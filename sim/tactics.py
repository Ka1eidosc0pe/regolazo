"""Team behaviour: shape by phase, block height, marking, pressing, support and set-piece positions.

All functions are vectorized over the batch and the 22 players. Shapes are defined in the team frame (the team
attacks +x, +y on its left) by the formation slots in ``sim.squads`` and converted to the world frame by the
team's attacking sign.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from sim.constants import HALF_LENGTH, HALF_WIDTH
from sim.world import (
    BS_DEAD,
    BS_OWNED,
    N_ON,
    R_CORNER,
    R_FK_DIRECT,
    R_FK_INDIRECT,
    R_GOAL_KICK,
    R_KICK_OFF,
    R_PENALTY,
    Squad,
    World,
)

F = npt.NDArray[np.float64]
IA = npt.NDArray[np.int64]


@dataclass(frozen=True)
class TeamStyle:
    press: float
    """0..1 pressing intensity."""
    line: float
    """Defensive line offset (m, team frame)."""
    width: float
    tempo: float
    """Scales decision speed."""
    directness: float
    """0..1 preference for forward / long options."""


def styles_for(seed: int) -> tuple[TeamStyle, TeamStyle]:
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x57E1]))
    out = []
    for _ in range(2):
        out.append(
            TeamStyle(
                press=float(rng.uniform(0.3, 0.9)),
                line=float(rng.uniform(-4.0, 4.0)),
                width=float(rng.uniform(0.92, 1.08)),
                tempo=float(rng.uniform(0.7, 1.3)),
                directness=float(rng.uniform(0.0, 1.0)),
            )
        )
    return out[0], out[1]


def attack_sign(w: World, home_positive_first: bool) -> F:
    """(B, 2): +1 where team (home, away) attacks +x in the current period."""
    home_pos = np.where(w.period == 1, home_positive_first, not home_positive_first)
    hs = np.where(home_pos, 1.0, -1.0)
    return np.stack([hs, -hs], axis=1)


def offside_lines(w: World, sq: Squad, sign: F) -> F:
    """(B, 2) offside line for each attacking team, in that team's frame (>= halfway line, >= ball x)."""
    out = np.zeros((w.b, 2))
    for t in (0, 1):
        opp = slice(N_ON, 2 * N_ON) if t == 0 else slice(0, N_ON)
        xs = w.pos[:, opp, 0] * sign[:, t : t + 1]
        second_last = np.sort(xs, axis=1)[:, -2]
        ball_x = w.ball_pos[:, 0] * sign[:, t]
        out[:, t] = np.maximum(np.maximum(second_last, ball_x), 0.0)
    return out


def last_defender_x(w: World, sq: Squad, sign: F) -> F:
    """(B, 2) x of the deepest outfield opponent for each attacking team, in that team's frame."""
    out = np.zeros((w.b, 2))
    for t in (0, 1):
        opp = np.arange(N_ON, 2 * N_ON) if t == 0 else np.arange(0, N_ON)
        opp = opp[~sq.is_gk[opp]]
        xs = w.pos[:, opp, 0] * sign[:, t : t + 1]
        out[:, t] = xs.max(axis=1)
    return out


def shape_targets(
    w: World,
    sq: Squad,
    sign: F,
    ref: F,
    poss: IA,
    styles: tuple[TeamStyle, TeamStyle],
) -> tuple[F, F]:
    """Off-ball targets (B, P, 2) in the world frame and speed caps (B, P) from team shape and marking.

    ``ref`` is the ball reference point (B, 2): the ball, or where a ball in flight will be collected.
    ``poss`` is the team in (or last in) possession, -1 for none.
    """
    b = w.b
    s_p = sign[:, sq.team]  # (B, P)
    bx = ref[:, None, 0] * s_p
    by = ref[:, None, 1] * s_p
    in_poss = sq.team[None, :] == poss[:, None]
    press = np.array([styles[0].press, styles[1].press])[sq.team][None, :]
    line_off = np.array([styles[0].line, styles[1].line])[sq.team][None, :]
    width = np.array([styles[0].width, styles[1].width])[sq.team][None, :]
    ax_in = np.clip(0.55 * bx + 5.0, -22.0, 23.0)
    ax_out = np.clip(0.62 * bx - 7.0 + line_off + 6.0 * (press - 0.5), -31.0, 12.0)
    ay_in = 0.25 * by
    ay_out = 0.45 * by
    ox = np.where(in_poss, sq.in_xy[None, :, 0], sq.out_xy[None, :, 0])
    oy = np.where(in_poss, sq.in_xy[None, :, 1] * width, sq.out_xy[None, :, 1] * 0.8)
    # out of possession the block is built line by line, compact (8-14 m between lines) and deeper near goal
    l1 = np.clip(0.55 * bx - 19.0 + line_off + 6.0 * (press - 0.5), -45.0, -4.0)
    l2 = l1 + np.clip(8.0 + 0.07 * (bx + 52.5), 8.0, 14.0)
    l3 = l2 + np.clip(7.0 + 0.1 * (bx + 52.5), 7.0, 16.0)
    line = sq.line[None, :]
    base = np.where(line == 1, l1, np.where(line == 2, l2, l3))
    ref_x = np.where(line == 1, -15.0, np.where(line == 2, -4.0, 9.0))
    tx_out = base + (sq.out_xy[None, :, 0] - ref_x)
    _ = ax_out
    tx = np.where(in_poss, ax_in + ox, tx_out)
    ty = np.where(in_poss, ay_in, ay_out) + oy
    # fullbacks push on, wingers hold width: already in slot offsets. Defence stays goal-side of the ball.
    defending = ~in_poss & (sq.line[None, :] == 1)
    tx = np.where(defending, np.minimum(tx, np.maximum(bx - 3.0, -47.0)), tx)
    # attackers in possession hold the offside line unless running
    if b:
        lines = offside_lines(w, sq, sign)  # (B, 2)
        off = lines[:, sq.team]
        hold = in_poss & (sq.line[None, :] >= 2)
        tx = np.where(hold, np.minimum(tx, off - 0.7), tx)
    # marking: out-of-possession defenders and midfielders pick up the nearest opponent near their zone
    tx, ty = _marking(w, sq, s_p, tx, ty, in_poss)
    # the back line never leaves an attacker behind it: it drops to the deepest opponent
    deepest = _deepest_opponent(w, sq, s_p)
    tx = np.where(defending, np.minimum(tx, np.maximum(deepest - 1.0, -50.0)), tx)
    # goalkeepers
    gk_goal_x = -HALF_LENGTH
    d_ball = np.hypot(bx - gk_goal_x, by)
    depth = np.clip(0.9 + 0.1 * d_ball, 0.9, 5.5)
    sweeper = np.array([p.archetype == "sweeper_keeper" for p in sq.players])[None, :]
    depth = depth + np.where(sweeper & (bx > 0), np.clip((bx - 0) * 0.25, 0, 8.0), 0.0)
    depth = depth + np.where(in_poss & (bx > -10), 3.0, 0.0)
    ux = (bx - gk_goal_x) / np.maximum(d_ball, 1e-6)
    uy = by / np.maximum(d_ball, 1e-6)
    gx = gk_goal_x + ux * depth
    gy = np.clip(uy * depth, -3.0, 3.0)
    tx = np.where(sq.is_gk[None, :], gx, tx)
    ty = np.where(sq.is_gk[None, :], gy, ty)
    tx = np.clip(tx, -HALF_LENGTH + 0.5, HALF_LENGTH - 1.0)
    ty = np.clip(ty, -HALF_WIDTH + 1.0, HALF_WIDTH - 1.0)
    target = np.stack([tx * s_p, ty * s_p], axis=-1)
    dist = np.linalg.norm(target - w.pos, axis=-1)
    urgency = np.where(in_poss, 0.85, 1.1)
    cap = np.clip(0.2 + 0.13 * dist, 0.35, 2.9) * urgency
    cap = np.where(dist > 15.0, 0.75 * sq.max_speed[None, :], cap)
    return target, np.minimum(cap, sq.max_speed[None, :])


def _deepest_opponent(w: World, sq: Squad, s_p: F) -> F:
    """(B, P): x (in each player's team frame) of the opposing outfield player nearest that player's goal."""
    out = np.zeros_like(s_p)
    for t in (0, 1):
        opp = np.arange((1 - t) * N_ON, (2 - t) * N_ON)
        opp = opp[~sq.is_gk[opp]]
        mine = slice(t * N_ON, (t + 1) * N_ON)
        xs = w.pos[:, opp, 0] * s_p[:, t * N_ON : t * N_ON + 1]
        out[:, mine] = xs.min(axis=1)[:, None]
    return out


def _marking(w: World, sq: Squad, s_p: F, tx: F, ty: F, in_poss: npt.NDArray[np.bool_]) -> tuple[F, F]:
    """Out of possession, each outfield defender/midfielder goes goal-side of the opponent nearest his zone."""
    for t in (0, 1):
        mine = np.arange(t * N_ON, (t + 1) * N_ON)
        opp = np.arange((1 - t) * N_ON, (2 - t) * N_ON)
        mine = mine[~sq.is_gk[mine] & (sq.line[mine] <= 2)]
        opp = opp[~sq.is_gk[opp]]
        defending = ~in_poss[:, mine[0]]  # (B,)
        if not defending.any():
            continue
        sgn = s_p[:, mine[0]]
        zx, zy = tx[:, mine], ty[:, mine]  # team frame
        ox = w.pos[:, opp, 0] * sgn[:, None]
        oy = w.pos[:, opp, 1] * sgn[:, None]
        d = np.hypot(zx[:, :, None] - ox[:, None, :], zy[:, :, None] - oy[:, None, :])  # (B, M, O)
        j = np.argmin(d, axis=2)
        dmin = np.take_along_axis(d, j[..., None], axis=2)[..., 0]
        mx = np.take_along_axis(ox, j, axis=1)
        my = np.take_along_axis(oy, j, axis=1)
        # goal side: 1.8 m toward own goal centre
        gx, gy = -HALF_LENGTH - mx, -my
        gn = np.maximum(np.hypot(gx, gy), 1e-6)
        mark_x, mark_y = mx + 1.8 * gx / gn, my + 1.8 * gy / gn
        radius = np.where(sq.line[mine] == 1, 11.0, 8.0)[None, :]
        weight = np.clip(1.0 - dmin / radius, 0.0, 1.0) * 0.8
        weight = np.where(defending[:, None], weight, 0.0)
        mx_blend = zx + weight * (mark_x - zx)
        # defenders keep the line: they shadow attackers sideways but never step up out of it to mark
        back = (sq.line[mine] == 1)[None, :]
        tx[:, mine] = np.where(back, np.minimum(zx, mx_blend), mx_blend)
        ty[:, mine] = zy + weight * (my * 0 + mark_y - zy)
        w.mark[:, mine] = np.where(weight > 0.2, opp[j], -1)
    return tx, ty


def restart_targets(w: World, sq: Squad, sign: F, b: int, styles: tuple[TeamStyle, TeamStyle]) -> F | None:
    """Set-piece positions (P, 2) in the world frame for batch element ``b`` while the ball is dead."""
    if w.ball_state[b] != BS_DEAD or w.restart[b] < 0:
        return None
    r = int(w.restart[b])
    team = int(w.restart_team[b])
    rp = w.restart_pos[b]
    taker = int(w.restart_taker[b])
    s = sign[b]
    out = np.zeros((22, 2))
    # generic shape around the restart point first
    ref = rp[None, :]
    poss = np.array([team])
    sub = _single(w, b)
    tgt, _ = shape_targets(sub, sq, sign[b : b + 1], ref, poss, styles)
    out[:] = tgt[0]
    att = np.arange(team * N_ON, (team + 1) * N_ON)
    dfn = np.arange((1 - team) * N_ON, (2 - team) * N_ON)
    sa, sd = s[team], s[1 - team]
    if r == R_KICK_OFF:
        for t in (0, 1):
            idx = np.arange(t * N_ON, (t + 1) * N_ON)
            st = s[t]
            x_t = np.minimum(-12.0 + sq.out_xy[idx, 0] * 1.1, -1.0)
            y_t = sq.out_xy[idx, 1] * 1.1
            if t != team:
                r_c = np.hypot(x_t, y_t)
                push = r_c < 9.6
                x_t = np.where(push, np.minimum(x_t, -9.6), x_t)
            x_t[sq.is_gk[idx]] = -HALF_LENGTH + 1.0
            y_t[sq.is_gk[idx]] = 0.0
            out[idx, 0], out[idx, 1] = x_t * st, y_t * st
        if taker >= 0:
            out[taker] = (-0.3 * sa, 0.0)
    elif r == R_GOAL_KICK:
        gk = att[sq.is_gk[att]][0]
        out[gk] = rp
        for i in att:
            if sq.line[i] == 1:
                out[i, 0] = (-38.0 if sq.in_xy[i, 0] < -10 else -30.0) * sa
                out[i, 1] = np.sign(sq.in_xy[i, 1]) * (16.0 if abs(sq.in_xy[i, 1]) < 15 else 28.0) * sa
    elif r == R_CORNER:
        box = [(47.5, -1.5), (46.0, 3.0), (44.5, -4.5), (43.0, 1.0), (41.5, -3.0)]
        outfield = att[~sq.is_gk[att]]
        aerial = sorted(outfield, key=lambda i: -(sq.attr["heading"][i] + sq.height[i] * 40))
        corner_side = np.sign(rp[1] * sa) or 1.0
        for k, i in enumerate(aerial[:5]):
            if i == taker:
                continue
            out[i] = (box[k][0] * sa, box[k][1] * corner_side * sa)
        rest = [i for i in aerial[5:] if i != taker]
        for k, i in enumerate(rest):
            spots = [(33.0, -7.0), (33.0, 8.0), (2.0, -12.0), (2.0, 12.0), (-8.0, 0.0)]
            out[i] = (spots[k % 5][0] * sa, spots[k % 5][1] * sa)
        gk = dfn[sq.is_gk[dfn]][0]
        out[gk] = (-(HALF_LENGTH - 0.6) * sd, 0.0)
        markers = [i for i in dfn if not sq.is_gk[i]]
        markers = sorted(markers, key=lambda i: sq.line[i])
        targets_att = [out[i] for i in aerial[:5]]
        for k, i in enumerate(markers):
            if k < len(targets_att):
                ax_, ay_ = targets_att[k]
                gxs = -HALF_LENGTH * sd
                vx, vy = gxs - ax_, -ay_
                n = max(np.hypot(vx, vy), 1e-6)
                out[i] = (ax_ + 1.0 * vx / n, ay_ + 1.0 * vy / n)
            elif k == len(targets_att):
                out[i] = (-(HALF_LENGTH - 1.0) * sd, 3.2 * corner_side * sa)
            elif k < 9:
                out[i] = ((-37.0) * sd, (-5.0 + 5.0 * (k - 6)) * sd)
            else:
                out[i] = (-5.0 * sd, 0.0)
    elif r in (R_FK_DIRECT, R_FK_INDIRECT):
        gxs = HALF_LENGTH * sa
        d_goal = np.hypot(gxs - rp[0], rp[1])
        if d_goal < 33.0:
            u = np.array([gxs - rp[0], -rp[1]]) / max(d_goal, 1e-6)
            perp = np.array([-u[1], u[0]])
            wall_n = 4 if d_goal < 24 else 3
            wall = [i for i in dfn if not sq.is_gk[i]][:wall_n]
            for k, i in enumerate(wall):
                out[i] = rp + 9.15 * u + perp * (k - (wall_n - 1) / 2) * 0.6
            gk = dfn[sq.is_gk[dfn]][0]
            out[gk] = (gxs - 0.6 * sa, np.clip(-rp[1] * 0.08, -1.5, 1.5))
            box = [(44.0, -4.0), (43.0, 2.0), (41.0, 5.5), (40.0, -1.0), (45.5, 0.5)]
            fwd = [i for i in att if i != taker and not sq.is_gk[i] and sq.line[i] >= 2][:5]
            for k, i in enumerate(fwd):
                out[i] = (box[k][0] * sa, box[k][1] * sa)
    elif r == R_PENALTY:
        for k, j in enumerate([i for i in range(22) if i != taker and not sq.is_gk[i]]):
            out[j] = (34.0 * sa, (-14.0 + 1.5 * k) * sa)
        gk = dfn[sq.is_gk[dfn]][0]
        out[gk] = (HALF_LENGTH * sa - 0.1 * sa, 0.0)
        gk_att = att[sq.is_gk[att]][0]
        out[gk_att] = (-40.0 * sa, 0.0)
    if taker >= 0 and r != R_KICK_OFF:
        # stand just behind the ball, facing the attacking direction
        out[taker] = rp - np.array([0.6 * sa, 0.0])
    if r == R_KICK_OFF and taker >= 0:
        out[taker] = rp - np.array([0.3 * sa, 0.0])
    out[:, 0] = np.clip(out[:, 0], -HALF_LENGTH - 1.5, HALF_LENGTH + 1.5)
    out[:, 1] = np.clip(out[:, 1], -HALF_WIDTH - 1.5, HALF_WIDTH + 1.5)
    return out


def _single(w: World, b: int) -> World:
    """View of one batch element as a B=1 world (copies; used for restart set-ups only)."""
    from dataclasses import fields

    kw = {f.name: getattr(w, f.name)[b : b + 1].copy() for f in fields(w)}
    sub = World(**kw)
    sub.ball_state[:] = BS_OWNED
    return sub


def goal_side(w: World, sq: Squad, sign: F, poss: IA, tgt: F, cap: F) -> tuple[F, F]:
    """Hard defensive constraints, applied after the (smoothed) shape: the back line stays goal-side of the ball
    and of the deepest attacker, midfielders stay goal-side of the ball in their own third, and anybody caught on
    the wrong side sprints back."""
    s_p = sign[:, sq.team]
    in_poss = sq.team[None, :] == poss[:, None]
    bx = w.ball_pos[:, None, 0] * s_p
    tx = tgt[..., 0] * s_p
    deepest = _deepest_opponent(w, sq, s_p)
    back = ~in_poss & (sq.line[None, :] == 1)
    mid = ~in_poss & (sq.line[None, :] == 2) & (bx < -15.0)
    limit = np.where(back, np.minimum(bx - 1.5, deepest - 1.0), np.where(mid, bx - 1.0, np.inf))
    limit = np.maximum(limit, -HALF_LENGTH + 1.5)
    new_tx = np.minimum(tx, limit)
    px = w.pos[..., 0] * s_p
    behind = (back | mid) & (px > limit + 2.0)
    out = tgt.copy()
    out[..., 0] = new_tx * s_p
    cap = np.where(behind, 0.95 * sq.max_speed[None, :], cap)
    return out, cap


def protect_goal(w: World, sq: Squad, sign: F, poss: IA, tgt: F) -> F:
    """With the ball near their goal, defenders and midfielders narrow onto the line between ball and goal centre, so
    shots have to go through bodies."""
    s_p = sign[:, sq.team]
    bx = w.ball_pos[:, None, 0] * s_p
    by = w.ball_pos[:, None, 1] * s_p
    defending = (sq.team[None, :] != poss[:, None]) & ~sq.is_gk[None, :]
    k = np.clip((-bx - 15.0) / 15.0, 0.0, 1.0) * (np.abs(by) < 30.0)
    pull = np.where(sq.line[None, :] == 1, 0.5, np.where(sq.line[None, :] == 2, 0.35, 0.0)) * k * defending
    if not pull.any():
        return tgt
    tx, ty = tgt[..., 0] * s_p, tgt[..., 1] * s_p
    gx = -HALF_LENGTH
    frac = np.clip((tx - bx) / np.minimum(gx - bx, -1e-3), 0.0, 1.0)
    y_line = by * (1.0 - frac)
    out = tgt.copy()
    out[..., 1] = (ty + pull * (y_line - ty)) * s_p
    return out


def box_marking(w: World, sq: Squad, sign: F, poss: IA, tgt: F, cap: F, n_pairs: int = 4) -> tuple[F, F]:
    """Attackers inside (or arriving at) the box are picked up tightly, goal-side, by the nearest free defenders."""
    out, cap = tgt.copy(), cap.copy()
    rows = np.arange(w.b)
    for t in (0, 1):
        defending = poss != t
        if not defending.any():
            continue
        mine = np.arange(t * N_ON, (t + 1) * N_ON)
        mine = mine[~sq.is_gk[mine] & (sq.line[mine] <= 2)]
        opp = np.arange((1 - t) * N_ON, (2 - t) * N_ON)
        opp = opp[~sq.is_gk[opp]]
        s = sign[:, t][:, None]
        ox, oy = w.pos[:, opp, 0] * s, w.pos[:, opp, 1] * s
        danger = (ox < -BOX_EDGE) & (np.abs(oy) < 22.0) & defending[:, None]
        if not danger.any():
            continue
        d = np.linalg.norm(w.pos[:, mine][:, :, None, :] - w.pos[:, opp][:, None, :, :], axis=3)
        d = np.where(danger[:, None, :], d, np.inf)
        for _ in range(n_pairs):
            flat = d.reshape(w.b, -1)
            k = np.argmin(flat, axis=1)
            ok = np.isfinite(flat[rows, k]) & (flat[rows, k] < 14.0)
            if not ok.any():
                break
            i, j = np.unravel_index(k, d.shape[1:])
            bi = rows[ok]
            di, aj = mine[i[ok]], opp[j[ok]]
            a_pos = w.pos[bi, aj]
            goal = np.column_stack([-HALF_LENGTH * sign[bi, t], np.zeros(len(bi))])
            to_goal = goal - a_pos
            to_goal /= np.maximum(np.linalg.norm(to_goal, axis=1, keepdims=True), 1e-6)
            out[bi, di] = a_pos + 0.9 * to_goal + 0.4 * w.vel[bi, aj]
            cap[bi, di] = np.maximum(cap[bi, di], 0.9 * sq.max_speed[di])
            d[bi, i[ok], :] = np.inf
            d[bi, :, j[ok]] = np.inf
    return out, cap


BOX_EDGE = 27.0
"""Attackers this close to goal (team frame x beyond -33) count as box threats."""


def lane_blocker(w: World, sq: Squad, sign: F, poss: IA, tgt: F, cap: F) -> tuple[F, F]:
    """Near their own goal, the defender best placed steps into the shooting lane between ball and goal."""
    out, cap = tgt.copy(), cap.copy()
    rows = np.arange(w.b)
    for t in (0, 1):
        s = sign[:, t]
        bx, by = w.ball_pos[:, 0] * s, w.ball_pos[:, 1] * s
        active = (poss != t) & (bx < -28.0) & (np.abs(by) < 26.0)
        if not active.any():
            continue
        mine = np.arange(t * N_ON, (t + 1) * N_ON)
        mine = mine[~sq.is_gk[mine]]
        gx = -HALF_LENGTH
        vx, vy = gx - bx, -by
        n = np.maximum(np.hypot(vx, vy), 1e-6)
        step = np.minimum(3.0, 0.4 * n)
        lx, ly = bx + vx / n * step, by + vy / n * step
        lane = np.stack([lx * s, ly * s], axis=1)  # world frame
        d = np.linalg.norm(w.pos[:, mine] - lane[:, None, :], axis=2)
        j = mine[np.argmin(d, axis=1)]
        bi = rows[active]
        out[bi, j[active]] = lane[active]
        cap[bi, j[active]] = sq.max_speed[j[active]] * 0.95
    return out, cap
