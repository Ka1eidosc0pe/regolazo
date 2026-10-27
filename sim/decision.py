"""On-ball decision model: enumerate options, score them with decision value, choose with a shaped softmax.

Every decider gets the same fixed slot layout (passes to the 10 teammates, three through balls, a shot, three
carries, a shield, a take-on and a clearance), so a whole batch of deciders is scored in one vectorized pass
through the resolver and the analytics models. Vision widens the options a player perceives, Composure keeps the
choice sharp under pressure, and archetypes add tendencies (poachers shoot, wingers cross, playmakers thread).
"""

from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from shared.model_api import AnalyticsModels, decision_value
from sim.constants import HALF_LENGTH, HALF_WIDTH
from sim.evaluate import (
    SIT_COUNTER,
    SIT_DIRECT_FK,
    SIT_OPEN,
    SIT_PENALTY,
    SIT_SET_PIECE,
    carry_retention,
    lane_count,
    race_logit,
    score_passes,
    score_shots,
)
from sim.resolver import (
    CTX_CORNER,
    CTX_FREE_KICK,
    CTX_GK_HANDS,
    CTX_GOAL_KICK,
    CTX_KICK_OFF,
    CTX_OPEN,
    CTX_PENALTY,
    CTX_THROW_IN,
    I_CARRY,
    I_CLEARANCE,
    I_PASS_PLAYER,
    I_PASS_SPACE,
    I_SHOT,
    Context,
    Resolved,
    default_shot_aim,
    resolve,
)
from sim.tactics import TeamStyle
from sim.world import N_ON, Squad

F = npt.NDArray[np.float64]
IA = npt.NDArray[np.int64]
BA = npt.NDArray[np.bool_]

N_SLOTS: Final = 20
S_PASS0, S_THROUGH0, S_SHOT, S_CARRY0, S_HOLD, S_DRIBBLE, S_CLEAR = 0, 10, 13, 14, 17, 18, 19
SLOT_NAMES: Final = tuple(
    [f"pass_{k}" for k in range(10)]
    + ["through_0", "through_1", "through_2", "shot", "carry_fwd", "carry_left", "carry_right", "hold", "dribble"]
    + ["clearance"]
)

ARCH_SHOT_BIAS: Final = {
    "poacher": 0.022,
    "inside_forward": 0.016,
    "target_forward": 0.012,
    "advanced_playmaker": 0.006,
    "box_to_box": 0.004,
}


@dataclass
class DeciderInput:
    """Everything the decision model needs for D deciders (world frame)."""

    carrier: IA
    sign: F
    ball: F
    """(D, 3)"""
    pos: F
    """(D, 22, 2) all players"""
    vel: F
    run_target: F
    """(D, 22, 2)"""
    running: BA
    """(D, 22)"""
    ctx: IA
    situation: IA
    last_def_x: F
    offside_line: F
    """Team-frame offside line for the deciding team."""
    t: F
    first_time: BA | None = None
    bounced: BA | None = None
    ball_vel: F | None = None


@dataclass
class OptionTable:
    valid: BA
    """(D, O)"""
    kind: IA
    receiver: IA
    target: F
    """(D, O, 3)"""
    power: F
    curve: F
    res: Resolved
    """Flattened over D * O rows."""
    p: F
    p0: F
    value_success: F
    value_fail: F
    dv: F
    xg: F
    utility: F
    prob: F
    """(D, O) softmax choice probabilities (0 for invalid)."""
    p_base: F
    """(D, O) success probability for an average-attribute player."""
    pressure: F
    """(D,)"""
    receiver_offside: BA


def _team_rows(carrier: IA) -> IA:
    """(D, 10) teammates of each carrier in squad order."""
    team = carrier // N_ON
    base = team[:, None] * N_ON + np.arange(N_ON)[None, :]
    keep = base != carrier[:, None]
    return np.asarray(base[keep].reshape(len(carrier), N_ON - 1))


def _opp_rows(carrier: IA) -> IA:
    team = carrier // N_ON
    return np.asarray((1 - team)[:, None] * N_ON + np.arange(N_ON)[None, :])


@dataclass
class Slots:
    kind: IA
    receiver: IA
    target: F
    power: F
    curve: F
    valid: BA
    receiver_offside: BA
    cat: IA
    """Slot category (S_PASS0, S_THROUGH0, S_SHOT, S_CARRY0, S_HOLD, S_DRIBBLE, S_CLEAR) for the tendencies."""


@dataclass
class Common:
    rows: IA
    cpos: F
    cpos_t: F
    opp_pos: F
    opp_t: F
    opp_out: BA
    gk_idx: IA
    gk_pos: F
    d_opp_out: F
    nearest: F
    pressure: F
    sdist: F


def common(sq: Squad, di: DeciderInput) -> Common:
    d = len(di.carrier)
    c = di.carrier
    s = di.sign
    rows = np.arange(d)
    opps = _opp_rows(c)
    cpos = di.pos[rows, c]
    cpos_t = cpos * s[:, None]
    opp_pos = np.take_along_axis(di.pos, opps[..., None], axis=1)
    opp_t = opp_pos * s[:, None, None]
    opp_out = ~sq.is_gk[opps]
    gk_idx = opps[:, 0]
    gk_pos = di.pos[rows, gk_idx]
    d_opp = np.linalg.norm(opp_pos - cpos[:, None, :], axis=2)
    d_opp_out = np.where(opp_out, d_opp, 99.0)
    nearest = d_opp_out.min(axis=1)
    pressure = np.clip((3.5 - nearest) / 2.5, 0.0, 1.0)
    sdist = np.hypot(HALF_LENGTH - cpos_t[:, 0], cpos_t[:, 1])
    return Common(rows, cpos, cpos_t, opp_pos, opp_t, opp_out, gk_idx, gk_pos, d_opp_out, nearest, pressure, sdist)


def build_options(
    sq: Squad,
    models: AnalyticsModels,
    styles: tuple[TeamStyle, TeamStyle],
    di: DeciderInput,
    rng: np.random.Generator | None,
) -> OptionTable:
    cm = common(sq, di)
    slots = enumerate_slots(sq, di, cm, rng)
    return score_table(sq, models, styles, di, cm, slots, rng)


def enumerate_slots(sq: Squad, di: DeciderInput, cm: Common, rng: np.random.Generator | None) -> Slots:
    d = len(di.carrier)
    o = N_SLOTS
    c = di.carrier
    s = di.sign
    rows = cm.rows
    mates = _team_rows(c)
    cpos, cpos_t, opp_pos = cm.cpos, cm.cpos_t, cm.opp_pos
    gk_pos, d_opp_out, nearest, pressure = cm.gk_pos, cm.d_opp_out, cm.nearest, cm.pressure

    kind = np.full((d, o), I_PASS_PLAYER, dtype=np.int64)
    receiver = np.full((d, o), -1, dtype=np.int64)
    target = np.zeros((d, o, 3))
    power = np.full((d, o), 0.5)
    curve = np.zeros((d, o))
    valid = np.ones((d, o), dtype=bool)

    # passes to each teammate, led by their movement
    mpos = np.take_along_axis(di.pos, mates[..., None], axis=1)
    mvel = np.take_along_axis(di.vel, mates[..., None], axis=1)
    # aim where the receiver will be when the ball gets there (ground passes average ~11 m/s)
    first = mpos + 0.6 * mvel
    t_fl = np.clip(np.linalg.norm(first - cpos[:, None, :], axis=2) / 11.0, 0.3, 2.0)
    lead = mpos + t_fl[..., None] * mvel
    target[:, S_PASS0:S_THROUGH0, :2] = lead
    receiver[:, S_PASS0:S_THROUGH0] = mates
    if rng is not None:
        power[:, S_PASS0:S_THROUGH0] = rng.uniform(0.35, 0.92, (d, N_ON - 1))
    pass_len = np.linalg.norm(lead - cpos[:, None, :], axis=2)
    valid[:, S_PASS0:S_THROUGH0] &= pass_len > 4.0
    receiver_offside = np.zeros((d, o), dtype=bool)
    mates_x_t = mpos[..., 0] * s[:, None]
    receiver_offside[:, S_PASS0:S_THROUGH0] = (mates_x_t > di.offside_line[:, None] + 0.05) & (mates_x_t > 0)

    # through balls: three most advanced outfield teammates
    adv_key = np.where(sq.is_gk[mates], -1e9, mates_x_t)
    adv = np.argsort(-adv_key, axis=1)[:, :3]
    for k in range(3):
        j = mates[rows, adv[:, k]]
        p_j = di.pos[rows, j]
        run = di.running[rows, j]
        p_t = p_j * s[:, None]
        v_t = di.vel[rows, j] * s[:, None]
        # into the runner's stride: where he will be when the ball arrives, plus a couple of metres
        t_fl = np.clip(np.linalg.norm(p_t - cpos_t, axis=1) / 11.0, 0.6, 2.2)
        stride = p_t + v_t * t_fl[:, None] + np.array([2.5, 0.0])
        tgt_t = np.where(
            run[:, None],
            np.maximum(stride, di.run_target[rows, j] * s[:, None]),
            p_t + np.column_stack([np.full(d, 8.0), -0.15 * p_t[:, 1]]),
        )
        tgt_t[:, 0] = np.minimum(tgt_t[:, 0], HALF_LENGTH - 3.0)
        tgt_t[:, 1] = np.clip(tgt_t[:, 1], -HALF_WIDTH + 2.0, HALF_WIDTH - 2.0)
        slot = S_THROUGH0 + k
        target[:, slot, :2] = tgt_t * s[:, None]
        kind[:, slot] = I_PASS_SPACE
        receiver[:, slot] = j
        valid[:, slot] = (tgt_t[:, 0] > cpos_t[:, 0] + 6.0) & (p_t[:, 0] > -10.0)
        receiver_offside[:, slot] = (p_t[:, 0] > di.offside_line + 0.05) & (p_t[:, 0] > 0)
        power[:, slot] = 0.25  # weighted to die near the runner
        # technicians bend it round the last man
        bender = sq.attr["curve"][c] >= 66
        bend_sign = np.where(sq.left_foot[c], 1.0, -1.0)
        if rng is not None:
            bend_sign = np.where(rng.random(d) < 0.35, -bend_sign, bend_sign)
        curve[:, slot] = np.where(bender, 0.6 * bend_sign, 0.0)

    # shot
    aim = default_shot_aim(s, di.ball, gk_pos, rng)
    target[:, S_SHOT] = aim
    kind[:, S_SHOT] = I_SHOT
    power[:, S_SHOT] = 0.75 if rng is None else rng.uniform(0.55, 0.95, d)
    sdist = cm.sdist
    long_ok = sq.attr["long_shots"][c] >= 72
    valid[:, S_SHOT] = (
        ((sdist < 24.0) | ((sdist < 30.0) & long_ok)) & (cpos_t[:, 0] > 18.0) & (np.abs(cpos_t[:, 1]) < 24.0)
    )

    # carries: ahead, diagonals, shield
    for k, ang in enumerate((0.0, 0.7, -0.7)):
        slot = S_CARRY0 + k
        dir_t = np.array([np.cos(ang), np.sin(ang)])
        tgt_t = cpos_t + 8.0 * dir_t[None, :]
        target[:, slot, :2] = tgt_t * s[:, None]
        kind[:, slot] = I_CARRY
        valid[:, slot] = (np.abs(tgt_t[:, 0]) < HALF_LENGTH - 1.5) & (np.abs(tgt_t[:, 1]) < HALF_WIDTH - 1.0)
    j_near = np.argmin(d_opp_out, axis=1)
    away = cpos - opp_pos[rows, j_near]
    away /= np.maximum(np.linalg.norm(away, axis=1, keepdims=True), 1e-6)
    hold_t = cpos + 3.0 * away
    target[:, S_HOLD, :2] = hold_t
    kind[:, S_HOLD] = I_CARRY
    valid[:, S_HOLD] = (np.abs(hold_t[:, 0]) < HALF_LENGTH - 1.0) & (np.abs(hold_t[:, 1]) < HALF_WIDTH - 1.0)
    # take-on: the nearest opponent is in front and close
    to_opp_t = (opp_pos[rows, j_near] - cpos) * s[:, None]
    in_front = (to_opp_t[:, 0] > 0.3) & (nearest < 3.2)
    drib_t = cpos_t + np.column_stack([np.full(d, 7.0), np.sign(-to_opp_t[:, 1] + 1e-6) * 2.0])
    target[:, S_DRIBBLE, :2] = drib_t * s[:, None]
    kind[:, S_DRIBBLE] = I_CARRY
    valid[:, S_DRIBBLE] = in_front & (np.abs(drib_t[:, 0]) < HALF_LENGTH - 1.0) & (np.abs(drib_t[:, 1]) < HALF_WIDTH)
    # clearance
    side = np.where(cpos_t[:, 1] >= 0, 1.0, -1.0)
    clr_t = np.column_stack([np.minimum(cpos_t[:, 0] + 42.0, 30.0), np.clip(cpos_t[:, 1] + side * 14.0, -30, 30)])
    target[:, S_CLEAR, :2] = clr_t * s[:, None]
    kind[:, S_CLEAR] = I_CLEARANCE
    power[:, S_CLEAR] = 0.8
    valid[:, S_CLEAR] = (cpos_t[:, 0] < -25.0) & (pressure > 0.3) & ~sq.is_gk[c]

    # restart contexts restrict the menu
    ctx = di.ctx
    only_pass = np.isin(ctx, [CTX_THROW_IN, CTX_GOAL_KICK, CTX_CORNER, CTX_GK_HANDS, CTX_KICK_OFF])
    valid[only_pass, S_SHOT:] = False
    valid[ctx == CTX_PENALTY, :] = False
    valid[ctx == CTX_PENALTY, S_SHOT] = True
    fk = ctx == CTX_FREE_KICK
    valid[fk, S_CARRY0:] = False
    valid[fk, S_SHOT] &= (di.situation[fk] == SIT_DIRECT_FK) & (sdist[fk] < 30.0)
    thr = ctx == CTX_THROW_IN
    valid[thr, S_PASS0:S_SHOT] &= np.linalg.norm(target[thr, S_PASS0:S_SHOT, :2] - cpos[thr, None, :], axis=2) < 28.0
    ko = ctx == CTX_KICK_OFF
    valid[ko, S_PASS0:S_SHOT] &= (target[ko, S_PASS0:S_SHOT, 0] * s[ko, None] < 1.0) & (
        np.linalg.norm(target[ko, S_PASS0:S_SHOT, :2] - cpos[ko, None, :], axis=2) < 22.0
    )
    cor = ctx == CTX_CORNER
    if cor.any():
        in_box_t = target[cor, S_PASS0:S_THROUGH0, 0] * s[cor, None] > 36.0
        close = np.linalg.norm(target[cor, S_PASS0:S_THROUGH0, :2] - cpos[cor, None, :], axis=2) < 16.0
        valid[cor, S_PASS0:S_THROUGH0] &= in_box_t | close
        valid[cor, S_THROUGH0:S_SHOT] = False
    gk_hands = ctx == CTX_GK_HANDS
    valid[gk_hands, S_THROUGH0:S_SHOT] = False
    # keepers do not dribble
    gk_c = sq.is_gk[c]
    valid[gk_c, S_SHOT:S_CLEAR] = False
    valid[gk_c, S_HOLD] = ctx[gk_c] == CTX_OPEN

    cat = np.zeros((d, o), dtype=np.int64)
    cat[:, S_THROUGH0:S_SHOT] = S_THROUGH0
    cat[:, S_SHOT] = S_SHOT
    cat[:, S_CARRY0:S_HOLD] = S_CARRY0
    cat[:, S_HOLD] = S_HOLD
    cat[:, S_DRIBBLE] = S_DRIBBLE
    cat[:, S_CLEAR] = S_CLEAR
    return Slots(kind, receiver, target, power, curve, valid, receiver_offside, cat)


def score_table(
    sq: Squad,
    models: AnalyticsModels,
    styles: tuple[TeamStyle, TeamStyle],
    di: DeciderInput,
    cm: Common,
    sl: Slots,
    rng: np.random.Generator | None,
    pass_override: IA | None = None,
    shot_override: IA | None = None,
) -> OptionTable:
    """Resolve and score a (D, O) table of options; also used for a single fan intent (O = 1)."""
    d, o = sl.kind.shape
    c = di.carrier
    s = di.sign
    kind, receiver, target, power, curve = sl.kind, sl.receiver, sl.target, sl.power, sl.curve
    valid, receiver_offside, cat = sl.valid.copy(), sl.receiver_offside, sl.cat
    cpos, cpos_t, opp_pos, opp_t, opp_out = cm.cpos, cm.cpos_t, cm.opp_pos, cm.opp_t, cm.opp_out
    gk_idx, gk_pos, d_opp_out, nearest, pressure, sdist = (
        cm.gk_idx,
        cm.gk_pos,
        cm.d_opp_out,
        cm.nearest,
        cm.pressure,
        cm.sdist,
    )
    ctx = di.ctx
    # --- resolve every option -----------------------------------------------------------------------------
    n = d * o

    def flat(a: npt.NDArray[Any]) -> npt.NDArray[Any]:
        return a.reshape(n, *a.shape[2:])

    def rep(a: npt.NDArray[Any]) -> npt.NDArray[Any]:
        return np.repeat(a, o, axis=0)

    lane_blocked = np.zeros((d, o), dtype=bool)
    for slot in range(o):
        tgt = target[:, slot, :2]
        seg = tgt - cpos
        ll = np.maximum(np.sum(seg * seg, axis=1), 1e-9)
        rel = opp_pos - cpos[:, None, :]
        u = np.sum(rel * seg[:, None, :], axis=2) / ll[:, None]
        closest = cpos[:, None, :] + np.clip(u, 0, 1)[..., None] * seg[:, None, :]
        dl = np.linalg.norm(opp_pos - closest, axis=2)
        lane_blocked[:, slot] = ((u > 0.1) & (u < 0.9) & (dl < 1.4)).any(axis=1) & (kind[:, slot] <= I_PASS_SPACE)
    ctxo = Context(
        player=rep(c),
        sign=rep(s),
        ball=rep(di.ball),
        ball_vel=np.zeros((n, 3)) if di.ball_vel is None else rep(di.ball_vel),
        bounced_recently=np.zeros(n, dtype=bool) if di.bounced is None else rep(di.bounced),
        pressure=rep(pressure),
        nearest_opp_m=rep(nearest),
        lane_blocked=lane_blocked.reshape(n),
        gk_pos=rep(gk_pos),
        last_def_x=rep(di.last_def_x),
        restart=rep(ctx),
        first_time=np.zeros(n, dtype=bool) if di.first_time is None else rep(di.first_time),
    )
    ov_p = None if pass_override is None else np.repeat(pass_override, o)
    ov_s = None if shot_override is None else np.repeat(shot_override, o)
    res = resolve(sq, ctxo, flat(kind), flat(target), flat(power), flat(curve), flat(receiver), ov_p, ov_s, rng=rng)

    # --- score --------------------------------------------------------------------------------------------
    p = np.zeros(n)
    p0 = np.zeros(n)
    pb = np.zeros(n)
    vs = np.zeros(n)
    vf = np.zeros(n)
    xg = np.zeros(n)
    kf = flat(kind)
    origin_t = rep(cpos_t)
    end_t = flat(target)[:, :2] * rep(s)[:, None]
    opp_t_rep = rep(opp_t)
    is_pass = (kf == I_PASS_PLAYER) | (kf == I_PASS_SPACE)
    if is_pass.any():
        sub = _subset(res, is_pass)
        opps_rep = rep(_opp_rows(c))[is_pass]
        rec_f = flat(receiver)[is_pass]
        rows_n = rep(np.arange(d))[is_pass]
        rec_pos = di.pos[rows_n, np.maximum(rec_f, 0)]
        rec_vel = di.vel[rows_n, np.maximum(rec_f, 0)]
        race = race_logit(
            sq,
            sub,
            rep(cpos)[is_pass],
            flat(target)[is_pass, :2],
            flat(power)[is_pass],
            opps_rep,
            rep(opp_pos)[is_pass],
            np.take_along_axis(di.vel, _opp_rows(c)[..., None], axis=1).repeat(o, axis=0)[is_pass],
            rec_f,
            rec_pos,
            rec_vel,
            models.pitch_control_params,
        )
        sc = score_passes(
            models, sq, sub, origin_t[is_pass], end_t[is_pass], opp_t_rep[is_pass], rep(pressure)[is_pass], race
        )
        p[is_pass], p0[is_pass], vs[is_pass], vf[is_pass] = sc.p, sc.p0, sc.value_success, sc.value_fail
        pb[is_pass] = sc.p_base if sc.p_base is not None else sc.p0
    is_shot = kf == I_SHOT
    if is_shot.any():
        sub = _subset(res, is_shot)
        gk_t = rep(gk_pos * s[:, None])[is_shot]
        keeper = rep(gk_idx)[is_shot]
        outfield_t = np.where(rep(opp_out)[is_shot][..., None], opp_t_rep[is_shot], 999.0)
        sc = score_shots(
            models,
            sq,
            sub,
            origin_t[is_shot],
            gk_t,
            keeper,
            outfield_t,
            rep(pressure)[is_shot],
            rep(di.situation)[is_shot],
            np.zeros(int(is_shot.sum()), dtype=bool),
        )
        pb[is_shot] = sc.p0
        p[is_shot], p0[is_shot], vs[is_shot], vf[is_shot], xg[is_shot] = (
            sc.p,
            sc.p0,
            sc.value_success,
            sc.value_fail,
            sc.xg,
        )
    is_carry = kf == I_CARRY
    xt_opp_here = np.asarray(models.xt(-origin_t[:, 0], -origin_t[:, 1]))
    if is_carry.any():
        n_close = rep((d_opp_out < 2.5).sum(axis=1).astype(float))[is_carry]
        dribble = flat(cat) == S_DRIBBLE
        # opponents standing in (or able to step into) the carry path make it unlikely to survive
        seg_o = rep(cpos)[is_carry]
        seg_e = flat(target)[is_carry, :2]
        in_path = lane_count(seg_o, seg_e, rep(opp_pos)[is_carry], width=3.0).astype(float)
        in_path = np.where(dribble[is_carry], np.maximum(in_path - 1.0, 0.0), in_path)
        # and opponents waiting where the carry ends (a run into a packed box rarely survives)
        near_end = (np.linalg.norm(rep(opp_pos)[is_carry] - seg_e[:, None, :], axis=2) < 3.5).sum(axis=1)
        crowd = n_close + 1.3 * in_path + 0.8 * near_end
        pr = carry_retention(sq, rep(c)[is_carry], crowd, rep(pressure)[is_carry], dribble[is_carry])
        p[is_carry] = pr
        p0[is_carry] = pr
        pb[is_carry] = pr
        vs[is_carry] = np.asarray(models.xt(end_t[is_carry, 0], end_t[is_carry, 1]))
        vf[is_carry] = -0.6 * xt_opp_here[is_carry] - 0.004
    is_clear = kf == I_CLEARANCE
    if is_clear.any():
        p[is_clear] = 0.9
        p0[is_clear] = 0.9
        pb[is_clear] = 0.9
        vs[is_clear] = 0.002
        vf[is_clear] = -0.3 * xt_opp_here[is_clear]
    # a pass to a receiver standing offside at release is called back if it reaches him: it cannot succeed
    off = flat(receiver_offside)
    p_seen = p.copy()
    p = np.where(off, 0.0, p)
    pb = np.where(off, 0.0, pb)
    dv = decision_value(p, vs, vf)
    # the AI sometimes misreads the line and plays the pass anyway (that is where offsides come from)
    dv_seen = dv
    if rng is not None:
        vis_rep = np.repeat(sq.attr["vision"][c] / 99.0, o)
        misread = off & (rng.random(n) < 0.18 * (1.25 - vis_rep))
        dv_seen = np.where(misread, decision_value(p_seen, vs, vf), dv)

    # --- utility: decision value plus tendencies ------------------------------------------------------------
    util = dv_seen.copy().reshape(d, o)
    arch = np.array([sq.players[i].archetype for i in range(len(sq.players))])
    arch_c = arch[c]
    shot_bias = 0.5 * np.array([ARCH_SHOT_BIAS.get(a, -0.006) for a in arch_c]) - 0.01
    util += (cat == S_SHOT) * (shot_bias + 0.012 * (sq.attr["long_shots"][c] - 70) / 20 * (sdist > 20))[:, None]
    vision = sq.attr["vision"][c] / 99.0
    directness = np.array([styles[0].directness, styles[1].directness])[c // N_ON]
    util += (cat == S_THROUGH0) * (0.006 * (arch_c == "advanced_playmaker") + 0.003 * directness)[:, None]
    ptype = res.pass_type.reshape(d, o)
    winger = np.isin(arch_c, ["touchline_winger", "overlapping_fullback"])
    util += (ptype == 1) * (0.012 + 0.02 * winger[:, None])
    dribbler = np.isin(arch_c, ["inside_forward", "touchline_winger", "advanced_playmaker"])
    util += (cat == S_CARRY0) * (0.004 + 0.006 * dribbler - 0.006 * (sq.line[c] == 1))[:, None]
    # in the box players look to finish or set up, rather than dribble at the keeper
    in_box_now = (cpos_t[:, 0] > 36.0) & (np.abs(cpos_t[:, 1]) < 20.16)
    util -= ((cat == S_CARRY0) | (cat == S_DRIBBLE)) * (0.012 * in_box_now)[:, None]
    util += (cat == S_DRIBBLE) * (0.006 * dribbler - 0.012)[:, None]
    util -= (cat == S_HOLD) * 0.012
    util += (cat == S_CLEAR) * (0.02 * pressure)[:, None]
    # forward progress tendency
    fwd = (end_t[:, 0] - origin_t[:, 0]).reshape(d, o)
    util += 0.0004 * directness[:, None] * np.clip(fwd, -20, 30)
    # perception: long balls and through balls are only seen by players with vision
    if rng is not None:
        long_ = (np.linalg.norm(end_t - origin_t, axis=1) > 30.0).reshape(d, o)
        long_ |= cat == S_THROUGH0
        seen = rng.random((d, o)) < (0.3 + 0.7 * vision[:, None] ** 1.3)
        valid &= ~long_ | seen
    composure = sq.attr["composure"][c] / 99.0
    tau = 0.011 * (1.25 - 0.5 * vision) * (1.0 + 0.8 * pressure * (1.0 - composure))
    z = np.where(valid, util / tau[:, None], -np.inf)
    zmax = np.where(valid.any(axis=1, keepdims=True), np.max(np.where(valid, z, -1e300), axis=1, keepdims=True), 0.0)
    z = z - zmax
    e = np.where(valid, np.exp(z), 0.0)
    prob = e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)
    return OptionTable(
        valid=valid,
        kind=kind,
        receiver=receiver,
        target=target,
        power=power,
        curve=curve,
        res=res,
        p=p.reshape(d, o),
        p0=p0.reshape(d, o),
        p_base=pb.reshape(d, o),
        value_success=vs.reshape(d, o),
        value_fail=vf.reshape(d, o),
        dv=dv.reshape(d, o),
        xg=xg.reshape(d, o),
        utility=util,
        prob=prob,
        pressure=pressure,
        receiver_offside=receiver_offside,
    )


def _subset(res: Resolved, m: BA) -> Resolved:
    return Resolved(**{k: v[m] for k, v in res.__dict__.items()})


def choose(table: OptionTable, rng: np.random.Generator) -> IA:
    cum = np.cumsum(table.prob, axis=1)
    r = rng.random(len(cum))[:, None] * cum[:, -1:]
    return np.asarray(np.argmax(cum > r, axis=1), dtype=np.int64)


def situation_code(restart_ctx: IA, direct: BA, counter: BA, set_phase: BA) -> IA:
    sit = np.full(len(restart_ctx), SIT_OPEN, dtype=np.int64)
    sit = np.where(counter, SIT_COUNTER, sit)
    sit = np.where(set_phase, SIT_SET_PIECE, sit)
    sit = np.where((restart_ctx == CTX_FREE_KICK) & direct, SIT_DIRECT_FK, sit)
    sit = np.where(restart_ctx == CTX_PENALTY, SIT_PENALTY, sit)
    return sit


def option_entropy(prob: F) -> F:
    pr = np.clip(prob, 1e-12, 1)
    return np.asarray(-(prob * np.log(pr)).sum(axis=1))


__all__ = [
    "I_CARRY",
    "I_CLEARANCE",
    "I_PASS_PLAYER",
    "I_PASS_SPACE",
    "I_SHOT",
    "N_SLOTS",
    "DeciderInput",
    "OptionTable",
    "build_options",
    "choose",
]
