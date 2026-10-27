"""The action resolver (PLAN.md section 2.6): intent + context + attributes -> technique, launch, uncertainty.

One vectorized implementation serves the AI decision model (hundreds of candidate options per tick) and the fan's
Rewind gestures (one intent), so both are executed with the same technique choice, physics and skill noise.
Inputs are in the world frame; rules are applied in the acting team's attacking frame.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

from sim import kick as K
from sim.constants import (
    BALL_RADIUS,
    BOX_HALF_WIDTH,
    BOX_X,
    GOAL_HEIGHT,
    HALF_GOAL,
    HALF_LENGTH,
    SWITCH_LATERAL_M,
)
from sim.world import Squad

F = npt.NDArray[np.float64]
IA = npt.NDArray[np.int64]
BA = npt.NDArray[np.bool_]

# intent kinds
I_PASS_PLAYER, I_PASS_SPACE, I_SHOT, I_CARRY, I_CLEARANCE = range(5)
INTENT_NAMES: Final = ("pass_player", "pass_space", "shot", "carry", "clearance")

PASS_HEIGHTS: Final = ("ground", "low", "high")
PASS_TECHNIQUES: Final = ("standard", "driven", "lofted", "chipped", "curled", "outside_of_boot", "backheel")
PASS_TYPES: Final = ("standard", "cross", "cutback", "through_ball", "layoff", "flick_on", "back_pass")
SHOT_TECHNIQUES: Final = (
    "driven",
    "placed",
    "finesse",
    "chip",
    "volley",
    "half_volley",
    "header",
    "diving_header",
    "toe_poke",
    "backheel",
    "overhead_kick",
)
BODY_PARTS: Final = ("right_foot", "left_foot", "head", "chest", "other", "hands")

# restart context codes passed to the resolver
CTX_OPEN, CTX_THROW_IN, CTX_GOAL_KICK, CTX_CORNER, CTX_FREE_KICK, CTX_PENALTY, CTX_KICK_OFF, CTX_GK_HANDS = range(8)

ATTR_REF: Final = 70.0
"""Rating of an "average" professional: model probabilities apply unadjusted at this level."""


@dataclass
class Context:
    """Per-intent situation (N rows)."""

    player: IA
    sign: F
    """+1 if the acting team attacks +x."""
    ball: F
    """(N, 3) ball position at contact."""
    ball_vel: F
    """(N, 3) ball velocity at contact (first-time actions)."""
    bounced_recently: BA
    pressure: F
    """0..1 (1 = opponent on top of the player)."""
    nearest_opp_m: F
    lane_blocked: BA
    gk_pos: F
    """(N, 2) opposing goalkeeper, world frame."""
    last_def_x: F
    """Team-frame x of the last outfield defender."""
    restart: IA
    """CTX_* code."""
    first_time: BA


@dataclass
class Resolved:
    """Vectorized ResolvedAction (N rows)."""

    player: IA
    kind: IA
    kick: IA
    """sim.kick technique code (-1 for carries)."""
    pass_tech: IA
    pass_type: IA
    shot_tech: IA
    body: IA
    weak_foot: BA
    curve_applied: F
    max_curve: F
    quality: F
    """Relevant skill 0..1 before pressure / weak foot."""
    noise_mult: F
    target: F
    """(N, 3) intended end point (receiver / landing spot / goal-mouth point)."""
    override_rejected: BA
    pass_height: IA
    """Intended height class: 0 ground, 1 low, 2 high."""


def _q(sq: Squad, name: str, idx: IA) -> F:
    return np.asarray(sq.attr[name][idx] / 99.0)


def _adj(rating: F, scale: float = 20.0) -> F:
    return np.asarray((rating - ATTR_REF) / scale)


def max_curve(sq: Squad, player: IA, kind: IA) -> F:
    """Largest |curve| a player can put on a pass or shot (Curve, Technique and Vision cap it)."""
    a = sq.attr
    pass_cap = (0.5 * a["curve"][player] + 0.3 * a["technique"][player] + 0.2 * a["vision"][player] - 48.0) / 40.0
    shot_cap = (0.6 * a["curve"][player] + 0.4 * a["finishing"][player] - 48.0) / 40.0
    cap = np.where(kind == I_SHOT, shot_cap, pass_cap)
    return np.asarray(np.clip(cap, 0.0, 1.0))


def resolve(
    sq: Squad,
    ctx: Context,
    kind: IA,
    target: F,
    power: F,
    curve: F,
    receiver: IA,
    pass_override: IA | None = None,
    shot_override: IA | None = None,
    rng: np.random.Generator | None = None,
) -> Resolved:
    """Pick technique, body part and launch class for each intent (see module docstring).

    ``target``: receiver position (pass_player), point in space (pass_space), goal-mouth point (shot, world frame
    with x on the goal line), carry end (carry). ``rng`` adds the AI's occasional weak-foot use; the fan's
    preview passes None and gets the deterministic choice.
    """
    n = len(kind)
    p = ctx.player
    s = ctx.sign
    ox, oy = ctx.ball[:, 0] * s, ctx.ball[:, 1] * s
    ex, ey = target[:, 0] * s, target[:, 1] * s
    dx, dy = ex - ox, ey - oy
    dist = np.hypot(dx, dy)
    ang = np.arctan2(dy, dx)
    z_ball = ctx.ball[:, 2]
    airborne = z_ball > 0.35
    head_high = z_ball > 1.25
    a = sq.attr
    is_pass = (kind == I_PASS_PLAYER) | (kind == I_PASS_SPACE)
    is_shot = kind == I_SHOT
    is_clear = kind == I_CLEARANCE

    kick = np.full(n, K.K_GROUND, dtype=np.int64)
    ptech = np.zeros(n, dtype=np.int64)
    ptype = np.zeros(n, dtype=np.int64)
    stech = np.full(n, -1, dtype=np.int64)
    height = np.zeros(n, dtype=np.int64)
    rejected = np.zeros(n, dtype=bool)
    mc = max_curve(sq, p, kind)
    req = np.clip(curve, -1.0, 1.0)
    curve_applied = np.sign(req) * np.minimum(np.abs(req), mc)

    # ---- passes ------------------------------------------------------------------------------------------
    cross_zone = (np.abs(oy) >= 18.0) & (ox >= 20.0) & (ex >= 33.0) & (np.abs(ey) <= BOX_HALF_WIDTH)
    cross_zone &= (np.abs(ey) <= np.abs(oy) - 8.0) & (dist >= 8.0)
    corner = ctx.restart == CTX_CORNER
    cross_zone |= corner | ((ctx.restart == CTX_FREE_KICK) & (ex >= 33.0) & (np.abs(ey) <= BOX_HALF_WIDTH))
    cutback = (ox >= 46.0) & (np.abs(oy) >= 9.0) & (ex >= 36.0) & (ex <= ox - 3.0) & (np.abs(ey) <= 14.0)
    switch = np.abs(ey - oy) >= SWITCH_LATERAL_M
    back = (np.abs(ang) >= np.radians(135)) & (dist >= 3.0)
    through = (kind == I_PASS_SPACE) & (ex > ctx.last_def_x + 1.0) & (dist >= 10.0)
    long_ball = dist > 32.0
    blocked = ctx.lane_blocked & (dist > 9.0)

    pk = np.full(n, K.K_GROUND, dtype=np.int64)
    pk = np.where((power > 0.75) & (dist > 14.0) & ~blocked, K.K_DRIVEN, pk)
    pk = np.where(long_ball | switch, K.K_LOFTED, pk)
    pk = np.where(blocked & (dist <= 20.0) & (a["technique"][p] >= 55), K.K_CHIPPED, pk)
    pk = np.where(blocked & ((dist > 20.0) | (a["technique"][p] < 55)), K.K_LOFTED, pk)
    curled = np.abs(curve_applied) > 0.15
    pk = np.where(curled & ~long_ball & ~blocked, K.K_CURLED, pk)
    outside = (
        curled & (a["technique"][p] >= 80) & (a["curve"][p] >= 72) & (np.sign(curve_applied) == _natural_out(sq, p))
    )
    pk = np.where(outside & ~long_ball & ~blocked, K.K_OUTSIDE, pk)
    low_cross = (power > 0.7) | ((ex > 44.0) & (np.abs(ey) < 9.0) & (ox > 38.0))
    pk = np.where(cross_zone, np.where(low_cross & ~corner, K.K_CROSS_LOW, K.K_CROSS), pk)
    pk = np.where(cutback & ~cross_zone, K.K_GROUND, pk)
    pk = np.where(through & ~blocked & ~curled & ~cross_zone, K.K_GROUND, pk)
    pk = np.where(through & blocked & ~cross_zone, np.where(dist < 22, K.K_CHIPPED, K.K_LOFTED), pk)
    pk = np.where(head_high, K.K_HEADER_PASS, pk)
    pk = np.where(ctx.restart == CTX_THROW_IN, K.K_THROW_IN, pk)
    gk_hands = ctx.restart == CTX_GK_HANDS
    pk = np.where(gk_hands, np.where(dist < 34.0, K.K_GK_THROW, np.where(power > 0.6, K.K_PUNT, K.K_DROP_KICK)), pk)
    pk = np.where((ctx.restart == CTX_GOAL_KICK), np.where(dist < 28.0, K.K_GROUND, K.K_LOFTED), pk)

    # technique overrides (radial picker)
    if pass_override is not None:
        ov = pass_override
        want = ov >= 0
        avail = available_pass_mask(sq, p, dist, mc)
        ok = want & avail[np.arange(n), np.maximum(ov, 0)] & ~head_high & (ctx.restart == CTX_OPEN)
        rejected |= is_pass & want & ~ok
        map_k = np.array([K.K_GROUND, K.K_DRIVEN, K.K_LOFTED, K.K_CHIPPED, K.K_CURLED, K.K_OUTSIDE, K.K_GROUND])
        pk = np.where(is_pass & ok, map_k[np.maximum(ov, 0)], pk)
        curve_applied = np.where(
            is_pass & ok & (ov == 4) & (np.abs(curve_applied) < 0.2), 0.35 * np.sign(req + 1e-9), curve_applied
        )
        curve_applied = np.clip(curve_applied, -mc, mc)

    tech = np.zeros(n, dtype=np.int64)
    tech = np.where(pk == K.K_DRIVEN, 1, tech)
    tech = np.where(np.isin(pk, [K.K_LOFTED, K.K_CROSS, K.K_PUNT, K.K_DROP_KICK, K.K_GK_THROW, K.K_THROW_IN]), 2, tech)
    tech = np.where(pk == K.K_CROSS_LOW, 1, tech)
    tech = np.where(pk == K.K_CHIPPED, 3, tech)
    tech = np.where(pk == K.K_CURLED, 4, tech)
    tech = np.where(pk == K.K_OUTSIDE, 5, tech)
    tech = np.where(pk == K.K_HEADER_PASS, 0, tech)
    tech = np.where(pk == K.K_GK_THROW, 0, tech)
    if pass_override is not None:
        tech = np.where(is_pass & (pass_override == 6) & ~rejected, 6, tech)  # backheel
    ptype = np.where(back, 6, ptype)
    first_time_air = ctx.first_time & airborne
    ptype = np.where(first_time_air & (np.abs(ang) <= np.radians(60)) & head_high, 5, ptype)  # flick-on
    layoff = ctx.first_time & (dist <= 15.0) & (ex <= ox - 2.0)
    ptype = np.where(layoff & (ptype != 5), 4, ptype)
    ptype = np.where(through, 3, ptype)
    ptype = np.where(cross_zone & ~cutback, 1, ptype)
    ptype = np.where(cutback, 2, ptype)
    height = np.where(np.isin(pk, [K.K_LOFTED, K.K_CHIPPED, K.K_CROSS, K.K_PUNT, K.K_DROP_KICK, K.K_THROW_IN]), 2, 0)
    height = np.where(np.isin(pk, [K.K_DRIVEN, K.K_CROSS_LOW, K.K_GK_THROW, K.K_HEADER_PASS, K.K_CURLED]), 1, height)
    kick = np.where(is_pass, pk, kick)
    ptech = np.where(is_pass, tech, 0)

    # ---- shots -------------------------------------------------------------------------------------------
    gx = HALF_LENGTH
    sdist = np.hypot(gx - ox, oy)
    gk_x_t = ctx.gk_pos[:, 0] * s
    gk_off_line = gx - gk_x_t
    st = np.where(sdist < 12.0, 1, np.where(a["shot_power"][p] > a["finishing"][p] + 4, 0, 1))  # placed / driven
    st = np.where(sdist > 22.0, 0, st)
    finesse = (a["curve"][p] >= 68) & (a["finishing"][p] >= 62) & (np.abs(oy) > 5.0) & (sdist > 13) & (sdist < 26)
    st = np.where(finesse, 2, st)
    chip = (gk_off_line > 9.0) & (sdist > 13) & (sdist < 30) & (a["technique"][p] >= 62)
    st = np.where(chip, 3, st)
    toe = (ctx.nearest_opp_m < 1.1) & (sdist < 14.0)
    st = np.where(toe & ~chip, 8, st)
    st = np.where(airborne & ~head_high, np.where(ctx.bounced_recently, 5, 4), st)
    st = np.where(ctx.bounced_recently & ~airborne & (z_ball > 0.2), 5, st)
    st = np.where(head_high, 6, st)
    st = np.where(ctx.restart == CTX_PENALTY, np.where(a["shot_power"][p] > 80, 0, 1), st)
    st = np.where(ctx.restart == CTX_FREE_KICK, np.where(a["curve"][p] >= 60, 2, 0), st)
    if shot_override is not None:
        want = shot_override >= 0
        avail_s = available_shot_mask(sq, p, sdist, z_ball, ctx.bounced_recently, gk_off_line)
        ok = want & avail_s[np.arange(n), np.maximum(shot_override, 0)]
        rejected |= is_shot & want & ~ok
        st = np.where(is_shot & ok, shot_override, st)
    shot_kick = np.array(
        [
            K.K_SHOT_DRIVEN,
            K.K_SHOT_PLACED,
            K.K_SHOT_FINESSE,
            K.K_SHOT_CHIP,
            K.K_VOLLEY,
            K.K_HALF_VOLLEY,
            K.K_HEADER,
            K.K_HEADER,
            K.K_TOE_POKE,
            K.K_SHOT_PLACED,
            K.K_VOLLEY,
        ]
    )
    kick = np.where(is_shot, shot_kick[np.maximum(st, 0)], kick)
    stech = np.where(is_shot, st, -1)
    # finesse shots bend by default toward the far post
    finesse_now = is_shot & (st == 2)
    natural = np.where(sq.left_foot[p], 1.0, -1.0)  # left foot curls right (+), right foot curls left (-)
    curve_applied = np.where(finesse_now & (np.abs(curve_applied) < 0.2), natural * np.minimum(0.7, mc), curve_applied)
    curve_applied = np.where(is_shot & ~np.isin(st, [2, 0, 1]), 0.0, curve_applied)
    if (is_pass & cross_zone).any():
        # whipped crosses get the kicking foot's natural curl
        curve_applied = np.where(is_pass & cross_zone, natural * np.maximum(0.45, np.minimum(mc, 0.7)), curve_applied)

    kick = np.where(is_clear, np.where(head_high, K.K_HEADER_PASS, K.K_CLEARANCE), kick)
    kick = np.where(kind == I_CARRY, -1, kick)

    # ---- body part and weak foot --------------------------------------------------------------------------
    strong_left = sq.left_foot[p]
    side_natural_left = oy > 0  # on the left (team frame), the left foot is the natural crossing / wide foot
    wide_play = cross_zone | (np.abs(oy) > 22.0)
    two_footed = sq.weak_foot[p] >= 4
    use_left = np.where(wide_play & two_footed, side_natural_left, strong_left)
    if rng is not None:
        flip = rng.random(n) < (0.05 + 0.035 * sq.weak_foot[p])
        use_left = np.where(flip, ~use_left, use_left)
    weak = use_left != strong_left
    body = np.where(use_left, 1, 0)
    body = np.where(np.isin(kick, [K.K_HEADER, K.K_HEADER_PASS]), 2, body)
    body = np.where(np.isin(kick, [K.K_THROW_IN, K.K_GK_THROW]), 5, body)
    weak &= body <= 1

    # ---- skill and noise ----------------------------------------------------------------------------------
    quality = skill_for(sq, p, kick, dist, sdist, kind)
    wf_level = np.where(weak, (5.0 - sq.weak_foot[p]) / 4.0, 0.0)
    mult = K.noise_multiplier(quality, ctx.pressure, _q(sq, "composure", p), wf_level)
    # curled balls near a player's curve cap come out less accurate
    near_cap = np.where(mc > 0, np.abs(curve_applied) / np.maximum(mc, 1e-6), 0.0)
    mult = mult * (1.0 + 0.25 * near_cap * np.isin(kick, [K.K_CURLED, K.K_OUTSIDE, K.K_SHOT_FINESSE]))
    mult = mult * np.where(ctx.first_time & ~is_shot, 1.1, 1.0)

    tgt = target.copy()
    # passes aimed at a player: lofted balls drop at the feet, crosses arrive at head height
    tgt[:, 2] = np.where(is_pass, 0.0, tgt[:, 2])
    tgt[:, 2] = np.where(is_pass & np.isin(kick, [K.K_CROSS]), 1.7, tgt[:, 2])
    tgt[:, 2] = np.where(is_pass & np.isin(kick, [K.K_CROSS_LOW]), 0.6, tgt[:, 2])
    tgt[:, 2] = np.where(is_pass & np.isin(kick, [K.K_LOFTED, K.K_PUNT, K.K_DROP_KICK]), 0.4, tgt[:, 2])
    tgt[:, 2] = np.where(is_shot, np.clip(tgt[:, 2], BALL_RADIUS, GOAL_HEIGHT - BALL_RADIUS), tgt[:, 2])
    return Resolved(
        player=p,
        kind=kind,
        kick=kick,
        pass_tech=ptech,
        pass_type=np.where(is_pass, ptype, 0),
        shot_tech=stech,
        body=body,
        weak_foot=weak,
        curve_applied=curve_applied,
        max_curve=mc,
        quality=quality,
        noise_mult=mult,
        target=tgt,
        override_rejected=rejected,
        pass_height=height,
    )


def _natural_out(sq: Squad, p: IA) -> F:
    """Bend direction of an outside-of-the-boot ball: right foot bends right (+1), left foot left (-1)."""
    return np.where(sq.left_foot[p], -1.0, 1.0)


def skill_for(sq: Squad, p: IA, kick: IA, dist: F, shot_dist: F, kind: IA) -> F:
    a = sq.attr
    sp, lp = a["short_passing"][p], a["long_passing"][p]
    passing = np.where(dist < 25.0, sp, 0.4 * sp + 0.6 * lp)
    v = np.full(len(p), 60.0)
    v = np.where(np.isin(kick, [K.K_GROUND, K.K_DRIVEN]), passing, v)
    v = np.where(kind == I_PASS_SPACE, 0.6 * passing + 0.4 * a["vision"][p], v)
    v = np.where(np.isin(kick, [K.K_LOFTED]), 0.8 * lp + 0.2 * a["vision"][p], v)
    v = np.where(kick == K.K_CHIPPED, 0.5 * lp + 0.5 * a["technique"][p], v)
    v = np.where(np.isin(kick, [K.K_CROSS, K.K_CROSS_LOW]), a["crossing"][p], v)
    v = np.where(kick == K.K_CURLED, 0.45 * a["curve"][p] + 0.35 * passing + 0.2 * a["vision"][p], v)
    v = np.where(kick == K.K_OUTSIDE, 0.5 * a["technique"][p] + 0.5 * a["curve"][p], v)
    fin = np.where(shot_dist < 18.0, a["finishing"][p], 0.4 * a["finishing"][p] + 0.6 * a["long_shots"][p])
    v = np.where(kick == K.K_SHOT_PLACED, fin, v)
    v = np.where(kick == K.K_SHOT_DRIVEN, 0.45 * a["shot_power"][p] + 0.55 * fin, v)
    v = np.where(kick == K.K_SHOT_FINESSE, 0.5 * a["curve"][p] + 0.5 * fin, v)
    v = np.where(kick == K.K_SHOT_CHIP, 0.5 * a["technique"][p] + 0.5 * a["finishing"][p], v)
    v = np.where(np.isin(kick, [K.K_VOLLEY, K.K_HALF_VOLLEY]), 0.7 * a["volleys"][p] + 0.3 * fin, v)
    v = np.where(np.isin(kick, [K.K_HEADER, K.K_HEADER_PASS]), a["heading"][p], v)
    v = np.where(kick == K.K_TOE_POKE, 0.5 * a["finishing"][p] + 0.5 * a["technique"][p], v)
    v = np.where(kick == K.K_CLEARANCE, 0.5 * lp + 0.5 * a["strength"][p], v)
    v = np.where(kick == K.K_THROW_IN, 0.7 * sp + 0.3 * a["strength"][p], v)
    v = np.where(np.isin(kick, [K.K_GK_THROW, K.K_PUNT, K.K_DROP_KICK]), a["gk_distribution"][p], v)
    # 0.5 corresponds to an average professional (rating ATTR_REF)
    return np.asarray(np.clip(0.5 + (v - ATTR_REF) / 60.0, 0.0, 1.0))


def available_pass_mask(sq: Squad, p: IA, dist: F, mc: F) -> BA:
    """(N, 7) which PASS_TECHNIQUES the player can choose here."""
    a = sq.attr
    n = len(p)
    m = np.ones((n, len(PASS_TECHNIQUES)), dtype=bool)
    m[:, 3] = (a["technique"][p] >= 55) & (dist <= 28.0)  # chipped
    m[:, 4] = mc >= 0.2  # curled
    m[:, 5] = (a["technique"][p] >= 78) & (a["curve"][p] >= 70) & (dist <= 40.0)  # outside of the boot
    m[:, 6] = (a["technique"][p] >= 65) & (dist <= 12.0)  # backheel
    return m


def available_shot_mask(sq: Squad, p: IA, dist: F, z: F, bounced: BA, gk_off_line: F) -> BA:
    """(N, 11) which SHOT_TECHNIQUES the player can choose here."""
    a = sq.attr
    n = len(p)
    m = np.zeros((n, len(SHOT_TECHNIQUES)), dtype=bool)
    ground = z <= 0.35
    head = z > 1.25
    m[:, 0] = ground | (~head & (z <= 0.7))  # driven
    m[:, 1] = ground  # placed
    m[:, 2] = ground & (a["curve"][p] >= 55) & (dist < 30.0)  # finesse
    m[:, 3] = ground & (a["technique"][p] >= 60) & (dist < 32.0) & (gk_off_line > 3.0)  # chip
    m[:, 4] = (z > 0.35) & ~head  # volley
    m[:, 5] = bounced & (z <= 0.6)  # half volley
    m[:, 6] = head  # header
    m[:, 8] = ground & (dist < 16.0)  # toe poke
    return m


def goal_target(sign: F, y: F, z: F) -> F:
    """Goal-mouth point (world frame) on the attacked goal line."""
    return np.column_stack([HALF_LENGTH * sign, y * sign, z])


def default_shot_aim(sign: F, ball: F, gk_pos: F, rng: np.random.Generator | None) -> F:
    """AI aim: the corner farther from the keeper, a little inside the post, low or high."""
    gy = gk_pos[:, 1] * sign
    side = np.where(gy > ball[:, 1] * sign * 0.2, -1.0, 1.0)
    n = len(sign)
    if rng is None:
        y = side * (HALF_GOAL - 0.7)
        z = np.full(n, 0.5)
    else:
        y = side * rng.uniform(HALF_GOAL - 2.2, HALF_GOAL - 0.8, n)
        z = np.where(rng.random(n) < 0.6, rng.uniform(0.2, 0.7, n), rng.uniform(1.0, 2.1, n))
    return goal_target(sign, y, z)


def in_box(x_t: F, y_t: F) -> BA:
    return np.asarray((x_t >= BOX_X) & (np.abs(y_t) <= BOX_HALF_WIDTH))
