"""Scoring options with the analytics models (shared.model_api), plus the player adjustment.

The models give probabilities for an average professional in the attacking frame. The simulator shifts them on
the logit scale by the relevant attributes (rating 70 = no shift), and that player-adjusted number is both what
Rewind displays and what the outcome resolution samples, so the two agree by construction.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from shared.model_api import (
    PASS_KIND,
    SHOT_BODY_PART,
    SHOT_SITUATION,
    SHOT_TECHNIQUE,
    AnalyticsModels,
    PassFeatures,
    PitchControlParams,
    ShotFeatures,
)
from sim import kick as K
from sim.constants import HALF_GOAL, HALF_LENGTH
from sim.resolver import ATTR_REF, SHOT_TECHNIQUES, Resolved
from sim.world import Squad

F = npt.NDArray[np.float64]
IA = npt.NDArray[np.int64]
BA = npt.NDArray[np.bool_]

SIT_OPEN, SIT_COUNTER, SIT_SET_PIECE, SIT_DIRECT_FK, SIT_PENALTY = (
    SHOT_SITUATION["open_play"],
    SHOT_SITUATION["counter_attack"],
    SHOT_SITUATION["set_piece"],
    SHOT_SITUATION["direct_free_kick"],
    SHOT_SITUATION["penalty"],
)


def logit(p: F) -> F:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.asarray(np.log(p / (1 - p)))


def sigmoid(z: F) -> F:
    return np.asarray(1.0 / (1.0 + np.exp(-z)))


def lane_count(origin: F, target: F, opp: F, width: float = 2.0) -> IA:
    """Opponents within ``width`` of the straight segment origin->target. origin/target (N, 2), opp (N, M, 2)."""
    d = target - origin
    ll = np.maximum(np.sum(d * d, axis=1), 1e-9)
    rel = opp - origin[:, None, :]
    s = np.sum(rel * d[:, None, :], axis=2) / ll[:, None]
    closest = origin[:, None, :] + np.clip(s, 0, 1)[..., None] * d[:, None, :]
    dist = np.linalg.norm(opp - closest, axis=2)
    inside = (s > 0.08) & (s < 0.97) & (dist < width)
    return np.asarray(inside.sum(axis=1), dtype=np.int64)


def cone_count(shooter: F, opp: F) -> IA:
    """Opponents inside the triangle shooter - post - post (attacking frame, goal at +52.5)."""
    a = shooter
    b = np.array([HALF_LENGTH, HALF_GOAL])
    c = np.array([HALF_LENGTH, -HALF_GOAL])

    def cross(o: F, p1: F, p2: F) -> F:
        return np.asarray(
            (p1[..., 0] - o[..., 0]) * (p2[..., 1] - o[..., 1]) - (p1[..., 1] - o[..., 1]) * (p2[..., 0] - o[..., 0])
        )

    pa = a[:, None, :]
    d1 = cross(opp, pa, np.broadcast_to(b, opp.shape))
    d2 = cross(opp, np.broadcast_to(b, opp.shape), np.broadcast_to(c, opp.shape))
    d3 = cross(opp, np.broadcast_to(c, opp.shape), pa)
    neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    return np.asarray((~(neg & pos)).sum(axis=1), dtype=np.int64)


@dataclass
class Scored:
    p: F
    """Player-adjusted success probability."""
    p0: F
    """Average-player model probability."""
    value_success: F
    value_fail: F
    xg: F
    """xG of the option (shots), else 0."""
    xt_end: F
    p_base: F | None = None
    """Same model for an average-attribute player (race term included for passes)."""


def pass_kind_code(res: Resolved, switch: BA) -> IA:
    k = np.zeros(len(res.kind), dtype=np.int64)
    k = np.where(res.pass_type == 6, PASS_KIND["back_pass"], k)
    k = np.where(switch, PASS_KIND["switch"], k)
    k = np.where(res.pass_tech == 3, PASS_KIND["chipped"], k)
    k = np.where(res.pass_type == 3, PASS_KIND["through_ball"], k)
    k = np.where(res.pass_type == 2, PASS_KIND["cutback"], k)
    k = np.where(res.pass_type == 1, PASS_KIND["cross"], k)
    return k


def score_passes(
    models: AnalyticsModels,
    sq: Squad,
    res: Resolved,
    origin_t: F,
    end_t: F,
    opp_t: F,
    pressure: F,
    race: F | None = None,
) -> Scored:
    """origin_t / end_t (N, 2) and opp_t (N, 11, 2) in the passer's attacking frame.

    ``race`` (logit, <= 0 matters) is the time-to-intercept race from ``race_logit``: passes a defender can
    clearly cut out are less likely to arrive than the event-data model alone says.
    """
    p_idx = res.player
    lanes = lane_count(origin_t, end_t, opp_t)
    near_recv = np.min(np.linalg.norm(opp_t - end_t[:, None, :], axis=2), axis=1)
    switch = np.abs(end_t[:, 1] - origin_t[:, 1]) >= 34.0
    feats = PassFeatures(
        x=origin_t[:, 0],
        y=origin_t[:, 1],
        end_x=end_t[:, 0],
        end_y=end_t[:, 1],
        height=res.pass_height.astype(np.int64),
        kind=pass_kind_code(res, switch),
        nearest_defender_to_receiver_m=near_recv,
        defenders_in_lane=lanes,
        under_pressure=pressure > 0.45,
    )
    p0 = np.asarray(models.xpass(feats), dtype=np.float64)
    wf = np.where(res.weak_foot, (5.0 - sq.weak_foot[p_idx]) / 4.0, 0.0)
    delta = 1.35 * (res.quality - 0.5) + 0.5 * pressure * (sq.attr["composure"][p_idx] / 99.0 - 0.7) - 0.3 * wf
    race_term = np.zeros(len(p0))
    if race is not None:
        # the event-data model is reliable for ordinary passes and weak on crosses / through / lofted balls
        # (Step 2 report): there the time-to-intercept physics carries more of the weight
        weak_model = (res.pass_type == 1) | (res.pass_type == 3) | (res.pass_height == 2)
        weight = np.where(weak_model, RACE_WEIGHT_WEAK_MODEL, RACE_WEIGHT)
        race_term = weight * np.clip(race, -6.0, 0.0)
    p = sigmoid(logit(p0) + delta + race_term)
    p_base = sigmoid(logit(p0) + race_term)
    xt_end = np.asarray(models.xt(end_t[:, 0], end_t[:, 1]), dtype=np.float64)
    xt_opp = np.asarray(models.xt(-end_t[:, 0], -end_t[:, 1]), dtype=np.float64)
    return Scored(
        p=p,
        p0=p0,
        value_success=xt_end,
        value_fail=-0.6 * xt_opp - 0.004,
        xg=np.zeros(len(p)),
        xt_end=xt_end,
        p_base=p_base,
    )


RACE_WEIGHT = 0.35
RACE_WEIGHT_WEAK_MODEL = 0.9
RACE_SAMPLES = np.linspace(0.1, 1.0, 10)
_GROUND_KICKS = (K.K_GROUND, K.K_CURLED, K.K_OUTSIDE)
_LOW_KICKS = (K.K_DRIVEN, K.K_CROSS_LOW, K.K_GK_THROW)


def race_logit(
    sq: Squad,
    res: Resolved,
    origin: F,
    end: F,
    power: F,
    opp_idx: IA,
    opp_pos: F,
    opp_vel: F,
    rec_idx: IA,
    rec_pos: F,
    rec_vel: F,
    pc: PitchControlParams,
) -> F:
    """Pitch-control style race for each pass (world frame, N rows; opponents (N, 11, 2)).

    Ball schedule along the straight path from the kick model; players' time-to-intercept is Spearman's
    (pitch-control reaction time and speed from ``pc``). Returns pi / sqrt(3) / sigma * (defender slack - receiver
    lateness): negative when an opponent can reach the ball before the receiver gets it.
    """
    n = len(origin)
    if n == 0:
        return np.zeros(0)
    d_vec = end - origin
    dist = np.maximum(np.linalg.norm(d_vec, axis=1), 0.5)
    kick = res.kick
    ground = np.isin(kick, _GROUND_KICKS)
    low = np.isin(kick, _LOW_KICKS)
    v_ground = np.minimum(K.ground_speed_for(dist, 2.5 + 4.5 * power), 24.0)
    v_drive = 17.0 + 11.0 * power
    sk = RACE_SAMPLES
    dk = dist[:, None] * sk[None, :]
    t_ground = K.ground_time(np.repeat(v_ground, len(sk)), dk.ravel()).reshape(n, len(sk))
    t_drive = dk / np.maximum(v_drive[:, None] * 0.85, 1.0)
    origin3 = np.column_stack([origin, np.full(n, 0.11)])
    end3 = np.column_stack([end, np.zeros(n)])
    air = ~(ground | low)
    t_air = np.zeros((n, len(sk)))
    if air.any():
        plan = K.plan_kicks(origin3[air], end3[air], kick[air], power[air], np.zeros(int(air.sum())))
        tf = K.flight_time(plan.launch.speed, plan.launch.elev, dist[air])
        tf = np.where(np.isfinite(tf), tf, dist[air] / 15.0)
        t_air[air] = tf[:, None] * sk[None, :]
    t_ball = np.where(ground[:, None], t_ground, np.where(low[:, None], t_drive, t_air))
    t_ball = np.where(np.isfinite(t_ball), t_ball, 99.0)
    reachable = np.ones((n, len(sk)), dtype=bool)
    reachable[air] = (sk[None, :] <= 0.12) | (sk[None, :] >= 0.8)
    pts = origin[:, None, :] + d_vec[:, None, :] * sk[None, :, None]  # (N, K, 2)
    react, vmax = pc.reaction_time_s, pc.max_player_speed_mps
    ahead = opp_pos + opp_vel * react
    dd = np.linalg.norm(pts[:, :, None, :] - ahead[:, None, :, :], axis=3) - 0.9  # (N, K, 11)
    tti = react + np.maximum(dd, 0.0) / vmax
    slack = np.where(reachable[..., None], tti - t_ball[..., None], 99.0)
    slack_d = slack.reshape(n, -1).min(axis=1)
    ra = rec_pos + rec_vel * react
    dr = np.maximum(np.linalg.norm(end - ra, axis=1) - 0.9, 0.0)
    slack_r = react + dr / vmax - t_ball[:, -1]
    slack_r = np.where(rec_idx >= 0, slack_r, 0.0)
    return np.asarray(np.pi / np.sqrt(3.0) / pc.tti_sigma * (slack_d - np.maximum(slack_r, 0.0)))


def score_shots(
    models: AnalyticsModels,
    sq: Squad,
    res: Resolved,
    shooter_t: F,
    gk_t: F,
    keeper: IA,
    opp_t: F,
    pressure: F,
    situation: IA,
    first_time: BA,
) -> Scored:
    names = [SHOT_TECHNIQUES[max(int(s), 0)] for s in res.shot_tech]
    tech = np.array([SHOT_TECHNIQUE[nm] for nm in names], dtype=np.int64)
    body = np.where(res.body == 2, SHOT_BODY_PART["head"], np.where(res.body == 4, SHOT_BODY_PART["other"], 0))
    outfield = opp_t
    nearest = np.min(np.linalg.norm(outfield - shooter_t[:, None, :], axis=2), axis=1)
    feats = ShotFeatures(
        x=shooter_t[:, 0],
        y=shooter_t[:, 1],
        body_part=body.astype(np.int64),
        technique=tech,
        situation=situation.astype(np.int64),
        gk_x=gk_t[:, 0],
        gk_y=gk_t[:, 1],
        n_defenders_in_cone=cone_count(shooter_t, outfield),
        nearest_defender_m=nearest,
        under_pressure=pressure > 0.45,
        first_time=first_time,
    )
    x0 = np.asarray(models.xg(feats), dtype=np.float64)
    p = shot_adjust(sq, res, keeper, x0, pressure, situation)
    return Scored(
        p=p, p0=x0, value_success=np.ones(len(p)), value_fail=np.full(len(p), -0.01), xg=x0, xt_end=x0, p_base=x0
    )


def shot_adjust(sq: Squad, res: Resolved, keeper: IA, x0: F, pressure: F, situation: IA) -> F:
    p_idx = res.player
    gk = (sq.attr["gk_reflexes"][keeper] + sq.attr["gk_diving"][keeper] + sq.attr["gk_positioning"][keeper]) / 3
    wf = np.where(res.weak_foot, (5.0 - sq.weak_foot[p_idx]) / 4.0, 0.0)
    rating = (res.quality - 0.5) * 60.0 + ATTR_REF
    delta = 0.35 * (rating - ATTR_REF) / 20.0 - 0.3 * (gk - ATTR_REF) / 20.0 - 0.25 * wf
    delta = delta + 0.4 * pressure * (sq.attr["composure"][p_idx] / 99.0 - 0.7)
    delta = np.where(situation == SIT_PENALTY, 0.25 * delta, delta)
    return sigmoid(logit(x0) + delta)


def carry_retention(sq: Squad, carrier: IA, n_close: F, pressure: F, dribble: BA) -> F:
    a = sq.attr
    z = 2.6 - 1.1 * n_close - 0.8 * pressure + 0.035 * (a["dribbling"][carrier] - ATTR_REF)
    z = z + 0.02 * (a["strength"][carrier] - ATTR_REF) * (~dribble)
    z = np.where(dribble, z - 1.6, z)
    return sigmoid(z)
