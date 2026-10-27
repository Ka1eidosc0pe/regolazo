"""Player kinematics measured from Metrica tracking: top speed, acceleration, a reaction-time proxy, distance covered.

Also validates pitch control on real frames: at the start of each pass, the passing team's pitch control at the
pass destination should be higher for completed passes than for passes that were lost.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.metrics import roc_auc_score  # type: ignore[import-untyped]

from analytics.data import metrica
from analytics.pitch_control import pitch_control
from shared.model_api import PitchControlParams

FloatArray = npt.NDArray[np.float64]

MIN_MINUTES_FOR_DISTANCE = 85.0
SPEED_WINDOW_S = 1.0
"""Top speed is the best 1 s rolling mean (robust to single-frame tracking glitches)."""
ACCEL_WINDOW_S = 0.5
ACCEL_PERCENTILE = 99.9
REACTION_ACCEL_THRESHOLD = 1.5
"""Acceleration towards the pass target (m/s^2) that marks a defender's reaction onset."""


@dataclass
class GameKinematics:
    game: int
    team: str
    players: list[str]
    gk_index: int
    pos: FloatArray
    vel: FloatArray
    period: npt.NDArray[np.int64]
    frame: npt.NDArray[np.int64]
    ball: FloatArray


def load_game(game: int) -> dict[str, GameKinematics]:
    out: dict[str, GameKinematics] = {}
    for team in ("Home", "Away"):
        tr = metrica.load_tracking(game, team)
        vel = metrica.velocities(tr.pos, tr.period)
        out[team] = GameKinematics(
            game, team, tr.players, metrica.goalkeeper_index(tr), tr.pos, vel, tr.period, tr.frame, tr.ball
        )
    return out


def _rolling_mean(a: FloatArray, n: int) -> FloatArray:
    kernel = np.ones(n) / n
    return np.convolve(a, kernel, mode="valid") if a.size >= n else np.array([])


def player_kinematics(g: GameKinematics) -> list[dict[str, float]]:
    """Per outfield player: top speed, peak acceleration, minutes and distance covered."""
    rows: list[dict[str, float]] = []
    n_speed = round(SPEED_WINDOW_S * metrica.FPS)
    n_acc = round(ACCEL_WINDOW_S * metrica.FPS)
    for j in range(len(g.players)):
        if j == g.gk_index:
            continue
        on = ~np.isnan(g.pos[:, j, 0])
        minutes = on.sum() / metrica.FPS / 60.0
        if minutes < 10:
            continue
        speed = np.linalg.norm(g.vel[:, j], axis=1)
        top = 0.0
        acc_peak = 0.0
        dist = 0.0
        for per in np.unique(g.period):
            sel = (g.period == per) & ~np.isnan(speed)
            s = speed[sel]
            if s.size < n_speed:
                continue
            top = max(top, float(np.max(_rolling_mean(s, n_speed))))
            # Peak acceleration: speed gain over 0.5 s of a 0.5 s-smoothed speed, at the ACCEL_PERCENTILE (99.9th)
            # percentile (the raw maximum is dominated by tracking glitches).
            s_sm = _rolling_mean(s, n_acc)
            if s_sm.size > n_acc:
                gain = (s_sm[n_acc:] - s_sm[:-n_acc]) / ACCEL_WINDOW_S
                acc_peak = max(acc_peak, float(np.percentile(gain, ACCEL_PERCENTILE)))
            dist += float(s.sum() / metrica.FPS)
        rows.append(
            {
                "game": float(g.game),
                "top_speed_mps": top,
                "peak_accel_mps2": acc_peak,
                "minutes": float(minutes),
                "distance_m": dist,
                "full_match": float(minutes >= MIN_MINUTES_FOR_DISTANCE),
            }
        )
    return rows


def reaction_times(game: int, kin: dict[str, GameKinematics], events: list[metrica.MetricaEvent]) -> list[float]:
    """Reaction-time proxy: delay from a pass being struck to the nearest defender (to the target) accelerating
    towards the target at > REACTION_ACCEL_THRESHOLD, for passes of >= 10 m lasting >= 0.8 s."""
    out: list[float] = []
    for ev in events:
        if ev.type != "PASS" or ev.end_time_s - ev.start_time_s < 0.8:
            continue
        if np.hypot(ev.end[0] - ev.start[0], ev.end[1] - ev.start[1]) < 10:
            continue
        defending = kin["Away" if ev.team == "Home" else "Home"]
        f0 = ev.start_frame - 1
        f1 = min(f0 + 40, defending.pos.shape[0] - 1)
        target = np.array(ev.end)
        p0 = defending.pos[f0]
        d = np.linalg.norm(p0 - target, axis=1)
        d[defending.gk_index] = np.inf
        if not np.isfinite(d).any():
            continue
        j = int(np.nanargmin(np.where(np.isnan(d), np.inf, d)))
        seg_p = defending.pos[f0:f1, j]
        seg_v = defending.vel[f0:f1, j]
        to_target = target - seg_p
        unit = to_target / np.maximum(np.linalg.norm(to_target, axis=1, keepdims=True), 1e-6)
        v_toward = np.sum(seg_v * unit, axis=1)
        if np.isnan(v_toward).any():
            continue
        a_toward = np.gradient(v_toward, 1.0 / metrica.FPS)
        if a_toward[0] > 0.5:
            continue  # already accelerating towards the target before the pass
        onset = np.flatnonzero(a_toward > REACTION_ACCEL_THRESHOLD)
        if onset.size:
            out.append(float(onset[0] / metrica.FPS))
    return out


def attacking_frame_snapshot(
    kin: dict[str, GameKinematics], team: str, frame_idx: int
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, int, int, float]:
    """Positions/velocities at a frame rotated so ``team`` attacks +x. Returns att_pos, att_vel, def_pos, def_vel,
    ball, att_gk (index into att arrays), def_gk, and the sign applied (+1 or -1)."""
    att, dfn = kin[team], kin["Away" if team == "Home" else "Home"]
    period = int(att.period[frame_idx])
    sign = 1.0 if metrica.attacks_positive_x(_as_tracking(att), period, att.gk_index) else -1.0

    def take(g: GameKinematics) -> tuple[FloatArray, FloatArray, int]:
        on = ~np.isnan(g.pos[frame_idx, :, 0])
        idx = np.flatnonzero(on)
        gk = int(np.flatnonzero(idx == g.gk_index)[0]) if g.gk_index in idx else -1
        return g.pos[frame_idx, idx] * sign, np.nan_to_num(g.vel[frame_idx, idx]) * sign, gk

    ap, av, agk = take(att)
    dp, dv, dgk = take(dfn)
    return ap, av, dp, dv, att.ball[frame_idx] * sign, agk, dgk, sign


def _as_tracking(g: GameKinematics) -> metrica.TeamTracking:
    return metrica.TeamTracking(g.period, g.frame, g.frame / metrica.FPS, g.players, g.pos, g.ball)


def pass_control_validation(
    kin: dict[str, GameKinematics], events: list[metrica.MetricaEvent], params: PitchControlParams
) -> dict[str, Any]:
    """Pitch control of the passing team at the pass target, completed passes vs lost balls."""
    pc_ok: list[float] = []
    pc_lost: list[float] = []
    for ev in events:
        if ev.type not in ("PASS", "BALL LOST") or np.isnan(ev.end[0]) or np.isnan(ev.start[0]):
            continue
        if ev.type == "BALL LOST" and ev.subtype not in ("", "INTERCEPTION", "HEAD-INTERCEPTION", "CROSS-INTERCEPTION"):
            continue
        fi = ev.start_frame - 1
        ap, av, dp, dv, _ball, _, dgk, sign = attacking_frame_snapshot(kin, ev.team, fi)
        target = np.array([ev.end]) * sign
        start = np.array(ev.start) * sign
        res = pitch_control(ap, av, dp, dv, start, target, params, def_gk=dgk if dgk >= 0 else None)
        (pc_ok if ev.type == "PASS" else pc_lost).append(float(res.normalized_att[0]))
    y = np.r_[np.ones(len(pc_ok)), np.zeros(len(pc_lost))]
    s = np.r_[pc_ok, pc_lost]
    return {
        "n_completed": len(pc_ok),
        "n_lost": len(pc_lost),
        "mean_pc_completed": float(np.mean(pc_ok)),
        "mean_pc_lost": float(np.mean(pc_lost)),
        "auc_pc_at_target": float(roc_auc_score(y, s)),
    }
