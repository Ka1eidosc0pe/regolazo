"""Spearman (2018) physics-based pitch control, vectorized over target points.

Follows the Friends of Tracking reference implementation (Laurie Shaw, LaurieOnTracking ``Metrica_PitchControl.py``)
with its default parameters: each player keeps moving at their current velocity for the reaction time, then runs at
``max_player_speed`` to the target; arrival is uncertain with a logistic of std ``tti_sigma``; once there, a player
controls the ball at rate ``lambda`` (the defending goalkeeper 3x faster). The control ODE is integrated from the
ball's arrival time until the two teams' probabilities sum to ``1 - CONVERGE_TOL``.

Everything is in the attacking frame of the team in possession (attacking +x), so the defending goalkeeper is the one
near (+52.5, 0).
"""

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from shared.model_api import GOAL_X, PitchControlParams

FloatArray = npt.NDArray[np.float64]

INT_DT_S = 0.04
MAX_INT_TIME_S = 10.0
CONVERGE_TOL = 0.01
GK_LAMBDA_FACTOR = 3.0
TIME_TO_CONTROL_VETO = 3.0
GK_MAX_DIST_FROM_GOAL_M = 25.0
"""The defender nearest the defended goal is treated as the keeper if within this distance of the goal centre."""


@dataclass(frozen=True)
class ControlResult:
    att: FloatArray
    """Raw attacking-team control probability per point."""
    defn: FloatArray
    """Raw defending-team control probability per point. ``att + defn >= 1 - CONVERGE_TOL``."""

    @property
    def normalized_att(self) -> FloatArray:
        return np.asarray(self.att / np.maximum(self.att + self.defn, 1e-12), dtype=np.float64)


def time_to_control(params: PitchControlParams, lam: float) -> float:
    return TIME_TO_CONTROL_VETO * math.log(10) * (math.sqrt(3) * params.tti_sigma / math.pi + 1.0 / lam)


def defending_gk_index(def_pos: FloatArray) -> int | None:
    """Index of the defender acting as goalkeeper (nearest the defended goal at +x), if any is close enough."""
    if def_pos.shape[0] == 0:
        return None
    d = np.hypot(def_pos[:, 0] - GOAL_X, def_pos[:, 1])
    i = int(np.argmin(d))
    return i if d[i] <= GK_MAX_DIST_FROM_GOAL_M else None


def time_to_intercept(pos: FloatArray, vel: FloatArray, points: FloatArray, params: PitchControlParams) -> FloatArray:
    """(M, N) arrival time of each player at each point."""
    ahead = pos + vel * params.reaction_time_s
    d = np.linalg.norm(points[:, None, :] - ahead[None, :, :], axis=2)
    return np.asarray(params.reaction_time_s + d / params.max_player_speed_mps, dtype=np.float64)


def pitch_control(
    att_pos: FloatArray,
    att_vel: FloatArray,
    def_pos: FloatArray,
    def_vel: FloatArray,
    ball: FloatArray | None,
    points: FloatArray,
    params: PitchControlParams | None = None,
    def_gk: int | None = -1,
) -> ControlResult:
    """Pitch control at ``points`` (M x 2). ``def_gk=-1`` detects the keeper; ``None`` means no keeper bonus."""
    p = params or PitchControlParams()
    att_pos = np.asarray(att_pos, dtype=np.float64).reshape(-1, 2)
    def_pos = np.asarray(def_pos, dtype=np.float64).reshape(-1, 2)
    att_vel = np.nan_to_num(np.asarray(att_vel, dtype=np.float64).reshape(-1, 2))
    def_vel = np.nan_to_num(np.asarray(def_vel, dtype=np.float64).reshape(-1, 2))
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    m = points.shape[0]
    att_out = np.zeros(m)
    def_out = np.zeros(m)
    if att_pos.shape[0] == 0 or def_pos.shape[0] == 0:
        (att_out if def_pos.shape[0] == 0 else def_out)[:] = 1.0
        return ControlResult(att_out, def_out)

    if ball is None or np.any(np.isnan(ball)):
        btt = np.zeros(m)
    else:
        btt = np.linalg.norm(points - np.asarray(ball, dtype=np.float64)[:2], axis=1) / p.average_ball_speed_mps
    tti_a = time_to_intercept(att_pos, att_vel, points, p)
    tti_d = time_to_intercept(def_pos, def_vel, points, p)
    lam_d = np.full(def_pos.shape[0], p.lambda_def)
    gk = defending_gk_index(def_pos) if def_gk == -1 else def_gk
    if gk is not None:
        lam_d[gk] = p.lambda_def * GK_LAMBDA_FACTOR
    ttc_att = time_to_control(p, p.lambda_att)
    ttc_def = time_to_control(p, p.lambda_def)

    tau_a = tti_a.min(axis=1)
    tau_d = tti_d.min(axis=1)
    def_wins = tau_a - np.maximum(btt, tau_d) >= ttc_def
    att_wins = ~def_wins & (tau_d - np.maximum(btt, tau_a) >= ttc_att)
    def_out[def_wins] = 1.0
    att_out[att_wins] = 1.0
    active = np.flatnonzero(~(def_wins | att_wins))
    if active.size == 0:
        return ControlResult(att_out, def_out)

    ta, td, b = tti_a[active], tti_d[active], btt[active]
    use_a = (ta - tau_a[active, None]) < ttc_att
    use_d = (td - tau_d[active, None]) < ttc_def
    rate_a = np.where(use_a, p.lambda_att, 0.0)
    rate_d = np.where(use_d, lam_d[None, :], 0.0)
    k = math.pi / math.sqrt(3.0) / p.tti_sigma
    ppcf_a = np.zeros_like(ta)
    ppcf_d = np.zeros_like(td)
    tot_a = np.zeros(active.size)
    tot_d = np.zeros(active.size)
    n_steps = math.ceil((MAX_INT_TIME_S + INT_DT_S) / INT_DT_S)
    for i in range(1, n_steps):
        running = (1.0 - (tot_a + tot_d)) > CONVERGE_TOL
        if not running.any():
            break
        t = (b - INT_DT_S + i * INT_DT_S)[:, None]
        remaining = np.where(running, 1.0 - tot_a - tot_d, 0.0)[:, None]
        pa = 1.0 / (1.0 + np.exp(np.clip(-k * (t - ta), -50, 50)))
        pd = 1.0 / (1.0 + np.exp(np.clip(-k * (t - td), -50, 50)))
        ppcf_a += remaining * pa * rate_a * INT_DT_S
        ppcf_d += remaining * pd * rate_d * INT_DT_S
        tot_a = ppcf_a.sum(axis=1)
        tot_d = ppcf_d.sum(axis=1)
    att_out[active] = tot_a
    def_out[active] = tot_d
    return ControlResult(att_out, def_out)


def pitch_grid(nx: int = 32, ny: int = 21) -> FloatArray:
    """Cell centres of an nx x ny grid over the 105 x 68 pitch, as (nx * ny, 2), x-major."""
    xs = (np.arange(nx) + 0.5) / nx * 105.0 - 52.5
    ys = (np.arange(ny) + 0.5) / ny * 68.0 - 34.0
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    return np.column_stack([gx.ravel(), gy.ravel()])
