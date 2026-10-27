"""Flight planning: who can reach a ball in flight, and how a pass or shot resolves.

At release the simulator predicts the (noisy) ball path with the same physics that will then move the ball, and
computes for every player the earliest tick at which he can reach it: reaction time, then the time to cover the
distance from his projected position with his acceleration and top speed (the time-to-intercept of Spearman's
pitch-control model, evaluated with each player's own physical limits). Outcomes are then sampled so that their
rates match the player-adjusted model probabilities, and the physics renders them: the designated collector runs
to the reachable point on the path and takes the ball there.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

from sim.constants import BALL_RADIUS, DT, GOAL_HEIGHT, HALF_GOAL, HALF_LENGTH, HALF_WIDTH
from sim.physics import step_ball

F = npt.NDArray[np.float64]
IA = npt.NDArray[np.int64]
BA = npt.NDArray[np.bool_]

NEVER: Final = 10**6
CONTROL_REACH: Final = 0.9
MAX_CONTROL_SPEED: Final = 34.0


@dataclass
class Paths:
    xs: F
    """(T + 1, N, 3) positions, index 0 = launch."""
    vs: F
    wood: BA
    """(T + 1, N) woodwork contact on the step ending at that index."""
    out_k: IA
    """(N,) first index where the ball has left the field of play (NEVER if it stays in)."""
    goal_k: IA
    """(N,) first index where the ball is fully over the goal line between the posts and under the bar."""
    goal_side: F
    """(N,) sign of x at goal_k."""


def predict_paths(pos: F, vel: F, spin: F, steps: int) -> Paths:
    x, v, w = pos.astype(np.float64).copy(), vel.astype(np.float64).copy(), spin.astype(np.float64).copy()
    n = len(x)
    xs = np.empty((steps + 1, n, 3))
    vs = np.empty((steps + 1, n, 3))
    wood = np.zeros((steps + 1, n), dtype=bool)
    xs[0], vs[0] = x, v
    for k in range(1, steps + 1):
        f = step_ball(x, v, w)
        xs[k], vs[k], wood[k] = x, v, f.woodwork
        if k % 10 == 0:
            # stop early once every ball has come to rest or left the field by a margin
            gone = (np.abs(x[:, 0]) > HALF_LENGTH + 3.0) | (np.abs(x[:, 1]) > HALF_WIDTH + 3.0)
            still = np.linalg.norm(v, axis=1) < 0.05
            if (gone | still).all():
                xs[k + 1 :], vs[k + 1 :] = x, v
                break
    over_goal = np.abs(xs[..., 0]) > HALF_LENGTH + BALL_RADIUS
    in_mouth = (np.abs(xs[..., 1]) < HALF_GOAL) & (xs[..., 2] < GOAL_HEIGHT)
    # the first time the ball leaves the field decides: in the mouth it is a goal, anywhere else it is out
    # (a ball over the bar that drops onto the net behind is out, as the match engine judges it)
    leaves = over_goal | (np.abs(xs[..., 1]) > HALF_WIDTH + BALL_RADIUS)
    first = np.where(leaves.any(axis=0), np.argmax(leaves, axis=0), NEVER)
    k_first = np.minimum(first, steps)
    cols = np.arange(n)
    scored = (first < NEVER) & over_goal[k_first, cols] & in_mouth[k_first, cols]
    goal_k = np.where(scored, first, NEVER)
    out_k = np.where(scored, NEVER, first)
    side = np.sign(xs[np.minimum(goal_k, steps), np.arange(n), 0])
    return Paths(xs=xs, vs=vs, wood=wood, out_k=out_k.astype(np.int64), goal_k=goal_k.astype(np.int64), goal_side=side)


def kinematic_time(d: F, vmax: F, amax: F) -> F:
    """Time to cover ``d`` from rest with constant acceleration up to top speed."""
    d1 = vmax**2 / (2 * amax)
    return np.asarray(np.where(d <= d1, np.sqrt(2 * np.maximum(d, 0) / amax), vmax / amax + (d - d1) / vmax))


def reach_ticks(
    path: F,
    vel_path: F,
    pos: F,
    vel: F,
    react: F,
    vmax: F,
    amax: F | None,
    reach_z: F,
    earliest: IA,
    reach_xy: F | float = CONTROL_REACH,
    stop_k: int = NEVER,
) -> IA:
    """(P,) first path index each player can reach (NEVER if none before ``stop_k``).

    path / vel_path (T + 1, 3); pos / vel (P, 2); the rest (P,). With ``amax=None`` this is exactly Spearman's
    time-to-intercept (reaction, then straight-line running at ``vmax``), which is how the simulator resolves
    interceptions so they stay consistent with the pitch-control model.
    """
    t_k = np.arange(len(path)) * DT
    ahead = pos + vel * react[:, None]  # where momentum takes them during the reaction time
    d = np.linalg.norm(path[:, None, :2] - ahead[None, :, :], axis=2) - np.asarray(reach_xy)
    if amax is None:
        need = react[None, :] + np.maximum(d, 0.0) / vmax[None, :]
    else:
        need = react[None, :] + kinematic_time(np.maximum(d, 0.0), vmax[None, :], amax[None, :])
        # a player running the wrong way has to brake and turn first
        to = path[:, None, :2] - pos[None, :, :]
        u = to / np.maximum(np.linalg.norm(to, axis=2, keepdims=True), 1e-6)
        away = np.maximum(-np.sum(u * vel[None, :, :], axis=2), 0.0)
        need = need + 1.5 * away / amax[None, :]
    ok = need <= t_k[:, None] + 1e-9
    ok &= path[:, None, 2] <= reach_z[None, :]
    ok &= np.linalg.norm(vel_path, axis=1)[:, None] <= MAX_CONTROL_SPEED
    k_idx = np.arange(len(path))[:, None]
    ok &= k_idx >= earliest[None, :]
    ok &= k_idx < stop_k
    first = np.where(ok.any(axis=0), np.argmax(ok, axis=0), NEVER)
    return np.asarray(first, dtype=np.int64)


@dataclass
class PassPlan:
    collector: int
    mode: int
    k: int
    sub: int


def choose_interceptor(k_opp: IA, opp_idx: IA, rng: np.random.Generator, sigma_ticks: float) -> int:
    """Among reachable opponents, earlier reachers are likelier (logistic in arrival time, pitch-control style)."""
    ok = k_opp < NEVER
    if not ok.any():
        return -1
    kk = k_opp[ok].astype(float)
    w = np.exp(-(kk - kk.min()) / max(sigma_ticks, 1e-6))
    return int(opp_idx[ok][int(rng.choice(int(ok.sum()), p=w / w.sum()))])


def interception_time_sigma(tti_sigma: float) -> float:
    """Spread (ticks) of the interceptor choice, from the pitch-control time-to-intercept uncertainty."""
    return float(tti_sigma * np.sqrt(3.0) / np.pi / DT)
