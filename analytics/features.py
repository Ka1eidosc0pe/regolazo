"""Freeze-frame geometry shared by training (StatsBomb) and inference (simulator scenes).

All coordinates are in the attacking frame (metres, centre origin, attacking +x). Opponent sets have variable size,
so they are passed as padded ``(N, K, 2)`` arrays with NaN rows for missing players.
"""

import numpy as np
import numpy.typing as npt

from shared.model_api import GOAL_X, HALF_GOAL

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

LANE_RADIUS_M = 2.0
NO_DEFENDER_M = 30.0
"""Distance reported when no opponent is visible (clips the long tail of the training data)."""


def pad_players(groups: list[list[tuple[float, float]]]) -> FloatArray:
    """Ragged per-row player lists -> ``(N, K, 2)`` array padded with NaN."""
    k = max((len(g) for g in groups), default=0)
    out = np.full((len(groups), max(k, 1), 2), np.nan)
    for i, g in enumerate(groups):
        if g:
            out[i, : len(g)] = np.asarray(g, dtype=np.float64)
    return out


def _cross(
    ax: FloatArray | float, ay: FloatArray | float, bx: FloatArray | float, by: FloatArray | float
) -> FloatArray:
    return np.asarray(ax * by - ay * bx, dtype=np.float64)


def count_in_shot_cone(x: FloatArray, y: FloatArray, opp: FloatArray) -> IntArray:
    """Number of ``opp`` points (N, K, 2) inside the triangle shooter -> left post -> right post."""
    px, py = opp[..., 0], opp[..., 1]
    sx, sy = x[:, None], y[:, None]
    # Triangle vertices: shooter S, post A (+y), post B (-y). Point P is inside if it is on the same side of all edges.
    ax, ay = GOAL_X, HALF_GOAL
    bx, by = GOAL_X, -HALF_GOAL
    d1 = _cross(ax - sx, ay - sy, px - sx, py - sy)
    d2 = _cross(bx - ax, by - ay, px - ax, py - ay)
    d3 = _cross(sx - bx, sy - by, px - bx, py - by)
    has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    inside = ~(has_neg & has_pos) & ~np.isnan(px)
    return inside.sum(axis=1).astype(np.int64)


def nearest_distance(x: FloatArray, y: FloatArray, opp: FloatArray, cap: float = NO_DEFENDER_M) -> FloatArray:
    """Distance from (x, y) to the nearest ``opp`` point, ``cap`` when there is none."""
    d = np.hypot(opp[..., 0] - x[:, None], opp[..., 1] - y[:, None])
    d = np.where(np.isnan(d), np.inf, d)
    return np.asarray(np.minimum(d.min(axis=1), cap), dtype=np.float64)


def count_in_lane(
    x: FloatArray, y: FloatArray, end_x: FloatArray, end_y: FloatArray, opp: FloatArray, radius: float = LANE_RADIUS_M
) -> IntArray:
    """Opponents within ``radius`` of the straight segment start -> end (excluding points behind the passer)."""
    dx = (end_x - x)[:, None]
    dy = (end_y - y)[:, None]
    length2 = np.maximum(dx**2 + dy**2, 1e-9)
    rx = opp[..., 0] - x[:, None]
    ry = opp[..., 1] - y[:, None]
    t = (rx * dx + ry * dy) / length2
    tc = np.clip(t, 0.0, 1.0)
    dist = np.hypot(rx - tc * dx, ry - tc * dy)
    hit = (t > 0.0) & (dist <= radius) & ~np.isnan(opp[..., 0])
    return hit.sum(axis=1).astype(np.int64)


def gk_geometry(
    x: FloatArray, y: FloatArray, gk_x: FloatArray, gk_y: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Keeper distance off the goal line, distance to the shooter, and lateral offset from the shooter-goal line."""
    off_line = GOAL_X - gk_x
    to_shooter = np.hypot(gk_x - x, gk_y - y)
    # Signed perpendicular distance of the keeper from the line shooter -> goal centre, as an absolute value.
    dx, dy = GOAL_X - x, -y
    norm = np.maximum(np.hypot(dx, dy), 1e-9)
    lateral = np.abs(((gk_x - x) * dy - (gk_y - y) * dx) / norm)
    return off_line, to_shooter, lateral
