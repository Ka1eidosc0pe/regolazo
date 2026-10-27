"""Expected Threat (Karun Singh, 2018) on a 16 x 12 grid in the attacking frame.

xT(z) = s(z) g(z) + m(z) sum_z' T(z -> z') xT(z'), where s / m are the shot / move shares of actions in zone z,
g is the goal rate of shots from z, and T counts *successful* moves z -> z' over all moves from z (failed moves are
worth 0). Solved by fixed-point iteration.
"""

import numpy as np
import numpy.typing as npt

from shared.schemas.common import PITCH_LENGTH_M, PITCH_WIDTH_M

FloatArray = npt.NDArray[np.float64]

NX, NY = 16, 12


def cell_index(x: FloatArray, y: FloatArray, nx: int = NX, ny: int = NY) -> tuple[npt.NDArray[np.int64], ...]:
    """Grid cell of attacking-frame points; ix along the length (0 = own goal), iy across (0 = right touchline)."""
    ix = np.clip(((x + PITCH_LENGTH_M / 2) / PITCH_LENGTH_M * nx).astype(np.int64), 0, nx - 1)
    iy = np.clip(((y + PITCH_WIDTH_M / 2) / PITCH_WIDTH_M * ny).astype(np.int64), 0, ny - 1)
    return ix, iy


def fit_xt(
    move_x: FloatArray,
    move_y: FloatArray,
    move_end_x: FloatArray,
    move_end_y: FloatArray,
    move_success: npt.NDArray[np.bool_],
    shot_x: FloatArray,
    shot_y: FloatArray,
    shot_goal: npt.NDArray[np.bool_],
    nx: int = NX,
    ny: int = NY,
    tol: float = 1e-7,
    max_iter: int = 200,
) -> tuple[FloatArray, dict[str, FloatArray]]:
    """Return the (nx, ny) xT grid and the component grids."""
    n_cells = nx * ny
    sx, sy = cell_index(shot_x, shot_y, nx, ny)
    mx, my = cell_index(move_x, move_y, nx, ny)
    ex, ey = cell_index(move_end_x, move_end_y, nx, ny)
    shot_c = np.bincount(sx * ny + sy, minlength=n_cells).astype(np.float64)
    goal_c = np.bincount(sx * ny + sy, weights=shot_goal.astype(np.float64), minlength=n_cells)
    move_c = np.bincount(mx * ny + my, minlength=n_cells).astype(np.float64)
    total = shot_c + move_c
    with np.errstate(invalid="ignore", divide="ignore"):
        s = np.where(total > 0, shot_c / total, 0.0)
        m = np.where(total > 0, move_c / total, 0.0)
        g = np.where(shot_c > 0, goal_c / shot_c, 0.0)
    ok = move_success.astype(bool)
    src = (mx * ny + my)[ok]
    dst = (ex * ny + ey)[ok]
    trans = np.zeros((n_cells, n_cells))
    np.add.at(trans, (src, dst), 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        trans = np.where(move_c[:, None] > 0, trans / move_c[:, None], 0.0)
    xt = np.zeros(n_cells)
    for _ in range(max_iter):
        new = s * g + m * (trans @ xt)
        done = np.max(np.abs(new - xt)) < tol
        xt = new
        if done:
            break
    shape = (nx, ny)
    parts = {
        "shot_prob": s.reshape(shape),
        "move_prob": m.reshape(shape),
        "goal_prob": g.reshape(shape),
        "n_actions": total.reshape(shape),
    }
    return xt.reshape(shape), parts


def interpolate(grid: FloatArray, x: FloatArray, y: FloatArray) -> FloatArray:
    """Bilinear interpolation between cell centres (clamped at the edges)."""
    nx, ny = grid.shape
    fx = np.clip((np.asarray(x, dtype=np.float64) + PITCH_LENGTH_M / 2) / PITCH_LENGTH_M * nx - 0.5, 0, nx - 1)
    fy = np.clip((np.asarray(y, dtype=np.float64) + PITCH_WIDTH_M / 2) / PITCH_WIDTH_M * ny - 0.5, 0, ny - 1)
    x0 = np.minimum(np.floor(fx).astype(np.int64), nx - 2)
    y0 = np.minimum(np.floor(fy).astype(np.int64), ny - 2)
    tx, ty = fx - x0, fy - y0
    v00, v10 = grid[x0, y0], grid[x0 + 1, y0]
    v01, v11 = grid[x0, y0 + 1], grid[x0 + 1, y0 + 1]
    out = (1 - tx) * (1 - ty) * v00 + tx * (1 - ty) * v10 + (1 - tx) * ty * v01 + tx * ty * v11
    return np.asarray(out, dtype=np.float64)
