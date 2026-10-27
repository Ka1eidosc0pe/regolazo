import json

import numpy as np
import pytest

from analytics import xt
from analytics.models import load_trained
from analytics.paths import ARTIFACT_DIR


def test_interpolate_hits_cell_centres_and_clamps() -> None:
    grid = np.arange(16 * 12, dtype=float).reshape(16, 12)
    cx = (np.arange(16) + 0.5) / 16 * 105 - 52.5
    cy = (np.arange(12) + 0.5) / 12 * 68 - 34
    assert xt.interpolate(grid, np.array([cx[3]]), np.array([cy[5]]))[0] == pytest.approx(grid[3, 5])
    assert xt.interpolate(grid, np.array([60.0]), np.array([-40.0]))[0] == pytest.approx(grid[15, 0])
    mid = xt.interpolate(grid, np.array([(cx[3] + cx[4]) / 2]), np.array([cy[5]]))[0]
    assert mid == pytest.approx((grid[3, 5] + grid[4, 5]) / 2)


def test_fit_xt_on_a_synthetic_corridor() -> None:
    """Moves always go one cell forward; shots only from the last column score half the time."""
    nx, ny = 4, 1
    xs = (np.arange(nx) + 0.5) / nx * 105 - 52.5
    move_x = np.repeat(xs[:-1], 10)
    move_end_x = move_x + 105 / nx
    zeros = np.zeros_like(move_x)
    shot_x = np.full(10, xs[-1])
    grid, parts = xt.fit_xt(
        move_x,
        zeros,
        move_end_x,
        zeros,
        np.ones_like(move_x, dtype=bool),
        shot_x,
        np.zeros(10),
        np.arange(10) < 5,
        nx=nx,
        ny=ny,
    )
    # Last cell: only shots -> xT = 0.5; earlier cells carry it back unchanged (move prob 1, success 1).
    np.testing.assert_allclose(grid[:, 0], 0.5, atol=1e-6)
    assert parts["shot_prob"][-1, 0] == 1.0


def test_cell_index_covers_pitch() -> None:
    ix, iy = xt.cell_index(np.array([-52.5, 52.5, 0.0]), np.array([-34.0, 34.0, 0.0]))
    assert ix.tolist() == [0, 15, 8] and iy.tolist() == [0, 11, 6]


def test_physical_priors_are_measured_and_plausible() -> None:
    models = load_trained(ARTIFACT_DIR)
    p = models.physical_priors
    assert p.source.startswith("metrica")
    assert 7.5 < p.max_speed_mean < 9.5 and 0.2 < p.max_speed_std < 1.2
    assert 3.0 < p.max_accel_mean < 6.0
    assert 0.2 < p.reaction_time_s < 1.0
    raw = json.loads((ARTIFACT_DIR / "physical_priors.json").read_text())
    assert raw["n_players"] >= 40 and raw["n_reaction_samples"] >= 50
