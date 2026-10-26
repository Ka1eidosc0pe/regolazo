import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from analytics.models import TrainedModels, load_trained
from analytics.paths import ARTIFACT_DIR
from analytics.trees import TreeEnsemble, from_sklearn
from shared.model_api import (
    PASS_HEIGHT,
    PASS_KIND,
    SHOT_BODY_PART,
    SHOT_SITUATION,
    SHOT_TECHNIQUE,
    PassFeatures,
    PostShotFeatures,
    ShotFeatures,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def models() -> TrainedModels:
    return load_trained(ARTIFACT_DIR)


def _golden(name: str, cls: type[Any]) -> tuple[Any, np.ndarray]:
    """Golden inputs (the feature dataclass fields only) and the exported model's outputs at training time."""
    raw = json.loads((FIXTURES / f"golden_{name}.json").read_text())
    assert "StatsBomb" in raw["note"]
    kwargs = {f.name: np.asarray(raw[f.name]) for f in fields(cls)}
    return cls(**kwargs), np.asarray(raw["expected"])


def test_golden_predictions_pin_the_numpy_evaluator(models: TrainedModels) -> None:
    """The golden outputs were written by the same numpy evaluator at training time (where it was also checked
    against scikit-learn on every training row); this pins the shipped artifacts and evaluator to them."""
    f, expected = _golden("xg", ShotFeatures)
    np.testing.assert_allclose(models.xg(f), expected, atol=1e-6)
    f, expected = _golden("psxg", PostShotFeatures)
    np.testing.assert_allclose(models.psxg(f), expected, atol=1e-6)
    f, expected = _golden("xpass", PassFeatures)
    np.testing.assert_allclose(models.xpass(f), expected, atol=1e-6)


def test_tree_export_matches_sklearn(tmp_path: Path) -> None:
    ensemble_mod = pytest.importorskip("sklearn.ensemble")
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2000, 4))
    x[rng.random(2000) < 0.05, 2] = np.nan
    y = ((x[:, 0] + np.nan_to_num(x[:, 2]) * x[:, 1] + rng.normal(size=2000)) > 0).astype(float)
    model = ensemble_mod.HistGradientBoostingClassifier(max_iter=50, random_state=0).fit(x, y)
    ens = from_sklearn(model, ["a", "b", "c", "d"])
    ens.save(tmp_path / "m.npz")
    loaded = TreeEnsemble.load(tmp_path / "m.npz")
    np.testing.assert_allclose(loaded.predict_proba(x), model.predict_proba(x)[:, 1], atol=1e-5)
    assert loaded.feature_names == ("a", "b", "c", "d")


def _shots(n: int, **over: Any) -> ShotFeatures:
    base: dict[str, Any] = {
        "x": np.full(n, 41.5),
        "y": np.zeros(n),
        "body_part": np.full(n, SHOT_BODY_PART["foot"], dtype=np.int64),
        "technique": np.full(n, SHOT_TECHNIQUE["placed"], dtype=np.int64),
        "situation": np.full(n, SHOT_SITUATION["open_play"], dtype=np.int64),
        "gk_x": np.full(n, 51.0),
        "gk_y": np.zeros(n),
        "n_defenders_in_cone": np.ones(n, dtype=np.int64),
        "nearest_defender_m": np.full(n, 2.0),
        "under_pressure": np.zeros(n, dtype=bool),
        "first_time": np.zeros(n, dtype=bool),
    }
    base.update({k: np.asarray(v) for k, v in over.items()})
    return ShotFeatures(**base)


def test_xg_behaves_like_football(models: TrainedModels) -> None:
    xs = np.array([47.0, 41.5, 36.0, 30.0, 20.0])
    p = models.xg(_shots(5, x=xs))
    assert np.all(np.diff(p) < 0), p  # further out -> lower
    assert 0.15 < p[1] < 0.5  # penalty-spot open-play shot with one defender in the cone
    assert p[-1] < 0.05
    head = models.xg(_shots(1, body_part=[SHOT_BODY_PART["head"]], technique=[SHOT_TECHNIQUE["header"]]))
    assert head[0] < p[1]
    crowded = models.xg(_shots(1, n_defenders_in_cone=[4]))
    assert crowded[0] < p[1]
    open_goal = models.xg(_shots(1, n_defenders_in_cone=[0], gk_x=[40.0], gk_y=[6.0], nearest_defender_m=[8.0]))
    assert open_goal[0] > p[1]
    pen = models.xg(_shots(1, situation=[SHOT_SITUATION["penalty"]]))
    assert pen[0] == pytest.approx(models.penalty_xg)
    assert 0.7 < models.penalty_xg < 0.85


def _post(n: int, **over: Any) -> PostShotFeatures:
    base: dict[str, Any] = {
        "end_y": np.zeros(n),
        "end_z": np.full(n, 1.0),
        "speed_mps": np.full(n, 22.0),
        "distance_m": np.full(n, 16.0),
        "gk_x": np.full(n, 51.5),
        "gk_y": np.zeros(n),
        "gk_dive": np.full(n, 50.0),
    }
    base.update({k: np.asarray(v, dtype=np.float64) for k, v in over.items()})
    return PostShotFeatures(**base)


def test_psxg_placement_keeper_and_rating(models: TrainedModels) -> None:
    centre, corner = models.psxg(_post(2, end_y=[0.0, 3.4], end_z=[1.0, 0.3]))
    assert corner > centre + 0.3
    assert centre < 0.3
    good, avg, poor = models.psxg(_post(3, end_y=[2.5] * 3, gk_dive=[90.0, 50.0, 20.0]))
    assert good < avg < poor
    speeds = models.psxg(_post(4, end_y=[2.5] * 4, speed_mps=[10.0, 20.0, 30.0, 38.0]))
    assert np.all(np.diff(speeds) > 0)  # speed prior: harder strikes are harder to save


def _passes(n: int, **over: Any) -> PassFeatures:
    base: dict[str, Any] = {
        "x": np.full(n, -10.0),
        "y": np.zeros(n),
        "end_x": np.full(n, 5.0),
        "end_y": np.full(n, 5.0),
        "height": np.full(n, PASS_HEIGHT["ground"], dtype=np.int64),
        "kind": np.full(n, PASS_KIND["standard"], dtype=np.int64),
        "nearest_defender_to_receiver_m": np.full(n, 4.0),
        "defenders_in_lane": np.zeros(n, dtype=np.int64),
        "under_pressure": np.zeros(n, dtype=bool),
    }
    base.update({k: np.asarray(v) for k, v in over.items()})
    return PassFeatures(**base)


def test_xpass_behaves_like_football(models: TrainedModels) -> None:
    easy = models.xpass(_passes(1))[0]
    assert easy > 0.85
    lanes = models.xpass(_passes(3, defenders_in_lane=[0, 1, 3]))
    assert np.all(np.diff(lanes) < 0)
    tight = models.xpass(_passes(1, nearest_defender_to_receiver_m=[0.7]))[0]
    assert tight < easy
    cross = models.xpass(
        _passes(1, x=[38.0], y=[28.0], end_x=[47.0], end_y=[1.0], height=[PASS_HEIGHT["high"]], kind=[1])
    )[0]
    assert cross < 0.5
    long_ball = models.xpass(_passes(1, x=[-40.0], end_x=[30.0], end_y=[20.0], height=[PASS_HEIGHT["high"]]))[0]
    assert long_ball < easy


def test_xt_grid_rises_towards_goal(models: TrainedModels) -> None:
    grid = models.xt_grid
    assert grid.shape == (16, 12)
    col = grid.mean(axis=1)
    assert np.sum(np.diff(col) > 0) >= 14  # "monotonic-ish": at most one dip along the length
    assert col[-1] > 5 * col[0]
    assert grid.max() == grid[-1].max()  # most dangerous zone is next to goal
    central, wide = models.xt(np.array([45.0, 45.0]), np.array([0.0, 30.0]))
    assert central > 3 * wide
    assert np.all((grid >= 0) & (grid < 0.5))


def test_outputs_are_probabilities_and_vectorized(models: TrainedModels) -> None:
    rng = np.random.default_rng(3)
    n = 500
    shots = _shots(
        n,
        x=rng.uniform(10, 52, n),
        y=rng.uniform(-30, 30, n),
        body_part=rng.integers(0, 3, n),
        technique=rng.integers(0, 11, n),
        situation=rng.integers(0, 5, n),
        n_defenders_in_cone=rng.integers(0, 5, n),
        nearest_defender_m=rng.uniform(0, 20, n),
    )
    p = models.xg(shots)
    assert p.shape == (n,) and np.all((p > 0) & (p < 1))
    post = _post(n, end_y=rng.uniform(-3.66, 3.66, n), end_z=rng.uniform(0, 2.44, n), speed_mps=rng.uniform(5, 40, n))
    q = models.psxg(post)
    assert q.shape == (n,) and np.all((q > 0) & (q < 1))
    passes = _passes(n, end_x=rng.uniform(-50, 50, n), end_y=rng.uniform(-33, 33, n), kind=rng.integers(0, 7, n))
    r = models.xpass(passes)
    assert r.shape == (n,) and np.all((r > 0) & (r < 1))


# ----------------------------------------------------------------------------------------- edge cases (review)


def test_psxg_height_sweep_is_non_decreasing_towards_the_corners(models: TrainedModels) -> None:
    ys = [0.0, 1.0, 2.0, 3.0, 3.4]
    for z in (0.2, 0.8, 1.4, 2.0, 2.3, 2.44):
        p = models.psxg(_post(5, end_y=ys, end_z=[z] * 5))
        assert np.all(np.diff(p) >= -1e-9), (z, p)
        assert p[-1] > p[0] + 0.3, (z, p)


def test_psxg_top_corner_is_not_a_certain_save(models: TrainedModels) -> None:
    top_corner = models.psxg(_post(1, end_y=[3.2], end_z=[2.2]))[0]
    central = models.psxg(_post(1, end_y=[0.0], end_z=[1.2]))[0]
    assert top_corner > central + 0.3
    assert top_corner > 0.4
    # No cliff at the bar: just under it is about as hard to save as mid-height.
    mid, high, bar = models.psxg(_post(3, end_y=[3.0] * 3, end_z=[1.2, 2.3, 2.44]))
    assert high > 0.8 * mid and bar > 0.8 * high


def test_xg_is_bounded_for_unmarked_shooters(models: TrainedModels) -> None:
    nd = [10.0, 15.0, 20.0, 30.0]
    p = models.xg(_shots(4, nearest_defender_m=nd, n_defenders_in_cone=[0] * 4))
    np.testing.assert_allclose(p, p[0])  # clipped at ~p99 of training (10 m): no extrapolation
    assert np.all(p < models.penalty_xg)
    # Keeper distances are clipped as well: two keepers far behind a shooter 32.5 m out look the same.
    far_keeper = models.xg(_shots(2, x=[20.0, 20.0], gk_x=[-15.0, -30.0], gk_y=[0.0, 0.0]))
    assert far_keeper[0] == pytest.approx(far_keeper[1])


def test_xpass_has_no_touchline_cliff(models: TrainedModels) -> None:
    p = models.xpass(_passes(4, x=[10.0] * 4, y=[25.0] * 4, end_x=[22.0] * 4, end_y=[33.0, 33.5, 34.0, 35.0]))
    assert np.max(p) - np.min(p) < 0.02


def test_xpass_falls_with_lane_defenders(models: TrainedModels) -> None:
    p = models.xpass(_passes(11, defenders_in_lane=list(range(11))))
    assert np.all(np.diff(p[:4]) < 0), p  # strictly decreasing 0 -> 3
    assert np.all(np.diff(p) <= 1e-12), p  # non-increasing after (capped at 3)
