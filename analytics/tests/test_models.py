import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from analytics import design
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


def _golden(name: str) -> dict[str, Any]:
    raw = json.loads((FIXTURES / f"golden_{name}.json").read_text())
    return {k: np.asarray(v, dtype=np.float64) for k, v in raw.items()}


def test_golden_predictions_match_training_time_sklearn_outputs(models: TrainedModels) -> None:
    g = _golden("xg")
    np.testing.assert_allclose(models.xg(design.shot_features_from_table(g)), g["expected"], atol=1e-5)
    g = _golden("psxg")
    np.testing.assert_allclose(models.psxg(design.post_shot_features_from_table(g)), g["expected"], atol=1e-5)
    g = _golden("xpass")
    np.testing.assert_allclose(models.xpass(design.pass_features_from_table(g)), g["expected"], atol=1e-5)


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
