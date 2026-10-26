"""The simulator and agents get the trained models through shared.model_api.load_models()."""

import json
from pathlib import Path

import numpy as np
import pytest

from analytics.models import TrainedModels
from analytics.paths import ARTIFACT_DIR, REPO_ROOT
from analytics.scene import Scene, pass_features, pitch_control_grid, scene_from_state, shot_features
from analytics.xt import cell_index
from shared.model_api import SHOT_BODY_PART, SHOT_TECHNIQUE, AnalyticsModels, load_models
from shared.schemas.models import ModelOutputs
from shared.schemas.state import SimState

FIXTURES = REPO_ROOT / "shared" / "fixtures"


@pytest.fixture(scope="module")
def models() -> AnalyticsModels:
    return load_models()


@pytest.fixture(scope="module")
def scene() -> Scene:
    state = SimState.model_validate_json((FIXTURES / "SimState.json").read_text())
    return scene_from_state(state)


def test_load_models_returns_trained_models(models: AnalyticsModels) -> None:
    assert isinstance(models, AnalyticsModels)
    assert isinstance(models, TrainedModels)
    assert set(models.versions) == {"xg", "psxg", "xpass", "xt", "pc"}
    assert all(v != "stub" for v in models.versions.values())
    assert models.physical_priors.source != "stub"


def test_manifest_files_exist_and_artifacts_are_small() -> None:
    manifest = json.loads((ARTIFACT_DIR / "manifest.json").read_text())
    for name in manifest["files"].values():
        assert (ARTIFACT_DIR / name).exists(), name
    total = sum(p.stat().st_size for p in Path(ARTIFACT_DIR).iterdir())
    assert total < 20e6


def test_fixture_scene_outputs_are_sane(models: AnalyticsModels, scene: Scene) -> None:
    carrier = scene.player_xy("H10")
    assert scene.team == "home" and len(scene.att_ids) == 11 and len(scene.def_ids) == 11
    assert scene.def_ids[scene.def_gk or 0] == "A01"

    # Long-range shot by the carrier (34 m out, defenders in the way) is a poor chance; a header from the six-yard
    # box is a much better one, a tap-in better still.
    long = models.xg(shot_features(scene, carrier))[0]
    header = models.xg(shot_features(scene, np.array([47.0, 1.0]), SHOT_BODY_PART["head"], SHOT_TECHNIQUE["header"]))[0]
    tap_in = models.xg(shot_features(scene, np.array([49.5, 0.5])))[0]
    assert long < 0.06
    assert long < header < tap_in

    # Passing options to every teammate: all probabilities; among forward options the shortest is easier than the
    # longest (backward passes to the free keeper are near-certain, as in the data).
    mates = [i for i, pid in enumerate(scene.att_ids) if pid != "H10"]
    targets = scene.att_pos[mates]
    p = models.xpass(pass_features(scene, carrier, targets))
    assert p.shape == (10,) and np.all((p > 0) & (p < 1))
    dist = np.linalg.norm(targets - carrier, axis=1)
    fwd = np.flatnonzero(targets[:, 0] > carrier[0])
    assert p[fwd[np.argmin(dist[fwd])]] > p[fwd[np.argmax(dist[fwd])]]

    # xT rises from the carrier's own half to the box.
    xt_vals = models.xt(np.array([-30.0, carrier[0], 45.0]), np.array([0.0, carrier[1], 0.0]))
    assert np.all(np.diff(xt_vals) > 0)


def test_fixture_pitch_control_grid(models: AnalyticsModels, scene: Scene) -> None:
    grid = ModelOutputs.model_validate_json((FIXTURES / "ModelOutputs.json").read_text()).pitch_control
    assert grid is not None
    xs = grid.x0 + grid.dx * np.arange(grid.nx)
    ys = grid.y0 + grid.dy * np.arange(grid.ny)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    pc = models.pitch_control(scene.att_pos, scene.att_vel, scene.def_pos, scene.def_vel, scene.ball[:2], pts)
    assert pc.shape == (grid.nx * grid.ny,)
    assert np.all((pc >= 0) & (pc <= 1))

    def near(x: float, y: float) -> float:
        return float(pc[np.argmin(np.hypot(pts[:, 0] - x, pts[:, 1] - y))])

    assert near(-45.0, 0.0) > 0.95  # around the home keeper, deep in their own half
    assert near(50.0, 0.0) < 0.05  # the away keeper's six-yard box
    assert 0.2 < pc.mean() < 0.8


def test_xt_cell_lookup_agrees_with_grid(models: AnalyticsModels) -> None:
    assert isinstance(models, TrainedModels)
    grid = models.xt_grid
    cx = (np.arange(16) + 0.5) / 16 * 105 - 52.5
    cy = (np.arange(12) + 0.5) / 12 * 68 - 34
    ix, iy = cell_index(np.array([cx[10]]), np.array([cy[3]]))
    assert models.xt(np.array([cx[10]]), np.array([cy[3]]))[0] == pytest.approx(grid[ix[0], iy[0]])


def test_realism_targets_cover_the_simulator_test() -> None:
    targets = json.loads((ARTIFACT_DIR / "realism_targets.json").read_text())
    expected = {
        "goals_per_match": (2.0, 3.5),
        "shots_per_team": (9.0, 17.0),
        "pass_completion_pct": (70.0, 90.0),
        "passes_per_team": (350.0, 650.0),
        "header_share_of_shots": (0.1, 0.25),
        "mean_shot_distance_m": (14.0, 20.0),
        "distance_per_outfield_player_m": (9000.0, 12000.0),
    }
    for key, (lo, hi) in expected.items():
        t = targets[key]
        assert lo < t["mean"] < hi, key
        assert t["std"] > 0 and t["band"][0] < t["mean"] < t["band"][1], key
    assert "std_band" in targets["possession_share_by_passes_pct"]


def test_pitch_control_grid_contract_in_world_frame(models: AnalyticsModels) -> None:
    """The away team (attacking -x in this fixture) gets a grid in world coordinates that mirrors the home grid."""
    state = SimState.model_validate_json((FIXTURES / "SimState.json").read_text())
    home = pitch_control_grid(models, scene_from_state(state, "home"))
    away = pitch_control_grid(models, scene_from_state(state, "away"))
    assert home.team == "home" and away.team == "away" and len(home.values) == home.nx * home.ny
    h, a = np.array(home.values), np.array(away.values)
    # Each view gives the control bonus to its own defending keeper, so they are near (not exact) complements.
    assert np.mean(np.abs(h + a - 1.0)) < 0.1
    # values[0] is the cell nearest (-52.5, -34): home's defensive corner.
    assert home.values[0] > 0.5 > away.values[0]
