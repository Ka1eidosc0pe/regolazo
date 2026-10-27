import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from analytics.models import load_trained
from analytics.paths import ARTIFACT_DIR
from analytics.pitch_control import CONVERGE_TOL, defending_gk_index, pitch_control, pitch_grid
from shared.model_api import PitchControlParams

FIXTURE = Path(__file__).parent / "fixtures" / "metrica_pc_frame.json"


@pytest.fixture(scope="module")
def frame() -> dict[str, Any]:
    raw = json.loads(FIXTURE.read_text())
    return {k: np.asarray(v) if isinstance(v, list) else v for k, v in raw.items()}


def test_matches_laurie_on_tracking_reference_on_metrica_frame(frame: dict[str, Any]) -> None:
    res = pitch_control(
        frame["att_pos"], frame["att_vel"], frame["def_pos"], frame["def_vel"], frame["ball"], frame["points"]
    )
    np.testing.assert_allclose(res.att, frame["reference_att"], atol=1e-9)
    np.testing.assert_allclose(res.defn, frame["reference_def"], atol=1e-9)
    # The keeper heuristic finds the same defending keeper the reference was given.
    assert defending_gk_index(frame["def_pos"]) == frame["def_gk"]
    # The frame is genuinely contested: both teams control part of the pitch.
    assert 0.2 < res.normalized_att.mean() < 0.8


def test_probabilities_sum_to_one_per_cell(frame: dict[str, Any]) -> None:
    pts = pitch_grid(32, 21)
    res = pitch_control(frame["att_pos"], frame["att_vel"], frame["def_pos"], frame["def_vel"], frame["ball"], pts)
    total = res.att + res.defn
    assert np.all(total >= 1 - CONVERGE_TOL - 1e-9)
    assert np.all(total <= 1 + 1e-9)
    norm = res.normalized_att
    assert np.all((norm >= 0) & (norm <= 1))


def test_team_swap_symmetry(frame: dict[str, Any]) -> None:
    """With equal control rates and no keeper bonus, swapping the teams swaps the probabilities exactly."""
    pts = pitch_grid(20, 13)
    a = pitch_control(
        frame["att_pos"], frame["att_vel"], frame["def_pos"], frame["def_vel"], frame["ball"], pts, def_gk=None
    )
    b = pitch_control(
        frame["def_pos"], frame["def_vel"], frame["att_pos"], frame["att_vel"], frame["ball"], pts, def_gk=None
    )
    np.testing.assert_allclose(a.att, b.defn, atol=1e-12)
    np.testing.assert_allclose(a.defn, b.att, atol=1e-12)


def test_mirror_symmetry(frame: dict[str, Any]) -> None:
    """Reflecting the scene across the long axis reflects the control surface."""
    flip = np.array([1.0, -1.0])
    pts = pitch_grid(20, 13)
    a = pitch_control(frame["att_pos"], frame["att_vel"], frame["def_pos"], frame["def_vel"], frame["ball"], pts)
    b = pitch_control(
        frame["att_pos"] * flip,
        frame["att_vel"] * flip,
        frame["def_pos"] * flip,
        frame["def_vel"] * flip,
        frame["ball"] * flip,
        pts * flip,
    )
    np.testing.assert_allclose(a.att, b.att, atol=1e-12)


def test_identical_teams_split_control_evenly() -> None:
    rng = np.random.default_rng(0)
    pos = rng.uniform([-40, -30], [40, 30], size=(10, 2))
    vel = rng.normal(size=(10, 2))
    pts = pitch_grid(12, 8)
    res = pitch_control(pos, vel, pos.copy(), vel.copy(), np.array([0.0, 0.0]), pts, def_gk=None)
    np.testing.assert_allclose(res.normalized_att, 0.5, atol=1e-12)


def test_lone_attacker_controls_his_space_and_keeper_bonus_helps_defence() -> None:
    att = np.array([[30.0, 0.0]])
    dfn = np.array([[0.0, 20.0], [50.0, 0.0]])
    zero2, zero1 = np.zeros((2, 2)), np.zeros((1, 2))
    pts = np.array([[30.0, 0.0], [-20.0, 20.0], [47.0, 0.0]])
    p = pitch_control(att, zero1, dfn, zero2, np.array([25.0, 0.0]), pts)
    assert p.normalized_att[0] > 0.99
    assert p.normalized_att[1] < 0.01  # nobody near, but the defender at (0, 20) is closer
    no_gk = pitch_control(att, zero1, dfn, zero2, np.array([25.0, 0.0]), pts, def_gk=None)
    assert p.normalized_att[2] < no_gk.normalized_att[2]


def test_trained_models_expose_consistent_params_and_normalised_control(frame: dict[str, Any]) -> None:
    models = load_trained(ARTIFACT_DIR)
    assert models.pitch_control_params == PitchControlParams()
    out = models.pitch_control(
        frame["att_pos"], frame["att_vel"], frame["def_pos"], frame["def_vel"], frame["ball"], frame["points"]
    )
    ref = frame["reference_att"] / (frame["reference_att"] + frame["reference_def"])
    np.testing.assert_allclose(out, ref, atol=1e-9)


def test_metrica_pass_validation_recorded_in_metrics() -> None:
    metrics = json.loads((ARTIFACT_DIR / "metrics.json").read_text())["pitch_control_validation"]
    for game in metrics.values():
        assert game["n_completed"] > 500 and game["n_lost"] > 100
        assert game["mean_pc_completed"] > game["mean_pc_lost"] + 0.3
        assert game["auc_pc_at_target"] > 0.85
