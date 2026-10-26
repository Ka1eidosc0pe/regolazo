"""Interface between the analytics models (Step 2) and their consumers (simulator, agents, server).

Everything is vectorized over a batch (numpy arrays of equal length) so the simulator can score options and
run 1,000 rollouts cheaply. All positions are in the **attacking frame** of the acting team (metres, centre
origin, attacking +x, +y = left; see shared.geometry).

``load_models()`` returns the trained models when their artifacts exist (written by analytics/) and falls back
to ``StubModels``, a hand-tuned analytic approximation, so the simulator can be built before Step 2 lands.
The trained implementation must satisfy the same ``AnalyticsModels`` protocol.
"""

import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from shared.schemas.common import GOAL_WIDTH_M, PITCH_LENGTH_M, PITCH_WIDTH_M

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

GOAL_X = PITCH_LENGTH_M / 2
HALF_GOAL = GOAL_WIDTH_M / 2
ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "analytics" / "artifacts"

# Integer codes keep the feature arrays numeric. They map to shared.schemas.enums values.
SHOT_BODY_PART = {"foot": 0, "head": 1, "other": 2}
SHOT_TECHNIQUE = {
    "placed": 0,
    "driven": 1,
    "finesse": 2,
    "chip": 3,
    "volley": 4,
    "half_volley": 5,
    "header": 6,
    "diving_header": 7,
    "toe_poke": 8,
    "backheel": 9,
    "overhead_kick": 10,
}
SHOT_SITUATION = {"open_play": 0, "counter_attack": 1, "set_piece": 2, "direct_free_kick": 3, "penalty": 4}
PASS_HEIGHT = {"ground": 0, "low": 1, "high": 2}
PASS_KIND = {"standard": 0, "cross": 1, "cutback": 2, "through_ball": 3, "switch": 4, "back_pass": 5, "chipped": 6}


@dataclass(frozen=True)
class ShotFeatures:
    """Pre-shot state for xG. One row per shot."""

    x: FloatArray
    y: FloatArray
    body_part: IntArray
    """SHOT_BODY_PART code."""
    technique: IntArray
    """SHOT_TECHNIQUE code."""
    situation: IntArray
    """SHOT_SITUATION code."""
    gk_x: FloatArray
    gk_y: FloatArray
    n_defenders_in_cone: IntArray
    """Outfield opponents inside the triangle shooter-post-post."""
    nearest_defender_m: FloatArray
    under_pressure: BoolArray
    first_time: BoolArray


@dataclass(frozen=True)
class PostShotFeatures:
    """Ball state as it reaches the goal plane, for the save model (post-shot xG). One row per on-target shot."""

    end_y: FloatArray
    """Goal-mouth crossing point, metres from the goal centre (attacking frame y)."""
    end_z: FloatArray
    speed_mps: FloatArray
    """Ball speed at the goal plane."""
    distance_m: FloatArray
    """Shot distance (affects keeper reaction time)."""
    gk_x: FloatArray
    gk_y: FloatArray
    gk_dive: FloatArray
    """Keeper diving / reflexes rating 1..99 (50 = average; the trained model is fit at average)."""


@dataclass(frozen=True)
class PassFeatures:
    """Pass attempt for xPass. One row per pass."""

    x: FloatArray
    y: FloatArray
    end_x: FloatArray
    end_y: FloatArray
    height: IntArray
    """PASS_HEIGHT code."""
    kind: IntArray
    """PASS_KIND code."""
    nearest_defender_to_receiver_m: FloatArray
    defenders_in_lane: IntArray
    """Opponents within 2 m of the straight passing lane."""
    under_pressure: BoolArray


@dataclass(frozen=True)
class PitchControlParams:
    """Spearman (2018) parameters, shared by the pitch-control model and the simulator's interception physics."""

    reaction_time_s: float = 0.7
    max_player_speed_mps: float = 5.0
    tti_sigma: float = 0.45
    """Std of time-to-intercept uncertainty (the logistic temperature uses pi / sqrt(3) / sigma)."""
    lambda_att: float = 4.3
    lambda_def: float = 4.3
    """Control rate (1/s); Spearman uses the same for both, with GK lambda = 3x for aerial balls."""
    average_ball_speed_mps: float = 15.0


@dataclass(frozen=True)
class PhysicalPriors:
    """Population distributions for player kinematics, measured from Metrica tracking (Step 2)."""

    max_speed_mean: float = 8.6
    max_speed_std: float = 0.55
    max_accel_mean: float = 3.9
    max_accel_std: float = 0.5
    reaction_time_s: float = 0.25
    source: str = "stub"


@runtime_checkable
class AnalyticsModels(Protocol):
    versions: dict[str, str]
    pitch_control_params: PitchControlParams
    physical_priors: PhysicalPriors

    def xg(self, f: ShotFeatures) -> FloatArray:
        """P(goal) for an average shooter against an average keeper."""
        ...

    def psxg(self, f: PostShotFeatures) -> FloatArray:
        """P(goal | on target, ball state at the goal plane)."""
        ...

    def xpass(self, f: PassFeatures) -> FloatArray:
        """P(pass completed) for an average passer."""
        ...

    def xt(self, x: FloatArray, y: FloatArray) -> FloatArray:
        """Expected threat of having the ball at (x, y)."""
        ...

    def pitch_control(
        self,
        att_pos: FloatArray,
        att_vel: FloatArray,
        def_pos: FloatArray,
        def_vel: FloatArray,
        ball: FloatArray,
        points: FloatArray,
    ) -> FloatArray:
        """P(attacking team controls) at each of ``points`` (M x 2). Player arrays are N x 2, ball is (2,)."""
        ...


def decision_value(p_success: FloatArray, value_success: FloatArray, value_fail: FloatArray) -> FloatArray:
    """Expected value of an option: success weighted by its value, failure by the (usually negative) cost."""
    return p_success * value_success + (1.0 - p_success) * value_fail


def shot_geometry(x: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Distance to goal centre and the angle subtended by the posts, in the attacking frame."""
    dx = GOAL_X - x
    dist = np.hypot(dx, y)
    a1 = np.arctan2(HALF_GOAL - y, dx)
    a2 = np.arctan2(-HALF_GOAL - y, dx)
    return dist, np.abs(a1 - a2)


def _sigmoid(z: FloatArray) -> FloatArray:
    return np.asarray(1.0 / (1.0 + np.exp(-z)), dtype=np.float64)


@dataclass
class StubModels:
    """Analytic stand-ins with plausible magnitudes. Replaced by the trained models once Step 2 lands."""

    versions: dict[str, str] = field(default_factory=lambda: {k: "stub" for k in ("xg", "psxg", "xpass", "xt", "pc")})
    pitch_control_params: PitchControlParams = field(default_factory=PitchControlParams)
    physical_priors: PhysicalPriors = field(default_factory=PhysicalPriors)

    def xg(self, f: ShotFeatures) -> FloatArray:
        dist, angle = shot_geometry(f.x, f.y)
        z = -1.0 + 1.6 * angle - 0.11 * dist
        z = z - 0.9 * (f.body_part == SHOT_BODY_PART["head"]) - 0.35 * f.n_defenders_in_cone
        z = z - 0.3 * f.under_pressure + 0.2 * (f.situation == SHOT_SITUATION["counter_attack"])
        return np.where(f.situation == SHOT_SITUATION["penalty"], 0.76, _sigmoid(z))

    def psxg(self, f: PostShotFeatures) -> FloatArray:
        placement = 1.6 * np.clip(np.abs(f.end_y) / HALF_GOAL, 0, 1) + 0.6 * np.clip(f.end_z / 2.44, 0, 1)
        reach = 0.05 * (f.speed_mps - 20.0) + 0.4 * (f.distance_m < 12) - 0.025 * (f.gk_dive - 50)
        keeper_offset = 0.25 * np.abs(f.gk_y - f.end_y)
        return _sigmoid(-1.2 + placement + reach + keeper_offset)

    def xpass(self, f: PassFeatures) -> FloatArray:
        length = np.hypot(f.end_x - f.x, f.end_y - f.y)
        z = 3.2 - 0.06 * length - 0.9 * f.defenders_in_lane + 0.25 * np.minimum(f.nearest_defender_to_receiver_m, 6)
        z = z - 0.5 * (f.height == PASS_HEIGHT["high"]) - 0.4 * f.under_pressure - 0.02 * np.maximum(f.end_x, 0)
        z = z - 0.6 * (f.kind == PASS_KIND["through_ball"]) - 0.7 * (f.kind == PASS_KIND["cross"])
        return _sigmoid(z)

    def xt(self, x: FloatArray, y: FloatArray) -> FloatArray:
        gx = np.clip((x + GOAL_X) / PITCH_LENGTH_M, 0, 1)
        central = np.exp(-((y / (PITCH_WIDTH_M / 4)) ** 2))
        return np.asarray(0.005 + 0.3 * gx**4 * (0.35 + 0.65 * central), dtype=np.float64)

    def pitch_control(
        self,
        att_pos: FloatArray,
        att_vel: FloatArray,
        def_pos: FloatArray,
        def_vel: FloatArray,
        ball: FloatArray,
        points: FloatArray,
    ) -> FloatArray:
        p = self.pitch_control_params

        def arrival(pos: FloatArray, vel: FloatArray) -> FloatArray:
            ahead = pos + vel * p.reaction_time_s  # (N, 2)
            d = np.linalg.norm(points[:, None, :] - ahead[None, :, :], axis=2)  # (M, N)
            return np.asarray(p.reaction_time_s + d / p.max_player_speed_mps, dtype=np.float64)

        t_att = arrival(att_pos, att_vel).min(axis=1)
        t_def = arrival(def_pos, def_vel).min(axis=1)
        return _sigmoid(np.pi / np.sqrt(3.0) / p.tti_sigma * (t_def - t_att))


def load_models(artifact_dir: Path = ARTIFACT_DIR) -> AnalyticsModels:
    """Trained models if ``analytics.models.load_trained`` and its artifacts exist, else the stub."""
    if (artifact_dir / "manifest.json").exists():
        module = importlib.import_module("analytics.models")
        models: AnalyticsModels = module.load_trained(artifact_dir)
        return models
    return StubModels()
