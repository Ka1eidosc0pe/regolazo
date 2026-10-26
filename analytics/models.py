"""Trained analytics models behind ``shared.model_api.AnalyticsModels``.

``shared.model_api.load_models()`` calls ``load_trained`` once ``analytics/artifacts/manifest.json`` exists. Inference
needs only numpy: tree ensembles are evaluated by ``analytics.trees``, xT is a JSON grid and pitch control is the
analytic Spearman model.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from analytics import design
from analytics import pitch_control as pc
from analytics.trees import TreeEnsemble
from analytics.xt import interpolate
from shared.model_api import (
    SHOT_SITUATION,
    PassFeatures,
    PhysicalPriors,
    PitchControlParams,
    PostShotFeatures,
    ShotFeatures,
)

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class LogisticModel:
    """Standardised logistic regression exported from scikit-learn."""

    mean: FloatArray
    scale: FloatArray
    coef: FloatArray
    intercept: float

    def raw(self, x: FloatArray) -> FloatArray:
        return np.asarray(self.intercept + ((x - self.mean) / self.scale) @ self.coef, dtype=np.float64)

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "LogisticModel":
        return cls(
            mean=np.asarray(d["mean"], dtype=np.float64),
            scale=np.asarray(d["scale"], dtype=np.float64),
            coef=np.asarray(d["coef"], dtype=np.float64),
            intercept=float(d["intercept"]),
        )


@dataclass
class TrainedModels:
    """Implements ``shared.model_api.AnalyticsModels`` with models trained on StatsBomb / Metrica data.

    xG is the mean of the logits of a gradient-boosted model and a logistic model (penalties use the empirical
    conversion rate); post-shot xG and xPass are gradient-boosted; xT is bilinear on a 16 x 12 grid; pitch control is
    Spearman (2018).
    """

    versions: dict[str, str]
    pitch_control_params: PitchControlParams
    physical_priors: PhysicalPriors
    xg_model: TreeEnsemble
    xg_logistic: LogisticModel
    psxg_model: TreeEnsemble
    xpass_model: TreeEnsemble
    xt_grid: FloatArray
    penalty_xg: float
    psxg_gk_dive_logit_per_point: float
    psxg_speed_logit_per_mps: float
    psxg_reference_speed_mps: float
    manifest: dict[str, Any] = field(default_factory=dict)

    def xg(self, f: ShotFeatures) -> FloatArray:
        x = design.xg_matrix(f)
        z = (self.xg_model.raw(x) + self.xg_logistic.raw(design.xg_logistic_matrix(x))) / 2
        p = 1.0 / (1.0 + np.exp(-z))
        return np.where(np.asarray(f.situation) == SHOT_SITUATION["penalty"], self.penalty_xg, p)

    def psxg(self, f: PostShotFeatures) -> FloatArray:
        """Learned placement / keeper model plus two documented priors: ball speed and keeper rating."""
        speed = np.clip(np.asarray(f.speed_mps, dtype=np.float64), *design.SPEED_RANGE_MPS)
        z = self.psxg_logit(f)
        z = z + self.psxg_speed_logit_per_mps * (speed - self.psxg_reference_speed_mps)
        z = z - self.psxg_gk_dive_logit_per_point * (np.asarray(f.gk_dive) - 50.0)
        return np.asarray(1.0 / (1.0 + np.exp(-z)), dtype=np.float64)

    def psxg_logit(self, f: PostShotFeatures) -> FloatArray:
        """Learned logit of post-shot xG (average keeper, typical shot speed)."""
        return self.psxg_model.raw(design.psxg_matrix(f))

    def xpass(self, f: PassFeatures) -> FloatArray:
        return self.xpass_model.predict_proba(design.xpass_matrix(f))

    def xt(self, x: FloatArray, y: FloatArray) -> FloatArray:
        return interpolate(self.xt_grid, x, y)

    def pitch_control(
        self,
        att_pos: FloatArray,
        att_vel: FloatArray,
        def_pos: FloatArray,
        def_vel: FloatArray,
        ball: FloatArray,
        points: FloatArray,
    ) -> FloatArray:
        """P(attacking team controls) per point, normalised so that P(att) + P(def) = 1."""
        res = pc.pitch_control(att_pos, att_vel, def_pos, def_vel, ball, points, self.pitch_control_params)
        return res.normalized_att


def load_trained(artifact_dir: Path) -> TrainedModels:
    manifest: dict[str, Any] = json.loads((artifact_dir / "manifest.json").read_text())
    files = manifest["files"]
    priors_raw = json.loads((artifact_dir / files["physical_priors"]).read_text())
    priors = PhysicalPriors(
        max_speed_mean=float(priors_raw["max_speed_mean"]),
        max_speed_std=float(priors_raw["max_speed_std"]),
        max_accel_mean=float(priors_raw["max_accel_mean"]),
        max_accel_std=float(priors_raw["max_accel_std"]),
        reaction_time_s=float(priors_raw["reaction_time_s"]),
        source=str(priors_raw["source"]),
    )
    pcp = PitchControlParams(**manifest["pitch_control_params"])
    xt_raw = json.loads((artifact_dir / files["xt"]).read_text())
    return TrainedModels(
        versions=dict(manifest["versions"]),
        pitch_control_params=pcp,
        physical_priors=priors,
        xg_model=TreeEnsemble.load(artifact_dir / files["xg"]),
        xg_logistic=LogisticModel.from_json(json.loads((artifact_dir / files["xg_logistic"]).read_text())),
        psxg_model=TreeEnsemble.load(artifact_dir / files["psxg"]),
        xpass_model=TreeEnsemble.load(artifact_dir / files["xpass"]),
        xt_grid=np.asarray(xt_raw["grid"], dtype=np.float64),
        penalty_xg=float(manifest["penalty_xg"]),
        psxg_gk_dive_logit_per_point=float(manifest["psxg_gk_dive_logit_per_point"]),
        psxg_speed_logit_per_mps=float(manifest["psxg_speed_logit_per_mps"]),
        psxg_reference_speed_mps=float(manifest["psxg_reference_speed_mps"]),
        manifest=manifest,
    )
