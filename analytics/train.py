"""Train, evaluate and export every analytics model.

Run with: uv run python -m analytics.train   (after scripts/download_*.py and python -m analytics.build_dataset)

Writes analytics/artifacts/ (models, xT grid, priors, realism targets, metrics, manifest), calibration plots under
docs/img/, and golden prediction vectors under analytics/tests/fixtures/ for the numpy evaluator tests.
"""

import json
import time
from dataclasses import dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from sklearn.calibration import calibration_curve  # type: ignore[import-untyped]
from sklearn.ensemble import HistGradientBoostingClassifier  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from analytics import design, physical, xt
from analytics.build_dataset import cache_dir, load_table
from analytics.data import metrica
from analytics.features import gk_geometry
from analytics.models import LogisticModel, psxg_with_priors
from analytics.paths import ARTIFACT_DIR, DOCS_IMG_DIR, REPO_ROOT
from analytics.trees import TreeEnsemble, from_sklearn
from shared.model_api import (
    PASS_KIND,
    SHOT_BODY_PART,
    SHOT_SITUATION,
    PitchControlParams,
    PostShotFeatures,
    ShotFeatures,
)

FloatArray = npt.NDArray[np.float64]
Table = dict[str, npt.NDArray[Any]]

SEED = 7
TEST_FRACTION = 0.2
GOLDEN_DIR = REPO_ROOT / "analytics" / "tests" / "fixtures"
VERSION = "1.0.0"
PSXG_GK_DIVE_LOGIT_PER_POINT = 0.01
"""Prior, not fit: elite keepers (rating ~90) save ~8 points more of on-target shots than average (PSxG - GA spread)."""
PSXG_SPEED_LOGIT_PER_MPS = 0.06
PSXG_REFERENCE_SPEED_MPS = 22.0
"""Prior, not fit: +0.6 logit per extra 10 m/s at the goal plane, centred on a typical 22 m/s (80 km/h) strike."""

# Plot styling (validated categorical slots 1-3 of the dataviz reference palette, light surface).
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]


# ----------------------------------------------------------------------------------------------- helpers


def match_split(match_ids: FloatArray, seed: int = SEED) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    """Hold out TEST_FRACTION of matches (grouped, so no match leaks into both sides)."""
    ids = np.unique(match_ids)
    rng = np.random.default_rng(seed)
    test_ids = rng.choice(ids, size=round(len(ids) * TEST_FRACTION), replace=False)
    test = np.isin(match_ids, test_ids)
    return ~test, test


def scores(y: FloatArray, p: FloatArray) -> dict[str, float]:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "n": len(y),
        "base_rate": float(y.mean()),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "auc": float(roc_auc_score(y, p)) if 0 < y.mean() < 1 else float("nan"),
        "mean_pred": float(p.mean()),
    }


def calibration(y: FloatArray, p: FloatArray, bins: int = 10) -> list[list[float]]:
    frac, mean = calibration_curve(y, p, n_bins=bins, strategy="quantile")
    return [[float(a), float(b)] for a, b in zip(mean, frac, strict=True)]


VAL_FRACTION = 0.2
"""Share of the *training* matches carved out (grouped by match) to choose GBM settings, tree counts and blend
weights. The held-out test matches are never used for any choice."""


def inner_split(
    match_ids: FloatArray, train: npt.NDArray[np.bool_], seed: int = SEED
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    """Split the training rows into fit / validation by match."""
    ids = np.unique(match_ids[train])
    rng = np.random.default_rng(seed + 1000)
    val_ids = rng.choice(ids, size=round(len(ids) * VAL_FRACTION), replace=False)
    val = train & np.isin(match_ids, val_ids)
    return train & ~val, val


@dataclass(frozen=True)
class GbmChoice:
    params: dict[str, Any]
    n_iter: int
    val_log_loss: float
    grid: list[dict[str, Any]]


def _hgb(monotonic: list[int], n_iter: int, **params: Any) -> Any:
    return HistGradientBoostingClassifier(
        max_iter=n_iter, early_stopping=False, monotonic_cst=monotonic, random_state=SEED, **params
    )


def select_gbm(
    x: FloatArray,
    y: FloatArray,
    fit: npt.NDArray[np.bool_],
    val: npt.NDArray[np.bool_],
    monotonic: list[int],
    grid: list[dict[str, Any]],
    max_iter: int,
) -> tuple[GbmChoice, Any]:
    """Grid over GBM settings; tree count = the iteration with the lowest validation log-loss (early stopping on the
    grouped validation matches). Returns the choice and the winning model fitted on ``fit`` (all ``max_iter`` trees)."""
    best: tuple[float, int, dict[str, Any], Any] | None = None
    table: list[dict[str, Any]] = []
    for params in grid:
        model = _hgb(monotonic, max_iter, **params).fit(x[fit], y[fit])
        losses = [log_loss(y[val], np.clip(p[:, 1], 1e-9, 1 - 1e-9)) for p in model.staged_predict_proba(x[val])]
        i = int(np.argmin(losses))
        table.append({**params, "best_iter": i + 1, "val_log_loss": float(losses[i])})
        if best is None or losses[i] < best[0]:
            best = (float(losses[i]), i + 1, params, model)
    assert best is not None
    return GbmChoice(best[2], best[1], best[0], table), best[3]


def staged_at(model: Any, x: FloatArray, n_iter: int) -> FloatArray:
    """Predictions of the first ``n_iter`` trees of a model fitted with more."""
    for i, p in enumerate(model.staged_predict_proba(x), start=1):
        if i == n_iter:
            return np.asarray(p[:, 1], dtype=np.float64)
    raise ValueError(n_iter)


def fit_gbm(x: FloatArray, y: FloatArray, monotonic: list[int], choice: GbmChoice) -> Any:
    return _hgb(monotonic, choice.n_iter, **choice.params).fit(x, y)


def fit_logit(x: FloatArray, y: FloatArray) -> tuple[Any, Any]:
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(C=1.0, max_iter=2000).fit(scaler.transform(x), y)
    return scaler, model


def logit_json(scaler: Any, model: Any, names: list[str]) -> dict[str, Any]:
    return {
        "features": names,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coef": model.coef_.ravel().tolist(),
        "intercept": float(model.intercept_[0]),
    }


def export_checked(model: Any, x: FloatArray, names: list[str]) -> TreeEnsemble:
    """Flatten a fitted model and check the numpy evaluator against scikit-learn on every training row."""
    ens = from_sklearn(model, names)
    diff = np.max(np.abs(ens.predict_proba(x) - model.predict_proba(x)[:, 1]))
    assert diff < 1e-5, f"numpy evaluator disagrees with sklearn by {diff}"
    return ens


def save_golden(name: str, features: Any, idx: npt.NDArray[np.int64], expected: FloatArray) -> None:
    """Store only the model inputs (the shared feature dataclass fields) and the numpy evaluator's outputs."""
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "note": "Model inputs derived from StatsBomb Open Data (https://github.com/statsbomb/open-data); "
        "expected = outputs of the exported model at training time.",
    }
    for f in fields(features):
        payload[f.name] = np.asarray(getattr(features, f.name))[idx].tolist()
    payload["expected"] = expected.tolist()
    (GOLDEN_DIR / f"golden_{name}.json").write_text(json.dumps(payload))


def _style(ax: Any, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=12, loc="left")
    ax.set_xlabel(xlabel, color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    for s in ax.spines.values():
        s.set_visible(False)


def plot_calibration(path: Path, title: str, curves: dict[str, list[list[float]]], max_p: float = 1.0) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 5.0), dpi=110)
    fig.patch.set_facecolor(SURFACE)
    ax.plot([0, max_p], [0, max_p], color=INK_2, linewidth=1, linestyle="--", label="perfect calibration")
    for i, (label, pts) in enumerate(curves.items()):
        arr = np.array(pts)
        ax.plot(arr[:, 0], arr[:, 1], color=SERIES[i], linewidth=2, marker="o", markersize=5, label=label)
    ax.set_xlim(0, max_p)
    ax.set_ylim(0, max_p)
    _style(ax, title, "predicted probability (decile mean)", "observed frequency")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------- models


XG_GRID: list[dict[str, Any]] = [
    {"learning_rate": 0.05, "max_leaf_nodes": 15, "min_samples_leaf": 40, "l2_regularization": 1.0},
    {"learning_rate": 0.02, "max_leaf_nodes": 15, "min_samples_leaf": 60, "l2_regularization": 3.0},
    {"learning_rate": 0.03, "max_leaf_nodes": 8, "min_samples_leaf": 80, "l2_regularization": 1.0},
]
PSXG_GRID: list[dict[str, Any]] = [
    {"learning_rate": 0.05, "max_leaf_nodes": 15, "min_samples_leaf": 30, "l2_regularization": 1.0},
    {"learning_rate": 0.03, "max_leaf_nodes": 8, "min_samples_leaf": 50, "l2_regularization": 3.0},
]
XPASS_GRID: list[dict[str, Any]] = [
    {"learning_rate": 0.1, "max_leaf_nodes": 31, "min_samples_leaf": 100, "l2_regularization": 1.0},
    {"learning_rate": 0.1, "max_leaf_nodes": 63, "min_samples_leaf": 200, "l2_regularization": 1.0},
]
BLEND_WEIGHTS = np.round(np.linspace(0.0, 1.0, 11), 2)


def _blend(p_gbm: FloatArray, p_lr: FloatArray, w: float) -> FloatArray:
    """xG = sigmoid(w * logit(GBM) + (1 - w) * logit(logistic))."""
    z = w * design.logit(p_gbm) + (1 - w) * design.logit(p_lr)
    return np.asarray(1.0 / (1.0 + np.exp(-z)), dtype=np.float64)


@dataclass(frozen=True)
class XgFit:
    choice: GbmChoice
    weight: float
    val_log_loss_by_weight: dict[str, float]
    gbm: Any
    scaler: Any
    lr: Any


def fit_xg(x: FloatArray, xl: FloatArray, y: FloatArray, match_ids: FloatArray, train: npt.NDArray[np.bool_]) -> XgFit:
    """All choices (GBM settings, tree count, blend weight) on grouped validation matches inside ``train``; then the
    two models are refit on all of ``train``."""
    mono = design.monotonic_vector(design.XG_FEATURES, design.XG_MONOTONIC)
    fit, val = inner_split(match_ids, train)
    choice, probe = select_gbm(x, y, fit, val, mono, XG_GRID, max_iter=1500)
    pg = staged_at(probe, x[val], choice.n_iter)
    sc, lr = fit_logit(xl[fit], y[fit])
    pl = lr.predict_proba(sc.transform(xl[val]))[:, 1]
    losses = {f"{w:.1f}": float(log_loss(y[val], _blend(pg, pl, float(w)))) for w in BLEND_WEIGHTS}
    weight = float(min(losses, key=losses.__getitem__))
    gbm = fit_gbm(x[train], y[train], mono, choice)
    sc_t, lr_t = fit_logit(xl[train], y[train])
    return XgFit(choice, weight, losses, gbm, sc_t, lr_t)


def _xg_predict(f: XgFit, x: FloatArray, xl: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    pg = f.gbm.predict_proba(x)[:, 1]
    pl = f.lr.predict_proba(f.scaler.transform(xl))[:, 1]
    return _blend(pg, pl, f.weight), pg, pl


def train_xg(shots: Table) -> tuple[dict[str, Any], dict[str, Any]]:
    """Shipped xG: logit blend of a gradient-boosted and a logistic model, weight chosen on validation matches."""
    pen = shots["situation"] == SHOT_SITUATION["penalty"]
    penalty_rate = float(shots["goal"][pen].mean())
    t = {k: v[~pen] for k, v in shots.items()}
    feats = design.shot_features_from_table(t)
    x = design.xg_matrix(feats)
    xl = design.xg_logistic_matrix(x)
    y = t["goal"]
    tr, te = match_split(t["match_id"])
    xf = fit_xg(x, xl, y, t["match_id"], tr)
    p_blend, p_gbm, p_lr = _xg_predict(xf, x[te], xl[te])
    sb_ref = t["sb_xg"][te]
    head = t["body_part"][te] == SHOT_BODY_PART["head"]
    yt = y[te]
    m: dict[str, Any] = {
        "n_train": int(tr.sum()),
        "n_test": int(te.sum()),
        "selection": {
            "gbm_params": xf.choice.params,
            "n_trees": xf.choice.n_iter,
            "grid": xf.choice.grid,
            "blend_weight_gbm": xf.weight,
            "val_log_loss_by_weight": xf.val_log_loss_by_weight,
        },
        "feature_p99_non_penalty": _xg_p99(feats),
        "repeated_split": _repeated_split(x, xl, y, t),
        "penalty_conversion": penalty_rate,
        "n_penalties": int(pen.sum()),
        "shipped_blend": scores(yt, p_blend),
        "gbm": scores(yt, p_gbm),
        "logistic": scores(yt, p_lr),
        "statsbomb_xg_reference": scores(yt, sb_ref),
        "blend_headers": scores(yt[head], p_blend[head]),
        "blend_foot": scores(yt[~head], p_blend[~head]),
        "statsbomb_headers": scores(yt[head], sb_ref[head]),
        "calibration": {
            "shipped_blend": calibration(yt, p_blend),
            "gbm": calibration(yt, p_gbm),
            "logistic": calibration(yt, p_lr),
            "statsbomb_xg_reference": calibration(yt, sb_ref),
        },
    }
    plot_calibration(
        DOCS_IMG_DIR / "xg_calibration.png",
        "xG calibration (held-out matches)",
        {
            "shipped xG (GBM + logistic blend)": m["calibration"]["shipped_blend"],
            "logistic baseline": m["calibration"]["logistic"],
            "StatsBomb xG (reference)": m["calibration"]["statsbomb_xg_reference"],
        },
        max_p=0.7,
    )
    # Final artifact: same settings, tree count and weight, refit on every non-penalty shot.
    mono = design.monotonic_vector(design.XG_FEATURES, design.XG_MONOTONIC)
    final = fit_gbm(x, y, mono, xf.choice)
    ens = export_checked(final, x, design.XG_FEATURES)
    s_final, lr_final = fit_logit(xl, y)
    logit = logit_json(s_final, lr_final, design.XG_LOGISTIC_FEATURES)
    idx = np.random.default_rng(1).choice(len(y), 64, replace=False)
    lr_model = LogisticModel.from_json(logit)
    z = xf.weight * ens.raw(x[idx]) + (1 - xf.weight) * lr_model.raw(xl[idx])
    save_golden("xg", feats, idx, 1.0 / (1.0 + np.exp(-z)))
    return {
        "ensemble": ens,
        "logit": logit,
        "penalty": penalty_rate,
        "weight": xf.weight,
        "n_trees_final": int(final.n_iter_),
    }, m


def _xg_p99(f: ShotFeatures) -> dict[str, float]:
    off, to_shooter, lateral = gk_geometry(f.x, f.y, f.gk_x, f.gk_y)
    return {
        "nearest_defender_m": float(np.percentile(f.nearest_defender_m, 99)),
        "gk_off_line": float(np.percentile(off, 99)),
        "gk_to_shooter": float(np.percentile(to_shooter, 99)),
        "gk_lateral": float(np.percentile(lateral, 99)),
        "n_nearest_defender_ge_15m": int((f.nearest_defender_m >= 15).sum()),
    }


def _repeated_split(x: FloatArray, xl: FloatArray, y: FloatArray, t: Table) -> dict[str, Any]:
    """Held-out AUC / log-loss over 5 more match splits, each with its own inner selection."""
    out: dict[str, list[tuple[float, float]]] = {"shipped_blend": [], "gbm": [], "logistic": [], "statsbomb": []}
    for seed in (11, 23, 31, 47, 59):
        tr, te = match_split(t["match_id"], seed=seed)
        xf = fit_xg(x, xl, y, t["match_id"], tr)
        pb, pg, pl = _xg_predict(xf, x[te], xl[te])
        for k, p in (("shipped_blend", pb), ("gbm", pg), ("logistic", pl), ("statsbomb", t["sb_xg"][te])):
            sc = scores(y[te], p)
            out[k].append((sc["auc"], sc["log_loss"]))
    return {
        k: {
            "auc_mean": float(np.mean([a for a, _ in v])),
            "auc_min": float(np.min([a for a, _ in v])),
            "auc_max": float(np.max([a for a, _ in v])),
            "log_loss_mean": float(np.mean([b for _, b in v])),
        }
        for k, v in out.items()
    }


def train_psxg(shots: Table) -> tuple[dict[str, Any], dict[str, Any]]:
    ot = (shots["on_target"] == 1) & np.isfinite(shots["end_y"]) & np.isfinite(shots["end_z"])
    t = {k: v[ot] for k, v in shots.items()}
    feats = design.post_shot_features_from_table(t)
    x = design.psxg_matrix(feats)
    y = t["goal"]
    tr, te = match_split(t["match_id"])
    mono = design.monotonic_vector(design.PSXG_FEATURES, design.PSXG_MONOTONIC)
    fit, val = inner_split(t["match_id"], tr)
    choice, _ = select_gbm(x, y, fit, val, mono, PSXG_GRID, max_iter=800)
    gbm = fit_gbm(x[tr], y[tr], mono, choice)
    p = gbm.predict_proba(x[te])[:, 1]
    scaler, lr = fit_logit(x[tr], y[tr])
    p_lr = lr.predict_proba(scaler.transform(x[te]))[:, 1]
    te_idx = np.flatnonzero(te)
    z_top = t["end_z"] >= 2.0
    m: dict[str, Any] = {
        "n_train": int(tr.sum()),
        "n_test": int(te.sum()),
        "selection": {"gbm_params": choice.params, "n_trees": choice.n_iter, "grid": choice.grid},
        "projection": {
            "saves_with_ratio_gt_cap": int(((t["projection_ratio"] > 1.3) & (y == 0)).sum()),
            "shots_at_bar_height_2.44": int((t["end_z"] >= 2.439).sum()),
            "goal_rate_z_ge_2m": float(y[z_top].mean()),
            "n_z_ge_2m": int(z_top.sum()),
        },
        "gbm": scores(y[te], p),
        "shipped_with_priors": scores(y[te], _psxg_shipped(gbm.decision_function(x[te]), feats, te_idx)),
        "keeper_beaten_prior_gt_0.5_share": float((design.keeper_beaten(feats) > 0.5).mean()),
        "logistic": scores(y[te], p_lr),
        "statsbomb_pre_shot_xg_on_same_shots": scores(y[te], t["sb_xg"][te]),
        "calibration": {"gbm": calibration(y[te], p), "logistic": calibration(y[te], p_lr)},
        "speed_proxy_median_mps": float(np.nanmedian(t["speed_proxy_mps"])),
        "speed_proxy_goal_rate_by_band": _speed_bands(t),
    }
    plot_calibration(
        DOCS_IMG_DIR / "psxg_calibration.png",
        "Post-shot xG calibration (on-target, held-out)",
        {"gradient boosting": m["calibration"]["gbm"], "logistic baseline": m["calibration"]["logistic"]},
    )
    final = fit_gbm(x, y, mono, choice)
    ens = export_checked(final, x, design.PSXG_FEATURES)
    idx = np.random.default_rng(2).choice(len(y), 64, replace=False)
    save_golden("psxg", feats, idx, _psxg_shipped(ens.raw(x[idx]), feats, idx))
    return {"ensemble": ens, "n_trees_final": int(final.n_iter_)}, m


def _psxg_shipped(raw_logit: FloatArray, feats: PostShotFeatures, idx: npt.NDArray[np.int64]) -> FloatArray:
    """The learned post-shot model with the priors exactly as analytics.models applies them (average keeper)."""
    sub = PostShotFeatures(**{f.name: np.asarray(getattr(feats, f.name))[idx] for f in fields(feats)})
    return psxg_with_priors(
        raw_logit, sub, PSXG_SPEED_LOGIT_PER_MPS, PSXG_REFERENCE_SPEED_MPS, PSXG_GK_DIVE_LOGIT_PER_POINT
    )


def _speed_bands(t: Table) -> dict[str, Any]:
    """Why speed is not learned: goal rate by StatsBomb speed-proxy band (falls with 'speed')."""
    out: dict[str, Any] = {}
    for lo, hi in ((5, 15), (15, 22), (22, 30), (30, 40)):
        sel = (t["speed_proxy_mps"] >= lo) & (t["speed_proxy_mps"] < hi)
        out[f"{lo}-{hi} m/s"] = {"goal_rate": float(t["goal"][sel].mean()), "n": int(sel.sum())}
    return out


def train_xpass(passes: Table) -> tuple[dict[str, Any], dict[str, Any]]:
    feats = design.pass_features_from_table(passes)
    x = design.xpass_matrix(feats)
    y = passes["completed"]
    tr, te = match_split(passes["match_id"])
    mono = design.monotonic_vector(design.XPASS_FEATURES, design.XPASS_MONOTONIC)
    fit, val = inner_split(passes["match_id"], tr)
    choice, _ = select_gbm(x, y, fit, val, mono, XPASS_GRID, max_iter=600)
    gbm = fit_gbm(x[tr], y[tr], mono, choice)
    p = gbm.predict_proba(x[te])[:, 1]
    scaler, lr = fit_logit(x[tr], y[tr])
    p_lr = lr.predict_proba(scaler.transform(x[te]))[:, 1]
    by_kind = {
        name: scores(y[te][passes["kind"][te] == code], p[passes["kind"][te] == code])
        for name, code in PASS_KIND.items()
        if (passes["kind"][te] == code).sum() > 50
    }
    failed = y == 0
    on_line = (np.abs(passes["end_y"]) >= 33.99) | (np.abs(passes["end_x"]) >= 52.49)
    m: dict[str, Any] = {
        "n_train": int(tr.sum()),
        "n_test": int(te.sum()),
        "selection": {"gbm_params": choice.params, "n_trees": choice.n_iter, "grid": choice.grid},
        "targets": {
            "target_to_recorded_end_m_p25_p50_p75_p95": {
                "completed": np.percentile(passes["target_proxy_m"][~failed], [25, 50, 75, 95]).tolist(),
                "failed": np.percentile(passes["target_proxy_m"][failed], [25, 50, 75, 95]).tolist(),
            },
            "share_on_line_failed": float(on_line[failed].mean()),
            "share_on_line_completed": float(on_line[~failed].mean()),
        },
        "gbm": scores(y[te], p),
        "logistic": scores(y[te], p_lr),
        "gbm_by_kind": by_kind,
        "calibration": {"gbm": calibration(y[te], p), "logistic": calibration(y[te], p_lr)},
    }
    plot_calibration(
        DOCS_IMG_DIR / "xpass_calibration.png",
        "xPass calibration (360 passes, held-out)",
        {"gradient boosting": m["calibration"]["gbm"], "logistic baseline": m["calibration"]["logistic"]},
    )
    final = fit_gbm(x, y, mono, choice)
    ens = export_checked(final, x, design.XPASS_FEATURES)
    idx = np.random.default_rng(3).choice(len(y), 64, replace=False)
    save_golden("xpass", feats, idx, ens.predict_proba(x[idx]))
    return {"ensemble": ens, "n_trees_final": int(final.n_iter_)}, m


def build_xt(moves: Table) -> tuple[FloatArray, dict[str, Any]]:
    mv = moves["kind"] == 0
    sh = ~mv
    grid, parts = xt.fit_xt(
        moves["x"][mv],
        moves["y"][mv],
        moves["end_x"][mv],
        moves["end_y"][mv],
        moves["success"][mv] == 1,
        moves["x"][sh],
        moves["y"][sh],
        moves["success"][sh] == 1,
    )
    col_mean = grid.mean(axis=1)
    rises = int(np.sum(np.diff(col_mean) > 0))
    m = {
        "n_moves": int(mv.sum()),
        "n_shots": int(sh.sum()),
        "column_means": col_mean.tolist(),
        "column_mean_increases": f"{rises}/{len(col_mean) - 1}",
        "max": float(grid.max()),
        "min": float(grid.min()),
    }
    fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=110)
    fig.patch.set_facecolor(SURFACE)
    im = ax.imshow(grid.T, origin="lower", extent=(-52.5, 52.5, -34, 34), cmap="Blues", aspect="equal")
    _style(ax, "Expected threat (16 x 12), attacking left to right", "x (m)", "y (m)")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03)
    cb.ax.tick_params(colors=INK_2, labelsize=8)
    fig.tight_layout()
    fig.savefig(DOCS_IMG_DIR / "xt_grid.png", facecolor=SURFACE)
    plt.close(fig)
    return grid, {**m, "parts": {k: v.tolist() for k, v in parts.items() if k != "n_actions"}}


def metrica_models() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Physical priors, pitch-control validation and distance covered from Metrica games 1-2."""
    kin_rows: list[dict[str, float]] = []
    rts: list[float] = []
    pc_val: dict[str, Any] = {}
    for game in (1, 2):
        kin = physical.load_game(game)
        for g in kin.values():
            kin_rows += physical.player_kinematics(g)
        ev = metrica.load_events(game)
        rts += physical.reaction_times(game, kin, ev)
        pc_val[f"game_{game}"] = physical.pass_control_validation(kin, ev, PitchControlParams())
    top = np.array([r["top_speed_mps"] for r in kin_rows])
    acc = np.array([r["peak_accel_mps2"] for r in kin_rows])
    dist = np.array([r["distance_m"] for r in kin_rows if r["full_match"]])
    priors = {
        "max_speed_mean": float(top.mean()),
        "max_speed_std": float(top.std(ddof=1)),
        "max_accel_mean": float(acc.mean()),
        "max_accel_std": float(acc.std(ddof=1)),
        "reaction_time_s": float(np.median(rts)),
        "reaction_time_definition": (
            "Decision + reaction time measured in hindsight, not a pure reflex: median delay from a Metrica pass "
            "(>= 10 m, >= 0.8 s flight) being struck until the defender nearest its target accelerates towards the "
            "target at > 1.5 m/s^2, over the passes in games 1-2 where that defender was not already accelerating "
            "towards it. The shared field name is reaction_time_s."
        ),
        "source": "metrica-sample-games-1-2",
        "n_players": len(kin_rows),
        "n_reaction_samples": len(rts),
        "reaction_time_iqr_s": [float(np.percentile(rts, 25)), float(np.percentile(rts, 75))],
    }
    distance = {
        "n_players": len(dist),
        "mean_m": float(dist.mean()),
        "std_m": float(dist.std(ddof=1)),
        "values_m": sorted(float(v) for v in dist),
    }
    return priors, pc_val, distance


def realism_targets(stats: Table, distance: dict[str, Any], matches: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-match / per-team distributions with a suggested band for the mean over >= 200 simulated matches."""
    male = {int(m["match_id"]) for m in matches if m["gender"] == "male"}
    sel = np.isin(stats["match_id"].astype(int), list(male))
    s = {k: v[sel] for k, v in stats.items()}
    mids = s["match_id"].astype(int)
    goals_match = np.array([s["goals"][mids == m].sum() for m in np.unique(mids)])
    passes = s["passes"]
    with np.errstate(invalid="ignore", divide="ignore"):
        completion = s["passes_completed"] / passes
        header_share = s["header_shots"] / s["shots"]
        shot_dist = s["shot_distance_sum"] / s["shots"]
    total_shots = s["shots"].sum()
    # Possession proxy: a team's share of the match's passes (StatsBomb has no possession clock).
    match_passes = {m: s["passes"][mids == m].sum() for m in np.unique(mids)}
    pass_share = 100 * passes / np.array([match_passes[m] for m in mids])
    comp_names = {int(m["competition_id"]): str(m["competition"]) for m in matches}

    def dist_stats(v: FloatArray, unit: str, rel_tol: float, pooled: float | None = None) -> dict[str, Any]:
        v = v[np.isfinite(v)]
        mean = float(v.mean()) if pooled is None else pooled
        tol = max(rel_tol * abs(mean), 0.0)
        return {
            "unit": unit,
            "mean": mean,
            "std": float(v.std(ddof=1)),
            "p10": float(np.percentile(v, 10)),
            "p50": float(np.percentile(v, 50)),
            "p90": float(np.percentile(v, 90)),
            "n": len(v),
            "band": [mean - tol, mean + tol],
            "band_rule": f"mean +/- {round(rel_tol * 100)}% (for the mean over >= 200 simulated matches)",
        }

    return {
        "version": VERSION,
        "description": (
            "Real-football distributions for the simulator realism test (PLAN.md Step 3). Box-score stats come "
            "from StatsBomb Open Data men's matches, regulation time only (periods 1-2); passes exclude throw-ins, "
            "completion = pass without an outcome (StatsBomb convention). Distance covered comes from Metrica "
            "sample games 1-2 (outfield players with >= 85 minutes). 'band' is a suggested tolerance for the "
            "simulator's mean over many matches, not for a single match."
        ),
        "sources": {
            "statsbomb_matches": len(np.unique(mids)),
            "statsbomb_team_matches": int(sel.sum()),
            "metrica_players": distance["n_players"],
        },
        "goals_per_match": dist_stats(goals_match.astype(float), "goals (both teams)", 0.15),
        "shots_per_team": dist_stats(s["shots"], "shots per team per match", 0.15),
        "pass_completion_pct": dist_stats(
            100 * completion,
            "% of passes completed",
            0.05,
            pooled=float(100 * s["passes_completed"].sum() / passes.sum()),
        ),
        "passes_per_team": dist_stats(passes, "passes per team per match (excl. throw-ins)", 0.15),
        "header_share_of_shots": dist_stats(
            header_share,
            "fraction of shots that are headers",
            0.25,
            pooled=float(s["header_shots"].sum() / total_shots),
        ),
        "mean_shot_distance_m": dist_stats(
            shot_dist, "metres from goal centre", 0.10, pooled=float(s["shot_distance_sum"].sum() / total_shots)
        ),
        "possession_share_by_passes_pct": {
            **dist_stats(pass_share, "team share of the match's passes (possession proxy), %", 0.0),
            "band_rule": "compare the std (match-to-match spread of the split): sim std within +/- 30% of this std",
            "std_band": [0.7 * float(pass_share.std(ddof=1)), 1.3 * float(pass_share.std(ddof=1))],
        },
        "distance_per_outfield_player_m": {
            **dist_stats(np.array(distance["values_m"]), "metres per outfield player per 90 min", 0.10),
        },
        "by_competition": {
            comp_names.get(int(c), str(int(c))): {
                "team_matches": int((s["competition_id"] == c).sum()),
                "goals_per_match": float(2 * s["goals"][s["competition_id"] == c].mean()),
                "shots_per_team": float(s["shots"][s["competition_id"] == c].mean()),
                "passes_per_team": float(s["passes"][s["competition_id"] == c].mean()),
                "pass_completion_pct": float(
                    100
                    * s["passes_completed"][s["competition_id"] == c].sum()
                    / s["passes"][s["competition_id"] == c].sum()
                ),
            }
            for c in np.unique(s["competition_id"])
        },
    }


# ----------------------------------------------------------------------------------------------- main


def _json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def main() -> None:
    t0 = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_IMG_DIR.mkdir(parents=True, exist_ok=True)
    shots, passes, moves, stats = (load_table(n) for n in ("shots", "passes", "moves", "team_stats"))
    matches = json.loads((cache_dir() / "matches.json").read_text())

    xg_art, xg_m = train_xg(shots)
    print(f"xG: {xg_m['shipped_blend']}  ({time.time() - t0:.0f}s)")
    psxg_art, psxg_m = train_psxg(shots)
    print(f"psxG: {psxg_m['gbm']}  ({time.time() - t0:.0f}s)")
    xpass_art, xpass_m = train_xpass(passes)
    print(f"xPass: {xpass_m['gbm']}  ({time.time() - t0:.0f}s)")
    grid, xt_m = build_xt(moves)
    print(f"xT: max {xt_m['max']:.3f}, column increases {xt_m['column_mean_increases']}")
    priors, pc_val, distance = metrica_models()
    print(f"priors: {priors}\npitch control validation: {pc_val}")
    targets = realism_targets(stats, distance, matches)

    xg_art["ensemble"].save(ARTIFACT_DIR / "xg_gbm.npz")
    psxg_art["ensemble"].save(ARTIFACT_DIR / "psxg_gbm.npz")
    xpass_art["ensemble"].save(ARTIFACT_DIR / "xpass_gbm.npz")
    _json(ARTIFACT_DIR / "xg_logistic.json", xg_art["logit"])
    _json(ARTIFACT_DIR / "xt_grid.json", {"nx": xt.NX, "ny": xt.NY, "frame": "attacking", "grid": grid.tolist()})
    _json(ARTIFACT_DIR / "physical_priors.json", priors)
    _json(ARTIFACT_DIR / "realism_targets.json", targets)
    metrics = {
        "xg": xg_m,
        "psxg": psxg_m,
        "xpass": xpass_m,
        "xt": xt_m,
        "pitch_control_validation": pc_val,
        "physical_priors": priors,
        "distance_covered": distance,
    }
    _json(ARTIFACT_DIR / "metrics.json", metrics)
    pc = PitchControlParams()
    manifest = {
        "version": VERSION,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "versions": {
            "xg": f"xg-hgb+logit-blend-{VERSION}",
            "psxg": f"psxg-hgb-{VERSION}",
            "xpass": f"xpass-hgb-{VERSION}",
            "xt": f"xt-singh-16x12-{VERSION}",
            "pc": f"pc-spearman2018-{VERSION}",
        },
        "files": {
            "xg": "xg_gbm.npz",
            "xg_logistic": "xg_logistic.json",
            "psxg": "psxg_gbm.npz",
            "xpass": "xpass_gbm.npz",
            "xt": "xt_grid.json",
            "physical_priors": "physical_priors.json",
            "realism_targets": "realism_targets.json",
            "metrics": "metrics.json",
        },
        "penalty_xg": xg_art["penalty"],
        "xg_blend_weight_gbm": xg_art["weight"],
        "physical_priors_note": (
            "reaction_time_s is decision + reaction measured in hindsight on Metrica passes (see "
            "physical_priors.json); pitch_control_params are Spearman's model constants, used for pitch control and "
            "the simulator's interception physics, not player kinematics."
        ),
        "psxg_gk_dive_logit_per_point": PSXG_GK_DIVE_LOGIT_PER_POINT,
        "psxg_speed_logit_per_mps": PSXG_SPEED_LOGIT_PER_MPS,
        "psxg_reference_speed_mps": PSXG_REFERENCE_SPEED_MPS,
        "pitch_control_params": {
            "reaction_time_s": pc.reaction_time_s,
            "max_player_speed_mps": pc.max_player_speed_mps,
            "tti_sigma": pc.tti_sigma,
            "lambda_att": pc.lambda_att,
            "lambda_def": pc.lambda_def,
            "average_ball_speed_mps": pc.average_ball_speed_mps,
        },
        "headline_metrics": {
            "xg_auc": xg_m["shipped_blend"]["auc"],
            "xg_log_loss": xg_m["shipped_blend"]["log_loss"],
            "psxg_auc": psxg_m["gbm"]["auc"],
            "xpass_auc": xpass_m["gbm"]["auc"],
        },
        "data": {
            "statsbomb": "StatsBomb Open Data, https://github.com/statsbomb/open-data",
            "metrica": "Metrica Sports sample data, https://github.com/metrica-sports/sample-data",
        },
    }
    _json(ARTIFACT_DIR / "manifest.json", manifest)
    print(f"done in {time.time() - t0:.0f}s -> {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
