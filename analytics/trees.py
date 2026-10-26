"""Numpy-only evaluation of gradient-boosted tree ensembles.

Models are trained with scikit-learn's ``HistGradientBoostingClassifier`` (a dev dependency) and exported to flat
node arrays in an ``.npz`` file, so inference needs only numpy and loads in milliseconds.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class TreeEnsemble:
    """Binary-classification ensemble: logit = baseline + sum of one leaf value per tree."""

    feature: npt.NDArray[np.int32]
    threshold: FloatArray
    left: npt.NDArray[np.int32]
    right: npt.NDArray[np.int32]
    missing_left: npt.NDArray[np.bool_]
    is_leaf: npt.NDArray[np.bool_]
    value: FloatArray
    roots: npt.NDArray[np.int32]
    baseline: float
    max_depth: int
    feature_names: tuple[str, ...]

    def raw(self, x: FloatArray) -> FloatArray:
        """Logit for each row of ``x`` (N x F, columns in ``feature_names`` order)."""
        n = x.shape[0]
        idx = np.broadcast_to(self.roots, (n, self.roots.size)).copy()
        rows = np.arange(n)[:, None]
        for _ in range(self.max_depth):
            leaf = self.is_leaf[idx]
            if leaf.all():
                break
            v = x[rows, self.feature[idx]]
            go_left = np.where(np.isnan(v), self.missing_left[idx], v <= self.threshold[idx])
            nxt = np.where(go_left, self.left[idx], self.right[idx])
            idx = np.where(leaf, idx, nxt)
        return np.asarray(self.baseline + self.value[idx].sum(axis=1), dtype=np.float64)

    def predict_proba(self, x: FloatArray) -> FloatArray:
        return np.asarray(1.0 / (1.0 + np.exp(-self.raw(x))), dtype=np.float64)

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            feature=self.feature,
            threshold=self.threshold,
            left=self.left,
            right=self.right,
            missing_left=self.missing_left,
            is_leaf=self.is_leaf,
            value=self.value.astype(np.float32),
            roots=self.roots,
            baseline=np.array([self.baseline]),
            max_depth=np.array([self.max_depth]),
            feature_names=np.array(self.feature_names),
        )

    @classmethod
    def load(cls, path: Path) -> "TreeEnsemble":
        with np.load(path) as z:
            return cls(
                feature=z["feature"].astype(np.int32),
                threshold=z["threshold"].astype(np.float64),
                left=z["left"].astype(np.int32),
                right=z["right"].astype(np.int32),
                missing_left=z["missing_left"].astype(bool),
                is_leaf=z["is_leaf"].astype(bool),
                value=z["value"].astype(np.float64),
                roots=z["roots"].astype(np.int32),
                baseline=float(z["baseline"][0]),
                max_depth=int(z["max_depth"][0]),
                feature_names=tuple(str(s) for s in z["feature_names"]),
            )


def from_sklearn(model: Any, feature_names: list[str]) -> TreeEnsemble:
    """Flatten a fitted binary ``HistGradientBoostingClassifier`` (numeric features only)."""
    feats, thr, left, right, miss, leaf, val, roots = [], [], [], [], [], [], [], []
    offset = 0
    depth = 0
    for stage in model._predictors:
        (pred,) = stage
        nodes = pred.nodes
        if np.any(nodes["is_categorical"]):
            raise ValueError("categorical splits are not supported by the exporter")
        roots.append(offset)
        feats.append(nodes["feature_idx"].astype(np.int32))
        thr.append(nodes["num_threshold"].astype(np.float64))
        left.append((nodes["left"] + offset).astype(np.int32))
        right.append((nodes["right"] + offset).astype(np.int32))
        miss.append(nodes["missing_go_to_left"].astype(bool))
        leaf.append(nodes["is_leaf"].astype(bool))
        val.append(nodes["value"].astype(np.float64))
        depth = max(depth, int(nodes["depth"].max()))
        offset += len(nodes)
    baseline = float(np.asarray(model._baseline_prediction).ravel()[0])
    return TreeEnsemble(
        feature=np.concatenate(feats),
        threshold=np.concatenate(thr),
        left=np.concatenate(left),
        right=np.concatenate(right),
        missing_left=np.concatenate(miss),
        is_leaf=np.concatenate(leaf),
        value=np.concatenate(val),
        roots=np.array(roots, dtype=np.int32),
        baseline=baseline,
        max_depth=depth + 1,
        feature_names=tuple(feature_names),
    )
