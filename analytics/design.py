"""Design matrices for the learned models, built from the shared feature dataclasses.

The same functions build the training matrices (from StatsBomb tables) and the inference matrices (from simulator
scenes), so the two cannot drift. Integer codes are expanded into indicator columns.
"""

from typing import Any

import numpy as np
import numpy.typing as npt

from analytics.features import gk_geometry
from shared.model_api import (
    GOAL_X,
    HALF_GOAL,
    PASS_HEIGHT,
    PASS_KIND,
    SHOT_BODY_PART,
    SHOT_SITUATION,
    SHOT_TECHNIQUE,
    PassFeatures,
    PostShotFeatures,
    ShotFeatures,
    shot_geometry,
)

FloatArray = npt.NDArray[np.float64]

SPEED_RANGE_MPS = (5.0, 40.0)
"""Speed proxy clip range: StatsBomb durations are noisy, so the tails of distance / duration are not trusted."""

XG_FEATURES = [
    "distance",
    "angle",
    "abs_y",
    "is_head",
    "is_other_body",
    "tech_chip",
    "tech_volley",
    "tech_half_volley",
    "tech_diving_header",
    "tech_backheel",
    "tech_overhead",
    "sit_counter",
    "sit_set_piece",
    "sit_free_kick",
    "gk_off_line",
    "gk_to_shooter",
    "gk_lateral",
    "n_defenders_in_cone",
    "nearest_defender_m",
    "under_pressure",
    "first_time",
]
XG_MONOTONIC = {"distance": -1, "angle": 1, "n_defenders_in_cone": -1}

PSXG_FEATURES = ["abs_end_y", "end_z", "distance", "gk_off_line", "gk_to_end", "gk_to_end_y"]
"""Ball speed is deliberately not learned: the StatsBomb speed proxy (distance / event duration) is anti-correlated
with goals (fast "shots" are mostly saves), an artefact of how durations are recorded. Speed enters post-shot xG as a
documented prior in ``analytics.models`` instead."""
PSXG_MONOTONIC = {"abs_end_y": 1, "gk_to_end": 1, "gk_to_end_y": 1}

XPASS_FEATURES = [
    "x",
    "abs_y",
    "end_x",
    "abs_end_y",
    "length",
    "dir_cos",
    "dir_sin_abs",
    "end_goal_dist",
    "height_low",
    "height_high",
    "kind_cross",
    "kind_cutback",
    "kind_through_ball",
    "kind_switch",
    "kind_back_pass",
    "kind_chipped",
    "nearest_defender_to_receiver_m",
    "defenders_in_lane",
    "under_pressure",
]
XPASS_MONOTONIC = {"defenders_in_lane": -1, "nearest_defender_to_receiver_m": 1}


XG_LOGISTIC_FEATURES = [*XG_FEATURES, "log_distance", "head_x_distance"]


def logit(p: FloatArray) -> FloatArray:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.asarray(np.log(p / (1 - p)), dtype=np.float64)


def xg_logistic_matrix(x: FloatArray) -> FloatArray:
    """xG design matrix plus the two terms the linear model needs (log distance, header x distance)."""
    dist = x[:, XG_FEATURES.index("distance")]
    head = x[:, XG_FEATURES.index("is_head")]
    return np.column_stack([x, np.log(np.maximum(dist, 0.5)), head * dist])


def monotonic_vector(names: list[str], cst: dict[str, int]) -> list[int]:
    return [cst.get(n, 0) for n in names]


def _f(a: Any) -> FloatArray:
    return np.asarray(a, dtype=np.float64)


def xg_matrix(f: ShotFeatures) -> FloatArray:
    dist, angle = shot_geometry(_f(f.x), _f(f.y))
    off_line, to_shooter, lateral = gk_geometry(_f(f.x), _f(f.y), _f(f.gk_x), _f(f.gk_y))
    t, s, b = f.technique, f.situation, f.body_part
    cols = {
        "distance": dist,
        "angle": angle,
        "abs_y": np.abs(f.y),
        "is_head": b == SHOT_BODY_PART["head"],
        "is_other_body": b == SHOT_BODY_PART["other"],
        "tech_chip": t == SHOT_TECHNIQUE["chip"],
        "tech_volley": t == SHOT_TECHNIQUE["volley"],
        "tech_half_volley": t == SHOT_TECHNIQUE["half_volley"],
        "tech_diving_header": t == SHOT_TECHNIQUE["diving_header"],
        "tech_backheel": t == SHOT_TECHNIQUE["backheel"],
        "tech_overhead": t == SHOT_TECHNIQUE["overhead_kick"],
        "sit_counter": s == SHOT_SITUATION["counter_attack"],
        "sit_set_piece": s == SHOT_SITUATION["set_piece"],
        "sit_free_kick": s == SHOT_SITUATION["direct_free_kick"],
        "gk_off_line": off_line,
        "gk_to_shooter": to_shooter,
        "gk_lateral": lateral,
        "n_defenders_in_cone": f.n_defenders_in_cone,
        "nearest_defender_m": np.minimum(f.nearest_defender_m, 30.0),
        "under_pressure": f.under_pressure,
        "first_time": f.first_time,
    }
    return np.column_stack([_f(cols[n]) for n in XG_FEATURES])


def psxg_matrix(f: PostShotFeatures) -> FloatArray:
    end_y = np.clip(_f(f.end_y), -HALF_GOAL, HALF_GOAL)
    end_z = np.clip(_f(f.end_z), 0.0, 2.44)
    cols = {
        "abs_end_y": np.abs(end_y),
        "end_z": end_z,
        "distance": _f(f.distance_m),
        "gk_off_line": GOAL_X - _f(f.gk_x),
        "gk_to_end": np.hypot(_f(f.gk_y) - end_y, np.maximum(end_z - 1.0, 0.0)),
        "gk_to_end_y": np.abs(_f(f.gk_y) - end_y),
    }
    return np.column_stack([cols[n] for n in PSXG_FEATURES])


def xpass_matrix(f: PassFeatures) -> FloatArray:
    dx = _f(f.end_x) - _f(f.x)
    dy = _f(f.end_y) - _f(f.y)
    length = np.hypot(dx, dy)
    safe = np.maximum(length, 1e-6)
    k, h = f.kind, f.height
    cols = {
        "x": _f(f.x),
        "abs_y": np.abs(_f(f.y)),
        "end_x": _f(f.end_x),
        "abs_end_y": np.abs(_f(f.end_y)),
        "length": length,
        "dir_cos": dx / safe,
        "dir_sin_abs": np.abs(dy) / safe,
        "end_goal_dist": np.hypot(GOAL_X - _f(f.end_x), _f(f.end_y)),
        "height_low": h == PASS_HEIGHT["low"],
        "height_high": h == PASS_HEIGHT["high"],
        "kind_cross": k == PASS_KIND["cross"],
        "kind_cutback": k == PASS_KIND["cutback"],
        "kind_through_ball": k == PASS_KIND["through_ball"],
        "kind_switch": k == PASS_KIND["switch"],
        "kind_back_pass": k == PASS_KIND["back_pass"],
        "kind_chipped": k == PASS_KIND["chipped"],
        "nearest_defender_to_receiver_m": np.minimum(_f(f.nearest_defender_to_receiver_m), 30.0),
        "defenders_in_lane": f.defenders_in_lane,
        "under_pressure": f.under_pressure,
    }
    return np.column_stack([_f(cols[n]) for n in XPASS_FEATURES])


def _i(a: npt.NDArray[Any]) -> npt.NDArray[np.int64]:
    return np.asarray(a, dtype=np.int64)


def _b(a: npt.NDArray[Any]) -> npt.NDArray[np.bool_]:
    return np.asarray(a, dtype=bool)


def shot_features_from_table(t: dict[str, npt.NDArray[Any]]) -> ShotFeatures:
    return ShotFeatures(
        x=_f(t["x"]),
        y=_f(t["y"]),
        body_part=_i(t["body_part"]),
        technique=_i(t["technique"]),
        situation=_i(t["situation"]),
        gk_x=_f(t["gk_x"]),
        gk_y=_f(t["gk_y"]),
        n_defenders_in_cone=_i(t["n_defenders_in_cone"]),
        nearest_defender_m=_f(t["nearest_defender_m"]),
        under_pressure=_b(t["under_pressure"]),
        first_time=_b(t["first_time"]),
    )


def post_shot_features_from_table(t: dict[str, npt.NDArray[Any]]) -> PostShotFeatures:
    dist, _ = shot_geometry(_f(t["x"]), _f(t["y"]))
    return PostShotFeatures(
        end_y=_f(t["end_y"]),
        end_z=_f(t["end_z"]),
        speed_mps=np.nan_to_num(_f(t["speed_proxy_mps"]), nan=20.0),
        distance_m=dist,
        gk_x=_f(t["gk_x"]),
        gk_y=_f(t["gk_y"]),
        gk_dive=np.full(len(t["x"]), 50.0),
    )


def pass_features_from_table(t: dict[str, npt.NDArray[Any]]) -> PassFeatures:
    return PassFeatures(
        x=_f(t["x"]),
        y=_f(t["y"]),
        end_x=_f(t["end_x"]),
        end_y=_f(t["end_y"]),
        height=_i(t["height"]),
        kind=_i(t["kind"]),
        nearest_defender_to_receiver_m=_f(t["nearest_defender_to_receiver_m"]),
        defenders_in_lane=_i(t["defenders_in_lane"]),
        under_pressure=_b(t["under_pressure"]),
    )
