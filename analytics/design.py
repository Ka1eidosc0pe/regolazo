"""Design matrices for the learned models, built from the shared feature dataclasses.

The same functions build the training matrices (from StatsBomb tables) and the inference matrices (from simulator
scenes), so the two cannot drift. Integer codes are expanded into indicator columns.
"""

from typing import Any

import numpy as np
import numpy.typing as npt

from analytics.features import count_in_shot_cone, gk_geometry
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
from shared.schemas.common import PITCH_WIDTH_M

FloatArray = npt.NDArray[np.float64]

XG_CLIP: dict[str, tuple[float, float]] = {
    "nearest_defender_m": (0.0, 10.0),
    "gk_off_line": (0.0, 9.0),
    "gk_to_shooter": (0.0, 31.0),
    "gk_lateral": (0.0, 3.0),
}
"""Clip ranges at about the 99th percentile of non-penalty training shots (p99: 9.4 m, 8.7 m, 30.9 m, 2.9 m), applied
at train and inference. Only 8 training shots have the nearest defender >= 15 m away, so without the clip the linear
half of the blend extrapolates (an unmarked shot from the penalty spot scored 0.86, above the 0.77 penalty rate)."""
LANE_DEFENDERS_CAP = 3
"""Opponents in the passing lane are counted up to 3: beyond that the 360 data is too thin (about 1.0 % of passes)."""
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
    "gk_out_of_cone",
    "n_defenders_in_cone",
    "nearest_defender_m",
    "under_pressure",
    "first_time",
]
XG_MONOTONIC = {"distance": -1, "angle": 1, "n_defenders_in_cone": -1, "gk_out_of_cone": 1}

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


XG_LOGISTIC_FEATURES = [*XG_FEATURES, "log_distance", "head_x_distance", "gk_out_x_distance"]


def logit(p: FloatArray) -> FloatArray:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.asarray(np.log(p / (1 - p)), dtype=np.float64)


def xg_logistic_matrix(x: FloatArray) -> FloatArray:
    """xG design matrix plus the interactions the linear model needs: log distance, header x distance, and keeper out
    of the cone x distance (an empty net makes distance matter much less)."""
    dist = x[:, XG_FEATURES.index("distance")]
    head = x[:, XG_FEATURES.index("is_head")]
    gk_out = x[:, XG_FEATURES.index("gk_out_of_cone")]
    return np.column_stack([x, np.log(np.maximum(dist, 0.5)), head * dist, gk_out * dist])


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
        "gk_off_line": np.clip(off_line, *XG_CLIP["gk_off_line"]),
        "gk_to_shooter": np.clip(to_shooter, *XG_CLIP["gk_to_shooter"]),
        "gk_lateral": np.clip(lateral, *XG_CLIP["gk_lateral"]),
        # Keeper outside the shooter-post-post triangle (beaten / out of position): 548 training shots, which score
        # 2-5x as often as shots from the same distance with the keeper in the cone. Unlike gk_off_line it does not
        # need clipping, so it carries the empty-net effect at any distance.
        "gk_out_of_cone": count_in_shot_cone(_f(f.x), _f(f.y), np.stack([_f(f.gk_x), _f(f.gk_y)], axis=1)[:, None, :])
        == 0,
        "n_defenders_in_cone": f.n_defenders_in_cone,
        "nearest_defender_m": np.clip(_f(f.nearest_defender_m), *XG_CLIP["nearest_defender_m"]),
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


KEEPER_REACH_M = 4.0
"""Full-stretch dive plus a step, with no time to move (prior, not fit)."""
KEEPER_SPEED_MPS = 7.0
KEEPER_REACTION_S = 0.25
KEEPER_BEATEN_MARGIN_M = 6.0
"""How far beyond his reach the crossing point must be before the keeper counts as beaten."""
BEATEN_PSXG = 0.84
"""Observed goal rate of the 37 on-target training shots whose crossing point was > 6 m beyond the keeper's reach
(at a 22 m/s reference speed); 18 shots > 8 m beyond: 0.83."""


def keeper_beaten(f: PostShotFeatures) -> FloatArray:
    """P(the keeper cannot get to the ball), a guard for keepers far out of position.

    The keeper covers KEEPER_REACH_M plus KEEPER_SPEED_MPS for the ball's flight time after reacting. Keepers more than
    ~9 m off their line are 1 % of training shots, so the learned model extrapolates flat there (a keeper 30 m out
    would still save about half of the shots on target); this term lifts post-shot xG to at least the observed
    BEATEN_PSXG once the crossing point is KEEPER_BEATEN_MARGIN_M beyond his reach. For a keeper near his line it is ~0.
    """
    flight = _f(f.distance_m) / np.clip(_f(f.speed_mps), *SPEED_RANGE_MPS)
    reach = KEEPER_REACH_M + KEEPER_SPEED_MPS * np.maximum(flight - KEEPER_REACTION_S, 0.0)
    end_y = np.clip(_f(f.end_y), -HALF_GOAL, HALF_GOAL)
    gap = np.hypot(GOAL_X - _f(f.gk_x), _f(f.gk_y) - end_y)
    return np.asarray(1.0 / (1.0 + np.exp(-(gap - reach - KEEPER_BEATEN_MARGIN_M))), dtype=np.float64)


def xpass_matrix(f: PassFeatures) -> FloatArray:
    """End points are clipped inside the touchlines and goal lines (training targets never lie on or beyond them)."""
    end_x = np.clip(_f(f.end_x), -GOAL_X, GOAL_X)
    end_y = np.clip(_f(f.end_y), -PITCH_WIDTH_M / 2, PITCH_WIDTH_M / 2)
    dx = end_x - _f(f.x)
    dy = end_y - _f(f.y)
    length = np.hypot(dx, dy)
    safe = np.maximum(length, 1e-6)
    k, h = f.kind, f.height
    cols = {
        "x": _f(f.x),
        "abs_y": np.abs(_f(f.y)),
        "end_x": end_x,
        "abs_end_y": np.abs(end_y),
        "length": length,
        "dir_cos": dx / safe,
        "dir_sin_abs": np.abs(dy) / safe,
        "end_goal_dist": np.hypot(GOAL_X - end_x, end_y),
        "height_low": h == PASS_HEIGHT["low"],
        "height_high": h == PASS_HEIGHT["high"],
        "kind_cross": k == PASS_KIND["cross"],
        "kind_cutback": k == PASS_KIND["cutback"],
        "kind_through_ball": k == PASS_KIND["through_ball"],
        "kind_switch": k == PASS_KIND["switch"],
        "kind_back_pass": k == PASS_KIND["back_pass"],
        "kind_chipped": k == PASS_KIND["chipped"],
        "nearest_defender_to_receiver_m": np.minimum(_f(f.nearest_defender_to_receiver_m), 30.0),
        "defenders_in_lane": np.minimum(f.defenders_in_lane, LANE_DEFENDERS_CAP),
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
    """Training rows -> PostShotFeatures. The keeper's y uses the goal-plane scale, like the ball's end y."""
    dist, _ = shot_geometry(_f(t["x"]), _f(t["y"]))
    return PostShotFeatures(
        end_y=_f(t["end_y"]),
        end_z=_f(t["end_z"]),
        speed_mps=np.nan_to_num(_f(t["speed_proxy_mps"]), nan=20.0),
        distance_m=dist,
        gk_x=_f(t["gk_x"]),
        gk_y=_f(t["gk_y_goal_plane"]),
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
