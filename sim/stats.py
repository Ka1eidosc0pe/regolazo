"""Match-level realism statistics from ground-truth events and frame arrays (docs/sim_realism.md)."""

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from shared.schemas.events import MatchEvent

REPO = Path(__file__).resolve().parents[1]
TARGETS_FILE = REPO / "analytics" / "artifacts" / "realism_targets.json"

DEFAULT_TARGETS: dict[str, tuple[float, float]] = {
    "goals_per_match": (2.5, 3.0),
    "shots_per_team": (10.0, 15.0),
    "pass_completion": (0.75, 0.85),
    "passes_per_team": (350.0, 550.0),
    "header_share_of_shots": (0.10, 0.20),
    "mean_shot_distance_m": (16.0, 19.0),
    "distance_per_outfield_player_km": (9.0, 12.0),
}


def targets() -> dict[str, tuple[float, float]]:
    """Target bands; Step 2's realism_targets.json replaces the defaults when it exists."""
    out = dict(DEFAULT_TARGETS)
    if not TARGETS_FILE.exists():
        return out
    data = json.loads(TARGETS_FILE.read_text())
    # file key -> (our key, unit factor)
    mapping = {
        "goals_per_match": ("goals_per_match", 1.0),
        "shots_per_team": ("shots_per_team", 1.0),
        "pass_completion_pct": ("pass_completion", 0.01),
        "passes_per_team": ("passes_per_team", 1.0),
        "header_share_of_shots": ("header_share_of_shots", 1.0),
        "mean_shot_distance_m": ("mean_shot_distance_m", 1.0),
        "distance_per_outfield_player_m": ("distance_per_outfield_player_km", 0.001),
    }
    for src, (dst, f) in mapping.items():
        band = data.get(src, {}).get("band") if isinstance(data.get(src), dict) else None
        if band and len(band) == 2:
            out[dst] = (float(band[0]) * f, float(band[1]) * f)
    poss = data.get("possession_share_by_passes_pct", {})
    if isinstance(poss, dict) and poss.get("std_band"):
        out["possession_share_sd"] = (float(poss["std_band"][0]), float(poss["std_band"][1]))
    return out


@dataclass
class MatchStats:
    goals: int
    shots: int
    shots_home: int
    shots_away: int
    passes: int
    passes_complete: int
    headers: int
    shot_distance_sum: float
    distance_km_per_outfield: float
    possession_home: float
    xg: float
    fouls: int
    corners: int
    offsides: int
    type_counts: dict[str, int]

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def match_stats(events: list[MatchEvent], arrays: dict[str, npt.NDArray[Any]] | None) -> MatchStats:
    shots = [e for e in events if e.type == "shot"]
    passes = [e for e in events if e.type == "pass" and getattr(e.qualifiers, "restart", None) != "throw_in"]
    goals = sum(1 for e in shots if e.qualifiers is not None and getattr(e.qualifiers, "outcome", None) == "goal")
    goals += sum(1 for e in events if e.type == "own_goal")
    headers = sum(1 for e in shots if getattr(e.qualifiers, "body_part", None) == "head")
    sd = sum(float(getattr(e.qualifiers, "distance_to_goal_m", 0.0)) for e in shots)
    complete = sum(1 for e in passes if getattr(e.qualifiers, "outcome", None) == "complete")
    xg = sum(float(getattr(e.qualifiers, "xg", 0.0) or 0.0) for e in shots)
    dist_km = 0.0
    poss_home = 0.5
    if arrays is not None and len(arrays.get("player_pos", [])):
        pos = arrays["player_pos"].astype(np.float64)
        on = arrays["player_on_pitch"]
        step = np.linalg.norm(np.diff(pos, axis=0), axis=2)
        step = np.where(on[1:] & on[:-1] & (step < 1.0), step, 0.0)  # drop teleports (half time)
        per_player = step.sum(axis=0)
        outfield = on.any(axis=0)
        # goalkeepers: the first player of each squad
        from shared.recording import ANIMS

        _ = ANIMS
        totals = per_player[outfield]
        gk_mask = np.zeros(len(per_player), dtype=bool)
        anim = arrays["player_anim"]
        gk_anim = (anim == list(ANIMS).index("gk_idle")).mean(axis=0) > 0.05
        gk_mask[gk_anim] = True
        keep = outfield & ~gk_mask
        dist_km = float(per_player[keep].mean() / 1000.0) if keep.any() else float(totals.mean() / 1000)
        pt = arrays["poss_team"]
        live = pt >= 0
        poss_home = float((pt[live] == 0).mean()) if live.any() else 0.5
    return MatchStats(
        goals=goals,
        shots=len(shots),
        shots_home=sum(1 for e in shots if e.team == "home"),
        shots_away=sum(1 for e in shots if e.team == "away"),
        passes=len(passes),
        passes_complete=complete,
        headers=headers,
        shot_distance_sum=sd,
        distance_km_per_outfield=dist_km,
        possession_home=poss_home,
        xg=xg,
        fouls=sum(1 for e in events if e.type == "foul_committed"),
        corners=sum(1 for e in events if e.type == "restart" and getattr(e.qualifiers, "restart_type", "") == "corner"),
        offsides=sum(1 for e in events if e.type == "offside"),
        type_counts=dict(Counter(e.type for e in events)),
    )


def aggregate(stats: list[MatchStats]) -> dict[str, float]:
    n = max(len(stats), 1)
    shots = sum(s.shots for s in stats)
    passes = sum(s.passes for s in stats)
    return {
        "matches": float(len(stats)),
        "goals_per_match": sum(s.goals for s in stats) / n,
        "shots_per_team": shots / (2 * n),
        "pass_completion": sum(s.passes_complete for s in stats) / max(passes, 1),
        "passes_per_team": passes / (2 * n),
        "header_share_of_shots": sum(s.headers for s in stats) / max(shots, 1),
        "mean_shot_distance_m": sum(s.shot_distance_sum for s in stats) / max(shots, 1),
        "distance_per_outfield_player_km": float(np.mean([s.distance_km_per_outfield for s in stats])),
        "xg_per_match": sum(s.xg for s in stats) / n,
        "possession_home_sd": float(np.std([s.possession_home for s in stats])),
        "fouls_per_match": sum(s.fouls for s in stats) / n,
        "corners_per_match": sum(s.corners for s in stats) / n,
        "offsides_per_match": sum(s.offsides for s in stats) / n,
    }
