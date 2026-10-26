"""StatsSnapshot (Analyst output) and Recap. Also the stat-key grammar used by ``Evidence(kind="stat")``.

Stat key grammar: a dotted path into ``StatsSnapshot.model_dump()``, e.g. ``teams.home.xg``,
``players.H07.top_speed_kmh``, ``control_chaos``. ``stat_value(snapshot, key)`` resolves one, so the
Fact-Checker can compare any number in a narrative with the value the Analyst reported at that time.
"""

from typing import Annotated, Any

from pydantic import Field

from shared.schemas.common import SCHEMA_VERSION, Contract, PlayerId, TeamSide

NonNeg = Annotated[float, Field(ge=0)]
Count = Annotated[int, Field(ge=0)]


class TeamStats(Contract):
    goals: Count = 0
    shots: Count = 0
    shots_on_target: Count = 0
    xg: NonNeg = 0.0
    passes: Count = 0
    passes_completed: Count = 0
    pass_accuracy: Annotated[float, Field(ge=0, le=1)] = 0.0
    possession: Annotated[float, Field(ge=0, le=1)] = 0.5
    progressive_passes: Count = 0
    final_third_entries: Count = 0
    pressures: Count = 0
    ppda: NonNeg | None = None
    field_tilt: Annotated[float, Field(ge=0, le=1)] = 0.5
    xt_generated: float = 0.0
    distance_km: NonNeg = 0.0
    sprints: Count = 0
    corners: Count = 0
    fouls: Count = 0


class PlayerStats(Contract):
    team: TeamSide
    touches: Count = 0
    passes: Count = 0
    passes_completed: Count = 0
    key_passes: Count = 0
    progressive_passes: Count = 0
    progressive_carries: Count = 0
    shots: Count = 0
    goals: Count = 0
    assists: Count = 0
    xg: NonNeg = 0.0
    xt_generated: float = 0.0
    take_ons_won: Count = 0
    tackles_won: Count = 0
    interceptions: Count = 0
    pressures: Count = 0
    distance_m: NonNeg = 0.0
    top_speed_kmh: NonNeg = 0.0
    sprints: Count = 0
    high_intensity_runs: Count = 0
    longest_pass_m: NonNeg = 0.0
    fastest_shot_kmh: NonNeg = 0.0


class StatsSnapshot(Contract):
    schema_version: str = SCHEMA_VERSION
    t: NonNeg
    teams: dict[TeamSide, TeamStats]
    players: dict[PlayerId, PlayerStats]
    control_chaos: Annotated[float, Field(ge=-1, le=1)]
    """-1 = total control, +1 = total chaos (rolling window)."""
    momentum: Annotated[float, Field(ge=-1, le=1)]
    """+1 = home dominating, -1 = away dominating (rolling xT/xG flow)."""
    milestones: list[str] = []
    """Milestone keys reached so far, e.g. "first_shot_on_target:home"."""


def stat_value(snapshot: StatsSnapshot, key: str) -> Any:
    """Resolve a stat key (dotted path) against a snapshot. Raises KeyError if the path does not exist."""
    node: Any = snapshot.model_dump(mode="json")
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(key)
        node = node[part]
    return node


class WhatIf(Contract):
    moment_id: str
    verdict_framing: str | None = None
    p_goal_alternative: Annotated[float, Field(ge=0, le=1)]
    p_goal_real: Annotated[float, Field(ge=0, le=1)]
    narrative_id: str | None = None


class Recap(Contract):
    """Full-time recap (PLAN.md Step 10). Text lives in Narratives; this holds structure and references."""

    schema_version: str = SCHEMA_VERSION
    match_id: str
    final_stats: StatsSnapshot
    key_moment_ids: list[str]
    what_ifs: list[WhatIf]
    player_of_the_match: PlayerId
    narrative_ids: list[str] = []
