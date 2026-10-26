"""DecisionMoment: a frozen on-ball moment that can be rewound and rewritten (PLAN.md section 2.5)."""

from typing import Annotated, Literal

from pydantic import Field

from shared.schemas.actions import ResolvedAction
from shared.schemas.common import SCHEMA_VERSION, Contract, EventId, Period, PlayerId, TeamSide
from shared.schemas.models import OptionEvaluation

MomentType = Literal[
    "final_pass",
    "shot",
    "one_v_one",
    "counter_attack",
    "decisive_turnover",
    "set_piece",
    "build_up",
    "other",
]

ConsequenceKind = Literal[
    "goal",
    "shot",
    "big_chance",
    "turnover_to_chance_against",
    "possession_value_swing",
    "none",
]


class RewindScores(Contract):
    """Judgement-free selection signals, each normalised to 0..1.

    The value gap between the real choice and the alternatives is deliberately NOT a selection signal;
    it is used only in the verdict after the fan commits.
    """

    richness: Annotated[float, Field(ge=0, le=1)]
    stakes: Annotated[float, Field(ge=0, le=1)]
    consequence: Annotated[float, Field(ge=0, le=1)]
    salience: Annotated[float, Field(ge=0, le=1)]
    total: Annotated[float, Field(ge=0, le=1)]
    viable_options: Annotated[int, Field(ge=0)]
    option_entropy: Annotated[float, Field(ge=0)]


class Consequence(Contract):
    """What happened in the ~10 s after the moment in the real match."""

    kind: ConsequenceKind
    xt_swing: float
    horizon_s: Annotated[float, Field(gt=0)] = 10.0
    event_ids: list[EventId] = []


class DecisionMoment(Contract):
    schema_version: str = SCHEMA_VERSION
    id: str
    tick: Annotated[int, Field(ge=0)]
    t: Annotated[float, Field(ge=0)]
    period: Period
    snapshot_ref: str
    """Handle of the sim snapshot to fork from."""
    team: TeamSide
    ball_carrier: PlayerId
    moment_type: MomentType
    origin: Literal["scout", "fan"]
    """Flagged by the Moment Scout, or picked by the fan while scrubbing."""
    real_action: ResolvedAction
    real_event_id: EventId | None = None
    consequence: Consequence
    options: list[OptionEvaluation]
    """Includes the real option (``is_real``)."""
    scores: RewindScores | None = None
    """Set for Scout-flagged moments."""
    flagged: bool = False
    hook: str | None = None
    """Neutral English hook from the Moment Scout; localized copies are Narratives."""
