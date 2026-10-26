"""Fork-and-rollout contracts: request, single result, and the "1,000 realities" distribution."""

from typing import Annotated, Literal

from pydantic import Field

from shared.schemas.actions import ActionIntent, ResolvedAction
from shared.schemas.common import SCHEMA_VERSION, Contract, EventId, Probability, Vec2
from shared.schemas.events import MatchEvent
from shared.schemas.models import OptionEvaluation
from shared.schemas.state import SimState

RolloutOutcomeLabel = Literal[
    "goal",
    "shot_saved",
    "shot_off_target",
    "shot_blocked",
    "woodwork",
    "chance_created",
    "possession_retained",
    "turnover",
    "out_of_play",
    "foul_won",
    "offside",
]


class RolloutRequest(Contract):
    snapshot_ref: str
    intent: ActionIntent
    horizon_s: Annotated[float, Field(ge=1, le=20)] = 10.0
    n: Annotated[int, Field(ge=1, le=10_000)] = 1
    seed: int | None = None
    """None = server picks; the chosen seed is echoed back for reproducibility."""
    return_frames: bool = True
    """Only honoured for n == 1."""
    frame_stride: Annotated[int, Field(ge=1)] = 1


class RolloutOutcome(Contract):
    label: RolloutOutcomeLabel
    primary_success: bool
    """Whether the fan's action itself succeeded (pass completed / shot scored / carry kept the ball)."""
    terminal_t: Annotated[float, Field(ge=0)]
    xg_generated: Annotated[float, Field(ge=0)]
    """Sum of xG of shots in the horizon."""
    xt_delta: float
    ended_by_event_id: EventId | None = None


class RolloutResult(Contract):
    schema_version: str = SCHEMA_VERSION
    id: str
    snapshot_ref: str
    seed: int
    resolved: ResolvedAction
    evaluation: OptionEvaluation
    """Model view of the fan's action at the fork (player-adjusted p_success, xG): revealed after commit."""
    frames: list[SimState] = []
    events: list[MatchEvent] = []
    """Events inside the rollout: ``rollout_id`` set, ids ``ev_r_<rollout>_<n>``."""
    outcome: RolloutOutcome


class Interval(Contract):
    lo: Probability
    hi: Probability


class OutcomeCount(Contract):
    label: RolloutOutcomeLabel
    count: Annotated[int, Field(ge=0)]


class RolloutSample(Contract):
    """One reality, compact enough to plot hundreds of them."""

    label: RolloutOutcomeLabel
    primary_success: bool
    end_xy: Vec2
    """Where the fan's ball ended up (pass arrival, shot goal-plane y and z packed as (y, z) for shots)."""
    xg: Annotated[float, Field(ge=0)]


class RolloutDistribution(Contract):
    schema_version: str = SCHEMA_VERSION
    snapshot_ref: str
    intent: ActionIntent
    resolved: ResolvedAction
    n: Annotated[int, Field(ge=1)]
    seed: int
    outcome_counts: list[OutcomeCount]
    p_primary_success: Probability
    p_primary_success_ci95: Interval
    p_goal: Probability
    p_goal_ci95: Interval
    mean_xg: Annotated[float, Field(ge=0)]
    mean_xt_delta: float
    model_p_success: Probability
    """Displayed model probability (player-adjusted xPass / xG) for the consistency check."""
    elapsed_ms: Annotated[float, Field(ge=0)]
    samples: list[RolloutSample] = []
    """Up to 300 individual realities for the "1,000 realities" visualization."""


VerdictFraming = Literal["beat_the_odds", "unlucky", "as_expected", "bold_and_paid_off", "same_call_as_reality"]


class Verdict(Contract):
    """Shown after a Rewind: the fan's call vs the real one (PLAN.md section 2.1 step 6)."""

    schema_version: str = SCHEMA_VERSION
    moment_id: str
    user_option: OptionEvaluation
    real_option: OptionEvaluation
    rollout_id: str
    user_outcome: RolloutOutcome
    real_outcome_label: RolloutOutcomeLabel
    framing: VerdictFraming
    value_gap: float
    """user decision_value - real decision_value (only ever shown after the fan commits)."""
    distribution: RolloutDistribution | None = None
    narrative_id: str | None = None
