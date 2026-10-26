"""MatchEvent: one football event, emitted as simulator ground truth or detected by the Spotter.

Structure follows docs/event_taxonomy.md: a common envelope plus a typed qualifier block selected by ``kind``.
Locations are in the world frame of ``shared.schemas.common`` (not the attacking frame); use
``shared.geometry.to_attacking_frame`` before applying the taxonomy's direction-dependent thresholds.
"""

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from shared.schemas.common import (
    SCHEMA_VERSION,
    Contract,
    EventId,
    Period,
    PlayerId,
    Probability,
    TeamSide,
    Vec3,
)
from shared.schemas.enums import (
    AccelType,
    BlockType,
    BodyOrientation,
    BodyPart,
    Card,
    CarryOutcome,
    ClearanceOutcome,
    CornerDelivery,
    DeflectionTarget,
    DuelOutcome,
    DuelType,
    EventType,
    FoulType,
    GkActionType,
    GkDistribution,
    GkOutcome,
    GkPosition,
    GkTechnique,
    InterceptionOutcome,
    OutLine,
    PassHeight,
    PassLength,
    PassOutcome,
    PassTechnique,
    PassType,
    PhaseOfPlay,
    PlayPattern,
    PossessionAfter,
    PressedRole,
    PressSequenceOutcome,
    PressureOutcome,
    ReceiptOutcome,
    RestartType,
    RunOutcome,
    RunType,
    ShotOutcome,
    ShotSituation,
    ShotTechnique,
    StepUpOutcome,
    TakeOnOutcome,
    TouchQuality,
)

NonNeg = Annotated[float, Field(ge=0)]

ON_TARGET_OUTCOMES: frozenset[str] = frozenset({"goal", "saved", "saved_to_post"})


# --- Tier A qualifiers --------------------------------------------------------------------------------------


class PassQ(Contract):
    kind: Literal["pass"] = "pass"
    recipient: PlayerId | None
    """Intended receiver; None for passes into space with no clear target."""
    outcome: PassOutcome
    height: PassHeight | None
    length_class: PassLength
    body_part: BodyPart | None
    technique: PassTechnique | None
    pass_type: PassType | None
    restart: RestartType | None = None
    corner_delivery: CornerDelivery | None = None
    gk_distribution: GkDistribution | None = None
    length_m: NonNeg
    angle_rad: float
    """World-frame direction, radians from +x, counter-clockwise."""
    ball_speed_mps: NonNeg
    peak_height_m: NonNeg
    duration_s: NonNeg | None = None
    is_switch: bool = False
    is_progressive: bool = False
    is_line_breaking: bool = False
    lines_broken: Annotated[int, Field(ge=0, le=3)] = 0
    is_key_pass: bool = False
    is_assist: bool = False
    first_time: bool = False
    deflected: bool = False
    recipient_offside: bool = False
    assisted_shot_id: EventId | None = None
    receipt_id: EventId | None = None
    xpass: Probability | None = None
    """Model completion probability at release (ground truth / Analyst)."""


class ReceiptQ(Contract):
    kind: Literal["ball_receipt"] = "ball_receipt"
    passer: PlayerId | None
    outcome: ReceiptOutcome
    touch_quality: TouchQuality | None
    body_orientation: BodyOrientation | None
    first_touch_body_part: BodyPart | None = None
    pressure_distance_m: NonNeg | None = None
    aerial: bool = False
    pass_id: EventId | None = None


class CarryQ(Contract):
    kind: Literal["carry"] = "carry"
    outcome: CarryOutcome
    distance_m: NonNeg
    path_m: NonNeg
    max_speed_mps: NonNeg
    is_progressive: bool = False


class TakeOnQ(Contract):
    kind: Literal["take_on"] = "take_on"
    opponent: PlayerId | None
    outcome: TakeOnOutcome
    nutmeg: bool = False
    overrun: bool = False
    no_touch: bool = False


class ShotQ(Contract):
    kind: Literal["shot"] = "shot"
    keeper: PlayerId | None
    outcome: ShotOutcome
    body_part: BodyPart | None
    technique: ShotTechnique | None
    situation: ShotSituation | None
    ball_speed_mps: NonNeg
    distance_to_goal_m: NonNeg
    angle_rad: NonNeg
    """Angle subtended by the goal posts at the shot location."""
    peak_height_m: NonNeg | None = None
    on_target: bool
    xg: Probability | None = None
    psxg: Probability | None = None
    first_time: bool = False
    one_on_one: bool = False
    open_goal: bool = False
    follows_dribble: bool = False
    aerial_won: bool = False
    deflected: bool = False
    n_defenders_in_cone: Annotated[int, Field(ge=0)] | None = None
    gk_position: GkPosition | None = None
    key_pass_id: EventId | None = None
    block_event_id: EventId | None = None
    gk_event_id: EventId | None = None
    freeze_frame_ref: str | None = None
    """Snapshot handle of the SimState at contact (ground truth)."""

    @model_validator(mode="after")
    def _check_target(self) -> Self:
        if self.on_target != (self.outcome in ON_TARGET_OUTCOMES):
            raise ValueError(f"on_target={self.on_target} contradicts outcome {self.outcome!r}")
        if self.psxg is not None and not self.on_target:
            raise ValueError("psxg is only defined for on-target shots")
        return self


class DuelQ(Contract):
    kind: Literal["duel"] = "duel"
    opponent: PlayerId | None
    duel_type: DuelType
    outcome: DuelOutcome
    possession_after: PossessionAfter | None = None


class InterceptionQ(Contract):
    kind: Literal["interception"] = "interception"
    passer: PlayerId | None
    outcome: InterceptionOutcome
    pass_id: EventId | None = None


class BlockQ(Contract):
    kind: Literal["block"] = "block"
    block_type: BlockType
    deflection: bool = False
    save_block: bool = False
    blocked_event_id: EventId | None = None


class ClearanceQ(Contract):
    kind: Literal["clearance"] = "clearance"
    body_part: BodyPart | None
    outcome: ClearanceOutcome
    aerial_won: bool = False


class RecoveryQ(Contract):
    kind: Literal["ball_recovery"] = "ball_recovery"
    offensive: bool = False
    recovery_failure: bool = False


class TouchLossQ(Contract):
    """Dispossessed, miscontrol or deflection."""

    kind: Literal["touch_loss"] = "touch_loss"
    opponent: PlayerId | None = None
    aerial_won: bool = False
    deflection_target: DeflectionTarget | None = None


class FoulQ(Contract):
    kind: Literal["foul"] = "foul"
    opponent: PlayerId | None
    """Fouled player (for foul_committed) or offender (for foul_won)."""
    foul_type: FoulType = "foul"
    card: Card = "none"
    advantage: bool = False
    penalty: bool = False
    offensive: bool = False


class GkQ(Contract):
    kind: Literal["gk_action"] = "gk_action"
    action: GkActionType
    outcome: GkOutcome
    technique: GkTechnique | None = None
    body_part: BodyPart | None = None
    shot_id: EventId | None = None
    position_before: GkPosition | None = None
    dive_distance_m: NonNeg | None = None


class RestartQ(Contract):
    kind: Literal["restart"] = "restart"
    restart_type: RestartType
    awarded_to: TeamSide
    cause_event_id: EventId | None = None
    wall_size: Annotated[int, Field(ge=0, le=11)] | None = None
    corner_delivery: CornerDelivery | None = None


class OutOfPlayQ(Contract):
    kind: Literal["ball_out_of_play"] = "ball_out_of_play"
    line: OutLine
    last_touch_team: TeamSide | None
    last_touch_player: PlayerId | None
    result_restart_type: RestartType | None


class OffsideQ(Contract):
    kind: Literal["offside"] = "offside"
    passer: PlayerId | None
    offside_line_x: float
    margin_m: float
    """Positive = distance beyond the offside line."""
    pass_id: EventId | None = None


# --- Tier B qualifiers --------------------------------------------------------------------------------------


class PressureQ(Contract):
    kind: Literal["pressure"] = "pressure"
    pressed_player: PlayerId | None
    """None when the Spotter only knows the pressed player's track (see MatchEvent.secondary_track_id)."""
    pressed_role: PressedRole
    min_distance_m: NonNeg
    closing_speed_mps: float
    n_pressers: Annotated[int, Field(ge=1)] = 1
    outcome: PressureOutcome = "none"


class PressingSequenceQ(Contract):
    kind: Literal["pressing_sequence"] = "pressing_sequence"
    n_pressures: Annotated[int, Field(ge=2)]
    pressing_players: list[PlayerId]
    outcome: PressSequenceOutcome
    pressure_ids: list[EventId] = []


class RunQ(Contract):
    kind: Literal["off_ball_run"] = "off_ball_run"
    run_type: RunType
    outcome: RunOutcome
    max_speed_mps: NonNeg
    distance_m: NonNeg
    carrier: PlayerId | None = None
    """Ball carrier the run relates to (overlap / underlap / checking)."""
    targeted: bool = False
    offside_at_release: bool = False
    dragged_defender: PlayerId | None = None
    lateral_gap_m: float | None = None


class PhysicalQ(Contract):
    """Sprint, high-intensity run or acceleration."""

    kind: Literal["physical"] = "physical"
    distance_m: NonNeg
    peak_speed_mps: NonNeg
    accel_type: AccelType | None = None
    peak_accel_mps2: float | None = None
    speed_start_mps: NonNeg | None = None
    speed_end_mps: NonNeg | None = None


class StepUpQ(Contract):
    kind: Literal["line_step_up"] = "line_step_up"
    line_players: list[PlayerId]
    line_height_before: float
    line_height_after: float
    """Mean world x of the back line before / after the step-up."""
    outcome: StepUpOutcome


Qualifiers = Annotated[
    PassQ
    | ReceiptQ
    | CarryQ
    | TakeOnQ
    | ShotQ
    | DuelQ
    | InterceptionQ
    | BlockQ
    | ClearanceQ
    | RecoveryQ
    | TouchLossQ
    | FoulQ
    | GkQ
    | RestartQ
    | OutOfPlayQ
    | OffsideQ
    | PressureQ
    | PressingSequenceQ
    | RunQ
    | PhysicalQ
    | StepUpQ,
    Field(discriminator="kind"),
]

QUALIFIER_KIND: dict[str, str | None] = {
    "pass": "pass",
    "ball_receipt": "ball_receipt",
    "carry": "carry",
    "take_on": "take_on",
    "shot": "shot",
    "duel": "duel",
    "interception": "interception",
    "block": "block",
    "clearance": "clearance",
    "ball_recovery": "ball_recovery",
    "dispossessed": "touch_loss",
    "miscontrol": "touch_loss",
    "deflection": "touch_loss",
    "foul_committed": "foul",
    "foul_won": "foul",
    "gk_action": "gk_action",
    "restart": "restart",
    "ball_out_of_play": "ball_out_of_play",
    "offside": "offside",
    "own_goal": None,
    "period_start": None,
    "period_end": None,
    "pressure": "pressure",
    "pressing_sequence": "pressing_sequence",
    "off_ball_run": "off_ball_run",
    "sprint": "physical",
    "high_intensity_run": "physical",
    "acceleration": "physical",
    "line_step_up": "line_step_up",
}
"""Which qualifier block each event type carries (None = no qualifiers)."""

GROUND_TRUTH_REQUIRED: dict[str, tuple[str, ...]] = {
    "pass": ("height", "body_part", "technique", "pass_type"),
    "ball_receipt": ("touch_quality", "body_orientation"),
    "shot": ("body_part", "technique", "situation"),
    "clearance": ("body_part",),
    "pressure": ("pressed_player",),
}
"""Qualifier fields that are nullable only for detected events (the Spotter may not see them)."""


class MatchEvent(Contract):
    schema_version: str = SCHEMA_VERSION
    id: EventId
    index: Annotated[int, Field(ge=0)]
    """Order within the match (per source)."""
    source: Literal["ground_truth", "detected"]
    confidence: Probability | None = None
    """Detected events only."""
    type: EventType
    period: Period
    t: NonNeg
    """Match seconds since the first kickoff (start of the event)."""
    tick: Annotated[int, Field(ge=0)] | None = None
    duration_s: NonNeg = 0.0
    team: TeamSide | None
    player: PlayerId | None
    """Primary actor; None for team-level events (period_start, pressing_sequence) or unidentified tracks."""
    track_id: Annotated[int, Field(ge=0)] | None = None
    """Spotter track of the primary actor (detected events; the Spotter may not know the player id)."""
    secondary_track_id: Annotated[int, Field(ge=0)] | None = None
    """Spotter track of the second actor (recipient, opponent, pressed player)."""
    rollout_id: str | None = None
    """Set on events produced inside a Rewind rollout; their ids are namespaced ``ev_r_<rollout>_<n>``."""
    location: Vec3 | None
    """World frame. Ball position at the start for on-ball events, actor position otherwise."""
    end_location: Vec3 | None = None
    possession_id: Annotated[int, Field(ge=0)] | None = None
    possession_team: TeamSide | None = None
    play_pattern: PlayPattern = "regular_play"
    phase_of_play: PhaseOfPlay | None = None
    under_pressure: bool = False
    counterpress: bool = False
    related_events: list[EventId] = []
    qualifiers: Qualifiers | None = None

    @model_validator(mode="after")
    def _check_qualifiers(self) -> Self:
        expected = QUALIFIER_KIND[self.type]
        actual = self.qualifiers.kind if self.qualifiers is not None else None
        if expected is None and actual is not None:
            raise ValueError(f"event type {self.type!r} takes no qualifiers, got {actual!r}")
        if expected is not None and actual != expected and not (self.source == "detected" and actual is None):
            raise ValueError(f"event type {self.type!r} needs qualifiers of kind {expected!r}, got {actual!r}")
        if self.source == "ground_truth":
            if self.confidence is not None:
                raise ValueError("ground-truth events carry no confidence")
            if self.track_id is not None or self.secondary_track_id is not None:
                raise ValueError("track ids are for detected events")
            if self.qualifiers is not None:
                required = GROUND_TRUTH_REQUIRED.get(self.qualifiers.kind, ())
                missing = [f for f in required if getattr(self.qualifiers, f) is None]
                if missing:
                    raise ValueError(f"ground-truth {self.type!r} needs qualifiers {missing}")
        if self.rollout_id is not None and not self.id.startswith("ev_r_"):
            raise ValueError("rollout events must use ids of the form ev_r_<rollout>_<n>")
        return self


class EventPair(Contract):
    """A detected event matched to a ground-truth event (Spotter evaluation)."""

    detected_id: EventId
    ground_truth_id: EventId
    dt_s: float
    distance_m: NonNeg | None = None
