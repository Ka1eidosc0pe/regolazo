"""OverlayCue: machine-readable timed overlay instructions emitted by the Director.

Payloads carry numbers and references only. Display text comes from Narratives (localized) or from UI string
tables, so one cue renders in any language and mode.
"""

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from shared.schemas.common import (
    SCHEMA_VERSION,
    Contract,
    EventId,
    PlayerId,
    Probability,
    TeamSide,
    Vec2,
    ViewerMode,
)
from shared.schemas.enums import EventType, PassOutcome, PassTechnique, PassType, ShotOutcome, ShotTechnique

ScreenRegion = Literal["top_left", "top_right", "top_center", "bottom_left", "bottom_right", "lower_third", "center"]


class CueAnchor(Contract):
    kind: Literal["player", "screen", "pitch"]
    player: PlayerId | None = None
    region: ScreenRegion | None = None
    pitch_xy: Vec2 | None = None

    @model_validator(mode="after")
    def _check_target(self) -> Self:
        needed = {"player": self.player, "screen": self.region, "pitch": self.pitch_xy}[self.kind]
        if needed is None:
            raise ValueError(f"anchor kind {self.kind!r} needs its matching field set")
        return self


class NameTagPayload(Contract):
    type: Literal["name_tag"] = "name_tag"
    player: PlayerId
    emphasis: Literal["normal", "highlight"] = "normal"


class ThresholdBadgePayload(Contract):
    type: Literal["threshold_badge"] = "threshold_badge"
    player: PlayerId
    metric: Literal["top_speed", "sprint_speed", "distance_covered", "sprint_count", "shot_speed", "pass_distance"]
    value: float
    unit: Literal["km/h", "m/s", "m", "km", "count"]
    threshold: float
    record: bool = False
    """True when this beats the match record for the metric."""


class PassCardPayload(Contract):
    type: Literal["pass_card"] = "pass_card"
    event_id: EventId
    passer: PlayerId
    recipient: PlayerId | None
    length_m: float
    ball_speed_kmh: float
    xpass: Probability
    difficulty: Annotated[int, Field(ge=1, le=10)]
    """1 (routine) .. 10 (outrageous), from 1 - xPass."""
    technique: PassTechnique
    pass_type: PassType
    outcome: PassOutcome


class ShotCardPayload(Contract):
    type: Literal["shot_card"] = "shot_card"
    event_id: EventId
    shooter: PlayerId
    xg: Probability
    psxg: Probability | None = None
    ball_speed_kmh: float
    distance_m: float
    technique: ShotTechnique
    outcome: ShotOutcome


class SeriesPoint(Contract):
    t: float
    value: float


class MomentumChartPayload(Contract):
    type: Literal["momentum_chart"] = "momentum_chart"
    home: list[SeriesPoint]
    away: list[SeriesPoint]
    window_s: float


class ControlChaosPayload(Contract):
    type: Literal["control_chaos"] = "control_chaos"
    index: Annotated[float, Field(ge=-1, le=1)]
    """-1 = total control, +1 = total chaos."""
    series: list[SeriesPoint] = []
    drivers: dict[str, float] = {}
    """Component contributions, e.g. {"turnover_rate": 0.4, "pressure_intensity": 0.3}."""


class AccuracyRow(Contract):
    event_type: EventType
    precision: Probability
    recall: Probability
    f1: Probability
    support: Annotated[int, Field(ge=0)]


class AccuracyPanelPayload(Contract):
    type: Literal["accuracy_panel"] = "accuracy_panel"
    rows: list[AccuracyRow]
    window: Literal["live_window", "match_so_far"] = "match_so_far"


class RewindPromptPayload(Contract):
    type: Literal["rewind_prompt"] = "rewind_prompt"
    moment_id: str
    narrative_id: str | None = None


class CaptionPayload(Contract):
    type: Literal["caption"] = "caption"
    narrative_id: str
    speak: bool = False
    """Play the narrative's TTS audio if the viewer has voice on."""


class MilestonePayload(Contract):
    type: Literal["milestone"] = "milestone"
    narrative_id: str
    milestone: str
    """Stable key, e.g. "first_shot_on_target", "pass_50", "sprint_record"."""
    team: TeamSide | None = None
    player: PlayerId | None = None


class HighlightPayload(Contract):
    type: Literal["highlight"] = "highlight"
    players: list[PlayerId]
    style: Literal["ring", "spotlight", "trail", "arrow"] = "ring"


CuePayload = Annotated[
    NameTagPayload
    | ThresholdBadgePayload
    | PassCardPayload
    | ShotCardPayload
    | MomentumChartPayload
    | ControlChaosPayload
    | AccuracyPanelPayload
    | RewindPromptPayload
    | CaptionPayload
    | MilestonePayload
    | HighlightPayload,
    Field(discriminator="type"),
]


class OverlayCue(Contract):
    schema_version: str = SCHEMA_VERSION
    id: str
    t_start: Annotated[float, Field(ge=0)]
    t_end: Annotated[float, Field(ge=0)]
    priority: Annotated[int, Field(ge=0, le=100)]
    """Higher wins when the Director resolves collisions."""
    anchor: CueAnchor
    payload: CuePayload
    modes: list[ViewerMode] = ["analyst", "casual"]
    """Viewer modes that may see this cue (the Personal Producer filters further)."""
    team_focus: TeamSide | None = None
    evidence: list[EventId] = []

    @model_validator(mode="after")
    def _check_window(self) -> Self:
        if self.t_end < self.t_start:
            raise ValueError("t_end must be >= t_start")
        return self
