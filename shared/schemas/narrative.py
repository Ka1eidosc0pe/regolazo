"""Narrative (verified, localized text with evidence) and ViewerProfile (per-session personalization)."""

from typing import Annotated, Literal

from pydantic import Field

from shared.schemas.common import SCHEMA_VERSION, Contract, LanguageCode, TeamSide, ViewerMode

NarrativeKind = Literal[
    "live_commentary",
    "why_it_matters",
    "milestone",
    "rewind_hook",
    "verdict",
    "recap",
]

EvidenceKind = Literal["event", "model_output", "stat", "rollout", "moment"]


class Evidence(Contract):
    """A tool output that backs a claim. Every number in a narrative must trace to one of these."""

    kind: EvidenceKind
    ref: str
    """Event id, moment id, rollout id, or stat key (e.g. "player.H07.top_speed_kmh")."""
    field: str | None = None
    value: float | str | None = None
    """The value as the tool reported it."""


class FactCheck(Contract):
    verified: bool
    attempts: Annotated[int, Field(ge=1)]
    issues: list[str] = []
    """Reasons for rejection from the last failed attempt (empty when verified first time)."""


class Narrative(Contract):
    schema_version: str = SCHEMA_VERSION
    id: str
    kind: NarrativeKind
    t: Annotated[float, Field(ge=0)]
    """Match time the narrative is about."""
    language: LanguageCode
    mode: ViewerMode
    club_perspective: TeamSide | None = None
    text: str
    evidence: list[Evidence] = []
    fact_check: FactCheck
    source_id: str | None = None
    """Id of the narrative this one was translated / adapted from."""
    audio_url: str | None = None
    voice: str | None = None


class ViewerProfile(Contract):
    schema_version: str = SCHEMA_VERSION
    session_id: str
    mode: ViewerMode = "casual"
    favorite_club: TeamSide | None = None
    language: LanguageCode = "en"
    """Must be one of the configured languages (validated by the server against config)."""
    voice_enabled: bool = False
    voice: str | None = None
    """Override of the default Azure neural voice for the language."""
    show_numbers_first: bool = False
    """Analyst toggle: show probabilities in Rewind before the fan commits."""
