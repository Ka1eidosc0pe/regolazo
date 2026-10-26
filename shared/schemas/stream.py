"""WebSocket protocol between the live engine (server/) and the browser (web/).

One JSON message per WebSocket text frame, discriminated by ``type``. ``ServerMessage`` flows server -> client,
``ClientMessage`` client -> server. Frames are full ``SimState`` JSON (no delta encoding; ~150 KB/s at 25 Hz).
Fork, preview and rollouts are REST calls (see shared.engine_api), not messages. The first server message on
every connection is ``hello`` with the ``MatchMeta``; frames follow at the playback rate.
"""

from typing import Annotated, Literal

from pydantic import Field

from shared.schemas.common import SCHEMA_VERSION, Contract
from shared.schemas.events import MatchEvent
from shared.schemas.meta import MatchMeta
from shared.schemas.moments import DecisionMoment
from shared.schemas.narrative import Narrative, ViewerProfile
from shared.schemas.overlay import OverlayCue
from shared.schemas.state import SimState

PlaybackSpeed = Annotated[float, Field(gt=0, le=16)]


class ClockStatus(Contract):
    t: Annotated[float, Field(ge=0)]
    tick: Annotated[int, Field(ge=0)]
    playing: bool
    speed: PlaybackSpeed
    duration_s: Annotated[float, Field(ge=0)]
    """Total length of the match timeline."""


# --- server -> client ---------------------------------------------------------------------------------------


class HelloMsg(Contract):
    type: Literal["hello"] = "hello"
    schema_version: str = SCHEMA_VERSION
    session_id: str
    meta: MatchMeta
    profile: ViewerProfile
    clock: ClockStatus


class FrameMsg(Contract):
    type: Literal["frame"] = "frame"
    state: SimState


class EventsMsg(Contract):
    type: Literal["events"] = "events"
    events: list[MatchEvent]


class ClockMsg(Contract):
    """Sent on every play / pause / seek / speed change, and after a seek before the next frame."""

    type: Literal["clock"] = "clock"
    clock: ClockStatus


class CueMsg(Contract):
    type: Literal["cues"] = "cues"
    cues: list[OverlayCue]


class NarrativeMsg(Contract):
    type: Literal["narrative"] = "narrative"
    narrative: Narrative


class MomentMsg(Contract):
    """A rewind opportunity (Scout-flagged) became available."""

    type: Literal["moment"] = "moment"
    moment: DecisionMoment


class ErrorMsg(Contract):
    type: Literal["error"] = "error"
    code: str
    message: str


ServerMessage = Annotated[
    HelloMsg | FrameMsg | EventsMsg | ClockMsg | CueMsg | NarrativeMsg | MomentMsg | ErrorMsg,
    Field(discriminator="type"),
]


# --- client -> server ---------------------------------------------------------------------------------------


class PlayCmd(Contract):
    type: Literal["play"] = "play"


class PauseCmd(Contract):
    type: Literal["pause"] = "pause"


class SeekCmd(Contract):
    type: Literal["seek"] = "seek"
    t: Annotated[float, Field(ge=0)]


class SpeedCmd(Contract):
    type: Literal["speed"] = "speed"
    speed: PlaybackSpeed


class ProfileCmd(Contract):
    type: Literal["profile"] = "profile"
    profile: ViewerProfile


ClientMessage = Annotated[PlayCmd | PauseCmd | SeekCmd | SpeedCmd | ProfileCmd, Field(discriminator="type")]


class StreamProtocol(Contract):
    """Container so the protocol unions get a JSON Schema and TS types. Not sent on the wire."""

    server: ServerMessage
    client: ClientMessage
