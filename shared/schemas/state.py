"""SimState: the full observable world at one simulator tick (streamed to the browser, recorded for training)."""

from typing import Annotated, Literal

from pydantic import Field

from shared.schemas.common import SCHEMA_VERSION, Contract, Period, PlayerId, TeamSide, Vec2, Vec3
from shared.schemas.enums import PhaseOfPlay, RestartType

PlayerAnim = Literal[
    "idle",
    "jog",
    "run",
    "sprint",
    "jog_back",
    "strafe_left",
    "strafe_right",
    "turn",
    "receive_ground",
    "receive_aerial",
    "dribble",
    "pass_short",
    "pass_long",
    "cross",
    "shoot",
    "chip",
    "volley",
    "header_standing",
    "header_jumping",
    "tackle",
    "slide_tackle",
    "fall",
    "get_up",
    "throw_in",
    "gk_idle",
    "gk_dive_left",
    "gk_dive_right",
    "gk_block_left",
    "gk_block_right",
    "gk_catch_high",
    "gk_catch_low",
    "gk_scoop",
    "gk_throw",
    "gk_drop_kick",
    "celebrate",
    "disappointed",
]
"""Animation state that drives the renderer's state machine (PLAN.md Step 3/5)."""

MatchStatus = Literal["pre_match", "first_half", "half_time", "second_half", "full_time"]


class BallState(Contract):
    pos: Vec3
    vel: Vec3
    spin: Vec3 = (0.0, 0.0, 0.0)
    """Angular velocity (rad/s) in the pitch frame; drives Magnus curl."""
    in_play: bool = True
    last_touch: PlayerId | None = None


class PlayerState(Contract):
    id: PlayerId
    pos: Vec2
    vel: Vec2
    facing: float
    """Body orientation, radians from +x, counter-clockwise."""
    z: Annotated[float, Field(ge=0)] = 0.0
    """Vertical offset of the body (jumps, dives)."""
    anim: PlayerAnim = "idle"
    anim_t: Annotated[float, Field(ge=0)] = 0.0
    """Seconds since ``anim`` started, so one-shot clips stay in sync with the sim."""
    stamina: Annotated[float, Field(ge=0, le=1)] = 1.0
    on_pitch: bool = True


class RefereeState(Contract):
    pos: Vec2
    vel: Vec2
    facing: float = 0.0


class Possession(Contract):
    team: TeamSide | None
    player: PlayerId | None
    """Player in control of the ball; None while the ball is in flight or loose."""
    sequence_id: Annotated[int, Field(ge=0)]
    """Increments every time possession changes team."""


class Score(Contract):
    home: Annotated[int, Field(ge=0)] = 0
    away: Annotated[int, Field(ge=0)] = 0


class SimState(Contract):
    schema_version: str = SCHEMA_VERSION
    tick: Annotated[int, Field(ge=0)]
    t: Annotated[float, Field(ge=0)]
    period: Period
    status: MatchStatus
    ball: BallState
    players: list[PlayerState]
    referee: RefereeState | None = None
    possession: Possession
    phase: PhaseOfPlay
    """Phase of the team in possession (in-possession values), or ``dead_ball``."""
    restart: RestartType | None = None
    """Set while the ball is dead and a restart is pending."""
    score: Score
    attacking_positive_x: TeamSide
    """Team attacking towards +x in the current period."""
    keyframe: bool = False
    """True on ticks where a touch occurred: these are the fan's scrub-back points for Rewind."""
    snapshot_ref: str | None = None
    """Handle for forking from this tick (set on keyframes); opaque to clients."""
