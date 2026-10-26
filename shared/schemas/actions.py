"""Gesture -> intent -> technique (PLAN.md section 2.6).

``ActionIntent`` is what a fan gesture (or the AI decision model) asks for. The shared action resolver turns it
into a ``ResolvedAction`` (technique, body part, ball launch and execution uncertainty) using context and the
player's attributes. Fan and AI actions go through the same resolver and the same physics.
"""

from typing import Annotated, Literal

from pydantic import Field

from shared.schemas.common import Contract, PlayerId, Probability, Vec2, Vec3
from shared.schemas.enums import BodyPart, PassHeight, PassTechnique, PassType, ShotTechnique

IntentKind = Literal["pass_player", "pass_space", "shot", "carry", "clearance"]
"""pass_player: click a teammate. pass_space: drag into space (through ball). shot: click a spot in the goal.
carry: drag a path from the ball carrier (becomes a take-on if a defender is in the way). clearance: AI only."""

ActionType = Literal["pass", "shot", "carry", "dribble", "clearance"]


class ActionIntent(Contract):
    kind: IntentKind
    player: PlayerId
    """Ball carrier performing the action."""
    target_player: PlayerId | None = None
    """Required for pass_player."""
    target: Vec3 | None = None
    """pass_space: point on the pitch (z = 0). shot: point on the goal-mouth plane (x = goal line, y, z).
    carry: end of the path. Required for pass_space and shot."""
    path: list[Vec2] = []
    """Raw drag path (through-ball bend or carry route), pitch coordinates."""
    curve: Annotated[float, Field(ge=-1, le=1)] = 0.0
    """Requested bend, -1 (hard left) .. +1 (hard right) from the passer's view. Capped by attributes."""
    power: Annotated[float, Field(ge=0, le=1)] = 0.5
    """From gesture length / hold duration."""
    pass_technique_override: PassTechnique | None = None
    shot_technique_override: ShotTechnique | None = None
    source: Literal["ai", "fan"] = "ai"


class BallLaunch(Contract):
    speed_mps: Annotated[float, Field(ge=0, le=45)]
    elevation_rad: Annotated[float, Field(ge=-0.5, le=1.5)]
    azimuth_rad: float
    """Horizontal direction, radians from +x, counter-clockwise."""
    spin: Vec3 = (0.0, 0.0, 0.0)


class ExecutionEllipse(Contract):
    """1-sigma execution uncertainty at the intended end point, sized by the player's skill and pressure."""

    center: Vec3
    semi_major_m: Annotated[float, Field(ge=0)]
    semi_minor_m: Annotated[float, Field(ge=0)]
    angle_rad: float
    """Orientation of the major axis on the target plane."""


class ResolvedAction(Contract):
    intent: ActionIntent
    action_type: ActionType
    body_part: BodyPart | None = None
    pass_technique: PassTechnique | None = None
    pass_type: PassType | None = None
    pass_height: PassHeight | None = None
    shot_technique: ShotTechnique | None = None
    intended_end: Vec3
    launch: BallLaunch | None = None
    """None for carries and dribbles."""
    curve_applied: Annotated[float, Field(ge=-1, le=1)] = 0.0
    """Bend actually available after attribute caps (<= requested)."""
    ellipse: ExecutionEllipse | None = None
    preview_arc: list[Vec3] = []
    """Noise-free ball path samples for the preview UI."""
    override_rejected: bool = False
    """True when a requested technique override was not possible for this player or context."""
    weak_foot: bool = False
    execution_quality: Probability = 1.0
    """0..1 skill factor applied to noise (1 = elite, unpressured)."""
