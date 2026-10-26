"""Shared primitives for all ReGolazo contracts.

Coordinate frame (global, fixed for the whole match):
- Pitch 105 x 68 m, origin at the centre spot.
- x runs along the length, from -52.5 (left goal line) to +52.5 (right goal line).
- y runs across the width, from -34 to +34; y = +34 is the far touchline as seen from the main camera.
- z is height above the grass, in metres (up is positive).
- Which team attacks +x in each period is given by ``MatchMeta.home_attacks_positive_x_first_half``.
- Velocities are m/s, angles are radians, time ``t`` is match seconds since the first kickoff
  (continuous across periods; ``period`` disambiguates).

The renderer maps this frame to Three.js (y-up) as (x, y, z) -> (x, z, -y).
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0.0"

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0
GOAL_WIDTH_M = 7.32
GOAL_HEIGHT_M = 2.44


class Contract(BaseModel):
    """Base class for every contract model: strict about unknown fields, defaults always serialized."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_serialization_defaults_required=True,
        use_enum_values=True,
        validate_default=True,
    )


Vec2 = tuple[float, float]
"""(x, y) on the pitch plane, metres."""

Vec3 = tuple[float, float, float]
"""(x, y, z) in metres (or m/s for velocities, rad/s for spin)."""

TeamSide = Literal["home", "away"]

PlayerId = Annotated[str, Field(pattern=r"^[HA]\d{2}$", examples=["H07", "A10"])]
"""Team letter (H/A) followed by the two-digit shirt number, e.g. ``H07``."""

EventId = Annotated[str, Field(pattern=r"^ev_[0-9a-z_]+$", examples=["ev_000123"])]

LanguageCode = Annotated[str, Field(pattern=r"^[a-z]{2}(-[A-Z]{2})?$", examples=["en", "es", "fr"])]
"""BCP-47 style language code. The supported list is configuration, not an enum."""

Probability = Annotated[float, Field(ge=0.0, le=1.0)]

ViewerMode = Literal["analyst", "casual"]

Period = Literal[1, 2]
