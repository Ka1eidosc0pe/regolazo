"""FrameLabels: per-frame ground truth written by the renderer's recording mode and consumed by the Spotter.

Pixel coordinates: origin top-left, x right, y down, in the rendered image of size ``width`` x ``height``.
Camera matrices map the world frame of shared.schemas.common (z up) to pixels:
``p ~ K [R | t] X_world`` with K the 3x3 intrinsics (row-major) and [R | t] the 3x4 world-to-camera extrinsics
(row-major, OpenCV convention: camera looks down +z, image y down).
"""

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from shared.schemas.common import SCHEMA_VERSION, Contract, PlayerId, TeamSide, Vec3

LabelClass = Literal["player_home", "player_away", "goalkeeper_home", "goalkeeper_away", "referee", "ball"]

PixelBox = tuple[float, float, float, float]
"""(x_min, y_min, x_max, y_max) in pixels."""


class CameraCalibration(Contract):
    intrinsics: Annotated[list[float], Field(min_length=9, max_length=9)]
    extrinsics: Annotated[list[float], Field(min_length=12, max_length=12)]
    fov_y_deg: Annotated[float, Field(gt=0, lt=180)]
    position: Vec3
    """Camera centre in world coordinates."""


class ObjectLabel(Contract):
    cls: LabelClass
    track_id: str
    """Player id (``H07``), ``ball`` or ``referee``; stable across frames."""
    team: TeamSide | None = None
    player: PlayerId | None = None
    box: PixelBox
    """Tight box of the projected 3D bounds, clipped to the image."""
    visible_fraction: Annotated[float, Field(ge=0, le=1)]
    """Share of the object's projected area that is not occluded and inside the image."""
    world: Vec3
    """Ground-truth world position (feet midpoint for people, centre for the ball)."""
    center_px: tuple[float, float]
    """Projection of ``world`` (ball centre, or feet point for people)."""


class FrameLabels(Contract):
    schema_version: str = SCHEMA_VERSION
    match_id: str
    frame: Annotated[int, Field(ge=0)]
    t: Annotated[float, Field(ge=0)]
    tick: Annotated[int, Field(ge=0)]
    """Simulator tick the frame was rendered from (interpolated frames use the earlier tick)."""
    image: str
    """Path of the frame image relative to the recording directory."""
    width: Annotated[int, Field(gt=0)]
    height: Annotated[int, Field(gt=0)]
    camera_mode: Literal["broadcast", "tactical", "replay"]
    camera: CameraCalibration
    objects: list[ObjectLabel]
    ball_shadow_px: tuple[float, float] | None = None
    """Projection of the ball's ground shadow point (x, y, 0): a single-camera height cue."""

    @model_validator(mode="after")
    def _check_boxes(self) -> Self:
        for o in self.objects:
            x0, y0, x1, y1 = o.box
            if not (0 <= x0 <= x1 <= self.width and 0 <= y0 <= y1 <= self.height):
                raise ValueError(f"box {o.box} of {o.track_id} is outside the {self.width}x{self.height} image")
        return self
