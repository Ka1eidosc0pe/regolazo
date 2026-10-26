"""ModelOutputs: analytics model results at one moment (xG, xPass, pitch control, xT, decision values)."""

from typing import Annotated, Self

from pydantic import Field, model_validator

from shared.schemas.actions import ActionIntent, ResolvedAction
from shared.schemas.common import SCHEMA_VERSION, Contract, PlayerId, Probability, TeamSide, Vec2


class PitchControlGrid(Contract):
    """Probability that ``team`` controls each cell. Row-major: ``values[iy * nx + ix]``.

    Cell (ix, iy) is centred at ``(x0 + ix * dx, y0 + iy * dy)`` in world coordinates. To tile the pitch exactly,
    use ``dx = 105 / nx``, ``dy = 68 / ny``, ``x0 = -52.5 + dx / 2``, ``y0 = -34 + dy / 2``.
    """

    team: TeamSide
    nx: Annotated[int, Field(gt=0)]
    ny: Annotated[int, Field(gt=0)]
    x0: float
    y0: float
    dx: Annotated[float, Field(gt=0)]
    dy: Annotated[float, Field(gt=0)]
    values: list[Probability]

    @model_validator(mode="after")
    def _check_size(self) -> Self:
        if len(self.values) != self.nx * self.ny:
            raise ValueError(f"values has {len(self.values)} cells, expected nx*ny = {self.nx * self.ny}")
        return self


class OptionEvaluation(Contract):
    """One option available to the ball carrier, scored by the analytics models."""

    option_id: str
    intent: ActionIntent
    resolved: ResolvedAction | None = None
    label: str
    """Short English label, e.g. "Through ball to Okafor" (logs, analyst tooling)."""
    label_key: str
    """UI string-table key, e.g. "option.through_ball", localized client-side with ``label_args``."""
    label_args: dict[str, str] = {}
    """e.g. {"player": "H09"}; player ids are rendered as display names by the UI."""
    p_success: Probability
    """Player-adjusted completion probability (pass) or xG (shot) or retention probability (carry)."""
    p_success_baseline: Probability | None = None
    """Same model for an average-attribute player (shows the player adjustment)."""
    value_if_success: float
    """xT at the destination, or xG of the resulting chance."""
    xg: Probability | None = None
    xt_gain: float | None = None
    decision_value: float
    """p_success * value_if_success minus the expected cost of failure."""
    viable: bool = True
    is_real: bool = False
    """The option the player actually took in the original match."""


class ModelOutputs(Contract):
    schema_version: str = SCHEMA_VERSION
    tick: Annotated[int, Field(ge=0)]
    t: Annotated[float, Field(ge=0)]
    team_in_possession: TeamSide | None
    ball_carrier: PlayerId | None
    ball_xy: Vec2
    xt_at_ball: float
    shot_xg_now: Probability | None = None
    """xG if the carrier shot from here now (placed technique), when in range."""
    pitch_control: PitchControlGrid | None = None
    options: list[OptionEvaluation] = []
    model_versions: dict[str, str] = {}
    """e.g. {"xg": "xg-gbm-2026-10-27", "xpass": "...", "pitch_control": "spearman2018"}."""
