"""Frame helpers. Contracts store world coordinates; models and taxonomy thresholds use the attacking frame."""

from shared.schemas.common import Period, TeamSide, Vec2, Vec3


def attacks_positive_x(team: TeamSide, period: Period, home_attacks_positive_x_first_half: bool = True) -> bool:
    """Whether ``team`` attacks towards +x in ``period``."""
    home_positive = home_attacks_positive_x_first_half == (period == 1)
    return home_positive if team == "home" else not home_positive


def to_attacking_frame[V: (Vec2, Vec3)](
    v: V, team: TeamSide, period: Period, home_attacks_positive_x_first_half: bool = True
) -> V:
    """Rotate a world-frame point so ``team`` attacks +x (y then points to the team's left). Inverse of itself."""
    if attacks_positive_x(team, period, home_attacks_positive_x_first_half):
        return v
    if len(v) == 2:
        return (-v[0], -v[1])
    return (-v[0], -v[1], v[2])


def statsbomb_to_attacking_frame(x_sb: float, y_sb: float) -> Vec2:
    """StatsBomb 120x80 (origin top-left, acting team attacking +x) -> metres, centre origin, +y = left."""
    return ((x_sb - 60.0) * 105.0 / 120.0, (40.0 - y_sb) * 68.0 / 80.0)


def statsbomb_goal_plane(y_sb: float, z_sb: float) -> tuple[float, float]:
    """StatsBomb shot end location on the goal mouth -> (y, z) metres, preserving the true goal frame."""
    return ((40.0 - y_sb) * 7.32 / 8.0, z_sb * 2.44 / 2.67)
