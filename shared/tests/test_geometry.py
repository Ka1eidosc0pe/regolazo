from shared.geometry import (
    attacks_positive_x,
    statsbomb_goal_plane,
    statsbomb_to_attacking_frame,
    to_attacking_frame,
)


def test_direction_swaps_at_half_time() -> None:
    assert attacks_positive_x("home", 1)
    assert not attacks_positive_x("away", 1)
    assert not attacks_positive_x("home", 2)
    assert attacks_positive_x("away", 2)
    assert not attacks_positive_x("home", 1, home_attacks_positive_x_first_half=False)


def test_attacking_frame_is_rotation_and_involution() -> None:
    p = (30.0, -10.0, 1.2)
    assert to_attacking_frame(p, "home", 1) == p
    flipped = to_attacking_frame(p, "away", 1)
    assert flipped == (-30.0, 10.0, 1.2)
    assert to_attacking_frame(flipped, "away", 1) == p
    assert to_attacking_frame((1.0, 2.0), "home", 2) == (-1.0, -2.0)


def test_statsbomb_conversion_landmarks() -> None:
    assert statsbomb_to_attacking_frame(60, 40) == (0.0, 0.0)
    assert statsbomb_to_attacking_frame(120, 40) == (52.5, 0.0)
    x, y = statsbomb_to_attacking_frame(108, 0)  # SB penalty spot is x=108; y=0 is the attacker's left
    assert abs(x - 42.0) < 1e-9 and y == 34.0
    gy, gz = statsbomb_goal_plane(36, 2.67)  # SB left post, crossbar height
    assert abs(gy - 3.66) < 1e-9 and abs(gz - 2.44) < 1e-9


def test_angle_to_attacking_frame() -> None:
    import math

    from shared.geometry import angle_to_attacking_frame

    assert angle_to_attacking_frame(0.3, "home", 1) == 0.3
    # away in period 1 attacks -x: a world pass along -x is straight forward for them
    assert abs(angle_to_attacking_frame(math.pi, "away", 1)) < 1e-12
    assert abs(angle_to_attacking_frame(-math.pi / 2, "away", 1) - math.pi / 2) < 1e-12
