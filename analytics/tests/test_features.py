import numpy as np
import pytest

from analytics.features import (
    NO_DEFENDER_M,
    count_in_lane,
    count_in_shot_cone,
    gk_geometry,
    nearest_distance,
    pad_players,
)


def test_pad_players_pads_with_nan() -> None:
    arr = pad_players([[(1.0, 2.0)], [], [(3.0, 4.0), (5.0, 6.0)]])
    assert arr.shape == (3, 2, 2)
    assert np.isnan(arr[0, 1]).all() and np.isnan(arr[1]).all()
    assert arr[2, 1].tolist() == [5.0, 6.0]


def test_cone_counts_only_points_inside_triangle() -> None:
    x, y = np.array([36.0]), np.array([0.0])
    opp = pad_players(
        [
            [
                (45.0, 0.0),  # in the cone
                (50.0, 3.0),  # in the cone near the post
                (45.0, 5.0),  # wide of the cone
                (30.0, 0.0),  # behind the shooter
                (53.0, 0.0),  # behind the goal line
            ]
        ]
    )
    assert count_in_shot_cone(x, y, opp).tolist() == [2]


def test_cone_ignores_padding_and_works_for_wide_angles() -> None:
    x, y = np.array([50.0, 40.0]), np.array([-10.0, 0.0])
    opp = pad_players([[(51.0, -6.0)], [(41.0, 0.0), (46.0, 0.0)]])
    assert count_in_shot_cone(x, y, opp).tolist() == [1, 2]


def test_nearest_distance_caps_when_no_opponent() -> None:
    d = nearest_distance(np.array([0.0, 0.0]), np.array([0.0, 0.0]), pad_players([[(3.0, 4.0)], []]))
    assert d.tolist() == [5.0, NO_DEFENDER_M]


def test_lane_counts_players_near_segment_ahead_of_passer() -> None:
    x, y, ex, ey = (np.array([v]) for v in (0.0, 0.0, 20.0, 0.0))
    opp = pad_players([[(10.0, 1.5), (10.0, 2.5), (-1.0, 0.0), (21.0, 0.5), (5.0, -1.9)]])
    # (10, 1.5) and (5, -1.9) are within 2 m of the segment; (21, 0.5) is within 2 m of the end point.
    assert count_in_lane(x, y, ex, ey, opp).tolist() == [3]


def test_gk_geometry() -> None:
    off, to_shooter, lateral = gk_geometry(np.array([40.5]), np.array([0.0]), np.array([51.5]), np.array([2.0]))
    assert off[0] == pytest.approx(1.0)
    assert to_shooter[0] == pytest.approx(np.hypot(11.0, 2.0))
    assert lateral[0] == pytest.approx(2.0)
