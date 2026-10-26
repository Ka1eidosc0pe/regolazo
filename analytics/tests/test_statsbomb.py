from typing import Any

import numpy as np
import pytest

from analytics.data import statsbomb as sb
from analytics.paths import statsbomb_dir
from shared.model_api import PASS_HEIGHT, PASS_KIND, SHOT_BODY_PART, SHOT_SITUATION, SHOT_TECHNIQUE


def _shot(**over: Any) -> dict[str, Any]:
    shot: dict[str, Any] = {
        "type": {"name": "Open Play"},
        "outcome": {"name": "Goal"},
        "technique": {"name": "Normal"},
        "body_part": {"name": "Right Foot"},
        "end_location": [120.0, 38.0, 1.0],
        "statsbomb_xg": 0.2,
        "freeze_frame": [
            {"location": [119.0, 40.0], "teammate": False, "position": {"name": "Goalkeeper"}},
            {"location": [110.0, 40.0], "teammate": False, "position": {"name": "Center Back"}},
            {"location": [105.0, 40.0], "teammate": True, "position": {"name": "Center Forward"}},
        ],
    }
    shot.update(over.pop("shot", {}))
    ev: dict[str, Any] = {
        "id": "s1",
        "type": {"name": "Shot"},
        "period": 1,
        "minute": 10,
        "timestamp": "00:10:05.000",
        "possession": 7,
        "play_pattern": {"name": "Regular Play"},
        "team": {"id": 1},
        "location": [102.0, 40.0],
        "duration": 0.6,
        "shot": shot,
    }
    ev.update(over)
    return ev


def _start(possession: int = 7, ts: str = "00:10:00.000") -> dict[str, Any]:
    return {"type": {"name": "Pass"}, "period": 1, "timestamp": ts, "possession": possession}


def test_extract_shot_features() -> None:
    (row,) = sb.extract_shots([_start(), _shot()], match_id=5)
    assert row["x"] == pytest.approx((102 - 60) * 105 / 120)
    assert row["y"] == pytest.approx(0.0)
    assert row["gk_found"] == 1.0
    assert row["gk_x"] == pytest.approx(59 * 105 / 120)
    assert row["n_defenders_in_cone"] == 1.0  # the keeper and teammates are not counted
    assert row["nearest_defender_m"] == pytest.approx(8 * 105 / 120)
    assert row["body_part"] == SHOT_BODY_PART["foot"]
    assert row["technique"] == SHOT_TECHNIQUE["placed"]
    assert row["situation"] == SHOT_SITUATION["open_play"]
    assert row["goal"] == 1.0 and row["on_target"] == 1.0
    assert row["end_y"] == pytest.approx(2 * 7.32 / 8)
    assert row["end_z"] == pytest.approx(1.0 * 2.44 / 2.67)
    assert row["speed_proxy_mps"] > 20


@pytest.mark.parametrize(
    ("over", "expected"),
    [
        ({"shot": {"type": {"name": "Penalty"}}}, "penalty"),
        ({"shot": {"type": {"name": "Free Kick"}}}, "direct_free_kick"),
        ({"play_pattern": {"name": "From Counter"}}, "counter_attack"),
        ({"play_pattern": {"name": "From Corner"}}, "set_piece"),
        ({"play_pattern": {"name": "From Corner"}, "timestamp": "00:10:30.000"}, "open_play"),
    ],
)
def test_shot_situation(over: dict[str, Any], expected: str) -> None:
    (row,) = sb.extract_shots([_start(), _shot(**over)])
    assert row["situation"] == SHOT_SITUATION[expected]


def test_shot_technique_and_body_part() -> None:
    head = _shot(shot={"body_part": {"name": "Head"}})
    lob = _shot(shot={"technique": {"name": "Lob"}})
    dive = _shot(shot={"body_part": {"name": "Head"}, "technique": {"name": "Diving Header"}})
    rows = sb.extract_shots([_start(), head, lob, dive])
    assert [r["technique"] for r in rows] == [
        SHOT_TECHNIQUE["header"],
        SHOT_TECHNIQUE["chip"],
        SHOT_TECHNIQUE["diving_header"],
    ]
    assert rows[0]["body_part"] == SHOT_BODY_PART["head"]


def test_shootout_shots_are_dropped() -> None:
    assert sb.extract_shots([_start(), _shot(period=5)]) == []


def test_saved_shot_end_is_projected_to_goal_line_on_the_sb_lattice() -> None:
    start = sb.to_m([100.0, 40.0])
    # From (100, 40) through (116, 38): ratio 20 / 16 = 1.25, goal line crossed at y = 37.5; z stays the touch height.
    gy, gz, gx, ratio = sb.goal_plane_end(start, [116.0, 38.0, 1.0])
    assert ratio == pytest.approx(1.25)
    assert gx == pytest.approx(sb.to_m([116.0, 38.0])[0])
    assert gy == pytest.approx(2.5 * 7.32 / 8)
    assert gz == pytest.approx(1.0 * 2.44 / 2.67)
    gy2, _, _, _ = sb.goal_plane_end(start, [116.0, 38.02, 1.0])
    assert gy2 == pytest.approx(gy)  # snapped to the 0.1-yard lattice
    # A save far off the line (ratio 2) is extrapolated by at most MAX_PROJECTION_RATIO, and its z is never scaled.
    gy3, gz3, _, ratio3 = sb.goal_plane_end(start, [110.0, 38.0, 2.4])
    assert ratio3 == pytest.approx(2.0)
    assert gy3 == pytest.approx((40.0 - (40.0 - 2.0 * sb.MAX_PROJECTION_RATIO)) * 7.32 / 8)
    assert gz3 == pytest.approx(2.4 * 2.44 / 2.67)


def test_keeper_y_uses_goal_plane_scale_for_post_shot_features() -> None:
    (row,) = sb.extract_shots(
        [
            _start(),
            _shot(
                shot={
                    "freeze_frame": [
                        {"location": [119.0, 43.0], "teammate": False, "position": {"name": "Goalkeeper"}},
                    ]
                }
            ),
        ]
    )
    assert row["gk_y"] == pytest.approx(-3.0 * 68 / 80)
    assert row["gk_y_goal_plane"] == pytest.approx(-3.0 * 7.32 / 8)


def _pass(pid: str, loc: list[float], end: list[float], **q: Any) -> dict[str, Any]:
    return {
        "id": pid,
        "type": {"name": "Pass"},
        "period": 1,
        "location": loc,
        "pass": {"end_location": end, "height": {"name": q.pop("height", "Ground Pass")}, **q},
    }


def _mate(x: float, y: float) -> dict[str, Any]:
    return {"location": [x, y], "teammate": True}


def test_extract_passes_kinds_outcomes_and_geometry() -> None:
    frames = {
        "p1": [
            {"location": [70.0, 41.0], "teammate": False},
            {"location": [81.0, 40.0], "teammate": False},
            {"location": [55.0, 40.0], "teammate": True, "actor": True},
            _mate(79.0, 40.0),  # the receiver when the ball is struck (it arrives at 80, 40)
            _mate(60.0, 70.0),
        ],
        # Failed cross intercepted at (110, 40): the target is the striker at (112, 38), not the interception point.
        "p2": [{"location": [110.0, 40.0], "teammate": False}, _mate(112.0, 38.0), _mate(90.0, 20.0)],
        "p3": [_mate(50.0, 40.0)],
        "p4": [_mate(70.0, 40.0)],
        "p5": [_mate(70.0, 40.0)],
        "p7": [{"location": [70.0, 41.0], "teammate": False}],  # no visible teammate -> dropped
        "p8": [_mate(70.0, 80.0)],  # target on the touchline -> dropped
    }
    events = [
        _pass("p1", [60.0, 40.0], [80.0, 40.0]),
        _pass("p2", [100.0, 75.0], [110.0, 40.0], cross=True, height="High Pass", outcome={"name": "Incomplete"}),
        _pass("p3", [60.0, 40.0], [50.0, 40.0]),
        _pass("p4", [60.0, 40.0], [70.0, 40.0], height="High Pass"),
        _pass("p5", [60.0, 40.0], [70.0, 40.0], type={"name": "Throw-in"}),
        _pass("p6", [60.0, 40.0], [70.0, 40.0]),  # no 360 frame
        _pass("p7", [60.0, 40.0], [70.0, 40.0], outcome={"name": "Incomplete"}),
        _pass("p8", [60.0, 40.0], [70.0, 81.0], outcome={"name": "Out"}),
    ]
    rows = sb.extract_passes(events, frames)
    assert len(rows) == 4
    p1, p2, p3, p4 = rows
    assert p1["kind"] == PASS_KIND["standard"] and p1["completed"] == 1.0
    assert (p1["end_x"], p1["end_y"]) == pytest.approx(sb.to_m([79.0, 40.0]))
    assert p1["target_proxy_m"] == pytest.approx(105 / 120)
    assert p1["defenders_in_lane"] == 2.0  # (70, 41) on the lane, (81, 40) within 2 m of the target
    assert p1["nearest_defender_to_receiver_m"] == pytest.approx(2 * 105 / 120)
    assert p2["kind"] == PASS_KIND["cross"] and p2["completed"] == 0.0 and p2["height"] == PASS_HEIGHT["high"]
    assert (p2["end_x"], p2["end_y"]) == pytest.approx(sb.to_m([112.0, 38.0]))
    assert p3["kind"] == PASS_KIND["back_pass"]
    assert p4["kind"] == PASS_KIND["chipped"]


def test_team_match_stats_counts() -> None:
    info = sb.MatchInfo(1, 2, 27, "PL", "2015/2016", "male", 10, 20, 2, 1, False)
    events = [
        _shot(team={"id": 10}),
        _shot(team={"id": 10}, shot={"body_part": {"name": "Head"}}),
        {**_pass("a", [60.0, 40.0], [70.0, 40.0]), "team": {"id": 20}},
        {**_pass("b", [60.0, 40.0], [70.0, 40.0], outcome={"name": "Out"}), "team": {"id": 20}},
        {**_pass("c", [60.0, 40.0], [70.0, 40.0], type={"name": "Throw-in"}), "team": {"id": 20}},
        {"type": {"name": "Own Goal For"}, "period": 2, "team": {"id": 20}},
        _shot(team={"id": 20}, period=3),  # extra time is excluded
    ]
    home, away = sb.team_match_stats(events, info)
    assert (home["goals"], home["shots"], home["header_shots"]) == (2.0, 2.0, 1.0)
    assert (away["goals"], away["passes"], away["passes_completed"]) == (1.0, 2.0, 1.0)


def test_moves_for_xt() -> None:
    carry = {"type": {"name": "Carry"}, "period": 1, "location": [60.0, 40.0], "carry": {"end_location": [70.0, 40.0]}}
    pen = _shot(shot={"type": {"name": "Penalty"}})
    rows = sb.extract_moves([carry, _shot(), pen])
    assert [r["kind"] for r in rows] == [0.0, 1.0]
    assert rows[1]["success"] == 1.0


@pytest.mark.skipif(not (statsbomb_dir() / "competitions.json").exists(), reason="StatsBomb cache not downloaded")
def test_real_match_extracts_finite_features() -> None:
    info = next(m for m in sb.list_matches() if m.has_360)
    events = sb.load_events(info.match_id)
    shots = sb.rows_to_table(sb.extract_shots(events, info.match_id))
    passes = sb.rows_to_table(sb.extract_passes(events, sb.load_frames_360(info.match_id)))
    assert 5 <= len(shots["x"]) <= 60
    assert len(passes["x"]) > 300
    for t in (shots, passes):
        for k in ("x", "y"):
            assert np.isfinite(t[k]).all()
            assert (np.abs(t[k]) <= 53).all()
