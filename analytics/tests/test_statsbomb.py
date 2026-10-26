import math
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


def test_saved_shot_end_is_projected_to_goal_line() -> None:
    gy, gz, gx = sb.goal_plane_end((10.0, 0.0), [114.0, 38.0, 1.0])
    ex, ey = sb.to_m([114.0, 38.0])
    ratio = (52.5 - 10.0) / (ex - 10.0)
    assert gx == pytest.approx(ex)
    assert gy == pytest.approx(ey * ratio)
    assert gz == pytest.approx(1.0 * 2.44 / 2.67 * ratio)


def _pass(pid: str, loc: list[float], end: list[float], **q: Any) -> dict[str, Any]:
    return {
        "id": pid,
        "type": {"name": "Pass"},
        "period": 1,
        "location": loc,
        "pass": {"end_location": end, "height": {"name": q.pop("height", "Ground Pass")}, **q},
    }


def test_extract_passes_kinds_outcomes_and_geometry() -> None:
    frames = {
        "p1": [
            {"location": [70.0, 41.0], "teammate": False},
            {"location": [81.0, 40.0], "teammate": False},
            {"location": [55.0, 40.0], "teammate": True, "actor": True},
        ],
        "p2": [{"location": [10.0, 10.0], "teammate": False}],
        "p3": [],
        "p4": [],
        "p5": [],
    }
    events = [
        _pass("p1", [60.0, 40.0], [80.0, 40.0]),
        _pass("p2", [100.0, 75.0], [110.0, 40.0], cross=True, height="High Pass", outcome={"name": "Incomplete"}),
        _pass("p3", [60.0, 40.0], [50.0, 40.0]),
        _pass("p4", [60.0, 40.0], [70.0, 40.0], height="High Pass"),
        _pass("p5", [60.0, 40.0], [70.0, 40.0], type={"name": "Throw-in"}),
        _pass("p6", [60.0, 40.0], [70.0, 40.0]),  # no 360 frame
    ]
    rows = sb.extract_passes(events, frames)
    assert len(rows) == 4
    p1, p2, p3, p4 = rows
    assert p1["kind"] == PASS_KIND["standard"] and p1["completed"] == 1.0
    assert p1["defenders_in_lane"] == 2.0
    assert p1["nearest_defender_to_receiver_m"] == pytest.approx(math.hypot(1.0, 0.0) * 105 / 120)
    assert p2["kind"] == PASS_KIND["cross"] and p2["completed"] == 0.0 and p2["height"] == PASS_HEIGHT["high"]
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
