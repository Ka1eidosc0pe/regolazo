"""Stats tools: deterministic, hand-checkable on the tiny synthetic recording (agents/tests/conftest.py)."""

import numpy as np

from agents.stats import MatchData, numbers_in


def test_numbers_in_handles_ordinals_scores_and_decimal_commas() -> None:
    assert numbers_in("74th minute, 1-1, xG 0.22 and 0,31, 90+2") == {74.0, 1.0, 0.22, 0.31, 90.0, 2.0}
    assert numbers_in("no digits here, one two three") == set()


def test_key_moments_classified_and_ordered(match: MatchData) -> None:
    ms = match.key_moments()
    kinds = {m.event_id: m.kind for m in ms if m.event_id}
    assert kinds == {"ev_100": "goal", "ev_101": "big_chance", "ev_102": "save"}
    assert [m.kind for m in ms][-1] == "full_time" and "half_time" in [m.kind for m in ms]
    assert [m.t for m in ms] == sorted(m.t for m in ms)


def test_goal_fact_sheet_has_names_and_exact_numbers(match: MatchData) -> None:
    goal = next(m for m in match.key_moments() if m.kind == "goal")
    f = goal.facts.facts
    assert f["shooter"] == match.name("H09")
    assert f["team"] == match.meta.home.name
    assert f["xg"] == "0.31" and f["post_shot_xg"] == "0.40"
    assert f["shot_speed_kmh"] == 90  # 25 m/s
    assert f["shot_distance_m"] == 18
    assert f["score_now"] == f"{match.meta.home.name} 1-0 {match.meta.away.name}" and f["minute"] == 1
    assert goal.t == 5.0 + 1.5
    assert not any(v in match.players for v in f.values() if isinstance(v, str)), "ids must not leak into facts"


def test_big_chance_threshold_and_running_totals(match: MatchData) -> None:
    chance = next(m for m in match.key_moments() if m.event_id == "ev_101")
    assert chance.kind == "big_chance"
    assert chance.facts.facts["team_shots_so_far"] == 1 and chance.facts.facts["team_xg_so_far"] == "0.22"
    home_second = next(m for m in match.key_moments() if m.event_id == "ev_102")
    assert home_second.facts.facts["team_shots_so_far"] == 2 and home_second.facts.facts["team_xg_so_far"] == "0.39"


def test_match_stats_totals(match: MatchData) -> None:
    s = match.match_stats(18.0)
    assert s["score"] == "1-0"
    assert (s["home_shots"], s["away_shots"]) == (2, 1)
    assert (s["home_shots_on_target"], s["away_shots_on_target"]) == (2, 0)
    assert s["home_xg"] == "0.39" and s["away_xg"] == "0.22"
    assert int(s["home_possession_pct"]) + int(s["away_possession_pct"]) == 100
    assert int(s["top_speed_kmh"]) > 0 and float(s["distance_leader_km"]) > 0


def test_physical_stats_match_frames(match: MatchData) -> None:
    top, dist = match.player_physical("H09", 18.0)
    i = match.rec.index_at(18.0)
    c = match.col["H09"]
    vel = match.rec.arrays["player_vel"][: i + 1, c]
    assert abs(top - float(np.linalg.norm(vel, axis=1).max()) * 3.6) < 1e-3
    assert dist > 0
    assert match.player_physical("H09", 9.0)[0] <= top


def test_evidence_only_for_numeric_facts(match: MatchData) -> None:
    goal = next(m for m in match.key_moments() if m.kind == "goal")
    ev = goal.facts.evidence()
    keys = {e.field for e in ev}
    assert {"xg", "minute", "shot_speed_kmh"} <= keys and "shooter" not in keys
    xg = next(e for e in ev if e.field == "xg")
    assert xg.kind == "model_output" and xg.value == "0.31"
