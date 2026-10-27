"""Sim <-> model consistency (PLAN.md section 3): realized rates over many realities match the displayed,
player-adjusted probabilities; for average-attribute players those equal the analytics models' own numbers
(xG; xPass with the pitch-control interception term); elite / poor players deviate in the right direction.
"""

from functools import cache

import pytest

from shared.schemas.actions import ActionIntent
from shared.schemas.models import OptionEvaluation
from shared.schemas.rollout import RolloutDistribution, RolloutRequest
from sim.engine import wilson
from sim.match import Match
from sim.tests.scenes import away_ids, base_match, engine_for, home_ids, place

ST, MID = 9, 6
SHOOTING = ("finishing", "shot_power", "long_shots", "composure", "technique", "curve", "volleys")
PASSING = ("short_passing", "long_passing", "vision", "technique", "composure", "curve")
GK = ("gk_reflexes", "gk_diving", "gk_positioning", "gk_handling")
N = 500


def _attrs(names: tuple[str, ...], level: int) -> dict[str, int]:
    return dict.fromkeys(names, level)


def _run(m: Match, intent: ActionIntent, horizon: float) -> tuple[RolloutDistribution, OptionEvaluation]:
    eng, ref = engine_for(m)
    ev = eng.rollout(RolloutRequest(snapshot_ref=ref, intent=intent, horizon_s=horizon, n=1, seed=1)).evaluation
    d = eng.distribution(RolloutRequest(snapshot_ref=ref, intent=intent, horizon_s=horizon, n=N, seed=2026))
    return d, ev


@cache
def _shot(level: int) -> tuple[RolloutDistribution, OptionEvaluation]:
    m0, _ = base_match()
    h, a = home_ids(m0), away_ids(m0)
    _, meta = base_match(attr_overrides={h[ST]: _attrs(SHOOTING, level)})
    away = meta.away
    players = [  # an average keeper
        p.model_copy(update={"attributes": p.attributes.model_copy(update=_attrs(GK, 70))}) if p.position == "GK" else p
        for p in away.players
    ]
    meta = meta.model_copy(update={"away": away.model_copy(update={"players": players})})
    m = Match(meta, batch=1)
    m.start()
    m.step()
    place(m, h[ST], (36.0, 5.0), opps={a[0]: (51.5, 0.5), a[4]: (40.0, 9.0)})
    intent = ActionIntent(kind="shot", player=h[ST], target=(52.5, -2.2, 0.5), power=0.6, source="fan")
    return _run(m, intent, 3.0)


@cache
def _pass(level: int) -> tuple[RolloutDistribution, OptionEvaluation]:
    m0, _ = base_match()
    h, a = home_ids(m0), away_ids(m0)
    m, _ = base_match(attr_overrides={h[MID]: _attrs(PASSING, level)})
    place(m, h[MID], (0.0, 0.0), mates={h[ST]: (22.0, 6.0)}, opps={a[5]: (11.0, 5.0), a[2]: (24.0, 9.0)})
    intent = ActionIntent(kind="pass_player", player=h[MID], target_player=h[ST], power=0.5, source="fan")
    return _run(m, intent, 4.0)


def _within_ci(d: RolloutDistribution, p: float) -> bool:
    k = round(d.p_primary_success * d.n)
    lo, hi = wilson(k, d.n, z=2.58)
    return lo <= p <= hi


def test_average_shooter_scores_at_the_model_xg() -> None:
    d, ev = _shot(70)
    assert ev.xg is not None and ev.p_success_baseline is not None
    assert ev.p_success == pytest.approx(ev.p_success_baseline, abs=0.015)  # average player: no adjustment
    assert d.model_p_success == pytest.approx(ev.p_success)
    assert _within_ci(d, d.model_p_success), (d.p_primary_success, d.model_p_success)
    assert d.p_goal == pytest.approx(d.p_primary_success)


def test_finishers_deviate_in_the_expected_direction() -> None:
    (poor, pe), (avg, ae), (elite, ee) = _shot(40), _shot(70), _shot(95)
    assert pe.p_success < ae.p_success < ee.p_success
    assert pe.p_success < (pe.p_success_baseline or 0) and ee.p_success > (ee.p_success_baseline or 1)
    for d in (poor, elite):
        assert _within_ci(d, d.model_p_success), (d.p_primary_success, d.model_p_success)
    assert poor.p_primary_success < elite.p_primary_success
    assert ee.p_success < 2.0 * ae.p_success  # plausible: a great finisher beats xG, he does not double it


def test_average_passer_completes_at_the_displayed_xpass() -> None:
    d, ev = _pass(70)
    assert ev.p_success == pytest.approx(ev.p_success_baseline, abs=0.02)
    assert _within_ci(d, d.model_p_success), (d.p_primary_success, d.model_p_success)


def test_passers_deviate_in_the_expected_direction() -> None:
    (poor, pe), (_, ae), (elite, ee) = _pass(35), _pass(70), _pass(95)
    assert pe.p_success < ae.p_success < ee.p_success
    for d in (poor, elite):
        assert _within_ci(d, d.model_p_success), (d.p_primary_success, d.model_p_success)
    assert poor.p_primary_success < elite.p_primary_success
