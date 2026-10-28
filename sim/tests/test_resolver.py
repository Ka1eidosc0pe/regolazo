"""Resolver picks the expected technique in hand-built situations (PLAN.md Step 3 technique test, part 1)."""

import numpy as np
import pytest

from shared.schemas.actions import ActionIntent
from sim.tests.scenes import away_ids, base_match, engine_for, home_ids, place

MID, ST, WING = 6, 9, 10  # home formation slots: a central midfielder, the striker, a wide forward


def _setup(**kw: object):  # type: ignore[no-untyped-def]
    m, _ = base_match()
    return m


def _preview(m, carrier: str, intent: ActionIntent):  # type: ignore[no-untyped-def]
    eng, ref = engine_for(m)
    return eng.preview(ref, intent)


def test_wide_to_box_is_a_cross() -> None:
    m = _setup()
    h = home_ids(m)
    place(m, h[WING], (38.0, -28.0), mates={h[ST]: (45.0, 2.0)})
    r = _preview(m, h[WING], ActionIntent(kind="pass_player", player=h[WING], target_player=h[ST], source="fan"))
    assert r.pass_type == "cross"
    assert r.pass_height in ("low", "high")
    assert r.preview_arc and max(p[2] for p in r.preview_arc) > 1.0


def test_byline_ball_back_is_a_cutback() -> None:
    m = _setup()
    h = home_ids(m)
    place(m, h[WING], (49.0, -14.0), mates={h[ST]: (40.0, -3.0)})
    r = _preview(m, h[WING], ActionIntent(kind="pass_player", player=h[WING], target_player=h[ST], source="fan"))
    assert r.pass_type == "cutback"
    assert r.pass_height == "ground"


def test_blocked_lane_goes_over_the_top() -> None:
    m = _setup()
    h, a = home_ids(m), away_ids(m)
    place(m, h[MID], (0.0, 0.0), mates={h[ST]: (24.0, 0.5)}, opps={a[5]: (12.0, 0.0)})
    blocked = _preview(m, h[MID], ActionIntent(kind="pass_player", player=h[MID], target_player=h[ST], source="fan"))
    assert blocked.pass_technique in ("lofted", "chipped")
    place(m, h[MID], (0.0, 0.0), mates={h[ST]: (24.0, 0.5)}, opps={a[5]: (12.0, 9.0)})
    open_ = _preview(m, h[MID], ActionIntent(kind="pass_player", player=h[MID], target_player=h[ST], source="fan"))
    assert open_.pass_technique in ("standard", "driven")


@pytest.mark.parametrize(("z", "expected"), [(1.75, "header"), (0.6, "volley"), (0.11, None)])
def test_airborne_ball_is_headed_or_volleyed(z: float, expected: str | None) -> None:
    m = _setup()
    h = home_ids(m)
    place(m, h[ST], (42.0, 1.0), ball_z=z)
    r = _preview(m, h[ST], ActionIntent(kind="shot", player=h[ST], target=(52.5, -2.0, 0.6), source="fan"))
    if expected is None:
        assert r.shot_technique not in ("header", "volley", "half_volley")
        assert r.body_part in ("left_foot", "right_foot")
    else:
        assert r.shot_technique == expected
        if expected == "header":
            assert r.body_part == "head"


def _with_attrs(level: int):  # type: ignore[no-untyped-def]
    m0, _ = base_match()
    h = home_ids(m0)
    upd = {a: level for a in ("curve", "technique", "vision", "short_passing", "long_passing", "finishing")}
    m, _ = base_match(attr_overrides={h[MID]: upd})
    return m, h


def test_curl_is_capped_by_curve_technique_and_vision() -> None:
    results = {}
    for level in (30, 95):
        m, h = _with_attrs(level)
        place(m, h[MID], (5.0, 10.0), mates={h[ST]: (30.0, -5.0)})
        intent = ActionIntent(kind="pass_space", player=h[MID], target=(32.0, -6.0, 0.0), curve=0.9, source="fan")
        results[level] = _preview(m, h[MID], intent)
    poor, elite = results[30], results[95]
    assert poor.max_curve < 0.15 < 0.8 < elite.max_curve
    assert abs(poor.curve_applied) <= poor.max_curve + 1e-9
    assert abs(elite.curve_applied) > 0.6
    assert "curled" in elite.available_pass_techniques and "curled" not in poor.available_pass_techniques
    assert "outside_of_boot" in elite.available_pass_techniques
    assert "outside_of_boot" not in poor.available_pass_techniques
    # the poor passer's attempt is less accurate: a bigger execution ellipse
    assert poor.ellipse is not None and elite.ellipse is not None
    assert poor.ellipse.semi_major_m > elite.ellipse.semi_major_m
    assert poor.execution_quality < elite.execution_quality


def test_override_rejected_when_the_player_cannot_do_it() -> None:
    m, h = _with_attrs(30)
    place(m, h[MID], (0.0, 0.0), mates={h[ST]: (18.0, 3.0)})
    intent = ActionIntent(
        kind="pass_player", player=h[MID], target_player=h[ST], pass_technique_override="outside_of_boot", source="fan"
    )
    r = _preview(m, h[MID], intent)
    assert r.override_rejected
    m2, h2 = _with_attrs(95)
    place(m2, h2[MID], (0.0, 0.0), mates={h2[ST]: (18.0, 3.0)})
    upd = {"player": h2[MID], "target_player": h2[ST], "pass_technique_override": "chipped"}
    intent = intent.model_copy(update=upd)
    r2 = _preview(m2, h2[MID], intent)
    assert not r2.override_rejected and r2.pass_technique == "chipped"


def test_chip_when_the_keeper_is_off_his_line() -> None:
    m, h = _with_attrs(95)
    a = away_ids(m)
    place(m, h[MID], (26.0, 3.0), opps={a[0]: (40.0, 1.0)})
    r = _preview(m, h[MID], ActionIntent(kind="shot", player=h[MID], target=(52.5, 1.0, 1.6), source="fan"))
    assert r.shot_technique == "chip"
    assert r.ellipse is not None and r.ellipse.plane == "goal_mouth"
    assert max(p[2] for p in r.preview_arc) > 3.0


def test_preview_is_deterministic_and_lands_near_the_target() -> None:
    m = _setup()
    h = home_ids(m)
    place(m, h[MID], (0.0, 0.0), mates={h[ST]: (20.0, 6.0)})
    i = ActionIntent(kind="pass_space", player=h[MID], target=(20.0, 6.0, 0.0), source="fan")
    a, b = _preview(m, h[MID], i), _preview(m, h[MID], i)
    assert a == b
    end = np.array(a.preview_arc[-1][:2])
    s = 1.0 if m.sign[0, 0] > 0 else -1.0
    assert np.linalg.norm(end - np.array([20.0 * s, 6.0 * s])) < 2.5
