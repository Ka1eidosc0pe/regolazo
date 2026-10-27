"""Milestone 3: fork from snapshots (in memory and from a written recording) and the RolloutEngine API."""

from functools import cache
from pathlib import Path

import numpy as np
import pytest

from shared.engine_api import RolloutEngine
from shared.recording import Recording
from shared.schemas.models import OptionEvaluation
from shared.schemas.rollout import RolloutRequest
from sim.engine import SimEngine, parse_ref
from sim.match import Match
from sim.record import simulate, write_match
from sim.squads import generate_meta


@cache
def _engine() -> tuple[SimEngine, list[str]]:
    meta = generate_meta(31, "enginetest")
    m = Match(meta, snapshots=True)
    m.start()
    m.run(until_t=150.0)
    eng = SimEngine(models=m.models, workers=1)
    eng.register(meta, m.snapshots)
    refs = [f"snap:enginetest:{t}" for t, _ in m.snapshots]
    return eng, refs


def _option(eng: SimEngine, ref: str, kind: str = "pass_player") -> OptionEvaluation:
    mo = eng.options(ref)
    cands = [o for o in mo.options if o.intent.kind == kind]
    assert cands, f"no {kind} option at {ref}"
    return max(cands, key=lambda o: o.p_success)


def test_engine_satisfies_protocol() -> None:
    eng, refs = _engine()
    assert isinstance(eng, RolloutEngine)
    assert len(refs) > 30


def test_options_are_scored_with_the_models() -> None:
    eng, refs = _engine()
    mo = eng.options(refs[len(refs) // 2])
    assert mo.ball_carrier is not None and mo.pitch_control is not None
    assert len(mo.pitch_control.values) == mo.pitch_control.nx * mo.pitch_control.ny
    assert len(mo.options) >= 5
    for o in mo.options:
        assert 0.0 <= o.p_success <= 1.0
        assert o.p_success_baseline is not None
    assert set(mo.model_versions) >= {"xg", "xpass", "xt"}


def test_preview_has_arc_ellipse_and_launch() -> None:
    eng, refs = _engine()
    ref = refs[len(refs) // 3]
    opt = _option(eng, ref)
    r = eng.preview(ref, opt.intent)
    assert r.action_type == "pass" and r.launch is not None and r.ellipse is not None
    assert len(r.preview_arc) >= 2 and r.pass_technique is not None and r.pass_type is not None
    assert r.available_pass_techniques and r.body_part in ("left_foot", "right_foot", "head", "hands")


def test_rollout_returns_frames_and_namespaced_events() -> None:
    eng, refs = _engine()
    ref = refs[len(refs) // 2]
    opt = _option(eng, ref)
    req = RolloutRequest(snapshot_ref=ref, intent=opt.intent, horizon_s=6.0, n=1, seed=11)
    a = eng.rollout(req)
    b = eng.rollout(req)
    assert len(a.frames) == 150
    assert a.events and all(e.rollout_id == a.id and e.id.startswith(f"ev_r_{a.id}_") for e in a.events)
    on_ball = [e for e in a.events if e.type in ("pass", "shot", "carry", "take_on", "clearance")]
    assert on_ball[0].type == "pass" and on_ball[0].player == opt.intent.player
    # same seed, same reality
    assert a.outcome.model_dump(exclude={"ended_by_event_id"}) == b.outcome.model_dump(exclude={"ended_by_event_id"})
    assert [f.ball.pos for f in a.frames] == [f.ball.pos for f in b.frames]
    _, tick = parse_ref(ref)
    assert a.frames[0].tick >= tick
    assert a.evaluation.p_success == pytest.approx(opt.p_success, abs=1e-9)


def test_distribution_counts_and_intervals() -> None:
    eng, refs = _engine()
    ref = refs[len(refs) // 2]
    opt = _option(eng, ref)
    d = eng.distribution(RolloutRequest(snapshot_ref=ref, intent=opt.intent, horizon_s=5.0, n=120, seed=5))
    assert sum(c.count for c in d.outcome_counts) == 120
    assert d.p_primary_success_ci95.lo <= d.p_primary_success <= d.p_primary_success_ci95.hi
    assert len(d.samples) == 120
    assert d.model_p_success == pytest.approx(opt.p_success, abs=1e-9)


def test_fork_from_a_written_recording(tmp_path: Path) -> None:
    meta = generate_meta(32, "recfork")
    res = simulate(meta, until_t=60.0)
    write_match(res, tmp_path / "recfork")
    rec = Recording.open(tmp_path / "recfork")
    keys = np.nonzero(rec.arrays["keyframe"])[0]
    assert len(keys) > 10
    eng = SimEngine(data_dir=tmp_path, workers=1)
    for i in keys[:: max(1, len(keys) // 5)]:
        state = rec.state_at(int(i))
        assert state.snapshot_ref is not None
        mo = eng.options(state.snapshot_ref)
        assert mo.ball_carrier is not None and mo.tick >= state.tick
        assert mo.ball_carrier == state.possession.player or state.possession.player is None
