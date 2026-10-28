"""Milestone 2 gate: a simulated match obeys the rendering contract and basic football physics.

One module-level simulation (8 simulated minutes) is shared by the tests.
"""

from functools import cache
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from shared.recording import ANIMS
from shared.schemas.events import MatchEvent
from sim.constants import DT
from sim.eventlog import to_match_events
from sim.match import Match
from sim.record import FrameRecorder
from sim.squads import generate_meta

SECONDS = 480.0


@cache
def _sim(seed: int = 11, seconds: float = SECONDS) -> tuple[Match, dict[str, npt.NDArray[Any]], list[MatchEvent]]:
    m = Match(generate_meta(seed), batch=1, snapshots=True)
    rec = FrameRecorder(m)
    m.on_frame = rec
    m.start()
    m.run(until_t=seconds)
    return m, rec.result(), to_match_events(m.logs[0])


@pytest.fixture(scope="module")
def sim() -> tuple[Match, dict[str, npt.NDArray[Any]], list[MatchEvent]]:
    return _sim()


def _carrier_frames(a: dict[str, npt.NDArray[Any]]) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    k = np.nonzero(a["poss_player"] >= 0)[0]
    return k, a["poss_player"][k].astype(int)


def test_carrier_faces_and_dribbles_where_he_goes(sim: Any) -> None:
    _, a, _ = sim
    k, p = _carrier_frames(a)
    vel = a["player_vel"][k, p].astype(float)
    speed = np.linalg.norm(vel, axis=1)
    moving = speed >= 1.5
    assert moving.sum() > 500
    k, p, vel, speed = k[moving], p[moving], vel[moving], speed[moving]
    bad = {ANIMS.index(n) for n in ("jog_back", "strafe_left", "strafe_right")}
    anim = a["player_anim"][k, p]
    assert not np.isin(anim, list(bad)).any()
    heading = np.arctan2(vel[:, 1], vel[:, 0])
    facing = a["player_facing"][k, p].astype(float)
    diff = np.abs(np.arctan2(np.sin(facing - heading), np.cos(facing - heading)))
    assert diff.max() <= np.pi / 2
    ahead = a["ball_pos"][k, :2].astype(float) - a["player_pos"][k, p].astype(float)
    assert (np.sum(ahead * vel, axis=1) > 0).all()


def test_possession_needs_the_ball_within_reach(sim: Any) -> None:
    _, a, _ = sim
    pp = a["poss_player"].astype(int)
    gains = np.nonzero((pp[1:] >= 0) & (pp[1:] != pp[:-1]))[0] + 1
    assert len(gains) > 50
    who = pp[gains]
    d = np.linalg.norm(a["ball_pos"][gains - 1, :2] - a["player_pos"][gains - 1, who], axis=1)
    assert d.max() <= 1.05, f"gain from {d.max():.2f} m"
    # above 1.6 m only a keeper can take the ball (hands); outfield players head it on instead
    gk = {k for k, p in enumerate(_players_by_rec(sim[0])) if p.position == "GK"}
    z = a["ball_pos"][gains - 1, 2]
    outfield = np.array([int(w) not in gk for w in who])
    assert (z[outfield] <= 1.65).all()
    assert (z <= 3.4).all()


def test_ball_never_teleports(sim: Any) -> None:
    _, a, _ = sim
    pos = a["ball_pos"].astype(float)
    vel = a["ball_vel"].astype(float)
    step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    speed = np.maximum(np.linalg.norm(vel[1:], axis=1), np.linalg.norm(vel[:-1], axis=1))
    excess = step - speed * DT
    assert excess.max() <= 0.3, f"ball jumped {excess.max():.2f} m beyond its speed at frame {int(excess.argmax()) + 1}"


def test_players_respect_kinematic_limits(sim: Any) -> None:
    m, a, _ = sim
    on = a["player_on_pitch"][0]
    idx = np.nonzero(on)[0]
    vel = a["player_vel"][:, idx].astype(float)
    speed = np.linalg.norm(vel, axis=2)
    max_speed = np.array([p.physical.max_speed_mps for p in _players_by_rec(m)])[idx]
    dives = np.isin(a["player_anim"][:, idx], [ANIMS.index("gk_dive_left"), ANIMS.index("gk_dive_right")])
    over = (speed > max_speed[None, :] + 1e-3) & ~dives
    assert not over.any()
    acc = np.linalg.norm(np.diff(vel, axis=0), axis=2) / DT
    max_acc = np.array([max(p.physical.max_accel_mps2, p.physical.max_decel_mps2) for p in _players_by_rec(m)])[idx]
    big = (acc > max_acc[None, :] + 1e-3) & ~dives[1:] & ~dives[:-1]
    # half time re-positions players: ignore status changes
    status_change = np.diff(a["status"]) != 0
    big &= ~status_change[:, None]
    assert not big.any()


def _players_by_rec(m: Match) -> list[Any]:
    return [*m.meta.home.players, *m.meta.away.players]


def test_events_validate_and_cover_the_basics(sim: Any) -> None:
    _, _, events = sim
    types = {e.type for e in events}
    for t in ("pass", "ball_receipt", "shot", "duel", "restart", "period_start"):
        assert t in types
    passes = [e for e in events if e.type == "pass" and e.t < SECONDS - 6.0]  # the last one may still be in flight
    assert all(e.qualifiers is not None and e.end_location is not None for e in passes)
    # every completed pass has a receipt
    complete = [e for e in passes if getattr(e.qualifiers, "outcome", "") == "complete"]
    assert all(getattr(e.qualifiers, "receipt_id", None) for e in complete)


def test_formation_stays_compact(sim: Any) -> None:
    _, a, _ = sim
    live = a["ball_in_play"].astype(bool)
    pos = a["player_pos"][live][:, a["player_on_pitch"][0]].astype(float)
    for team in (slice(1, 11), slice(12, 22)):
        length = np.ptp(pos[:, team, 0], axis=1)
        width = np.ptp(pos[:, team, 1], axis=1)
        assert 15.0 < np.median(length) < 50.0
        assert 25.0 < np.median(width) < 62.0


def test_determinism_same_seed_same_match() -> None:
    def run(seed: int) -> npt.NDArray[Any]:
        m = Match(generate_meta(seed), batch=1)
        rec = FrameRecorder(m, capacity=2000)
        m.on_frame = rec
        m.start()
        m.run(until_t=40.0)
        out: npt.NDArray[Any] = rec.result()["player_pos"]
        return out

    np.testing.assert_array_equal(run(3), run(3))
    assert not np.array_equal(run(3), run(4))


def test_strike_clips_make_contact_on_the_kick_tick(sim: Any) -> None:
    """Renderer convention: strike clips start 0.4 s before ball contact (anim_t = 0.4 on the kick tick)."""
    from sim.match import STRIKE_ANIMS

    m, a, events = sim
    strike = {ANIMS.index(n) for n in STRIKE_ANIMS}
    rec_of = {p.id: k for k, p in enumerate(_players_by_rec(m))}
    frame_of = {int(t): k for k, t in enumerate(a["tick"])}
    kicks = [e for e in events if e.type in ("pass", "shot", "clearance") and int(e.tick or -1) in frame_of]
    assert len(kicks) > 50
    for e in kicks:
        k, r = frame_of[int(e.tick or 0)], rec_of[e.player or ""]
        assert int(a["player_anim"][k, r]) in strike, (e.type, ANIMS[int(a["player_anim"][k, r])])
        assert abs(float(a["player_anim_t"][k, r]) - 0.4) <= DT + 1e-4
        if k >= 10:  # the clip started 0.4 s earlier and ran continuously
            assert int(a["player_anim"][k - 10, r]) == int(a["player_anim"][k, r])
            assert float(a["player_anim_t"][k - 10, r]) <= DT + 1e-4
