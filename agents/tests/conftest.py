"""A tiny synthetic recording with known shots, so the stats tools have hand-checkable expected values."""

from pathlib import Path

import pytest

from agents.stats import MatchData
from server.fixture import fixture_events, fixture_moments, fixture_states
from shared.recording import player_order, states_to_arrays, write_recording
from shared.schemas.events import MatchEvent, ShotQ
from shared.schemas.fixtures import event_examples, match_meta


def _shot(base: MatchEvent, eid: str, t: float, team: str, player: str, outcome: str, xg: float) -> MatchEvent:
    q = base.qualifiers
    assert isinstance(q, ShotQ)
    on_target = outcome in ("goal", "saved")
    q2 = q.model_copy(
        update={
            "outcome": outcome,
            "xg": xg,
            "on_target": on_target,
            "psxg": 0.4 if on_target else None,
            "ball_speed_mps": 25.0,
            "distance_to_goal_m": 17.6,
            "key_pass_id": None,
            "gk_event_id": None,
            "keeper": "A01" if team == "home" else "H01",
        }
    )
    return base.model_copy(
        update={
            "id": eid,
            "index": int(t * 10),
            "t": t,
            "tick": round(t * 25),
            "team": team,
            "player": player,
            "qualifiers": q2,
        }
    )


@pytest.fixture(scope="session")
def match_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """18 s clip: home goal (xG 0.31) at 5 s, away big chance (0.22, blocked) at 9 s, home saved (0.08) at 12 s."""
    d = tmp_path_factory.mktemp("rec") / "tiny-0001"
    meta = match_meta()
    states = fixture_states(450)
    arrays = states_to_arrays(states, player_order(meta))
    shot0 = next(e for e in event_examples() if e.type == "shot")
    base = [e for e in fixture_events(18.0) if e.type == "period_start"][:1]
    events = [
        *base,
        _shot(shot0, "ev_100", 5.0, "home", "H09", "goal", 0.31),
        _shot(shot0, "ev_101", 9.0, "away", "A09", "blocked", 0.22),
        _shot(shot0, "ev_102", 12.0, "home", "H10", "saved", 0.08),
    ]
    write_recording(d, meta, arrays, events, fixture_moments(), created_by="agents.tests")
    return d


@pytest.fixture(scope="session")
def match(match_dir: Path) -> MatchData:
    return MatchData(match_dir)
