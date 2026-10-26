"""Synthetic recording for tests and development, until the simulator writes real ones.

``write_fixture_recording`` builds a short clip from ``shared.schemas.fixtures``: players and ball oscillate
around the fixture scene, the fixture events are re-timed into the clip, and one DecisionMoment is placed on a
keyframe. ``python -m server --fixture`` writes it under ``$REGOLAZO_DATA_DIR/matches/demo-0001``.
"""

import math
from pathlib import Path

from shared.recording import player_order, snapshot_ref, states_to_arrays, write_recording
from shared.schemas.events import MatchEvent
from shared.schemas.fixtures import decision_moment, event_examples, match_meta, sim_state
from shared.schemas.moments import DecisionMoment
from shared.schemas.state import SimState

FIXTURE_HZ = 25
FIXTURE_MOMENT_TICK = 200


def fixture_states(n_frames: int = 500) -> list[SimState]:
    """``n_frames`` ticks starting at tick 0 / t = 0; positions are smooth periodic functions of time."""
    base = sim_state()
    states: list[SimState] = []
    for k in range(n_frames):
        t = k / FIXTURE_HZ
        players = []
        for i, p in enumerate(base.players):
            w, phase = 2 * math.pi / (8.0 + i % 5), i * 0.7
            amp_x, amp_y = 3.0 + i % 4, 2.0 + i % 3
            dx = amp_x * math.sin(w * t + phase)
            dy = amp_y * math.cos(w * t + phase)
            vx = amp_x * w * math.cos(w * t + phase)
            vy = -amp_y * w * math.sin(w * t + phase)
            players.append(
                p.model_copy(
                    update={
                        "pos": (round(p.pos[0] + dx, 3), round(p.pos[1] + dy, 3)),
                        "vel": (round(vx, 3), round(vy, 3)),
                        "facing": round(math.atan2(vy, vx), 3),
                        "anim_t": round(t % 1.0, 3),
                    }
                )
            )
        carrier = players[7]  # H10 in fixture order (home lineup index 7)
        keyframe = k % FIXTURE_HZ == 0 or k == FIXTURE_MOMENT_TICK
        states.append(
            base.model_copy(
                update={
                    "tick": k,
                    "t": round(t, 6),
                    "players": players,
                    "ball": base.ball.model_copy(
                        update={
                            "pos": (round(carrier.pos[0] + 0.6, 3), round(carrier.pos[1] + 0.1, 3), 0.11),
                            "vel": (carrier.vel[0], carrier.vel[1], 0.0),
                        }
                    ),
                    "keyframe": keyframe,
                    "snapshot_ref": snapshot_ref(base_match_id(), k) if keyframe else None,
                }
            )
        )
    return states


def base_match_id() -> str:
    return match_meta().match_id


def fixture_events(duration_s: float) -> list[MatchEvent]:
    """The fixture ground-truth events, in their original order, spread evenly over the clip."""
    originals = sorted((e for e in event_examples() if e.source == "ground_truth"), key=lambda e: e.t)
    step = (duration_s - 1.0) / len(originals)
    out = []
    for i, e in enumerate(originals):
        t = 0.0 if i == 0 else round(0.5 + i * step, 3)  # the first event is the kickoff, at exactly t = 0
        out.append(e.model_copy(update={"index": i, "t": t, "tick": round(t * FIXTURE_HZ)}))
    return out


def fixture_moments() -> list[DecisionMoment]:
    t = FIXTURE_MOMENT_TICK / FIXTURE_HZ
    m = decision_moment()
    return [
        m.model_copy(
            update={
                "tick": FIXTURE_MOMENT_TICK,
                "t": t,
                "snapshot_ref": snapshot_ref(m.snapshot_ref.split(":")[1], FIXTURE_MOMENT_TICK),
            }
        )
    ]


def write_fixture_recording(directory: Path, n_frames: int = 500) -> Path:
    """Write a synthetic recording to ``directory`` and return it."""
    meta = match_meta()
    states = fixture_states(n_frames)
    arrays = states_to_arrays(states, player_order(meta))
    duration = float(arrays["t"][-1])
    write_recording(directory, meta, arrays, fixture_events(duration), fixture_moments(), created_by="server.fixture")
    return directory
