from pathlib import Path

import numpy as np

from shared.recording import FRAME_ARRAYS, Recording, player_order, snapshot_ref, states_to_arrays, write_recording
from shared.schemas.fixtures import decision_moment, event_examples, match_event, match_meta, sim_state


def test_recording_round_trip(tmp_path: Path) -> None:
    meta = match_meta()
    first = sim_state()
    second = first.model_copy(
        update={"tick": first.tick + 1, "t": first.t + 0.04, "keyframe": False, "snapshot_ref": None}
    )
    arrays = states_to_arrays([first, second], player_order(meta))
    assert set(arrays) == set(FRAME_ARRAYS)
    events = [*event_examples(), match_event()]
    write_recording(tmp_path / "rec", meta, arrays, events, [decision_moment()])

    rec = Recording.open(tmp_path / "rec")
    assert rec.n_frames == 2 and rec.meta == meta
    assert rec.events() == events and rec.moments() == [decision_moment()]
    assert rec.index_at(first.t + 0.01) == 0 and rec.index_at(1e9) == 1 and rec.index_at(0) == 0

    restored = rec.state_at(0)
    assert restored.snapshot_ref == snapshot_ref(meta.match_id, first.tick)
    assert rec.state_at(1).snapshot_ref is None
    assert restored.possession == first.possession and restored.score == first.score
    assert restored.phase == first.phase and restored.status == first.status
    by_id = {p.id: p for p in restored.players}
    for p in first.players:
        q = by_id[p.id]
        assert q.anim == p.anim
        np.testing.assert_allclose(q.pos, p.pos, atol=1e-3)
        np.testing.assert_allclose(q.vel, p.vel, atol=1e-3)
    np.testing.assert_allclose(restored.ball.pos, first.ball.pos, atol=1e-3)
    assert restored.ball.last_touch == first.ball.last_touch
