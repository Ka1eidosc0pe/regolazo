"""Short in-process load tests (server and clients share one interpreter, so this is a floor, not a benchmark).

The full gate run is ``uv run python -m server.loadtest --sessions 20 --seconds 10 [--spread 4]`` against a
separate server process. Tolerances here are wide so slow CI runners do not flake.
"""

import asyncio
import statistics
from pathlib import Path

import pytest

from server.app import create_app
from server.config import Settings
from server.fake_engine import FakeEngine
from server.fixture import write_fixture_recording
from server.loadtest import SessionStats, format_report, run_load
from server.testing import LiveServer


@pytest.mark.parametrize("spread_s", [0.0, 2.5], ids=["same-position", "divergent-positions"])
def test_many_concurrent_viewers_keep_the_frame_rate(tmp_path: Path, spread_s: float) -> None:
    rec = write_fixture_recording(tmp_path / "demo-0001", n_frames=1500)  # 60 s: sessions spread over 0..27.5 s
    app = create_app(rec, FakeEngine("demo-0001"), Settings())
    with LiveServer(app) as srv:
        results: list[SessionStats] = asyncio.run(run_load(srv.ws_url, sessions=12, seconds=4.0, spread_s=spread_s))
    print(f"\n[spread {spread_s} s]\n" + format_report(results))
    assert len(results) == 12
    for r in results:
        assert not r.errors
        assert 21.0 <= r.fps <= 27.0, f"session {r.index}: {r.fps:.2f} fps"
        assert statistics.quantiles(r.lateness_ms, n=20)[18] < 250.0  # p95 lateness vs the 25 Hz schedule
        assert r.rtt_ms
        assert max(r.rtt_ms) < 1000.0  # command round trip while streaming
        assert r.gaps <= 10
