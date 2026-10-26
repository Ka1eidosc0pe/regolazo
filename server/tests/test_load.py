"""Short in-process load test (server and clients share one interpreter, so this is a floor, not a benchmark).

The full gate run is ``uv run python -m server.loadtest --sessions 20 --seconds 10`` against a separate process.
"""

import asyncio
import statistics
from pathlib import Path

from server.app import create_app
from server.config import Settings
from server.fake_engine import FakeEngine
from server.fixture import write_fixture_recording
from server.loadtest import SessionStats, format_report, run_load
from server.testing import LiveServer


def test_many_concurrent_viewers_keep_the_frame_rate(tmp_path: Path) -> None:
    rec = write_fixture_recording(tmp_path / "demo-0001", n_frames=1000)  # 40 s, longer than the test
    app = create_app(rec, FakeEngine("demo-0001"), Settings())
    with LiveServer(app) as srv:
        results: list[SessionStats] = asyncio.run(run_load(srv.ws_url, sessions=12, seconds=4.0))
    print("\n" + format_report(results))
    assert len(results) == 12
    for r in results:
        assert not r.errors
        assert 24.0 <= r.fps <= 26.5, f"session {r.index}: {r.fps:.2f} fps"
        assert statistics.quantiles(r.lateness_ms, n=20)[18] < 100.0  # p95 lateness vs the 25 Hz schedule
        assert r.rtt_ms
        assert max(r.rtt_ms) < 250.0  # command round trip while streaming
        assert r.gaps <= 3
