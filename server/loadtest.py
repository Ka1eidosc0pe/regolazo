"""Load test: N concurrent viewer sessions playing the match, reporting frame rate and latency per session.

    uv run python -m server.loadtest --sessions 20 --seconds 10            # spawns a server subprocess
    uv run python -m server.loadtest --sessions 20 --seconds 10 --spread 4 # divergent positions: no shared frames
    uv run python -m server.loadtest --url ws://host:8000/ws --sessions 20 # an already running server

Per session it reports the frame rate (frames/s), the mean/p95/max lateness of each frame against the ideal
25 Hz schedule (``arrival - (t0 + (tick - tick0) / 25)``) and the round trip of a control command (``speed``
sent, matching ``clock`` received), which is queued behind frames and so also shows head-of-line delay.
"""

import argparse
import asyncio
import json
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from websockets.asyncio.client import connect

SIM_HZ = 25.0


@dataclass
class SessionStats:
    index: int
    frames: int = 0
    seconds: float = 0.0
    lateness_ms: list[float] = field(default_factory=list)
    interval_ms: list[float] = field(default_factory=list)
    rtt_ms: list[float] = field(default_factory=list)
    gaps: int = 0
    """Frames whose tick did not follow the previous one (skipped frames at 1x)."""
    errors: list[str] = field(default_factory=list)

    @property
    def fps(self) -> float:
        return self.frames / self.seconds if self.seconds else 0.0


def _pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    return s[min(len(s) - 1, round(q * (len(s) - 1)))]


async def _session(url: str, index: int, seconds: float, start_at: float, offset_s: float) -> SessionStats:
    st = SessionStats(index)
    async with connect(url, max_size=None) as ws:
        hello = json.loads(await ws.recv())
        assert hello["type"] == "hello", hello["type"]
        await ws.send(json.dumps({"type": "seek", "t": offset_s}))
        seen_clock = False
        while True:  # consume the initial frame, then the seek's own `clock` + `frame`
            kind = json.loads(await ws.recv())["type"]
            seen_clock = seen_clock or kind == "clock"
            if seen_clock and kind == "frame":
                break
        await asyncio.sleep(max(0.0, start_at - time.monotonic()))  # start all sessions together
        await ws.send(json.dumps({"type": "play"}))
        t_begin = time.monotonic()
        end = t_begin + seconds
        first_tick: int | None = None
        t_first = 0.0
        prev_tick: int | None = None
        prev_arrival: float | None = None
        ping_sent: float | None = None
        next_ping = t_begin + 1.0
        while (now := time.monotonic()) < end:
            if ping_sent is None and now >= next_ping:
                ping_sent = now
                await ws.send(json.dumps({"type": "speed", "speed": 1.0}))
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=end - now)
            except TimeoutError:
                break
            arrival = time.monotonic()
            msg = json.loads(raw)
            kind = msg["type"]
            if kind == "frame":
                tick = msg["state"]["tick"]
                if first_tick is None:
                    first_tick, t_first = tick, arrival
                st.frames += 1
                st.lateness_ms.append((arrival - (t_first + (tick - first_tick) / SIM_HZ)) * 1000)
                if prev_arrival is not None:
                    st.interval_ms.append((arrival - prev_arrival) * 1000)
                if prev_tick is not None and tick != prev_tick + 1:
                    st.gaps += 1
                prev_tick, prev_arrival = tick, arrival
            elif kind == "clock" and ping_sent is not None:
                st.rtt_ms.append((arrival - ping_sent) * 1000)
                ping_sent, next_ping = None, arrival + 1.0
            elif kind == "error":
                st.errors.append(msg["message"])
        st.seconds = time.monotonic() - t_begin
    return st


async def run_load(url: str, sessions: int, seconds: float, spread_s: float = 0.0) -> list[SessionStats]:
    """Session ``i`` starts watching at match time ``i * spread_s`` (0 = everyone at the same position)."""
    start_at = time.monotonic() + 1.0 + 0.05 * sessions
    return list(await asyncio.gather(*(_session(url, i, seconds, start_at, i * spread_s) for i in range(sessions))))


def format_report(results: list[SessionStats]) -> str:
    lines = [
        f"{'sess':>4} {'fps':>6} {'frames':>6} {'gaps':>4} {'late_mean':>9} {'late_p95':>8} {'late_max':>8} "
        f"{'dt_p50':>6} {'dt_p95':>6} {'rtt_ms':>7}"
    ]
    for r in results:
        lines.append(
            f"{r.index:>4} {r.fps:>6.2f} {r.frames:>6} {r.gaps:>4} {statistics.fmean(r.lateness_ms):>9.1f} "
            f"{_pct(r.lateness_ms, 0.95):>8.1f} {max(r.lateness_ms):>8.1f} {_pct(r.interval_ms, 0.5):>6.1f} "
            f"{_pct(r.interval_ms, 0.95):>6.1f} {statistics.fmean(r.rtt_ms) if r.rtt_ms else float('nan'):>7.1f}"
        )
    fps = [r.fps for r in results]
    lines.append(
        f"sessions={len(results)} fps min/mean/max = {min(fps):.2f}/{statistics.fmean(fps):.2f}/{max(fps):.2f}; "
        f"worst lateness p95 = {max(_pct(r.lateness_ms, 0.95) for r in results):.1f} ms; "
        f"errors = {sum(len(r.errors) for r in results)}"
    )
    return "\n".join(lines)


def _spawn_server(port: int) -> tuple[subprocess.Popen[bytes], str]:
    tmp = tempfile.mkdtemp(prefix="regolazo_load_")
    from server.fixture import write_fixture_recording

    rec = write_fixture_recording(Path(tmp) / "demo-0001", n_frames=2500)  # 100 s: longer than the test
    proc = subprocess.Popen(
        [sys.executable, "-m", "server", "--recording", str(rec), "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=0.5).read()
            return proc, f"ws://127.0.0.1:{port}/ws"
        except OSError:
            time.sleep(0.2)
    proc.kill()
    raise RuntimeError("server subprocess did not start")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", help="ws URL of a running server (default: spawn one)")
    ap.add_argument("--sessions", type=int, default=20)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--spread", type=float, default=0.0, help="seconds between sessions' start positions")
    args = ap.parse_args()
    proc = None
    url = args.url
    if url is None:
        proc, url = _spawn_server(args.port)
    try:
        print(format_report(asyncio.run(run_load(url, args.sessions, args.seconds, args.spread))))
    finally:
        if proc is not None:
            proc.terminate()
            proc.wait(timeout=10)


if __name__ == "__main__":
    main()
