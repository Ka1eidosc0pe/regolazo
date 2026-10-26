"""Engine calls: off the event loop, bounded, serialized by default, cached when seeded; snapshot validation."""

import concurrent.futures
import threading
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from server.app import create_app
from server.config import Settings
from server.fake_engine import FakeEngine
from server.testing import LiveServer
from shared.schemas.fixtures import rollout_request
from shared.schemas.rollout import RolloutDistribution, RolloutRequest, RolloutResult

SNAP = "snap:demo-0001:200"
DELAY_S = 0.6


def _req(**kw: object) -> dict[str, object]:
    base: dict[str, object] = rollout_request().model_dump(mode="json")
    base.update({"snapshot_ref": SNAP, **kw})
    return base


class SlowEngine(FakeEngine):
    """Sleeps in every heavy call and records how many calls overlapped."""

    def __init__(self) -> None:
        super().__init__("demo-0001")
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def _slow(self) -> None:
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(DELAY_S)
        with self._lock:
            self.active -= 1

    def rollout(self, request: RolloutRequest) -> RolloutResult:
        self._slow()
        return super().rollout(request)

    def distribution(self, request: RolloutRequest) -> RolloutDistribution:
        self._slow()
        return super().distribution(request)


def _fire(base_url: str, n: int) -> list[concurrent.futures.Future[httpx.Response]]:
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=n)
    return [pool.submit(httpx.post, f"{base_url}/api/rollout", json=_req(seed=100 + i), timeout=30) for i in range(n)]


def test_cheap_routes_stay_fast_while_slow_engine_calls_are_in_flight(recording_dir: Path) -> None:
    engine = SlowEngine()
    app = create_app(recording_dir, engine, Settings(engine_concurrency=2, engine_thread_safe=True))
    with LiveServer(app) as srv:
        futures = _fire(srv.http_url, 6)  # 6 x 0.6 s of engine work, 2 at a time: ~1.8 s in flight
        time.sleep(0.2)
        worst = 0.0
        t_end = time.monotonic() + 1.0
        while time.monotonic() < t_end:
            for path in ("/healthz", "/api/meta", "/api/events", "/api/moments", "/api/state?t=3"):
                t0 = time.monotonic()
                assert httpx.get(srv.http_url + path, timeout=5).status_code == 200
                worst = max(worst, time.monotonic() - t0)
        assert engine.active > 0, "the slow calls must still be running while we measured"
        assert worst < 0.25, f"cheap route took {worst * 1000:.0f} ms behind engine calls"
        assert all(f.result().status_code == 200 for f in futures)
    assert engine.max_active == 2  # bounded by engine_concurrency


def test_engine_calls_are_serialized_by_default(recording_dir: Path) -> None:
    engine = SlowEngine()
    app = create_app(recording_dir, engine, Settings(engine_concurrency=4))  # thread_safe defaults to False
    with LiveServer(app) as srv:
        t0 = time.monotonic()
        assert all(f.result().status_code == 200 for f in _fire(srv.http_url, 3))
        assert time.monotonic() - t0 >= 3 * DELAY_S * 0.95
    assert engine.max_active == 1


def test_concurrency_limit_is_configurable(recording_dir: Path) -> None:
    engine = SlowEngine()
    app = create_app(recording_dir, engine, Settings(engine_concurrency=3, engine_thread_safe=True))
    with LiveServer(app) as srv:
        assert all(f.result().status_code == 200 for f in _fire(srv.http_url, 6))
    assert engine.max_active == 3


def test_seeded_rollouts_and_distributions_are_cached(recording_dir: Path) -> None:
    engine = FakeEngine("demo-0001")
    app = create_app(recording_dir, engine, Settings(result_cache_size=2))
    with TestClient(app) as c:
        a = c.post("/api/rollout", json=_req(seed=1)).json()
        assert c.post("/api/rollout", json=_req(seed=1)).json() == a
        assert engine.calls == ["rollout"]  # second request served from the cache
        c.post("/api/rollout", json=_req(seed=2))
        assert engine.calls == ["rollout", "rollout"]  # a different seed is a different request
        c.post("/api/rollout", json=_req(seed=1, horizon_s=5.0))
        assert engine.calls == ["rollout"] * 3  # so is any other field
        c.post("/api/rollout", json=_req(seed=1, horizon_s=5.0))
        assert engine.calls == ["rollout"] * 3
        # LRU of 2: seed=1 (10 s) is the oldest and was evicted by the two newer entries
        c.post("/api/rollout", json=_req(seed=1))
        assert engine.calls == ["rollout"] * 4
        # distributions are cached separately from rollouts of the same request
        d = c.post("/api/distribution", json=_req(seed=1, n=100)).json()
        assert c.post("/api/distribution", json=_req(seed=1, n=100)).json() == d
        assert engine.calls.count("distribution") == 1
        # no seed: the server picks one, so every call is a fresh simulation
        before = len(engine.calls)
        c.post("/api/rollout", json=_req(seed=None))
        c.post("/api/rollout", json=_req(seed=None))
        assert len(engine.calls) == before + 2


@pytest.mark.parametrize(
    "ref",
    [
        "snap:demo-0001:201",  # a real frame, but not a keyframe
        "snap:demo-0001:99999",  # beyond the recording
        "snap:demo-0001:-5",
        "snap:demo-0001:٢٠٠",  # non-ASCII digits
        "snap:demo-0001:2 0",
    ],
)
def test_snapshot_must_be_a_keyframe_of_the_recording(client: TestClient, ref: str) -> None:
    assert client.get("/api/options", params={"snapshot_ref": ref}).status_code == 404
    assert client.post("/api/rollout", json=_req(snapshot_ref=ref)).status_code == 404
    assert client.post("/api/distribution", json=_req(snapshot_ref=ref)).status_code == 404
    preview = {"snapshot_ref": ref, "intent": rollout_request().intent.model_dump(mode="json")}
    assert client.post("/api/preview", json=preview).status_code == 404


def test_keyframe_snapshot_is_accepted_for_any_engine(client: TestClient, engine: FakeEngine) -> None:
    for tick in (0, 25, 200, 475):
        assert client.get("/api/options", params={"snapshot_ref": f"snap:demo-0001:{tick}"}).status_code == 200
    assert engine.calls == ["options"] * 4
