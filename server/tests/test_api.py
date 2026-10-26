from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.app import create_app, get_engine
from server.config import Settings
from server.fake_engine import FakeEngine
from shared.engine_api import RolloutEngine
from shared.recording import Recording
from shared.schemas.actions import ActionIntent, ResolvedAction
from shared.schemas.common import SCHEMA_VERSION
from shared.schemas.events import MatchEvent
from shared.schemas.fixtures import rollout_request
from shared.schemas.meta import MatchMeta
from shared.schemas.models import ModelOutputs
from shared.schemas.moments import DecisionMoment
from shared.schemas.rollout import RolloutDistribution, RolloutRequest, RolloutResult
from shared.schemas.state import SimState

SNAP = "snap:demo-0001:200"


def _req(**kw: object) -> dict[str, object]:
    base: dict[str, object] = rollout_request().model_dump(mode="json")
    base.update({"snapshot_ref": SNAP, **kw})
    return base


def test_fake_engine_implements_protocol(engine: FakeEngine) -> None:
    assert isinstance(engine, RolloutEngine)


def test_healthz(client: TestClient) -> None:
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {
        "status": "ok",
        "schema_version": SCHEMA_VERSION,
        "match_id": "demo-0001",
        "n_frames": 500,
        "duration_s": pytest.approx(19.96),
        "sessions": 0,
    }


def test_meta(client: TestClient, recording: Recording) -> None:
    meta = MatchMeta.model_validate(client.get("/api/meta").json())
    assert meta == recording.meta


def test_events_all_and_window(client: TestClient, recording: Recording) -> None:
    expected = recording.events()
    allev = [MatchEvent.model_validate(e) for e in client.get("/api/events").json()]
    assert allev == expected
    assert [e.t for e in allev] == sorted(e.t for e in allev)
    lo, hi = expected[3].t, expected[7].t
    win = client.get("/api/events", params={"from": lo, "to": hi}).json()
    assert [e["id"] for e in win] == [e.id for e in expected[3:8]]  # inclusive on both ends
    assert client.get("/api/events", params={"from": 1000}).json() == []
    assert client.get("/api/events", params={"from": 5, "to": 1}).status_code == 422
    assert client.get("/api/events", params={"from": -1}).status_code == 422


def test_moments(client: TestClient, recording: Recording) -> None:
    moments = [DecisionMoment.model_validate(m) for m in client.get("/api/moments").json()]
    assert moments == recording.moments()
    assert moments[0].snapshot_ref == SNAP


def test_state(client: TestClient, recording: Recording) -> None:
    r = client.get("/api/state", params={"t": 10.0})
    assert r.status_code == 200
    state = SimState.model_validate(r.json())
    assert state.tick == 250
    assert state == recording.state_at(250)
    assert SimState.model_validate(client.get("/api/state", params={"t": 10.01}).json()).tick == 250
    assert SimState.model_validate(client.get("/api/state", params={"t": 9999}).json()).tick == 499  # clamped
    assert client.get("/api/state").status_code == 422
    assert client.get("/api/state", params={"t": -1}).status_code == 422


def test_preview(client: TestClient, engine: FakeEngine) -> None:
    intent = rollout_request().intent
    r = client.post("/api/preview", json={"snapshot_ref": SNAP, "intent": intent.model_dump(mode="json")})
    assert r.status_code == 200
    resolved = ResolvedAction.model_validate(r.json())
    assert resolved.intent == intent
    assert engine.calls == ["preview"]


def test_options(client: TestClient) -> None:
    r = client.get("/api/options", params={"snapshot_ref": SNAP})
    assert r.status_code == 200
    out = ModelOutputs.model_validate(r.json())
    assert out.tick == 200
    assert out.options
    assert client.get("/api/options").status_code == 422


def test_rollout_with_and_without_frames(client: TestClient) -> None:
    r = client.post("/api/rollout", json=_req(horizon_s=4.0, frame_stride=5))
    assert r.status_code == 200
    res = RolloutResult.model_validate(r.json())
    assert res.snapshot_ref == SNAP
    assert res.seed == 7
    assert len(res.frames) == 20
    assert res.frames[0].tick == 200
    assert all(e.rollout_id == res.id for e in res.events)
    r2 = client.post("/api/rollout", json=_req(return_frames=False))
    assert RolloutResult.model_validate(r2.json()).frames == []


def test_distribution(client: TestClient) -> None:
    r = client.post("/api/distribution", json=_req(n=500))
    assert r.status_code == 200
    dist = RolloutDistribution.model_validate(r.json())
    assert dist.n == 500
    assert sum(c.count for c in dist.outcome_counts) == 500
    assert dist.snapshot_ref == SNAP


@pytest.mark.parametrize(
    ("method", "path", "payload"),
    [
        ("get", "/api/options?snapshot_ref=snap:other-match:5", None),
        ("get", "/api/options?snapshot_ref=garbage", None),
        ("post", "/api/rollout", _req(snapshot_ref="snap:other-match:5")),
        ("post", "/api/distribution", _req(snapshot_ref="nope")),
    ],
)
def test_unknown_snapshot_is_404(client: TestClient, method: str, path: str, payload: object) -> None:
    r = client.request(method, path, json=payload)
    assert r.status_code == 404
    assert "detail" in r.json()


def test_invalid_bodies_are_422(client: TestClient) -> None:
    assert client.post("/api/rollout", json={"snapshot_ref": SNAP}).status_code == 422
    assert client.post("/api/rollout", json=_req(n=0)).status_code == 422
    assert client.post("/api/rollout", json=_req(horizon_s=99)).status_code == 422
    assert client.post("/api/preview", json={"snapshot_ref": SNAP, "intent": {"kind": "shot"}}).status_code == 422
    bad = ActionIntent(kind="carry", player="H10", path=[(1.0, 2.0)]).model_dump(mode="json")
    bad["kind"] = "pass_player"  # needs target_player: a cross-field rule enforced by Pydantic
    assert client.post("/api/preview", json={"snapshot_ref": SNAP, "intent": bad}).status_code == 422


class _Raising:
    """Engine whose errors the server must translate."""

    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    def preview(self, snapshot_ref: str, intent: ActionIntent) -> ResolvedAction:
        raise self.exc

    def options(self, snapshot_ref: str) -> ModelOutputs:
        raise self.exc

    def rollout(self, request: RolloutRequest) -> RolloutResult:
        raise self.exc

    def distribution(self, request: RolloutRequest) -> RolloutDistribution:
        raise self.exc


def test_engine_errors_map_to_http(recording_dir: Path) -> None:
    app = create_app(recording_dir, _Raising(KeyError(SNAP)), Settings())
    with TestClient(app) as c:
        assert c.get("/api/options", params={"snapshot_ref": SNAP}).status_code == 404
    app = create_app(recording_dir, _Raising(ValueError("horizon too long")), Settings())
    with TestClient(app) as c:
        r = c.post("/api/rollout", json=_req())
        assert r.status_code == 422
        assert "horizon" in r.json()["detail"]


def test_engine_can_be_overridden_via_dependency(app: FastAPI) -> None:
    sentinel = FakeEngine("demo-0001")
    app.dependency_overrides[get_engine] = lambda: sentinel
    with TestClient(app) as c:
        assert c.get("/api/options", params={"snapshot_ref": SNAP}).status_code == 200
    assert sentinel.calls == ["options"]


def test_cors(client: TestClient) -> None:
    r = client.options(
        "/api/meta",
        headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "GET"},
    )
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == "http://localhost:5173"
    r = client.get("/api/meta", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in r.headers


def test_static_frontend_served_when_configured(recording_dir: Path, tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text("<h1>ReGolazo</h1>")
    app = create_app(recording_dir, FakeEngine(), Settings(static_dir=tmp_path))
    with TestClient(app) as c:
        assert "ReGolazo" in c.get("/").text
        assert c.get("/healthz").json()["status"] == "ok"  # API routes still win
    plain = create_app(recording_dir, FakeEngine(), Settings())
    with TestClient(plain) as c:
        assert c.get("/").status_code == 404
