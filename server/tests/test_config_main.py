import asyncio
import shutil
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.config import Settings, host_port
from server.engine import UnknownSnapshotError, load_engine, parse_snapshot_ref
from server.fake_engine import FakeEngine
from server.fixture import fixture_events, fixture_moments, write_fixture_recording
from server.main import app_factory, build_app, recording_match_id, resolve_recording_dir
from server.session import Session
from server.timeline import Timeline
from shared.recording import Recording
from shared.schemas.narrative import ViewerProfile
from shared.schemas.stream import ErrorMsg


def test_settings_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("REGOLAZO_LANGUAGES", "en, de ,pt-BR")
    monkeypatch.setenv("REGOLAZO_CORS_ORIGINS", "https://app.example")
    monkeypatch.setenv("REGOLAZO_STATIC_DIR", str(tmp_path))
    s = Settings.from_env()
    assert s.languages == ("en", "de", "pt-BR")
    assert s.cors_origins == ("https://app.example",)
    assert s.static_dir == tmp_path
    monkeypatch.delenv("REGOLAZO_LANGUAGES")
    assert Settings.from_env().languages == ("en", "es", "fr")


def test_snapshot_ref_parsing() -> None:
    assert parse_snapshot_ref("snap:demo-0001:58310") == ("demo-0001", 58310)
    for bad in ("", "snap:demo", "snap:demo:x", "foo:demo:1", "snap::1", "snap:a:b:1"):
        with pytest.raises(UnknownSnapshotError):
            parse_snapshot_ref(bad)


def test_load_engine() -> None:
    assert isinstance(load_engine("fake", "m"), FakeEngine)
    assert isinstance(load_engine("server.fake_engine:FakeEngine", "m"), FakeEngine)
    with pytest.raises(ValueError, match="module:callable"):
        load_engine("nonsense")
    with pytest.raises(TypeError):
        load_engine("server.fixture:base_match_id", "m")


def test_fixture_recording_is_valid_and_self_consistent(tmp_path: Path) -> None:
    d = write_fixture_recording(tmp_path / "demo-0001", n_frames=500)
    rec = Recording.open(d)
    assert rec.n_frames == 500
    assert rec.state_at(499).tick == 499
    events = rec.events()
    assert events == fixture_events(rec.manifest["duration_s"])
    assert [e.t for e in events] == sorted(e.t for e in events)
    assert all(0 <= e.t < rec.manifest["duration_s"] for e in events)
    (moment,) = rec.moments()
    assert moment == fixture_moments()[0]
    assert rec.state_at(moment.tick).snapshot_ref == moment.snapshot_ref  # moment sits on a keyframe
    assert rec.state_at(0).players != rec.state_at(100).players  # things move


def test_app_factory_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    write_fixture_recording(tmp_path / "matches" / "demo-0001", n_frames=50)
    monkeypatch.setenv("REGOLAZO_DATA_DIR", str(tmp_path))
    assert resolve_recording_dir() == tmp_path / "matches" / "demo-0001"
    assert resolve_recording_dir("other") == tmp_path / "matches" / "other"
    monkeypatch.setenv("REGOLAZO_ENGINE", "fake")
    app = app_factory()
    assert app.state.timeline.match_id == "demo-0001"
    monkeypatch.setenv("REGOLAZO_DATA_DIR", str(tmp_path / "empty"))
    with pytest.raises(SystemExit):
        resolve_recording_dir()


def test_engine_gets_the_recorded_match_id_not_the_directory_name(
    recording_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    odd = tmp_path / "some-other-directory-name"
    shutil.copytree(recording_dir, odd)
    assert recording_match_id(odd) == "demo-0001"
    for app in (build_app(odd, "fake"), _factory_app(monkeypatch, odd)):
        with TestClient(app) as c:
            r = c.get("/api/options", params={"snapshot_ref": "snap:demo-0001:200"})
            assert r.status_code == 200, r.text
            assert (
                c.get("/api/options", params={"snapshot_ref": "snap:some-other-directory-name:200"}).status_code == 404
            )


def _factory_app(monkeypatch: pytest.MonkeyPatch, directory: Path) -> FastAPI:
    monkeypatch.setenv("REGOLAZO_RECORDING_DIR", str(directory))
    monkeypatch.setenv("REGOLAZO_ENGINE", "fake")
    return app_factory()


def test_settings_from_env_covers_pacing_and_engine_knobs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REGOLAZO_MAX_FPS", "30")
    monkeypatch.setenv("REGOLAZO_FRAME_CACHE", "99")
    monkeypatch.setenv("REGOLAZO_ENGINE_CONCURRENCY", "5")
    monkeypatch.setenv("REGOLAZO_ENGINE_THREAD_SAFE", "true")
    monkeypatch.setenv("REGOLAZO_RESULT_CACHE", "7")
    s = Settings.from_env()
    assert (s.max_fps, s.frame_cache_size, s.engine_concurrency, s.engine_thread_safe, s.result_cache_size) == (
        30.0,
        99,
        5,
        True,
        7,
    )
    assert Settings().engine_thread_safe is False
    assert Settings().engine_concurrency == 2


def test_host_and_port_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    assert host_port() == ("127.0.0.1", 8000)
    monkeypatch.setenv("REGOLAZO_HOST", "0.0.0.0")
    monkeypatch.setenv("REGOLAZO_PORT", "8080")
    assert host_port() == ("0.0.0.0", 8080)


def test_session_refuses_pushes_after_it_finished(recording_dir: Path) -> None:
    async def scenario() -> None:
        tl = Timeline.open(recording_dir)
        session = Session("sess_x", tl, ViewerProfile(session_id="sess_x"), ("en",))
        sent: list[str] = []

        async def send(text: str) -> None:
            sent.append(text)

        task = asyncio.create_task(session.run(send))
        await asyncio.sleep(0.05)
        assert session.push(ErrorMsg(code="x", message="while running"))
        await asyncio.sleep(0.05)
        session.close()
        await task
        assert not session.push(ErrorMsg(code="x", message="after the end"))  # refused, no exception
        assert sum('"while running"' in m for m in sent) == 1
        assert not any("after the end" in m for m in sent)

    asyncio.run(scenario())


def test_push_to_a_session_whose_loop_is_closed_is_swallowed(recording_dir: Path) -> None:
    loop = asyncio.new_event_loop()
    session = loop.run_until_complete(_make_session(recording_dir))
    loop.close()
    assert session.push(ErrorMsg(code="x", message="nobody home")) is False


async def _make_session(recording_dir: Path) -> Session:
    return Session("sess_y", Timeline.open(recording_dir), ViewerProfile(session_id="sess_y"), ("en",))
