from pathlib import Path

import pytest

from server.config import Settings
from server.engine import UnknownSnapshotError, load_engine, parse_snapshot_ref
from server.fake_engine import FakeEngine
from server.fixture import fixture_events, fixture_moments, write_fixture_recording
from server.main import app_factory, resolve_recording_dir
from shared.recording import Recording


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
