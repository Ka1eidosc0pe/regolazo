"""The match-files route serves component outputs next to the recording and nothing outside it."""

from pathlib import Path

from fastapi.testclient import TestClient

from server.app import create_app
from server.fake_engine import FakeEngine
from server.fixture import write_fixture_recording


def test_serves_match_files_and_refuses_escapes(tmp_path: Path) -> None:
    rec = write_fixture_recording(tmp_path / "rec")
    (rec / "narratives").mkdir()
    (rec / "narratives" / "en.jsonl").write_text('{"a": 1}\n')
    (tmp_path / "secret.json").write_text("{}")
    (rec / "notes.txt").write_text("x")
    client = TestClient(create_app(rec, FakeEngine()))
    assert client.get("/api/files/narratives/en.jsonl").text == '{"a": 1}\n'
    assert client.get("/api/files/../secret.json").status_code == 404
    assert client.get("/api/files/%2e%2e/secret.json").status_code == 404
    assert client.get("/api/files/notes.txt").status_code == 404
    assert client.get("/api/files/missing.json").status_code == 404
