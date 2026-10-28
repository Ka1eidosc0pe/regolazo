"""Narratives are loaded at startup and delivered per session (language + mode) as the match clock passes ``t``."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from server.app import create_app
from server.config import Settings
from server.fake_engine import FakeEngine
from server.fixture import write_fixture_recording
from server.narration import NarrationStore
from shared.schemas.narrative import FactCheck, Narrative


def narrative(language: str, mode: str, t: float, text: str, audio: bool = True) -> Narrative:
    return Narrative(
        id=f"nar_{language}_{mode}_{int(t)}",
        kind="live_commentary",
        t=t,
        language=language,
        mode=mode,
        text=text,
        fact_check=FactCheck(verified=True, attempts=1),
        audio_url=f"/api/files/narratives/audio/m_{language}_{mode}_{int(t)}.mp3" if audio else None,
    )


ROWS = [
    narrative("en", "casual", 5.0, "A goal in minute 1."),
    narrative("en", "casual", 12.0, "A save."),
    narrative("en", "analyst", 5.0, "xG 0.31."),
    narrative("es", "casual", 5.0, "Un gol."),
]


@pytest.fixture
def rec(tmp_path: Path) -> Path:
    d = write_fixture_recording(tmp_path / "demo-0001")  # 19.96 s
    folder = d / "narratives"
    (folder / "audio").mkdir(parents=True)
    for lang, mode in (("en", "casual"), ("en", "analyst"), ("es", "casual")):
        rows = [r for r in ROWS if (r.language, r.mode) == (lang, mode)]
        (folder / f"{lang}_{mode}.jsonl").write_text("".join(r.model_dump_json() + "\n" for r in rows))
    (folder / "audio" / "m_en_casual_5.mp3").write_bytes(b"ID3fake")
    (folder / "broken.jsonl").write_text("not json\n")  # skipped, not fatal
    return d


@pytest.fixture
def client(rec: Path) -> Iterator[TestClient]:
    with TestClient(create_app(rec, FakeEngine("demo-0001"), Settings(languages=("en", "es", "fr")))) as c:
        yield c


def play_to_end(client: TestClient, query: str, seek: float | None = None) -> list[dict[str, Any]]:
    """Play the whole 20 s clip at 16x and return every non-frame message."""
    out: list[dict[str, Any]] = []
    with client.websocket_connect("/ws" + query) as ws:
        if seek is not None:
            ws.send_json({"type": "seek", "t": seek})
        ws.send_json({"type": "speed", "speed": 16})
        ws.send_json({"type": "play"})
        while True:
            msg = ws.receive_json()
            if msg["type"] != "frame":
                out.append(msg)
            if msg["type"] == "clock" and not msg["clock"]["playing"] and msg["clock"]["t"] > 19:
                return out


def test_store_loads_valid_lines_only_and_slices_by_time(rec: Path) -> None:
    store = NarrationStore.load(rec)
    assert len(store) == 4 and store.profiles() == [("en", "analyst"), ("en", "casual"), ("es", "casual")]
    assert [n.t for n in store.between("en", "casual", -1.0, 5.0)] == [5.0]
    assert [n.t for n in store.between("en", "casual", 5.0, 20.0)] == [12.0]
    assert store.between("fr", "casual", -1.0, 20.0) == []
    assert len(NarrationStore.load(rec / "nowhere")) == 0


def test_session_gets_only_its_language_and_mode_in_time_order(client: TestClient) -> None:
    msgs = play_to_end(client, "?language=en&mode=casual")
    got = [m["narrative"] for m in msgs if m["type"] == "narrative"]
    assert [(n["id"], n["text"]) for n in got] == [
        ("nar_en_casual_5", "A goal in minute 1."),
        ("nar_en_casual_12", "A save."),
    ]
    assert got[0]["audio_url"] == "/api/files/narratives/audio/m_en_casual_5.mp3"
    spanish = play_to_end(client, "?language=es&mode=casual")
    assert [m["narrative"]["text"] for m in spanish if m["type"] == "narrative"] == ["Un gol."]
    analyst = play_to_end(client, "?language=en&mode=analyst")
    assert [m["narrative"]["text"] for m in analyst if m["type"] == "narrative"] == ["xG 0.31."]


def test_narrative_arrives_after_the_frame_that_covers_it(client: TestClient) -> None:
    with client.websocket_connect("/ws?language=en&mode=casual") as ws:
        ws.send_json({"type": "speed", "speed": 16})
        ws.send_json({"type": "play"})
        last_t = -1.0
        while True:
            msg = ws.receive_json()
            if msg["type"] == "frame":
                last_t = msg["state"]["t"]
            if msg["type"] == "narrative":
                assert last_t >= msg["narrative"]["t"]
                return


def test_forward_seek_skips_and_backward_seek_redelivers(client: TestClient) -> None:
    skipped = play_to_end(client, "?language=en&mode=casual", seek=8.0)
    assert [m["narrative"]["t"] for m in skipped if m["type"] == "narrative"] == [12.0]
    again = play_to_end(client, "?language=en&mode=casual", seek=0.0)
    assert [m["narrative"]["t"] for m in again if m["type"] == "narrative"] == [5.0, 12.0]


def test_rest_list_and_audio_file(client: TestClient) -> None:
    rows = client.get("/api/narratives", params={"language": "en", "mode": "casual"}).json()
    assert [r["t"] for r in rows] == [5.0, 12.0] and rows[0]["fact_check"]["verified"] is True
    assert client.get("/api/narratives", params={"language": "fr"}).json() == []
    audio = client.get(rows[0]["audio_url"])
    assert audio.status_code == 200 and audio.content == b"ID3fake"
