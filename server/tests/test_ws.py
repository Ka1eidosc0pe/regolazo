"""WebSocket integration tests against a real uvicorn server on an ephemeral port (real clock, real sockets)."""

import asyncio
import itertools
import json
import statistics
import time
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from websockets.asyncio.client import ClientConnection, connect

from server.app import create_app
from server.config import Settings
from server.fake_engine import FakeEngine
from server.testing import LiveServer
from shared.recording import Recording
from shared.schemas.stream import ErrorMsg, FrameMsg, HelloMsg, MomentMsg

Msg = dict[str, Any]


@pytest.fixture(scope="module")
def server(recording_dir: Path) -> Iterator[LiveServer]:
    app = create_app(recording_dir, FakeEngine("demo-0001"), Settings(languages=("en", "es", "fr")))
    with LiveServer(app) as srv:
        yield srv


def run(scenario: Callable[[], Awaitable[None]]) -> None:
    asyncio.run(asyncio.wait_for(scenario(), timeout=60))


async def recv(ws: ClientConnection, timeout: float = 3.0) -> Msg:
    msg: Msg = json.loads(await asyncio.wait_for(ws.recv(), timeout))
    return msg


async def send(ws: ClientConnection, **cmd: Any) -> None:
    await ws.send(json.dumps(cmd))


async def until(ws: ClientConnection, pred: Callable[[Msg], bool], timeout: float = 10.0) -> list[Msg]:
    """Receive messages up to and including the first that satisfies ``pred``."""
    out: list[Msg] = []
    end = time.monotonic() + timeout
    while True:
        out.append(await recv(ws, max(0.05, end - time.monotonic())))
        if pred(out[-1]):
            return out


async def silence(ws: ClientConnection, seconds: float) -> None:
    """Assert that nothing arrives for ``seconds``."""
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(ws.recv(), seconds)


async def open_session(server: LiveServer, query: str = "") -> tuple[ClientConnection, Msg]:
    ws = await connect(server.ws_url + query, max_size=None)
    hello = await recv(ws)
    first = await recv(ws)  # the frame at the initial clock position follows hello
    assert first["type"] == "frame"
    assert first["state"]["tick"] == 0
    return ws, hello


async def collect_frames(ws: ClientConnection, seconds: float) -> tuple[list[Msg], list[float]]:
    frames: list[Msg] = []
    arrivals: list[float] = []
    end = time.monotonic() + seconds
    while (left := end - time.monotonic()) > 0:
        try:
            msg = await recv(ws, left)
        except TimeoutError:
            break
        if msg["type"] == "frame":
            frames.append(msg)
            arrivals.append(time.monotonic())
    return frames, arrivals


def ticks(frames: list[Msg]) -> list[int]:
    return [f["state"]["tick"] for f in frames]


def test_hello_then_first_frame_and_contracts(server: LiveServer) -> None:
    async def scenario() -> None:
        ws = await connect(server.ws_url, max_size=None)
        raw_hello = json.loads(await ws.recv())
        hello = HelloMsg.model_validate(raw_hello)  # first message is a valid hello
        assert hello.schema_version == "1.0.0"
        assert hello.meta.match_id == "demo-0001"
        assert hello.profile.session_id == hello.session_id
        assert (hello.profile.language, hello.profile.mode) == ("en", "casual")
        assert hello.clock.playing is False
        assert hello.clock.tick == 0
        assert hello.clock.speed == 1.0
        assert hello.clock.duration_s == pytest.approx(19.96)
        first = FrameMsg.model_validate(await recv(ws))
        assert first.state.tick == 0
        await silence(ws, 0.3)  # paused: nothing else is pushed
        await ws.close()

    run(scenario)


def test_play_streams_ordered_frames_at_25hz(server: LiveServer) -> None:
    async def scenario() -> None:
        ws, _ = await open_session(server)
        await send(ws, type="play")
        frames, arrivals = await collect_frames(ws, 3.0)
        t = ticks(frames)
        assert t == sorted(t)
        assert len(set(t)) == len(t)
        rate = (len(frames) - 1) / (arrivals[-1] - arrivals[0])
        assert rate == pytest.approx(25.0, abs=1.5), f"rate {rate:.2f} Hz"
        consecutive = sum(b - a == 1 for a, b in itertools.pairwise(t)) / (len(t) - 1)
        assert consecutive > 0.95, "at 1x every recorded tick should be delivered"
        gaps = [(b - a) * 1000 for a, b in itertools.pairwise(arrivals)]
        assert statistics.median(gaps) == pytest.approx(40.0, abs=4.0)
        # game time keeps up with wall time
        game_elapsed = (t[-1] - t[0]) / 25.0
        assert game_elapsed == pytest.approx(arrivals[-1] - arrivals[0], abs=0.15)
        await ws.close()

    run(scenario)


def test_pause_stops_frames_and_resume_continues(server: LiveServer) -> None:
    async def scenario() -> None:
        ws, _ = await open_session(server)
        await send(ws, type="play")
        await asyncio.sleep(0.6)
        await send(ws, type="pause")
        msgs = await until(ws, lambda m: m["type"] == "clock" and m["clock"]["playing"] is False)
        paused_clock = msgs[-1]["clock"]
        last_frame = [m for m in msgs if m["type"] == "frame"][-1]["state"]
        assert paused_clock["tick"] == last_frame["tick"]  # the clock freezes on the last frame sent
        assert paused_clock["t"] == pytest.approx(last_frame["t"])
        await silence(ws, 0.5)  # nothing while paused
        await send(ws, type="play")
        resumed = await until(ws, lambda m: m["type"] == "frame")
        assert resumed[0]["type"] == "clock"
        assert resumed[0]["clock"]["playing"] is True
        assert resumed[-1]["state"]["tick"] == last_frame["tick"] + 1  # picks up exactly where it stopped
        await ws.close()

    run(scenario)


def test_seek_lands_on_the_right_tick_without_replaying_events(server: LiveServer, recording: Recording) -> None:
    async def scenario() -> None:
        ws, _ = await open_session(server)
        await send(ws, type="seek", t=10.0)
        msgs = await until(ws, lambda m: m["type"] == "frame")
        assert [m["type"] for m in msgs] == ["clock", "frame"]  # clock first, then the frame at the new time
        assert msgs[0]["clock"]["tick"] == 250
        assert msgs[0]["clock"]["t"] == pytest.approx(10.0)
        assert msgs[0]["clock"]["playing"] is False
        assert msgs[1]["state"]["tick"] == 250
        assert msgs[1]["state"] == recording.state_at(250).model_dump(mode="json")
        await silence(ws, 0.3)  # events before t=10 are not sent on seek

        await send(ws, type="seek", t=10.019)  # between ticks: lands on the frame at or before
        msgs = await until(ws, lambda m: m["type"] == "frame")
        assert msgs[-1]["state"]["tick"] == 250

        await send(ws, type="seek", t=1e6)  # beyond the end: clamped
        msgs = await until(ws, lambda m: m["type"] == "frame")
        assert msgs[-1]["state"]["tick"] == 499
        assert msgs[0]["clock"]["t"] == pytest.approx(19.96)
        await send(ws, type="seek", t=0)
        msgs = await until(ws, lambda m: m["type"] == "frame")
        assert msgs[-1]["state"]["tick"] == 0
        await ws.close()

    run(scenario)


def test_seek_while_playing_continues_from_new_position(server: LiveServer) -> None:
    async def scenario() -> None:
        ws, _ = await open_session(server)
        await send(ws, type="play")
        await asyncio.sleep(0.3)
        await send(ws, type="seek", t=12.0)
        msgs = await until(ws, lambda m: m["type"] == "clock" and m["clock"]["tick"] == 300)
        assert msgs[-1]["clock"]["playing"] is True
        frames, _ = await collect_frames(ws, 0.5)
        assert frames[0]["state"]["tick"] in (300, 301)  # the in-flight seek frame, or the first after it
        assert all(t >= 300 for t in ticks(frames))
        assert 8 <= len(frames) <= 15
        await ws.close()

    run(scenario)


@pytest.mark.parametrize(
    ("speed", "wall_rate", "tick_rate"), [(0.5, 12.5, 12.5), (0.25, 6.25, 6.25), (2.0, 25.0, 50.0)]
)
def test_speed_changes_the_rate(server: LiveServer, speed: float, wall_rate: float, tick_rate: float) -> None:
    async def scenario() -> None:
        ws, _ = await open_session(server)
        await send(ws, type="speed", speed=speed)
        clock = await until(ws, lambda m: m["type"] == "clock")
        assert clock[-1]["clock"]["speed"] == speed
        await send(ws, type="play")
        frames, arrivals = await collect_frames(ws, 3.0)
        span = arrivals[-1] - arrivals[0]
        t = ticks(frames)
        assert (len(frames) - 1) / span == pytest.approx(wall_rate, rel=0.12)
        assert (t[-1] - t[0]) / span == pytest.approx(tick_rate, rel=0.1)
        await ws.close()

    run(scenario)


def test_high_speed_skips_frames_instead_of_flooding(server: LiveServer) -> None:
    async def scenario() -> None:
        ws, _ = await open_session(server)
        await send(ws, type="speed", speed=8.0)
        await send(ws, type="play")
        frames, arrivals = await collect_frames(ws, 2.0)  # 16 s of game time of a 20 s clip
        span = arrivals[-1] - arrivals[0]
        t = ticks(frames)
        assert (len(frames) - 1) / span <= 27.0, "never above the 25 fps cap"
        assert (t[-1] - t[0]) / span == pytest.approx(200.0, rel=0.1)  # 8 x 25 ticks per second
        assert min(b - a for a, b in itertools.pairwise(t)) >= 5  # frames skipped, ~8 ticks apart
        await ws.close()

    run(scenario)


def test_invalid_speed_is_an_error_not_a_disconnect(server: LiveServer) -> None:
    async def scenario() -> None:
        ws, _ = await open_session(server)
        for bad in (0, -1, 16.5, 100):
            await send(ws, type="speed", speed=bad)
            err = ErrorMsg.model_validate((await until(ws, lambda m: m["type"] == "error"))[-1])
            assert err.code == "invalid_message"
        await send(ws, type="speed", speed=16)  # the boundary is valid
        assert (await until(ws, lambda m: m["type"] == "clock"))[-1]["clock"]["speed"] == 16
        await ws.close()

    run(scenario)


def test_events_and_moments_arrive_in_order_exactly_once(server: LiveServer, recording: Recording) -> None:
    async def scenario() -> None:
        ws, _ = await open_session(server)
        await send(ws, type="speed", speed=16.0)
        await send(ws, type="play")
        msgs = await until(
            ws, lambda m: m["type"] == "clock" and m["clock"]["playing"] is False and m["clock"]["t"] > 19
        )
        last_t = -1.0
        got: list[str] = []
        moments: list[str] = []
        frame_t = 0.0
        for m in msgs:
            if m["type"] == "frame":
                frame_t = m["state"]["t"]
                assert frame_t > last_t
                last_t = frame_t
            elif m["type"] == "events":
                for e in m["events"]:
                    assert e["t"] <= frame_t + 1e-9, "an event is never delivered before the frame it belongs to"
                    got.append(e["id"])
            elif m["type"] == "moment":
                MomentMsg.model_validate(m)
                assert m["moment"]["t"] <= frame_t + 1e-9
                moments.append(m["moment"]["id"])
        expected = recording.events()
        assert got == [e.id for e in expected]  # all of them, in order, each exactly once
        assert moments == [mo.id for mo in recording.moments()]
        # the match ended: clock reports not playing at the final tick
        assert msgs[-1]["clock"]["tick"] == 499
        assert msgs[-1]["clock"]["playing"] is False
        assert [m["state"]["tick"] for m in msgs if m["type"] == "frame"][-1] == 499
        await ws.close()

    run(scenario)


def test_seek_back_then_play_resends_events_once_per_pass(server: LiveServer, recording: Recording) -> None:
    async def scenario() -> None:
        ws, _ = await open_session(server)
        events = recording.events()
        cut = events[10].t  # seek to just after event 10: events 0..10 must not be replayed, 11.. must come
        await send(ws, type="seek", t=cut)
        await until(ws, lambda m: m["type"] == "frame")
        await send(ws, type="speed", speed=16.0)
        await send(ws, type="play")
        msgs = await until(
            ws, lambda m: m["type"] == "clock" and m["clock"]["playing"] is False and m["clock"]["t"] > 19
        )
        got = [e["id"] for m in msgs if m["type"] == "events" for e in m["events"]]
        landed_t = recording.arrays["t"][recording.index_at(cut)]
        assert got == [e.id for e in events if e.t > landed_t]
        # play again at the end restarts from the beginning and delivers everything once more
        await send(ws, type="play")
        msgs = await until(
            ws, lambda m: m["type"] == "clock" and m["clock"]["playing"] is False and m["clock"]["t"] > 19
        )
        got = [e["id"] for m in msgs if m["type"] == "events" for e in m["events"]]
        assert got == [e.id for e in events]
        await ws.close()

    run(scenario)


def test_profile_update_and_language_validation(server: LiveServer) -> None:
    async def scenario() -> None:
        ws, hello = await open_session(server)
        sid = hello["session_id"]
        session = server.app.state.sessions[sid]
        profile = {"session_id": "ignored-by-server", "mode": "analyst", "favorite_club": "away", "language": "es"}
        await send(ws, type="profile", profile=profile)
        await send(ws, type="speed", speed=1.0)  # sync point: processed after the profile command
        await until(ws, lambda m: m["type"] == "clock")
        assert session.profile.language == "es"
        assert session.profile.mode == "analyst"
        assert session.profile.favorite_club == "away"
        assert session.profile.session_id == sid  # the server owns the session id

        await send(ws, type="profile", profile={**profile, "language": "de"})  # valid code, not configured
        err = (await until(ws, lambda m: m["type"] == "error"))[-1]
        assert err["code"] == "unsupported_language"
        assert "en, es, fr" in err["message"]
        assert session.profile.language == "es"  # unchanged

        await send(ws, type="profile", profile={**profile, "language": "Spanish!"})  # not a language code at all
        assert (await until(ws, lambda m: m["type"] == "error"))[-1]["code"] == "invalid_message"
        await send(ws, type="profile", profile={**profile, "mode": "pundit"})
        assert (await until(ws, lambda m: m["type"] == "error"))[-1]["code"] == "invalid_message"
        await ws.close()

    run(scenario)


def test_language_list_is_configuration(recording_dir: Path) -> None:
    app = create_app(recording_dir, FakeEngine(), Settings(languages=("en", "de")))

    async def scenario(srv: LiveServer) -> None:
        ws, hello = await open_session(srv, "?language=de&mode=analyst&club=home")
        assert hello["profile"]["language"] == "de"
        assert hello["profile"]["mode"] == "analyst"
        assert hello["profile"]["favorite_club"] == "home"
        await send(ws, type="profile", profile={**hello["profile"], "language": "fr"})  # fr is not configured here
        assert (await until(ws, lambda m: m["type"] == "error"))[-1]["code"] == "unsupported_language"
        await ws.close()

    with LiveServer(app) as srv:
        run(lambda: scenario(srv))


def test_bad_query_params_fall_back_to_defaults_with_an_error(server: LiveServer) -> None:
    async def scenario() -> None:
        ws, hello = await open_session(server, "?language=xx&mode=boss&club=left")
        assert (hello["profile"]["language"], hello["profile"]["mode"], hello["profile"]["favorite_club"]) == (
            "en",
            "casual",
            None,
        )
        errors = [m for m in [await recv(ws) for _ in range(3)] if m["type"] == "error"]
        assert {e["code"] for e in errors} == {"unsupported_language", "invalid_query"}
        assert len(errors) == 3
        await ws.close()

    run(scenario)


def test_garbage_gets_error_messages_and_the_connection_survives(server: LiveServer) -> None:
    async def scenario() -> None:
        ws, _ = await open_session(server)
        cases: list[tuple[str | bytes, str]] = [
            ("this is not json", "invalid_json"),
            ("[1, 2, 3]", "invalid_message"),
            ('"play"', "invalid_message"),
            ("{}", "invalid_message"),
            ('{"type": "teleport"}', "invalid_message"),
            ('{"type": "seek"}', "invalid_message"),
            ('{"type": "seek", "t": -5}', "invalid_message"),
            ('{"type": "seek", "t": "soon"}', "invalid_message"),
            ('{"type": "play", "extra": 1}', "invalid_message"),
            ('{"type": "profile"}', "invalid_message"),
            (b"\xff\xfe binary junk", "invalid_json"),
        ]
        for payload, code in cases:
            await ws.send(payload)
            err = ErrorMsg.model_validate((await until(ws, lambda m: m["type"] == "error"))[-1])
            assert err.code == code, (payload, err)
            assert err.message
        # still alive and functional
        await send(ws, type="seek", t=5.0)
        msgs = await until(ws, lambda m: m["type"] == "frame")
        assert msgs[-1]["state"]["tick"] == 125
        await ws.close()

    run(scenario)


def test_sessions_are_independent_and_cleaned_up(server: LiveServer) -> None:
    async def scenario() -> None:
        a, ha = await open_session(server)
        b, hb = await open_session(server)
        assert ha["session_id"] != hb["session_id"]
        assert {ha["session_id"], hb["session_id"]} <= set(server.app.state.sessions)
        await send(a, type="play")
        await send(b, type="seek", t=15.0)
        frames_a, _ = await collect_frames(a, 0.5)
        msgs_b = await until(b, lambda m: m["type"] == "frame" and m["state"]["tick"] == 375)
        assert len(frames_a) >= 8  # a streams
        await silence(b, 0.4)  # b stays paused at its own time
        assert msgs_b[-1]["state"]["tick"] == 375
        await a.close()
        await b.close()
        for _ in range(50):
            if not ({ha["session_id"], hb["session_id"]} & set(server.app.state.sessions)):
                break
            await asyncio.sleep(0.05)
        assert not ({ha["session_id"], hb["session_id"]} & set(server.app.state.sessions))

    run(scenario)


def test_server_can_push_messages_in_order_to_a_session(server: LiveServer) -> None:
    """Other components (agents) push cues / narratives through ``Session.push``; they interleave in order."""
    from shared.schemas.fixtures import narrative, overlay_cue
    from shared.schemas.stream import CueMsg, NarrativeMsg

    async def scenario() -> None:
        ws, hello = await open_session(server)
        session = server.app.state.sessions[hello["session_id"]]
        session.push(CueMsg(cues=[overlay_cue()]))  # thread-safe: called here from the test thread
        session.push(NarrativeMsg(narrative=narrative()))
        first = await recv(ws)
        second = await recv(ws)
        assert [first["type"], second["type"]] == ["cues", "narrative"]
        await ws.close()

    run(scenario)
