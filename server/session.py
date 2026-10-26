"""One viewer session: a match clock plus a ViewerProfile, driving what is pushed over one WebSocket.

All sends happen from a single task (``Session.run``), so message order is deterministic: commands from the
client arrive through an inbox queue, are applied to the clock, and the resulting messages are sent in order.

Clock semantics
- The clock is wall-time driven: ``t = t_ref + (now - wall_ref) * speed``. A slow client never delays the
  clock; it simply receives fewer frames (the newest frame at or before ``t`` is always the one sent).
- Frames are sent at the recorded rate scaled by speed (25 Hz at 1x, 6.25 Hz at 0.25x) and never faster than
  ``max_fps``: at 2x and above, frames are skipped, not queued.
- ``events`` / ``moment`` messages are delivered when the clock passes them during playback, on every pass,
  always after the frame whose time covers them. A backward seek followed by play therefore re-delivers
  everything after the new position, in order, once per pass; clients dedupe by ``MatchEvent.id`` /
  ``DecisionMoment.id``. A forward seek skips the span without sending it; clients fetch history from
  ``GET /api/events`` and ``GET /api/moments``. Seeking to the first frame rewinds the cursors fully, so
  events at ``t <= 0`` (the kickoff) are delivered again with that frame.
- After a seek the clock snaps to the frame at the requested time, then ``clock`` and ``frame`` are sent.
  Pause freezes the clock at the last frame sent.
- Consecutive ``seek`` (or ``speed``) commands that are already queued are coalesced: only the last is applied.
- The inbox is bounded; a client flooding commands is back-pressured through its socket.
"""

import asyncio
import bisect
import contextlib
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from pydantic import BaseModel, TypeAdapter, ValidationError

from server.timeline import Timeline
from shared.schemas.narrative import ViewerProfile
from shared.schemas.stream import (
    ClientMessage,
    ClockMsg,
    ClockStatus,
    ErrorMsg,
    EventsMsg,
    HelloMsg,
    MomentMsg,
    PauseCmd,
    PlayCmd,
    ProfileCmd,
    SeekCmd,
    SpeedCmd,
)

log = logging.getLogger("regolazo.session")

Send = Callable[[str], Awaitable[None]]
ClientCmd = PlayCmd | PauseCmd | SeekCmd | SpeedCmd | ProfileCmd

_EPS = 1e-4
INBOX_SIZE = 256
ClientMessageAdapter: TypeAdapter[ClientMessage] = TypeAdapter(ClientMessage)


@dataclass
class _Close:
    """Inbox sentinel: the client went away."""


@dataclass
class _Raw:
    """Inbox item: one raw text frame from the client, parsed on the session task."""

    text: str


@dataclass
class _Push:
    """Inbox item: a server-originated message (cues, narrative, ...) to deliver in order."""

    text: str


class Session:
    def __init__(
        self,
        session_id: str,
        timeline: Timeline,
        profile: ViewerProfile,
        languages: tuple[str, ...],
        max_fps: float = 25.0,
    ) -> None:
        self.id = session_id
        self.timeline = timeline
        self.profile = profile
        self.languages = languages
        self.max_fps = max_fps
        self._loop = asyncio.get_running_loop()  # Sessions are created by the /ws handler, on the server loop
        self.inbox: asyncio.Queue[_Close | _Raw | _Push] = asyncio.Queue(maxsize=INBOX_SIZE)
        self._carry: _Close | _Raw | _Push | None = None
        self._closed = False
        # Clock
        self.playing = False
        self.speed = 1.0
        self._t_ref = 0.0
        self._wall_ref = 0.0
        self._idx = 0  # index of the last frame sent
        self._pace_wall = float("-inf")
        """Earliest wall time of the next frame (drift-free grid of 1 / max_fps)."""
        self._event_cursor = 0
        self._moment_cursor = 0
        self._send: Send | None = None
        self.frames_sent = 0

    # --- public ---------------------------------------------------------------------------------------------

    def push(self, message: BaseModel) -> bool:
        """Queue a server message (e.g. cues, a narrative) for in-order delivery to this viewer.

        Thread-safe. Returns False (and drops the message) once the session has finished or its inbox is full.
        """
        if self._closed:
            return False
        item = _Push(message.model_dump_json())
        try:
            self._loop.call_soon_threadsafe(self._put_nowait, item)
        except RuntimeError:  # the event loop is closed (server shutting down)
            return False
        return True

    def _put_nowait(self, item: _Close | _Raw | _Push) -> None:
        try:
            self.inbox.put_nowait(item)
        except asyncio.QueueFull:
            log.warning("session %s: inbox full, dropped %s", self.id, type(item).__name__)

    async def receive(self, text: str) -> None:
        """Queue one raw client frame; waits when the inbox is full, which back-pressures the socket."""
        await self.inbox.put(_Raw(text))

    def close(self) -> None:
        self._closed = True
        with contextlib.suppress(asyncio.QueueFull):
            self.inbox.put_nowait(_Close())  # if full, the run loop sees ``_closed`` after its next item

    def clock_status(self) -> ClockStatus:
        tl = self.timeline
        t = tl.time_of(self._idx)
        if self.playing:
            t = min(self._now_t(self._loop_time()), tl.duration_s)
        return ClockStatus(
            t=t,
            tick=tl.tick_of(tl.index_at(t)),
            playing=self.playing,
            speed=self.speed,
            duration_s=tl.duration_s,
        )

    def hello(self) -> HelloMsg:
        return HelloMsg(
            session_id=self.id,
            meta=self.timeline.meta,
            profile=self.profile,
            clock=self.clock_status(),
        )

    async def run(self, send: Send) -> None:
        """Send ``hello`` and the first frame, then serve commands and the clock until closed."""
        self._send = send
        try:
            await send(self.hello().model_dump_json())
            await self._emit_frame(0)
            while not self._closed:
                item = self._carry
                self._carry = None
                if item is None:
                    timeout: float | None = None
                    if self.playing:
                        timeout = max(0.0, self._next_wall() - self._loop_time())
                    try:
                        item = await asyncio.wait_for(self.inbox.get(), timeout)
                    except TimeoutError:
                        await self._tick()
                        continue
                if isinstance(item, _Close):
                    return
                if isinstance(item, _Push):
                    await send(item.text)
                else:
                    await self._handle_raw(item.text)
        finally:
            self._closed = True

    # --- clock ----------------------------------------------------------------------------------------------

    @staticmethod
    def _loop_time() -> float:
        return asyncio.get_running_loop().time()

    def _now_t(self, now: float) -> float:
        return self._t_ref + (now - self._wall_ref) * self.speed

    def _rebase(self, now: float, t: float) -> None:
        self._t_ref, self._wall_ref = t, now

    def _next_wall(self) -> float:
        """Wall time of the next frame to send: when the clock reaches the next frame, but not above max_fps."""
        tl = self.timeline
        if self._idx + 1 >= tl.n_frames:
            return self._loop_time()
        reach = self._wall_ref + (tl.time_of(self._idx + 1) - self._t_ref) / self.speed + _EPS
        return max(reach, self._pace_wall)

    def _advance_pace(self) -> None:
        """Called when a frame goes out: step the pacing grid, resyncing (no catch-up burst) if we fell behind."""
        now = self._loop_time()
        interval = 1.0 / self.max_fps
        if self._pace_wall == float("-inf"):
            self._pace_wall = now + interval
        else:
            self._pace_wall = max(self._pace_wall + interval, now + interval / 2)

    async def _tick(self) -> None:
        tl = self.timeline
        now = self._loop_time()
        t = min(self._now_t(now), tl.duration_s)
        idx = tl.index_at(t)
        if idx > self._idx:
            await self._emit_frame(idx)
        if self.playing and self._idx >= tl.n_frames - 1:
            self.playing = False
            self._rebase(now, tl.time_of(self._idx))
            await self._send_clock()

    async def _emit_frame(self, idx: int) -> None:
        """Send frame ``idx`` then every event / moment up to its time (cursors only move forward here)."""
        tl = self.timeline
        assert self._send is not None
        self._idx = idx
        self._advance_pace()
        await self._send(tl.frame_json(idx))
        self.frames_sent += 1
        t_frame = tl.time_of(idx)
        hi = bisect.bisect_right(tl.event_times, t_frame)
        if hi > self._event_cursor:
            batch = tl.events[self._event_cursor : hi]
            self._event_cursor = hi
            await self._send(EventsMsg(events=batch).model_dump_json())
        hi = bisect.bisect_right(tl.moment_times, t_frame)
        while self._moment_cursor < hi:
            moment = tl.moments[self._moment_cursor]
            self._moment_cursor += 1
            await self._send(MomentMsg(moment=moment).model_dump_json())

    async def _send_clock(self) -> None:
        assert self._send is not None
        await self._send(ClockMsg(clock=self.clock_status()).model_dump_json())

    async def _error(self, code: str, message: str) -> None:
        assert self._send is not None
        await self._send(ErrorMsg(code=code, message=message).model_dump_json())

    # --- commands -------------------------------------------------------------------------------------------

    @staticmethod
    def _parse(text: str) -> ClientCmd | None:
        try:
            return ClientMessageAdapter.validate_json(text)
        except ValidationError:
            return None

    def _coalesce(self, cmd: SeekCmd | SpeedCmd) -> SeekCmd | SpeedCmd:
        """Fold already-queued consecutive commands of the same kind into the last one."""
        while not self.inbox.empty():
            nxt = self.inbox.get_nowait()
            if isinstance(nxt, _Raw):
                parsed = self._parse(nxt.text)
                if type(parsed) is type(cmd) and isinstance(parsed, SeekCmd | SpeedCmd):
                    cmd = parsed
                    continue
            self._carry = nxt
            break
        return cmd

    async def _handle_raw(self, text: str) -> None:
        try:
            cmd = ClientMessageAdapter.validate_json(text)
        except ValidationError as exc:
            first = exc.errors()[0]
            kind = first["type"]
            if kind in ("json_invalid", "json_type"):
                await self._error("invalid_json", "message is not a JSON object")
            else:
                where = ".".join(str(p) for p in first["loc"])
                await self._error("invalid_message", f"{where}: {first['msg']}"[:300])
            return
        if isinstance(cmd, SeekCmd | SpeedCmd):
            cmd = self._coalesce(cmd)
        now = self._loop_time()
        tl = self.timeline
        if isinstance(cmd, PlayCmd):
            if self.playing:
                await self._send_clock()
                return
            if self._idx >= tl.n_frames - 1:
                await self._seek(0.0, now, play=True)
                return
            self.playing = True
            self._rebase(now, tl.time_of(self._idx))
            self._pace_wall = float("-inf")
            await self._send_clock()
        elif isinstance(cmd, PauseCmd):
            if self.playing:
                self.playing = False
                self._rebase(now, tl.time_of(self._idx))
            await self._send_clock()
        elif isinstance(cmd, SeekCmd):
            await self._seek(cmd.t, now, play=self.playing)
        elif isinstance(cmd, SpeedCmd):
            if self.playing:
                self._rebase(now, min(self._now_t(now), tl.duration_s))
            self.speed = cmd.speed
            await self._send_clock()
        elif isinstance(cmd, ProfileCmd):
            if cmd.profile.language not in self.languages:
                await self._error(
                    "unsupported_language",
                    f"language {cmd.profile.language!r} is not supported; choose one of {', '.join(self.languages)}",
                )
                return
            self.profile = cmd.profile.model_copy(update={"session_id": self.id})

    async def _seek(self, t: float, now: float, play: bool) -> None:
        tl = self.timeline
        idx = tl.index_at(min(t, tl.duration_s))
        t_frame = tl.time_of(idx)
        self._rebase(now, t_frame)
        self.playing = play and idx < tl.n_frames - 1
        self._pace_wall = float("-inf")
        if idx == 0:
            # Back to the very start: rewind the cursors so events at t <= 0 (kickoff) go out with this frame.
            self._event_cursor = self._moment_cursor = 0
            self._idx = 0
            await self._send_clock()
            await self._emit_frame(0)
            return
        self._event_cursor = bisect.bisect_right(tl.event_times, t_frame)
        self._moment_cursor = bisect.bisect_right(tl.moment_times, t_frame)
        self._idx = idx
        await self._send_clock()
        assert self._send is not None
        self._advance_pace()
        await self._send(tl.frame_json(idx))
        self.frames_sent += 1
