"""FastAPI application: REST for match data and rollouts, one WebSocket endpoint for the live stream.

    app = create_app(recording_dir, engine)

``engine`` is anything implementing ``shared.engine_api.RolloutEngine`` (``server.fake_engine.FakeEngine`` in
tests; the simulator's engine in production). It is held on ``app.state.engine`` and read by the
``get_engine`` dependency, so tests can also use ``app.dependency_overrides[get_engine]``.

Cheap routes (healthz, meta, events, moments, state) are ``async def`` and never touch a thread pool. Engine calls
run on ``EngineRunner``'s own pool (serialized by default, see ``Settings.engine_thread_safe``), so a slow rollout
cannot stall them.
"""

import asyncio
import contextlib
import logging
import math
import secrets
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError
from starlette.websockets import WebSocketDisconnect, WebSocketState
from websockets.exceptions import ConnectionClosed

from server.config import Settings
from server.engine import UnknownSnapshotError, parse_snapshot_ref
from server.engine_runner import EngineRunner
from server.session import Session
from server.timeline import Timeline
from shared.engine_api import RolloutEngine
from shared.schemas.actions import ActionIntent, ResolvedAction
from shared.schemas.common import SCHEMA_VERSION, Contract
from shared.schemas.events import MatchEvent
from shared.schemas.meta import MatchMeta
from shared.schemas.models import ModelOutputs
from shared.schemas.moments import DecisionMoment
from shared.schemas.narrative import ViewerProfile
from shared.schemas.rollout import RolloutDistribution, RolloutRequest, RolloutResult
from shared.schemas.state import SimState
from shared.schemas.stream import ErrorMsg

log = logging.getLogger("regolazo.server")


class PreviewRequest(Contract):
    snapshot_ref: str
    intent: ActionIntent


class Health(Contract):
    status: str
    schema_version: str
    match_id: str
    n_frames: int
    duration_s: float
    sessions: int


async def get_engine(request: Request) -> RolloutEngine:
    engine: RolloutEngine = request.app.state.engine
    return engine


async def get_timeline(request: Request) -> Timeline:
    timeline: Timeline = request.app.state.timeline
    return timeline


async def get_runner(request: Request) -> EngineRunner:
    runner: EngineRunner = request.app.state.engine_runner
    return runner


EngineDep = Annotated[RolloutEngine, Depends(get_engine)]
RunnerDep = Annotated[EngineRunner, Depends(get_runner)]
TimelineDep = Annotated[Timeline, Depends(get_timeline)]


def _check_snapshot(timeline: Timeline, snapshot_ref: str) -> None:
    """404 unless the ref names a keyframe of this recording (applies to every engine)."""
    try:
        match_id, tick = parse_snapshot_ref(snapshot_ref)
    except UnknownSnapshotError as exc:
        raise HTTPException(404, str(exc.args[0])) from exc
    if match_id != timeline.match_id:
        raise HTTPException(404, f"snapshot {snapshot_ref!r} is not from match {timeline.match_id!r}")
    if tick not in timeline.keyframe_ticks:
        raise HTTPException(404, f"snapshot {snapshot_ref!r}: tick {tick} is not a keyframe of this recording")


async def _call_engine[T](
    runner: EngineRunner,
    timeline: Timeline,
    snapshot_ref: str,
    fn: Callable[[], T],
    cache_key: str | None = None,
) -> T:
    """Validate the snapshot ref, run the engine call off the event loop, map engine errors to HTTP statuses."""
    _check_snapshot(timeline, snapshot_ref)
    try:
        if cache_key is None:
            return await runner.run(fn)
        return await runner.run_cached(cache_key, fn)  # type: ignore[type-var]
    except KeyError as exc:
        raise HTTPException(404, f"unknown snapshot: {exc.args[0] if exc.args else snapshot_ref}") from exc
    except ValidationError:
        raise
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


def _is_disconnect(exc: BaseException, ws: WebSocket) -> bool:
    if isinstance(exc, WebSocketDisconnect | ConnectionClosed):
        return True
    return isinstance(exc, RuntimeError) and (
        ws.client_state == WebSocketState.DISCONNECTED or ws.application_state == WebSocketState.DISCONNECTED
    )


def create_app(recording_dir: Path, engine: RolloutEngine, settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    timeline = Timeline.open(Path(recording_dir), settings.frame_cache_size)
    runner = EngineRunner(settings.engine_concurrency, settings.engine_thread_safe, settings.result_cache_size)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        runner.close()

    app = FastAPI(title="ReGolazo live engine", version=SCHEMA_VERSION, lifespan=lifespan)
    app.state.engine_runner = runner
    app.state.timeline = timeline
    app.state.engine = engine
    app.state.settings = settings
    app.state.sessions = {}  # session_id -> Session, for other components to push cues / narratives

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- REST ---------------------------------------------------------------------------------------------

    @app.get("/healthz", response_model=Health)
    async def healthz(tl: TimelineDep) -> Health:
        return Health(
            status="ok",
            schema_version=SCHEMA_VERSION,
            match_id=tl.match_id,
            n_frames=tl.n_frames,
            duration_s=tl.duration_s,
            sessions=len(app.state.sessions),
        )

    @app.get("/api/meta", response_model=MatchMeta)
    async def get_meta(tl: TimelineDep) -> MatchMeta:
        return tl.meta

    @app.get("/api/events", response_model=list[MatchEvent])
    async def get_events(
        tl: TimelineDep,
        t_from: Annotated[float, Query(alias="from", ge=0)] = 0.0,
        t_to: Annotated[float, Query(alias="to", ge=0)] = math.inf,
    ) -> list[MatchEvent]:
        if t_to < t_from:
            raise HTTPException(422, "'to' must be >= 'from'")
        return tl.events_between(t_from, t_to)

    @app.get("/api/moments", response_model=list[DecisionMoment])
    async def get_moments(tl: TimelineDep) -> list[DecisionMoment]:
        return tl.moments

    @app.get("/api/state", response_model=SimState)
    async def get_state(tl: TimelineDep, t: Annotated[float, Query(ge=0)]) -> SimState:
        """The frame at or before match time ``t`` (clamped to the match)."""
        return tl.state_at(tl.index_at(t))

    @app.post("/api/preview", response_model=ResolvedAction)
    async def post_preview(body: PreviewRequest, eng: EngineDep, run: RunnerDep, tl: TimelineDep) -> ResolvedAction:
        return await _call_engine(run, tl, body.snapshot_ref, lambda: eng.preview(body.snapshot_ref, body.intent))

    @app.get("/api/options", response_model=ModelOutputs)
    async def get_options(snapshot_ref: str, eng: EngineDep, run: RunnerDep, tl: TimelineDep) -> ModelOutputs:
        return await _call_engine(run, tl, snapshot_ref, lambda: eng.options(snapshot_ref))

    @app.post("/api/rollout", response_model=RolloutResult)
    async def post_rollout(body: RolloutRequest, eng: EngineDep, run: RunnerDep, tl: TimelineDep) -> RolloutResult:
        key = None if body.seed is None else "rollout:" + body.model_dump_json()
        return await _call_engine(run, tl, body.snapshot_ref, lambda: eng.rollout(body), key)

    @app.post("/api/distribution", response_model=RolloutDistribution)
    async def post_distribution(
        body: RolloutRequest, eng: EngineDep, run: RunnerDep, tl: TimelineDep
    ) -> RolloutDistribution:
        key = None if body.seed is None else "distribution:" + body.model_dump_json()
        return await _call_engine(run, tl, body.snapshot_ref, lambda: eng.distribution(body), key)

    # --- WebSocket ----------------------------------------------------------------------------------------

    @app.websocket("/ws")
    async def stream(
        ws: WebSocket,
        language: str | None = None,
        mode: str | None = None,
        club: str | None = None,
    ) -> None:
        """Live stream. Optional query params seed the ViewerProfile: ``language``, ``mode``, ``club``."""
        await ws.accept()
        languages = settings.languages
        session_id = f"sess_{secrets.token_hex(4)}"
        default_language = "en" if "en" in languages else languages[0]
        problems: list[ErrorMsg] = []
        if language is not None and language not in languages:
            msg = f"language {language!r} is not supported; choose one of {', '.join(languages)}"
            problems.append(ErrorMsg(code="unsupported_language", message=msg))
            language = None
        if mode not in (None, "casual", "analyst"):
            problems.append(ErrorMsg(code="invalid_query", message=f"mode {mode!r} must be casual or analyst"))
            mode = None
        if club not in (None, "home", "away"):
            problems.append(ErrorMsg(code="invalid_query", message=f"club {club!r} must be home or away"))
            club = None
        profile = ViewerProfile.model_validate(
            {
                "session_id": session_id,
                "mode": mode or "casual",
                "favorite_club": club,
                "language": language or default_language,
            }
        )
        session = Session(session_id, timeline, profile, languages, settings.max_fps)
        for problem in problems:
            session.push(problem)  # delivered right after hello and the first frame
        app.state.sessions[session_id] = session

        async def reader() -> None:
            try:
                while True:
                    message = await ws.receive()
                    if message["type"] == "websocket.disconnect":
                        return
                    text = message.get("text")
                    if text is None:
                        text = (message.get("bytes") or b"").decode("utf-8", "replace")
                    await session.receive(text)
            except Exception as exc:
                if not _is_disconnect(exc, ws):
                    log.exception("session %s: reader failed", session_id)
            finally:
                session.close()

        reader_task = asyncio.create_task(reader())
        close_code = 1000
        try:
            await session.run(ws.send_text)
        except Exception as exc:
            if not _is_disconnect(exc, ws):
                log.exception("session %s: server error, closing with 1011", session_id)
                close_code = 1011
                with contextlib.suppress(Exception):
                    await ws.send_text(
                        ErrorMsg(code="internal_error", message="the server failed while streaming").model_dump_json()
                    )
        finally:
            reader_task.cancel()
            app.state.sessions.pop(session_id, None)
            with contextlib.suppress(Exception):
                await ws.close(code=close_code)

    # --- static frontend (single-container deploy) --------------------------------------------------------

    if settings.static_dir is not None and settings.static_dir.is_dir():
        app.mount("/", StaticFiles(directory=settings.static_dir, html=True), name="web")

    return app
