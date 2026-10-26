"""FastAPI application: REST for match data and rollouts, one WebSocket endpoint for the live stream.

    app = create_app(recording_dir, engine)

``engine`` is anything implementing ``shared.engine_api.RolloutEngine`` (``server.fake_engine.FakeEngine`` in
tests; the simulator's engine in production). It is held on ``app.state.engine`` and read by the
``get_engine`` dependency, so tests can also use ``app.dependency_overrides[get_engine]``.
"""

import asyncio
import logging
import math
import secrets
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from server.config import Settings
from server.engine import UnknownSnapshotError, parse_snapshot_ref
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


def get_engine(request: Request) -> RolloutEngine:
    engine: RolloutEngine = request.app.state.engine
    return engine


def get_timeline(request: Request) -> Timeline:
    timeline: Timeline = request.app.state.timeline
    return timeline


EngineDep = Annotated[RolloutEngine, Depends(get_engine)]
TimelineDep = Annotated[Timeline, Depends(get_timeline)]


def _call_engine[T](timeline: Timeline, snapshot_ref: str, fn: Callable[[], T]) -> T:
    """Validate the snapshot ref against this match, call the engine, map its errors to HTTP statuses."""
    try:
        match_id, _ = parse_snapshot_ref(snapshot_ref)
    except UnknownSnapshotError as exc:
        raise HTTPException(404, str(exc.args[0])) from exc
    if match_id != timeline.match_id:
        raise HTTPException(404, f"snapshot {snapshot_ref!r} is not from match {timeline.match_id!r}")
    try:
        return fn()
    except KeyError as exc:
        raise HTTPException(404, f"unknown snapshot: {exc.args[0] if exc.args else snapshot_ref}") from exc
    except ValidationError:
        raise
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


def create_app(recording_dir: Path, engine: RolloutEngine, settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    timeline = Timeline.open(Path(recording_dir), settings.frame_cache_size)
    app = FastAPI(title="ReGolazo live engine", version=SCHEMA_VERSION)
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
    def healthz(tl: TimelineDep) -> Health:
        return Health(
            status="ok",
            schema_version=SCHEMA_VERSION,
            match_id=tl.match_id,
            n_frames=tl.n_frames,
            duration_s=tl.duration_s,
            sessions=len(app.state.sessions),
        )

    @app.get("/api/meta", response_model=MatchMeta)
    def get_meta(tl: TimelineDep) -> MatchMeta:
        return tl.meta

    @app.get("/api/events", response_model=list[MatchEvent])
    def get_events(
        tl: TimelineDep,
        t_from: Annotated[float, Query(alias="from", ge=0)] = 0.0,
        t_to: Annotated[float, Query(alias="to", ge=0)] = math.inf,
    ) -> list[MatchEvent]:
        if t_to < t_from:
            raise HTTPException(422, "'to' must be >= 'from'")
        return tl.events_between(t_from, t_to)

    @app.get("/api/moments", response_model=list[DecisionMoment])
    def get_moments(tl: TimelineDep) -> list[DecisionMoment]:
        return tl.moments

    @app.get("/api/state", response_model=SimState)
    def get_state(tl: TimelineDep, t: Annotated[float, Query(ge=0)]) -> SimState:
        """The frame at or before match time ``t`` (clamped to the match)."""
        return tl.state_at(tl.index_at(t))

    @app.post("/api/preview", response_model=ResolvedAction)
    def post_preview(body: PreviewRequest, eng: EngineDep, tl: TimelineDep) -> ResolvedAction:
        return _call_engine(tl, body.snapshot_ref, lambda: eng.preview(body.snapshot_ref, body.intent))

    @app.get("/api/options", response_model=ModelOutputs)
    def get_options(snapshot_ref: str, eng: EngineDep, tl: TimelineDep) -> ModelOutputs:
        return _call_engine(tl, snapshot_ref, lambda: eng.options(snapshot_ref))

    @app.post("/api/rollout", response_model=RolloutResult)
    def post_rollout(body: RolloutRequest, eng: EngineDep, tl: TimelineDep) -> RolloutResult:
        return _call_engine(tl, body.snapshot_ref, lambda: eng.rollout(body))

    @app.post("/api/distribution", response_model=RolloutDistribution)
    def post_distribution(body: RolloutRequest, eng: EngineDep, tl: TimelineDep) -> RolloutDistribution:
        return _call_engine(tl, body.snapshot_ref, lambda: eng.distribution(body))

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
                    session.receive(text)
            except Exception:  # connection reset etc.: treat as a disconnect
                log.debug("reader ended", exc_info=True)
            finally:
                session.close()

        reader_task = asyncio.create_task(reader())
        try:
            await session.run(ws.send_text)
        except Exception:  # client went away mid-send
            log.debug("session %s ended", session_id, exc_info=True)
        finally:
            reader_task.cancel()
            app.state.sessions.pop(session_id, None)
            try:
                await ws.close()
            except Exception:
                log.debug("close failed", exc_info=True)

    # --- static frontend (single-container deploy) --------------------------------------------------------

    if settings.static_dir is not None and settings.static_dir.is_dir():
        app.mount("/", StaticFiles(directory=settings.static_dir, html=True), name="web")

    return app
