"""Entry points.

    uv run python -m server --fixture                # write the synthetic demo recording, then serve it
    uv run python -m server --match demo-0001        # serve $REGOLAZO_DATA_DIR/matches/demo-0001
    uv run uvicorn server.main:app_factory --factory # same, configured by environment variables

Environment: REGOLAZO_DATA_DIR, REGOLAZO_MATCH_ID, REGOLAZO_RECORDING_DIR, REGOLAZO_ENGINE ("fake" or
"module:factory", the factory takes the match id and returns a RolloutEngine), REGOLAZO_LANGUAGES,
REGOLAZO_CORS_ORIGINS, REGOLAZO_STATIC_DIR (built web app), REGOLAZO_HOST / REGOLAZO_PORT, REGOLAZO_MAX_FPS,
REGOLAZO_FRAME_CACHE, REGOLAZO_ENGINE_CONCURRENCY, REGOLAZO_ENGINE_THREAD_SAFE, REGOLAZO_RESULT_CACHE.
"""

import argparse
import json
import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI

from server.app import create_app
from server.config import data_dir, host_port
from server.engine import load_engine
from server.fixture import base_match_id, write_fixture_recording


def resolve_recording_dir(match_id: str | None = None, recording_dir: str | None = None) -> Path:
    explicit = recording_dir or os.environ.get("REGOLAZO_RECORDING_DIR")
    if explicit:
        return Path(explicit)
    matches = data_dir() / "matches"
    match_id = match_id or os.environ.get("REGOLAZO_MATCH_ID")
    if match_id:
        return matches / match_id
    candidates = sorted(p for p in matches.glob("*/manifest.json")) if matches.is_dir() else []
    if not candidates:
        raise SystemExit(f"no recordings under {matches}; pass --match / --recording or run with --fixture")
    return candidates[-1].parent


def recording_match_id(directory: Path) -> str:
    """The match id recorded in the manifest (the directory name is only a convention)."""
    return str(json.loads((directory / "manifest.json").read_text())["match_id"])


def build_app(directory: Path, engine_spec: str) -> FastAPI:
    return create_app(directory, load_engine(engine_spec, recording_match_id(directory)))


def app_factory() -> FastAPI:
    """uvicorn ``--factory`` entry: everything from environment variables."""
    return build_app(resolve_recording_dir(), os.environ.get("REGOLAZO_ENGINE", "fake"))


def main(argv: list[str] | None = None) -> None:
    default_host, default_port = host_port()
    ap = argparse.ArgumentParser(prog="python -m server", description=__doc__)
    ap.add_argument("--match", help="match id under $REGOLAZO_DATA_DIR/matches")
    ap.add_argument("--recording", help="recording directory (overrides --match)")
    ap.add_argument("--engine", default=os.environ.get("REGOLAZO_ENGINE", "fake"), help="'fake' or module:factory")
    ap.add_argument("--fixture", action="store_true", help="write the synthetic demo recording first and serve it")
    ap.add_argument("--host", default=default_host, help="default $REGOLAZO_HOST or 127.0.0.1")
    ap.add_argument("--port", type=int, default=default_port, help="default $REGOLAZO_PORT or 8000")
    args = ap.parse_args(argv)
    if args.fixture:
        directory = write_fixture_recording(data_dir() / "matches" / base_match_id())
    else:
        directory = resolve_recording_dir(args.match, args.recording)
    uvicorn.run(build_app(directory, args.engine), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
