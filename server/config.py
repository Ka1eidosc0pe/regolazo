"""Server configuration, read from the environment (all keys prefixed ``REGOLAZO_``)."""

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LANGUAGES = ("en", "es", "fr")
DEFAULT_CORS_ORIGINS = ("http://localhost:5173", "http://127.0.0.1:5173")


def data_dir() -> Path:
    return Path(os.environ.get("REGOLAZO_DATA_DIR", REPO_ROOT / "data"))


def host_port() -> tuple[str, int]:
    """Bind address from ``REGOLAZO_HOST`` (default 127.0.0.1; containers set 0.0.0.0) and ``REGOLAZO_PORT``."""
    return os.environ.get("REGOLAZO_HOST", "127.0.0.1"), int(os.environ.get("REGOLAZO_PORT", "8000"))


def _csv(value: str | None, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None or not value.strip():
        return default
    return tuple(part.strip() for part in value.split(",") if part.strip())


@dataclass(frozen=True)
class Settings:
    languages: tuple[str, ...] = DEFAULT_LANGUAGES
    """Supported viewer languages (configuration, not an enum): ``REGOLAZO_LANGUAGES=en,es,fr``."""
    cors_origins: tuple[str, ...] = DEFAULT_CORS_ORIGINS
    static_dir: Path | None = None
    """Built frontend (``web/dist``) served at ``/`` when set and present: single-container deploy."""
    max_fps: float = 25.0
    """Upper bound on frames per second pushed to one session (frames are skipped above this)."""
    frame_cache_size: int = 4096
    """Serialized frames kept in memory, shared by all sessions."""

    engine_concurrency: int = 2
    """Worker threads for engine calls (preview / options / rollout / distribution)."""
    engine_thread_safe: bool = False
    """False (default): engine calls are serialized with a lock. True: up to ``engine_concurrency`` run at once."""
    result_cache_size: int = 32
    """Rollout / distribution results kept for repeat requests (only requests with an explicit seed are cached)."""

    @classmethod
    def from_env(cls) -> "Settings":
        static = os.environ.get("REGOLAZO_STATIC_DIR")
        return cls(
            languages=_csv(os.environ.get("REGOLAZO_LANGUAGES"), DEFAULT_LANGUAGES),
            cors_origins=_csv(os.environ.get("REGOLAZO_CORS_ORIGINS"), DEFAULT_CORS_ORIGINS),
            static_dir=Path(static) if static else None,
            max_fps=float(os.environ.get("REGOLAZO_MAX_FPS", cls.max_fps)),
            frame_cache_size=int(os.environ.get("REGOLAZO_FRAME_CACHE", cls.frame_cache_size)),
            engine_concurrency=int(os.environ.get("REGOLAZO_ENGINE_CONCURRENCY", cls.engine_concurrency)),
            engine_thread_safe=os.environ.get("REGOLAZO_ENGINE_THREAD_SAFE", "").lower() in ("1", "true", "yes"),
            result_cache_size=int(os.environ.get("REGOLAZO_RESULT_CACHE", cls.result_cache_size)),
        )
