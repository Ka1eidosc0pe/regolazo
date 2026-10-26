"""Server configuration, read from the environment (all keys prefixed ``REGOLAZO_``)."""

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LANGUAGES = ("en", "es", "fr")
DEFAULT_CORS_ORIGINS = ("http://localhost:5173", "http://127.0.0.1:5173")


def data_dir() -> Path:
    return Path(os.environ.get("REGOLAZO_DATA_DIR", REPO_ROOT / "data"))


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

    @classmethod
    def from_env(cls) -> "Settings":
        static = os.environ.get("REGOLAZO_STATIC_DIR")
        return cls(
            languages=_csv(os.environ.get("REGOLAZO_LANGUAGES"), DEFAULT_LANGUAGES),
            cors_origins=_csv(os.environ.get("REGOLAZO_CORS_ORIGINS"), DEFAULT_CORS_ORIGINS),
            static_dir=Path(static) if static else None,
        )
