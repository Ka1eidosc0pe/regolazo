"""Where raw data and artifacts live.

Raw downloads are large and gitignored. They live under ``$REGOLAZO_DATA_DIR/raw/`` so several worktrees can share
one cache; the default is ``<repo>/data``.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = REPO_ROOT / "analytics" / "artifacts"
DOCS_IMG_DIR = REPO_ROOT / "docs" / "img"


def data_dir() -> Path:
    return Path(os.environ.get("REGOLAZO_DATA_DIR", str(REPO_ROOT / "data")))


def raw_dir() -> Path:
    return data_dir() / "raw"


def statsbomb_dir() -> Path:
    return raw_dir() / "statsbomb"


def metrica_dir() -> Path:
    return raw_dir() / "metrica"
