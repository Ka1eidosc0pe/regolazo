"""Export a time window of a recorded match as a renderer clip ({meta, frames}).

    uv run python scripts/spotter_clips.py --match candidate-2004 --start 600 --end 720 --out web/public/spotter

Writes <out>/<match>-<start>-<end>.json; the renderer loads it with ?clip=/spotter/<name>.json.
"""

import argparse
import os
from pathlib import Path

from vision.clips import export_clip

DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", required=True)
    ap.add_argument("--start", type=float, required=True)
    ap.add_argument("--end", type=float, required=True)
    ap.add_argument("--out", type=Path, default=Path("web/public/spotter"))
    a = ap.parse_args()
    data_dir = Path(os.environ.get("REGOLAZO_DATA_DIR", DEFAULT_DATA))
    print(export_clip(data_dir, a.match, a.start, a.end, a.out))


if __name__ == "__main__":
    main()
