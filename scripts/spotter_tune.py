"""Grid-search the event-rule thresholds on a validation-match segment (candidate-2021), never the held-out match.

uv run python scripts/spotter_tune.py
"""

import itertools
import json
import os
from pathlib import Path

import vision.events as ev
from shared.recording import Recording
from vision.evaluate import evaluate
from vision.pipeline import default_weights, load_results, run_frames, save_results
from vision.tracking import Spotter, kits_from_meta, load_frame_meta, to_tracks

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("REGOLAZO_DATA_DIR", REPO / "data"))
MATCH = "candidate-2021"
RENDER = DATA / "vision" / "renders" / "candidate-2021-2000-2090-broadcast-25fps"
GRID = {
    "POSSESSION_RADIUS_M": [1.0, 1.5, 2.0],
    "MIN_CONTROL_FRAMES": [2, 3, 5],
    "MIN_PASS_TRAVEL_M": [3.0, 4.0, 6.0],
}


def main() -> None:
    cache = DATA / "vision" / "tune" / "detections.json"
    if not cache.exists():
        save_results(run_frames(Spotter(default_weights(DATA), fps=25), load_frame_meta(RENDER)), cache)
    rec = Recording.open(DATA / "matches" / MATCH)
    tracks = to_tracks(load_results(cache), kits_from_meta(rec.meta))
    rows = []
    for values in itertools.product(*GRID.values()):
        for k, v in zip(GRID, values, strict=True):
            setattr(ev, k, v)
        events, runs = ev.detect(tracks, 1, rec.meta.home_attacks_positive_x_first_half)
        r = evaluate(rec, events, runs, tracks, tol=1.0)
        f1 = {k: s.f1 for k, s in r.scores.items()}
        rows.append(
            {**dict(zip(GRID, values, strict=True)), **f1, "objective": (f1["pass"] + f1["possession_change"]) / 2}
        )
    rows.sort(key=lambda x: -x["objective"])
    for row in rows[:5]:
        print(json.dumps(row))
    (DATA / "vision" / "tune" / "grid.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
