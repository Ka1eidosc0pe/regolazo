"""Render the Spotter dataset: frames + FrameLabels from recorded matches, then a YOLO-format dataset.

    uv run python scripts/spotter_dataset.py render     # clips -> web/scripts/record.ts -> $DATA/vision/renders/*
    uv run python scripts/spotter_dataset.py yolo       # renders -> $DATA/vision/yolo/{images,labels}/{train,val,test}

Split by match: train early-0007, candidate-2012, candidate-2020; val candidate-2021; test candidate-2004 (demo match,
held out). The demo segment of candidate-2004 is rendered at 25 fps for tracking and event evaluation.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from vision.clips import export_clip
from vision.dataset import Window, labels_to_yolo

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("REGOLAZO_DATA_DIR", REPO / "data"))
VISION = DATA / "vision"
CLIPS = REPO / "web" / "public" / "spotter"

TRAIN = ["early-0007", "candidate-2012", "candidate-2020"]
VAL = ["candidate-2021"]
DEMO_MATCH = "candidate-2004"
DEMO_START, DEMO_END = 1180.0, 1300.0

WINDOWS: list[Window] = []
for _m in TRAIN + VAL:
    _split = "train" if _m in TRAIN else "val"
    for _k, (_t0, _cam) in enumerate([(420, "broadcast"), (1500, "replay"), (3300, "broadcast"), (4700, "replay")]):
        WINDOWS.append(Window(_m, float(_t0), float(_t0 + 40), _cam, 4, _split))
WINDOWS.append(Window(DEMO_MATCH, DEMO_START, DEMO_END, "broadcast", 25, "test"))
WINDOWS.append(Window(DEMO_MATCH, 2890.0, 2920.0, "replay", 4, "test"))
# second-half event-evaluation segment (two shots); not part of the YOLO test split
WINDOWS.append(Window(DEMO_MATCH, 4060.0, 4120.0, "broadcast", 25, "eval"))


def render(only: str | None) -> None:
    for w in WINDOWS:
        if only and w.match != only:
            continue
        out = VISION / "renders" / w.name
        if (out / "manifest.json").exists():
            print("skip", out)
            continue
        clip = export_clip(DATA, w.match, w.start - 3.0, w.end + 1.0, CLIPS)
        cmd = [
            "node",
            "scripts/record.ts",
            "--clip",
            f"/spotter/{clip.name}",
            "--out",
            str(out),
            "--fps",
            str(w.fps),
            "--width",
            "960",
            "--height",
            "540",
            "--start",
            str(w.start),
            "--end",
            str(w.end),
            "--camera",
            w.camera,
        ]
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=REPO / "web", check=True)
        clip.unlink()


def yolo(splits: list[str]) -> None:
    root = VISION / "yolo"
    counts: dict[str, int] = {}
    for w in WINDOWS:
        if w.split not in splits:
            continue
        src = VISION / "renders" / w.name
        if not (src / "manifest.json").exists():
            print("missing", src, file=sys.stderr)
            continue
        stride = 5 if w.fps == 25 else 1  # thin the 25 fps demo segment for detection metrics
        counts[w.split] = counts.get(w.split, 0) + labels_to_yolo(src, root, w.split, w.name, stride)
    (root / "data.yaml").write_text(
        f"path: {root}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "names:\n  0: player_home\n  1: player_away\n  2: goalkeeper\n  3: referee\n  4: ball\n"
    )
    print(json.dumps(counts))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["render", "yolo"])
    ap.add_argument("--match")
    ap.add_argument("--split", default="train,val,test")
    a = ap.parse_args()
    if a.cmd == "render":
        render(a.match)
    else:
        yolo(a.split.split(","))


if __name__ == "__main__":
    main()
