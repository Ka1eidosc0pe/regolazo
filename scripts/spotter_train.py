"""Fine-tune YOLO11n on the rendered Spotter dataset (MPS), then evaluate on the held-out demo match.

    uv run python scripts/spotter_train.py train [--hours 0.28]
    uv run python scripts/spotter_train.py test   # per-class mAP on candidate-2004 -> detector_metrics.json

Weights: $REGOLAZO_DATA_DIR/vision/runs/spotter/weights/best.pt (Ultralytics, AGPL-3.0).
"""

import argparse
import json
import os
from pathlib import Path

from ultralytics import YOLO

from vision.dataset import CLASSES

REPO = Path(__file__).resolve().parents[1]
VISION = Path(os.environ.get("REGOLAZO_DATA_DIR", REPO / "data")) / "vision"
DATA_YAML = VISION / "yolo" / "data.yaml"
RUNS = VISION / "runs"
WEIGHTS = RUNS / "spotter" / "weights" / "best.pt"


def train(hours: float, imgsz: int) -> None:
    model = YOLO("yolo11n.pt")
    model.train(
        data=str(DATA_YAML),
        imgsz=imgsz,
        epochs=60,
        time=hours,
        batch=16,
        device="mps",
        workers=4,
        project=str(RUNS),
        name="spotter",
        exist_ok=True,
        close_mosaic=3,
        patience=100,
        plots=False,
        verbose=False,
    )


def test(imgsz: int) -> None:
    model = YOLO(str(WEIGHTS))
    m = model.val(
        data=str(DATA_YAML), split="test", imgsz=imgsz, device="mps", plots=False, project=str(RUNS), name="test"
    )
    per_class = {}
    for i, c in enumerate(m.box.ap_class_index):
        name = CLASSES[int(c)]
        per_class[name] = {"map50": float(m.box.ap50[i]), "map50_95": float(m.box.ap[i])}
    out = {"match": "candidate-2004", "map50": float(m.box.map50), "map50_95": float(m.box.map), "per_class": per_class}
    (VISION / "detector_metrics.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "test"])
    ap.add_argument("--hours", type=float, default=0.28)
    ap.add_argument("--imgsz", type=int, default=960)
    a = ap.parse_args()
    if a.cmd == "train":
        train(a.hours, a.imgsz)
    else:
        test(a.imgsz)


if __name__ == "__main__":
    main()
