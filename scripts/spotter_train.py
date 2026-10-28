"""Fine-tune YOLO11n on the rendered Spotter dataset (MPS), then evaluate on the held-out demo match.

    uv run python scripts/spotter_train.py train [--hours 0.28]
    uv run python scripts/spotter_train.py test   # per-class mAP on candidate-2004 -> detector_metrics.json

Weights: $REGOLAZO_DATA_DIR/vision/runs/spotter/weights/best.pt (Ultralytics, AGPL-3.0).
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from vision.dataset import CLASSES

REPO = Path(__file__).resolve().parents[1]
VISION = Path(os.environ.get("REGOLAZO_DATA_DIR", REPO / "data")) / "vision"
DATA_YAML = VISION / "yolo" / "data.yaml"
RUNS = VISION / "runs"


def train(hours: float, imgsz: int, init: str = "yolo11n.pt", name: str = "spotter") -> None:
    model = YOLO(init)
    model.train(
        data=str(DATA_YAML),
        imgsz=imgsz,
        epochs=60,
        time=hours,
        batch=16,
        device="mps",
        workers=4,
        project=str(RUNS),
        name=name,
        exist_ok=True,
        close_mosaic=3,
        patience=100,
        plots=False,
        verbose=False,
    )


def test(imgsz: int, name: str = "spotter") -> None:
    model = YOLO(str(RUNS / name / "weights" / "best.pt"))
    m = model.val(
        data=str(DATA_YAML), split="test", imgsz=imgsz, device="mps", plots=False, project=str(RUNS), name="test"
    )
    per_class = {}
    for i, c in enumerate(m.box.ap_class_index):
        cls_name = CLASSES[int(c)]
        per_class[cls_name] = {"map50": float(m.box.ap50[i]), "map50_95": float(m.box.ap[i])}
    out = {
        "match": "candidate-2004",
        "images": len(list((VISION / "yolo" / "images" / "test").glob("*.jpg"))),
        "map50": float(m.box.map50),
        "map50_95": float(m.box.map),
        "per_class": per_class,
        "player_any_team_ap50": player_ap50(model, imgsz),
    }
    suffix = "" if name == "spotter" else f"_{name}"
    (VISION / f"detector_metrics{suffix}.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between boxes a (N, 4) and b (M, 4), xyxy."""
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    out: np.ndarray = inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)
    return out


def player_ap50(model: YOLO, imgsz: int) -> float:
    """AP50 for outfield players with home/away merged (the team comes from kit colour downstream)."""
    scored: list[tuple[float, bool]] = []
    n_gt = 0
    for img in sorted((VISION / "yolo" / "images" / "test").glob("*.jpg")):
        txt = img.parents[2] / "labels" / "test" / f"{img.stem}.txt"
        lab = np.loadtxt(txt, ndmin=2) if txt.stat().st_size else np.zeros((0, 5))
        r = model.predict(str(img), imgsz=imgsz, conf=0.001, device="mps", verbose=False)[0]
        w, h = r.orig_shape[1], r.orig_shape[0]
        gt = lab[np.isin(lab[:, 0], [0, 1])] if len(lab) else np.zeros((0, 5))
        g = np.column_stack(
            [
                (gt[:, 1] - gt[:, 3] / 2) * w,
                (gt[:, 2] - gt[:, 4] / 2) * h,
                (gt[:, 1] + gt[:, 3] / 2) * w,
                (gt[:, 2] + gt[:, 4] / 2) * h,
            ]
        )
        b = r.boxes.cpu().numpy()
        keep = np.isin(b.cls.astype(int), [0, 1])
        pred, conf = b.xyxy[keep], b.conf[keep]
        order = np.argsort(-conf)
        used = np.zeros(len(g), dtype=bool)
        m = iou(pred[order], g) if len(g) and len(pred) else np.zeros((len(pred), len(g)))
        for k, i in enumerate(order):
            j = int(np.argmax(np.where(used, -1, m[k]))) if len(g) else -1
            hit = j >= 0 and m[k, j] >= 0.5 and not used[j]
            if hit:
                used[j] = True
            scored.append((float(conf[i]), bool(hit)))
        n_gt += len(g)
    scored.sort(key=lambda x: -x[0])
    tp = np.cumsum([s[1] for s in scored])
    prec = tp / np.arange(1, len(scored) + 1)
    rec = tp / max(n_gt, 1)
    grid = np.linspace(0, 1, 101)
    return float(np.mean([prec[rec >= r].max() if np.any(rec >= r) else 0.0 for r in grid]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "test"])
    ap.add_argument("--hours", type=float, default=0.28)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--init", default="yolo11n.pt", help="starting weights (e.g. a previous best.pt to train longer)")
    ap.add_argument("--name", default="spotter", help="run name under $DATA/vision/runs")
    a = ap.parse_args()
    if a.cmd == "train":
        train(a.hours, a.imgsz, a.init, a.name)
    else:
        test(a.imgsz, a.name)


if __name__ == "__main__":
    main()
