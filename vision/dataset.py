"""Rendered FrameLabels -> YOLO detection dataset (5 classes; goalkeepers of both teams share one class)."""

import shutil
from dataclasses import dataclass
from pathlib import Path

from shared.schemas.labels import FrameLabels, ObjectLabel

CLASSES: tuple[str, ...] = ("player_home", "player_away", "goalkeeper", "referee", "ball")
CLASS_OF_LABEL: dict[str, int] = {
    "player_home": 0,
    "player_away": 1,
    "goalkeeper_home": 2,
    "goalkeeper_away": 2,
    "referee": 3,
    "ball": 4,
}
MIN_VISIBLE = 0.3
MIN_BALL_PX = 4.0


@dataclass(frozen=True)
class Window:
    match: str
    start: float
    end: float
    camera: str
    fps: int
    split: str

    @property
    def name(self) -> str:
        return f"{self.match}-{int(self.start)}-{int(self.end)}-{self.camera}-{self.fps}fps"


def yolo_box(o: ObjectLabel, width: int, height: int) -> tuple[int, float, float, float, float] | None:
    """Normalised (cls, cx, cy, w, h), or None if the object is too occluded or degenerate."""
    if o.visible_fraction < MIN_VISIBLE:
        return None
    x0, y0, x1, y1 = o.box
    cls = CLASS_OF_LABEL[o.cls]
    if cls == 4:  # the ball is a few pixels wide: keep a minimum box around its centre
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        half = max(MIN_BALL_PX, x1 - x0, y1 - y0) / 2
        x0, x1 = max(0.0, cx - half), min(float(width), cx + half)
        y0, y1 = max(0.0, cy - half), min(float(height), cy + half)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return cls, (x0 + x1) / 2 / width, (y0 + y1) / 2 / height, (x1 - x0) / width, (y1 - y0) / height


def labels_to_yolo(src: Path, root: Path, split: str, prefix: str, stride: int = 1) -> int:
    """Copy frames of one render into root/images/<split> with YOLO txt labels; returns the frame count."""
    img_dir, lab_dir = root / "images" / split, root / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for k, path in enumerate(sorted((src / "labels").glob("*.json"))):
        if k % stride:
            continue
        fl = FrameLabels.model_validate_json(path.read_text())
        stem = f"{prefix}_{path.stem}"
        shutil.copyfile(src / fl.image, img_dir / f"{stem}.jpg")
        rows = [b for o in fl.objects if (b := yolo_box(o, fl.width, fl.height)) is not None]
        (lab_dir / f"{stem}.txt").write_text(
            "".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in rows)
        )
        n += 1
    return n
