"""Detector + ByteTrack on rendered broadcast frames -> pitch-metre tracks.

Inference path inputs: the frame image and the camera calibration from the frame metadata (no ground-truth objects).
"""

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import ROOT, YAML, IterableSimpleNamespace

from shared.schemas.labels import CameraCalibration
from vision.geometry import BALL_RADIUS_M, F64, foot_point, image_to_pitch

BALL_CLS = 4
ROLE_OF_CLS = {0: "home", 1: "away", 2: "goalkeeper", 3: "referee"}


@dataclass
class FrameMeta:
    """What the Spotter may know about a frame besides its pixels: time, size and the broadcast camera calibration."""

    index: int
    t: float
    image: Path
    width: int
    height: int
    camera: CameraCalibration


def load_frame_meta(render_dir: Path) -> list[FrameMeta]:
    out = []
    for k, path in enumerate(sorted((render_dir / "labels").glob("*.json"))):
        d = json.loads(path.read_text())
        out.append(
            FrameMeta(
                k,
                float(d["t"]),
                render_dir / d["image"],
                int(d["width"]),
                int(d["height"]),
                CameraCalibration.model_validate(d["camera"]),
            )
        )
    return out


@dataclass
class Detection:
    track_id: int
    cls: int
    conf: float
    box: tuple[float, float, float, float]
    pitch: tuple[float, float]


@dataclass
class FrameResult:
    t: float
    players: list[Detection]
    ball: Detection | None


@dataclass
class Tracks:
    """Dense per-frame arrays: positions are NaN where a track is not seen."""

    t: F64
    ball: F64
    """(N, 2) ball ground-plane position."""
    players: dict[int, F64] = field(default_factory=dict)
    """track id -> (N, 2) foot position."""
    role: dict[int, str] = field(default_factory=dict)
    """track id -> home | away | goalkeeper | referee (majority vote of detected classes)."""


def make_tracker(fps: float) -> BYTETracker:
    cfg = YAML.load(ROOT / "cfg" / "trackers" / "bytetrack.yaml")
    cfg.update(track_buffer=50, track_low_thresh=0.1, new_track_thresh=0.3)
    return BYTETracker(IterableSimpleNamespace(**cfg), frame_rate=round(fps))


class Spotter:
    def __init__(self, weights: Path, fps: float, imgsz: int = 960, conf: float = 0.1, device: str = "mps") -> None:
        self.model = YOLO(str(weights))
        self.tracker = make_tracker(fps)
        self.imgsz, self.conf, self.device = imgsz, conf, device

    def step(self, fm: FrameMeta) -> FrameResult:
        r = self.model.predict(str(fm.image), imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False)[0]
        boxes = r.boxes.cpu().numpy()
        cls = boxes.cls.astype(int)
        people = boxes[cls != BALL_CLS]
        tracked = self.tracker.update(people) if len(people) else np.zeros((0, 8))
        players = []
        if len(tracked):
            feet = np.array([foot_point((float(a), float(b), float(c), float(d))) for a, b, c, d, *_ in tracked])
            pitch = image_to_pitch(fm.camera, feet, 0.0)
            for row, p in zip(tracked, pitch, strict=True):
                x0, y0, x1, y1, tid, score, c = (float(v) for v in row[:7])
                players.append(Detection(int(tid), int(c), score, (x0, y0, x1, y1), (float(p[0]), float(p[1]))))
        ball = None
        balls = boxes[cls == BALL_CLS]
        if len(balls):
            j = int(np.argmax(balls.conf))
            x0, y0, x1, y1 = (float(v) for v in balls.xyxy[j])
            p = image_to_pitch(fm.camera, np.array([[(x0 + x1) / 2, (y0 + y1) / 2]]), BALL_RADIUS_M)[0]
            ball = Detection(-1, BALL_CLS, float(balls.conf[j]), (x0, y0, x1, y1), (float(p[0]), float(p[1])))
        return FrameResult(fm.t, players, ball)


def to_tracks(results: list[FrameResult], min_frames: int = 5) -> Tracks:
    n = len(results)
    t = np.array([r.t for r in results], dtype=np.float64)
    ball = np.full((n, 2), np.nan)
    players: dict[int, F64] = {}
    votes: dict[int, Counter[int]] = {}
    for i, r in enumerate(results):
        if r.ball is not None:
            ball[i] = r.ball.pitch
        for d in r.players:
            if d.track_id not in players:
                players[d.track_id] = np.full((n, 2), np.nan)
                votes[d.track_id] = Counter()
            players[d.track_id][i] = d.pitch
            votes[d.track_id][d.cls] += 1
    keep = {k: v for k, v in players.items() if int(np.sum(~np.isnan(v[:, 0]))) >= min_frames}
    role = {k: ROLE_OF_CLS[votes[k].most_common(1)[0][0]] for k in keep}
    return Tracks(t, ball, keep, role)
