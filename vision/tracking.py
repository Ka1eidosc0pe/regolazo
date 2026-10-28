"""Detector + ByteTrack on rendered broadcast frames -> pitch-metre tracks.

Inference path inputs: the frame image and the camera calibration from the frame metadata (no ground-truth objects).
"""

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import ROOT, YAML, IterableSimpleNamespace

from shared.schemas.labels import CameraCalibration
from shared.schemas.meta import MatchMeta
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
    colour: tuple[float, ...] | None = None
    """Shirt hue histogram (players only)."""


@dataclass
class FrameResult:
    t: float
    players: list[Detection]
    ball: Detection | None
    """Most confident ball candidate (for display); tracks use the gated candidates."""
    balls: list[Detection] = field(default_factory=list)


@dataclass
class Tracks:
    """Dense per-frame arrays: positions are NaN where a track is not seen."""

    t: F64
    ball: F64
    """(N, 2) ball ground-plane position."""
    players: dict[int, F64] = field(default_factory=dict)
    """track id -> (N, 2) foot position."""
    role: dict[int, str] = field(default_factory=dict)
    """track id -> home | away | goalkeeper | referee."""


BALL_MIN_CONF = 0.3
BALL_MAX_MPS = 40.0


def make_tracker(fps: float) -> BYTETracker:
    cfg = YAML.load(ROOT / "cfg" / "trackers" / "bytetrack.yaml")
    cfg.update(track_buffer=round(2 * fps), track_low_thresh=0.1, new_track_thresh=0.3)  # keep lost tracks 2 s
    return BYTETracker(IterableSimpleNamespace(**cfg))


HUE_BINS = 12


def hue_hist(hsv_pixels: np.ndarray) -> np.ndarray:
    """Normalised histogram of OpenCV hue (0-180) over saturated, non-dark pixels; zeros if none."""
    h, sat, val = hsv_pixels[:, 0], hsv_pixels[:, 1], hsv_pixels[:, 2]
    keep = (sat > 60) & (val > 40)
    hist = np.bincount((h[keep].astype(int) * HUE_BINS) // 180, minlength=HUE_BINS)[:HUE_BINS].astype(np.float64)
    total = hist.sum()
    out: np.ndarray = hist / total if total > 0 else hist
    return out


def torso_colour(img: np.ndarray, box: tuple[float, float, float, float]) -> tuple[float, ...] | None:
    """Hue histogram of the shirt region (upper body, central part), pitch green removed."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    crop = img[int(y0 + 0.15 * h) : int(y0 + 0.45 * h) + 1, int(x0 + 0.2 * w) : int(x1 - 0.2 * w) + 1]
    if crop.size == 0:
        return None
    px = crop.reshape(-1, 3)
    b, g, r = (px[:, k].astype(np.int32) for k in range(3))
    px = px[~((g > r + 15) & (g > b + 15))]
    if len(px) < 3:
        return None
    hsv = cv2.cvtColor(np.ascontiguousarray(px.reshape(-1, 1, 3)), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    hist = hue_hist(hsv)
    return tuple(float(v) for v in hist) if hist.sum() > 0 else None


class Spotter:
    def __init__(self, weights: Path, fps: float, imgsz: int = 960, conf: float = 0.1, device: str = "mps") -> None:
        self.model = YOLO(str(weights))
        self.tracker = make_tracker(fps)
        self.imgsz, self.conf, self.device = imgsz, conf, device

    def step(self, fm: FrameMeta) -> FrameResult:
        img = cv2.imread(str(fm.image))
        assert img is not None, fm.image
        r = self.model.predict(img, imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False)[0]
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
                box = (x0, y0, x1, y1)
                players.append(
                    Detection(int(tid), int(c), score, box, (float(p[0]), float(p[1])), torso_colour(img, box))
                )
        balls = []
        for xyxy, conf in zip(boxes.xyxy[cls == BALL_CLS], boxes.conf[cls == BALL_CLS], strict=True):
            if conf < BALL_MIN_CONF:
                continue
            x0, y0, x1, y1 = (float(v) for v in xyxy)
            p = image_to_pitch(fm.camera, np.array([[(x0 + x1) / 2, (y0 + y1) / 2]]), BALL_RADIUS_M)[0]
            balls.append(Detection(-1, BALL_CLS, float(conf), (x0, y0, x1, y1), (float(p[0]), float(p[1]))))
        balls.sort(key=lambda d: -d.conf)
        return FrameResult(fm.t, players, balls[0] if balls else None, balls)


def gate_ball(results: list[FrameResult]) -> F64:
    """Ball track: per frame the candidate consistent with the last accepted position (<= BALL_MAX_MPS),
    re-acquiring on a confident detection after 0.5 s without one."""
    out = np.full((len(results), 2), np.nan)
    last: tuple[float, np.ndarray] | None = None
    for i, r in enumerate(results):
        cands = r.balls or ([r.ball] if r.ball is not None else [])
        best = None
        for d in cands:
            p = np.asarray(d.pitch)
            if last is None or r.t - last[0] > 0.5:
                if d.conf >= 0.5:
                    best = p
                    break
            elif np.linalg.norm(p - last[1]) <= BALL_MAX_MPS * (r.t - last[0]) + 1.0:
                best = p
                break
        if best is not None:
            out[i] = best
            last = (r.t, best)
    return out


def kit_hist(kit: tuple[str, str, str]) -> np.ndarray:
    """Expected shirt hue histogram from (primary, secondary, pattern)."""

    def hsv(hx: str) -> np.ndarray:
        hx = hx.lstrip("#")
        bgr = np.array([[[int(hx[4:6], 16), int(hx[2:4], 16), int(hx[0:2], 16)]]], dtype=np.uint8)
        out: np.ndarray = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).reshape(1, 3)
        return out

    primary, secondary, pattern = kit
    w = 0.85 if pattern == "plain" else 0.5
    out: np.ndarray = w * hue_hist(hsv(primary)) + (1 - w) * hue_hist(hsv(secondary))
    return out


def kits_from_meta(meta: MatchMeta) -> dict[str, np.ndarray]:
    """Pre-match kit colours (broadcast metadata, not tracking ground truth)."""
    return {
        side: kit_hist((team.kit.primary, team.kit.secondary, team.kit.pattern))
        for side, team in (("home", meta.home), ("away", meta.away))
    }


def assign_teams(colours: dict[int, np.ndarray], kits: dict[str, np.ndarray], iters: int = 20) -> dict[int, str]:
    """2-means over per-track shirt hue histograms, initialised at the kits' expected histograms; clusters are then
    matched to the kits (keep or swap, whichever is closer)."""
    if not colours:
        return {}
    ids = list(colours)
    x = np.stack([colours[k] for k in ids])
    names = ["home", "away"]
    ref = np.stack([kits[n] for n in names])
    cent = ref.copy()
    lab = np.zeros(len(ids), dtype=int)
    for _ in range(iters):
        lab = np.argmin(np.linalg.norm(x[:, None] - cent[None], axis=2), axis=1)
        for c in range(2):
            if np.any(lab == c):
                cent[c] = x[lab == c].mean(axis=0)
    keep = np.linalg.norm(cent[0] - ref[0]) + np.linalg.norm(cent[1] - ref[1])
    swap = np.linalg.norm(cent[0] - ref[1]) + np.linalg.norm(cent[1] - ref[0])
    if swap < keep:
        lab = 1 - lab
    return {k: names[int(c)] for k, c in zip(ids, lab, strict=True)}


def to_tracks(results: list[FrameResult], kits: dict[str, np.ndarray] | None = None, min_frames: int = 5) -> Tracks:
    """Dense tracks. Outfield team comes from kit colour when `kits` is given (the detector's home/away classes
    only know the training matches' kits), else from the detector's class vote."""
    n = len(results)
    t = np.array([r.t for r in results], dtype=np.float64)
    players: dict[int, F64] = {}
    votes: dict[int, Counter[int]] = {}
    cols: dict[int, list[tuple[float, ...]]] = {}
    for i, r in enumerate(results):
        for d in r.players:
            if d.track_id not in players:
                players[d.track_id] = np.full((n, 2), np.nan)
                votes[d.track_id] = Counter()
                cols[d.track_id] = []
            players[d.track_id][i] = d.pitch
            votes[d.track_id][d.cls] += 1
            if d.colour is not None:
                cols[d.track_id].append(d.colour)
    keep = {k: v for k, v in players.items() if int(np.sum(~np.isnan(v[:, 0]))) >= min_frames}
    role = {k: ROLE_OF_CLS[votes[k].most_common(1)[0][0]] for k in keep}
    if kits is not None:
        outfield = {k: np.mean(np.asarray(cols[k]), axis=0) for k in keep if role[k] in ("home", "away") and cols[k]}
        role.update(assign_teams(outfield, kits))
    return Tracks(t, gate_ball(results), keep, role)
