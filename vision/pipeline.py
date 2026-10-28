"""Spotter pipeline: rendered frames -> detections/tracks -> events; plus JSON caching of per-frame results."""

import json
from dataclasses import asdict
from pathlib import Path

from vision.tracking import Detection, FrameMeta, FrameResult, Spotter

REPO = Path(__file__).resolve().parents[1]


def default_weights(data_dir: Path) -> Path:
    return data_dir / "vision" / "runs" / "spotter" / "weights" / "best.pt"


def run_frames(spotter: Spotter, frames: list[FrameMeta], log_every: int = 250) -> list[FrameResult]:
    out = []
    for k, fm in enumerate(frames):
        out.append(spotter.step(fm))
        if log_every and k % log_every == 0:
            print(
                f"frame {k}/{len(frames)} t={fm.t:.2f} players={len(out[-1].players)} ball={out[-1].ball is not None}"
            )
    return out


def save_results(results: list[FrameResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(r) for r in results]))


def load_results(path: Path) -> list[FrameResult]:
    def det(d: dict[str, object]) -> Detection:
        return Detection(
            int(d["track_id"]),  # type: ignore[call-overload]
            int(d["cls"]),  # type: ignore[call-overload]
            float(d["conf"]),  # type: ignore[arg-type]
            tuple(d["box"]),  # type: ignore[arg-type]
            tuple(d["pitch"]),  # type: ignore[arg-type]
        )

    return [
        FrameResult(float(r["t"]), [det(p) for p in r["players"]], det(r["ball"]) if r["ball"] else None)
        for r in json.loads(path.read_text())
    ]
