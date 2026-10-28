"""Run the Spotter on rendered broadcast video.

Offline, on two rendered segments of the held-out demo match candidate-2004 (t 1180-1300 s, first half, the demo
clip; t 4060-4120 s, second half):
    uv run python scripts/spotter_run.py offline [--rerun]
  -> $DATA/matches/candidate-2004/spotter/{detections-*.json, detected_events.jsonl, accuracy.json, metrics.json}

Live, on a short clip: renders it in headless Chromium and runs detection + tracking + event rules on each frame as
it arrives, printing events as they are detected:
    uv run python scripts/spotter_run.py live --match candidate-2004 --start 1225 --end 1240
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from shared.recording import Recording
from shared.schemas.common import Period
from shared.schemas.events import MatchEvent
from vision.clips import export_clip
from vision.evaluate import Report, combine, evaluate, id_switches, shot_speeds
from vision.events import detect
from vision.pipeline import default_weights, load_results, run_frames, save_results
from vision.tracking import FrameMeta, FrameResult, Spotter, kits_from_meta, load_frame_meta, to_tracks

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("REGOLAZO_DATA_DIR", REPO / "data"))
DEMO = "candidate-2004"
SEGMENTS: list[tuple[str, Period]] = [
    ("candidate-2004-1180-1300-broadcast-25fps", 1),  # demo clip
    ("candidate-2004-4060-4120-broadcast-25fps", 2),
]


def cache_path(segment: str) -> Path:
    return DATA / "matches" / DEMO / "spotter" / f"detections-{segment}.json"


def scores_json(r: Report) -> dict[str, dict[str, float]]:
    return {
        k: {"precision": s.precision, "recall": s.recall, "f1": s.f1, "tp": s.tp, "fp": s.fp, "fn": s.fn}
        for k, s in r.scores.items()
    }


def offline(rerun: bool) -> None:
    out = DATA / "matches" / DEMO / "spotter"
    rec = Recording.open(DATA / "matches" / DEMO)
    events: list[MatchEvent] = []
    strict: list[Report] = []
    loose: list[Report] = []
    seg_info = []
    sw_total, minutes_total = 0, 0.0
    for n_seg, (segment, period) in enumerate(SEGMENTS):
        render = DATA / "vision" / "renders" / segment
        if not (render / "manifest.json").exists():
            print("missing render", render, file=sys.stderr)
            continue
        cache = cache_path(segment)
        if rerun or not cache.exists():
            frames = load_frame_meta(render)
            spotter = Spotter(default_weights(DATA), fps=25)
            t0 = time.time()
            results = run_frames(spotter, frames)
            secs = time.time() - t0
            print(f"spotter: {len(frames)} frames in {secs:.1f} s ({len(frames) / secs:.1f} frames/s)")
            save_results(results, cache)
        results = load_results(cache)
        tracks = to_tracks(results, kits_from_meta(rec.meta))
        seg_events, runs = detect(tracks, period, rec.meta.home_attacks_positive_x_first_half, f"ev_det_s{n_seg}")
        events += seg_events
        strict.append(evaluate(rec, seg_events, runs, tracks, tol=1.0))
        loose.append(evaluate(rec, seg_events, runs, tracks, tol=2.0))
        sw, minutes = id_switches(rec, tracks)
        sw_total, minutes_total = sw_total + sw, minutes_total + minutes
        seg_info.append(
            {
                "id_switches": sw,
                "matched_player_minutes": minutes,
                "render": segment,
                "period": period,
                "t": [float(tracks.t[0]), float(tracks.t[-1])],
                "evaluated_t": [strict[-1].t0, strict[-1].t1],
                "frames": len(results),
                "tracks": len(tracks.players),
                "detection_tol_1s": scores_json(strict[-1]),
            }
        )
    events = [e.model_copy(update={"index": k}) for k, e in enumerate(events)]
    (out / "detected_events.jsonl").write_text("".join(e.model_dump_json() + "\n" for e in events))
    report, report2 = combine(strict), combine(loose)
    (out / "accuracy.json").write_text(report.payload().model_dump_json(indent=2) + "\n")
    metrics = {
        "match": DEMO,
        "segments": seg_info,
        "events": {k: sum(e.type == k for e in events) for k in ("pass", "shot", "sprint")},
        "detection": {"tol_1s": scores_json(report), "tol_2s": scores_json(report2)},
        "mae": {k: {"mae": v[0], "support": v[1], "unit": v[2]} for k, v in report.mae.items()},
        "team_accuracy": {"accuracy": report.team_accuracy[0], "track_frames": report.team_accuracy[1]},
        "tracking": {
            "id_switches": sw_total,
            "matched_player_minutes": minutes_total,
            "id_switches_per_player_minute": sw_total / minutes_total if minutes_total else None,
        },
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


def live(match: str, start: float, end: float, fps: int) -> None:
    """Render a clip and spot it frame by frame while it renders."""
    name = f"live-{match}-{int(start)}-{int(end)}"
    out = DATA / "vision" / "renders" / name
    clip = export_clip(DATA, match, start - 3.0, end + 1.0, REPO / "web" / "public" / "spotter")
    cmd = ["node", "scripts/record.ts", "--clip", f"/spotter/{clip.name}", "--out", str(out), "--fps", str(fps)]
    cmd += ["--width", "960", "--height", "540", "--start", str(start), "--end", str(end)]
    rec = Recording.open(DATA / "matches" / match)
    period = 1 if start < float(rec.arrays["t"][rec.arrays["period"] == 2][0]) else 2
    spotter = Spotter(default_weights(DATA), fps=fps)
    proc = subprocess.Popen(cmd, cwd=REPO / "web", stdout=subprocess.DEVNULL)
    results: list[FrameResult] = []
    seen: set[tuple[str, float]] = set()
    k = 0
    n = round((end - start) * fps) + 1
    t_wall = time.time()
    try:
        while k < n:
            label = out / "labels" / f"{k:06d}.json"
            nxt = out / "labels" / f"{k + 1:06d}.json"
            if not (nxt.exists() or (label.exists() and proc.poll() is not None)):
                if proc.poll() is not None and not label.exists():
                    break
                time.sleep(0.02)
                continue
            d = json.loads(label.read_text())
            from shared.schemas.labels import CameraCalibration

            fm = FrameMeta(
                k, float(d["t"]), out / d["image"], d["width"], d["height"], CameraCalibration(**d["camera"])
            )
            results.append(spotter.step(fm))
            k += 1
            if k % fps == 0 and len(results) > 10:
                tracks = to_tracks(results, kits_from_meta(rec.meta))
                events, _ = detect(tracks, period, rec.meta.home_attacks_positive_x_first_half)  # type: ignore[arg-type]
                for e in events:
                    if (e.type, round(e.t, 1)) in seen or e.t > tracks.t[-1] - 1.0:
                        continue
                    seen.add((e.type, round(e.t, 1)))
                    extra = ""
                    if e.type == "shot":
                        extra = f" {shot_speeds(tracks, [e])[0] * 3.6:.0f} km/h"
                    elif e.qualifiers is not None and e.qualifiers.kind == "pass":
                        extra = f" {e.qualifiers.length_m:.0f} m {e.qualifiers.outcome}"
                    elif e.qualifiers is not None and e.qualifiers.kind == "physical":
                        extra = f" peak {e.qualifiers.peak_speed_mps * 3.6:.1f} km/h"
                    print(f"[t={e.t:7.2f}] {e.type:6s} {e.team} track {e.track_id}{extra}", flush=True)
        secs = time.time() - t_wall
        print(f"live: {k} frames in {secs:.1f} s ({k / secs:.1f} frames/s incl. rendering)", file=sys.stderr)
    finally:
        proc.wait()
        clip.unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["offline", "live"])
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--match", default=DEMO)
    ap.add_argument("--start", type=float, default=1225.0)
    ap.add_argument("--end", type=float, default=1240.0)
    ap.add_argument("--fps", type=int, default=25)
    a = ap.parse_args()
    if a.cmd == "offline":
        offline(a.rerun)
    else:
        live(a.match, a.start, a.end, a.fps)


if __name__ == "__main__":
    main()
