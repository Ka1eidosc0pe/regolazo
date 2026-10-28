"""Annotated Spotter demo clip: rendered broadcast frames with detected boxes, track ids, team colours, a pitch
minimap from the homography, and detected-event captions burned in.

    uv run python scripts/spotter_run.py offline     # first: detections + events
    uv run python scripts/spotter_demo.py [--start 1180 --end 1300]
  -> $DATA/matches/candidate-2004/spotter/annotated.mp4
"""

import argparse
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np

from shared.schemas.events import MatchEvent
from shared.schemas.meta import MatchMeta
from vision.evaluate import shot_speeds
from vision.events import track_teams
from vision.pipeline import load_results
from vision.tracking import kits_from_meta, load_frame_meta, to_tracks

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("REGOLAZO_DATA_DIR", REPO / "data"))
DEMO = "candidate-2004"
RENDER = DATA / "vision" / "renders" / "candidate-2004-1180-1300-broadcast-25fps"
CAPTION_S = 2.5
MAX_SHOWN_KMH = 120.0  # ball speeds above this come from ground-projecting a lofted ball: not shown
FFMPEG_IN = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24"]
MW, MH, MS = 210, 136, 2.0  # minimap: 105 x 68 m at 2 px/m


def bgr(hex_colour: str) -> tuple[int, int, int]:
    h = hex_colour.lstrip("#")
    return int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)


def caption(e: MatchEvent, team_name: dict[str, str], shot_kmh: dict[str, float]) -> str:
    who = f"{team_name.get(e.team or '', '?')} #{e.track_id}"
    q = e.qualifiers
    if e.type == "pass" and q is not None and q.kind == "pass":
        res = "completed" if q.outcome == "complete" else "INTERCEPTED"
        kmh = q.ball_speed_mps * 3.6
        speed = f"  ball {kmh:.0f} km/h" if kmh <= MAX_SHOWN_KMH else ""  # airborne balls project too fast
        return f"PASS  {who}  {q.length_m:.0f} m  {res}{speed}"
    if e.type == "shot":
        kmh = shot_kmh.get(e.id, 0.0)
        return f"SHOT  {who}  {kmh:.0f} km/h" if kmh <= MAX_SHOWN_KMH else f"SHOT  {who}"
    if e.type == "sprint" and q is not None and q.kind == "physical":
        return f"SPRINT  {who}  peak {q.peak_speed_mps * 3.6:.1f} km/h  {q.distance_m:.0f} m"
    return f"{e.type.upper()}  {who}"


def minimap_px(p: tuple[float, float], ox: int, oy: int) -> tuple[int, int]:
    return int(ox + MW / 2 + p[0] * MS), int(oy + MH / 2 - p[1] * MS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=1e9)
    a = ap.parse_args()
    out_dir = DATA / "matches" / DEMO / "spotter"
    meta = MatchMeta.model_validate_json((DATA / "matches" / DEMO / "meta.json").read_text())
    frames = load_frame_meta(RENDER)
    results = load_results(out_dir / f"detections-{RENDER.name}.json")
    tracks = to_tracks(results, kits_from_meta(meta))
    teams = track_teams(tracks, 1, meta.home_attacks_positive_x_first_half)
    events = [
        e
        for x in (out_dir / "detected_events.jsonl").read_text().splitlines()
        if x and (e := MatchEvent.model_validate_json(x)).period == 1
    ]
    shots = [e for e in events if e.type == "shot"]
    shot_kmh = {e.id: v * 3.6 for e, v in zip(shots, shot_speeds(tracks, shots), strict=True)}
    colour = {"home": bgr(meta.home.kit.primary), "away": bgr(meta.away.kit.primary)}
    team_name = {"home": meta.home.short_name, "away": meta.away.short_name}
    w, h = frames[0].width, frames[0].height
    path = out_dir / "annotated.mp4"
    ff = subprocess.Popen(
        [*FFMPEG_IN, "-s", f"{w}x{h}", "-r", "25", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
        stdin=subprocess.PIPE,
    )
    assert ff.stdin is not None
    for fm, r in zip(frames, results, strict=True):
        if not a.start <= fm.t <= a.end:
            continue
        img = cv2.imread(str(fm.image))
        assert img is not None, fm.image
        for d in r.players:
            x0, y0, x1, y1 = (int(v) for v in d.box)
            team = teams.get(d.track_id)
            c = colour[team] if team else (40, 40, 40)
            cv2.rectangle(img, (x0, y0), (x1, y1), c, 2)
            cv2.putText(img, str(d.track_id), (x0, y0 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1, cv2.LINE_AA)
        if r.ball is not None:
            bx, by = (int((r.ball.box[0] + r.ball.box[2]) / 2), int((r.ball.box[1] + r.ball.box[3]) / 2))
            cv2.circle(img, (bx, by), 9, (0, 255, 255), 2)
        # minimap
        ox, oy = w - MW - 12, 12
        cv2.rectangle(img, (ox, oy), (ox + MW, oy + MH), (30, 90, 30), -1)
        cv2.rectangle(img, (ox, oy), (ox + MW, oy + MH), (230, 230, 230), 1)
        cv2.line(img, (ox + MW // 2, oy), (ox + MW // 2, oy + MH), (230, 230, 230), 1)

        for d in r.players:
            team = teams.get(d.track_id)
            if team and abs(d.pitch[0]) < 60 and abs(d.pitch[1]) < 40:
                cv2.circle(img, minimap_px(d.pitch, ox, oy), 3, colour[team], -1)
        if r.ball is not None and abs(r.ball.pitch[0]) < 60 and abs(r.ball.pitch[1]) < 40:
            cv2.circle(img, minimap_px(r.ball.pitch, ox, oy), 3, (255, 255, 255), -1)
        # captions
        live = [e for e in events if e.type != "sprint" and 0 <= fm.t - e.t <= CAPTION_S]
        live += [e for e in events if e.type == "sprint" and e.t <= fm.t <= e.t + e.duration_s + 1.0][:2]
        y = h - 20
        for e in reversed(live[-3:]):
            text = caption(e, team_name, shot_kmh)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (12, y - th - 8), (24 + tw, y + 6), (20, 20, 20), -1)
            c = colour[e.team] if e.team else (255, 255, 255)
            cv2.rectangle(img, (12, y - th - 8), (17, y + 6), c, -1)
            cv2.putText(img, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            y -= th + 18
        label = f"SPOTTER (detected from video)  t={fm.t:7.2f}s"
        cv2.putText(img, label, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        ff.stdin.write(np.ascontiguousarray(img).tobytes())
    ff.stdin.close()
    ff.wait()
    print(path)


if __name__ == "__main__":
    main()
