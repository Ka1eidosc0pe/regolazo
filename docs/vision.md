# Spotter: auto-eventing from the rendered broadcast

The Spotter watches the rendered broadcast video and produces `MatchEvent(source="detected")` on its own.
The inference path reads two things only: the frame pixels, and the broadcast camera calibration that the renderer
writes next to each frame (the "camera data feed"). It also uses the pre-match kit colours from `MatchMeta` and the
period. It never reads the ground-truth object labels, positions or events. Those are used only for training labels
and for the evaluation below.

Sprint scope: this is a working slice of PLAN.md Step 6. It is not the full Tier A/B taxonomy.

## Pipeline

| Stage | What it does | Code |
|---|---|---|
| Dataset | Windows of recorded matches are exported as renderer clips. Headless Chromium (ANGLE/Metal) renders them with `web/scripts/record.ts` at 960×540 and writes a JPEG plus a `FrameLabels` per frame. The YOLO labels come from those. | `scripts/spotter_dataset.py`, `vision/clips.py`, `vision/dataset.py` |
| Detector | YOLO11n (COCO-pretrained), fine-tuned on MPS. Classes: player_home, player_away, goalkeeper, referee, ball. | `scripts/spotter_train.py` |
| Tracking | ByteTrack (Ultralytics) over the person boxes, with a 2 s buffer for lost tracks. The ball is not tracked by ByteTrack. Each frame keeps every ball candidate with confidence ≥ 0.3, and a gate accepts a candidate only if it is ≤ 40 m/s from the last accepted ball. After 0.5 s with no accepted ball, a candidate with confidence ≥ 0.5 re-acquires it. | `vision/tracking.py` |
| Teams | The detector's home and away classes only know the kits of the training matches: in `candidate-2004` it calls about 98 % of home players "away". So the outfield team comes from colour instead. Each track gets a mean shirt hue histogram, and a 2-means split of those histograms is initialised at the two kits' expected histograms (from the `MatchMeta` kit colours). Goalkeepers are assigned by the goal they stand in front of. | `vision/tracking.py` |
| Pitch positions | The plane homography `H = K [r1 r2 (z·r3 + t)]` comes from the frame's camera calibration (OpenCV convention). People use the box bottom-centre at z = 0. The ball uses the box centre on the z = 0.11 m plane, with no height estimate. | `vision/geometry.py` |
| Kinematics | Gaps up to 0.6 s are filled linearly, followed by a moving average (0.5 s for players, 0.12 s for the ball) and central-difference velocity. Player speed is capped at 12 m/s. | `vision/events.py` |
| Events | **Possession:** the owner is the nearest team player within 1.5 m of the ball. A control run needs 3 frames or more, and an opponent touch shorter than 0.4 s, after which the ball comes straight back, does not change possession. **Pass:** the ball moves 4 m or more from one control run to a different player's run; it is completed if the receiver is a teammate, intercepted if not. **Shot:** within 0.5 s of the release, the ball is faster than 12 m/s and heading toward the goal the team attacks, from 35 m or closer, aimed within ±7.66 m of the goal centre. **Sprint:** above 7.0 m/s (25.2 km/h) for at least 1.0 s, merging gaps under 0.5 s (taxonomy §4.6). | `vision/events.py` |

## Results

RESULTS_PLACEHOLDER
