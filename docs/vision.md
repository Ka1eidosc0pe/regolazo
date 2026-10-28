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

## Data

| Split | Matches | Frames |
|---|---|---|
| train | early-0007, candidate-2012, candidate-2020: four 40 s windows each at 4 fps, two with the broadcast camera and two with the replay camera | 1,932 |
| val | candidate-2021, same windows | 644 |
| test (detector) | **candidate-2004**, the held-out demo match: every 5th frame of t 1180–1300 s (broadcast), plus 2890–2920 s (replay, 4 fps) | 722 |
| eval (tracking/events) | candidate-2004 t 1180–1300 s (period 1, the demo clip) and t 4060–4120 s (period 2), both 25 fps broadcast | 3,001 + 1,501 |

Rendering ran at 12–15 frames/s at 960×540, so the whole dataset took about 8 minutes. Frames live under
`$REGOLAZO_DATA_DIR/vision/renders/`, and the YOLO copy under `$REGOLAZO_DATA_DIR/vision/yolo/`.

Training: YOLO11n at imgsz 960, batch 16, MPS (M3 Pro), time budget 0.28 h. That gave 6 epochs in 17 minutes.
The weights are at `$REGOLAZO_DATA_DIR/vision/runs/spotter/weights/best.pt`. Validation on candidate-2021 reached
mAP50 0.904 and mAP50-95 0.735.

## Results: detector on the held-out match (candidate-2004, 722 frames)

| Class | mAP50 | mAP50-95 |
|---|---|---|
| player_home | 0.338 | 0.271 |
| player_away | 0.509 | 0.442 |
| goalkeeper | 0.977 | 0.870 |
| referee | 0.989 | 0.869 |
| ball | 0.625 | 0.347 |
| **all (5 classes)** | **0.687** | **0.560** |
| player, either team (home and away merged, AP50 computed by `spotter_train.py test`) | **0.988** | n/a |

The home and away classes are low because "home" means a different kit in every match. The four training and
validation matches have four other kit pairs, so the class cannot transfer to a new pair of kits. Detection as a
player is fine: see the class-agnostic row. The Spotter therefore merges the two classes and takes the team from
kit colour. Against ground truth, the team is right on **81.1 %** of 39,124 outfield track-frames (each matched to a
ground-truth player within 1 m).

## Results: events and speeds (candidate-2004, held out)

We evaluate on t 1181–1299 s and 4061–4119 s: the two rendered segments minus 1 s at each edge. That is 178 s of
match. Matching is greedy and one-to-one: same team and |Δt| ≤ **1.0 s**. The 2.0 s numbers are shown for reference.

| Event | GT | Detected | TP | Precision | Recall | F1 | F1 @ 2 s |
|---|---|---|---|---|---|---|---|
| pass | 33 | 33 | 18 | 0.545 | 0.545 | **0.545** | 0.576 |
| shot | 3 | 1 | 1 | 1.000 | 0.333 | **0.500** | 0.500 |
| possession change | 12 | 8 | 4 | 0.500 | 0.333 | **0.400** | 0.500 |
| sprint (> 25.2 km/h for ≥ 1 s; matched on start time) | 25 | 15 | 12 | 0.800 | 0.480 | **0.600** | 0.650 |

The ground-truth possession changes come from the recording's `events.jsonl`, wherever `possession_team` switches
between consecutive events. The detected ones are wherever the team of the control run changes, after the
brief-touch smoothing.

| Speed / distance | MAE | Support |
|---|---|---|
| Ball speed, ground balls (GT z < 0.5 m) | **2.4 km/h** | 1,699 frames |
| Ball speed, all frames with a ball track | 5.4 km/h | 1,935 frames |
| Player speed (track matched to a GT player within 1 m) | **1.0 km/h** | 41,996 track-frames |
| Pass length (matched passes) | 6.3 m | 18 passes |
| Shot speed (matched shots) | 48 km/h | 1 shot |
| Sprint peak speed (matched sprints) | 1.6 km/h | 12 sprints |

Ball and player speeds are horizontal speeds, compared with the ground truth's horizontal `|vel|`.

Projection sanity check, against ground truth: re-projecting the labelled foot pixel of every person through the
z = 0 homography lands a median of 0.3 mm from the true position. The bottom-centre of the labelled box (not occluded)
lands at a median of 0.19 m and a 90th percentile of 0.35 m.

Throughput: detection plus tracking runs at 50–60 frames/s on MPS, about 2× real time at 25 fps. The live mode
renders and spots at the same time and runs at 12.8 frames/s. There the renderer is the bottleneck.

## Known limits (honest)

- **No ball height.** The ball is projected onto the ground plane. A lofted ball projects too far from the camera and
  looks too fast. Both missed shots are aerial: a header at z ≈ 2–3 m, and a chip that peaks at 4.7 m.
  The one detected shot was struck at z up to 2 m, which is why its speed is 48 km/h too high (32 m/s against 18.7 m/s
  in ground truth). Fitting a trajectory with the shadow cue (`ball_shadow_px`) is the next step.
- **ID fragmentation.** ByteTrack makes 282 tracks in 120 s on the zoomed broadcast camera, against about 22 players
  on the pitch. Tracks and ground-truth players are paired as mutual nearest neighbours within 1 m. On that pairing,
  a player's track id changes **728 times in 27.9 matched player-minutes (26 per player-minute)**. Players leaving and
  re-entering the moving frame account for part of that, and swaps inside clusters of players for the rest. Rules that need the same player across frames are therefore unreliable: passes between "different"
  tracks of the same player are dropped by the ≥ 4 m travel rule, but they still cost some precision.
- **Ball recall.** The gated ball track covers 50 % of frames in the first segment and 36 % in the second. Fast or airborne balls are often missed, which
  hurts the timing of pass receptions and possession changes.
- Sprints are matched by team and start time only, not by player, because tracks have no player id.
- Events are passes, shots, possession and sprints only: no receipts, carries, duels or qualifiers beyond pass
  length and outcome. Thresholds were set from the taxonomy and the training matches, not tuned on candidate-2004.

## Commands

```bash
export REGOLAZO_DATA_DIR=/path/to/data
uv run python scripts/spotter_dataset.py render && uv run python scripts/spotter_dataset.py yolo
uv run python scripts/spotter_train.py train      # ~17 min on MPS
uv run python scripts/spotter_train.py test       # -> $DATA/vision/detector_metrics.json
uv run python scripts/spotter_run.py offline      # -> $DATA/matches/candidate-2004/spotter/{detected_events.jsonl,accuracy.json,metrics.json}
uv run python scripts/spotter_demo.py             # -> $DATA/matches/candidate-2004/spotter/annotated.mp4 (t 1180-1300)
uv run python scripts/spotter_run.py live --start 1226 --end 1236   # render + spot a clip live, printing events
```

The UI fetches the accuracy panel at `/api/files/spotter/accuracy.json`. It is an `AccuracyPanelPayload`. The
possession-change row has `event_type: null, qualifier: "possession_change"`, because `EventType` has no
possession-change type.

Licence: Ultralytics (YOLO11, ByteTrack) is AGPL-3.0.
