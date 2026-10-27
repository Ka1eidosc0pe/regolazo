# ReGolazo analytics models: evaluation report

We trained these models on real public football data: StatsBomb Open Data and Metrica Sports sample tracking. Everything shown on screen runs on synthetic matches; the real data is used only for training and calibration (PLAN.md §3). The simulator and agents load the models through `shared.model_api.load_models()`. Once `analytics/artifacts/manifest.json` exists, it returns `analytics.models.TrainedModels`.

All numbers below come from `analytics/artifacts/metrics.json`, written by `python -m analytics.train`.

**Protocol.** 20 % of matches are held out as the test set (seed 7, grouped by match). Every modelling choice is made on a validation split carved out of the *training* matches, also grouped by match (20 % of training matches). That covers the GBM settings (small grid), the number of trees (early stopping on the validation log-loss) and the xG blend weight. The test matches are used only to report results. The models are then refit on all training matches for the test numbers, and on all rows for the shipped artifacts, with the same settings and tree counts.

## 1. Data

| Source | What we used | Volume |
|---|---|---|
| StatsBomb Open Data (events, lineups, 360) | Men's: UEFA Euro 2020 and 2024, FIFA World Cup 2018 and 2022, Premier League 2015/16, 1. Bundesliga 2023/24 | 644 matches, 16,511 shots, 1.16 M passes and carries |
| StatsBomb 360 freeze frames | Euro 2020, Euro 2024, World Cup 2022, Bundesliga 2023/24 | 200 matches, 170,750 usable passes with a freeze frame |
| Metrica Sports sample data | Sample games 1 and 2 (25 Hz tracking + events, anonymised) | 2 matches, 50 outfield player-games |

Reproduce:

```bash
export REGOLAZO_DATA_DIR=/path/to/shared/data      # default <repo>/data; raw files land in $REGOLAZO_DATA_DIR/raw/
uv run python scripts/download_statsbomb.py         # ~3.3 GB, about 1 minute
uv run python scripts/download_metrica.py           # ~160 MB
uv run python -m analytics.build_dataset            # feature tables -> $REGOLAZO_DATA_DIR/cache/analytics/*.npz
uv run python -m analytics.train                    # select, train, evaluate, export (~3 min)
```

**Coordinates.** StatsBomb's 120 × 80 yd frame always has the acting team attacking +x. `shared.geometry.statsbomb_to_attacking_frame` and `statsbomb_goal_plane` convert it to the metric attacking frame (105 × 68 m, centre origin). StatsBomb's frame is not true to scale across the width. The posts sit at y = 36 / 44 yd, which is ±3.40 m on the pitch-wide 68 / 80 scale but ±3.66 m on the goal-mouth scale. Two consequences:

- Pre-shot xG features (shooter, keeper, defenders) all use the pitch-wide scale, so they are consistent with each other. The post angle in xG, however, uses the true ±3.66 m posts, so StatsBomb shots near the goal line are about 7 % narrower in y than the angle assumes. This distortion is small and remains in the model.
- Post-shot features use the goal-mouth scale for *both* the ball's crossing point and the keeper's y, so the keeper-to-ball offsets are comparable.

Freeze-frame geometry (shot cone, nearest defender, passing lane) lives in `analytics/features.py`. The same code builds features for simulator scenes (`analytics/scene.py`), so the training data and the simulator describe a situation the same way.

## 2. xG (pre-shot)

**Features** (`ShotFeatures` → `analytics.design.xg_matrix`):

- Distance and angle to goal, and |y|.
- Body part: foot, head or other.
- Technique: chip, volley, half-volley, diving header, backheel or overhead. StatsBomb's "Normal" technique maps to `placed`, or to `header` for headed shots.
- Situation: open play, counter, set piece (a shot ≤ 10 s after a corner, free-kick or throw-in restart, per taxonomy §3.5.2) or direct free kick.
- Keeper: distance off his line, distance to the shooter, lateral offset from the shot line, and **keeper out of the shooter–post–post cone**.
- Outfield defenders in the cone, nearest outfield defender, under pressure, first time.

The keeper-out-of-cone flag matters. 548 training shots had the keeper beaten or out of position, and they score 2–5× as often as shots from the same distance with the keeper in the cone:

| Distance | Keeper out of cone | Keeper in cone |
|---|---|---|
| 0–6 m | 0.58 | 0.28 |
| 6–12 m | 0.32 | 0.14 |
| 12–20 m | 0.18 | 0.08 |
| 20–60 m | 0.15 | 0.03 |

**Clipping, at training and inference** (`design.XG_CLIP`, about the 99th percentile of non-penalty training shots):

| Feature | Clip (m) | p99 (m) |
|---|---|---|
| Nearest defender | 10 | 9.4 |
| Keeper off his line | 9 | 8.7 |
| Keeper to shooter | 31 | 30.9 |
| Keeper lateral offset | 3 | 2.9 |

Only 8 training shots have the nearest defender ≥ 15 m away. Before clipping, the linear half extrapolated: an unmarked shot from the penalty spot scored 0.86, above the 0.77 penalty rate. It now scores 0.52 whether the nearest defender is 10, 15, 20 or 30 m away.

**Model.** The shipped xG is sigmoid(w · logit(GBM) + (1 − w) · logit(logistic)):

- **GBM:** scikit-learn `HistGradientBoostingClassifier`, monotone in distance, angle, defenders in cone and keeper-out-of-cone. Validation chose learning rate 0.02, 15 leaves, min leaf 60, L2 3, and 312 trees.
- **Logistic:** the same features plus log distance, header × distance and keeper-out-of-cone × distance. Headers are handled through these interactions.
- **Blend weight:** w = 0.4, chosen on the validation matches (validation log-loss 0.2632 at w = 0.4, against 0.2647 at w = 0 and 0.2663 at w = 1).
- **Penalties** use the empirical conversion rate, **0.772** (180 penalties).

| Held-out (3,315 non-penalty shots, 297 goals) | AUC | Log-loss | Brier | Mean pred / base rate |
|---|---|---|---|---|
| **Shipped xG (blend)** | **0.799** | **0.248** | **0.0699** | 0.094 / 0.090 |
| GBM only | 0.789 | 0.251 | 0.0705 | |
| Logistic only | 0.799 | 0.249 | 0.0704 | |
| StatsBomb's own xG on the same shots (reference) | 0.803 | 0.245 | 0.0684 | 0.094 |
| Shipped xG, foot shots (2,770) | 0.816 | 0.238 | 0.0671 | |
| Shipped xG, headers (545) | 0.703 | 0.298 | 0.0845 | (StatsBomb xG: 0.706) |

**Five more match splits**, each with its own inner selection: the shipped blend has mean AUC 0.791 (range 0.782–0.803) and mean log-loss 0.261. On the same splits, GBM alone scores 0.784, the logistic 0.789 and StatsBomb xG 0.795 (log-loss 0.258). That is within the published 0.78–0.82 range and about 0.004 AUC below StatsBomb's commercial model, which also sees assist type and body orientation.

![xG calibration](img/xg_calibration.png)

Calibration (decile mean predicted → observed): 0.013→0.009, 0.038→0.030, 0.060→0.060, 0.077→0.100, 0.106→0.096, 0.167→0.184, 0.379→0.343.

## 3. Post-shot xG / save model

**Data:** on-target shots only (goal, saved, saved to post): 5,354 shots, 31.0 % goals.

**Goal-plane projection.** Saved shots end where the keeper touched the ball, so we project the end location onto the goal plane:

- **y** is extrapolated along the shot line to the goal line, with the factor capped at 1.3. A third of all saves (1,178) are touched further out than that, and extrapolating them 3–5× pushed them past the posts.
- **z** is kept as the recorded touch height. The first version also scaled z by the extrapolation factor and clipped it to the bar. That put 250 saves at exactly z = 2.44 m (goal rate 0.000) and taught the model that top-corner shots are always saved. Now no shot sits at the bar, and shots at z ≥ 2 m score 36.5 % (n = 370), about the overall rate.
- Projected values are snapped back to StatsBomb's 0.1-yard lattice. Otherwise goals (recorded on the lattice) and projected saves are separable by quantisation alone; a tree model learned this and scored AUC 0.98 on placement.

**Features** (`PostShotFeatures`): |end y|, end z, shot distance, keeper distance off his line, keeper distance to the crossing point, and keeper lateral distance to it. The GBM is monotone in |end y| and in the keeper-to-ball distances. Validation chose learning rate 0.05, 15 leaves, min leaf 30 and 175 trees.

**Documented priors**, not fit. All coefficients are in `manifest.json` and `analytics/design.py`; the simulator's consistency test can retune them.

- **Ball speed.** The StatsBomb speed proxy (distance ÷ event duration) is *anti*-correlated with goals: goal rate 50 % at 5–15 m/s, 38 % at 15–22, 15 % at 22–30 and 5 % at 30–40 m/s. That is how durations are recorded, not physics. The model therefore adds +0.06 logit per m/s above a 22 m/s (80 km/h) reference.
- **Keeper rating.** −0.01 logit per `gk_dive` point above 50.
- **Keeper beaten.** Keepers more than ~9 m off their line are 1 % of training shots, so the learned model extrapolates flat there: a keeper 30 m out would still have saved about half of the on-target shots. If the crossing point is more than 6 m beyond the keeper's reach, post-shot xG is raised to the observed goal rate of such shots, **0.84** (37 training shots; 18 shots more than 8 m beyond: 0.83). Reach is 4 m plus 7 m/s × (flight time − 0.25 s). The guard switches on for 0.4 % of training shots, so the normal regime is unchanged.

| Held-out (1,056 on-target shots) | AUC | Log-loss | Brier |
|---|---|---|---|
| **Learned post-shot model (GBM)** | **0.882** | **0.399** | **0.129** |
| Shipped with priors, evaluated on the StatsBomb speed proxy | 0.864 | 0.430 | 0.139 |
| Logistic baseline | 0.852 | 0.444 | 0.143 |
| Pre-shot StatsBomb xG on the same shots (what the ball state adds) | 0.786 | 0.603 | 0.191 |

The learned AUC fell from 0.896 to 0.882 with the z fix: part of the old score came from the bar-height artefact. The second row is pessimistic, because it feeds the unreliable speed proxy into the speed prior; the simulator passes the true speed at the goal plane.

Behaviour of the shipped model (average keeper on his line, 16 m shot, 22 m/s):

| end y (m) → | 0 | 1 | 2 | 3 | 3.4 |
|---|---|---|---|---|---|
| z = 0.2 m | 0.012 | 0.023 | 0.148 | 0.524 | 0.551 |
| z = 1.4 m | 0.014 | 0.020 | 0.105 | 0.361 | 0.390 |
| z = 2.3 m | 0.028 | 0.037 | 0.183 | 0.624 | 0.670 |
| z = 2.44 m | 0.028 | 0.037 | 0.211 | 0.629 | 0.702 |

The top corner (3.2, 2.2) scores 0.63, against 0.014 for a central mid-height shot.

![Post-shot xG calibration](img/psxg_calibration.png)

Calibration of the learned model: 0.028→0.028, 0.094→0.104, 0.229→0.321, 0.345→0.362, 0.539→0.557, 0.731→0.771, 0.905→0.830. It is slightly overconfident in the top decile.

## 4. xPass (pass completion)

**Data:** passes with a 360 freeze frame. Throw-ins and kick-offs are excluded; outcomes "Unknown" and "Injury Clearance" are dropped; out, incomplete and offside count as failures.

**The target is the intended receiver, not the outcome.** StatsBomb's end location is the reception point for a completed pass, but the interception or out-of-play point for a failed one. That encodes the outcome: 6 % of failed passes ended on or over a line, against 0.05 % of completed ones. In the first version, a 15 m pass to |y| = 33.5 scored 0.87 and to |y| = 34 scored 0.35.

The target for **every** pass, completed or failed, is now the visible teammate (passer excluded) nearest the recorded end location, at the moment the ball is struck. That is the receiver's position when the pass was played, which is also the question the simulator asks. The median distance from that teammate to the recorded end location is 4.0 m for completed passes and 8.2 m for failed ones.

Two less strict versions both still leaked:

- Using the teammate proxy only for failed passes, and the reception point for completed ones, scored held-out AUC 0.945. A reception point is, by construction, where the ball got through.
- Capping the proxy distance at 15 m dropped long balls whose receiver was off camera, and inflated completion from 84 % to 88 %.

Passes with no visible teammate, or whose target is on or over a line, are dropped. That leaves 170,750 passes with 84.0 % completion, the same as the raw rate. Target coordinates are clipped inside the lines at inference.

**Features** (`PassFeatures`): origin and target (x, |y|), length, direction, target distance to goal, height (ground / low / high), kind, distance from the target to the nearest opponent (keeper included), opponents within 2 m of the lane (capped at 3; only 1.0 % of passes have more), and under pressure.

Kind follows the taxonomy precedence: cutback > cross > through ball > switch > chipped (high and ≤ 25 m) > back pass (≥ 135°, ≥ 3 m) > standard. The GBM is monotone in lane defenders and receiver space. Validation chose 31 leaves, min leaf 100, learning rate 0.1 and 58 trees.

| Held-out (34,350 passes) | AUC | Log-loss | Brier |
|---|---|---|---|
| **GBM (honest target)** | **0.900** | **0.275** | **0.086** |
| Logistic baseline | 0.887 | 0.295 | 0.092 |
| *Before the fix (leaky end location), for reference* | *0.941* | *0.214* | *0.066* |

AUC by kind: standard 0.892 (n 25,473), back pass 0.877, switch 0.791, cutback 0.725, chipped 0.662, cross 0.577, through ball 0.561. Crosses and through balls are hard to rank: whether the receiver wins the ball depends on movement after the pass, which the frame at the moment of the pass doesn't show.

Lane defenders 0 → 3 give 0.949, 0.897, 0.830, 0.822 (strictly decreasing, then flat at the cap). Moving the target from |y| = 33 to 35 changes xPass by < 0.001.

![xPass calibration](img/xpass_calibration.png)

Calibration: 0.321→0.331, 0.543→0.550, 0.765→0.767, 0.875→0.877, and the upper deciles are within 0.007.

**Caveats for the simulator:**

- 360 frames show only players in the broadcast camera's view (median 9 of 11 opponents). In the simulator all 22 are known, so lane counts and receiver distances are complete, and xPass may be a few points lower than for an identical real situation.
- Through balls are trained on the receiver's position at the moment of the pass, not on the space he runs into. A simulator through ball aimed ahead of the runner is longer than its training analogue.
- The consistency test should calibrate against the displayed xPass.

## 5. Pitch control (Spearman 2018)

`analytics/pitch_control.py` is a vectorised port of the Friends of Tracking reference (Laurie Shaw, LaurieOnTracking `Metrica_PitchControl.py`). It uses that reference's default parameters, which are exactly `shared.model_api.PitchControlParams()`:

- reaction time 0.7 s, maximum speed 5 m/s
- arrival-time σ 0.45 s
- control rate λ 4.3 s⁻¹ for both teams; the defending keeper gets 3 × λ
- average ball speed 15 m/s
- integration step 0.04 s for up to 10 s, until P(att) + P(def) > 0.99

`TrainedModels.pitch_control` returns P(att) normalised so that P(att) + P(def) = 1.

**Which parameters apply where.**

- `pitch_control_params` (5 m/s, 0.7 s) are Spearman's *model* constants. They are effective average values for "arrive and control the ball", not player kinematics. They drive pitch control and, as PLAN.md requires, the simulator's interception physics, so displayed control and simulated interceptions agree.
- `physical_priors` (top speed 8.28 m/s, peak acceleration 4.37 m/s², §7) describe how fast simulated players *move*. They feed player kinematics only.

**Keeper identification.** The protocol call can't say which defender is the keeper. `TrainedModels.pitch_control` therefore takes an optional `def_gk` index:

- `-1`, the protocol default: guess the defender nearest the defended goal, within 25 m.
- `None`: no keeper.

`analytics.scene` passes the keeper by id, from `goalkeeper_ids(meta)` (MatchMeta position "GK"). The heuristic is only a fallback. It fails when the keeper has rushed more than 25 m out, or when a defender covers the line during a rush-out.

| Check | Result |
|---|---|
| Agreement with the reference code on a Metrica frame (game 1, 176 grid points, run through the original LaurieOnTracking function) | max abs difference 3e-16 (`analytics/tests/fixtures/metrica_pc_frame.json`) |
| Raw probabilities per cell (672-cell grid) | 0.99 ≤ P(att) + P(def) ≤ 1 everywhere |
| Team-swap symmetry (no keeper bonus) | P_att(A, D) = P_def(D, A) to 1e-12 |
| Mirror symmetry (y → −y) | identical surface to 1e-12 |
| Identical teams | 0.5 everywhere |
| Metrica passes: P(att) at the destination when the pass is struck | Game 1: completed 0.891 vs lost 0.316, AUC 0.963 (799 / 192). Game 2: 0.885 vs 0.341, AUC 0.969 (964 / 160) |
| Speed | 672 points in 13 ms, 1,600 in 26 ms, 5 in 0.6 ms (M3 Pro) |

For lost balls, Metrica's end location is often where the ball was lost, so the pass AUC is optimistic. It is a sanity check, not a benchmark.

## 6. Expected threat (xT)

This is Karun Singh's method on a 16 × 12 grid in the attacking frame. For each cell, xT = P(shot)·P(goal | shot) + P(move)·Σ T(z → z′)·xT(z′). Moves are the 1,145,378 StatsBomb passes and carries; T counts successful moves over all moves, so failed moves are worth 0. The 16,331 non-penalty shots give P(goal | shot). We solve it by fixed-point iteration. `xt(x, y)` interpolates bilinearly between cell centres.

![xT grid](img/xt_grid.png)

Mean xT by column, from own goal to opponent goal: 0.0035, 0.0043, 0.0051, 0.0059, 0.0067, 0.0076, 0.0087, 0.0099, 0.0115, 0.0134, 0.0158, 0.0188, 0.0236, 0.0344, 0.0528, 0.0758. It rises in **15 of 15** steps. The maximum is 0.267, in the cell in front of goal.

## 7. Physical priors (Metrica)

| Prior | Value | How it is measured |
|---|---|---|
| Top speed | 8.28 ± 0.59 m/s (29.8 km/h) | per outfield player: best 1 s rolling mean of the speed, from positions smoothed over 7 frames (0.28 s); samples above 12 m/s treated as glitches |
| Peak acceleration | 4.37 ± 1.33 m/s² | per player: 99.9th percentile of the speed gain over 0.5 s on 0.5 s-smoothed speed (the raw maximum is dominated by tracking noise) |
| `reaction_time_s` | 0.72 s (IQR 0.45–0.91, 106 passes) | see below |
| Distance covered | 10,357 ± 685 m per outfield player per match (30 players with ≥ 85 min) | sum of smoothed speed × dt |

**`reaction_time_s` is decision + reaction time measured in hindsight, not a reflex.** It is the median delay from a Metrica pass (≥ 10 m, ≥ 0.8 s of flight) being struck until the defender nearest its target accelerates towards it at > 1.5 m/s². Only passes where that defender was not already accelerating towards the target count, giving 106 samples from 2 matches. The delay includes reading the pass, which is why it agrees with Spearman's 0.7 s rather than the ~0.25 s of a pure reflex. The shared field name is frozen; the definition is also written into `physical_priors.json` and `manifest.json`.

## 8. Realism targets (`analytics/artifacts/realism_targets.json`)

These are for the simulator realism test (PLAN.md Step 3). StatsBomb men's matches, regulation time only (periods 1–2):

- Passes exclude throw-ins.
- Completion uses StatsBomb's convention: a pass with no outcome counts as completed.
- Distance covered comes from Metrica.

`band` is a suggested tolerance for the simulator's mean over at least 200 matches.

| Stat | Mean | Std (per match / team) | Band |
|---|---|---|---|
| Goals per match | 2.68 | 1.66 | 2.27–3.08 (±15 %) |
| Shots per team | 12.6 | 5.5 | 10.7–14.5 (±15 %) |
| Pass completion | 79.0 % (pooled) | 7.4 | 75.1–83.0 (±5 %) |
| Passes per team | 477 | 127 | 405–548 (±15 %) |
| Header share of shots | 0.167 (pooled) | 0.135 | 0.125–0.209 (±25 %) |
| Mean shot distance | 16.4 m (pooled) | 2.6 | 14.8–18.1 (±10 %) |
| Possession split (team share of passes) | 50 % | 11.6 | sim std in 8.1–15.1 |
| Distance per outfield player | 10,357 m | 685 | 9,321–11,393 (±10 %) |

The file also has per-competition means. For example, Premier League 2015/16 has 2.70 goals per match, 13.0 shots and 462 passes per team, and 76.4 % completion.

## 9. Artifacts and inference

`analytics/artifacts/` (about 270 KB in total, committed):

| File | Contents |
|---|---|
| `manifest.json` | model versions, file map, penalty xG, xG blend weight, post-shot priors, pitch-control parameters, notes, headline metrics |
| `xg_gbm.npz`, `psxg_gbm.npz`, `xpass_gbm.npz` | tree ensembles flattened to node arrays |
| `xg_logistic.json` | the logistic half of the xG blend |
| `xt_grid.json` | the 16 × 12 xT grid |
| `physical_priors.json` | the Metrica priors |
| `realism_targets.json` | the simulator realism targets |
| `metrics.json` | every number in this report, plus the selection grids |

Inference needs only numpy. `analytics/trees.py` evaluates the exported ensembles: 1,000 shots in about 40 ms, 1,000 passes in about 20 ms. At training time, `export_checked` verifies the numpy evaluator against scikit-learn on every training row (to within 1e-5).

The golden vectors in `analytics/tests/fixtures/` are a different check. They store 64 feature rows per model and the numpy evaluator's own outputs at training time. The tests pin the shipped artifacts and evaluator to those outputs; they do not re-run scikit-learn. A separate test (`test_tree_export_matches_sklearn`) checks the exporter against scikit-learn on synthetic data. scikit-learn and matplotlib are dev dependencies used only for training.

`analytics/scene.py` provides:

- `scene_from_state(state, team, goalkeepers=goalkeeper_ids(meta))`
- `shot_features(scene, shooter_xy, ...)`
- `pass_features(scene, passer_xy, targets, ...)`
- `pitch_control_grid(models, scene)`, which returns a world-frame `PitchControlGrid`

All of them use the keeper by id when it is given, so the simulator does not have to reimplement the freeze-frame geometry.

## 10. Data licences and attribution (for the README)

- **StatsBomb Open Data.** © StatsBomb, from https://github.com/statsbomb/open-data under the StatsBomb Public Data User Agreement (`LICENSE.pdf` in that repository). The terms require that anything published from the data states StatsBomb as the source and uses the StatsBomb logo from their media pack. Suggested README line: *"Event and 360 data provided by StatsBomb (StatsBomb Open Data)."*, plus the logo. The golden test vectors contain model inputs derived from this data and carry the same note.
- **Metrica Sports sample data.** From https://github.com/metrica-sports/sample-data, anonymised tracking and event data. The repository asks users to "be responsible with the use of this data" and to acknowledge the source in anything public. There is no separate licence file. Suggested README line: *"Tracking data: Metrica Sports sample data."* The pitch-control fixture frame comes from this data and says so.
- **Pitch-control reference.** William Spearman, "Beyond Expected Goals" (MIT Sloan Sports Analytics Conference, 2018), and Laurie Shaw's Friends of Tracking code (https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking). We used that code only to produce the reference numbers in the test fixture; none of it is vendored.
- **xT.** Karun Singh, "Introducing Expected Threat (xT)" (2018).

## 11. Known limitations

- **Header xG** is weaker (AUC 0.70) than foot xG (0.82), as it is for StatsBomb's own model (0.71). Without cross or assist type there is little to separate headers.
- **Empty net from long range.** With the keeper identified by id, a 40 m shot at an empty net (keeper 30 m out and beaten) scores xG 0.058, against 0.010 with the position heuristic. That is still low. Only 33 training shots from 20–60 m had the keeper out of the cone (goal rate 0.15), and the keeper distances are clipped at their p99. Shot *resolution* goes through post-shot xG, where the keeper-beaten guard gives 0.84 for such a shot on target.
- **Shot techniques.** StatsBomb has no placed / driven / finesse split, so these share the "placed" baseline in xG.
- **Priors.** The post-shot speed, keeper-rating and keeper-reach terms are priors, not fitted (§3).
- **xPass data.** xPass is trained on partially visible 360 frames, and crosses and through balls are hard to rank from the moment of the pass (§4).
- **xG coordinates.** StatsBomb's lateral squeeze near goal is left in the pre-shot xG coordinates (§1).
- **Physical priors** come from two anonymised matches (50 player-games), and `reaction_time_s` is a decision + reaction proxy (§7).
