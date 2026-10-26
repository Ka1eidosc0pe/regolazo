# ReGolazo — Design & Implementation Plan

> **Every fan has yelled at the screen. Now the screen answers back.**

ReGolazo is an AI broadcast crew for synthetic football matches. It watches a live match (through a vision model running on the rendered broadcast feed), turns events into explainable match intelligence, and — its signature feature — lets any fan **rewind a decision moment, make a different call FIFA-style, and watch that alternate reality play out**, with outcomes driven by real football analytics models (xG, pass success, pitch control, expected threat), not by an LLM.

This document is the build brief for the implementing agent. Read it fully before starting.

---

## 1. Context

- **Hackathon:** Premier League × Microsoft, "Inside the Game". Submission due ~48h from plan creation (2026-10-26).
- **Team:** one human (the user) + the implementing agent.
- **Deliverables:** public GitHub repo, short pitch/description (incl. Microsoft/Azure technologies used), **demo video < 2 minutes** (no third-party trademarks, no copyrighted music).
- **Judging (20% each):** Technological Implementation · Agentic Design & Innovation · Real-World Impact · UX & Presentation · Category Adherence.
- **Category requirements we must visibly satisfy:** ingest → interpret → explain → render → personalize; player name tagging with speed/distance threshold badges; pass distance / accuracy / difficulty; ball and shot speed; **auto-eventing from a video feed**; narratives, milestones, recaps; explainability (control vs chaos, pressure changes, rhythm); synthetic dataset creation; multi-language output.

## 2. Product summary

### 2.1 Core experience
1. A synthetic match plays live in a stylized 3D broadcast view (FIFA-lite look, original art style — no real player likenesses, club names, or crests).
2. The **Spotter** (vision model) watches the rendered broadcast feed and produces a rich event stream (full taxonomy in §2.4: passes by type, carries, take-ons, duels, pressures, interceptions, shots by type, keeper actions, off-ball runs, set pieces, team shape). A live accuracy panel compares its output with simulator ground truth.
3. Overlays appear in sync with play: player name tags, speed/distance badges on thresholds, pass and shot metrics, momentum / Control-vs-Chaos chart.
4. The **Moment Scout** surfaces **Rewind opportunities** using objective, judgement-free criteria (§2.5) — it never decides that a better option existed; the fan does. Fans can also rewind *any* on-ball moment themselves.
5. **Rewind & Rewrite:** play freezes, camera swings to tactical view, a pitch-control heatmap fades in, teammates are selectable. The user clicks a teammate (pass), drags an arrow (through ball, curvable), or clicks a spot in the goal (shot). Pass and shot types are context- and player-aware (§2.6) — a click on a teammate in the box from the wing is a cross, a playmaker can bend a through ball, a striker's finish looks and behaves differently from a fullback's. The sim forks from that exact state and plays 5–12 s forward with a ghost of the real play overlaid. Outcome is stochastic and model-consistent. **Probabilities are revealed after the fan commits** (analyst mode can toggle "show the numbers first").
6. **Verdict card** compares the user's choice with the real one; **"Run 1,000 realities"** shows the outcome distribution.
7. Commentary is narrated per viewer mode and language (text + Azure neural TTS).
8. At full time, an automated **recap** covers key moments and the biggest "what-ifs".

### 2.2 Personalization (two genuinely different modes)
- **Analyst mode:** tactical camera available, pitch-control heatmaps, xG/xT/probabilities, dense stat panels, technical commentary.
- **Casual mode:** broadcast camera, story-first graphics, "You be the manager — what would you have done?" framing, plain-language commentary, fewer numbers.
- **Favorite club:** framing, emphasis, and commentary perspective follow the chosen side.
- **Languages:** English + Spanish + French (configurable list; swapping French for German must be a config change only).
- Favorite-player follow-cam is **out of scope**.

### 2.3 Name and identity
- Name: **ReGolazo**. Two fictional clubs with original names, colors, and simple generated crests. Fictional player names (generated, culturally varied, plausible).

### 2.4 Event taxonomy (simulator ground truth and Spotter output share it)
Base definitions on the StatsBomb Open Data event specification, cross-checked with SPADL/Opta conventions; a research subagent produces `docs/event_taxonomy.md` with precise definitions and thresholds before Step 1 freezes `MatchEvent`. The simulator emits every type as ground truth; the Spotter must detect Tier A and B from video; Tier C is derived by the Analyst.

**Tier A — on-ball events**
- **Pass**, with qualifiers: height (ground / low / high), length (short / medium / long), technique (driven, lofted, chipped, curled, outside-of-boot), type (cross, cutback, through ball, switch of play, back pass, layoff, flick-on), progressive, line-breaking, key pass, assist, under pressure, outcome (complete / incomplete / intercepted / out of play / offside).
- **Ball receipt**: first-touch quality, body orientation (open / half-turn / back to goal), miscontrol.
- **Carry** (distance, progressive), **take-on / dribble** (defender beaten, success).
- **Shot**: body part (left / right foot, head, other), technique (placed, driven, finesse/curled, chip, volley, half-volley, header, toe-poke), situation (open play, counter, set piece, penalty), outcome (goal, saved, off target, blocked, woodwork), shot speed, xG, post-shot xG.
- **Defensive**: tackle (won / lost), interception, block, clearance, ball recovery, duel (ground / aerial, won / lost), foul, dispossessed, deflection.
- **Goalkeeper**: save (standing / diving, catch / parry), claim or punch on crosses, sweeper action, rush out, distribution (throw, kick, goal kick, pass).
- **Restarts and stoppages**: kickoff, throw-in, goal kick, corner (inswinger / outswinger), free kick (direct / indirect), penalty, ball out of play, offside.

**Tier B — off-ball and pressure (from tracking)**
- **Pressure** (defender closing on the ball carrier within distance and speed thresholds), **counterpress** (pressure within 5 s of losing the ball), pressing sequences.
- **Off-ball runs**: run in behind, overlap, underlap, checking run to the ball, decoy run.
- **Physical**: sprint (> 25 km/h), high-intensity run, distance, top speed, accelerations.
- Defensive line step-up / offside trap.

**Tier C — team and phase states (Analyst)**
- Possession sequences and phase of play (build-up, progression, final third, transition / counter-attack, set piece), formation detection in and out of possession, team width / length / compactness, defensive line height, PPDA, field tilt, possession-value (xT) flow.

### 2.5 Rewind opportunity selection (judgement-free)
The Moment Scout must **never** flag a moment because "the player made the wrong choice" — that is the fan's call. It flags moments where a **meaningful decision existed and mattered**, using only objective signals:
- **Decision richness:** the ball carrier had ≥ 3 viable options (completion probability and resulting value above thresholds), or high entropy across option values — the situation was genuinely open.
- **Stakes:** high possession value at the ball location (xT), plus match-context weight (score margin, minute — late and close counts more).
- **Consequence:** what happened in the next ~10 s materially changed the match — goal, shot, big chance, a turnover in a dangerous zone leading to an opposition chance, or a large possession-value swing. **Successful moments qualify too** (a goal is rewindable: "was there another way?").
- **Fan salience:** moment types broadcasters replay — final passes, shots, 1v1s, counters, decisive turnovers, set pieces.

Rewind score = weighted combination of these signals. The value gap between the actual choice and alternatives is **explicitly excluded** from selection (it is used only in the verdict after the fan commits). The Scout ranks candidates, enforces diversity (moment types, both teams, spread across the match) and a cooldown, targeting ~8–15 flagged moments per match, and writes a neutral hook ("Okafor had four options here. What would you have done?").

**Fan-initiated rewind:** the fan can pause at any time and scrub back through recent on-ball moments (a keyframe at every touch) to rewrite any of them, controlling that moment's ball carrier (either team).

### 2.6 Action controls (rich, context- and player-aware)
**Gesture → intent → technique.** The fan expresses intent with a gesture; an action resolver picks the technique from context and the player's attributes. Holding the gesture (or right-click) opens a small radial picker to override the technique; gesture length or hold duration sets power.
- **Click a teammate (pass):** the resolver picks ground, driven, lofted, chip, **cross** (passer in a wide zone, receiver in or near the box), **cutback** (passer near the byline, receiver arriving behind), **switch of play** (long diagonal across the width), or a header / flick if the ball is airborne. A blocked ground lane automatically becomes a lofted option.
- **Drag into space (through ball):** ground or lofted depending on the lane. **Bending the drag path curves the ball**, with the maximum curve and accuracy capped by the passer's Curve, Technique and Vision. A low-technique player's curved attempt comes out straighter and less accurate; elite technicians can play outside-of-the-boot balls.
- **Click a spot in the goal (shot):** the technique follows player profile and situation — power drive, placed side-foot, finesse curl (needs Curve and Finishing), chip (keeper off his line, enough Technique), volley / half-volley if the ball is in the air, header for aerial balls, toe-poke under heavy pressure. Power scales with Shot Power; the dispersion around the target scales with Finishing / Long Shots / Composure, under-pressure penalties, and weak-foot rating.
- **Drag from the ball carrier along a path (carry / dribble):** optional, for take-ons.
- **Preview before committing:** a trajectory arc for the chosen technique and an **execution-uncertainty ellipse** sized by the player's skill (analyst mode can also show probabilities first).
- **Visible difference:** animations, ball flight (speed, height, spin, curl) and outcomes differ by technique and by player — a striker's driven finish and a fullback's speculative effort must look and behave differently.

**Player attributes (synthetic, 1–99, plus Weak Foot 1–5 and Preferred Foot):** Pace, Acceleration, Stamina, Strength, Jumping · Short Passing, Long Passing, Vision, Crossing, Curve, Technique / Ball Control, Dribbling · Finishing, Shot Power, Long Shots, Volleys, Heading, Composure · Tackling, Interceptions, Marking, Positioning · GK: Reflexes, Diving, Handling, Positioning, Distribution.

**Archetypes (fictional, generated):** advanced creative playmaker (elite Vision, Curve, Long Passing — the "sees and bends the impossible pass" profile), deep-lying playmaker, box-to-box midfielder, ball-winning midfielder, poacher, target forward, inside forward, touchline winger, overlapping fullback, inverted fullback, ball-playing centre-back, stopper, sweeper keeper. Attributes drive execution noise, the AI decision model's tendencies, physical capabilities, and animation choice.

---

## 3. Key technical decisions (already made — do not relitigate)

| Area | Decision |
|---|---|
| Analytics models | **Trained on real public data** (StatsBomb Open Data incl. 360 freeze frames; Metrica Sports sample tracking data). Real data is used **only for training/calibration**; everything shown runs on synthetic matches. Include attribution per each dataset's license in README. |
| Simulation dimensionality | **Ball: full 3D physics** (gravity, quadratic air drag, bounce with restitution + friction, optional Magnus spin for curl). **Players: 2D kinematics on the pitch plane** plus a jump/height component for headers and keeper dives. Enables lofted passes, crosses, long shots, headers. |
| Sim ↔ model consistency | Execution is physical: intended action + player attributes → ball launch with attribute-scaled noise → 3D flight → resolution (pass interception from time-to-intercept physics consistent with pitch control; shots resolved by a goalkeeper save model trained as post-shot xG on StatsBomb shot end locations, plus keeper attributes). Calibration rule: **for average-attribute players, empirical rates match the displayed xG / xPass; for specific players, deviations go in the expected direction** (elite finisher above xG, poor finisher below). Displayed probabilities in Rewind are player-adjusted. The 1,000-realities numbers and the single replay must agree statistically — this is tested. |
| Sim location | Python (numpy, vectorized), server side. Streams state to browser. Rollouts batched for "1,000 realities". |
| Renderer | **Three.js in the browser**, single renderer used for both live UI and recording training/evaluation video (headless Chromium via Playwright). |
| Art style | **Stylized broadcast** (chosen by the user). The world is realistic: stadium bowl, floodlights, mowed-stripe grass with sheen, nets, crowd, broadcast camera language (in the spirit of EA FC). The athletes are stylized: real athletic proportions but smooth, sculpted, simplified forms; heads with hair silhouettes and minimal or no facial detail (avoid the uncanny valley and any real likeness); detailed kits with numbers, names, and fabric shading; realistic ball with strike trails. Target feel: "a TV broadcast where the players are premium figurines", with game-quality polish. Also fixed: original look (no real likenesses, clubs, crests or sponsors); **night match under floodlights** (floodlight bloom, light haze, wet-pitch sheen, dark sky hides background simplification); stadium bowl, instanced crowd, ball trails, broadcast-style camera work. **Rewind look: holographic ghost** — the world desaturates slightly, the fan's alternate play runs in full color, the real play replays as translucent cyan ghosts, and the pitch-control heatmap glows on the grass. Must look **stunning** — art direction is a first-class workstream. |
| Character animation | Rigged stylized humanoid + animation clips (idle, jog, sprint, pass, shoot, header, tackle, keeper dive, celebration), blended by velocity/state. Source: Mixamo (royalty-free; requires the user to download with an Adobe account) or an equivalent CC0 asset. Ask the user early if a download is needed. |
| Auto-eventing | Rendered broadcast frames + auto-generated labels → fine-tune a small detector (Ultralytics YOLO, players/ball/goalkeepers/referee) on the M3 Pro (MPS). Tracking (ByteTrack), image→pitch homography (camera calibration metadata, as real broadcast camera-tracking systems provide), possession/pass/shot/speed inference. Note Ultralytics is AGPL — fine for a public repo; state it in README. |
| Agents | **Microsoft Agent Framework (Python)** for orchestration, models served from **Azure AI Foundry / Azure OpenAI**. LLMs never invent numbers — all numbers come from tools (models, sim, stats store). |
| Voice | **Azure AI Speech** neural TTS, one voice per language/mode. |
| Hosting | **Azure Container Apps** (backend + static frontend), **Azure Container Registry**, **Azure Blob Storage** for match/video assets, **Application Insights** for tracing agents. Secrets via managed identity / Key Vault, not committed. |
| Backend | FastAPI + WebSockets, Pydantic schemas, `uv` for Python env. |
| Frontend | Vite + TypeScript + Three.js; lightweight UI layer (plain TS or a minimal framework — implementer's choice, justify in ARCHITECTURE.md). |

---

## 4. System architecture

```
                 ┌──────────────────────────── Simulation Engine (Python) ───────────────────────────┐
                 │ 3D ball physics · player kinematics · tactics · decision model · outcome models   │
                 │ snapshot/fork · batched rollouts · seeded, deterministic                           │
                 └──────┬───────────────────────────────┬─────────────────────────────┬──────────────┘
                        │ ground-truth state/events       │ fork + rollouts             │ state stream
                        ▼                                 ▼                             ▼
          ┌──────────────────────┐           ┌────────────────────────┐     ┌──────────────────────────┐
          │ Three.js Renderer    │──frames──▶│ Spotter (vision)       │     │ Live Engine / WebSocket   │
          │ broadcast/tactical   │           │ detect·track·homography│     │ timeline, pause, replay   │
          └──────────┬───────────┘           │ → detected event stream│     └────────────┬─────────────┘
                     │                       └───────────┬────────────┘                  │
                     │                                   ▼                               │
                     │                       ┌────────────────────────┐                  │
                     │                       │ Analyst (stats engine) │  Control-vs-Chaos,
                     │                       │ + analytics models     │  momentum, pass difficulty,
                     │                       │ xG · xPass · PC · xT   │  speeds, thresholds
                     │                       └───────────┬────────────┘
                     │                                   ▼
                     │   ┌──────────────────── Agent layer (Microsoft Agent Framework + Azure OpenAI) ───────────┐
                     │   │ Moment Scout → Tactician → (Simulation tool) → Storyteller → Fact-Checker → Director  │
                     │   │                                   └──► Personal Producer (per viewer: mode, club, lang)│
                     │   └────────────────────────────────────────────┬──────────────────────────────────────────┘
                     │                                                ▼ timed overlay cues (JSON) + narration + TTS
                     └──────────────────────────────▶  Browser UI: match view · overlays · Rewind & Rewrite · recap
```

### 4.1 Agent roles
| Agent | Responsibility | Tools |
|---|---|---|
| **Moment Scout** | Detects key moments (momentum swings, milestones) and Rewind opportunities using the judgement-free criteria in §2.5; writes neutral hooks. | stats store, option-richness / stakes / consequence scorers |
| **Tactician** | Enumerates realistic options at a frozen state, scores each, explains tactically why. | xG, xPass, pitch control, xT, decision value |
| **Simulation (tool, not LLM)** | Forks state, executes chosen action, runs single or batched rollouts. | sim engine |
| **Storyteller** | Writes "why it matters" narratives, milestones, recaps — with evidence references (event IDs, model outputs). | stats store, moment records |
| **Fact-Checker** | Verifies every factual claim/number in Storyteller output against the evidence; rejects with reasons → Storyteller revises (bounded loop). | stats store, claim extractor |
| **Director** | Decides what goes on screen and when: priority queue, cooldowns, collision avoidance between overlays; emits timed overlay cues. | overlay schema |
| **Personal Producer** (one per viewer session) | Adapts depth, framing, club perspective, language, voice; chooses which cues/narratives reach this viewer. | translation via LLM, Azure Speech |

### 4.2 Contracts (define first, freeze, version)
All in `shared/schemas/` as Pydantic models with exported JSON Schema → generated TS types for the frontend.
- `MatchMeta` (teams, players, attributes, kits, seed)
- `SimState` (t, ball pos/vel/spin 3D, players pos/vel/facing/state, possession, phase of play)
- `MatchEvent` (id, t, type, actor(s), location(s), outcome, qualifiers, source = `ground_truth|detected`)
- `ModelOutputs` (xG, pass probability, pitch-control grid, xT value, decision value per option)
- `DecisionMoment` (state snapshot ref, real action, options[], scores)
- `RolloutResult` (trajectory frames, events, outcome) / `RolloutDistribution`
- `OverlayCue` (id, t_start, t_end, priority, type, payload, anchor = player/screen region) — the "machine-readable timed overlay" deliverable
- `Narrative` (text, language, mode, evidence[], verified flag)
- `ViewerProfile` (mode, club, language, voice)

---

## 5. Implementation steps

Each step lists **build** and **test gates**. Do not proceed past a gate that fails; fix first. Commit at each green gate with a descriptive message.

### Step 0 — Environment & repo setup
**Build**
- Monorepo: `sim/`, `analytics/`, `vision/`, `agents/`, `server/`, `web/`, `shared/`, `infra/`, `docs/`, `scripts/`.
- Python via `uv` (3.12), ruff + mypy (or pyright), pytest. Web via Vite + TS, ESLint, Vitest, Playwright.
- GitHub Actions CI: lint + type check + unit tests for Python and web.
- Install Azure CLI (`brew install azure-cli`) and Azure Developer CLI (`brew install azd`); the **user** runs `az login`. Verify subscription, register providers, create resource group. Check Azure OpenAI model availability and quota in the user's region; pick a stronger model for narrative/tactics and a cheaper model for high-frequency calls. Record choices in `docs/ARCHITECTURE.md`.
- Verify `ffmpeg` works (video assembly).

**Test gate:** CI green on an empty skeleton; `az account show` works; a one-line Azure OpenAI chat call and a TTS call succeed from a script in `scripts/smoke_azure.py`.

### Step 1 — Contracts & architecture doc
**Build:** schemas from §4.2, JSON Schema export, TS type generation, `docs/ARCHITECTURE.md` (diagram, data flow, agent roles, decisions).
**Test gate:** schema round-trip tests (Python ↔ JSON ↔ TS types compile); example fixtures for every schema validate.

### Step 2 — Analytics models (trained on real data)
**Build**
- Data loaders for StatsBomb Open Data (events + 360 freeze frames) and Metrica sample tracking data, cached under `data/raw/` (gitignored; download script committed).
- **xG model:** features = distance, angle, body part (foot/head), shot type, goalkeeper position, defenders in shot cone, nearest defender distance, under-pressure flag; gradient boosting (or logistic baseline + GBM). Separate handling of headers.
- **Pass success model (xPass):** features = distance, height (ground/lofted), angle, receiver/defender geometry from freeze frames; and a physics-based interception component (time-to-intercept) used inside the sim.
- **Pitch control:** Spearman (2018) physics-based model; parameters from the paper / Friends-of-Tracking reference implementation; validate on Metrica tracking.
- **Post-shot xG / keeper save model:** from StatsBomb shot end locations (incl. height), shot speed proxy, and keeper position; used to resolve shots in the sim.
- **Technique-aware variants:** xG and xPass take technique and body part as features (header, volley, chip, curled, cross, through ball, etc.) so different shot/pass types have realistic baselines.
- **Expected Threat (xT):** compute grid from StatsBomb event data (Karun Singh's method).
- **Decision value:** P(success) × value at destination (xT / xG of resulting chance), compared across options.
- **Player physical params** (max speed, acceleration, reaction time) from Metrica tracking distributions.
- Export models as lightweight artifacts loaded by sim and agents.

**Test gate:** held-out evaluation — xG log-loss and AUC reported and in line with published baselines (AUC ≈ 0.78–0.82 range is typical); calibration curve plotted and reasonable; xPass AUC reported; pitch control sanity (sums to 1 per cell, symmetry tests, matches reference implementation on a fixture frame); xT grid monotonic-ish toward goal; unit tests for feature extraction. Save evaluation report to `docs/models.md`.

### Step 3 — Simulation engine
**Build**
- Fixed timestep (e.g. 25 Hz), seeded RNG, fully deterministic given seed + inputs.
- **Ball:** 3D rigid point with gravity, quadratic drag, ground bounce (restitution, rolling friction), optional Magnus curl; kick model maps intended target + power + player skill → initial velocity with skill-dependent noise.
- **Players:** steering kinematics (accel, max speed, turning radius) using params from Step 2; states (idle/jog/sprint/kick/header/tackle/dive) that drive animation.
- **Team behavior:** formation shapes by phase (build-up, attack, defend, transition), compactness, pressing triggers, marking, support runs, overlapping fullbacks, offside line.
- **Player attributes & archetypes** per §2.6; squads generated with coherent archetype mixes per formation.
- **Action execution & resolver** per §2.6: the same resolver serves AI players and the fan's Rewind gestures, so fan actions are executed with exactly the same physics and skill noise as the AI's.
- **On-ball decision model:** enumerate options (all pass types and techniques in §2.4, shot techniques, dribble, carry, clearance); score via decision value; choose with softmax over value, shaped by attributes (Vision widens the options considered, Composure reduces pressure errors) and archetype tendencies, so the AI plays believably and sometimes makes calls fans will want to rewrite.
- **Outcome resolution:** shots resolved via xG with full freeze-frame (plus on-target/keeper save/rebound physics consistent with it); passes via interception physics; tackles/duels via attribute-weighted contests; aerial duels for crosses/headers.
- **Restarts (simplified but present):** kickoff, throw-in, goal kick, corner, free kick (direct shot possible), fouls (rare), cards optional.
- **Goalkeeper** positioning, saves, dives, distribution.
- **Snapshot/fork API:** capture full state at any tick; fork with an injected action; run single rollout (returns frames + events) or N batched rollouts (vectorized or multiprocess) returning outcome distribution.
- **Seed search script:** simulate many matches fast (headless), score each for drama (goals, comebacks, late winner, # of high-regret decision moments, variety of moment types), pick the demo match. Save the chosen match as a versioned asset.

**Test gate**
- Physics unit tests: projectile range/time vs analytic solution (no drag), drag reduces range, bounce energy loss, ball stops rolling, determinism (same seed → identical trajectory).
- Behavioral tests: formation integrity, offside line respected, no player exceeds max speed/accel.
- **Realism test vs real-data distributions** across ≥200 simulated matches: goals/match, shots/team, pass completion %, passes/team, possession split variance, distance covered/player, share of headers among shots, average shot distance — each within a tolerance band of StatsBomb/Metrica-derived stats. Report in `docs/sim_realism.md`.
- **Consistency test:** for sampled decision moments with average-attribute players, empirical goal rate over N rollouts of a shot ≈ model xG (within statistical CI) and pass completion ≈ xPass; for high/low-attribute players, deviations go in the expected direction and stay within plausible bounds.
- **Technique test:** each pass/shot technique produces distinguishable ball flight (height profile, speed, curl) and realistic success rates; curl capped by attributes; resolver picks the expected technique on a suite of hand-built situations (wide-to-box = cross, byline = cutback, blocked lane = lofted, airborne ball = header/volley).
- **Event coverage test:** every event type and qualifier in §2.4 occurs in simulated matches at plausible frequencies.
- Performance: full 90-min match simulates headlessly in well under real time; 1,000 rollouts of 10 s complete in a few seconds on the M3 Pro.

### Step 4 — Live engine & streaming server
**Build:** FastAPI app; match clock with play/pause/seek/speed; WebSocket stream of `SimState` frames (delta/quantized for bandwidth) and events; REST endpoints for meta, fork/rollout, distributions; viewer sessions with `ViewerProfile`.
**Test gate:** API tests (pytest + httpx), WebSocket integration test (client receives ordered frames at target rate, pause/seek correct), load test with several concurrent viewers.

### Step 5 — Renderer & art direction (Three.js)
**Build**
- Art direction doc first (`docs/art_direction.md`): palette, kit designs, crest style, stadium mood, lighting, camera language, typography for overlays. Produce reference screenshots before full build-out.
- Stadium (procedural bowl, stands, floodlights, ad boards with original/fictional sponsors), pitch with mowing stripes and lines, goals with net deformation if feasible.
- Stylized rigged athletes with kit coloring and numbers; animation state machine blended by speed and action; ball with spin and trail; crowd (instanced).
- Cameras: broadcast (smooth follow with lead, zoom on danger), tactical (high top-down-ish), replay/cinematic (for Rewind and goals). Smooth transitions.
- Post-processing: tone mapping, bloom, subtle vignette, optional depth of field on replays.
- Interpolation between server ticks; client prediction not needed.
- **Recording mode:** headless Chromium (Playwright) renders the broadcast camera deterministically frame-by-frame and exports frames + per-frame ground-truth labels (2D bounding boxes from projected 3D bounds, ball pixel position, camera intrinsics/extrinsics).

**Test gate:** Playwright visual tests (screenshots at fixed timestamps compared against approved baselines; human/agent review of screenshots for art quality), frame-rate check (≥ 60 fps on the M3 Pro at 1080p for the live view), label projection test (rendered label boxes overlaid on frames line up — spot-checked programmatically and visually).

### Step 6 — Spotter: auto-eventing from video
**Build**
- Dataset generation: render several matches (different seeds, kits, lighting, camera behavior) → frames + labels; train/val/test split by match.
- Fine-tune small YOLO (players by team, goalkeepers, referee, ball) on MPS.
- Tracking (ByteTrack), team classification by kit color, pitch homography from camera calibration metadata, track smoothing (Kalman).
- **Ball height from a single camera:** estimate the ball's 3D position by fitting physically consistent trajectories (projectile + drag) to the 2D ball track, using apparent ball size and its shadow as cues; needed for lofted passes, crosses, headers and shot height.
- Event inference for the full Tier A and Tier B taxonomy in §2.4 (possession, passes with qualifiers, receipts, carries, take-ons, shots with technique, defensive and goalkeeper actions, restarts, pressures, off-ball runs, sprints), plus ball speed, shot speed, player speed/distance; emit `MatchEvent(source=detected)`. Rule-based inference over tracks is fine; a learned classifier for ambiguous types (e.g. shot technique, duel outcome) is welcome where it beats rules.
- Live mode: Spotter runs on the rendered feed during playback (pre-computed offline for the demo match is acceptable if real-time inference is too slow, but must also be demonstrable live on a clip).

**Test gate:** detection mAP on held-out match; tracking ID-switch rate; event precision/recall/F1 vs ground truth **per event type and per qualifier**; ball 3D position error (incl. height); speed MAE (ball, shot, players). Report in `docs/vision.md`. These metrics feed the live accuracy panel.

### Step 7 — Analyst: stats & explainability engine
**Build:** consumes detected (and, where needed, ground-truth) events + state; computes possession, pass distance/accuracy/difficulty rating (from xPass), ball/shot speed, distance and top speed per player with threshold triggers for badges, **Control-vs-Chaos index** (rolling possession turnover rate, pressure intensity, pass-chain length, spatial dispersion), momentum (rolling xT/xG flow), rhythm patterns, milestones (e.g. first shot on target, 50th completed pass, sprint record).
**Test gate:** unit tests with hand-built event fixtures and known expected values; property tests (index bounded, responds in expected direction to synthetic chaos/control sequences); streaming vs batch computation produce identical results.

### Step 8 — Agent layer
**Build**
- Microsoft Agent Framework orchestration, Azure OpenAI models, tools wrapping Steps 2/3/7.
- Agents per §4.1 with explicit system prompts, structured outputs (JSON matching schemas), evidence references.
- Storyteller ↔ Fact-Checker bounded revise loop; Director priority/cooldown logic (mostly deterministic code, LLM for editorial ranking where useful).
- Personal Producer per session: mode, club perspective, language (EN/ES/FR), TTS via Azure Speech; caching of generated narration for repeated moments.
- Tracing to Application Insights / Foundry tracing; latency budgets per agent.
- Prompt + config files versioned; no secrets in repo.

**Test gate**
- Golden tests on recorded fixtures (LLM calls recorded/replayed for CI determinism; live-call test suite runnable manually).
- **Fact-Checker test:** inject false claims/numbers into narratives → must catch ≥ 95%; true narratives must pass.
- Every number in final narratives traceable to a tool output (automated check).
- Mode differentiation test: analyst vs casual outputs differ materially (length, terminology, numbers density) — measured, not eyeballed.
- Language test: ES/FR outputs are in the right language (language detection) and preserve numbers/names.
- Latency: live narration for a key moment ready within a few seconds.

### Step 9 — Rewind & Rewrite interaction
**Build:** Rewind markers on timeline (from §2.5) plus fan-initiated pause-and-scrub through recent touches; freeze; camera swing to tactical; pitch-control heatmap shader overlay; selectable teammates; full gesture set and resolver UI from §2.6 (click teammate, drag-and-bend through ball, click goal spot, drag carry, radial technique override, power by hold/length); trajectory preview arc and execution-uncertainty ellipse; probabilities hidden until commit (analyst toggle to show first); fork + rollout playback with ghost replay of reality (translucent players/ball); verdict card (user choice vs real choice, player-adjusted probabilities, outcome, "beat the odds" / "unlucky" framing); "Run 1,000 realities" distribution visualization; narration in viewer's mode/language; return to live.
**Test gate:** Playwright E2E — trigger flagged and fan-initiated rewinds; exercise every gesture and technique override; resolver picks expected technique in scripted situations; curved through ball available only to sufficiently technical players; preview ellipse size varies with player skill; rollout plays; verdict numbers match backend; distribution matches backend statistics; UX review via screenshots; round trip latency (commit → playback start) under ~1 s, 1,000 realities under ~5 s.

### Step 10 — Overlays, personalization UI, recap
**Build:** overlay components driven only by `OverlayCue` (name tags, threshold badges, pass/shot cards, momentum/Control-vs-Chaos chart, accuracy panel); mode switcher, club picker, language picker, voice toggle; split-screen comparison view (Analyst vs Casual) for the demo; full-time recap screen (key moments, best what-ifs, player of the match, narrative in chosen language, optional TTS).
**Test gate:** overlay timing test (cues render within one frame window of `t_start` and never overlap per Director rules); E2E for each mode × language; visual baselines.

### Step 11 — Azure deployment
**Build:** Dockerfiles; `infra/` with Bicep or azd templates for Container Apps, ACR, Blob, App Insights, Key Vault, Azure OpenAI/Speech connections via managed identity; deployment script; public URL.
**Test gate:** deployed smoke test (E2E subset against public URL), cost check against the $200 credit (`az consumption` / cost analysis), no secrets in repo (secret scan).

### Step 12 — Documentation & submission assets
**Build:** README (pitch, features mapped to challenge requirements, Microsoft technologies used, architecture diagram, quickstart, data attribution & licenses incl. StatsBomb, Metrica, Ultralytics AGPL, animation assets), `docs/` reports from each test gate, LICENSE, project pitch text (short elevator pitch).
**Demo video plan (< 2 min, no copyrighted music/trademarks):** 0:00 hook ("Every fan has yelled at the screen…") → 0:10 live match with overlays + Spotter accuracy panel → 0:30 decision moment → Rewind & Rewrite → user picks the open fullback → alternate reality plays out → verdict + 1,000 realities → 1:10 split screen Analyst (EN) vs Casual (ES/FR with voice) → 1:30 recap + architecture flash (agents + Azure) → 1:50 close. The agent produces the screen recordings via Playwright and assembles with ffmpeg; the user records/approves voiceover and uploads to YouTube.
**Test gate:** fresh-clone quickstart works end to end following README only (run by a subagent with no prior context); video length < 2:00 verified with ffprobe.

---

## 6. Execution guidance for the implementing agent (Opus 5.5)

You own the outcome. Delegate to go faster and keep your context clean, but **you** own the contracts, integration, and final quality bar.

**What you do yourself**
- Steps 0–1 (environment, contracts, architecture). Contracts must be frozen before parallel work starts; changes after that go through you.
- Integration between components, cross-cutting debugging, final review of every merged piece, demo match selection, and the demo script.

**What to delegate, and how**
- **Parallel build subagents** (each in its own git worktree via `isolation: "worktree"`, merged by you): once contracts are frozen, these are largely independent —
  - Analytics models (Step 2)
  - Simulation engine core physics + kinematics (Step 3; tactics/decision model may need your closer involvement since it depends on Step 2 artifacts — integrate via the decision-value interface with a stub until Step 2 lands)
  - Renderer & art direction (Step 5)
  - Server / streaming (Step 4)
  Later wave: Spotter (Step 6, needs Steps 3+5), Analyst engine (Step 7), Agent layer (Step 8), Rewind UI (Step 9), Overlays/personalization (Step 10).
- **Research subagents** (`Explore` / general-purpose, cheaper model such as Sonnet or Haiku): look up formulas and reference implementations (Spearman pitch control, xT method, StatsBomb 360 schema, Microsoft Agent Framework APIs, Azure Speech voices), returning concise summaries — not file dumps.
- **Mechanical subagents** (cheaper model): data download scripts, test fixture generation, docs formatting, Dockerfiles, CI config.
- **Independent review subagents**: after each step, spawn a reviewer that receives the code and the step's spec (not your conclusions) to find bugs and gaps; also use `/code-review` on diffs. For visual work, have a reviewer judge screenshots against the art direction doc.
- **Fresh-eyes verification**: the README quickstart test (Step 12) must be run by a subagent with no context.

**Briefing subagents well:** give each the goal, the relevant section of this plan, the frozen schemas, file paths it owns, what it must not touch, and its test gate. Require it to report test results with actual output.

**Rules**
- Never mark a step done until its test gate passes with real output; report failures faithfully.
- Two writers never edit the same files concurrently.
- Commit at each green gate; keep commits small and descriptive.
- Ask the user only for things only they can do: `az login`, asset downloads requiring accounts, voiceover, YouTube upload, approval of art direction screenshots and the demo match.

## 7. Open items for the user
- Run `az login` once the Azure CLI is installed.
- Possibly download a character model + animation clips from Mixamo (agent will specify exact clips).
- Approve: art direction screenshots, fictional club identities, the selected demo match, the demo video cut.
- Upload the final video to YouTube and submit.
