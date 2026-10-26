# ReGolazo — Architecture

ReGolazo is an AI broadcast crew for synthetic football matches. A Python simulator plays a match, a Three.js renderer draws it as a TV broadcast, a vision model (the Spotter) watches that broadcast and detects events, analytics models trained on real public data explain what is happening, and a team of agents on Azure OpenAI turns it into commentary, overlays and the "Rewind & Rewrite" experience. [PLAN.md](../PLAN.md) is the product spec. This document covers how the parts fit together.

## 1. System diagram

```
                 ┌──────────────────────────── Simulation Engine (sim/) ─────────────────────────────┐
                 │ 3D ball physics · player kinematics · tactics · decision model · outcome models   │
                 │ snapshot/fork · batched rollouts · seeded, deterministic                           │
                 └──────┬───────────────────────────────┬─────────────────────────────┬──────────────┘
                        │ SimState + MatchEvent          │ RolloutResult /             │ SimState stream
                        │ (ground truth)                 │ RolloutDistribution         │
                        ▼                                 ▼                             ▼
          ┌──────────────────────┐           ┌────────────────────────┐     ┌──────────────────────────┐
          │ Renderer (web/)      │──frames──▶│ Spotter (vision/)      │     │ Live engine (server/)     │
          │ broadcast/tactical   │  +labels  │ detect·track·homography│     │ FastAPI + WebSocket       │
          └──────────┬───────────┘           │ → MatchEvent(detected) │     │ clock, pause, seek, fork  │
                     │                       └───────────┬────────────┘     └────────────┬─────────────┘
                     │                                   ▼                               │
                     │                       ┌────────────────────────┐                  │
                     │                       │ Analyst (analytics/)   │ ModelOutputs,    │
                     │                       │ xG · xPass · PC · xT   │ stats, indices   │
                     │                       └───────────┬────────────┘                  │
                     │                                   ▼                               │
                     │   ┌───────────── Agent layer (agents/: Microsoft Agent Framework + Azure OpenAI) ─────────┐
                     │   │ Moment Scout → Tactician → (Simulation tool) → Storyteller ⇄ Fact-Checker → Director │
                     │   │                                   └──► Personal Producer (per ViewerProfile)          │
                     │   └────────────────────────────────────────────┬──────────────────────────────────────────┘
                     │                    DecisionMoment · OverlayCue · Narrative (+ Azure Speech audio)
                     └──────────────────────────────▶  Browser UI: match view · overlays · Rewind & Rewrite · recap
```

## 2. Data flow

1. **Simulate.** `sim/` advances a seeded match at 25 Hz. Every tick produces a `SimState`. Every touch produces ground-truth `MatchEvent`s and marks the tick as a keyframe with a `snapshot_ref` (a fork handle).
2. **Stream.** `server/` runs the match clock (play, pause, seek, speed) and pushes `SimState` frames and events to each browser session over a WebSocket. REST endpoints serve `MatchMeta`, forks and rollouts.
3. **Render.** `web/` interpolates between ticks and draws the broadcast. In recording mode, Playwright drives the same renderer frame by frame and exports frames plus labels (projected boxes, ball pixel, camera matrices) for the Spotter.
4. **Spot.** `vision/` runs detection, tracking and homography on rendered frames and emits `MatchEvent(source="detected")`. The live accuracy panel compares those with ground truth.
5. **Analyse.** `analytics/` provides the models (xG, post-shot xG, xPass, pitch control, xT, decision value) and the stats engine. Its outputs are `ModelOutputs` (per moment) and stats keyed for evidence.
6. **Explain.** `agents/` turns events and model outputs into `DecisionMoment`s (Scout and Tactician), `Narrative`s (Storyteller, checked by the Fact-Checker), and `OverlayCue`s (Director). The Personal Producer filters and localizes those for each `ViewerProfile` and adds Azure Speech audio.
7. **Rewind.** The fan's gesture becomes an `ActionIntent`. The server sends a `RolloutRequest` to the simulator, which forks from `snapshot_ref`. The shared action resolver turns the intent into a `ResolvedAction`, and the simulator returns a `RolloutResult` (n = 1, with frames for playback) or a `RolloutDistribution` (n = 1,000).

**Rule:** LLMs never invent numbers. Every number in a `Narrative` has an `Evidence` entry that points to a tool output (an event, a model output, a stat or a rollout), and the Fact-Checker rejects text whose numbers don't match that evidence.

## 3. Contracts

All contracts live in `shared/schemas/` as Pydantic v2 models. They are **frozen at `SCHEMA_VERSION = 1.0.0`**. Any change goes through the lead engineer: bump the version, regenerate, and update fixtures.

| Contract | Module | Purpose |
|---|---|---|
| `MatchMeta` | `meta.py` | Teams, kits, crests, players (attributes 1–99, archetype, physical params), seed, pitch |
| `SimState` | `state.py` | One tick: ball (3D pos, vel, spin), players (2D pos, vel, facing, jump z, animation state), possession, phase, score, keyframe/snapshot handle |
| `MatchEvent` | `events.py` | One event: common envelope plus a typed qualifier block (`PassQ`, `ShotQ`, `PressureQ`, …) chosen by `kind`. A validator enforces the type→qualifier mapping. |
| `ModelOutputs` | `models.py` | xT at the ball, shot xG now, `PitchControlGrid`, scored `OptionEvaluation`s (player-adjusted and baseline) |
| `DecisionMoment` | `moments.py` | Rewindable moment: snapshot ref, real `ResolvedAction`, options, real consequence, judgement-free `RewindScores`, neutral hook |
| `ActionIntent` / `ResolvedAction` | `actions.py` | Gesture → intent → technique: what the fan or AI asked for, and what the resolver made of it (technique, launch, curve cap, uncertainty ellipse, preview arc) |
| `RolloutRequest` / `RolloutResult` / `RolloutDistribution` | `rollout.py` | Fork + single rollout (frames, namespaced events, outcome, the model's evaluation of the fan's action) and the 1,000-realities distribution with 95 % CIs, plottable samples and the displayed model probability for the consistency test |
| `Verdict` | `rollout.py` | The fan's call vs the real one after a Rewind: both options, outcome, framing (`beat_the_odds`, `unlucky`, …), value gap (post-commit only) |
| `OverlayCue` | `overlay.py` | Machine-readable timed overlay: window, priority, anchor (player / screen region / pitch point), typed payload, allowed modes, evidence |
| `Narrative` | `narrative.py` | Text in one language and mode, with evidence and fact-check result, optional TTS audio |
| `ViewerProfile` | `narrative.py` | Per-session personalization: mode, favorite club, language, voice, "show numbers first" |
| `StatsSnapshot` / `Recap` | `stats.py` | Analyst output (team and player stats, Control-vs-Chaos, momentum, milestones) and the full-time recap. `Evidence(kind="stat")` refs are dotted paths into a snapshot (`players.H07.top_speed_kmh`), resolved by `stat_value` |
| `FrameLabels` | `labels.py` | Recording-mode ground truth per rendered frame: camera intrinsics/extrinsics, per-object pixel boxes, visibility, world position, ball shadow pixel (renderer → Spotter) |
| `ServerMessage` / `ClientMessage` | `stream.py` | WebSocket protocol (`hello`, `frame`, `events`, `clock`, `cues`, `narrative`, `moment`, `error` / `play`, `pause`, `seek`, `speed`, `profile`); wrapped in `StreamProtocol` for codegen |

Vocabulary (event types, techniques, outcomes, phases) is in `shared/schemas/enums.py` as string `Literal`s, so the same strings appear in Python, JSON and TypeScript. Definitions and thresholds are in [event_taxonomy.md](event_taxonomy.md).

### 3.1 Conventions

- **World frame everywhere.** Pitch 105 × 68 m, origin at the centre spot, x along the length (goals at x = ±52.5), y across (±34), z up. `SimState`, `MatchEvent` locations, overlays and the Spotter's homography all use this one frame. Which team attacks +x is derived from `MatchMeta.home_attacks_positive_x_first_half` and the period; `shared/geometry.py` converts to the **attacking frame** (acting team attacks +x, +y on its left) that the taxonomy thresholds and the StatsBomb-trained models use. StatsBomb → metres conversions are in the same module.
- **Time.** `t` is continuous match seconds since the first kickoff; `period` disambiguates. `tick = round(t × sim_hz)` for simulator output.
- **Ids.** Players are `H07` / `A10` (team letter + shirt number); events are `ev_…` (detected events `ev_d_…`).
- **Three.js mapping.** The renderer maps world (x, y, z) to Three.js (x, z, −y) (y-up).
- **Wire format.** JSON in serialization mode: fields with defaults are always present, so TS types have no optional-vs-undefined ambiguity. WebSocket frames are full `SimState` JSON (about 150 KB/s at 25 Hz, no delta encoding). Fork, preview and rollouts are REST.
- **Detected events.** The Spotter may leave qualifiers it can't see as null (ground truth may not; see `GROUND_TRUTH_REQUIRED`) and identifies actors by `track_id` when it has no player id.
- **Possession.** `possession_id` follows taxonomy §4.7 (new id on a change of team *and* on every restart) and is the same number in `SimState` and `MatchEvent`.

### 3.2 Generation pipeline

```
shared/schemas/*.py ──(uv run python -m shared.schemas.export)──▶ shared/schemas/json/*.schema.json   (draft-07)
                                                               ├─▶ shared/fixtures/*.json              (one per root contract)
                                                               ├─▶ web/src/types/generated/enums.ts    (const arrays + types)
                                                               └─▶ web/tests/generated/fixtures.ts     (fixtures typed as TS contracts)
shared/schemas/json/contracts.schema.json ──(npm run gen:types)──▶ web/src/types/generated/contracts.ts (json-schema-to-typescript)
```

Exported schemas are draft-07: Pydantic's `prefixItems` tuples are rewritten to array-form `items`, so TypeScript gets typed tuples such as `[number, number, number]`. CI fails if any generated file is stale.

**Round-trip evidence (Step 1 gate):**
- Python → JSON: every root fixture, plus one example per event-qualifier kind (21) and per cue payload type (11), round-trips through JSON unchanged.
- JSON ↔ schema: all of them validate against the exported schemas in Python (`jsonschema`) and in TS (Ajv).
- JSON → TS: the typed TS fixtures and examples compile under strict `tsc`, so the Python output conforms to the generated TS types (an injected bad enum value fails `tsc`).
- TS → JSON → Python: Vitest builds client messages from the generated TS types and writes them to `shared/fixtures/from_ts/`; pytest parses them with Pydantic. CI fails if that file is stale.
- Negative tests: the JSON Schemas reject structural errors (id patterns, tuple lengths, enums, unknown fields). **Cross-field rules are enforced only by Pydantic**: event type ↔ qualifier kind, ground-truth-required qualifiers, shot outcome ↔ `on_target`, cue `t_end ≥ t_start`, anchor targets, grid size, intent targets. The Python server validates everything it receives, so it is the authority.

### 3.3 Cross-team interfaces (Python)

| Module | Between | Purpose |
|---|---|---|
| `shared/model_api.py` | analytics → sim, agents | `AnalyticsModels` protocol (vectorized `xg`, `psxg`, `xpass`, `xt`, `pitch_control`, plus Spearman params and physical priors), feature dataclasses in the attacking frame, and `StubModels`. `load_models()` returns the trained models once `analytics/artifacts/manifest.json` exists. |
| `shared/engine_api.py` | sim → server, agents | `RolloutEngine` protocol: `preview`, `options`, `rollout`, `distribution`, with latency targets |
| `shared/recording.py` | sim → server, vision, tooling | Recorded match directory (`manifest.json`, `meta.json`, columnar `frames.npz`, `events.jsonl`, `moments.jsonl`, `snapshots/`). `Recording.state_at(i)` rebuilds one `SimState` without loading a whole match as objects. Snapshot refs are `snap:<match_id>:<tick>`. |

## 4. Components and ownership

| Path | Component | Plan step | Depends on |
|---|---|---|---|
| `shared/` | Contracts, geometry helpers | 1 | — |
| `analytics/` | Data loaders, xG / psxG / xPass / pitch control / xT / decision value; later the stats and explainability engine | 2, 7 | shared |
| `sim/` | Physics, kinematics, tactics, action resolver, decision model, fork/rollout, seed search | 3 | shared, analytics artifacts (stubbed until Step 2 lands) |
| `server/` | FastAPI + WebSocket live engine, sessions | 4 | shared, sim |
| `web/` | Three.js renderer, recording mode, overlays, Rewind UI, personalization, recap | 5, 9, 10 | shared (generated types) |
| `vision/` | Dataset generation, YOLO fine-tune, tracking, homography, ball 3D, event inference | 6 | shared, sim, web recording mode |
| `agents/` | Microsoft Agent Framework agents and tools | 8 | shared, analytics, sim |
| `infra/` | Bicep / azd for Container Apps, ACR, Blob, App Insights, Key Vault | 11 | — |

Parallel build agents each own one of these directories and must not edit another's. `shared/` and the root `pyproject.toml` / `web/package.json` dependency lists are changed only by the lead engineer, or by an agent with an explicit brief to do so, merged by the lead.

## 5. Agents

| Agent | Model | Responsibility | Tools |
|---|---|---|---|
| Moment Scout | `gpt-4.1-mini` | Flags rewindable moments with the judgement-free score (richness, stakes, consequence, salience; never the value gap); writes neutral hooks | stats store, option-richness / stakes / consequence scorers |
| Tactician | `gpt-5-mini` | Enumerates and explains options at a frozen state | xG, xPass, pitch control, xT, decision value |
| Simulation | — (tool) | Fork, execute, single or batched rollouts | sim engine |
| Storyteller | `gpt-5-mini` | "Why it matters" narratives, milestones, recap, with evidence | stats store, moment records |
| Fact-Checker | `gpt-5-mini` + deterministic number matcher | Verifies every claim against evidence; bounded revise loop | stats store, claim extractor |
| Director | code + `gpt-4.1-mini` for editorial ranking | Priority queue, cooldowns, collision avoidance; emits `OverlayCue`s | overlay schema |
| Personal Producer | `gpt-4.1-mini` | Per-session mode, club perspective, translation, voice | Azure Speech |

Orchestration uses Microsoft Agent Framework (Python), with tracing to Application Insights. Deployment names come from configuration only.

## 6. Key decisions

| Decision | Choice | Why |
|---|---|---|
| Ball / player dimensionality | 3D ball physics, 2D player kinematics + jump height | Lofted passes, crosses, headers and shot height without a full-body simulation |
| Sim ↔ model consistency | Physical execution with attribute-scaled noise; shots resolved by a save model trained as post-shot xG | Displayed probabilities and simulated outcomes agree statistically, which is tested |
| One renderer | The Three.js app renders both the live view and the Spotter's training video | Labels are exact (projected from the same scene) and there is a single art pipeline |
| Contracts | Pydantic → JSON Schema → TS | One source of truth, with typed tuples and literal unions on both sides |
| Event frame | World frame in contracts, attacking frame computed on demand | One frame for sim, renderer, Spotter and overlays avoids flip bugs; models get their frame from one tested helper |
| Frontend UI layer | Plain TypeScript modules + DOM/CSS overlays over the WebGL canvas, no framework | The app is one full-screen 3D scene with a few overlay panels driven by `OverlayCue`s and a small state machine (live → rewind → verdict). A framework adds a second render loop and bundle weight for little gain. Overlays are small components (`mount(cue) / update(t) / unmount()`) that are easy to test with Vitest and Playwright. |
| Python layout | One uv project, one package per component | One lockfile, simple imports across components, each worktree gets its own `.venv` |

## 7. Azure environment (verified 2026-10-26)

| Item | Value |
|---|---|
| Subscription | "Azure subscription 1" (free trial, $200 credit) |
| Resource group | `rg-regolazo` (eastus2) |
| AI resource | `regolazo-ai-j76s9` (Azure AI Foundry, kind AIServices): serves Azure OpenAI and Azure AI Speech |
| Providers registered | Microsoft.App, ContainerRegistry, Storage, Insights, OperationalInsights, KeyVault, CognitiveServices, ManagedIdentity |

### Model choices

| Role | Deployment | Capacity | Used for |
|---|---|---|---|
| Main reasoning / narrative | `gpt-5-mini` (2025-08-07, GlobalStandard) | 400K TPM | Tactician, Storyteller, Fact-Checker, recap |
| Fast / high-frequency | `gpt-4.1-mini` (2025-04-14, GlobalStandard) | 150K TPM | Live hooks, Director ranking, translation, Personal Producer |

Newer models (gpt-5.x and later) have zero quota on this trial subscription. Deployment names are read only from configuration (`AZURE_OPENAI_DEPLOYMENT_MAIN`, `AZURE_OPENAI_DEPLOYMENT_FAST`), so a model swap is an `.env` change. Spare quota if needed: o4-mini 100K TPM, gpt-4o (Standard) 50K TPM.

### Voices (Azure AI Speech neural TTS)

| Language | Voice |
|---|---|
| English | `en-GB-RyanNeural` |
| Spanish | `es-ES-AlvaroNeural` |
| French | `fr-FR-HenriNeural` |

Languages are configuration (a list of codes with a voice per code), not an enum, so swapping French for German is a config change.

## 8. Tooling

- Python 3.12 via `uv`, one project at the repo root (`pyproject.toml`). Ruff (lint + format), mypy (strict, Pydantic plugin), pytest. Tests that call Azure are marked `live` and excluded from CI.
- Web: Vite + TypeScript (strict, `noUncheckedIndexedAccess`) + Three.js in `web/`, with ESLint, Vitest and Playwright.
- CI: GitHub Actions (`.github/workflows/ci.yml`) runs lint, type check, contract freshness and unit tests for both stacks, plus a web build and a Playwright smoke test.
- Video: ffmpeg 9 with libx264 + AAC, verified by `scripts/check_ffmpeg.sh`.
