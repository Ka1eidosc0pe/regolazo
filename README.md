# ReGolazo

> **Every fan has yelled at the screen. Now the screen answers back.**

ReGolazo is an AI broadcast crew for synthetic football matches. A simulated match plays in a stylized 3D broadcast. A vision model watches the rendered feed and logs the events. Agents explain the match in the viewer's language and style. The signature feature is **Rewind & Rewrite**: pause at a decision moment, make a different call, and watch that alternate reality play out. Outcomes come from football analytics models trained on real public data (xG, post-shot xG, pass completion, pitch control, expected threat), never from an LLM.

Built for the Premier League × Microsoft **"Inside the Game"** hackathon.

<!-- DEMO VIDEO LINK GOES HERE -->

## What it does

| Category requirement | How ReGolazo does it |
|---|---|
| **Synthetic dataset creation** | A full 90-minute match engine (`sim/`): 3D ball physics (drag, bounce, spin), player kinematics, tactics, and execution noise scaled by player attributes. Every frame is recorded with ground-truth events using a StatsBomb-style taxonomy (`docs/event_taxonomy.md`). Over 200 simulated matches it lands inside real-football bands for shots, pass completion, passes, headers, shot distance, distance covered and possession spread (`docs/sim_realism.md`). |
| **Auto-eventing from a video feed** | The **Spotter** (`vision/`) renders the broadcast and fine-tunes a YOLO detector on frames labelled by the renderer. It tracks players and ball, projects them onto the pitch with camera calibration, infers events from the tracks, and is scored against simulator ground truth (`docs/vision.md`). |
| **Ingest → interpret → explain → render → personalize** | Engine and Spotter (ingest) → analytics models and Moment Scout (interpret) → Storyteller and Fact-Checker agents (explain) → Three.js broadcast renderer with overlays (render) → mode, language and voice per viewer (personalize). |
| **Player name tags, speed and distance badges** | Overlays on the ball carrier and key players, with sprint and top-speed badges at the taxonomy's thresholds. |
| **Pass distance, accuracy, difficulty; ball and shot speed** | Pass cards (distance, difficulty from xPass) and shot cards (xG, shot speed). |
| **Narratives, milestones, recaps** | Agent-written commentary for every key moment, plus half-time and full-time recaps. |
| **Explainability** | Rewind shows the options the player had, the pitch-control map, and model probabilities adjusted for that player. The verdict compares the fan's choice with reality, and **"Run 1,000 realities"** shows the outcome distribution. |
| **Multi-language output** | English, Spanish and French, each in analyst and casual modes, with Azure neural voices. The language list is configuration. |

### Rewind & Rewrite

The **Moment Scout** (`sim/scout.py`) flags about 10 rewind opportunities per match using objective signals only: how many viable options the carrier had, what was at stake (expected threat, score, minute), what happened next, and whether it's the kind of moment broadcasters replay. It never flags a moment because a player "chose wrong"; that's the fan's call.

At a flagged moment, or any moment the fan pauses on:
- play freezes and the camera swings to tactical;
- the fan picks a teammate or a spot in the goal;
- the engine forks from that exact frame and plays the new reality forward, with the real play ghosted alongside;
- probabilities stay hidden until the fan commits.

The 1,000-reality distribution and the single replay come from the same engine, and a test checks that they agree statistically.

### The agents

All built on **Microsoft Agent Framework**, with models on **Azure AI Foundry**:

- **Storyteller** (`gpt-5-mini`): writes commentary for each moment from a fact sheet built by deterministic tools.
- **Fact-Checker:** verifies every number and name against that fact sheet, with a bounded revise loop. In an injected-falsehood evaluation it caught 137 of 138 false claims and passed 16 of 16 true narratives (`docs/agents.md`).
- **Translator** (`gpt-4.1-mini`): Spanish and French, preserving names and numbers; the translations are fact-checked too.
- **Voice:** Azure AI Speech neural voices: en-GB-RyanNeural, es-ES-AlvaroNeural, fr-FR-HenriNeural.

**Rule:** the LLMs never invent numbers. Every number in every narrative is traceable to a tool output, and a test checks this.

## Microsoft technologies

- **Azure AI Foundry / Azure OpenAI:** `gpt-5-mini` and `gpt-4.1-mini` deployments.
- **Microsoft Agent Framework (Python):** agent orchestration and tool calling.
- **Azure AI Speech:** neural text-to-speech in three languages.

## Architecture

```mermaid
flowchart LR
  subgraph Python
    SIM[Match engine<br/>sim/] -->|frames + ground-truth events| REC[(Recording)]
    AN[Analytics models<br/>xG · psxG · xPass · pitch control · xT] --> SIM
    REC --> SCOUT[Moment Scout]
    REC --> AG[Agents<br/>Storyteller · Fact-Checker · Translator]
    AG -->|Azure AI Foundry| LLM[(gpt-5-mini / gpt-4.1-mini)]
    AG -->|Azure AI Speech| TTS[(Neural TTS)]
    SRV[FastAPI server<br/>REST + WebSocket]
    REC --> SRV
    SCOUT --> SRV
    AG --> SRV
    SIM -->|fork · rollout · 1,000 realities| SRV
  end
  subgraph Browser
    UI[Three.js broadcast<br/>overlays · Rewind & Rewrite]
  end
  SRV <-->|/ws stream, /api| UI
  UI -->|rendered frames + labels| SPOT[Spotter<br/>YOLO · ByteTrack · homography · event rules]
  SPOT -->|detected events + accuracy| SRV
```

All components share frozen, versioned contracts in `shared/schemas/` (Pydantic → JSON Schema → TypeScript). The details are in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Quickstart

You need Python 3.12 with [uv](https://docs.astral.sh/uv/), Node 20+, and ffmpeg.

```bash
uv sync
cd web && npm ci && cd ..

# 1. Simulate the demo match (about 2 minutes; writes data/matches/candidate-2004/)
uv run python scripts/sim_match.py --seed 2004 --match-id candidate-2004

# 2. Flag rewind moments
uv run python -m sim.scout --match candidate-2004

# 3. (optional, needs Azure keys in .env, see .env.example) precompute narration and voice
uv run python -m agents.precompute --match candidate-2004

# 4. Serve it with the live engine
REGOLAZO_ENGINE=sim.engine:create_engine uv run python -m server --match candidate-2004

# 5. In another terminal: the web app
cd web && npm run dev
# open http://localhost:5173/?ws=ws://localhost:8000/ws
```

Without Azure keys, everything except narration works.

### Checks

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy . && uv run pytest -m "not live"
cd web && npm run lint && npm run typecheck && npm test
```

## Known limitations

- **Simulator realism:** over 200 matches goals average 3.12 per match, just above the 3.08 upper band, and one rare event (a successful offside trap) never occurs. Every other realism band passes.
- **Expected-goals edge cases:** the xG model is weak on empty-net long shots, so shots are resolved with post-shot xG. The pass model is weaker on crosses and through balls than on ground passes.
- **Narration is precomputed** for the recorded match, because a live Storyteller call takes 7–11 s. Rewind verdicts are not narrated yet.
- **Not deployed:** Azure Container Apps deployment (planned Step 11) was cut for time, and the app runs locally.

## Data, models and licences

- **StatsBomb Open Data** trains and calibrates xG, post-shot xG, xPass and the realism targets. © StatsBomb, used under the [StatsBomb Open Data licence](https://github.com/statsbomb/open-data/blob/master/LICENSE.pdf); thanks to StatsBomb for making it public.
- **Metrica Sports sample tracking data** gives the physical priors and the distance-covered targets. © Metrica Sports, [sample-data](https://github.com/metrica-sports/sample-data), used under its licence.
- **Ultralytics YOLO** is used by the Spotter and is licensed **AGPL-3.0**. The Spotter, and any distribution that includes it, is subject to AGPL-3.0.
- **Mixamo** (Adobe) provides the athlete rig and animation clips. Only processed in-app assets are shipped; raw Mixamo files are not redistributed.
- **Barlow Condensed** font: SIL Open Font License (`web/public/fonts/OFL.txt`).
- Pitch control follows Spearman (2018). Expected threat follows Karun Singh (2018).
- Everything shown on screen is synthetic: fictional clubs, players and kits, with no real likenesses, crests or sponsors.

ReGolazo's own code is released under the [MIT licence](LICENSE).
