# ReGolazo

> Every fan has yelled at the screen. Now the screen answers back.

ReGolazo is an AI broadcast crew for synthetic football matches, built for the Premier League × Microsoft "Inside the Game" hackathon. See [PLAN.md](PLAN.md) for the design and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the architecture.

## Repository layout

| Path | Contents |
|---|---|
| `shared/` | Contracts: Pydantic schemas, JSON Schema export |
| `sim/` | Simulation engine (3D ball physics, player kinematics, tactics) |
| `analytics/` | Models trained on public data (xG, xPass, pitch control, xT) |
| `vision/` | Spotter: detection, tracking, auto-eventing from rendered video |
| `agents/` | Agent layer (Microsoft Agent Framework + Azure OpenAI) |
| `server/` | FastAPI + WebSocket live engine |
| `web/` | Vite + TypeScript + Three.js frontend |
| `infra/` | Azure deployment templates |
| `scripts/` | Utility and smoke-test scripts |

## Development

```bash
uv sync
uv run ruff check . && uv run ruff format --check . && uv run mypy . && uv run pytest -m "not live"
cd web && npm ci && npm run lint && npm run typecheck && npm test
```

Azure smoke test (needs `.env`, see `.env.example`): `uv run scripts/smoke_azure.py`
