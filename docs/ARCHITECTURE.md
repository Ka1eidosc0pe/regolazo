# ReGolazo — Architecture

_Full architecture (diagram, data flow, contracts, agent roles) is written in Step 1. This file starts with the environment decisions recorded in Step 0._

## Azure environment (verified 2026-10-26)

| Item | Value |
|---|---|
| Subscription | "Azure subscription 1" (free trial, $200 credit) |
| Resource group | `rg-regolazo` (eastus2) |
| AI resource | `regolazo-ai-j76s9` (Azure AI Foundry, kind AIServices) — serves Azure OpenAI and Azure AI Speech |
| Providers registered | Microsoft.App, ContainerRegistry, Storage, Insights, OperationalInsights, KeyVault, CognitiveServices, ManagedIdentity |

## Model choices

| Role | Deployment | Capacity | Used for |
|---|---|---|---|
| Main reasoning / narrative | `gpt-5-mini` (2025-08-07, GlobalStandard) | 400K TPM | Tactician explanations, Storyteller, Fact-Checker, recap |
| Fast / high-frequency | `gpt-4.1-mini` (2025-04-14, GlobalStandard) | 150K TPM | Live hooks, Director editorial ranking, translation, Personal Producer |

Newer models (gpt-5.x and later) have zero quota on this trial subscription. Deployment names are read from configuration only (`AZURE_OPENAI_DEPLOYMENT_MAIN`, `AZURE_OPENAI_DEPLOYMENT_FAST`), so a model swap is an `.env` change. Spare quota if needed: o4-mini 100K TPM, gpt-4o (Standard) 50K TPM.

## Voices (Azure AI Speech neural TTS)

| Language | Voice |
|---|---|
| English | `en-GB-RyanNeural` |
| Spanish | `es-ES-AlvaroNeural` |
| French | `fr-FR-HenriNeural` |

## Tooling

- Python 3.12 via `uv`, one project at the repo root (`pyproject.toml`) with packages `shared`, `sim`, `analytics`, `vision`, `agents`, `server`. Ruff (lint + format), mypy (strict), pytest. Tests that call Azure are marked `live` and excluded from CI.
- Web: Vite + TypeScript (strict) + Three.js in `web/`, ESLint, Vitest, Playwright.
- CI: GitHub Actions (`.github/workflows/ci.yml`) runs lint, type check and unit tests for both, plus a web build and a Playwright smoke test.
- Video: ffmpeg 9 with libx264 + AAC verified by `scripts/check_ffmpeg.sh`.
