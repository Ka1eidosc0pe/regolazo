# Commentary agents (sprint build of Steps 7 and 8)

Mode-aware, multi-language match narration with voice. The LLMs write the words; **every number and name comes from
deterministic Python tools** and is checked against them. Built on Microsoft Agent Framework 1.23 (`Agent`, `@tool`,
structured output `response_format`), Azure OpenAI in Foundry (`gpt-5-mini`, `gpt-4.1-mini`) and Azure Speech
(neural voices `en-GB-RyanNeural`, `es-ES-AlvaroNeural`, `fr-FR-HenriNeural`).

## Pieces

| Piece | File | Role |
|---|---|---|
| Stats tools | `agents/stats.py` | No LLM. Reads `events.jsonl`, `frames.npz`, `meta.json`. Key moments (every shot: goal / save / big chance with xG >= 0.2 / other chance, plus half time and full time; 16 for the demo match) and a `FactSheet` per moment: display names (never ids), minute, running score, xG and post-shot xG, shot speed, pass length and xPass of the assist, shooter running speed from frames, running team shots and xG; at half/full time possession, shots, xG, pass accuracy, top speed and distance leaders. xG and xPass are the model outputs the simulator recorded in the events. |
| Storyteller | `agents/crew.py` | `gpt-5-mini` agent. Its only tool is `get_moment_facts(moment_id)`; prompt rules: digits copied from the tool, no number words, no unsupported claims. Analyst: terse, tactical, number-dense. Casual: warm, plain, at most two numbers. |
| Fact-Checker | `agents/crew.py`, `agents/checks.py` | `gpt-5-mini` agent with structured output `Verdict(verified, issues)`, run on the fact sheet and the text. Merged with code checks (digit numbers not in the sheet, roster names not in the sheet, raw field names). |
| Translator | `agents/crew.py` | `gpt-4.1-mini`. ES and FR from the checked EN text. Code verifies digits and names survived and the language is right; the Fact-Checker then reads the translation too. One retry. |
| Revise loop | `Crew.narrate` | draft, check, up to **2 revisions**, then a deterministic template built only from fact-sheet values (never needed on the demo match). |
| Precompute | `agents/precompute.py` | `uv run python -m agents.precompute --match candidate-2004` writes `narratives/<lang>_<mode>.jsonl` (the `Narrative` contract, `t` = shot time + 1.5 s, `evidence` = the numeric facts the text uses, `fact_check`, `voice`, `audio_url`), `narratives/audio/<moment>_<lang>_<mode>.mp3`, and a cache (`narratives/.cache`, audio `.sha` stamps), so a rerun costs nothing unless prompts or facts change. |
| Server | `server/narration.py`, `server/session.py`, `server/app.py` | Loads `narratives/*.jsonl` at startup. A session gets the `NarrativeMsg` for its own ViewerProfile language and mode once the clock passes `t` (same replay semantics as events and moments; a profile change takes effect immediately). `GET /api/narratives?language=&mode=` returns the list. Audio: `Narrative.audio_url` = `/api/files/narratives/audio/<file>.mp3` (the contract has this field; the existing files route serves it). |

## Number traceability

Rule: **every number written with digits in a final narrative appears in that moment's fact sheet**, which only
`agents/stats.py` produces. Enforced three ways: the Storyteller can only get numbers from its tool; code rejects any
other number during the loop (`checks.untraceable_numbers`); the test `agents/tests/test_modes.py` re-checks all 96
generated narratives (EN, ES, FR) against the fact sheets, and the same test requires `verified=true` and the right
language for each. Limits: the check is set-based, so an injected number that equals another fact (12 goals vs a 12 m
shot) is not caught by code, only by the LLM Fact-Checker; number words are not parsed by code (the prompt forbids
them and the LLM checker looks for them).

## Fact-Checker catch rate (measured)

`uv run python -m agents.eval_factcheck --record` (live). 20 true narratives were generated from 10 moments in both
modes; 16 passed the code checks and are the "true" set. From each, false variants were injected: swapped numbers,
numbers taken from a different fact, an invented number, a name outside the sheet, a fabricated name, two names
swapped, a team swapped, an invented claim, a contradiction with the outcome.

| | caught |
|---|---|
| LLM Fact-Checker alone (`gpt-5-mini`) | **137 / 138 = 99.3%** |
| Code checks alone | 61 / 138 = 44.2% (number injections except value collisions, and names outside the sheet; blind to the rest) |
| Combined (what the loop uses) | **137 / 138 = 99.3%** |
| True narratives wrongly rejected | 0 / 16 |

The one miss was a swapped team name (4 of 5 caught). Every other category was 100%: number swap 32/32, number from
another fact 11/11, invented number 16/16, name not in sheet 15/15, fabricated name 15/15, names swapped 14/14,
invented claim 16/16, contradiction 14/14. Caveat: the injections are mechanical, so subtler real errors are not
covered. The offline test `agents/tests/test_factcheck.py` replays the recorded verdicts (fixture
`agents/tests/fixtures/factcheck_cases.json`, gate >= 95% and 0 false rejections); `pytest -m live` asks the real model
about 6 of the cases.

## Analyst versus casual (measured on the 16 demo narratives per language and mode)

| | words (mean) | numbers per narrative | numbers per sentence |
|---|---|---|---|
| EN analyst | 40.0 | 9.0 | 4.48 |
| EN casual | 22.8 | 2.1 | 1.28 |
| ES analyst / casual | 47.6 / 25.9 | 9.0 / 2.1 | 4.48 / 1.28 |
| FR analyst / casual | 48.1 / 26.6 | 9.0 / 2.1 | 4.48 / 1.28 |

Analyst is 1.8x longer and 3.5x denser in numbers per sentence. `agents.mode_stats` reproduces the table; the test
asserts at least 1.4x words and 2x numbers per sentence in every language. Numbers per narrative are identical across
languages because translation preserves digits (checked).

## Run and cost

Precompute for the demo match (16 moments x 2 modes x 3 languages = 96 narratives, 96 mp3, 10 MB): about 100 s wall
clock with 6 parallel workers, roughly 250 model requests and 96 TTS calls. Of the 96, three
English drafts and three translations needed a second attempt, none a third and none used the fallback template.

Live per-moment latency (not on the live path yet): Storyteller with tool call 7 to 11 s on `gpt-5-mini`, translation 1.2 to
1.7 s on `gpt-4.1-mini`. Precompute exists precisely to keep this off the viewing path.

## Not done

Moment Scout, Tactician, Director and Personal Producer agents (Step 8 roles beyond the three above); tracing to
Application Insights; narratives for decision moments or rewind verdicts; club-perspective narration
(`club_perspective` is left `None`); live on-demand narration for moments that were not precomputed.
