# ReGolazo: submission pitch

- Demo video: https://youtu.be/-GfUATd-dwQ
- Repo: https://github.com/Ka1eidosc0pe/regolazo

## Elevator pitch (one line)

Every fan has yelled at the screen. ReGolazo lets the screen answer back: rewind any decision, make your own call, and watch that alternate reality play out, with outcomes from real football analytics, not an LLM.

## Short description (about 150 words)

ReGolazo is an AI broadcast crew for synthetic football matches. A full match engine generates realistic matches. A vision model, the Spotter, watches the rendered broadcast and logs passes, shots and sprints straight from the pixels, with its accuracy against ground truth shown live. Agents built on Microsoft Agent Framework and Azure AI Foundry write commentary in English, Spanish and French, in analyst or casual style. A Fact-Checker agent verifies every number before it goes out, and Azure AI Speech voices it.

The signature feature is **Rewind & Rewrite**:
- A Moment Scout flags the match's decision moments without judging them.
- The fan pauses, sees who controlled which space, and picks a different pass or shot.
- The engine forks from that exact frame and plays the new reality, with the real play ghosted alongside.
- A verdict compares the two, and "Run 1,000 realities" shows the full distribution of outcomes.

## Microsoft technologies

- **Azure AI Foundry / Azure OpenAI:** `gpt-5-mini` (Storyteller, Fact-Checker) and `gpt-4.1-mini` (Translator).
- **Microsoft Agent Framework (Python):** agent orchestration, tool calling, structured outputs.
- **Azure AI Speech:** neural TTS in three languages (en-GB-RyanNeural, es-ES-AlvaroNeural, fr-FR-HenriNeural).
