"""Mode differentiation and number traceability on the precomputed demo narratives (skipped if not generated)."""

import json

import pytest

from agents.checks import detect_language
from agents.mode_stats import load, summarize
from agents.stats import MatchData
from server.config import data_dir

MATCH = data_dir() / "matches" / "candidate-2004"
NARRATIVES = MATCH / "narratives"

pytestmark = pytest.mark.skipif(
    not (NARRATIVES / "fr_casual.jsonl").exists(), reason="run python -m agents.precompute --match candidate-2004"
)


def test_analyst_and_casual_differ_materially_in_every_language() -> None:
    for lang in ("en", "es", "fr"):
        a, c = summarize(load(NARRATIVES, lang, "analyst")), summarize(load(NARRATIVES, lang, "casual"))
        assert a["words_mean"] > 1.4 * c["words_mean"], (lang, a, c)
        assert a["numbers_per_sentence"] > 2 * c["numbers_per_sentence"], (lang, a, c)


def test_every_number_in_every_narrative_is_in_its_fact_sheet() -> None:
    sheets = {m.id: m.facts for m in MatchData(MATCH).key_moments()}
    from agents.stats import numbers_in

    n = 0
    for path in sorted(NARRATIVES.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            moment_id = row["id"].removeprefix("nar_").rsplit("_", 2)[0]
            assert numbers_in(row["text"]) <= sheets[moment_id].numbers(), row["id"]
            assert row["fact_check"]["verified"] is True
            assert detect_language(row["text"]) == row["language"], row["text"]
            n += 1
    assert n >= 60


def test_audio_exists_for_every_narrative() -> None:
    for path in sorted(NARRATIVES.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            audio = json.loads(line)["audio_url"]
            assert audio and (MATCH / audio.removeprefix("/api/files/")).stat().st_size > 1000
