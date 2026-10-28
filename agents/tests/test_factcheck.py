"""Fact-Checker catch rate on injected false numbers and names.

Offline: replays the LLM verdicts recorded by ``python -m agents.eval_factcheck --record`` (live gpt-5-mini) and
re-runs the code checks. Live (``-m live``): asks the real Fact-Checker about a handful of cases.
"""

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from agents.crew import Crew
from agents.eval_factcheck import metrics, sheet_from

FIXTURE = Path(__file__).parent / "fixtures" / "factcheck_cases.json"


def load() -> tuple[set[str], list[dict[str, Any]]]:
    data = json.loads(FIXTURE.read_text())
    return set(data["roster"]), data["cases"]


def test_recorded_catch_rate_meets_gate() -> None:
    roster, rows = load()
    m = metrics(rows, roster)
    assert m["n_false"] >= 100 and m["n_true"] >= 10
    assert m["catch_combined"] >= 0.95, m
    assert m["false_rejection_combined"] == 0.0, m
    assert m["catch_combined"] >= m["catch_llm_only"]


def test_code_checks_alone_catch_nearly_all_number_injections() -> None:
    """The set-based number check misses only injected values that collide with another fact (12 goals vs 12 m)."""
    from agents.checks import check_text

    roster, rows = load()
    numeric = [r for r in rows if r["mutation"] in ("number_swap", "invented_number")]
    caught = [bool(check_text(r["text"], sheet_from(r["facts"], r["moment_id"]), roster)) for r in numeric]
    assert sum(caught) / len(caught) >= 0.95
    assert all(not r["llm_verified"] for r, c in zip(numeric, caught, strict=True) if not c), "LLM covers collisions"


@pytest.mark.live
def test_live_fact_checker_flags_false_cases_and_passes_true_ones() -> None:
    from agents.llm import load_env

    load_env()
    roster, rows = load()
    pick = [r for r in rows if r["label"] == "true"][:2]
    for kind in ("number_swap", "names_swapped", "invented_claim", "contradiction"):
        pick += [r for r in rows if r["mutation"] == kind][:1]
    sheets = {r["moment_id"]: sheet_from(r["facts"], r["moment_id"]) for r in pick}
    crew = Crew(sheets, roster)

    async def go() -> list[bool]:
        verdicts = await asyncio.gather(*(crew.llm_verdict(r["text"], sheets[r["moment_id"]]) for r in pick))
        return [v.verified for v in verdicts]

    got = asyncio.run(go())
    for r, verified in zip(pick, got, strict=True):
        assert verified == (r["label"] == "true"), (r["mutation"], r["text"])
