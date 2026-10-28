"""Measure the Fact-Checker: inject false numbers and names into true narratives and count what it catches.

    uv run python -m agents.eval_factcheck --match candidate-2004 --record   # live; writes the fixture + prints metrics

The fixture (``agents/tests/fixtures/factcheck_cases.json``) stores every case with the recorded LLM verdict, so
``agents/tests/test_factcheck.py`` re-derives the same metrics offline (the code checks are re-run, the LLM verdict
is replayed).
"""

import argparse
import asyncio
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agents.checks import check_text
from agents.crew import Crew
from agents.llm import load_env
from agents.stats import FactSheet, MatchData, Moment
from server.config import data_dir

FIXTURE = Path(__file__).parent / "tests" / "fixtures" / "factcheck_cases.json"
NUM = re.compile(r"\d+(?:\.\d+)?")


@dataclass
class Case:
    moment_id: str
    mode: str
    label: str  # "true" or "false"
    mutation: str
    text: str


def _bump(token: str, allowed: set[float]) -> str:
    """A different number with the same number of decimals, not present in the fact sheet."""
    decimals = len(token.split(".")[1]) if "." in token else 0
    step = 0.17 if decimals else 7.0
    value = float(token)
    for k in range(1, 20):
        cand = value + step * k
        text = f"{cand:.{decimals}f}"
        if float(text) not in allowed and float(text) != value:
            return text
    return token


def mutations(text: str, moment: Moment, roster: set[str], rng: random.Random) -> list[tuple[str, str]]:
    """(mutation name, false text) pairs derived from one true text."""
    sheet = moment.facts
    allowed = sheet.numbers()
    out: list[tuple[str, str]] = []
    tokens = [m for m in NUM.finditer(text) if float(m.group()) in allowed]
    for m in rng.sample(tokens, min(2, len(tokens))):
        new = _bump(m.group(), allowed)
        out.append(("number_swap", text[: m.start()] + new + text[m.end() :]))
    in_text = sorted(n for n in sheet.names() if n in text and n in roster)
    people = [n for n in in_text if n not in (sheet.facts.get("home_team"), sheet.facts.get("away_team"))]
    outsiders = sorted(roster - set(sheet.names()))
    if people:
        out.append(("name_not_in_sheet", text.replace(people[0], rng.choice(outsiders))))
        out.append(("fabricated_name", text.replace(people[0], "Martinez")))
    if len(people) >= 2:
        a, b = people[0], people[1]
        out.append(("names_swapped", text.replace(a, "\0").replace(b, a).replace("\0", b)))
    out.append(("invented_number", text.rstrip() + " He has scored 12 goals this season."))
    out.append(("invented_claim", text.rstrip() + " It was his first goal for the club and the crowd erupted."))
    kind = moment.kind
    if kind == "goal":
        out.append(("contradiction", text.rstrip() + " The goalkeeper saved it comfortably."))
    elif kind in ("save", "big_chance", "chance"):
        out.append(("contradiction", text.rstrip() + " It ended up in the back of the net for a goal."))
    return out


def sheet_from(facts: dict[str, Any], moment_id: str) -> FactSheet:
    return FactSheet(moment_id, dict(facts), {k: ("stat", k) for k in facts})


def metrics(rows: list[dict[str, Any]], roster: set[str]) -> dict[str, Any]:
    """Catch rates from recorded rows: LLM only, code only, combined; plus false rejections on true texts."""
    out: dict[str, Any] = {}
    false = [r for r in rows if r["label"] == "false"]
    true = [r for r in rows if r["label"] == "true"]

    def code_issues(r: dict[str, Any]) -> list[str]:
        return check_text(r["text"], sheet_from(r["facts"], r["moment_id"]), roster)

    def rate(flags: list[bool]) -> float:
        return round(sum(flags) / len(flags), 4) if flags else float("nan")

    out["n_false"], out["n_true"] = len(false), len(true)
    out["catch_llm_only"] = rate([not r["llm_verified"] for r in false])
    out["catch_code_only"] = rate([bool(code_issues(r)) for r in false])
    out["catch_combined"] = rate([(not r["llm_verified"]) or bool(code_issues(r)) for r in false])
    out["false_rejection_combined"] = rate([(not r["llm_verified"]) or bool(code_issues(r)) for r in true])
    by: dict[str, list[bool]] = {}
    for r in false:
        by.setdefault(r["mutation"], []).append((not r["llm_verified"]) or bool(code_issues(r)))
    out["by_mutation_combined"] = {k: [sum(v), len(v)] for k, v in sorted(by.items())}
    by_llm: dict[str, list[bool]] = {}
    for r in false:
        by_llm.setdefault(r["mutation"], []).append(not r["llm_verified"])
    out["by_mutation_llm_only"] = {k: [sum(v), len(v)] for k, v in sorted(by_llm.items())}
    return out


async def run(match: str, n_moments: int, record: bool) -> dict[str, Any]:
    load_env()
    data = MatchData(data_dir() / "matches" / match)
    all_moments = data.key_moments()
    chosen = [m for m in all_moments if m.kind in ("goal", "save", "big_chance", "full_time")][:n_moments]
    crew = Crew({m.id: m.facts for m in all_moments}, data.all_names())
    rng = random.Random(7)
    sem = asyncio.Semaphore(8)
    roster = data.all_names()

    async def gen(m: Moment, mode: str) -> list[Case]:
        async with sem:
            text = await crew.story(m, mode)  # type: ignore[arg-type]
        cases: list[Case] = []
        if not check_text(text, m.facts, roster):
            cases.append(Case(m.id, mode, "true", "none", text))
        cases += [Case(m.id, mode, "false", name, bad) for name, bad in mutations(text, m, roster, rng)]
        return cases

    batches = await asyncio.gather(*(gen(m, mode) for m in chosen for mode in ("analyst", "casual")))
    cases = [c for b in batches for c in b]
    sheets = {m.id: m.facts for m in all_moments}

    async def judge(c: Case) -> dict[str, Any]:
        async with sem:
            v = await crew.llm_verdict(c.text, sheets[c.moment_id])
        return {
            "moment_id": c.moment_id,
            "mode": c.mode,
            "label": c.label,
            "mutation": c.mutation,
            "text": c.text,
            "facts": sheets[c.moment_id].facts,
            "llm_verified": v.verified,
            "llm_issues": v.issues,
        }

    rows = await asyncio.gather(*(judge(c) for c in cases))
    if record:
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE.write_text(
            json.dumps({"roster": sorted(roster), "cases": rows}, ensure_ascii=False, indent=1, sort_keys=True)
        )
    return metrics(list(rows), roster)


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m agents.eval_factcheck", description=__doc__)
    ap.add_argument("--match", default="candidate-2004")
    ap.add_argument("--moments", type=int, default=10)
    ap.add_argument("--record", action="store_true", help="write the replay fixture used by the offline test")
    args = ap.parse_args()
    import logging

    logging.getLogger("httpx").setLevel(logging.WARNING)
    print(json.dumps(asyncio.run(run(args.match, args.moments, args.record)), indent=1))


if __name__ == "__main__":
    main()
