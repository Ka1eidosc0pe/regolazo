"""The commentary crew: Storyteller, Fact-Checker and Translator, built on Microsoft Agent Framework.

- **Storyteller** (gpt-5-mini) has one tool, ``get_moment_facts``; it may only use numbers and names that tool returns.
- **Fact-Checker** (gpt-5-mini, structured output ``Verdict``) verifies every number and name against the fact sheet.
  Code checks (``agents.checks``) run first and their issues are merged into the verdict, so a hallucinated number
  is rejected even if the model misses it.
- **Translator** (gpt-4.1-mini) renders the checked English text in ES / FR; code verifies numbers and names survived.

``Crew.narrate`` is the bounded revise loop: draft, check, up to ``MAX_REVISIONS`` revisions, then a deterministic
template built only from fact-sheet values, so the output is always traceable.

LLM calls go through ``Crew.ask``, which tests replace (``ReplayAsk``) so CI never needs the network.
"""

import hashlib
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

from agent_framework import tool
from pydantic import BaseModel

from agents.checks import check_text, detect_language, preserved
from agents.stats import FactSheet, Moment, fact_sheet_dict
from shared.schemas.common import ViewerMode

MAX_REVISIONS = 2
LANG_NAMES = {"es": "Spanish", "fr": "French"}

STORYTELLER_RULES = """You are the match commentator for a football broadcast.
You write ONLY from the facts returned by the tool get_moment_facts. Call it first, with the moment id you are given.
Hard rules:
- Every number must be written as digits, copied exactly from the facts (for example 0.22, 74, 1-0). Never write a
  quantity as a word (no "one", "two", "twice", "double"), never round, convert, sum or invent a number.
- Use player and team names exactly as in the facts. Never mention a person or team that is not in the facts.
- Say nothing the facts do not support: no crowd, weather, emotions, history, injuries or tactics beyond the facts.
- shot_speed_kmh is the ball speed of the shot; shooter_running_speed_kmh is how fast the shooter was running.
- The text is read aloud by a speech voice: full spoken sentences, say "minute 74" (never 74'), no brackets, dashes
  as separators, bullet points or markdown.
- Write natural sentences. Never print field names or underscores (say "post-shot xG", not post_shot_xg).
- Output only the commentary text: no title, no quotes."""

MODE_STYLE: dict[str, str] = {
    "analyst": (
        "Style: ANALYST. Terse, tactical, number-dense. Use football analytics terms (xG, post-shot xG, shot speed in "
        "km/h, distance, possession, running totals). Pack in as many of the provided figures as fit. 2-3 short "
        "sentences, at most 45 words (recaps: at most 75 words)."
    ),
    "casual": (
        "Style: CASUAL. Warm, plain, friendly, like talking to a fan new to the game. No jargon (do not say xG or "
        "post-shot). Use at most two numbers (minute and score are best). 1-2 short sentences, at most 35 words "
        "(recaps: at most 55 words)."
    ),
}

CHECKER_INSTRUCTIONS = """You are a strict fact-checker for football commentary.
You are given a FACT SHEET and a commentary TEXT. Verify:
1. Every number in the text (digits, and also number words used as quantities) matches a number in the fact sheet,
   with the same meaning (an xG value is not a speed; a minute is not a score).
2. Every player and team name is in the fact sheet and is attributed correctly (shooter, goalkeeper, assist, team).
3. The outcome, score and minute agree with the fact sheet, and nothing unsupported is claimed.
Return verified=true only if all hold. Otherwise verified=false and list each problem in issues, briefly."""

TRANSLATOR_INSTRUCTIONS = """You translate football commentary. Keep every digit, decimal number, score and player or
team name EXACTLY as written (do not translate names, do not change decimals, do not spell numbers out). Keep the same
tone and length. Output only the translation."""


class Verdict(BaseModel):
    verified: bool
    issues: list[str] = []


Ask = Callable[[str, str, type[BaseModel] | None], Awaitable[str]]
"""(role, prompt, response_format) -> text, or the JSON of the structured output."""


@dataclass
class Narration:
    text: str
    verified: bool
    attempts: int
    issues: list[str] = field(default_factory=list)
    fallback: bool = False


def template_text(sheet: FactSheet, mode: ViewerMode) -> str:
    """Deterministic English fallback built only from fact-sheet values."""
    f = sheet.facts
    kind = str(f.get("moment_type", "moment"))
    if kind in ("half time", "full time"):
        label = "Half time" if kind == "half time" else "Full time"
        head = f"{label}: {f['scoreline']}."
        if mode == "casual":
            return head
        return (
            f"{head} Shots {f['home_shots']}-{f['away_shots']}, xG {f['home_xg']}-{f['away_xg']}, "
            f"possession {f['home_possession_pct']}-{f['away_possession_pct']}."
        )
    who = f.get("shooter", "The shooter")
    head = f"Minute {f['minute']}: {who} for {f.get('team', 'the team')}, {f.get('outcome', kind)}."
    return head if mode == "casual" else f"{head} xG {f.get('xg', '')}, score {f['score_now']}."


class Crew:
    def __init__(self, sheets: dict[str, FactSheet], roster: set[str], ask: Ask | None = None) -> None:
        self.sheets = sheets
        self.roster = roster
        self.ask: Ask = ask or self._ask_azure

    # --- Agent Framework backend -------------------------------------------------------------------------

    def _tools(self) -> list[object]:
        sheets = self.sheets

        @tool
        def get_moment_facts(moment_id: Annotated[str, "Id of the moment to narrate"]) -> dict[str, Any]:
            """Return the verified fact sheet (display names and numbers) for one match moment."""
            if moment_id not in sheets:
                return {"error": f"unknown moment {moment_id}"}
            return fact_sheet_dict(sheets[moment_id])

        return [get_moment_facts]

    async def _ask_azure(self, role: str, prompt: str, response_format: type[BaseModel] | None) -> str:
        from agents.llm import FAST_MODEL, MAIN_MODEL, make_agent

        spec = {
            "storyteller": (MAIN_MODEL, STORYTELLER_RULES, self._tools()),
            "fact_checker": (MAIN_MODEL, CHECKER_INSTRUCTIONS, None),
            "translator": (FAST_MODEL, TRANSLATOR_INSTRUCTIONS, None),
        }[role]
        agent = make_agent(spec[0], role, spec[1], spec[2])
        if response_format is not None:
            result = await agent.run(prompt, options={"response_format": response_format})
            value = result.value
            return value.model_dump_json() if isinstance(value, BaseModel) else result.text
        return str((await agent.run(prompt)).text).strip()

    # --- agents ------------------------------------------------------------------------------------------

    async def story(self, moment: Moment, mode: ViewerMode, feedback: tuple[str, list[str]] | None = None) -> str:
        recap = moment.kind in ("half_time", "full_time")
        task = "a short recap" if recap else "a live commentary line for this moment"
        prompt = f"{MODE_STYLE[mode]}\nMoment id: {moment.id} (kind: {moment.kind}). Write {task}."
        if feedback:
            draft, issues = feedback
            problems = "; ".join(issues)
            prompt += (
                f"\nYour previous draft was rejected.\nDraft: {draft}\nProblems: {problems}\nWrite a corrected version."
            )
        return (await self.ask("storyteller", prompt, None)).strip()

    async def llm_verdict(self, text: str, sheet: FactSheet) -> Verdict:
        """The Fact-Checker agent alone (no code checks)."""
        prompt = f"FACT SHEET:\n{sheet.render()}\n\nTEXT:\n{text}"
        return Verdict.model_validate_json(await self.ask("fact_checker", prompt, Verdict))

    async def verdict(self, text: str, sheet: FactSheet) -> Verdict:
        """Fact-Checker verdict merged with the deterministic code checks (either one can reject)."""
        llm = await self.llm_verdict(text, sheet)
        issues = list(llm.issues)
        for issue in check_text(text, sheet, self.roster):
            if issue not in issues:
                issues.append(issue)
        return Verdict(verified=llm.verified and not issues, issues=issues)

    async def narrate(self, moment: Moment, mode: ViewerMode) -> Narration:
        """Storyteller <-> Fact-Checker loop: at most MAX_REVISIONS revisions, then the deterministic template."""
        sheet = moment.facts
        feedback: tuple[str, list[str]] | None = None
        issues: list[str] = []
        for attempt in range(1, MAX_REVISIONS + 2):
            text = await self.story(moment, mode, feedback)
            verdict = await self.verdict(text, sheet)
            if verdict.verified:
                return Narration(text, True, attempt, issues)
            issues = verdict.issues
            feedback = (text, issues)
        return Narration(template_text(sheet, mode), True, MAX_REVISIONS + 1, issues, fallback=True)

    async def translate(self, text: str, language: str, mode: ViewerMode, sheet: FactSheet) -> Narration:
        """Translator: ES / FR from the checked English text; code verifies numbers, names and language survived."""
        if language == "en":
            return Narration(text, True, 1)
        prompt = f"Translate into {LANG_NAMES[language]} ({mode} tone):\n{text}"
        issues: list[str] = []
        for attempt in (1, 2):
            tr = (await self.ask("translator", prompt, None)).strip()
            issues = preserved(text, tr, self.roster)
            issues += [f"wrong language {detect_language(tr)}"] if detect_language(tr) != language else []
            issues += check_text(tr, sheet, self.roster)
            if not issues:
                return Narration(tr, True, attempt)
            problems = "; ".join(issues)
            prompt = (
                f"Translate into {LANG_NAMES[language]} ({mode} tone). "
                f"Previous attempt had problems: {problems}.\n{text}"
            )
        return Narration(tr, False, 2, issues)


class ReplayAsk:
    """Record / replay LLM calls at the Crew.ask boundary, keyed by sha256(role, prompt). Fixtures are JSON."""

    def __init__(self, path: Path, live: Ask | None = None) -> None:
        self.path = path
        self.live = live
        self.cache: dict[str, str] = json.loads(path.read_text()) if path.exists() else {}

    @staticmethod
    def key(role: str, prompt: str) -> str:
        return hashlib.sha256(f"{role}\n{prompt}".encode()).hexdigest()[:24]

    async def __call__(self, role: str, prompt: str, response_format: type[BaseModel] | None) -> str:
        k = self.key(role, prompt)
        if k not in self.cache:
            if self.live is None:
                raise KeyError(f"no recorded response for {role} {k}")
            self.cache[k] = await self.live(role, prompt, response_format)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.cache, indent=1, ensure_ascii=False, sort_keys=True))
        return self.cache[k]
