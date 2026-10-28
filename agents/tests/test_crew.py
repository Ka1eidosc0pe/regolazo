"""Crew logic with a scripted LLM (no network): the revise loop, the fallback, translation checks, traceability."""

import asyncio

from pydantic import BaseModel

from agents.checks import check_text, detect_language, preserved, untraceable_numbers
from agents.crew import MAX_REVISIONS, Crew, Verdict, template_text
from agents.stats import MatchData, Moment, numbers_in


class Script:
    """Scripted Crew.ask: storyteller drafts are consumed in order; the checker approves everything the model says."""

    def __init__(self, drafts: list[str], translations: list[str] | None = None) -> None:
        self.drafts = list(drafts)
        self.translations = list(translations or [])
        self.calls: list[tuple[str, str]] = []

    async def __call__(self, role: str, prompt: str, response_format: type[BaseModel] | None) -> str:
        self.calls.append((role, prompt))
        if role == "storyteller":
            return self.drafts.pop(0)
        if role == "translator":
            return self.translations.pop(0)
        return Verdict(verified=True).model_dump_json()  # a lenient LLM checker: the code checks must still bite


def goal(match: MatchData) -> Moment:
    return next(m for m in match.key_moments() if m.kind == "goal")


def crew_for(match: MatchData, script: Script) -> Crew:
    return Crew({m.id: m.facts for m in match.key_moments()}, match.all_names(), script)


def test_clean_draft_passes_first_time(match: MatchData) -> None:
    m = goal(match)
    who, team = m.facts.facts["shooter"], m.facts.facts["team"]
    script = Script([f"Minute 1: {who} scores for {team}, xG 0.31."])
    n = asyncio.run(crew_for(match, script).narrate(m, "analyst"))
    assert n.verified and n.attempts == 1 and not n.fallback
    assert [r for r, _ in script.calls] == ["storyteller", "fact_checker"]


def test_false_number_triggers_revision_then_passes(match: MatchData) -> None:
    m = goal(match)
    who = m.facts.facts["shooter"]
    script = Script([f"{who} scores, xG 0.55.", f"{who} scores, xG 0.31."])
    n = asyncio.run(crew_for(match, script).narrate(m, "analyst"))
    assert n.attempts == 2 and n.verified and "0.31" in n.text
    revise_prompt = script.calls[2][1]
    assert "0.55" in revise_prompt and "rejected" in revise_prompt, "feedback must reach the Storyteller"


def test_loop_is_bounded_and_falls_back_to_template(match: MatchData) -> None:
    m = goal(match)
    script = Script(["xG 0.99 for Nobody."] * (MAX_REVISIONS + 1))
    n = asyncio.run(crew_for(match, script).narrate(m, "analyst"))
    assert n.fallback and n.attempts == MAX_REVISIONS + 1
    assert [r for r, _ in script.calls].count("storyteller") == MAX_REVISIONS + 1
    assert not untraceable_numbers(n.text, m.facts), "fallback text is built from the fact sheet only"


def test_wrong_name_and_underscores_are_rejected(match: MatchData) -> None:
    m = goal(match)
    other = next(n for n in match.all_names() if n not in m.facts.names())
    issues = check_text(f"{other} scores in minute 1, xG 0.31", m.facts, match.all_names())
    assert any(other in i for i in issues)
    assert check_text("shot_speed_kmh 90", m.facts, match.all_names())


def test_every_template_number_traces_to_the_fact_sheet(match: MatchData) -> None:
    for m in match.key_moments():
        for mode in ("analyst", "casual"):
            text = template_text(m.facts, mode)
            assert numbers_in(text) <= m.facts.numbers(), (m.id, text)
            assert not check_text(text, m.facts, match.all_names()), (m.id, text)


def test_translation_checked_for_numbers_names_and_language(match: MatchData) -> None:
    m = goal(match)
    who = m.facts.facts["shooter"]
    en = f"Minute 1: {who} scores for the home side, xG 0.31."
    good = f"Minuto 1: {who} marca para el equipo local, xG 0.31 y la jugada de la banda."
    script = Script([], [f"Minuto 1: {who} marca, xG 0.35 para el equipo.", good])
    n = asyncio.run(crew_for(match, script).translate(en, "es", "analyst", m.facts))
    assert n.verified and n.attempts == 2 and n.text == good
    assert preserved(en, good, match.all_names()) == []
    assert preserved(en, "Minuto 1: marca, xG 0.35", match.all_names())  # name dropped, number changed
    assert detect_language(good) == "es" and detect_language(en) == "en"
