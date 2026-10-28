"""Deterministic narrative checks: numbers, names and language. No LLM; these run before and after every agent.

The number rule: every number written with digits in a narrative must also appear in the moment's fact sheet
(``FactSheet.numbers``), which only ``agents.stats`` produces. Number *words* are not parsed; the Storyteller prompt
forbids them and the LLM Fact-Checker looks for them.
"""

import re

from agents.stats import FactSheet, numbers_in

_TOL = 1e-9

_STOPWORDS = {
    "en": {"the", "and", "of", "to", "in", "is", "with", "from", "for", "that", "his", "they", "this", "at", "on"},
    "es": {"el", "la", "de", "que", "y", "en", "los", "con", "una", "por", "del", "para", "su", "se", "un", "las"},
    "fr": {"le", "la", "de", "et", "les", "des", "un", "une", "du", "en", "que", "pour", "dans", "sur", "son", "au"},
}


def untraceable_numbers(text: str, sheet: FactSheet) -> list[float]:
    """Numbers in ``text`` that are not in the fact sheet."""
    allowed = sheet.numbers()
    return sorted(n for n in numbers_in(text) if not any(abs(n - a) < _TOL for a in allowed))


def mentioned_names(text: str, roster: set[str]) -> set[str]:
    return {n for n in roster if re.search(rf"(?<!\w){re.escape(n)}(?!\w)", text)}


def check_text(text: str, sheet: FactSheet, roster: set[str]) -> list[str]:
    """Issues found by code: untraceable numbers and names of people/teams that are not in this fact sheet."""
    issues: list[str] = []
    for n in untraceable_numbers(text, sheet):
        issues.append(f"number {n:g} is not in the fact sheet")
    allowed = {v for v in sheet.facts.values() if isinstance(v, str)}
    allowed_blob = " ".join(allowed)
    for name in sorted(mentioned_names(text, roster)):
        if name not in allowed_blob:
            issues.append(f"name {name!r} is not in the fact sheet")
    if "_" in text:
        issues.append("text contains raw field names (underscores); write natural sentences")
    if not text.strip():
        issues.append("empty text")
    return issues


def preserved(source: str, translation: str, roster: set[str]) -> list[str]:
    """Issues if a translation changed the numbers or dropped a name that the source had."""
    issues: list[str] = []
    src_nums, tr_nums = numbers_in(source), numbers_in(translation)
    for n in sorted(src_nums - tr_nums):
        issues.append(f"number {n:g} missing from the translation")
    for n in sorted(tr_nums - src_nums):
        issues.append(f"number {n:g} appears only in the translation")
    for name in sorted(mentioned_names(source, roster) - mentioned_names(translation, roster)):
        issues.append(f"name {name!r} missing from the translation")
    return issues


def detect_language(text: str) -> str:
    """Tiny stopword-based detector for en / es / fr (enough to catch a translator answering in the wrong language)."""
    words = re.findall(r"[a-zà-ÿ']+", text.lower())
    scores = {lang: sum(w in sw for w in words) for lang, sw in _STOPWORDS.items()}
    return max(scores, key=lambda k: scores[k])


def words(text: str) -> int:
    return len(re.findall(r"\S+", text))


def sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?¡¿])\s+", text.strip()) if s]
