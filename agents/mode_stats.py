"""Measure how differently analyst and casual narratives read: words, numbers per sentence, numbers per narrative.

uv run python -m agents.mode_stats --match candidate-2004 [--show]
"""

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from agents.checks import sentences, words
from agents.stats import numbers_in
from server.config import data_dir


def load(narratives: Path, language: str, mode: str) -> list[dict[str, Any]]:
    path = narratives / f"{language}_{mode}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    texts = [r["text"] for r in rows]
    n_sent = [max(len(sentences(t)), 1) for t in texts]
    n_num = [len(numbers_in(t)) for t in texts]
    return {
        "n": len(rows),
        "words_mean": round(mean(words(t) for t in texts), 1),
        "numbers_per_narrative": round(mean(n_num), 2),
        "numbers_per_sentence": round(mean(n / s for n, s in zip(n_num, n_sent, strict=True)), 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m agents.mode_stats")
    ap.add_argument("--match", default="candidate-2004")
    ap.add_argument("--show", action="store_true", help="print the EN texts and recaps")
    args = ap.parse_args()
    narratives = data_dir() / "matches" / args.match / "narratives"
    out: dict[str, Any] = {}
    for lang in ("en", "es", "fr"):
        if not (narratives / f"{lang}_analyst.jsonl").exists():
            continue
        for mode in ("analyst", "casual"):
            rows = load(narratives, lang, mode)
            out[f"{lang}_{mode}"] = summarize(rows)
            if args.show:
                for r in rows:
                    if lang == "en" or r["kind"] == "recap":
                        print(f"[{lang} {mode} t={r['t']:.0f} {r['kind']}] {r['text']}")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
