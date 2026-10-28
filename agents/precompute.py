"""Precompute narratives and voice for a recorded match.

    uv run python -m agents.precompute --match candidate-2004 [--langs en,es,fr] [--modes analyst,casual]

Writes, under ``$REGOLAZO_DATA_DIR/matches/<match>/narratives/``:
    <lang>_<mode>.jsonl                one ``Narrative`` per line, ordered by ``t``
    audio/<moment>_<lang>_<mode>.mp3   Azure neural voice per narrative (``Narrative.audio_url`` points at it)
    .cache/                            agent results (rerun is free) keyed by prompt+facts hash
"""

import argparse
import asyncio
import hashlib
import json
import logging
from pathlib import Path
from typing import cast

from agents.crew import MODE_STYLE, STORYTELLER_RULES, Crew, Narration
from agents.llm import VOICES, load_env, tts
from agents.stats import MatchData, Moment, numbers_in
from server.config import data_dir
from shared.schemas.common import ViewerMode
from shared.schemas.narrative import FactCheck, Narrative, NarrativeKind

log = logging.getLogger("regolazo.precompute")

PRIORITY = {"goal": 0, "save": 1, "big_chance": 2, "full_time": 3, "half_time": 4, "chance": 5}
KIND: dict[str, NarrativeKind] = {
    "goal": "live_commentary",
    "save": "live_commentary",
    "big_chance": "live_commentary",
    "chance": "live_commentary",
    "half_time": "milestone",
    "full_time": "recap",
}
AUDIO_URL = "/api/files/narratives/audio/{name}"


def narrative_id(moment_id: str, language: str, mode: str) -> str:
    return f"nar_{moment_id}_{language}_{mode}"


def select_moments(moments: list[Moment], limit: int | None) -> list[Moment]:
    if limit is None or limit >= len(moments):
        return moments
    keep = sorted(moments, key=lambda m: (PRIORITY[m.kind], m.t))[:limit]
    return sorted(keep, key=lambda m: m.t)


def build_narrative(moment: Moment, language: str, mode: ViewerMode, n: Narration, source_id: str | None) -> Narrative:
    used = numbers_in(n.text)
    evidence = [e for e in moment.facts.evidence() if numbers_in(str(e.value)) <= used]
    return Narrative(
        id=narrative_id(moment.id, language, mode),
        kind=KIND[moment.kind],
        t=moment.t,
        language=language,
        mode=mode,
        text=n.text,
        evidence=evidence,
        fact_check=FactCheck(verified=n.verified, attempts=n.attempts, issues=n.issues),
        source_id=source_id,
        voice=VOICES[language][1] if language in VOICES else None,
    )


class Cache:
    """Per-(moment, mode, language) agent results on disk, invalidated when prompts or facts change."""

    def __init__(self, directory: Path) -> None:
        self.dir = directory
        directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def stamp(moment: Moment, mode: str) -> str:
        blob = f"{STORYTELLER_RULES}\n{MODE_STYLE[mode]}\n{moment.facts.render()}"
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def get(self, moment: Moment, mode: str, language: str) -> Narration | None:
        path = self.dir / f"{moment.id}_{mode}_{language}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        if data["stamp"] != self.stamp(moment, mode):
            return None
        return Narration(**data["narration"])

    def put(self, moment: Moment, mode: str, language: str, n: Narration) -> None:
        path = self.dir / f"{moment.id}_{mode}_{language}.json"
        path.write_text(
            json.dumps({"stamp": self.stamp(moment, mode), "narration": n.__dict__}, ensure_ascii=False, indent=1)
        )


async def precompute(
    match_dir: Path,
    langs: list[str],
    modes: list[ViewerMode],
    limit: int | None = None,
    voice: bool = True,
    concurrency: int = 6,
) -> dict[str, int]:
    data = MatchData(match_dir)
    moments = select_moments(data.key_moments(), limit)
    out = match_dir / "narratives"
    cache = Cache(out / ".cache")
    crew = Crew({m.id: m.facts for m in moments}, data.all_names())
    sem = asyncio.Semaphore(concurrency)
    stats = {"narratives": 0, "audio": 0, "cached": 0, "fallbacks": 0, "errors": 0}
    results: dict[tuple[str, str, str], Narrative] = {}

    async def one(moment: Moment, mode: ViewerMode) -> None:
        async with sem:
            try:
                en = cache.get(moment, mode, "en")
                if en is None:
                    en = await crew.narrate(moment, mode)
                    cache.put(moment, mode, "en", en)
                else:
                    stats["cached"] += 1
                stats["fallbacks"] += int(en.fallback)
                results[(moment.id, "en", mode)] = build_narrative(moment, "en", mode, en, None)
                for lang in (la for la in langs if la != "en"):
                    tr = cache.get(moment, mode, lang)
                    if tr is None:
                        tr = await crew.translate(en.text, lang, mode, moment.facts)
                        cache.put(moment, mode, lang, tr)
                    results[(moment.id, lang, mode)] = build_narrative(
                        moment, lang, mode, tr, narrative_id(moment.id, "en", mode)
                    )
            except Exception:
                stats["errors"] += 1
                log.exception("moment %s (%s) failed", moment.id, mode)

    await asyncio.gather(*(one(m, mode) for m in moments for mode in modes))

    if voice:
        load_env()
        audio_dir = out / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        async def speak(key: tuple[str, str, str], nar: Narrative) -> None:
            name = f"{key[0]}_{key[1]}_{key[2]}.mp3"
            path = audio_dir / name
            async with sem:
                try:
                    if not path.exists():
                        path.write_bytes(await tts(nar.text, nar.language))
                        stats["audio"] += 1
                    results[key] = nar.model_copy(update={"audio_url": AUDIO_URL.format(name=name)})
                except Exception:
                    stats["errors"] += 1
                    log.exception("tts failed for %s", nar.id)

        await asyncio.gather(*(speak(k, v) for k, v in list(results.items())))

    for lang in langs:
        for mode in modes:
            rows = sorted((v for k, v in results.items() if k[1] == lang and k[2] == mode), key=lambda n: n.t)
            (out / f"{lang}_{mode}.jsonl").write_text("".join(r.model_dump_json() + "\n" for r in rows))
            stats["narratives"] += len(rows)
    return stats


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m agents.precompute", description=__doc__)
    ap.add_argument("--match", required=True, help="match id under $REGOLAZO_DATA_DIR/matches")
    ap.add_argument("--langs", default="en,es,fr")
    ap.add_argument("--modes", default="analyst,casual")
    ap.add_argument("--max-moments", type=int, default=None, help="keep the N highest-priority moments")
    ap.add_argument("--no-voice", action="store_true", help="skip Azure Speech")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    load_env()
    stats = asyncio.run(
        precompute(
            data_dir() / "matches" / args.match,
            args.langs.split(","),
            cast("list[ViewerMode]", [m for m in args.modes.split(",") if m in ("analyst", "casual")]),
            args.max_moments,
            not args.no_voice,
        )
    )
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
