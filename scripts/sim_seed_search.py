"""Seed search (PLAN.md Step 3): simulate many matches, score them for drama, save the best as recordings.

    uv run python scripts/sim_seed_search.py --seeds 2000-2099 --top 5 --workers 9

Writes $REGOLAZO_DATA_DIR/matches/candidate-<seed>/ (full recording with fork snapshots) plus summary.json there,
and a ranking of every simulated seed to $REGOLAZO_DATA_DIR/matches/seed_search.json. It does not pick the demo
match: that is the lead's and the user's call.
"""

import argparse
import json
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.model_api import load_models
from sim.drama import Drama, score_match
from sim.record import matches_dir, simulate, write_match
from sim.squads import generate_meta


def _score(seed: int) -> Drama:
    models = load_models()
    meta = generate_meta(seed, f"candidate-{seed}", models.physical_priors)
    res = simulate(meta, models, record=False, snapshots=False, decisions=True)
    return score_match(seed, res.events, res.match.decisions)


def _save(seed: int) -> str:
    models = load_models()
    meta = generate_meta(seed, f"candidate-{seed}", models.physical_priors)
    res = simulate(meta, models)
    out = matches_dir() / meta.match_id
    write_match(res, out)
    drama = score_match(seed, res.events, res.match.decisions)
    summary = {
        "match_id": meta.match_id,
        "seed": seed,
        "home": meta.home.name,
        "away": meta.away.name,
        "venue": meta.venue,
        **asdict(drama),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return str(out)


def _seeds(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", default="2000-2047")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--workers", type=int, default=9)
    args = ap.parse_args()
    seeds = _seeds(args.seeds)
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
        ranked = sorted(ex.map(_score, seeds), key=lambda d: -d.score)
        matches_dir().mkdir(parents=True, exist_ok=True)
        (matches_dir() / "seed_search.json").write_text(json.dumps([asdict(d) for d in ranked], indent=1) + "\n")
        best = [d.seed for d in ranked[: args.top]]
        paths = list(ex.map(_save, best))
    for d, p in zip(ranked[: args.top], paths, strict=True):
        goals = ", ".join(f"{g['minute']}' {g['team'][0].upper()} {g['technique']}" for g in d.goals)
        print(f"{d.score:.2f}  seed {d.seed}  {d.final[0]}-{d.final[1]}  shots {d.shots}  rich {d.rich_moments}  "
              f"{'; '.join(d.notes) or '-'}  [{goals}]  -> {p}")  # fmt: skip


if __name__ == "__main__":
    main()
