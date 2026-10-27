"""Turn the raw StatsBomb cache into compact training tables (``$REGOLAZO_DATA_DIR/cache/analytics/*.npz``).

Run with: uv run python -m analytics.build_dataset
"""

import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

from analytics.data import statsbomb as sb
from analytics.paths import data_dir

TABLES = ("shots", "passes", "moves", "team_stats")


def cache_dir() -> Path:
    return data_dir() / "cache" / "analytics"


def _one(info: sb.MatchInfo) -> dict[str, sb.Table]:
    events = sb.load_events(info.match_id)
    frames = sb.load_frames_360(info.match_id) if info.has_360 else {}
    shots = sb.rows_to_table(sb.extract_shots(events, info.match_id))
    passes = sb.rows_to_table(sb.extract_passes(events, frames))
    moves = sb.rows_to_table(sb.extract_moves(events))
    stats = sb.rows_to_table(sb.team_match_stats(events, info))
    for t in (shots, passes, moves):
        if t:
            n = len(next(iter(t.values())))
            t["match_id"] = np.full(n, float(info.match_id))
            t["competition_id"] = np.full(n, float(info.competition_id))
            t["season_id"] = np.full(n, float(info.season_id))
    return {"shots": shots, "passes": passes, "moves": moves, "team_stats": stats}


def build() -> dict[str, sb.Table]:
    matches = sb.list_matches()
    t0 = time.time()
    with ProcessPoolExecutor() as pool:
        results = list(pool.map(_one, matches, chunksize=4))
    out = {name: sb.concat_tables([r[name] for r in results]) for name in TABLES}
    cache_dir().mkdir(parents=True, exist_ok=True)
    for name, table in out.items():
        np.savez_compressed(cache_dir() / f"{name}.npz", **table)  # type: ignore[arg-type]
    infos = [asdict(m) for m in matches]
    (cache_dir() / "matches.json").write_text(json.dumps(infos))
    sizes = {k: len(next(iter(v.values()))) if v else 0 for k, v in out.items()}
    print(f"built {sizes} from {len(matches)} matches in {time.time() - t0:.0f}s -> {cache_dir()}")
    return out


def load_table(name: str) -> sb.Table:
    with np.load(cache_dir() / f"{name}.npz") as z:
        return {k: z[k] for k in z.files}


if __name__ == "__main__":
    build()
