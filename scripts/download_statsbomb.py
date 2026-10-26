"""Download StatsBomb Open Data (events, lineups, 360 freeze frames) for the competitions the models train on.

Run with: uv run python scripts/download_statsbomb.py [--quick]
Files go to $REGOLAZO_DATA_DIR/raw/statsbomb/ mirroring the open-data repo layout; existing files are skipped.
Data: https://github.com/statsbomb/open-data (StatsBomb Open Data licence, attribution required).
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import httpx

from analytics.paths import statsbomb_dir

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"

# (competition_id, season_id, label). 360 competitions feed xPass; all feed xG, psxG, xT and realism targets.
COMPETITIONS: list[tuple[int, int, str]] = [
    (55, 43, "UEFA Euro 2020 (360)"),
    (43, 106, "FIFA World Cup 2022 (360)"),
    (55, 282, "UEFA Euro 2024 (360)"),
    (9, 281, "1. Bundesliga 2023/24 (360)"),
    (2, 27, "Premier League 2015/16"),
    (43, 3, "FIFA World Cup 2018"),
]
QUICK = [(55, 43, "UEFA Euro 2020 (360)")]


async def fetch(client: httpx.AsyncClient, sem: asyncio.Semaphore, rel: str, dest_root: Path) -> bool:
    """Download ``rel`` once. Returns False when the file does not exist upstream (e.g. no 360 data)."""
    dest = dest_root / rel
    if dest.exists() and dest.stat().st_size > 0:
        return True
    async with sem:
        for attempt in range(4):
            try:
                resp = await client.get(f"{BASE}/{rel}")
                if resp.status_code == 404:
                    return False
                resp.raise_for_status()
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_suffix(".part")
                tmp.write_bytes(resp.content)
                tmp.replace(dest)
                return True
            except (httpx.HTTPError, OSError) as exc:
                if attempt == 3:
                    print(f"failed {rel}: {exc}", file=sys.stderr)
                    raise
                await asyncio.sleep(1.5 * (attempt + 1))
    return False


async def main(quick: bool, concurrency: int) -> None:
    root = statsbomb_dir()
    root.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
        await fetch(client, sem, "competitions.json", root)
        comps = QUICK if quick else COMPETITIONS
        await asyncio.gather(*(fetch(client, sem, f"matches/{c}/{s}.json", root) for c, s, _ in comps))
        jobs: list[str] = []
        for c, s, label in comps:
            matches = json.loads((root / f"matches/{c}/{s}.json").read_text())
            ids = [int(m["match_id"]) for m in matches]
            print(f"{label}: {len(ids)} matches")
            for mid in ids:
                jobs += [f"events/{mid}.json", f"lineups/{mid}.json", f"three-sixty/{mid}.json"]
        done = 0
        missing = 0

        async def run(rel: str) -> None:
            nonlocal done, missing
            ok = await fetch(client, sem, rel, root)
            done += 1
            missing += not ok
            if done % 200 == 0:
                print(f"  {done}/{len(jobs)} files ({missing} not available upstream)")

        await asyncio.gather(*(run(r) for r in jobs))
    print(f"done: {len(jobs)} files checked, {missing} not available upstream, cache at {root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="only Euro 2020 (for a fast smoke run)")
    parser.add_argument("--concurrency", type=int, default=24)
    args = parser.parse_args()
    asyncio.run(main(args.quick, args.concurrency))
