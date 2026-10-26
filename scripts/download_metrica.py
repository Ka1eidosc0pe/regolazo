"""Download the Metrica Sports sample tracking + event data (games 1 and 2, 25 Hz CSV).

Run with: uv run python scripts/download_metrica.py
Files go to $REGOLAZO_DATA_DIR/raw/metrica/; existing files are skipped.
Data: https://github.com/metrica-sports/sample-data (see that repository's licence; attribution to Metrica Sports).
"""

import sys

import httpx

from analytics.paths import metrica_dir

BASE = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data"
FILES = [
    f"Sample_Game_{g}/Sample_Game_{g}_{kind}.csv"
    for g in (1, 2)
    for kind in ("RawTrackingData_Home_Team", "RawTrackingData_Away_Team", "RawEventsData")
]


def main() -> None:
    root = metrica_dir()
    with httpx.Client(timeout=300.0, follow_redirects=True) as client:
        for rel in FILES:
            dest = root / rel
            if dest.exists() and dest.stat().st_size > 0:
                print(f"cached {rel}")
                continue
            resp = client.get(f"{BASE}/{rel}")
            if resp.status_code != 200:
                print(f"failed {rel}: HTTP {resp.status_code}", file=sys.stderr)
                sys.exit(1)
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(".part")
            tmp.write_bytes(resp.content)
            tmp.replace(dest)
            print(f"downloaded {rel} ({len(resp.content) / 1e6:.1f} MB)")
    print(f"done, cache at {root}")


if __name__ == "__main__":
    main()
