"""Simulate a full match headlessly and write a recording to $REGOLAZO_DATA_DIR/matches/<match_id>/.

uv run python scripts/sim_match.py --seed 7 [--match-id demo-0007] [--out DIR]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.model_api import load_models
from sim.record import matches_dir, simulate, write_match
from sim.squads import generate_meta
from sim.stats import match_stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--match-id", default=None)
    ap.add_argument("--out", type=Path, default=None, help="recording directory (default: data/matches/<id>)")
    args = ap.parse_args()
    models = load_models()
    meta = generate_meta(args.seed, args.match_id, models.physical_priors)
    res = simulate(meta, models)
    out = args.out or matches_dir() / meta.match_id
    write_match(res, out)
    st = match_stats(res.events, res.arrays)
    home, away = (int(v) for v in res.arrays["score"][-1])
    print(
        f"{meta.home.name} {home} - {away} {meta.away.name}  shots {st.shots_home}-{st.shots_away}, passes {st.passes}"
    )
    print(f"frames={len(res.arrays['t'])} events={len(res.events)} wall={res.wall_s:.1f}s -> {out}")


if __name__ == "__main__":
    main()
