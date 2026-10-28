"""Performance gate (PLAN.md Step 3): full-match speed and "1,000 realities" latency on this machine.

uv run python scripts/sim_bench.py [--seed 7] [--n 1000] [--horizon 10]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.model_api import load_models
from shared.schemas.rollout import RolloutRequest
from sim.engine import SimEngine
from sim.record import simulate
from sim.squads import generate_meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--horizon", type=float, default=10.0)
    ap.add_argument("--forks", type=int, default=5)
    args = ap.parse_args()
    models = load_models()
    meta = generate_meta(args.seed, f"bench{args.seed}", models.physical_priors)
    t0 = time.perf_counter()
    res = simulate(meta, models)
    wall = time.perf_counter() - t0
    sim_s = float(res.arrays["t"][-1])
    print(f"full match: {sim_s / 60:.1f} simulated min in {wall:.1f} s wall ({sim_s / wall:.0f}x real time)")
    eng = SimEngine(models=models)
    eng.register(meta, res.match.snapshots)
    t0 = time.perf_counter()
    eng.warm_up()
    print(f"worker pool warm-up: {time.perf_counter() - t0:.1f} s ({eng.workers} workers)")
    rng = np.random.default_rng(0)
    ticks = [t for t, _ in res.match.snapshots]
    times = []
    for tick in rng.choice(ticks, args.forks, replace=False):
        ref = f"snap:{meta.match_id}:{int(tick)}"
        mo = eng.options(ref)
        if not mo.options:
            continue
        opt = max(mo.options, key=lambda o: o.decision_value)
        t0 = time.perf_counter()
        eng.preview(ref, opt.intent)
        t_prev = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        eng.rollout(RolloutRequest(snapshot_ref=ref, intent=opt.intent, horizon_s=args.horizon, n=1, seed=1))
        t_roll = (time.perf_counter() - t0) * 1000
        req = RolloutRequest(snapshot_ref=ref, intent=opt.intent, horizon_s=args.horizon, n=args.n, seed=1)
        t0 = time.perf_counter()
        d = eng.distribution(req)
        t_dist = time.perf_counter() - t0
        times.append(t_dist)
        print(
            f"{opt.label:30s} preview {t_prev:5.1f} ms  rollout {t_roll:6.0f} ms  {args.n} x {args.horizon:.0f} s: "
            f"{t_dist:5.2f} s  p={d.p_primary_success:.3f} (model {d.model_p_success:.3f})"
        )
    if times:
        print(f"distribution: median {np.median(times):.2f} s, max {np.max(times):.2f} s")


if __name__ == "__main__":
    main()
