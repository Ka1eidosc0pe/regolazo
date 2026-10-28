"""Performance gate (PLAN.md Step 3), lenient so it holds on a loaded machine: a match runs well faster than real
time, and batched realities share the work. The measured numbers (full match, 1,000 x 10 s) come from
scripts/sim_bench.py and are reported in docs/sim_realism.md.
"""

import time

from sim.match import Match
from sim.squads import generate_meta


def test_single_match_runs_much_faster_than_real_time() -> None:
    m = Match(generate_meta(5), batch=1)
    m.start()
    m.run(until_t=5.0)  # warm-up: model caches, first decisions
    t0 = time.perf_counter()
    m.run(until_t=125.0)
    wall = time.perf_counter() - t0
    assert 120.0 / wall >= 5.0, f"{120.0 / wall:.1f}x real time"


def test_batched_realities_are_cheaper_than_separate_matches() -> None:
    def per_reality(b: int) -> float:
        m = Match(generate_meta(6), batch=b)
        m.start()
        m.run(until_t=2.0)
        t0 = time.perf_counter()
        m.run(until_t=12.0)
        return (time.perf_counter() - t0) / b

    single, batched = per_reality(1), per_reality(100)
    assert batched < 0.5 * single, (single, batched)
