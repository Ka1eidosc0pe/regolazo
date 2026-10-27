"""Event coverage gate (PLAN.md Step 3): every core event type and qualifier value occurs in simulated matches.

The full gate needs many matches (rare values such as penalties, own-goal-free woodwork rebounds or keeper rush
outs); it runs with ``REGOLAZO_FULL_GATES=1`` (20 matches in parallel, a few minutes) and is reported over 200
matches in docs/sim_realism.md by scripts/sim_realism.py. The default run checks the common values on the shared
8-minute simulation.
"""

import os

import pytest

from sim.realism import _values, core_values, run_many
from sim.tests.test_match import _sim

COMMON = {
    "type:pass",
    "type:ball_receipt",
    "type:carry",
    "type:shot",
    "type:duel",
    "type:ball_recovery",
    "type:restart",
    "type:off_ball_run",
    "pass.height:ground",
    "pass.height:high",
    "pass.length_class:short",
    "pass.length_class:medium",
    "pass.length_class:long",
    "pass.outcome:complete",
    "pass.outcome:intercepted",
    "ball_receipt.touch_quality:clean",
    "ball_receipt.body_orientation:open",
}


def test_common_values_occur_in_a_short_match() -> None:
    _, _, events = _sim()
    seen = _values(events)
    assert core_values() >= COMMON
    assert COMMON - seen == set()


@pytest.mark.skipif(os.environ.get("REGOLAZO_FULL_GATES") != "1", reason="full coverage gate: REGOLAZO_FULL_GATES=1")
def test_every_core_value_occurs_over_many_matches() -> None:
    res = run_many(list(range(5000, 5020)), workers=9)
    seen = set().union(*[r.values for r in res])
    missing = sorted(core_values() - seen)
    assert not missing, missing
