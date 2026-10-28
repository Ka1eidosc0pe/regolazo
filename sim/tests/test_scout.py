"""Moment Scout selection: cooldown, per-type cap, both teams, never ranks on the value gap."""

import itertools

from shared.schemas.moments import Consequence, MomentType, RewindScores
from sim.scout import COOLDOWN_S, PER_TYPE_CAP, Candidate, select


def _cand(t: float, team: str, mtype: MomentType, total: float) -> Candidate:
    scores = RewindScores(
        richness=0.5, stakes=0.5, consequence=0.5, salience=0.5, total=total, viable_options=4, option_entropy=1.8
    )
    cons = Consequence(kind="shot", xt_swing=0.1)
    return Candidate(int(t * 25), t, 1, team, "H07" if team == "home" else "A07", {}, 4, 1.8, mtype, cons, scores)


def test_select_respects_cooldown_type_cap_and_both_teams() -> None:
    cands = [_cand(100 + 10 * i, "away", "final_pass", 0.9 - i * 0.001) for i in range(40)]
    cands += [_cand(3000 + 200 * i, "home", "shot", 0.3) for i in range(5)]
    cands += [_cand(5000 + 200 * i, "away", "decisive_turnover", 0.5) for i in range(5)]
    chosen = select(cands, target=12)
    times = sorted(c.t for c in chosen)
    assert all(b - a >= COOLDOWN_S for a, b in itertools.pairwise(times))
    assert sum(c.moment_type == "final_pass" for c in chosen) <= PER_TYPE_CAP
    assert sum(c.team == "home" for c in chosen) >= 4
    assert len(chosen) <= 12
