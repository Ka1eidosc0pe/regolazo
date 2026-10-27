"""Drama score for seed search: goals, comebacks, late decisive goals, rich decision moments, variety."""

from dataclasses import dataclass, field
from typing import Any

from shared.schemas.events import MatchEvent
from sim.match import DecisionRecord

RICH_MIN_OPTIONS = 3
RICH_MIN_XT = 0.04


@dataclass
class Drama:
    seed: int
    score: float
    final: tuple[int, int]
    goals: list[dict[str, Any]] = field(default_factory=list)
    comeback: int = 0
    late_decider: bool = False
    rich_moments: int = 0
    moment_variety: int = 0
    shots: tuple[int, int] = (0, 0)
    xg: tuple[float, float] = (0.0, 0.0)
    notes: list[str] = field(default_factory=list)


def minute(t: float, second_half_t0: float | None) -> int:
    if second_half_t0 is None or t < second_half_t0:
        return int(t // 60) + 1
    return 45 + int((t - second_half_t0) // 60) + 1


def score_match(seed: int, events: list[MatchEvent], decisions: list[DecisionRecord]) -> Drama:
    starts = [e.t for e in events if e.type == "period_start"]
    h2 = starts[1] if len(starts) > 1 else None
    goals = []
    score = [0, 0]
    max_deficit = [0, 0]
    late = False
    for e in sorted(events, key=lambda e: e.t):
        is_goal = (e.type == "shot" and getattr(e.qualifiers, "outcome", None) == "goal") or e.type == "own_goal"
        if not is_goal:
            continue
        side = e.team if e.type == "shot" else ("away" if e.team == "home" else "home")
        k = 0 if side == "home" else 1
        before = score[k] - score[1 - k]
        score[k] += 1
        for j in (0, 1):
            max_deficit[j] = max(max_deficit[j], score[1 - j] - score[j])
        mnt = minute(e.t, h2)
        if mnt >= 80 and before in (-1, 0):
            late = True
        goals.append(
            {
                "minute": mnt,
                "team": side,
                "player": e.player,
                "technique": getattr(e.qualifiers, "technique", None) if e.type == "shot" else "own_goal",
                "situation": getattr(e.qualifiers, "situation", None) if e.type == "shot" else None,
                "xg": round(float(getattr(e.qualifiers, "xg", 0.0) or 0.0), 3),
                "score": f"{score[0]}-{score[1]}",
            }
        )
    final = (score[0], score[1])
    comeback = 0
    for j in (0, 1):
        if max_deficit[j] >= 1 and final[j] >= final[1 - j]:
            comeback = max(comeback, max_deficit[j])
    rich = [d for d in decisions if d.n_viable >= RICH_MIN_OPTIONS and d.xt >= RICH_MIN_XT]
    kinds = {d.slot for d in rich}
    shots_ev = [e for e in events if e.type == "shot"]
    techniques = {getattr(e.qualifiers, "technique", None) for e in shots_ev}
    situations = {getattr(e.qualifiers, "situation", None) for e in shots_ev}
    variety = len(kinds) + len(techniques) + len(situations)
    total_goals = sum(final)
    s = 1.2 * min(total_goals, 6) / 6 + 0.9 * min(comeback, 2) + (1.0 if late else 0.0)
    s += 0.6 * (abs(final[0] - final[1]) <= 1 and total_goals > 0)
    s += 0.8 * min(len(rich), 60) / 60 + 0.4 * min(variety, 30) / 30
    s -= 0.6 * (total_goals > 7)  # silly scorelines are not the demo we want
    notes = []
    if comeback:
        notes.append(f"comeback from {comeback} down")
    if late:
        notes.append("decisive goal after 80'")
    return Drama(
        seed=seed,
        score=float(s),
        final=final,
        goals=goals,
        comeback=comeback,
        late_decider=late,
        rich_moments=len(rich),
        moment_variety=variety,
        shots=(sum(e.team == "home" for e in shots_ev), sum(e.team == "away" for e in shots_ev)),
        xg=(
            round(sum(float(getattr(e.qualifiers, "xg", 0) or 0) for e in shots_ev if e.team == "home"), 2),
            round(sum(float(getattr(e.qualifiers, "xg", 0) or 0) for e in shots_ev if e.team == "away"), 2),
        ),
        notes=notes,
    )
