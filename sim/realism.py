"""Many-match realism runs: stats vs real-data targets, event coverage and displayed-vs-realized probabilities.

uv run python scripts/sim_realism.py --matches 200 --workers 9   # writes docs/sim_realism.md
"""

import multiprocessing as mp
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any, get_args

import numpy as np

from shared.model_api import load_models
from shared.schemas import enums
from shared.schemas.events import MatchEvent
from sim.record import simulate
from sim.squads import generate_meta
from sim.stats import MatchStats, match_stats


@dataclass
class MatchSummary:
    seed: int
    stats: MatchStats
    wall_s: float
    sim_s: float
    values: set[str] = field(default_factory=set)
    """Every event type and qualifier value seen (for the coverage gate)."""
    shots: list[tuple[float, float, bool, int]] = field(default_factory=list)
    """(displayed p, model xG, goal, finishing rating) per shot."""
    passes: list[tuple[float, float, bool, int]] = field(default_factory=list)
    """(displayed p, model xPass, completed, passing rating) per open-play pass."""
    score: tuple[int, int] = (0, 0)
    possession_share_home: float = 0.5
    extra: dict[str, float] = field(default_factory=dict)


def _values(events: list[MatchEvent]) -> set[str]:
    out: set[str] = set()
    for e in events:
        out.add(f"type:{e.type}")
        out.add(f"play_pattern:{e.play_pattern}")
        if e.phase_of_play:
            out.add(f"phase:{e.phase_of_play}")
        q = e.qualifiers
        if q is None:
            continue
        for k, v in q.model_dump().items():
            if isinstance(v, str) and k not in ("kind",) and not k.endswith("_id") and not _is_id(v):
                out.add(f"{q.kind}.{k}:{v}")
    return out


def _is_id(v: str) -> bool:
    return (len(v) == 3 and v[0] in "HA" and v[1:].isdigit()) or v.startswith("ev_") or v.startswith("snap:")


def run_one(seed: int) -> MatchSummary:
    models = load_models()
    meta = generate_meta(seed, f"real{seed:05d}", models.physical_priors)
    t0 = time.perf_counter()
    res = simulate(meta, models, snapshots=False)
    st = match_stats(res.events, res.arrays)
    recs = res.match.logs[0].records
    players = {p.id: p for p in meta.home.players + meta.away.players}
    shots = [
        (
            float(r["_p"]),
            float(r["q"]["xg"] or 0.0),
            r["q"]["outcome"] == "goal",
            players[r["player"]].attributes.finishing,
        )
        for r in recs
        if r["type"] == "shot" and "_p" in r
    ]
    passes = [
        (
            float(r["_p"]),
            float(r["q"].get("xpass") or 0.0),
            r["q"]["outcome"] == "complete",
            players[r["player"]].attributes.short_passing,
        )
        for r in recs
        if r["type"] == "pass" and "_p" in r and r["q"].get("restart") is None
    ]
    sc = res.arrays["score"][-1]
    return MatchSummary(
        seed=seed,
        stats=st,
        wall_s=time.perf_counter() - t0,
        sim_s=float(res.arrays["t"][-1]),
        values=_values(res.events),
        shots=shots,
        passes=passes,
        score=(int(sc[0]), int(sc[1])),
        possession_share_home=_pass_share(res.events),
        extra=_extra(res.events),
    )


def _extra(events: list[MatchEvent]) -> dict[str, float]:
    passes = [e for e in events if e.type == "pass"]
    crosses = [e for e in passes if getattr(e.qualifiers, "pass_type", None) == "cross"]
    shots = [e for e in events if e.type == "shot"]
    return {
        "crosses": len(crosses),
        "crosses_complete": sum(getattr(e.qualifiers, "outcome", "") == "complete" for e in crosses),
        "through_balls": sum(getattr(e.qualifiers, "pass_type", None) == "through_ball" for e in passes),
        "header_shots": sum(getattr(e.qualifiers, "body_part", None) == "head" for e in shots),
        "shots_in_box": sum(getattr(e.qualifiers, "distance_to_goal_m", 99) < 16.5 for e in shots),
        "carries": sum(e.type == "carry" for e in events),
        "own_goals": sum(e.type == "own_goal" for e in events),
        "on_target": sum(bool(getattr(e.qualifiers, "on_target", False)) for e in shots),
        "long_shots_25": sum(getattr(e.qualifiers, "distance_to_goal_m", 0) >= 25 for e in shots),
        "goals_from_shots": sum(getattr(e.qualifiers, "outcome", "") == "goal" for e in shots),
        "shots": len(shots),
        "xg_sum": float(sum(getattr(e.qualifiers, "xg", 0) or 0 for e in shots)),
        "psxg_on_target": float(sum(getattr(e.qualifiers, "psxg", 0) or 0 for e in shots)),
        "take_ons": sum(e.type == "take_on" for e in events),
    }


def _pass_share(events: list[MatchEvent]) -> float:
    c = Counter(e.team for e in events if e.type == "pass")
    tot = c["home"] + c["away"]
    return c["home"] / tot if tot else 0.5


def run_many(seeds: list[int], workers: int = 8) -> list[MatchSummary]:
    if workers <= 1:
        return [run_one(s) for s in seeds]
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
        return list(ex.map(run_one, seeds))


def core_values() -> set[str]:
    """Every value the coverage gate requires: event types and qualifier enums, minus OPTIONAL_FOR_COVERAGE."""
    from shared.schemas.events import (
        BlockQ,
        CarryQ,
        ClearanceQ,
        DuelQ,
        FoulQ,
        GkQ,
        InterceptionQ,
        OffsideQ,
        OutOfPlayQ,
        PassQ,
        PhysicalQ,
        PressingSequenceQ,
        PressureQ,
        ReceiptQ,
        RecoveryQ,
        RestartQ,
        RunQ,
        ShotQ,
        StepUpQ,
        TakeOnQ,
        TouchLossQ,
    )

    opt = enums.OPTIONAL_FOR_COVERAGE
    out = {f"type:{t}" for t in get_args(enums.EventType) if t not in opt}
    out |= {f"play_pattern:{p}" for p in get_args(enums.PlayPattern) if p not in opt}
    out |= {f"phase:{p}" for p in get_args(enums.PhaseOfPlay) if p not in opt and p != "dead_ball"}
    models: list[Any] = [
        PassQ, ReceiptQ, CarryQ, TakeOnQ, ShotQ, DuelQ, InterceptionQ, BlockQ, ClearanceQ, RecoveryQ, TouchLossQ,
        FoulQ, GkQ, RestartQ, OutOfPlayQ, OffsideQ, PressureQ, PressingSequenceQ, RunQ, PhysicalQ, StepUpQ,
    ]  # fmt: skip
    for qm in models:
        kind = qm.model_fields["kind"].default
        for name, f in qm.model_fields.items():
            if name == "kind":
                continue
            for v in _literal_values(f.annotation):
                if v not in opt:
                    out.add(f"{kind}.{name}:{v}")
    # values that only make sense on some fields
    out -= {f"pass.body_part:{v}" for v in ("chest", "both_hands", "left_hand", "right_hand", "other", "no_touch")}
    out -= {f"ball_receipt.first_touch_body_part:{v}" for v in ("hands", "both_hands", "left_hand", "right_hand",
                                                                  "no_touch", "other")}  # fmt: skip
    out -= {f"shot.body_part:{v}" for v in ("chest", "hands", "both_hands", "left_hand", "right_hand", "no_touch")}
    out -= {f"clearance.body_part:{v}" for v in ("chest", "hands", "both_hands", "left_hand", "right_hand", "no_touch",
                                                 "other")}  # fmt: skip
    out -= {f"gk_action.body_part:{v}" for v in ("hands", "other", "no_touch", "head")}
    out -= {f"pass.restart:{v}" for v in ("penalty", "drop_ball")}
    # a ball that crosses a line is restarted with a throw-in, goal kick or corner: never a free kick or penalty
    out -= {f"ball_out_of_play.result_restart_type:{v}" for v in ("free_kick_direct", "free_kick_indirect", "kick_off",
                                                                  "penalty", "drop_ball")}  # fmt: skip
    return out


def _literal_values(ann: Any) -> list[str]:
    out: list[str] = []
    for a in get_args(ann) or ():
        if isinstance(a, str):
            out.append(a)
        else:
            out.extend(_literal_values(a))
    if not out and getattr(ann, "__origin__", None) is not None:
        pass
    return out


def calibration(rows: list[tuple[float, float, bool, int]], bins: int = 5) -> list[tuple[float, float, float, int]]:
    """(mean displayed p, mean model p, realized rate, n) per displayed-p quantile bin."""
    if not rows:
        return []
    arr = np.array([(r[0], r[1], float(r[2])) for r in rows])
    edges = np.quantile(arr[:, 0], np.linspace(0, 1, bins + 1))
    out = []
    for i in range(bins):
        m = (arr[:, 0] >= edges[i]) & ((arr[:, 0] < edges[i + 1]) | (i == bins - 1))
        if m.any():
            out.append((float(arr[m, 0].mean()), float(arr[m, 1].mean()), float(arr[m, 2].mean()), int(m.sum())))
    return out
