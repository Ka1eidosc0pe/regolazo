"""Moment Scout (PLAN.md section 2.5): flag Rewind opportunities in a recorded match, judgement-free.

    uv run python -m sim.scout --match candidate-2004        # writes <recording>/moments.jsonl

Candidates are the simulator's decision keyframes (``snapshots/decisions.jsonl``). Each is scored on decision
richness (viable options, option entropy), stakes (xT at the ball, late-and-close match context), consequence (what
happened in the next 10 s) and fan salience (moment type). The value gap between the real choice and the
alternatives is never used for selection. The Scout keeps the best candidates subject to a cooldown, a per-type cap
and a minimum per team, then fills in the options and the real action from the engine so the UI can fork directly.
"""

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shared.recording import Recording
from shared.schemas.actions import ActionIntent
from shared.schemas.events import MatchEvent
from shared.schemas.moments import Consequence, ConsequenceKind, DecisionMoment, MomentType, RewindScores
from sim.engine import SimEngine, snapshot_ref

HORIZON_S = 10.0
COOLDOWN_S = 60.0
TARGET = 12
PER_TYPE_CAP = 4
MIN_PER_TEAM = 4
MIN_VIABLE = 3

SALIENCE: dict[MomentType, float] = {
    "shot": 1.0,
    "final_pass": 0.9,
    "one_v_one": 0.9,
    "counter_attack": 0.8,
    "decisive_turnover": 0.8,
    "set_piece": 0.7,
    "build_up": 0.4,
    "other": 0.3,
}
CONSEQUENCE_WEIGHT: dict[ConsequenceKind, float] = {
    "goal": 1.0,
    "big_chance": 0.85,
    "turnover_to_chance_against": 0.75,
    "shot": 0.6,
    "possession_value_swing": 0.4,
    "none": 0.0,
}


@dataclass
class Candidate:
    tick: int
    t: float
    period: int
    team: str
    carrier: str
    intent: dict[str, Any]
    n_viable: int
    entropy: float
    moment_type: MomentType
    consequence: Consequence
    scores: RewindScores


def _xg(e: MatchEvent) -> float:
    return float(getattr(e.qualifiers, "xg", 0.0) or 0.0)


def _is_goal(e: MatchEvent) -> bool:
    return e.type == "shot" and getattr(e.qualifiers, "outcome", None) == "goal"


def _consequence(team: str, t: float, events: list[MatchEvent]) -> tuple[Consequence, list[MatchEvent]]:
    window = [e for e in events if t <= e.t <= t + HORIZON_S]
    own_shots = [e for e in window if e.type == "shot" and e.team == team]
    opp_shots = [e for e in window if e.type == "shot" and e.team not in (None, team)]
    kind: ConsequenceKind = "none"
    picked: list[MatchEvent] = []
    if any(_is_goal(e) for e in own_shots):
        kind, picked = "goal", [e for e in own_shots if _is_goal(e)]
    elif any(_xg(e) >= 0.2 for e in own_shots):
        kind, picked = "big_chance", [max(own_shots, key=_xg)]
    elif opp_shots:
        kind, picked = "turnover_to_chance_against", opp_shots[:1]
    elif own_shots:
        kind, picked = "shot", own_shots[:1]
    swing = sum(_xg(e) for e in own_shots) - sum(_xg(e) for e in opp_shots)
    if kind == "none" and abs(swing) >= 0.05:
        kind = "possession_value_swing"
    return Consequence(kind=kind, xt_swing=round(swing, 4), event_ids=[e.id for e in picked]), window


def _moment_type(team: str, intent: dict[str, Any], consequence: Consequence, window: list[MatchEvent]) -> MomentType:
    if intent.get("kind") == "shot":
        return "shot"
    if consequence.kind == "turnover_to_chance_against":
        return "decisive_turnover"
    first_shot = next((e for e in window if e.type == "shot"), None)
    if first_shot is not None and first_shot.team == team:
        if first_shot.play_pattern in ("from_corner", "from_free_kick", "from_throw_in"):
            return "set_piece"
        if first_shot.play_pattern == "from_counter":
            return "counter_attack"
        return "final_pass"
    return "build_up"


def _context(t: float, minute_scale: float, margin: int) -> float:
    late = min(1.0, max(0.0, (t / minute_scale - 60.0) / 30.0))
    close = 1.0 if abs(margin) <= 1 else 0.3
    return 0.5 * late + 0.5 * close


def candidates(rec: Recording, decisions: list[dict[str, Any]]) -> list[Candidate]:
    events = rec.events()
    goals = [e for e in events if _is_goal(e)]
    duration = max(e.t for e in events)
    minute_scale = duration / 95.0  # match seconds per match minute, including stoppages
    out: list[Candidate] = []
    for d in decisions:
        t = float(d["t"])
        team = "home" if d["carrier"].startswith("H") else "away"
        cons, window = _consequence(team, t, events)
        if cons.kind == "none":
            continue
        # an open situation, or one that ended in a goal or a big chance (successful moments are rewindable too)
        rich = d["n_viable"] >= MIN_VIABLE or d["entropy"] >= 1.5
        if not rich and cons.kind not in ("goal", "big_chance"):
            continue
        home = sum(1 for g in goals if g.t < t and g.team == "home")
        away = sum(1 for g in goals if g.t < t and g.team == "away")
        richness = 0.6 * min(1.0, d["n_viable"] / 6) + 0.4 * min(1.0, d["entropy"] / 2.5)
        stakes = 0.7 * min(1.0, float(d["xt"]) / 0.15) + 0.3 * _context(t, minute_scale, home - away)
        mtype = _moment_type(team, d["intent"], cons, window)
        consequence = CONSEQUENCE_WEIGHT[cons.kind]
        salience = SALIENCE[mtype]
        total = 0.3 * richness + 0.2 * stakes + 0.3 * consequence + 0.2 * salience
        scores = RewindScores(
            richness=round(richness, 4),
            stakes=round(stakes, 4),
            consequence=consequence,
            salience=salience,
            total=round(total, 4),
            viable_options=int(d["n_viable"]),
            option_entropy=float(d["entropy"]),
        )
        out.append(
            Candidate(
                int(d["tick"]), t, 1 if t < rec_half(events) else 2, team, d["carrier"], d["intent"],
                int(d["n_viable"]), float(d["entropy"]), mtype, cons, scores,
            )
        )  # fmt: skip
    return out


def rec_half(events: list[MatchEvent]) -> float:
    starts = [e.t for e in events if e.type == "period_start"]
    return starts[1] if len(starts) > 1 else math.inf


def select(cands: list[Candidate], target: int = TARGET) -> list[Candidate]:
    """Best first, with a cooldown, a cap per moment type and a minimum per team."""
    ranked = sorted(cands, key=lambda c: -c.scores.total)
    chosen: list[Candidate] = []

    def ok(c: Candidate) -> bool:
        if any(abs(c.t - o.t) < COOLDOWN_S for o in chosen):
            return False
        return sum(o.moment_type == c.moment_type for o in chosen) < PER_TYPE_CAP

    for team in ("home", "away"):
        for c in ranked:
            if sum(o.team == team for o in chosen) >= MIN_PER_TEAM:
                break
            if c.team == team and ok(c):
                chosen.append(c)
    for c in ranked:
        if len(chosen) >= target:
            break
        if c not in chosen and ok(c):
            chosen.append(c)
    return sorted(chosen, key=lambda c: c.t)


def _same_choice(option: ActionIntent, real: ActionIntent) -> bool:
    """The option list is discrete: a pass to the same teammate or a shot counts as the real choice. Carries and
    passes into space rarely match an option exactly; the UI then shows ``real_action`` as the real choice."""
    if option.kind != real.kind:
        return False
    if real.kind == "pass_player":
        return option.target_player == real.target_player
    return real.kind == "shot" or option == real


def build_moments(
    rec: Recording, chosen: list[Candidate], engine: SimEngine, names: dict[str, str]
) -> list[DecisionMoment]:
    moments = []
    for i, c in enumerate(chosen):
        ref = snapshot_ref(rec.meta.match_id, c.tick)
        intent = ActionIntent.model_validate(c.intent)
        outputs = engine.options(ref)
        real = engine.preview(ref, intent)
        options = [o.model_copy(update={"is_real": _same_choice(o.intent, intent)}) for o in outputs.options]
        n = sum(o.viable for o in options) or c.n_viable
        who = names.get(c.carrier, c.carrier)
        moments.append(
            DecisionMoment(
                id=f"mo_{i + 1:02d}",
                tick=c.tick,
                t=c.t,
                period=c.period,
                snapshot_ref=ref,
                team=c.team,
                ball_carrier=c.carrier,
                moment_type=c.moment_type,
                origin="scout",
                real_action=real,
                consequence=c.consequence,
                options=options,
                scores=c.scores,
                flagged=True,
                hook=f"{who} had {n} options here. What would you have done?",
            )
        )
    return moments


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--match", required=True)
    ap.add_argument("--target", type=int, default=TARGET)
    args = ap.parse_args()
    data = Path(os.environ.get("REGOLAZO_DATA_DIR", Path(__file__).resolve().parents[1] / "data"))
    directory = data / "matches" / args.match
    rec = Recording.open(directory)
    decisions = [json.loads(line) for line in (directory / "snapshots" / "decisions.jsonl").read_text().splitlines()]
    chosen = select(candidates(rec, decisions), args.target)
    names = {p.id: p.display_name for team in (rec.meta.home, rec.meta.away) for p in team.players}
    moments = build_moments(rec, chosen, SimEngine(), names)
    (directory / "moments.jsonl").write_text("".join(m.model_dump_json() + "\n" for m in moments))
    for m in moments:
        s = m.scores
        assert s is not None
        print(f"{m.id} t={m.t:7.1f} {m.team:4} {m.moment_type:18} {m.consequence.kind:26} total={s.total:.2f} {m.hook}")


if __name__ == "__main__":
    main()
