"""Deterministic match statistics and key moments: the only source of numbers for the commentary agents.

No LLM in here. ``MatchData`` reads a recording (events, frames, meta); ``key_moments`` picks the moments worth
narrating; each ``Moment`` carries a ``FactSheet`` of display-ready facts (player *names*, never ids; numbers
already rounded the way they may be quoted). The Fact-Checker and the automated number check compare narratives
to exactly these facts, so "every number traces to a tool output" is a property of this module's output.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from shared.recording import Recording
from shared.schemas.events import MatchEvent, PassQ, ShotQ
from shared.schemas.narrative import Evidence

MS_TO_KMH = 3.6
BIG_CHANCE_XG = 0.2
OUTCOME_DELAY_S = 1.5
"""A shot's narrative becomes visible this long after contact, when the outcome is known on screen."""

MomentKind = Literal["goal", "save", "big_chance", "chance", "half_time", "full_time"]
FactValue = str | int
FactSource = tuple[Literal["event", "stat", "model_output"], str]


@dataclass
class FactSheet:
    """Display-ready facts for one moment: ``facts[key]`` is what a narrative may quote, verbatim."""

    moment_id: str
    facts: dict[str, FactValue] = field(default_factory=dict)
    sources: dict[str, FactSource] = field(default_factory=dict)
    """Where each fact came from (event id, stat key, model output), for Narrative.evidence."""

    def add(self, key: str, value: FactValue | None, source: FactSource) -> None:
        if value is not None:
            self.facts[key] = value
            self.sources[key] = source

    def render(self) -> str:
        return "\n".join(f"{k}: {v}" for k, v in self.facts.items())

    def numbers(self) -> set[float]:
        return numbers_in(" ".join(str(v) for v in self.facts.values()))

    def names(self) -> set[str]:
        """Every non-numeric string fact (player and team names, outcomes): what a narrative may mention."""
        return {str(v) for v in self.facts.values() if isinstance(v, str) and not NUMBER_RE.search(v)}

    def evidence(self) -> list[Evidence]:
        out: list[Evidence] = []
        for key, value in self.facts.items():
            if NUMBER_RE.search(str(value)):
                kind, ref = self.sources[key]
                out.append(Evidence(kind=kind, ref=ref, field=key, value=str(value)))
        return out


@dataclass
class Moment:
    id: str
    kind: MomentKind
    t: float
    """Match time at which the narrative becomes visible."""
    minute: int
    facts: FactSheet
    event_id: str | None = None


NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")


def numbers_in(text: str) -> set[float]:
    """All numbers written with digits (decimal comma or point); '74th' -> 74, '1-1' -> {1.0}."""
    return {float(m.replace(",", ".")) for m in NUMBER_RE.findall(text)}


def _human(value: str) -> str:
    return value.replace("_", " ")


class MatchData:
    """A recorded match with cached per-frame aggregates (possession, speed, distance)."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.rec = Recording.open(directory)
        self.meta = self.rec.meta
        self.events: list[MatchEvent] = sorted(self.rec.events(), key=lambda e: (e.t, e.index))
        self.by_id = {e.id: e for e in self.events}
        self.ids = self.rec.player_ids
        self.col = {pid: i for i, pid in enumerate(self.ids)}
        self.players = {p.id: p for team in (self.meta.home, self.meta.away) for p in team.players}
        a = self.rec.arrays
        self.t: npt.NDArray[np.float64] = a["t"]
        on = a["player_on_pitch"].astype(bool)
        speed = np.linalg.norm(a["player_vel"].astype(np.float64), axis=2) * on
        dt = 1.0 / self.meta.sim_hz
        self._speed = speed
        self._top = np.maximum.accumulate(speed, axis=0)
        self._dist = np.cumsum(speed * dt, axis=0)
        poss = a["poss_team"]
        self._poss = np.stack([np.cumsum(poss == 0), np.cumsum(poss == 1)], axis=1)
        self.duration_s = float(self.t[-1])
        self.half_end_t = next((e.t for e in self.events if e.type == "period_end" and e.period == 1), 2700.0)
        self.second_half_t = next((e.t for e in self.events if e.type == "period_start" and e.period == 2), 2700.0)

    def minute_of(self, t: float) -> int:
        """Broadcast minute number (the 90th-minute goal is minute 90): stoppage time is not shown separately."""
        if t < self.half_end_t:
            return min(int(t // 60) + 1, 45)
        return 45 + int((max(t, self.second_half_t) - self.second_half_t) // 60) + 1

    # --- names -------------------------------------------------------------------------------------------

    def name(self, player_id: str | None) -> str | None:
        return self.players[player_id].display_name if player_id in self.players else None

    def team_name(self, side: str | None) -> str | None:
        if side == "home":
            return self.meta.home.name
        if side == "away":
            return self.meta.away.name
        return None

    def all_names(self) -> set[str]:
        """Every player display name and team name in the match (used to catch wrong-name claims)."""
        return {p.display_name for p in self.players.values()} | {self.meta.home.name, self.meta.away.name}

    # --- match level -------------------------------------------------------------------------------------

    def _events_until(self, t: float) -> list[MatchEvent]:
        return [e for e in self.events if e.t <= t]

    def score_at(self, t: float) -> tuple[int, int]:
        h = a = 0
        for e in self._events_until(t):
            q = e.qualifiers
            if e.type == "shot" and isinstance(q, ShotQ) and q.outcome == "goal":
                if e.team == "home":
                    h += 1
                else:
                    a += 1
        return h, a

    def player_physical(self, player_id: str, t: float) -> tuple[float, float]:
        """(top speed in km/h, distance in km) of one player up to match time ``t``."""
        i = self.rec.index_at(t)
        c = self.col[player_id]
        return float(self._top[i, c]) * MS_TO_KMH, float(self._dist[i, c]) / 1000.0

    def speed_at(self, player_id: str, t: float) -> float:
        """Player speed in km/h at match time ``t`` (from the recorded frames)."""
        return float(self._speed[self.rec.index_at(t), self.col[player_id]]) * MS_TO_KMH

    def match_stats(self, t: float) -> dict[str, FactValue]:
        """Team totals and physical leaders up to match time ``t`` (possession, shots, xG, passing, speed, distance)."""
        i = self.rec.index_at(t)
        poss = self._poss[i].astype(float)
        total = max(float(poss.sum()), 1.0)
        home_pct = round(100 * poss[0] / total)
        shots = {"home": 0, "away": 0}
        sot = {"home": 0, "away": 0}
        xg = {"home": 0.0, "away": 0.0}
        passes = {"home": [0, 0], "away": [0, 0]}
        for e in self._events_until(t):
            q = e.qualifiers
            if e.team not in ("home", "away"):
                continue
            if e.type == "shot" and isinstance(q, ShotQ):
                shots[e.team] += 1
                sot[e.team] += int(q.on_target)
                xg[e.team] += q.xg or 0.0
            elif e.type == "pass" and isinstance(q, PassQ):
                passes[e.team][1] += 1
                passes[e.team][0] += int(q.outcome == "complete")
        h, a = self.score_at(t)
        speeds = self._top[i] * MS_TO_KMH
        c_top = int(np.argmax(speeds))
        dists = self._dist[i]
        c_dist = int(np.argmax(dists))
        out: dict[str, FactValue] = {
            "score": f"{h}-{a}",
            "home_possession_pct": int(home_pct),
            "away_possession_pct": int(100 - home_pct),
            "home_shots": shots["home"],
            "away_shots": shots["away"],
            "home_shots_on_target": sot["home"],
            "away_shots_on_target": sot["away"],
            "home_xg": f"{xg['home']:.2f}",
            "away_xg": f"{xg['away']:.2f}",
            "top_speed_player": self.name(self.ids[c_top]) or "",
            "top_speed_kmh": round(float(speeds[c_top])),
            "distance_leader": self.name(self.ids[c_dist]) or "",
            "distance_leader_km": f"{float(dists[c_dist]) / 1000.0:.1f}",
        }
        for side in ("home", "away"):
            done, att = passes[side]
            out[f"{side}_pass_accuracy_pct"] = round(100 * done / att) if att else 0
        return out

    # --- moments -----------------------------------------------------------------------------------------

    def _team_running(self, shot: MatchEvent) -> tuple[int, float]:
        n, x = 0, 0.0
        for e in self.events:
            if e.t > shot.t:
                break
            if e.type == "shot" and e.team == shot.team and isinstance(e.qualifiers, ShotQ):
                n += 1
                x += e.qualifiers.xg or 0.0
        return n, x

    def _shot_moment(self, e: MatchEvent) -> Moment:
        q = e.qualifiers
        assert isinstance(q, ShotQ) and e.player is not None
        kind: MomentKind
        if q.outcome == "goal":
            kind = "goal"
        elif q.outcome in ("saved", "saved_to_post"):
            kind = "save"
        elif (q.xg or 0.0) >= BIG_CHANCE_XG:
            kind = "big_chance"
        else:
            kind = "chance"
        mid = f"mom_{e.id}"
        t_vis = min(e.t + OUTCOME_DELAY_S, self.duration_s - 0.5)
        s = FactSheet(mid)
        ev: FactSource = ("event", e.id)
        minute = self.minute_of(e.t)
        s.add("moment_type", kind.replace("_", " "), ev)
        s.add("minute", minute, ev)
        s.add("half", "first half" if e.period == 1 else "second half", ev)
        sh, sa = self.score_at(e.t)
        s.add("score_now", f"{self.meta.home.name} {sh}-{sa} {self.meta.away.name}", ("stat", "score"))
        s.add("home_team", self.meta.home.name, ("stat", "meta.home"))
        s.add("away_team", self.meta.away.name, ("stat", "meta.away"))
        s.add("team", self.team_name(e.team), ev)
        s.add("shooter", self.name(e.player), ev)
        s.add("goalkeeper", self.name(q.keeper), ev)
        s.add("outcome", _human(q.outcome), ev)
        s.add("body_part", _human(q.body_part) if q.body_part else None, ev)
        s.add("technique", _human(q.technique) if q.technique else None, ev)
        s.add("situation", _human(q.situation) if q.situation else None, ev)
        s.add("xg", f"{q.xg:.2f}" if q.xg is not None else None, ("model_output", f"{e.id}.xg"))
        s.add("post_shot_xg", f"{q.psxg:.2f}" if q.psxg is not None else None, ("model_output", f"{e.id}.psxg"))
        s.add("shot_speed_kmh", round(q.ball_speed_mps * MS_TO_KMH), ev)
        s.add("shot_distance_m", round(q.distance_to_goal_m), ev)
        s.add("shooter_running_speed_kmh", round(self.speed_at(e.player, e.t)), ("stat", f"player.{e.player}.speed"))
        s.add("under_pressure", "yes" if e.under_pressure else "no", ev)
        if q.key_pass_id and q.key_pass_id in self.by_id:
            kp = self.by_id[q.key_pass_id]
            kq = kp.qualifiers
            if isinstance(kq, PassQ):
                kev: FactSource = ("event", kp.id)
                s.add("assist_by", self.name(kp.player), kev)
                s.add("assist_pass_length_m", round(kq.length_m), kev)
                s.add(
                    "assist_pass_xpass",
                    f"{kq.xpass:.2f}" if kq.xpass is not None else None,
                    ("model_output", f"{kp.id}.xpass"),
                )
        n, x = self._team_running(e)
        s.add("team_shots_so_far", n, ("stat", f"{e.team}.shots"))
        s.add("team_xg_so_far", f"{x:.2f}", ("stat", f"{e.team}.xg"))
        if q.outcome in ("saved", "saved_to_post") and q.gk_event_id and q.gk_event_id in self.by_id:
            gq = self.by_id[q.gk_event_id].qualifiers
            dive = getattr(gq, "dive_distance_m", None)
            s.add("keeper_dive_m", f"{dive:.1f}" if dive is not None else None, ("event", q.gk_event_id))
        return Moment(mid, kind, t_vis, minute, s, e.id)

    def _period_moment(self, kind: Literal["half_time", "full_time"], t_event: float) -> Moment:
        s = FactSheet(f"mom_{kind}")
        st: FactSource = ("stat", f"match_stats@{kind}")
        s.add("moment_type", kind.replace("_", " "), st)
        s.add("home_team", self.meta.home.name, ("stat", "meta.home"))
        s.add("away_team", self.meta.away.name, ("stat", "meta.away"))
        stats = self.match_stats(t_event)
        s.add("scoreline", f"{self.meta.home.name} {stats['score']} {self.meta.away.name}", st)
        for k, v in stats.items():
            s.add(k, v, ("stat", f"{k}@{kind}"))
        if kind == "full_time":
            goals = [
                e
                for e in self.events
                if e.type == "shot" and isinstance(e.qualifiers, ShotQ) and e.qualifiers.outcome == "goal"
            ]
            for n, g in enumerate(goals, 1):
                s.add(
                    f"goal_{n}",
                    f"{self.name(g.player)} for {self.team_name(g.team)}, minute {self.minute_of(g.t)}",
                    ("event", g.id),
                )
        t_vis = min(t_event, self.duration_s - 0.5)
        return Moment(f"mom_{kind}", kind, t_vis, self.minute_of(t_event), s)

    def key_moments(self) -> list[Moment]:
        """Goals, saves, big chances and other shots, plus half time and full time; ordered by time."""
        moments = [
            self._shot_moment(e)
            for e in self.events
            if e.type == "shot" and isinstance(e.qualifiers, ShotQ) and e.player
        ]
        moments.append(self._period_moment("half_time", self.half_end_t))
        moments.append(self._period_moment("full_time", self.duration_s))
        return sorted(moments, key=lambda m: m.t)


def fact_sheet_dict(sheet: FactSheet) -> dict[str, Any]:
    """The JSON an agent tool returns: display values only, no ids."""
    return {"moment_id": sheet.moment_id, "facts": dict(sheet.facts)}
