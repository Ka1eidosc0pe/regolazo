"""Lightweight ground-truth event records, converted to ``MatchEvent`` only when needed.

Records are plain dicts while a match runs (qualifiers are patched as outcomes become known, e.g. a pass's
outcome on arrival or its ``is_assist`` flag when the shot goes in). ``to_match_events`` validates them all.
"""

from dataclasses import dataclass, field
from typing import Any

from pydantic import TypeAdapter

from shared.schemas.events import QUALIFIER_KIND, MatchEvent, Qualifiers

_QUAL = TypeAdapter(Qualifiers)

PLAY_PATTERNS = (
    "regular_play",
    "from_corner",
    "from_free_kick",
    "from_throw_in",
    "from_goal_kick",
    "from_keeper",
    "from_kick_off",
    "from_counter",
    "other",
)


@dataclass
class EventLog:
    records: list[dict[str, Any]] = field(default_factory=list)

    def add(
        self,
        type: str,
        t: float,
        tick: int,
        period: int,
        team: str | None,
        player: str | None,
        location: tuple[float, float, float] | None,
        q: dict[str, Any] | None = None,
        **env: Any,
    ) -> int:
        rec: dict[str, Any] = {
            "type": type,
            "t": round(float(t), 3),
            "tick": int(tick),
            "period": int(period),
            "team": team,
            "player": player,
            "location": None if location is None else tuple(round(float(v), 3) for v in location),
            "q": dict(q or {}),
            "related": [],
        }
        rec.update(env)
        self.records.append(rec)
        return len(self.records) - 1

    def q(self, idx: int) -> dict[str, Any]:
        q: dict[str, Any] = self.records[idx]["q"]
        return q

    def link(self, a: int, b: int) -> None:
        if a < 0 or b < 0:
            return
        if b not in self.records[a]["related"]:
            self.records[a]["related"].append(b)
        if a not in self.records[b]["related"]:
            self.records[b]["related"].append(a)

    def __len__(self) -> int:
        return len(self.records)


def event_id(index: int, rollout_id: str | None = None) -> str:
    return f"ev_r_{rollout_id}_{index}" if rollout_id else f"ev_{index:06d}"


_ID_FIELDS = (
    "receipt_id",
    "assisted_shot_id",
    "key_pass_id",
    "block_event_id",
    "gk_event_id",
    "pass_id",
    "shot_id",
    "cause_event_id",
    "blocked_event_id",
)
_ID_LIST_FIELDS = ("pressure_ids",)


def to_match_events(log: EventLog, rollout_id: str | None = None) -> list[MatchEvent]:
    order = sorted(range(len(log.records)), key=lambda i: (log.records[i]["t"], i))
    new_index = {old: k for k, old in enumerate(order)}

    def ref(i: Any) -> str | None:
        if i is None or i < 0 or i not in new_index:
            return None
        return event_id(new_index[i], rollout_id)

    out: list[MatchEvent] = []
    for k, old in enumerate(order):
        r = log.records[old]
        kind = QUALIFIER_KIND[r["type"]]
        qual = None
        if kind is not None:
            qd = dict(r["q"])
            for f in _ID_FIELDS:
                if f in qd:
                    qd[f] = ref(qd[f])
            for f in _ID_LIST_FIELDS:
                if f in qd:
                    qd[f] = [x for x in (ref(i) for i in qd[f]) if x is not None]
            qd["kind"] = kind
            qual = _QUAL.validate_python(qd)
        env = {
            key: r[key]
            for key in (
                "end_location",
                "possession_id",
                "possession_team",
                "play_pattern",
                "phase_of_play",
                "under_pressure",
                "counterpress",
                "duration_s",
            )
            if key in r and r[key] is not None
        }
        if "end_location" in env:
            env["end_location"] = tuple(round(float(v), 3) for v in env["end_location"])
        related = [x for x in (ref(i) for i in r["related"]) if x is not None]
        out.append(
            MatchEvent(
                id=event_id(k, rollout_id),
                index=k,
                source="ground_truth",
                type=r["type"],
                period=r["period"],
                t=r["t"],
                tick=r["tick"],
                team=r["team"],
                player=r["player"],
                rollout_id=rollout_id,
                location=r["location"],
                related_events=related,
                qualifiers=qual,
                **env,
            )
        )
    return out
