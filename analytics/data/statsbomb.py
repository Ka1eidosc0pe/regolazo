"""StatsBomb Open Data loaders and feature extraction.

StatsBomb event coordinates are 120 x 80 yards with the acting team always attacking +x, so every event is already in
that team's attacking frame; ``shared.geometry`` converts them to metres. 360 freeze frames use the same frame as the
event they belong to.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from analytics.features import (
    count_in_lane,
    count_in_shot_cone,
    nearest_distance,
    pad_players,
)
from analytics.paths import statsbomb_dir
from shared.geometry import statsbomb_goal_plane, statsbomb_to_attacking_frame
from shared.model_api import GOAL_X, PASS_HEIGHT, PASS_KIND, SHOT_BODY_PART, SHOT_SITUATION, SHOT_TECHNIQUE

Event = dict[str, Any]
Table = dict[str, npt.NDArray[Any]]

SET_PIECE_WINDOW_S = 10.0
"""Shots within this long of a corner / free kick / throw-in restart count as ``set_piece`` (taxonomy 3.5.2)."""
ON_TARGET = {"Goal", "Saved", "Saved To Post"}
EXCLUDED_PASS_TYPES = {"Throw-in", "Kick Off"}
EXCLUDED_PASS_OUTCOMES = {"Unknown", "Injury Clearance"}
CHIP_MAX_LENGTH_M = 25.0
BACK_PASS_ANGLE_RAD = math.radians(135.0)

_SB_TECHNIQUE = {
    "Lob": "chip",
    "Volley": "volley",
    "Half Volley": "half_volley",
    "Diving Header": "diving_header",
    "Backheel": "backheel",
    "Overhead Kick": "overhead_kick",
}
_SB_HEIGHT = {"Ground Pass": "ground", "Low Pass": "low", "High Pass": "high"}


@dataclass(frozen=True)
class MatchInfo:
    match_id: int
    competition_id: int
    season_id: int
    competition: str
    season: str
    gender: str
    home_team_id: int
    away_team_id: int
    home_score: int
    away_score: int
    has_360: bool


def list_matches(root: Path | None = None) -> list[MatchInfo]:
    """Every match whose events file is in the cache."""
    root = root or statsbomb_dir()
    comps = {(c["competition_id"], c["season_id"]): c for c in json.loads((root / "competitions.json").read_text())}
    out: list[MatchInfo] = []
    for path in sorted((root / "matches").glob("*/*.json")):
        for m in json.loads(path.read_text()):
            mid = int(m["match_id"])
            if not (root / "events" / f"{mid}.json").exists():
                continue
            comp = comps.get((m["competition"]["competition_id"], m["season"]["season_id"]), {})
            out.append(
                MatchInfo(
                    match_id=mid,
                    competition_id=int(m["competition"]["competition_id"]),
                    season_id=int(m["season"]["season_id"]),
                    competition=str(m["competition"]["competition_name"]),
                    season=str(m["season"]["season_name"]),
                    gender=str(comp.get("competition_gender", "male")),
                    home_team_id=int(m["home_team"]["home_team_id"]),
                    away_team_id=int(m["away_team"]["away_team_id"]),
                    home_score=int(m["home_score"]),
                    away_score=int(m["away_score"]),
                    has_360=(root / "three-sixty" / f"{mid}.json").exists(),
                )
            )
    return out


def load_events(match_id: int, root: Path | None = None) -> list[Event]:
    root = root or statsbomb_dir()
    events: list[Event] = json.loads((root / "events" / f"{match_id}.json").read_text())
    return events


def load_frames_360(match_id: int, root: Path | None = None) -> dict[str, list[dict[str, Any]]]:
    """event uuid -> 360 freeze frame (visible players), empty if the match has no 360 data."""
    root = root or statsbomb_dir()
    path = root / "three-sixty" / f"{match_id}.json"
    if not path.exists():
        return {}
    frames: list[dict[str, Any]] = json.loads(path.read_text())
    return {f["event_uuid"]: f["freeze_frame"] for f in frames}


def to_m(loc: list[float]) -> tuple[float, float]:
    return statsbomb_to_attacking_frame(float(loc[0]), float(loc[1]))


def timestamp_s(ts: str) -> float:
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def _name(obj: Event, *keys: str) -> str | None:
    cur: Any = obj
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return str(cur) if cur is not None else None


def shot_body_part(shot: Event) -> int:
    bp = _name(shot, "body_part", "name") or ""
    if bp == "Head":
        return SHOT_BODY_PART["head"]
    if bp in ("Left Foot", "Right Foot"):
        return SHOT_BODY_PART["foot"]
    return SHOT_BODY_PART["other"]


def shot_technique(shot: Event) -> int:
    """SB technique -> SHOT_TECHNIQUE code. SB "Normal" maps to ``header`` for headed shots, else ``placed``."""
    tech = _name(shot, "technique", "name") or "Normal"
    if tech in _SB_TECHNIQUE:
        return SHOT_TECHNIQUE[_SB_TECHNIQUE[tech]]
    if shot_body_part(shot) == SHOT_BODY_PART["head"]:
        return SHOT_TECHNIQUE["header"]
    return SHOT_TECHNIQUE["placed"]


def shot_situation(event: Event, possession_start_s: float) -> int:
    shot_type = _name(event, "shot", "type", "name")
    if shot_type == "Penalty":
        return SHOT_SITUATION["penalty"]
    if shot_type == "Free Kick":
        return SHOT_SITUATION["direct_free_kick"]
    pattern = _name(event, "play_pattern", "name")
    if pattern == "From Counter":
        return SHOT_SITUATION["counter_attack"]
    since = timestamp_s(event["timestamp"]) - possession_start_s
    if pattern in ("From Corner", "From Free Kick", "From Throw In") and since <= SET_PIECE_WINDOW_S:
        return SHOT_SITUATION["set_piece"]
    return SHOT_SITUATION["open_play"]


def goal_plane_end(start: tuple[float, float], end: list[float]) -> tuple[float, float, float]:
    """Project a shot's SB end location onto the goal plane. Returns (end_y, end_z, end_x) in metres.

    Saved shots end where the keeper touched the ball (often 1-4 yd off the line); y is extrapolated along the shot
    line to the goal line and z is scaled by the same ratio (ball rising from ground level). The projected values are
    snapped back to StatsBomb's 0.1-yard lattice: otherwise every goal (recorded on the lattice) and every save
    (projected off it) would be separable by quantisation alone, which a tree model learns (AUC 0.98 on placement).
    """
    ex, ey = to_m(end)
    y_sb = float(end[1])
    z_sb = float(end[2]) if len(end) > 2 else float("nan")
    sx_sb, sy_sb = float(start[0]) * 120.0 / 105.0 + 60.0, 40.0 - float(start[1]) * 80.0 / 68.0
    if ex < GOAL_X - 0.05 and ex > start[0] + 0.5:
        ratio = (120.0 - sx_sb) / (float(end[0]) - sx_sb)
        y_sb = round(sy_sb + (y_sb - sy_sb) * ratio, 1)
        z_sb = round(z_sb * min(ratio, 3.0), 1)
    gy, gz = statsbomb_goal_plane(y_sb, z_sb)
    return gy, float(np.clip(gz, 0.0, 2.44)) if not math.isnan(gz) else float("nan"), ex


def possession_starts(events: list[Event]) -> dict[tuple[int, int], float]:
    starts: dict[tuple[int, int], float] = {}
    for e in events:
        key = (int(e["period"]), int(e["possession"]))
        starts.setdefault(key, timestamp_s(e["timestamp"]))
    return starts


def extract_shots(events: list[Event], match_id: int = 0) -> list[dict[str, float]]:
    """One feature row per shot (penalty shoot-outs excluded)."""
    starts = possession_starts(events)
    rows: list[dict[str, float]] = []
    for e in events:
        if e["type"]["name"] != "Shot" or int(e["period"]) > 4:
            continue
        shot = e["shot"]
        x, y = to_m(e["location"])
        opp: list[tuple[float, float]] = []
        gk: tuple[float, float] | None = None
        for p in shot.get("freeze_frame", []) or []:
            if p.get("teammate"):
                continue
            loc = to_m(p["location"])
            if _name(p, "position", "name") == "Goalkeeper":
                gk = loc
            else:
                opp.append(loc)
        opp_arr = pad_players([opp])
        xa, ya = np.array([x]), np.array([y])
        outcome = _name(shot, "outcome", "name") or ""
        on_target = outcome in ON_TARGET
        end = shot.get("end_location")
        end_y = end_z = end_x = float("nan")
        if end is not None:
            end_y, end_z, end_x = goal_plane_end((x, y), end)
            if len(end) == 2:
                end_z = float("nan")
        travel = float("nan")
        if end is not None:
            ez = end_z if not math.isnan(end_z) else 0.0
            travel = math.dist((x, y, 0.0), (end_x, end_y, ez))
        duration = float(e.get("duration") or float("nan"))
        rows.append(
            {
                "match_id": float(match_id),
                "team_id": float(e["team"]["id"]),
                "period": float(e["period"]),
                "minute": float(e["minute"]),
                "x": x,
                "y": y,
                "body_part": float(shot_body_part(shot)),
                "technique": float(shot_technique(shot)),
                "situation": float(shot_situation(e, starts[(int(e["period"]), int(e["possession"]))])),
                "gk_found": float(gk is not None),
                "gk_x": gk[0] if gk else GOAL_X,
                "gk_y": gk[1] if gk else 0.0,
                "n_defenders_in_cone": float(count_in_shot_cone(xa, ya, opp_arr)[0]),
                "nearest_defender_m": float(nearest_distance(xa, ya, opp_arr)[0]),
                "under_pressure": float(bool(e.get("under_pressure", False))),
                "first_time": float(bool(shot.get("first_time", False))),
                "goal": float(outcome == "Goal"),
                "on_target": float(on_target),
                "end_y": end_y,
                "end_z": end_z,
                "end_x": end_x,
                "duration_s": duration,
                "speed_proxy_mps": travel / duration if duration > 0 else float("nan"),
                "sb_xg": float(shot.get("statsbomb_xg", float("nan"))),
                "freeze_frame_n": float(len(opp)),
            }
        )
    return rows


def pass_kind(p: Event, x: float, y: float, end_x: float, end_y: float) -> int:
    """SB pass qualifiers -> PASS_KIND code, precedence cutback > cross > through ball > switch > chipped > back."""
    q = p["pass"]
    length = math.hypot(end_x - x, end_y - y)
    if q.get("cut_back"):
        return PASS_KIND["cutback"]
    if q.get("cross"):
        return PASS_KIND["cross"]
    if _name(q, "technique", "name") == "Through Ball" or q.get("through_ball"):
        return PASS_KIND["through_ball"]
    if q.get("switch"):
        return PASS_KIND["switch"]
    if _name(q, "height", "name") == "High Pass" and length <= CHIP_MAX_LENGTH_M:
        return PASS_KIND["chipped"]
    if abs(math.atan2(end_y - y, end_x - x)) >= BACK_PASS_ANGLE_RAD and length >= 3.0:
        return PASS_KIND["back_pass"]
    return PASS_KIND["standard"]


def extract_passes(events: list[Event], frames: dict[str, list[dict[str, Any]]]) -> list[dict[str, float]]:
    """One xPass row per open-play / set-piece pass that has a 360 freeze frame (throw-ins and kick-offs excluded)."""
    rows: list[dict[str, float]] = []
    for e in events:
        if e["type"]["name"] != "Pass" or int(e["period"]) > 4 or e["id"] not in frames:
            continue
        q = e["pass"]
        if _name(q, "type", "name") in EXCLUDED_PASS_TYPES:
            continue
        outcome = _name(q, "outcome", "name")
        if outcome in EXCLUDED_PASS_OUTCOMES:
            continue
        x, y = to_m(e["location"])
        end_x, end_y = to_m(q["end_location"])
        opp = [to_m(p["location"]) for p in frames[e["id"]] if not p.get("teammate")]
        mates = [to_m(p["location"]) for p in frames[e["id"]] if p.get("teammate") and not p.get("actor")]
        opp_arr = pad_players([opp])
        xa, ya, exa, eya = np.array([x]), np.array([y]), np.array([end_x]), np.array([end_y])
        rows.append(
            {
                "x": x,
                "y": y,
                "end_x": end_x,
                "end_y": end_y,
                "height": float(PASS_HEIGHT[_SB_HEIGHT.get(_name(q, "height", "name") or "", "ground")]),
                "kind": float(pass_kind(e, x, y, end_x, end_y)),
                "nearest_defender_to_receiver_m": float(nearest_distance(exa, eya, opp_arr)[0]),
                "defenders_in_lane": float(count_in_lane(xa, ya, exa, eya, opp_arr)[0]),
                "under_pressure": float(bool(e.get("under_pressure", False))),
                "completed": float(outcome is None),
                "n_opponents_visible": float(len(opp)),
                "n_teammates_visible": float(len(mates)),
                "head": float(_name(q, "body_part", "name") == "Head"),
            }
        )
    return rows


def extract_moves(events: list[Event]) -> list[dict[str, float]]:
    """Ball moves (passes and carries) and non-penalty shots for xT, in the acting team's attacking frame."""
    rows: list[dict[str, float]] = []
    for e in events:
        name = e["type"]["name"]
        if int(e["period"]) > 4 or "location" not in e:
            continue
        x, y = to_m(e["location"])
        if name == "Pass":
            q = e["pass"]
            if _name(q, "outcome", "name") in EXCLUDED_PASS_OUTCOMES:
                continue
            ex, ey = to_m(q["end_location"])
            rows.append({"kind": 0.0, "x": x, "y": y, "end_x": ex, "end_y": ey, "success": float("outcome" not in q)})
        elif name == "Carry":
            ex, ey = to_m(e["carry"]["end_location"])
            rows.append({"kind": 0.0, "x": x, "y": y, "end_x": ex, "end_y": ey, "success": 1.0})
        elif name == "Shot" and _name(e, "shot", "type", "name") != "Penalty":
            goal = _name(e, "shot", "outcome", "name") == "Goal"
            rows.append({"kind": 1.0, "x": x, "y": y, "end_x": x, "end_y": y, "success": float(goal)})
    return rows


def team_match_stats(events: list[Event], info: MatchInfo) -> list[dict[str, float]]:
    """Per-team box-score stats for the realism targets (periods 1-4; passes exclude throw-ins)."""
    out: list[dict[str, float]] = []
    for team_id in (info.home_team_id, info.away_team_id):
        shots = [e for e in events if e["type"]["name"] == "Shot" and e["team"]["id"] == team_id and e["period"] <= 4]
        passes = [
            e
            for e in events
            if e["type"]["name"] == "Pass"
            and e["team"]["id"] == team_id
            and e["period"] <= 4
            and _name(e, "pass", "type", "name") != "Throw-in"
        ]
        completed = sum(1 for e in passes if "outcome" not in e["pass"])
        goals = info.home_score if team_id == info.home_team_id else info.away_score
        dists = [math.hypot(GOAL_X - to_m(s["location"])[0], to_m(s["location"])[1]) for s in shots]
        headers = sum(1 for s in shots if _name(s, "shot", "body_part", "name") == "Head")
        out.append(
            {
                "match_id": float(info.match_id),
                "competition_id": float(info.competition_id),
                "team_id": float(team_id),
                "goals": float(goals),
                "shots": float(len(shots)),
                "passes": float(len(passes)),
                "passes_completed": float(completed),
                "header_shots": float(headers),
                "shot_distance_sum": float(sum(dists)),
            }
        )
    return out


def rows_to_table(rows: list[dict[str, float]]) -> Table:
    if not rows:
        return {}
    return {k: np.array([r[k] for r in rows], dtype=np.float64) for k in rows[0]}


def concat_tables(tables: list[Table]) -> Table:
    tables = [t for t in tables if t]
    if not tables:
        return {}
    return {k: np.concatenate([t[k] for t in tables]) for k in tables[0]}
