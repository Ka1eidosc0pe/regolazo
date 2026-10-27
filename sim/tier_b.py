"""Tier B ground truth derived from the simulator's exact tracks (docs/event_taxonomy.md section 5).

pressure (+ counterpress), pressing_sequence, sprint, high_intensity_run, acceleration and line_step_up are
computed after the match with the taxonomy's operational rules, so the Spotter (which applies the same rules
to detected tracks) is scored against like-for-like definitions. Off-ball runs come from the AI's intent and are
emitted by the match engine. Finally, on-ball events overlapping a pressure on their actor get
``under_pressure = True`` (taxonomy section 4.6).
"""

from typing import Any

import numpy as np
import numpy.typing as npt

from shared.recording import PHASES
from sim.constants import (
    ACCEL_MIN_DV,
    ACCEL_MIN_S,
    ACCEL_THRESHOLD,
    COUNTERPRESS_WINDOW_S,
    DT,
    HIR_MPS,
    PRESSURE_CLOSE_RADIUS,
    PRESSURE_CLOSING_MPS,
    PRESSURE_MIN_S,
    PRESSURE_RADIUS,
    SPEED_EVENT_MIN_S,
    SPRINT_MPS,
    STEP_UP_ADVANCE_M,
    STEP_UP_WINDOW_S,
)
from sim.eventlog import EventLog
from sim.match import Match

F = npt.NDArray[np.float64]
SIDE = ("home", "away")
ON_BALL = {
    "pass",
    "ball_receipt",
    "carry",
    "take_on",
    "shot",
    "clearance",
    "dispossessed",
    "miscontrol",
}


def runs(mask: npt.NDArray[np.bool_], min_len: int, merge_gap: int = 0) -> list[tuple[int, int]]:
    """[start, end) index pairs of True runs at least ``min_len`` long, merging gaps shorter than ``merge_gap``."""
    if not mask.any():
        return []
    m = np.concatenate([[False], mask, [False]]).astype(np.int8)
    d = np.diff(m)
    starts, ends = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
    out: list[list[int]] = []
    for s, e in zip(starts, ends, strict=True):
        if out and s - out[-1][1] < merge_gap:
            out[-1][1] = int(e)
        else:
            out.append([int(s), int(e)])
    return [(s, e) for s, e in out if e - s >= min_len]


class _Ctx:
    def __init__(self, m: Match, a: dict[str, npt.NDArray[Any]], log: EventLog) -> None:
        self.m, self.a, self.log = m, a, log
        meta = m.meta
        self.ids = [p.id for p in meta.home.players] + [p.id for p in meta.away.players]
        n_home = len(meta.home.players)
        self.team_of = np.array([0] * n_home + [1] * (len(self.ids) - n_home))
        self.on = np.nonzero(a["player_on_pitch"][0])[0]
        self.gk = {int(m.sq.rec_index[int(m._gk[t])]) for t in (0, 1)}
        self.t = a["t"]

    def add(
        self,
        type_: str,
        k0: int,
        k1: int,
        player: int | None,
        team: int | None,
        loc: Any,
        q: dict[str, Any],
        end: Any = None,
    ) -> int:
        a = self.a
        pt = int(a["poss_team"][k0])
        env: dict[str, Any] = {
            "duration_s": float(self.t[min(k1, len(self.t) - 1)] - self.t[k0]),
            "possession_id": int(a["possession_id"][k0]),
            "phase_of_play": PHASES[int(a["phase"][k0])],
        }
        if pt >= 0:
            env["possession_team"] = SIDE[pt]
            if team is not None and team != pt and PHASES[int(a["phase"][k0])] != "dead_ball":
                env["phase_of_play"] = "out_of_possession"
        if end is not None:
            env["end_location"] = (float(end[0]), float(end[1]), 0.0)
        return self.log.add(
            type_,
            float(self.t[k0]),
            int(a["tick"][k0]),
            int(a["period"][k0]),
            None if team is None else SIDE[team],
            None if player is None else self.ids[player],
            None if loc is None else (float(loc[0]), float(loc[1]), 0.0),
            q,
            **env,
        )


def derive_tier_b(m: Match, arrays: dict[str, npt.NDArray[Any]], log: EventLog) -> None:
    if len(arrays.get("t", [])) < 2:
        return
    c = _Ctx(m, arrays, log)
    _physical(c)
    pressures = _pressure(c)
    _pressing_sequences(c, pressures)
    _step_ups(c)
    _flag_under_pressure(c, pressures)


# --- physical: sprints, high-intensity runs, accelerations ---------------------------------------------------
def _physical(c: _Ctx) -> None:
    a = c.a
    vel = a["player_vel"].astype(np.float64)
    pos = a["player_pos"].astype(np.float64)
    live = a["status"] != 2  # skip half-time re-positioning
    n_min = round(SPEED_EVENT_MIN_S / DT)
    for j in c.on:
        speed = np.linalg.norm(vel[:, j], axis=1)
        speed = np.where(live, speed, 0.0)
        step = np.linalg.norm(np.diff(pos[:, j], axis=0), axis=1)
        step = np.concatenate([[0.0], np.where(step < 1.0, step, 0.0)])
        team = int(c.team_of[j])
        for s, e in runs(speed > SPRINT_MPS, n_min, merge_gap=int(0.5 / DT)):
            q: dict[str, Any] = {"distance_m": float(step[s:e].sum()), "peak_speed_mps": float(speed[s:e].max())}
            c.add("sprint", s, e, j, team, pos[s, j], q, pos[e - 1, j])
        for s, e in runs((speed > HIR_MPS), n_min):
            if speed[s:e].max() > SPRINT_MPS:
                continue
            q = {"distance_m": float(step[s:e].sum()), "peak_speed_mps": float(speed[s:e].max())}
            c.add("high_intensity_run", s, e, j, team, pos[s, j], q, pos[e - 1, j])
        acc = np.concatenate([[0.0], np.diff(speed) / DT])
        k_min = round(ACCEL_MIN_S / DT)
        for sign, name in ((1.0, "acceleration"), (-1.0, "deceleration")):
            for s, e in runs(sign * acc >= ACCEL_THRESHOLD, k_min):
                s0 = max(s - 1, 0)
                dv = speed[e - 1] - speed[s0]
                if abs(dv) < ACCEL_MIN_DV:
                    continue
                q = {
                    "distance_m": float(step[s:e].sum()),
                    "peak_speed_mps": float(speed[s0:e].max()),
                    "accel_type": name,
                    "peak_accel_mps2": float((sign * acc[s:e]).max() * sign),
                    "speed_start_mps": float(speed[s0]),
                    "speed_end_mps": float(speed[e - 1]),
                }
                c.add("acceleration", s, e, j, team, pos[s, j], q, pos[e - 1, j])


# --- pressure ---------------------------------------------------------------------------------------------------
def _pressed_series(c: _Ctx) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Per frame: pressed player (rec index or -1) and role (0 carrier, 1 receiver, 2 releaser)."""
    a = c.a
    n = len(c.t)
    who = a["poss_player"].astype(np.int64).copy()
    role = np.where(who >= 0, 0, -1)
    idx = {pid: i for i, pid in enumerate(c.ids)}
    for r in c.log.records:
        if r["type"] == "ball_receipt" and r["player"] is not None:
            k = int(np.searchsorted(c.t, r["t"]))
            k0 = max(k - int(1.0 / DT), 0)
            sel = slice(k0, min(k, n))
            free = who[sel] < 0
            who[sel] = np.where(free, idx[r["player"]], who[sel])
            role[sel] = np.where(free, 1, role[sel])
        elif r["type"] == "pass" and r["player"] is not None:
            k = int(np.searchsorted(c.t, r["t"]))
            sel = slice(k, min(k + int(0.5 / DT), n))
            free = who[sel] < 0
            who[sel] = np.where(free, idx[r["player"]], who[sel])
            role[sel] = np.where(free, 2, role[sel])
    live = a["ball_in_play"].astype(bool)
    who = np.where(live, who, -1)
    return who, role


def _pressure(c: _Ctx) -> list[dict[str, Any]]:
    a = c.a
    pos = a["player_pos"].astype(np.float64)
    vel = a["player_vel"].astype(np.float64)
    who, role = _pressed_series(c)
    n = len(c.t)
    out: list[dict[str, Any]] = []
    has = who >= 0
    tgt = np.maximum(who, 0)
    tpos = pos[np.arange(n), tgt]
    tteam = c.team_of[tgt]
    min_len = round(PRESSURE_MIN_S / DT)
    turnover_t = _turnover_times(c)
    for j in c.on:
        if j in c.gk:
            continue
        team = int(c.team_of[j])
        valid = has & (tteam != team)
        rel = tpos - pos[:, j]
        d = np.linalg.norm(rel, axis=1)
        dd = np.concatenate([[0.0], np.diff(d) / DT])
        toward = np.sum(vel[:, j] * rel, axis=1) / np.maximum(d, 1e-6)
        ok = valid & (d <= PRESSURE_RADIUS) & ((dd <= PRESSURE_CLOSING_MPS) | (d <= PRESSURE_CLOSE_RADIUS))
        ok &= (toward >= 1.0) | (d <= PRESSURE_CLOSE_RADIUS)
        # split runs when the pressed player changes
        change = np.concatenate([[False], who[1:] != who[:-1]])
        ok_split = ok & ~change
        for s, e in runs(ok_split, min_len, merge_gap=3):
            target = int(who[s])
            if target < 0:
                continue
            t0 = float(c.t[s])
            q = {
                "pressed_player": c.ids[target],
                "pressed_role": ("carrier", "receiver", "releaser")[int(max(role[s], 0))],
                "min_distance_m": float(d[s:e].min()),
                "closing_speed_mps": float(-np.mean(dd[s:e])),
                "n_pressers": 1,
                "outcome": "none",
            }
            counter = any(0 <= t0 - tt <= COUNTERPRESS_WINDOW_S for tt, tm in turnover_t if tm == team)
            idx = c.add("pressure", s, e, j, team, pos[s, j], q)
            c.log.records[idx]["counterpress"] = bool(counter)
            out.append(
                {
                    "idx": idx,
                    "s": s,
                    "e": e,
                    "player": j,
                    "team": team,
                    "target": target,
                    "pid": int(a["possession_id"][s]),
                }
            )
    _pressure_outcomes(c, out)
    return out


def _turnover_times(c: _Ctx) -> list[tuple[float, int]]:
    """(t, team that lost the ball) for every open-play change of possession team."""
    pt = c.a["poss_team"].astype(int)
    live = c.a["ball_in_play"].astype(bool)
    out = []
    last = -1
    for k in np.nonzero(pt >= 0)[0]:
        if pt[k] != last and last >= 0 and live[k]:
            out.append((float(c.t[k]), last))
        last = int(pt[k])
    return out


def _pressure_outcomes(c: _Ctx, pressures: list[dict[str, Any]]) -> None:
    a = c.a
    pt = a["poss_team"].astype(int)
    rec_t = [
        (r["t"], r["type"], r["player"])
        for r in c.log.records
        if r["type"] in ("pass", "clearance", "ball_out_of_play")
    ]
    rec_t.sort()
    times = np.array([x[0] for x in rec_t]) if rec_t else np.zeros(0)
    by_frame: dict[int, int] = {}
    for p in pressures:
        for k in range(p["s"], p["e"]):
            by_frame[k] = by_frame.get(k, 0) + 1
    for p in pressures:
        q = c.log.records[p["idx"]]["q"]
        q["n_pressers"] = max(1, max(by_frame.get(k, 1) for k in range(p["s"], p["e"])))
        e = p["e"]
        k3 = min(e + int(3.0 / DT), len(pt) - 1)
        if (pt[e : k3 + 1] == p["team"]).any():
            q["outcome"] = "forced_turnover"
            continue
        t_end = float(c.t[min(e, len(c.t) - 1)])
        i0 = int(np.searchsorted(times, float(c.t[p["s"]])))
        i1 = int(np.searchsorted(times, t_end + 2.0))
        pressed = c.ids[p["target"]]
        if any(rec_t[i][2] in (pressed, None) for i in range(i0, i1)):
            q["outcome"] = "forced_hurried_action"


def _pressing_sequences(c: _Ctx, pressures: list[dict[str, Any]]) -> None:
    a = c.a
    pt = a["poss_team"].astype(int)
    pressures = sorted(pressures, key=lambda p: (p["team"], p["s"]))
    groups: list[list[dict[str, Any]]] = []
    for p in pressures:
        g = groups[-1] if groups else None
        if (
            g is not None
            and g[-1]["team"] == p["team"]
            and g[-1]["pid"] == p["pid"]
            and (p["s"] - max(x["e"] for x in g)) * DT <= 3.0
        ):
            g.append(p)
        else:
            groups.append([p])
    whistles = {r["tick"] for r in c.log.records if r["type"] in ("foul_committed", "offside")}
    outs = {r["tick"] for r in c.log.records if r["type"] == "ball_out_of_play"}
    ticks = a["tick"]
    for g in groups:
        players = {x["player"] for x in g}
        if len(g) < 2 or len(players) < 2:
            continue
        s, e = min(x["s"] for x in g), max(x["e"] for x in g)
        team = g[0]["team"]
        k3 = min(e + int(3.0 / DT), len(pt) - 1)
        window = set(int(t) for t in ticks[s : k3 + 1])
        if (pt[e : k3 + 1] == team).any():
            outcome = "success"
        elif window & whistles:
            outcome = "whistle"
        elif window & outs:
            outcome = "out"
        else:
            outcome = "broken"
        q = {
            "n_pressures": len(g),
            "pressing_players": sorted(c.ids[x["player"]] for x in g),
            "outcome": outcome,
            "pressure_ids": [x["idx"] for x in g],
        }
        c.add("pressing_sequence", s, e, None, team, None, q)


# --- defensive line step-ups -----------------------------------------------------------------------------------
def _step_ups(c: _Ctx) -> None:
    a = c.a
    pos = a["player_pos"].astype(np.float64)
    vel = a["player_vel"].astype(np.float64)
    attack = a["attacking_positive_x"].astype(int)
    n = len(c.t)
    win = int(STEP_UP_WINDOW_S / DT)
    offs = [(r["t"], r["team"]) for r in c.log.records if r["type"] == "offside"]
    passes = [(r["t"], r["team"], r["q"].get("outcome")) for r in c.log.records if r["type"] == "pass"]
    for team in (0, 1):
        players = [j for j in c.on if c.team_of[j] == team and j not in c.gk]
        sgn = np.where((attack == team), 1.0, -1.0)  # +1 when this team attacks +x
        xs = pos[:, players, 0] * sgn[:, None]
        order = np.argsort(xs, axis=1)[:, :4]
        back = np.take_along_axis(xs, order, axis=1)
        line = back.mean(axis=1)
        vx = np.take_along_axis(vel[:, players, 0] * sgn[:, None], order, axis=1)
        k = win
        live = a["ball_in_play"].astype(bool)
        while k < n:
            adv = line[k] - line[k - win]
            moving = (vx[k] >= 1.5).sum() >= 3 and np.std(vx[k]) < 1.0
            if adv >= STEP_UP_ADVANCE_M and moving and live[k] and live[k - win]:
                s = k - win
                t_end = float(c.t[k])
                opp = SIDE[1 - team]
                trap = any(0 <= t - t_end <= 1.5 and tm == opp for t, tm in offs)
                beaten = any(0 <= t - t_end <= 1.5 and tm == opp and o == "complete" for t, tm, o in passes)
                outcome = "trap_success" if trap else ("trap_beaten" if beaten else "reset")
                ids = [c.ids[players[i]] for i in order[k]]
                q = {
                    "line_players": ids,
                    "line_height_before": float(line[s] * sgn[s]),
                    "line_height_after": float(line[k] * sgn[k]),
                    "outcome": outcome,
                }
                c.add("line_step_up", s, k, None, team, None, q)
                k += win * 2
            else:
                k += 5


def _flag_under_pressure(c: _Ctx, pressures: list[dict[str, Any]]) -> None:
    spans: dict[str, list[tuple[float, float]]] = {}
    for p in pressures:
        pid = c.ids[p["target"]]
        spans.setdefault(pid, []).append((float(c.t[p["s"]]), float(c.t[min(p["e"], len(c.t) - 1)])))
    for r in c.log.records:
        if r["type"] in ON_BALL and r["player"] is not None:
            t0 = r["t"]
            t1 = t0 + float(r.get("duration_s") or 0.0)
            r["under_pressure"] = any(s <= t1 + 0.05 and t0 - 0.05 <= e for s, e in spans.get(r["player"], []))
