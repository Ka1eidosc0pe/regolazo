"""Rule-based events on Spotter tracks: possession, passes, shots, sprints (docs/event_taxonomy.md thresholds).

Everything here works on pitch-metre tracks (vision.tracking.Tracks); no ground truth is read.
"""

import math
from dataclasses import dataclass

import numpy as np

from shared.geometry import attacks_positive_x
from shared.schemas.common import Period, TeamSide
from shared.schemas.enums import EventType, PassLength, PassOutcome
from shared.schemas.events import MatchEvent, PassQ, PhysicalQ
from vision.geometry import F64
from vision.tracking import Tracks

POSSESSION_RADIUS_M = 1.5
MIN_CONTROL_FRAMES = 3
MIN_POSSESSION_S = 0.4
MIN_PASS_TRAVEL_M = 4.0
SHOT_MIN_SPEED_MPS = 12.0
SHOT_MAX_DIST_M = 35.0
SHOT_GOAL_HALF_WIDTH_M = 3.66 + 4.0
SPRINT_MPS = 7.0
SPRINT_MIN_S = 1.0
SPRINT_MERGE_GAP_S = 0.5
MAX_PLAYER_MPS = 12.0
GOAL_X = 52.5


@dataclass
class Control:
    track: int
    team: TeamSide
    i0: int
    i1: int


@dataclass
class Kinematics:
    pos: F64
    vel: F64
    speed: F64


def kinematics(xy: F64, t: F64, max_gap_s: float = 0.6, win_s: float = 0.5) -> Kinematics:
    """Fill short gaps linearly, moving-average over win_s, central-difference velocity. NaN where unknown."""
    n = len(t)
    pos = np.full((n, 2), np.nan)
    ok = ~np.isnan(xy[:, 0])
    if ok.sum() >= 2:
        idx = np.flatnonzero(ok)
        filled = np.column_stack([np.interp(t, t[idx], xy[idx, k]) for k in range(2)])
        # only keep interpolated values inside short gaps
        nxt = np.searchsorted(idx, np.arange(n))
        prev_t = np.where(nxt > 0, t[idx[np.clip(nxt - 1, 0, len(idx) - 1)]], -np.inf)
        next_t = np.where(nxt < len(idx), t[idx[np.clip(nxt, 0, len(idx) - 1)]], np.inf)
        inside = ok | ((next_t - prev_t) <= max_gap_s)
        dt = float(np.median(np.diff(t))) if n > 1 else 0.04
        w = max(1, round(win_s / dt))
        if w > 1:
            kern = np.ones(w) / w
            pad = w // 2
            for k in range(2):
                col = np.pad(filled[:, k], (pad, w - 1 - pad), mode="edge")
                filled[:, k] = np.convolve(col, kern, mode="valid")
        pos[inside] = filled[inside]
    vel = np.full((n, 2), np.nan)
    if n >= 3:
        vel[1:-1] = (pos[2:] - pos[:-2]) / (t[2:] - t[:-2])[:, None]
    speed = np.linalg.norm(vel, axis=1)
    return Kinematics(pos, vel, speed)


def track_teams(tracks: Tracks, period: Period, home_pos_first: bool) -> dict[int, TeamSide]:
    """Outfield tracks by kit class; goalkeepers by the goal they stand in front of; referees excluded."""
    out: dict[int, TeamSide] = {}
    for k, role in tracks.role.items():
        if role in ("home", "away"):
            out[k] = "home" if role == "home" else "away"
        elif role == "goalkeeper":
            x = float(np.nanmean(tracks.players[k][:, 0]))
            home_attacks_pos = attacks_positive_x("home", period, home_pos_first)
            # a keeper near -x goal belongs to the team attacking +x
            out[k] = "home" if (x < 0) == home_attacks_pos else "away"
    return out


def owners(ball: F64, players: dict[int, F64], teams: dict[int, TeamSide]) -> np.ndarray:
    """Per frame: track id of the nearest team player within POSSESSION_RADIUS_M of the ball, else -1."""
    n = len(ball)
    ids = [k for k in players if k in teams]
    if not ids:
        return np.full(n, -1)
    stack = np.stack([players[k] for k in ids])  # (P, N, 2)
    d = np.linalg.norm(stack - ball[None], axis=2)
    d = np.where(np.isnan(d), np.inf, d)
    j = np.argmin(d, axis=0)
    best = d[j, np.arange(n)]
    return np.where(best <= POSSESSION_RADIUS_M, np.asarray(ids)[j], -1)


def controls(owner: np.ndarray, teams: dict[int, TeamSide], min_frames: int = MIN_CONTROL_FRAMES) -> list[Control]:
    """Runs of the same owner (ignoring frames with no owner); short contacts by a new owner are dropped."""
    runs: list[Control] = []
    i = 0
    n = len(owner)
    while i < n:
        if owner[i] < 0:
            i += 1
            continue
        j = i
        while j + 1 < n and owner[j + 1] == owner[i]:
            j += 1
        k = int(owner[i])
        if runs and runs[-1].track == k:
            runs[-1].i1 = j
        elif j - i + 1 >= min_frames:
            runs.append(Control(k, teams[k], i, j))
        i = j + 1
    return runs


def possession_ids(runs: list[Control], t: F64, min_s: float = MIN_POSSESSION_S) -> list[int]:
    """Possession id per control run. A brief opponent contact (< min_s, and the ball comes straight back) does not
    end a possession; otherwise a run by the other team starts a new one."""
    ids: list[int] = []
    current: TeamSide | None = None
    pid = 0
    for k, r in enumerate(runs):
        if current is None:
            current = r.team
        elif r.team != current:
            nxt = runs[k + 1] if k + 1 < len(runs) else None
            brief = t[r.i1] - t[r.i0] < min_s and nxt is not None and nxt.team == current
            if not brief:
                pid += 1
                current = r.team
        ids.append(pid)
    return ids


def possession_changes(runs: list[Control], t: F64) -> list[tuple[float, TeamSide]]:
    """(time, team gaining) at each new possession."""
    ids = possession_ids(runs, t)
    return [(float(t[r.i0]), r.team) for k, r in enumerate(runs) if k > 0 and ids[k] != ids[k - 1]]


def _length_class(m: float) -> PassLength:
    return "short" if m < 15 else "medium" if m <= 30 else "long"


def _shot_after(
    release: int, team: TeamSide, ball: Kinematics, t: F64, period: Period, home_pos_first: bool
) -> float | None:
    """Peak ball speed if the ball leaves `release` fast and toward the goal `team` attacks; else None."""
    goal_x = GOAL_X if attacks_positive_x(team, period, home_pos_first) else -GOAL_X
    end = int(np.searchsorted(t, t[release] + 0.5))
    window = ball.speed[release : max(end, release + 1)]
    if len(window) == 0 or np.all(np.isnan(window)):
        return None
    k = release + int(np.nanargmax(window))
    v = float(ball.speed[k])
    p, vel = ball.pos[release], ball.vel[k]
    if v < SHOT_MIN_SPEED_MPS or np.isnan(p[0]) or abs(goal_x - p[0]) > SHOT_MAX_DIST_M:
        return None
    if (goal_x - p[0]) * vel[0] <= 0:
        return None
    y_at_line = p[1] + vel[1] * (goal_x - p[0]) / vel[0]
    return v if abs(y_at_line) <= SHOT_GOAL_HALF_WIDTH_M else None


@dataclass
class Raw:
    type: EventType
    t: float
    team: TeamSide
    track: int
    loc: F64
    end: F64 | None = None
    poss: int | None = None
    recv: int | None = None
    outcome: PassOutcome = "complete"
    length: float = 0.0
    speed: float = 0.0
    duration: float = 0.0
    dist: float = 0.0


def detect(
    tracks: Tracks, period: Period, home_pos_first: bool = True, id_prefix: str = "ev_det"
) -> tuple[list[MatchEvent], list[Control]]:
    """Detected events (passes, shots, sprints) ordered by time, plus the control runs (possession)."""
    t = tracks.t
    teams = track_teams(tracks, period, home_pos_first)
    ball = kinematics(tracks.ball, t, max_gap_s=0.5, win_s=0.12)
    kin = {k: kinematics(v, t) for k, v in tracks.players.items() if k in teams}
    owner = owners(ball.pos, {k: v.pos for k, v in kin.items()}, teams)
    runs = controls(owner, teams)
    raw: list[Raw] = []
    poss_ids = possession_ids(runs, t)
    for r_i, r in enumerate(runs):
        possession = poss_ids[r_i]
        nxt = runs[r_i + 1] if r_i + 1 < len(runs) else None
        start = ball.pos[r.i1]
        tr = float(t[r.i1])
        # a run can span several touches (no one else came within reach in between): try each release
        releases = [
            i for i in range(r.i0, r.i1 + 1) if owner[i] == r.track and (i + 1 >= len(owner) or owner[i + 1] != r.track)
        ]
        shot = next(
            ((i, v) for i in releases if (v := _shot_after(i, r.team, ball, t, period, home_pos_first)) is not None),
            None,
        )
        if shot is not None:
            i, v = shot
            raw.append(Raw("shot", float(t[i]), r.team, r.track, ball.pos[i], poss=possession, speed=v))
            continue
        if nxt is None or nxt.track == r.track:
            continue
        end = ball.pos[nxt.i0]
        travel = float(np.linalg.norm(end - start))
        if np.isnan(travel) or travel < MIN_PASS_TRAVEL_M:
            continue
        seg = ball.speed[r.i1 : nxt.i0 + 1]
        raw.append(
            Raw(
                "pass",
                tr,
                r.team,
                r.track,
                start,
                end=end,
                poss=possession,
                recv=nxt.track,
                outcome="complete" if nxt.team == r.team else "intercepted",
                length=travel,
                speed=float(np.nanmax(seg)) if np.any(~np.isnan(seg)) else 0.0,
                duration=float(t[nxt.i0] - t[r.i1]),
            )
        )
    for k, kn in kin.items():
        for i0, i1, peak, dist in sprints(kn.speed, t):
            raw.append(
                Raw(
                    "sprint",
                    float(t[i0]),
                    teams[k],
                    k,
                    kn.pos[i0],
                    end=kn.pos[i1],
                    speed=peak,
                    dist=dist,
                    duration=float(t[i1] - t[i0]),
                )
            )
    raw.sort(key=lambda e: e.t)
    return [_to_event(i, e, period, id_prefix) for i, e in enumerate(raw)], runs


def sprints(speed: F64, t: F64) -> list[tuple[int, int, float, float]]:
    """(i0, i1, peak m/s, distance m) of runs above SPRINT_MPS for >= SPRINT_MIN_S, merging short gaps."""
    s = np.where(np.isnan(speed) | (speed > MAX_PLAYER_MPS), 0.0, speed)
    fast = np.flatnonzero(s > SPRINT_MPS)
    if len(fast) == 0:
        return []
    groups: list[list[int]] = [[int(fast[0]), int(fast[0])]]
    for i in fast[1:]:
        if t[i] - t[groups[-1][1]] < SPRINT_MERGE_GAP_S:
            groups[-1][1] = int(i)
        else:
            groups.append([int(i), int(i)])
    out = []
    for i0, i1 in groups:
        if t[i1] - t[i0] >= SPRINT_MIN_S:
            dt = np.diff(t[i0 : i1 + 1])
            dist = float(np.sum(s[i0:i1] * dt))
            out.append((i0, i1, float(np.max(s[i0 : i1 + 1])), dist))
    return out


def _v3(p: F64 | None) -> tuple[float, float, float] | None:
    if p is None or np.isnan(p).any():
        return None
    return (float(p[0]), float(p[1]), 0.0)


def _to_event(i: int, e: Raw, period: Period, prefix: str) -> MatchEvent:
    loc, end = _v3(e.loc), _v3(e.end)
    q: PassQ | PhysicalQ | None = None
    if e.type == "pass" and loc is not None and end is not None:
        q = PassQ(
            recipient=None,
            outcome=e.outcome,
            height=None,
            length_class=_length_class(e.length),
            body_part=None,
            technique=None,
            pass_type=None,
            length_m=e.length,
            angle_rad=math.atan2(end[1] - loc[1], end[0] - loc[0]),
            ball_speed_mps=e.speed,
            peak_height_m=0.0,  # ball height is not estimated (ground-plane projection only)
            duration_s=e.duration,
        )
    elif e.type == "sprint":
        q = PhysicalQ(distance_m=e.dist, peak_speed_mps=e.speed)
    return MatchEvent(
        id=f"{prefix}_{i:06d}",
        index=i,
        source="detected",
        type=e.type,
        period=period,
        t=e.t,
        tick=round(e.t * 25),
        duration_s=e.duration,
        team=e.team,
        player=None,
        track_id=e.track,
        secondary_track_id=e.recv,
        location=loc,
        end_location=end,
        possession_id=e.poss,
        possession_team=e.team if e.poss is not None else None,
        qualifiers=q,
    )
