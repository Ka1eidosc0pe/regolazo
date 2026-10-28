"""Spotter evaluation against the recording's ground truth (evaluation only; never used for inference)."""

from dataclasses import dataclass

import numpy as np

from shared.recording import Recording
from shared.schemas.common import TeamSide
from shared.schemas.events import MatchEvent
from shared.schemas.overlay import AccuracyPanelPayload, AccuracyRow
from vision.events import Control, ball_kinematics, kinematics, possession_changes
from vision.tracking import Tracks


@dataclass
class Score:
    tp: int
    fp: int
    fn: int
    pairs: list[tuple[int, int]]

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


def match(det: list[tuple[float, TeamSide]], gt: list[tuple[float, TeamSide]], tol: float) -> Score:
    """Greedy one-to-one matching by time (closest pairs first), same team, |dt| <= tol."""
    cands = sorted((abs(d[0] - g[0]), i, j) for i, d in enumerate(det) for j, g in enumerate(gt) if d[1] == g[1])
    used_d: set[int] = set()
    used_g: set[int] = set()
    pairs = []
    for dt, i, j in cands:
        if dt > tol:
            break
        if i in used_d or j in used_g:
            continue
        used_d.add(i)
        used_g.add(j)
        pairs.append((i, j))
    return Score(len(pairs), len(det) - len(pairs), len(gt) - len(pairs), pairs)


def gt_possession_changes(events: list[MatchEvent]) -> list[tuple[float, TeamSide]]:
    out: list[tuple[float, TeamSide]] = []
    last: TeamSide | None = None
    for e in sorted(events, key=lambda e: (e.t, e.index)):
        if e.possession_team is None:
            continue
        if last is not None and e.possession_team != last:
            out.append((e.t, e.possession_team))
        last = e.possession_team
    return out


@dataclass
class Report:
    t0: float
    t1: float
    tol: float
    scores: dict[str, Score]
    mae: dict[str, tuple[float, int, str]]
    """name -> (mae, support, unit)"""
    team_accuracy: tuple[float, int] = (0.0, 0)
    """Share of outfield track-frames (matched to a GT player within 1 m) whose team is right."""

    def payload(self) -> AccuracyPanelPayload:
        rows = []
        for name, s in self.scores.items():
            rows.append(
                AccuracyRow(
                    event_type=name if name in ("pass", "shot", "sprint") else None,
                    qualifier=None if name in ("pass", "shot", "sprint") else name,
                    precision=round(s.precision, 3),
                    recall=round(s.recall, 3),
                    f1=round(s.f1, 3),
                    support=s.tp + s.fn,
                )
            )
        for name, (mae, n, unit) in self.mae.items():
            et = next((x for x in ("shot", "pass", "sprint") if name.startswith(x)), None)
            rows.append(
                AccuracyRow(event_type=et, qualifier=name, metric="mae", mae=round(mae, 2), unit=unit, support=n)
            )
        return AccuracyPanelPayload(rows=rows, window="match_so_far")


def evaluate(
    rec: Recording, detected: list[MatchEvent], runs: list[Control], tracks: Tracks, tol: float = 1.0
) -> Report:
    t = tracks.t
    t0, t1 = float(t[0]) + 1.0, float(t[-1]) - 1.0  # ignore events cut by the segment edges
    gt = [e for e in rec.events() if t0 <= e.t <= t1]
    det = [e for e in detected if t0 <= e.t <= t1]

    def key(es: list[MatchEvent], typ: str) -> list[tuple[float, TeamSide]]:
        return [(e.t, e.team) for e in es if e.type == typ and e.team is not None]

    scores = {
        "pass": match(key(det, "pass"), key(gt, "pass"), tol),
        "shot": match(key(det, "shot"), key(gt, "shot"), tol),
        "sprint": match(key(det, "sprint"), key(gt, "sprint"), tol),
        "possession_change": match(
            [c for c in possession_changes(runs, t) if t0 <= c[0] <= t1], gt_possession_changes(gt), tol
        ),
    }
    mae: dict[str, tuple[float, int, str]] = {}
    # pass length / shot speed on matched pairs
    dp, gp = [e for e in det if e.type == "pass"], [e for e in gt if e.type == "pass"]
    errs = [
        abs(q.length_m - g.length_m)
        for i, j in scores["pass"].pairs
        if (q := dp[i].qualifiers) is not None
        and q.kind == "pass"
        and (g := gp[j].qualifiers) is not None
        and g.kind == "pass"
    ]
    if errs:
        mae["pass_length"] = (float(np.mean(errs)), len(errs), "m")
    # ball and player speed per frame
    a = rec.arrays
    idx = np.array([rec.index_at(float(x) + 1e-6) for x in t])
    ball = ball_kinematics(tracks.ball, t)
    gt_ball_v = np.linalg.norm(a["ball_vel"][idx, :2], axis=1)
    ground = a["ball_pos"][idx, 2] < 0.5
    seen = ~np.isnan(tracks.ball[:, 0]) & ~np.isnan(ball.speed)
    for name, m in (("ball_speed_ground", seen & ground), ("ball_speed_all", seen)):
        if m.any():
            mae[name] = (float(np.mean(np.abs(ball.speed[m] - gt_ball_v[m]))) * 3.6, int(m.sum()), "km/h")
    pp, pv = a["player_pos"][idx], np.linalg.norm(a["player_vel"][idx], axis=2)
    on = a["player_on_pitch"][idx].astype(bool)
    perr: list[float] = []
    team_hits: list[bool] = []
    for k, xy in tracks.players.items():
        if tracks.role.get(k) == "referee":
            continue
        kn = kinematics(xy, t)
        for i in np.flatnonzero(~np.isnan(kn.speed) & ~np.isnan(xy[:, 0])):
            d = np.linalg.norm(pp[i] - kn.pos[i], axis=1)
            d[~on[i]] = np.inf
            j = int(np.argmin(d))
            if d[j] < 1.0:
                perr.append(abs(min(float(kn.speed[i]), 12.0) - float(pv[i, j])))
                gt_team = rec.player_ids[j][0]
                if tracks.role.get(k) in ("home", "away"):
                    team_hits.append(tracks.role[k][0].upper() == gt_team)
    if perr:
        mae["player_speed"] = (float(np.mean(perr)) * 3.6, len(perr), "km/h")
    ds, gs = [e for e in det if e.type == "shot"], [e for e in gt if e.type == "shot"]
    shot_det_speed = shot_speeds(tracks, ds)
    serr = [
        abs(shot_det_speed[i] - g.ball_speed_mps) * 3.6
        for i, j in scores["shot"].pairs
        if (g := gs[j].qualifiers) is not None and g.kind == "shot"
    ]
    if serr:
        mae["shot_speed"] = (float(np.mean(serr)), len(serr), "km/h")
    dsp, gsp = [e for e in det if e.type == "sprint"], [e for e in gt if e.type == "sprint"]
    sp_err = [
        abs(q.peak_speed_mps - g.peak_speed_mps) * 3.6
        for i, j in scores["sprint"].pairs
        if (q := dsp[i].qualifiers) is not None
        and q.kind == "physical"
        and (g := gsp[j].qualifiers) is not None
        and g.kind == "physical"
    ]
    if sp_err:
        mae["sprint_peak_speed"] = (float(np.mean(sp_err)), len(sp_err), "km/h")
    team_acc = (float(np.mean(team_hits)) if team_hits else 0.0, len(team_hits))
    return Report(t0, t1, tol, scores, mae, team_acc)


def shot_speeds(tracks: Tracks, shots: list[MatchEvent]) -> list[float]:
    """Peak detected ball speed (m/s) in the 0.5 s after each detected shot."""
    ball = ball_kinematics(tracks.ball, tracks.t)
    out = []
    for e in shots:
        m = (tracks.t >= e.t) & (tracks.t <= e.t + 0.5) & ~np.isnan(ball.speed)
        out.append(float(np.max(ball.speed[m])) if m.any() else 0.0)
    return out


def combine(reports: list[Report]) -> Report:
    """Pool several evaluated segments: counts add up, MAEs are support-weighted."""
    scores = {
        k: Score(
            sum(r.scores[k].tp for r in reports),
            sum(r.scores[k].fp for r in reports),
            sum(r.scores[k].fn for r in reports),
            [],
        )
        for k in reports[0].scores
    }
    mae: dict[str, tuple[float, int, str]] = {}
    for r in reports:
        for k, (m, n, unit) in r.mae.items():
            m0, n0, _ = mae.get(k, (0.0, 0, unit))
            mae[k] = ((m0 * n0 + m * n) / (n0 + n), n0 + n, unit)
    hits = sum(r.team_accuracy[0] * r.team_accuracy[1] for r in reports)
    n = sum(r.team_accuracy[1] for r in reports)
    return Report(reports[0].t0, reports[-1].t1, reports[0].tol, scores, mae, (hits / n if n else 0.0, n))


def id_switches(rec: Recording, tracks: Tracks, radius: float = 1.0) -> tuple[int, float]:
    """(ID switches, matched player-minutes): per frame, tracks and GT players are paired as mutual nearest
    neighbours within `radius`; a switch is a GT player's paired track id changing from its previous pairing."""
    t = tracks.t
    idx = np.array([rec.index_at(float(x) + 1e-6) for x in t])
    pp = rec.arrays["player_pos"][idx]
    on = rec.arrays["player_on_pitch"][idx].astype(bool)
    ids = [k for k in tracks.players if tracks.role.get(k) != "referee"]
    if not ids:
        return 0, 0.0
    xy = np.stack([tracks.players[k] for k in ids])  # (T, N, 2)
    last: dict[int, int] = {}
    switches = 0
    matched = 0
    for i in range(len(t)):
        seen = ~np.isnan(xy[:, i, 0])
        if not seen.any():
            continue
        d = np.linalg.norm(xy[seen, i][:, None] - pp[i][None], axis=2)
        d[:, ~on[i]] = np.inf
        tids = np.asarray(ids)[seen]
        near_gt = np.argmin(d, axis=1)
        near_tr = np.argmin(d, axis=0)
        for a, j in enumerate(near_gt):
            if d[a, j] < radius and near_tr[j] == a:  # mutual nearest: one track per GT player per frame
                matched += 1
                k = int(tids[a])
                if j in last and last[int(j)] != k:
                    switches += 1
                last[int(j)] = k
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.04
    return switches, matched * dt / 60.0
