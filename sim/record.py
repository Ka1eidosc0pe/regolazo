"""Columnar frame recording straight from the engine's numpy state (shared.recording.FRAME_ARRAYS), plus
snapshots for forking and the full-match driver used by the CLI and the tests."""

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from shared.model_api import AnalyticsModels, load_models
from shared.recording import FRAME_ARRAYS, write_recording
from shared.schemas.events import MatchEvent
from shared.schemas.meta import MatchMeta
from sim.constants import SIM_HZ
from sim.eventlog import to_match_events
from sim.match import Match
from sim.squads import generate_meta
from sim.world import BS_DEAD, World, world_field_names

REPO = Path(__file__).resolve().parents[1]


def data_dir() -> Path:
    return Path(os.environ.get("REGOLAZO_DATA_DIR", str(REPO / "data")))


def matches_dir() -> Path:
    return data_dir() / "matches"


class FrameRecorder:
    """Fills FRAME_ARRAYS for batch element ``b`` every tick (called through ``Match.on_frame``)."""

    def __init__(self, match: Match, b: int = 0, capacity: int | None = None, stride: int = 1) -> None:
        self.m = match
        self.b = b
        self.stride = stride
        sq = match.sq
        self.p_rec = sq.n_rec
        cap = capacity or int((2 * (match.meta.period_duration_s + 480) + 60) * SIM_HZ)
        self.arrays: dict[str, npt.NDArray[Any]] = {}
        for name, (shape, dtype) in FRAME_ARRAYS.items():
            dims = tuple(self.p_rec if d == "P" else d for d in shape)
            self.arrays[name] = np.zeros((cap, *dims), dtype=dtype)
        self.n = 0
        # substitutes sit on the bench, off the pitch
        bench = np.zeros((self.p_rec, 2))
        on = np.zeros(self.p_rec, dtype=bool)
        on[sq.rec_index] = True
        home_n = len(match.meta.home.players)
        for k in range(self.p_rec):
            if not on[k]:
                home = k < home_n
                bench[k] = ((-14.0 if home else 4.0) + 1.3 * (k % home_n), -37.0)
        self.bench = bench
        self.on = on
        self._last_touch_map = np.full(23, -1, dtype=np.int16)
        self._last_touch_map[:22] = sq.rec_index

    def __call__(self, m: Match) -> None:
        w, b = m.w, self.b
        if int(w.tick[b]) % self.stride:
            return
        if self.n >= len(self.arrays["tick"]):
            self._grow()
        k = self.n
        a = self.arrays
        ri = m.sq.rec_index
        a["tick"][k] = w.tick[b]
        a["t"][k] = w.t[b]
        a["period"][k] = w.period[b]
        a["status"][k] = w.status[b]
        a["ball_pos"][k] = w.ball_pos[b]
        a["ball_vel"][k] = w.ball_vel[b]
        a["ball_spin"][k] = w.ball_spin[b]
        a["ball_in_play"][k] = w.ball_state[b] != BS_DEAD
        a["last_touch"][k] = self._last_touch_map[int(w.last_touch[b])]
        pp = np.empty((self.p_rec, 2), dtype=np.float32)
        pp[:] = self.bench
        pp[ri] = w.pos[b]
        a["player_pos"][k] = pp
        pv = np.zeros((self.p_rec, 2), dtype=np.float32)
        pv[ri] = w.vel[b]
        a["player_vel"][k] = pv
        fz = np.zeros(self.p_rec, dtype=np.float32)
        fz[ri] = w.facing[b]
        a["player_facing"][k] = fz
        zz = np.zeros(self.p_rec, dtype=np.float32)
        zz[ri] = w.z[b]
        a["player_z"][k] = zz
        an = np.zeros(self.p_rec, dtype=np.int8)
        an[ri] = w.anim[b]
        a["player_anim"][k] = an
        at = np.zeros(self.p_rec, dtype=np.float32)
        at[ri] = w.anim_t[b]
        a["player_anim_t"][k] = at
        am = np.zeros(self.p_rec, dtype=bool)
        am[ri] = w.anim_mirror[b]
        a["player_anim_mirror"][k] = am
        av = np.zeros(self.p_rec, dtype=np.int8)
        av[ri] = w.anim_variant[b]
        a["player_anim_variant"][k] = av
        st = np.ones(self.p_rec, dtype=np.float32)
        st[ri] = w.stamina[b]
        a["player_stamina"][k] = st
        a["player_on_pitch"][k] = self.on
        a["ref_pos"][k] = w.ref_pos[b]
        a["ref_vel"][k] = w.ref_vel[b]
        a["poss_team"][k] = w.poss_team[b]
        a["poss_player"][k] = self._last_touch_map[int(w.owner[b])] if w.owner[b] >= 0 else -1
        a["possession_id"][k] = w.possession_id[b]
        a["phase"][k] = m.phase[b]
        a["restart"][k] = w.restart[b] if w.ball_state[b] == BS_DEAD else -1
        a["score"][k] = w.score[b]
        a["attacking_positive_x"][k] = 0 if m.sign[b, 0] > 0 else 1
        a["keyframe"][k] = m.keyframe[b]
        self.n += 1
        for bb, tick, i, code, mirror, variant, nf in m.backfill:
            if bb != b:
                continue
            r = int(ri[i])
            for j in range(1, nf + 1):
                kk = k - j
                if kk < 0 or int(a["tick"][kk]) != tick - j * self.stride:
                    continue
                a["player_anim"][kk, r] = code
                a["player_anim_t"][kk, r] = max(0.0, (nf - j) / SIM_HZ)
                a["player_anim_mirror"][kk, r] = mirror
                a["player_anim_variant"][kk, r] = variant

    def _grow(self) -> None:
        for name, arr in self.arrays.items():
            self.arrays[name] = np.concatenate([arr, np.zeros_like(arr)])

    def result(self) -> dict[str, npt.NDArray[Any]]:
        return {k: v[: self.n].copy() for k, v in self.arrays.items()}


@dataclass
class MatchResult:
    meta: MatchMeta
    arrays: dict[str, npt.NDArray[Any]]
    events: list[MatchEvent]
    match: Match
    wall_s: float


def simulate(
    meta: MatchMeta,
    models: AnalyticsModels | None = None,
    record: bool = True,
    snapshots: bool = True,
    until_t: float | None = None,
) -> MatchResult:
    """Play a full match (B = 1) and return frames, ground-truth events and the engine (snapshots, decisions)."""
    from sim.tier_b import derive_tier_b

    t0 = time.perf_counter()
    m = Match(meta, models or load_models(), batch=1, record=record, snapshots=snapshots)
    rec = FrameRecorder(m) if record else None
    m.on_frame = rec
    m.start()
    m.run(until_t=until_t)
    if rec is not None:
        # a final frame at full time
        rec(m)
    arrays = rec.result() if rec is not None else {}
    if arrays:
        derive_tier_b(m, arrays, m.logs[0])
    events = to_match_events(m.logs[0])
    return MatchResult(meta=meta, arrays=arrays, events=events, match=m, wall_s=time.perf_counter() - t0)


def save_snapshots(directory: Path, m: Match) -> None:
    """Engine-private fork data: the full World of every keyframe plus the decision summaries."""
    snap_dir = directory / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    if not m.snapshots:
        return
    ticks = np.array([t for t, _ in m.snapshots], dtype=np.int64)
    stacked = {name: np.stack([s[name] for _, s in m.snapshots]) for name in world_field_names()}
    np.savez_compressed(snap_dir / "world.npz", ticks=ticks, **stacked)  # type: ignore[arg-type]
    with (snap_dir / "decisions.jsonl").open("w") as fh:
        for d in m.decisions:
            fh.write(
                json.dumps(
                    {
                        "tick": d.tick,
                        "t": round(d.t, 3),
                        "carrier": m.sq.ids[d.carrier],
                        "slot": d.slot,
                        "n_viable": d.n_viable,
                        "entropy": round(d.entropy, 4),
                        "xt": round(d.xt, 5),
                        "best_dv": round(d.best_dv, 5),
                        "chosen_dv": round(d.chosen_dv, 5),
                        "intent": d.intent,
                    }
                )
                + "\n"
            )
    (snap_dir / "engine.json").write_text(
        json.dumps({"format": 1, "n_snapshots": len(ticks), "fields": world_field_names()}, indent=2) + "\n"
    )


def load_snapshot(directory: Path, tick: int) -> World:
    with np.load(directory / "snapshots" / "world.npz") as data:
        ticks = data["ticks"]
        idx = np.nonzero(ticks == tick)[0]
        if len(idx) == 0:
            raise KeyError(f"no snapshot at tick {tick}")
        i = int(idx[0])
        snap = {name: data[name][i] for name in world_field_names()}
    return World.from_snapshot(snap, 1)


def write_match(result: MatchResult, directory: Path) -> Path:
    write_recording(directory, result.meta, result.arrays, result.events, created_by="sim")
    save_snapshots(directory, result.match)
    return directory


def simulate_seed(seed: int, match_id: str | None = None, out: Path | None = None) -> MatchResult:
    meta = generate_meta(seed, match_id)
    res = simulate(meta)
    if out is not None:
        write_match(res, out)
    return res
