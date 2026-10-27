"""``RolloutEngine`` implementation: preview, options, single rollouts and batched "1,000 realities".

Forks load a stored keyframe snapshot (``<recording>/snapshots/world.npz``, written by ``sim.record``) or one
registered in memory, tile it over the batch and inject the fan's intent as the first action of every reality.
The intent goes through the same resolver, models and physics as the AI's actions; only the execution noise and
everything after it differ between realities.
"""

import math
import multiprocessing as mp
import os
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from analytics.scene import goalkeeper_ids, pitch_control_grid, scene_from_state
from shared.model_api import AnalyticsModels, load_models
from shared.recording import Recording, snapshot_ref
from shared.schemas.actions import ActionIntent, BallLaunch, ExecutionEllipse, ResolvedAction
from shared.schemas.meta import MatchMeta
from shared.schemas.models import ModelOutputs, OptionEvaluation
from shared.schemas.rollout import (
    Interval,
    OutcomeCount,
    RolloutDistribution,
    RolloutOutcome,
    RolloutRequest,
    RolloutResult,
    RolloutSample,
)
from shared.schemas.state import SimState
from sim import kick as K
from sim import planner as PL
from sim.constants import BOX_HALF_WIDTH, BOX_X, HALF_LENGTH, SIM_HZ
from sim.decision import (
    S_CARRY0,
    S_CLEAR,
    S_DRIBBLE,
    S_PASS0,
    S_SHOT,
    S_THROUGH0,
    DeciderInput,
    OptionTable,
    Slots,
    build_options,
    common,
    score_table,
)
from sim.eventlog import to_match_events
from sim.match import RESTART_CTX, SIDE, Match
from sim.record import FrameRecorder, matches_dir
from sim.resolver import (
    BODY_PARTS,
    CTX_GK_HANDS,
    CTX_OPEN,
    I_CARRY,
    I_CLEARANCE,
    I_PASS_PLAYER,
    I_PASS_SPACE,
    I_SHOT,
    INTENT_NAMES,
    PASS_HEIGHTS,
    PASS_TECHNIQUES,
    PASS_TYPES,
    SHOT_TECHNIQUES,
    available_pass_mask,
    available_shot_mask,
    effective_power,
)
from sim.tactics import attack_sign, restart_targets
from sim.world import BS_OWNED, N_ON, World, world_field_names

F = npt.NDArray[np.float64]
_Row = tuple[str, bool, float, float, tuple[float, float]]
"""(label, primary success, xG generated, xT delta, end point) of one reality."""

MAX_SAMPLES: Final = 300
ACTION_TYPE: Final = {I_PASS_PLAYER: "pass", I_PASS_SPACE: "pass", I_SHOT: "shot", I_CARRY: "carry"}
SHOT_OUTCOME_LABEL: Final = {
    "goal": "goal",
    "saved": "shot_saved",
    "saved_to_post": "shot_saved",
    "off_target": "shot_off_target",
    "wayward": "shot_off_target",
    "blocked": "shot_blocked",
    "woodwork": "woodwork",
}


@dataclass
class _MatchData:
    meta: MatchMeta
    ticks: npt.NDArray[np.int64]
    snaps: dict[str, npt.NDArray[Any]]


def parse_ref(ref: str) -> tuple[str, int]:
    parts = ref.split(":")
    if len(parts) != 3 or parts[0] != "snap":
        raise ValueError(f"bad snapshot ref {ref!r} (expected snap:<match_id>:<tick>)")
    return parts[1], int(parts[2])


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


@dataclass
class _Prepared:
    """A fan intent resolved and scored at a snapshot (D = 1 option table)."""

    match: Match
    table: OptionTable
    di: DeciderInput
    slot_cat: int
    kind: int
    receiver: int
    target: F
    ctx: int
    intent: ActionIntent | None = None
    launch: K.Launch | None = None
    """Noise-free refined launch, computed once and shared by every reality."""


class SimEngine:
    """``shared.engine_api.RolloutEngine`` backed by the simulator."""

    def __init__(
        self, data_dir: Path | None = None, models: AnalyticsModels | None = None, workers: int | None = None
    ) -> None:
        self.matches_dir = data_dir or matches_dir()
        self.models = models or load_models()
        self._cache: dict[str, _MatchData] = {}
        self.workers = workers if workers is not None else max(1, min(8, (os.cpu_count() or 2) - 2))
        """Processes used for big distributions (1 = everything in-process)."""

    # --- snapshots -----------------------------------------------------------------------------------------
    def register(self, meta: MatchMeta, snapshots: list[tuple[int, dict[str, npt.NDArray[Any]]]]) -> None:
        """Make an in-memory match forkable (e.g. one just simulated, before it is written)."""
        ticks = np.array([t for t, _ in snapshots], dtype=np.int64)
        snaps = {name: np.stack([s[name] for _, s in snapshots]) for name in world_field_names()}
        self._cache[meta.match_id] = _MatchData(meta, ticks, snaps)

    def _data(self, match_id: str) -> _MatchData:
        if match_id not in self._cache:
            d = self.matches_dir / match_id
            meta = MatchMeta.model_validate_json((d / "meta.json").read_text())
            with np.load(d / "snapshots" / "world.npz") as z:
                ticks = z["ticks"]
                snaps = {name: z[name] for name in world_field_names()}
            self._cache[match_id] = _MatchData(meta, ticks, snaps)
        return self._cache[match_id]

    def snapshot_ticks(self, match_id: str) -> npt.NDArray[np.int64]:
        return self._data(match_id).ticks

    def fork(self, ref: str, n: int = 1, seed: int = 0) -> Match:
        """A Match with ``n`` identical copies of the world at ``ref``."""
        match_id, tick = parse_ref(ref)
        data = self._data(match_id)
        idx = np.nonzero(data.ticks == tick)[0]
        if len(idx) == 0:
            raise KeyError(f"no snapshot at {ref}")
        i = int(idx[-1])
        snap = {name: data.snaps[name][i] for name in world_field_names()}
        m = Match(data.meta, self.models, batch=n, seed=seed)
        m.w = World.from_snapshot(snap, n)
        w = m.w
        for name in ("last_pass_ev", "shot_ev", "clear_ev", "receipt_ev", "flight_event"):
            getattr(w, name)[:] = -1
        # a reality's carries start at the fork, not before it
        if w.owner[0] >= 0:
            w.carry_start[:] = w.pos[:, int(w.owner[0])]
        w.carry_start_t[:] = w.t
        w.carry_path[:] = 0.0
        # likewise off-ball runs in progress are measured from the fork
        w.run_start_t[:] = np.maximum(w.run_start_t, w.t[:, None])
        w.run_start[:] = w.pos
        m.sign = attack_sign(w, m.home_first)
        m.done[:] = False
        m.phase[:] = 0
        m._update_phase()
        if w.ball_state[0] != BS_OWNED and w.restart[0] >= 0:
            rc = restart_targets(w, m.sq, m.sign, 0, m.styles)
            m.restart_cache = [None if rc is None else rc.copy() for _ in range(n)]
        return m

    # --- intents -------------------------------------------------------------------------------------------
    def _prepare(self, ref: str, intent: ActionIntent, n: int = 1, seed: int = 0) -> _Prepared:
        m = self.fork(ref, n, seed)
        w, sq = m.w, m.sq
        if w.ball_state[0] != BS_OWNED or w.owner[0] < 0:
            raise ValueError(f"nobody controls the ball at {ref}")
        carrier = int(w.owner[0])
        if sq.ids[carrier] != intent.player:
            raise ValueError(f"{intent.player} is not the ball carrier at {ref} ({sq.ids[carrier]} is)")
        ctx = CTX_GK_HANDS if w.hands[0] else CTX_OPEN
        if abs(w.last_restart_t[0] - w.t[0]) < 1e-6 and w.last_restart_type[0] >= 0:
            ctx = RESTART_CTX[int(w.last_restart_type[0])]
        di = m.decider_input([0], [ctx])
        cm = common(sq, di)
        kind = INTENT_NAMES.index(intent.kind)
        team = int(sq.team[carrier])
        s = float(m.sign[0, team])
        cpos = w.pos[0, carrier]
        receiver = -1
        target = np.zeros(3)
        cat = S_PASS0
        if kind == I_PASS_PLAYER:
            assert intent.target_player is not None
            if intent.target_player not in sq.ids:
                raise ValueError(f"unknown player {intent.target_player}")
            receiver = sq.ids.index(intent.target_player)
            if sq.team[receiver] != team or receiver == carrier:
                raise ValueError("pass_player needs a teammate of the ball carrier")
            mp, mv = w.pos[0, receiver], w.vel[0, receiver]
            first = mp + 0.6 * mv
            t_fl = float(np.clip(np.linalg.norm(first - cpos) / 11.0, 0.3, 2.0))
            target[:2] = mp + t_fl * mv
        elif kind == I_PASS_SPACE:
            assert intent.target is not None
            target[:2] = intent.target[:2]
            mates = np.array([i for i in range(team * N_ON, (team + 1) * N_ON) if i != carrier])
            d = np.linalg.norm(w.pos[0, mates] - target[:2], axis=1)
            receiver = int(mates[int(np.argmin(d))])
            cat = S_THROUGH0
        elif kind == I_SHOT:
            assert intent.target is not None
            target[:] = (HALF_LENGTH * s, intent.target[1], intent.target[2])
            cat = S_SHOT
        elif kind == I_CARRY:
            end = intent.path[-1] if intent.path else (intent.target[0], intent.target[1])  # type: ignore[index]
            target[:2] = end
            to_opp = (cm.opp_pos[0] - cpos) @ (
                (target[:2] - cpos) / max(float(np.linalg.norm(target[:2] - cpos)), 1e-6)
            )
            close = np.linalg.norm(cm.opp_pos[0] - cpos, axis=1) < 3.2
            cat = S_DRIBBLE if bool((close & (to_opp > 0.3)).any()) else S_CARRY0
        else:
            if intent.target is not None:
                target[:2] = intent.target[:2]
            else:
                target[:2] = (cpos[0] + 40.0 * s, np.clip(cpos[1] * 1.6, -30, 30))
            cat = S_CLEAR
        lines = di.offside_line[0]
        rec_off = receiver >= 0 and kind in (I_PASS_PLAYER, I_PASS_SPACE)
        rec_off = rec_off and (w.pos[0, receiver, 0] * s > lines + 0.05) and (w.pos[0, receiver, 0] * s > 0)
        sl = Slots(
            kind=np.array([[kind]]),
            receiver=np.array([[receiver]]),
            target=target[None, None, :].copy(),
            power=np.array([[intent.power]]),
            curve=np.array([[intent.curve]]),
            valid=np.array([[True]]),
            receiver_offside=np.array([[bool(rec_off)]]),
            cat=np.array([[cat]]),
        )
        pov = (
            None
            if intent.pass_technique_override is None
            else np.array([PASS_TECHNIQUES.index(intent.pass_technique_override)])
        )
        sov = (
            None
            if intent.shot_technique_override is None
            else np.array([SHOT_TECHNIQUES.index(intent.shot_technique_override)])
        )
        table = score_table(sq, self.models, m.styles, di, cm, sl, None, pov, sov)
        return _Prepared(m, table, di, cat, kind, receiver, target, ctx, intent=intent)

    def _resolved(self, prep: _Prepared, intent: ActionIntent) -> ResolvedAction:
        m, res = prep.match, prep.table.res
        w, sq = m.w, m.sq
        c = int(w.owner[0])
        kind = prep.kind
        kick = int(res.kick[0])
        tgt = res.target[0]
        mult = float(res.noise_mult[0])
        quality = float(np.clip((1.55 - mult) / 1.1, 0.0, 1.0))
        cpos = w.pos[0, c]
        dist = float(np.linalg.norm(tgt[:2] - cpos))
        mc = float(res.max_curve[0])
        avail_p = available_pass_mask(sq, np.array([c]), np.array([dist]), np.array([mc]))[0]
        team = int(sq.team[c])
        s = float(m.sign[0, team])
        gk = int(m._gk[1 - team])
        gk_off = HALF_LENGTH - w.pos[0, gk, 0] * s
        sdist = float(np.hypot(HALF_LENGTH - cpos[0] * s, cpos[1] * s))
        avail_s = available_shot_mask(
            sq, np.array([c]), np.array([sdist]), np.array([w.ball_pos[0, 2]]), np.array([False]), np.array([gk_off])
        )[0]
        launch = None
        arc: list[tuple[float, float, float]] = []
        ellipse = None
        if kind == I_CARRY:
            q = sq.attr["dribbling"][c] / 99.0
            ang = math.atan2(tgt[1] - cpos[1], tgt[0] - cpos[0])
            ellipse = ExecutionEllipse(
                plane="pitch",
                center=(float(tgt[0]), float(tgt[1]), 0.0),
                semi_major_m=float(1.6 - q),
                semi_minor_m=float(0.9 - 0.5 * q),
                angle_rad=ang,
            )
        else:
            origin = w.ball_pos[0:1].copy()
            pw = effective_power(sq, np.array([c]), np.array([kick]), np.array([intent.power]))
            plan = K.plan_kicks(origin, res.target[0:1], np.array([kick]), pw, res.curve_applied[0:1])
            if kick != K.K_GROUND or abs(float(res.curve_applied[0])) > 0.05:
                plan = K.refine(plan)
            la = plan.launch
            prep.launch = la
            launch = BallLaunch(
                speed_mps=float(np.clip(la.speed[0], 0, 45)),
                elevation_rad=float(np.clip(la.elev[0], -0.5, 1.5)),
                azimuth_rad=float(la.azimuth[0]),
                spin=(float(la.spin[0, 0]), float(la.spin[0, 1]), float(la.spin[0, 2])),
            )
            vel = la.velocity()
            if kick == K.K_GROUND and abs(float(res.curve_applied[0])) < 0.05:
                vel[:, 2] = 0.0
            paths = PL.predict_paths(origin, vel, la.spin, 100)
            xs = paths.xs[:, 0]
            arc = _cut_arc(xs, origin[0], tgt, kind == I_SHOT, s)
            sig_az = float(K.profile_array("sig_az", np.array([kick]))[0]) * mult
            sig_el = float(K.profile_array("sig_el", np.array([kick]))[0]) * mult
            sig_v = float(K.profile_array("sig_v", np.array([kick]))[0]) * mult
            if kind == I_SHOT:
                sy, sz = sdist * sig_az, sdist * sig_el + 0.08
                ellipse = ExecutionEllipse(
                    plane="goal_mouth",
                    center=(float(tgt[0]), float(tgt[1]), float(tgt[2])),
                    semi_major_m=float(max(sy, sz)),
                    semi_minor_m=float(min(sy, sz)),
                    angle_rad=0.0 if sy >= sz else math.pi / 2,
                )
            else:
                lat, lon = dist * sig_az, dist * 2.0 * sig_v + 0.2
                ang = float(la.azimuth[0])
                ellipse = ExecutionEllipse(
                    plane="pitch",
                    center=(float(tgt[0]), float(tgt[1]), 0.0),
                    semi_major_m=float(max(lat, lon)),
                    semi_minor_m=float(min(lat, lon)),
                    angle_rad=ang if lon >= lat else ang + math.pi / 2,
                )
        action_type = ACTION_TYPE.get(kind, "clearance")
        if kind == I_CARRY and prep.slot_cat == S_DRIBBLE:
            action_type = "dribble"
        is_pass = kind in (I_PASS_PLAYER, I_PASS_SPACE)
        return ResolvedAction(
            intent=intent,
            action_type=action_type,
            body_part=None if kind == I_CARRY else BODY_PARTS[int(res.body[0])],
            pass_technique=PASS_TECHNIQUES[int(res.pass_tech[0])] if is_pass else None,
            pass_type=PASS_TYPES[int(res.pass_type[0])] if is_pass else None,
            pass_height=PASS_HEIGHTS[int(res.pass_height[0])] if is_pass else None,
            shot_technique=SHOT_TECHNIQUES[int(res.shot_tech[0])] if kind == I_SHOT else None,
            intended_end=(float(tgt[0]), float(tgt[1]), float(max(tgt[2], 0.0))),
            launch=launch,
            curve_applied=float(np.clip(res.curve_applied[0], -1, 1)),
            ellipse=ellipse,
            preview_arc=arc,
            override_rejected=bool(res.override_rejected[0]),
            weak_foot=bool(res.weak_foot[0]),
            execution_quality=quality,
            available_pass_techniques=[PASS_TECHNIQUES[i] for i in np.nonzero(avail_p)[0]],
            available_shot_techniques=[SHOT_TECHNIQUES[i] for i in np.nonzero(avail_s)[0]],
            max_curve=float(np.clip(mc, 0, 1)),
        )

    def _evaluation(
        self, m: Match, table: OptionTable, k: int, slot: int, intent: ActionIntent, resolved: ResolvedAction | None
    ) -> OptionEvaluation:
        sq = m.sq
        kind = int(table.kind[k, slot])
        rec = int(table.receiver[k, slot])
        ptype = PASS_TYPES[int(table.res.pass_type[k * table.kind.shape[1] + slot])]
        c = int(m.w.owner[0])
        team = int(sq.team[c])
        s = float(m.sign[0, team])
        bx, by = m.w.ball_pos[0, 0] * s, m.w.ball_pos[0, 1] * s
        xt_here = float(np.asarray(self.models.xt(np.array([bx]), np.array([by])))[0])
        name = sq.players[rec].display_name if rec >= 0 else ""
        if kind == I_PASS_PLAYER:
            key = "option.cross" if ptype == "cross" else ("option.cutback" if ptype == "cutback" else "option.pass")
            label = f"{'Cross' if ptype == 'cross' else 'Pass'} to {name}"
        elif kind == I_PASS_SPACE:
            key, label = "option.through_ball", f"Through ball for {name}"
        elif kind == I_SHOT:
            key, label = "option.shot", "Shot"
        elif kind == I_CARRY:
            dribble = resolved is not None and resolved.action_type == "dribble"
            key, label = ("option.dribble", "Take on") if dribble else ("option.carry", "Carry")
        else:
            key, label = "option.clearance", "Clearance"
        p = float(np.clip(table.p[k, slot], 0, 1))
        vs = float(table.value_success[k, slot])
        return OptionEvaluation(
            option_id=f"opt_{slot}",
            intent=intent,
            resolved=resolved,
            label=label,
            label_key=key,
            label_args={"player": sq.ids[rec]} if rec >= 0 and kind in (I_PASS_PLAYER, I_PASS_SPACE) else {},
            p_success=p,
            p_success_baseline=float(np.clip(table.p_base[k, slot], 0, 1)),
            value_if_success=vs,
            xg=float(np.clip(table.p0[k, slot], 0, 1)) if kind == I_SHOT else None,
            xt_gain=vs - xt_here if kind != I_SHOT else None,
            decision_value=float(table.dv[k, slot]),
            viable=bool(table.valid[k, slot] and p >= 0.5),
        )

    # --- RolloutEngine -------------------------------------------------------------------------------------
    def preview(self, snapshot_ref: str, intent: ActionIntent) -> ResolvedAction:
        prep = self._prepare(snapshot_ref, intent)
        return self._resolved(prep, intent)

    def options(self, snapshot_ref: str) -> ModelOutputs:
        m = self.fork(snapshot_ref, 1, 0)
        w, sq = m.w, m.sq
        state = state_of(m, 0)
        team_idx = int(w.poss_team[0]) if w.poss_team[0] >= 0 else int(max(w.last_touch_team[0], 0))
        scene = scene_from_state(state, SIDE[team_idx], goalkeeper_ids(m.meta))
        grid = pitch_control_grid(self.models, scene)
        s = float(m.sign[0, team_idx])
        bx, by = float(w.ball_pos[0, 0]), float(w.ball_pos[0, 1])
        xt_ball = float(np.asarray(self.models.xt(np.array([bx * s]), np.array([by * s])))[0])
        out = ModelOutputs(
            tick=int(w.tick[0]),
            t=float(w.t[0]),
            team_in_possession=SIDE[int(w.poss_team[0])] if w.poss_team[0] >= 0 else None,
            ball_carrier=sq.ids[int(w.owner[0])] if w.owner[0] >= 0 else None,
            ball_xy=(bx, by),
            xt_at_ball=xt_ball,
            pitch_control=grid,
            model_versions=dict(self.models.versions),
        )
        if w.ball_state[0] != BS_OWNED or w.owner[0] < 0:
            return out
        ctx = CTX_GK_HANDS if w.hands[0] else CTX_OPEN
        di = m.decider_input([0], [ctx])
        table = build_options(sq, self.models, m.styles, di, None)
        opts: list[OptionEvaluation] = []
        carrier = sq.ids[int(w.owner[0])]
        for slot in np.nonzero(table.valid[0])[0]:
            slot = int(slot)
            intent = _intent_from_table(m, table, 0, slot, carrier)
            if intent is None:
                continue
            opts.append(self._evaluation(m, table, 0, slot, intent, None))
        shot_xg = float(table.p0[0, S_SHOT]) if table.valid[0, S_SHOT] else None
        return out.model_copy(update={"options": opts, "shot_xg_now": shot_xg})

    def rollout(self, request: RolloutRequest) -> RolloutResult:
        seed = request.seed if request.seed is not None else int(np.random.default_rng().integers(2**31))
        rid = uuid.uuid4().hex[:10]
        prep = self._prepare(request.snapshot_ref, request.intent, 1, seed)
        resolved = self._resolved(prep, request.intent)
        evaluation = self._evaluation(prep.match, prep.table, 0, 0, request.intent, resolved)
        m = prep.match
        rec = FrameRecorder(m, capacity=int(request.horizon_s * SIM_HZ) + 5, stride=request.frame_stride)
        if request.return_frames:
            m.on_frame = rec
        start = _start_value(self.models, m)
        self._inject(prep)
        m.run(max_ticks=round(request.horizon_s * SIM_HZ))
        events = to_match_events(m.logs[0], rid)
        outcome = _outcome(self.models, m, 0, prep, start, events)
        frames: list[SimState] = []
        if request.return_frames and rec.n:
            frames = _states(m, rec)
        return RolloutResult(
            id=rid,
            snapshot_ref=request.snapshot_ref,
            seed=seed,
            resolved=resolved,
            evaluation=evaluation,
            frames=frames,
            events=events,
            outcome=outcome,
        )

    def distribution(self, request: RolloutRequest) -> RolloutDistribution:
        """``n`` realities, vectorized in-process per chunk; large requests are split over a process pool."""
        t0 = time.perf_counter()
        seed = request.seed if request.seed is not None else int(np.random.default_rng().integers(2**31))
        n = request.n
        prep = self._prepare(request.snapshot_ref, request.intent, 1, seed)
        resolved = self._resolved(prep, request.intent)
        chunks = _chunks(n, self.workers)
        if len(chunks) == 1:
            rows = self._simulate_chunk(request, n, seed, 0)
        else:
            match_id, tick = parse_ref(request.snapshot_ref)
            data = self._data(match_id)
            i = int(np.nonzero(data.ticks == tick)[0][-1])
            snap = {name: data.snaps[name][i] for name in world_field_names()}
            payload = (data.meta.model_dump_json(), tick, snap, request.model_dump_json())
            jobs = [(payload, size, seed, k) for k, size in enumerate(chunks)]
            rows = [row for part in self._executor().map(_worker_chunk, jobs) for row in part]
        labels = [r[0] for r in rows]
        counts = [OutcomeCount(label=lab, count=labels.count(lab)) for lab in dict.fromkeys(labels)]
        k_succ = sum(1 for r in rows if r[1])
        k_goal = labels.count("goal")
        lo, hi = wilson(k_succ, n)
        glo, ghi = wilson(k_goal, n)
        samples = [RolloutSample(label=r[0], primary_success=r[1], end_xy=r[4], xg=r[2]) for r in rows[:MAX_SAMPLES]]
        return RolloutDistribution(
            snapshot_ref=request.snapshot_ref,
            intent=request.intent,
            resolved=resolved,
            n=n,
            seed=seed,
            outcome_counts=counts,
            p_primary_success=k_succ / n,
            p_primary_success_ci95=Interval(lo=lo, hi=hi),
            p_goal=k_goal / n,
            p_goal_ci95=Interval(lo=glo, hi=ghi),
            mean_xg=float(np.mean([r[2] for r in rows])),
            mean_xt_delta=float(np.mean([r[3] for r in rows])),
            model_p_success=float(np.clip(prep.table.p[0, 0], 0, 1)),
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            samples=samples,
        )

    def _simulate_chunk(self, request: RolloutRequest, n: int, seed: int, k: int) -> list[_Row]:
        """``n`` realities in one vectorized batch; chunk ``k`` gets its own deterministic stream."""
        prep = self._prepare(request.snapshot_ref, request.intent, n, seed * 1000 + k)
        self._resolved(prep, request.intent)
        m = prep.match
        start = _start_value(self.models, m)
        self._inject(prep)
        m.run(max_ticks=round(request.horizon_s * SIM_HZ))
        out: list[_Row] = []
        for b in range(n):
            o = _outcome(self.models, m, b, prep, start, None)
            out.append((o.label, o.primary_success, o.xg_generated, o.xt_delta, _end_xy(m, b, prep)))
        return out

    def _executor(self) -> ProcessPoolExecutor:
        global _EXECUTOR
        if _EXECUTOR is None:
            _EXECUTOR = ProcessPoolExecutor(max_workers=self.workers, mp_context=mp.get_context("spawn"))
        return _EXECUTOR

    def warm_up(self) -> None:
        """Start the rollout worker pool (each worker loads the models once) before the first big request."""
        if self.workers > 1:
            list(self._executor().map(_worker_ping, range(self.workers)))

    def _inject(self, prep: _Prepared) -> None:
        """Every reality executes the same resolved intent (its own noise) as its first action."""
        m, table = prep.match, prep.table
        restart = None
        if prep.ctx not in (CTX_OPEN, CTX_GK_HANDS):
            restart = (-1, int(m.w.last_restart_type[0]))
        info: dict[str, Any] = {
            "ctx": prep.ctx,
            "situation": int(prep.di.situation[0]),
            "restart": restart,
            "pressure": float(table.pressure[0]),
            "receiver_offside": bool(table.receiver_offside[0, 0]),
            "slot": prep.slot_cat,
            "power": float(table.power[0, 0]),
            "forced": True,
        }
        if prep.launch is None and prep.kind != I_CARRY:
            assert prep.intent is not None
            self._resolved(prep, prep.intent)
        p, p0 = float(table.p[0, 0]), float(table.p0[0, 0])

        def act(mm: Match, b: int) -> None:
            mm.keyframe[b] = True
            inf = dict(info)
            if prep.launch is not None:
                la = prep.launch
                inf["plan"] = K.Launch(la.speed.copy(), la.elev.copy(), la.azimuth.copy(), la.spin.copy())
            mm.execute(b, prep.kind, table.res, 0, table.target[0, 0], prep.receiver, p, p0, inf)
            mm.w.act_timer[b] = max(float(mm.w.act_timer[b]), 0.0) if prep.kind == I_CARRY else mm.w.act_timer[b]

        for b in range(m.w.b):
            m.forced[b] = act


_EXECUTOR: ProcessPoolExecutor | None = None
_WORKER_ENGINE: "SimEngine | None" = None
CHUNK: Final = 125
"""Realities per vectorized batch: the batched engine's sweet spot on the M3 Pro."""


def _chunks(n: int, workers: int) -> list[int]:
    if workers <= 1 or n <= 1.5 * CHUNK:
        return [n]
    k = min(workers, math.ceil(n / CHUNK))
    base, extra = divmod(n, k)
    return [base + (1 if i < extra else 0) for i in range(k)]


def _worker_ping(_: int) -> int:
    global _WORKER_ENGINE
    if _WORKER_ENGINE is None:
        _WORKER_ENGINE = SimEngine(workers=1)
    K.flight_tables()
    return os.getpid()


def _worker_chunk(job: tuple[tuple[str, int, dict[str, npt.NDArray[Any]], str], int, int, int]) -> list[_Row]:
    (meta_json, tick, snap, req_json), n, seed, k = job
    global _WORKER_ENGINE
    if _WORKER_ENGINE is None:
        _WORKER_ENGINE = SimEngine(workers=1)
    eng = _WORKER_ENGINE
    meta = MatchMeta.model_validate_json(meta_json)
    eng.register(meta, [(tick, snap)])
    return eng._simulate_chunk(RolloutRequest.model_validate_json(req_json), n, seed, k)


def _cut_arc(xs: F, origin: F, tgt: F, shot: bool, s: float) -> list[tuple[float, float, float]]:
    """Noise-free ball path for the preview, until it reaches the target (or the goal plane for shots)."""
    if shot:
        past = xs[:, 0] * s >= HALF_LENGTH
        end = int(np.argmax(past)) + 1 if past.any() else len(xs)
    else:
        chord = tgt[:2] - origin[:2]
        dist = float(np.linalg.norm(chord))
        u = chord / max(dist, 1e-6)
        along = (xs[:, :2] - origin[:2]) @ u
        past = along >= dist
        end = int(np.argmax(past)) + 1 if past.any() else len(xs)
        moving = np.linalg.norm(np.diff(xs[:, :2], axis=0), axis=1) > 1e-3
        if not moving.all():
            end = min(end, int(np.argmin(moving)) + 1)
    pts = xs[: max(end, 2)]
    step = max(1, len(pts) // 40)
    sel = list(range(0, len(pts), step))
    if sel[-1] != len(pts) - 1:
        sel.append(len(pts) - 1)
    return [(round(float(p[0]), 3), round(float(p[1]), 3), round(float(max(p[2], 0.0)), 3)) for p in pts[sel]]


def _intent_from_table(m: Match, table: OptionTable, k: int, slot: int, carrier: str) -> ActionIntent | None:
    kind = int(table.kind[k, slot])
    tgt = table.target[k, slot]
    rec = int(table.receiver[k, slot])
    power = float(np.clip(table.power[k, slot], 0, 1))
    if kind == I_PASS_PLAYER:
        return ActionIntent(kind="pass_player", player=carrier, target_player=m.sq.ids[rec], power=power, source="ai")
    if kind == I_CLEARANCE:
        return ActionIntent(kind="clearance", player=carrier, target=(float(tgt[0]), float(tgt[1]), 0.0), source="ai")
    name = INTENT_NAMES[kind]
    z = float(tgt[2]) if kind == I_SHOT else 0.0
    return ActionIntent(kind=name, player=carrier, target=(float(tgt[0]), float(tgt[1]), z), power=power, source="ai")


def state_of(m: Match, b: int) -> SimState:
    rec = FrameRecorder(m, b=b, capacity=1)
    rec(m)
    arrays = rec.result()
    arrays["keyframe"][:] = False
    ids = [p.id for p in m.meta.home.players] + [p.id for p in m.meta.away.players]
    r = Recording(Path("."), m.meta, {"n_frames": 1, "player_ids": ids}, arrays)
    return r.state_at(0)


def _states(m: Match, rec: FrameRecorder) -> list[SimState]:
    arrays = rec.result()
    arrays["keyframe"][:] = False
    ids = [p.id for p in m.meta.home.players] + [p.id for p in m.meta.away.players]
    r = Recording(Path("."), m.meta, {"n_frames": rec.n, "player_ids": ids}, arrays)
    return [r.state_at(i) for i in range(rec.n)]


def _start_value(models: AnalyticsModels, m: Match) -> float:
    w = m.w
    team = int(m.sq.team[int(w.owner[0])])
    s = float(m.sign[0, team])
    return float(np.asarray(models.xt(np.array([w.ball_pos[0, 0] * s]), np.array([w.ball_pos[0, 1] * s])))[0])


def _end_value(models: AnalyticsModels, m: Match, b: int, team: int) -> float:
    w = m.w
    holder = int(w.poss_team[b])
    if holder < 0:
        holder = int(w.last_touch_team[b])
    s = float(m.sign[b, holder]) if holder >= 0 else 1.0
    v = float(np.asarray(models.xt(np.array([w.ball_pos[b, 0] * s]), np.array([w.ball_pos[b, 1] * s])))[0])
    return v if holder == team else -v


def _outcome(models: AnalyticsModels, m: Match, b: int, prep: _Prepared, start: float, events: Any) -> RolloutOutcome:
    """Classify one reality from its ground-truth event records (taxonomy-level, not frames)."""
    sq = m.sq
    team = int(sq.team[int(prep.di.carrier[0])])
    side = SIDE[team]
    recs = m.logs[b].records
    order = sorted(range(len(recs)), key=lambda i: (recs[i]["t"], i))
    label: str | None = None
    end_idx: int | None = None
    end_t = float(m.w.t[b])
    xg = 0.0
    chance = False
    primary = None
    for i in order:
        r = recs[i]
        if (
            primary is None
            and r["player"] == _carrier_id(prep)
            and r["type"] in ("pass", "shot", "clearance", "take_on")
        ):
            primary = r
        typ, rteam, q = r["type"], r["team"], r["q"]
        if typ == "shot" and rteam == side:
            xg += float(q.get("xg") or 0.0)
        if label is not None:
            continue
        if typ == "shot" and rteam == side:
            label = SHOT_OUTCOME_LABEL.get(q.get("outcome", ""), "shot_off_target")
        elif typ == "own_goal" and rteam != side:
            label = "goal"
        elif typ == "foul_won" and rteam == side:
            label = "foul_won"
        elif typ == "offside" and rteam == side:
            label = "offside"
        elif typ == "ball_out_of_play":
            nxt = q.get("result_restart_type")
            awarded_to_us = (q.get("last_touch_team") != side) if nxt in ("throw_in", "corner") else False
            if nxt == "goal_kick":
                awarded_to_us = q.get("last_touch_team") != side
            if not awarded_to_us:
                label = "out_of_play"
        elif typ in ("interception", "ball_recovery", "duel", "gk_action") and rteam != side:
            if typ == "duel" and (q.get("outcome") != "won" or q.get("duel_type") == "aerial"):
                continue
            if typ == "duel" and q.get("possession_after") in ("loose", "out_of_play", "opponent"):
                continue
            if typ == "interception" and q.get("outcome") == "lost":
                continue
            if typ == "gk_action" and q.get("action") not in ("save", "claim", "sweeper", "rush_out"):
                continue
            label = "turnover"
        elif typ in ("ball_receipt",) and rteam == side and r["location"] is not None:
            s = float(m.sign[b, team])
            x_t, y_t = r["location"][0] * s, r["location"][1] * s
            if x_t >= BOX_X and abs(y_t) <= BOX_HALF_WIDTH:
                chance = True
        if label is not None:
            end_idx = i
            end_t = float(r["t"])
    if label is None:
        label = "chance_created" if chance or xg > 0 else "possession_retained"
    kind = prep.kind
    if kind in (I_PASS_PLAYER, I_PASS_SPACE):
        success = primary is not None and primary["type"] == "pass" and primary["q"].get("outcome") == "complete"
    elif kind == I_SHOT:
        success = primary is not None and primary["type"] == "shot" and primary["q"].get("outcome") == "goal"
    elif kind == I_CARRY:
        if primary is not None and primary["type"] == "take_on":
            success = primary["q"].get("outcome") == "success"
        else:
            success = bool(m.forced_carry_kept[b])
    else:
        success = label not in ("turnover",) or end_t > _fork_t(prep) + 3.0
    xt_delta = _end_value(models, m, b, team) - start
    ended = None
    if events is not None and end_idx is not None:
        new_index = {old: k for k, old in enumerate(order)}
        ended = events[new_index[end_idx]].id
    return RolloutOutcome(
        label=label,
        primary_success=bool(success),
        terminal_t=end_t,
        xg_generated=xg,
        xt_delta=xt_delta,
        ended_by_event_id=ended,
    )


def _carrier_id(prep: _Prepared) -> str:
    return prep.match.sq.ids[int(prep.di.carrier[0])]


def _fork_t(prep: _Prepared) -> float:
    return float(prep.di.t[0])


def _end_xy(m: Match, b: int, prep: _Prepared) -> tuple[float, float]:
    cid = _carrier_id(prep)
    for r in m.logs[b].records:
        if r["player"] == cid and r["type"] in ("pass", "shot", "clearance"):
            end = r.get("end_location")
            if r["type"] == "shot" and end is not None:
                return (float(end[1]), float(end[2]))
            if end is not None:
                return (float(end[0]), float(end[1]))
            break
    if prep.kind == I_CARRY:
        c = prep.match.sq.ids.index(cid)
        return (float(m.w.pos[b, c, 0]), float(m.w.pos[b, c, 1]))
    return (float(m.w.ball_pos[b, 0]), float(m.w.ball_pos[b, 1]))


__all__ = ["SimEngine", "parse_ref", "snapshot_ref", "state_of", "wilson"]
