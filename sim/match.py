"""The match engine: a batched, seeded, deterministic 25 Hz football simulation.

``Match`` advances B independent realities at once (B = 1 for a full match, B = 1,000 for "1,000 realities").
Continuous work (team shape, steering, ball physics, animation) is vectorized over the batch; discrete football
events (a touch, a tackle, the ball going out) are handled per element when they happen, and all decisions taken
in the same tick are scored together by the decision model.
"""

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from shared.model_api import AnalyticsModels, PostShotFeatures, load_models
from shared.recording import ANIMS, PHASES, STATUSES
from shared.schemas.meta import MatchMeta
from sim import kick as K
from sim import planner as PL
from sim.constants import (
    BALL_RADIUS,
    BOX_HALF_WIDTH,
    BOX_X,
    CONTROL_RADIUS,
    COUNTER_ADVANCE_M,
    COUNTER_DIRECTNESS,
    COUNTER_WINDOW_S,
    COUNTERPRESS_WINDOW_S,
    DT,
    FINAL_THIRD_X,
    FIRST_TIME_S,
    GOAL_HEIGHT,
    HALF_GOAL,
    HALF_LENGTH,
    HALF_WIDTH,
    KEY_PASS_WINDOW_S,
    PASS_GROUND_MAX_Z,
    PASS_LOW_MAX_Z,
    PASS_MEDIUM_MAX,
    PASS_SHORT_MAX,
    PENALTY_SPOT_X,
    SET_PIECE_WINDOW_S,
    SIM_HZ,
    TRANSITION_WINDOW_S,
)
from sim.decision import (
    S_DRIBBLE,
    S_HOLD,
    DeciderInput,
    OptionTable,
    build_options,
    choose,
    option_entropy,
    situation_code,
)
from sim.evaluate import SIT_COUNTER, SIT_DIRECT_FK, SIT_OPEN, SIT_PENALTY, SIT_SET_PIECE, sigmoid
from sim.eventlog import EventLog
from sim.kinematics import drain_stamina, effective_max_speed, jump_speed, steer, step_jump, turn_toward
from sim.physics import step_ball
from sim.resolver import (
    BODY_PARTS,
    CTX_CORNER,
    CTX_FREE_KICK,
    CTX_GK_HANDS,
    CTX_GOAL_KICK,
    CTX_KICK_OFF,
    CTX_OPEN,
    CTX_PENALTY,
    CTX_THROW_IN,
    I_CARRY,
    I_CLEARANCE,
    I_PASS_PLAYER,
    I_SHOT,
    PASS_TECHNIQUES,
    PASS_TYPES,
    SHOT_TECHNIQUES,
    Resolved,
    effective_power,
)
from sim.tactics import (
    attack_sign,
    box_marking,
    goal_side,
    lane_blocker,
    last_defender_x,
    offside_lines,
    restart_targets,
    shape_targets,
    styles_for,
)
from sim.world import (
    BS_DEAD,
    BS_FLIGHT,
    BS_OWNED,
    FM_BLOCK,
    FM_CLAIM,
    FM_FREE,
    FM_INTERCEPT,
    FM_LOOSE,
    FM_MISCONTROL,
    FM_NONE,
    FM_RECEIVE,
    FM_SAVE,
    N_ON,
    R_CORNER,
    R_FK_DIRECT,
    R_FK_INDIRECT,
    R_GOAL_KICK,
    R_KICK_OFF,
    R_PENALTY,
    R_THROW_IN,
    P,
    Squad,
    World,
)

F = npt.NDArray[np.float64]
IA = npt.NDArray[np.int64]
BA = npt.NDArray[np.bool_]

A: Final = {name: i for i, name in enumerate(ANIMS)}
PH: Final = {name: i for i, name in enumerate(PHASES)}
ST: Final = {name: i for i, name in enumerate(STATUSES)}
SIDE: Final = ("home", "away")
RESTART_NAMES: Final = (
    "kick_off",
    "throw_in",
    "goal_kick",
    "corner",
    "free_kick_direct",
    "free_kick_indirect",
    "penalty",
    "drop_ball",
)
PP: Final = {
    name: i
    for i, name in enumerate(
        (
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
    )
}
PP_NAMES: Final = tuple(PP)
RESTART_PATTERN: Final = {
    R_KICK_OFF: "from_kick_off",
    R_THROW_IN: "from_throw_in",
    R_GOAL_KICK: "from_goal_kick",
    R_CORNER: "from_corner",
    R_FK_DIRECT: "from_free_kick",
    R_FK_INDIRECT: "from_free_kick",
    R_PENALTY: "other",
    7: "other",
}
RESTART_CTX: Final = {
    R_KICK_OFF: CTX_KICK_OFF,
    R_THROW_IN: CTX_THROW_IN,
    R_GOAL_KICK: CTX_GOAL_KICK,
    R_CORNER: CTX_CORNER,
    R_FK_DIRECT: CTX_FREE_KICK,
    R_FK_INDIRECT: CTX_FREE_KICK,
    R_PENALTY: CTX_PENALTY,
    7: CTX_OPEN,
}
FLIGHT_PASS, FLIGHT_SHOT, FLIGHT_CLEAR, FLIGHT_LOOSE = 0, 1, 2, 3
SHAPE_SMOOTHING: Final = 0.08
"""Fraction of the way a shape target moves toward its new value per update (every 3 ticks)."""
CONTACT_LEAD_S: Final = 0.4
"""Strike clips (renderer convention): ball contact happens 0.4 s after the clip starts."""
RECEIVE_LEAD_S: Final = 0.15
STRIKE_ANIMS: Final = frozenset(
    {
        "pass_short",
        "pass_long",
        "cross",
        "shoot",
        "chip",
        "volley",
        "header_standing",
        "header_jumping",
        "tackle",
        "slide_tackle",
        "gk_drop_kick",
        "gk_throw",
        "throw_in",
    }
)
RECEIVE_ANIMS: Final = frozenset({"receive_ground", "receive_aerial"})
HEAD_ONLY_Z: Final = 1.6
"""Above this height a ball can only be headed (or handled by a keeper), never controlled."""
CARRIER_TURN_RATE: Final = 7.0
"""rad/s: a standing ball carrier shaping his body toward the next action (receive-and-turn)."""

# interception / save / claim sub-outcomes
SUB_WON, SUB_IN_PLAY, SUB_LOST, SUB_DEFLECT = 0, 1, 2, 3
SAVE_CATCH, SAVE_PARRY_SAFE, SAVE_PARRY_DANGER, SAVE_PARRY_OUT, SAVE_TO_POST = 0, 1, 2, 3, 4
CLAIM_CATCH, CLAIM_PUNCH, CLAIM_FAIL = 0, 1, 2

KICK_ANIM: Final = {
    K.K_GROUND: "pass_short",
    K.K_DRIVEN: "pass_long",
    K.K_LOFTED: "pass_long",
    K.K_CHIPPED: "chip",
    K.K_CROSS: "cross",
    K.K_CROSS_LOW: "cross",
    K.K_CURLED: "pass_short",
    K.K_OUTSIDE: "pass_short",
    K.K_SHOT_DRIVEN: "shoot",
    K.K_SHOT_PLACED: "shoot",
    K.K_SHOT_FINESSE: "shoot",
    K.K_SHOT_CHIP: "chip",
    K.K_VOLLEY: "volley",
    K.K_HALF_VOLLEY: "volley",
    K.K_HEADER: "header_jumping",
    K.K_TOE_POKE: "shoot",
    K.K_CLEARANCE: "pass_long",
    K.K_THROW_IN: "throw_in",
    K.K_GK_THROW: "gk_throw",
    K.K_PUNT: "gk_drop_kick",
    K.K_DROP_KICK: "gk_drop_kick",
    K.K_HEADER_PASS: "header_standing",
}
ANIM_LOCK: Final = {
    "pass_short": 0.55,
    "pass_long": 0.85,
    "cross": 0.85,
    "shoot": 0.85,
    "chip": 0.8,
    "volley": 0.85,
    "header_standing": 0.7,
    "header_jumping": 1.0,
    "tackle": 0.8,
    "slide_tackle": 1.2,
    "fall": 0.9,
    "get_up": 1.0,
    "throw_in": 1.1,
    "receive_ground": 0.45,
    "receive_aerial": 0.65,
    "gk_dive_left": 1.3,
    "gk_dive_right": 1.3,
    "gk_block_left": 0.9,
    "gk_block_right": 0.9,
    "gk_catch_high": 0.8,
    "gk_catch_low": 0.8,
    "gk_scoop": 0.8,
    "gk_throw": 1.0,
    "gk_drop_kick": 1.1,
    "celebrate": 5.0,
    "disappointed": 3.5,
    "turn": 0.4,
}


@dataclass
class Pending:
    """A kick launched this tick, waiting for path prediction and planning."""

    b: int
    kicker: int
    kind: int
    flight: int
    res: Resolved
    row: int
    plan: K.KickPlan
    noisy: K.Launch | None
    receiver: int
    p_success: float
    p0: float
    event: int
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionRecord:
    """Summary of one on-ball decision (written for seed search and the Moment Scout)."""

    b: int
    tick: int
    t: float
    carrier: int
    slot: int
    kind: int
    n_viable: int
    entropy: float
    xt: float
    best_dv: float
    chosen_dv: float
    intent: dict[str, Any]


class Match:
    def __init__(
        self,
        meta: MatchMeta,
        models: AnalyticsModels | None = None,
        batch: int = 1,
        seed: int | None = None,
        record: bool = False,
        snapshots: bool = False,
    ) -> None:
        self.meta = meta
        self.sq = Squad.from_meta(meta)
        self.models = models or load_models()
        self.styles = styles_for(meta.seed)
        self.rng = np.random.default_rng(np.random.SeedSequence([meta.seed if seed is None else seed, 0x3A7C]))
        self.w = World.empty(batch)
        self.logs = [EventLog() for _ in range(batch)]
        self.record = record
        self.take_snapshots = snapshots
        self.record_decisions = snapshots
        """Keep per-decision option summaries (seed search, Moment Scout)."""
        self.snapshots: list[tuple[int, dict[str, npt.NDArray[Any]]]] = []
        self.decisions: list[DecisionRecord] = []
        self.home_first = meta.home_attacks_positive_x_first_half
        self.done = np.zeros(batch, dtype=bool)
        self.restart_cache: list[F | None] = [None] * batch
        self.pending: list[Pending] = []
        self.keyframe = np.zeros(batch, dtype=bool)
        self.sign = attack_sign(self.w, self.home_first)
        self.targets = np.zeros((batch, P, 2))
        self.caps = np.ones((batch, P))
        self.frames: list[dict[str, Any]] = []
        self.on_frame: Callable[[Match], None] | None = None
        self.forced: list[Any] = [None] * batch
        """Per element, a callable that injects the first action of a fork."""
        self.rollout_tick0 = 0
        self.phase = np.full(batch, PH["dead_ball"], dtype=np.int64)
        self.flight_res: list[dict[str, Any] | None] = [None] * batch
        self.tables_seen = 0
        sq = self.sq
        self._gk = np.array([np.nonzero(sq.is_gk & (sq.team == t))[0][0] for t in (0, 1)])
        self._wood = np.zeros(batch, dtype=bool)
        self._shape: F | None = None
        self.backfill: list[tuple[int, int, int, int, bool, int, int]] = []
        """(b, tick, player, anim, mirror, variant, n_frames): clips that started before the current tick."""
        self._dead_since = np.full(batch, -1e9)
        self._replan_after = np.full(batch, -1e9)
        self.diag: Counter[str] = Counter()
        """Engine diagnostics (how plans resolved), for tuning and tests."""
        self._wood_shot = np.full(batch, -1, dtype=np.int64)
        self._replan_queue: dict[int, int] = {}
        self._carry_safe_until = np.full(batch, -np.inf)
        self._carry_fail_t = np.full(batch, np.inf)
        self._carry_who = np.full(batch, -1, dtype=np.int64)
        self.forced_carry_kept = np.ones(batch, dtype=bool)
        """Outcome drawn for an injected (fan) carry: the physics then plays it out."""

    # ------------------------------------------------------------------------------------------------------
    # set-up
    # ------------------------------------------------------------------------------------------------------
    def start(self) -> None:
        """Kick off the first half."""
        w = self.w
        w.t[:] = 0.0
        w.period[:] = 1
        w.status[:] = ST["first_half"]
        stoppage = 60.0 * self.rng.integers(1, 4)
        w.period_end_t[:] = self.meta.period_duration_s + stoppage
        self.sign = attack_sign(w, self.home_first)
        kick_team = SIDE.index(self.meta.kickoff_team)
        for b in range(w.b):
            self._ev(b, "period_start", None, None, None)
            self._setup_kickoff(b, kick_team, teleport=True)

    def _setup_kickoff(self, b: int, team: int, teleport: bool, delay: float = 1.5) -> None:
        w, sq = self.w, self.sq
        w.ball_state[b] = BS_DEAD
        w.owner[b] = -1
        if w.t[b] <= 0.0:
            w.ball_pos[b] = (0.0, 0.0, BALL_RADIUS)
            w.ball_vel[b] = 0.0
            w.ball_spin[b] = 0.0
        self._dead_since[b] = w.t[b]
        w.restart[b] = R_KICK_OFF
        w.restart_team[b] = team
        w.restart_pos[b] = (0.0, 0.0)
        w.restart_timer[b] = delay
        w.restart_cause[b] = -1
        fwd = [i for i in range(team * N_ON, (team + 1) * N_ON) if sq.line[i] == 3]
        w.restart_taker[b] = fwd[0] if fwd else team * N_ON + 9
        self.restart_cache[b] = restart_targets(w, sq, self.sign, b, self.styles)
        if teleport and self.restart_cache[b] is not None:
            w.pos[b] = self.restart_cache[b]
            w.vel[b] = 0.0
            w.facing[b] = np.where(self.sign[b, sq.team] > 0, 0.0, np.pi)
            w.stamina[b] = np.minimum(1.0, w.stamina[b] + 0.15 * teleport)

    # ------------------------------------------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------------------------------------------
    def step(self) -> None:
        w = self.w
        self.sign = attack_sign(w, self.home_first)
        self.keyframe[:] = False
        self.backfill.clear()
        self._clock()
        self._forced_actions()  # a forked reality's injected action happens before anything else
        self._plan_pending()
        # touches are judged on the state at the start of the tick (the previous frame)
        self._interactions()
        self._flush_replans()
        self._restarts()
        self._decisions()
        self._plan_pending()
        self._flush_replans()
        self._move_players()
        self._move_ball()
        self._update_phase()
        self._animate()
        self._end_of_tick_snapshot()
        if self.on_frame is not None:
            self.on_frame(self)
        w.tick += 1
        w.t = w.tick / SIM_HZ  # exact: t = tick / sim_hz, no float drift over 135k ticks

    def run(self, until_t: float | None = None, max_ticks: int | None = None) -> None:
        n = 0
        while not self.done.all():
            if until_t is not None and self.w.t[0] >= until_t - 1e-9:
                break
            if max_ticks is not None and n >= max_ticks:
                break
            self.step()
            n += 1

    # ------------------------------------------------------------------------------------------------------
    # clock and periods
    # ------------------------------------------------------------------------------------------------------
    def _clock(self) -> None:
        w = self.w
        live = (w.status == ST["first_half"]) | (w.status == ST["second_half"])
        over = live & (w.t >= w.period_end_t) & ((w.ball_state != BS_FLIGHT) | (w.t >= w.period_end_t + 60.0))
        if not over.any():
            ht = w.status == ST["half_time"]
            if ht.any():
                for b in np.nonzero(ht & (w.t >= w.period_end_t))[0]:
                    self._start_second_half(int(b))
            return
        for b in np.nonzero(over)[0]:
            b = int(b)
            # do not end a half while a team is through on goal
            if w.ball_state[b] == BS_OWNED and w.t[b] < w.period_end_t[b] + 20.0:
                o = int(w.owner[b])
                x_t = w.ball_pos[b, 0] * self.sign[b, self.sq.team[o]]
                if x_t > FINAL_THIRD_X:
                    continue
            self._close_open_carry(b, end_of_period=True)
            self._ev(b, "period_end", None, None, None)
            if w.period[b] == 1:
                w.status[b] = ST["half_time"]
                w.period_end_t[b] = w.t[b] + 12.0
                self._dead_since[b] = w.t[b]
                w.ball_state[b] = BS_DEAD
                w.owner[b] = -1
                w.restart[b] = -1
                w.ball_vel[b] = 0.0
            else:
                w.status[b] = ST["full_time"]
                w.ball_state[b] = BS_DEAD
                w.owner[b] = -1
                w.ball_vel[b] = 0.0
                self.done[b] = True

    def _start_second_half(self, b: int) -> None:
        w = self.w
        w.period[b] = 2
        w.status[b] = ST["second_half"]
        w.period_end_t[b] = w.t[b] + self.meta.period_duration_s + 60.0 * self.rng.integers(2, 6)
        self.sign = attack_sign(w, self.home_first)
        first = SIDE.index(self.meta.kickoff_team)
        self._ev(b, "period_start", None, None, None)
        w.hands[b] = False
        self._setup_kickoff(b, 1 - first, teleport=True)

    # ------------------------------------------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------------------------------------------
    def x_t(self, b: int, x: float, team: int) -> float:
        return float(x * self.sign[b, team])

    def _pressure(self, b: int, i: int) -> tuple[float, float, int]:
        """(pressure 0..1, nearest opponent distance, nearest opponent index) for player i."""
        sq, w = self.sq, self.w
        opp = np.arange((1 - sq.team[i]) * N_ON, (2 - sq.team[i]) * N_ON)
        d = np.linalg.norm(w.pos[b, opp] - w.pos[b, i], axis=1)
        d = np.where(sq.is_gk[opp], d + 3.0, d)
        j = int(np.argmin(d))
        return float(np.clip((3.5 - d[j]) / 2.5, 0.0, 1.0)), float(d[j]), int(opp[j])

    def _ev(
        self,
        b: int,
        type_: str,
        player: int | None,
        team: int | None,
        loc: Any,
        q: dict[str, Any] | None = None,
        **env: Any,
    ) -> int:
        w = self.w
        if team is None and player is not None:
            team = int(self.sq.team[player])
        pt = int(w.poss_team[b])
        if "phase_of_play" not in env:
            ph = int(self.phase[b])
            if team is not None and pt >= 0 and team != pt and ph != PH["dead_ball"]:
                ph = PH["defensive_transition"] if ph == PH["attacking_transition"] else PH["out_of_possession"]
            env["phase_of_play"] = PHASES[ph]
        if "possession_id" not in env:
            env["possession_id"] = int(w.possession_id[b])
        if "possession_team" not in env and pt >= 0:
            env["possession_team"] = SIDE[pt]
        if "play_pattern" not in env:
            env["play_pattern"] = PP_NAMES[int(w.play_pattern[b])]
        if "counterpress" not in env and team is not None and pt >= 0 and team != pt:
            env["counterpress"] = bool(w.t[b] - w.last_turnover_t[b] <= COUNTERPRESS_WINDOW_S)
        if "under_pressure" not in env and player is not None:
            env["under_pressure"] = self._pressure(b, player)[1] < 2.6
        location = None if loc is None else (float(loc[0]), float(loc[1]), float(loc[2]) if len(loc) > 2 else 0.0)
        return self.logs[b].add(
            type_,
            float(w.t[b]),
            int(w.tick[b]),
            int(w.period[b]),
            None if team is None else SIDE[team],
            None if player is None else self.sq.ids[player],
            location,
            q,
            **env,
        )

    def _set_anim(self, b: int, i: int, name: str, mirror: bool = False, variant: int = 0) -> None:
        """Start a one-shot clip. Strike clips make contact 0.4 s in, receive clips 0.15 s before the ball arrives,
        so those are started in the past: the recorder back-fills the preceding frames (``backfill``)."""
        w = self.w
        lead = CONTACT_LEAD_S if name in STRIKE_ANIMS else (RECEIVE_LEAD_S if name in RECEIVE_ANIMS else 0.0)
        playing = ANIMS[int(w.anim[b, i])]
        if lead > 0 and w.anim_lock[b, i] > 0 and playing in STRIKE_ANIMS and w.anim_t[b, i] < 0.6:
            return  # the player is still in the strike he just made: do not restart (and rewrite) it
        w.anim[b, i] = A[name]
        # _animate adds one tick before the frame is recorded
        w.anim_t[b, i] = lead - DT
        w.anim_lock[b, i] = ANIM_LOCK.get(name, 0.6) - lead + DT
        w.anim_mirror[b, i] = mirror
        w.anim_variant[b, i] = variant
        if lead > 0:
            self.backfill.append((b, int(w.tick[b]), i, A[name], mirror, variant, round(lead / DT)))

    def _snapshot(self, b: int) -> None:
        if self.take_snapshots and b == 0:
            tick = int(self.w.tick[0])
            if not self.snapshots or self.snapshots[-1][0] != tick:
                self.snapshots.append((tick, self.w.snapshot(0)))

    def _end_of_tick_snapshot(self) -> None:
        """Touches that did not lead to an immediate action (receipts, recoveries) are keyframes too: store the
        state at the start of the next tick, with the player in control, under this tick's number."""
        if not self.take_snapshots or not self.keyframe[0]:
            return
        w = self.w
        tick = int(w.tick[0])
        if self.snapshots and self.snapshots[-1][0] == tick:
            return
        if w.ball_state[0] != BS_OWNED or w.owner[0] < 0:
            return
        snap = w.snapshot(0)
        snap["tick"] = np.array(tick + 1)
        snap["t"] = np.array((tick + 1) / SIM_HZ)
        self.snapshots.append((tick, snap))

    def has_snapshot(self, tick: int) -> bool:
        return bool(self.snapshots) and self.snapshots[-1][0] == tick

    def _new_possession(self, b: int, team: int, pattern: str, turnover: bool) -> None:
        w = self.w
        w.possession_id[b] += 1
        w.poss_team[b] = team
        w.poss_start_t[b] = w.t[b]
        w.poss_start_x[b] = self.x_t(b, w.ball_pos[b, 0], team)
        w.poss_path[b] = 0.0
        w.poss_counter_ok[b] = turnover and w.poss_start_x[b] < FINAL_THIRD_X
        w.play_pattern[b] = PP[pattern]
        if turnover:
            w.last_turnover_t[b] = w.t[b]

    def _gain(self, b: int, i: int, pattern: str = "regular_play") -> None:
        """Player i takes control of the ball."""
        w, sq = self.w, self.sq
        team = int(sq.team[i])
        if w.poss_team[b] != team:
            turnover = w.poss_team[b] >= 0 and pattern == "regular_play"
            self._new_possession(b, team, pattern if pattern != "regular_play" else "regular_play", turnover)
            if pattern == "from_keeper":
                w.play_pattern[b] = PP["from_keeper"]
        w.owner[b] = i
        self._wood_shot[b] = -1
        w.ball_state[b] = BS_OWNED
        w.last_touch[b] = i
        w.last_touch_team[b] = team
        w.collector[b] = -1
        w.flight_mode[b] = FM_NONE
        w.receive_t[b] = w.t[b]
        w.carry_start[b] = w.pos[b, i]
        w.carry_start_t[b] = w.t[b]
        w.carry_path[b] = 0.0
        w.carry_max_speed[b] = 0.0
        w.carry_target[b] = w.pos[b, i]
        w.carry_speed[b] = 2.0
        press, _, _ = self._pressure(b, i)
        tempo = self.styles[team].tempo
        w.act_timer[b] = (0.5 + 1.3 * self.rng.random() * (1.0 - 0.6 * press)) / tempo
        w.ball_vel[b] = 0.0
        w.ball_spin[b] = 0.0
        self._resolve_clearance(b, team)

    def _resolve_clearance(self, b: int, team: int) -> None:
        w = self.w
        ce = int(w.clear_ev[b])
        if ce < 0:
            return
        q = self.logs[b].q(ce)
        clr_team = SIDE.index(self.logs[b].records[ce]["team"])
        if team == clr_team:
            q["outcome"] = "in_play_safe"
        else:
            x_t = self.x_t(b, w.ball_pos[b, 0], clr_team)
            q["outcome"] = "in_play_danger" if x_t < 0 else "in_play_safe"
        w.clear_ev[b] = -1

    # ------------------------------------------------------------------------------------------------------
    # restarts
    # ------------------------------------------------------------------------------------------------------
    def _restarts(self) -> None:
        w = self.w
        dead = (w.ball_state == BS_DEAD) & (w.restart >= 0) & ~self.done
        if not dead.any():
            return
        w.restart_timer[dead] -= DT
        for b in np.nonzero(dead & (w.restart_timer <= 0))[0]:
            b = int(b)
            taker = int(w.restart_taker[b])
            spot = self._dead_ball_spot(b)
            if np.linalg.norm(w.ball_pos[b] - spot) > 0.05:
                continue
            if np.linalg.norm(w.pos[b, taker] - w.ball_pos[b, :2]) > CONTROL_RADIUS - 0.05:
                continue
            self._take_restart(b)

    def _set_restart(self, b: int, r: int, team: int, pos: F, delay: float, cause: int = -1) -> None:
        w, sq = self.w, self.sq
        self._dead_since[b] = w.t[b]
        self._wood_shot[b] = -1
        self._close_open_carry(b)
        w.ball_state[b] = BS_DEAD
        w.owner[b] = -1
        w.collector[b] = -1
        w.flight_mode[b] = FM_NONE
        w.hands[b] = False
        w.restart[b] = r
        w.restart_team[b] = team
        w.restart_pos[b] = pos
        w.restart_timer[b] = delay
        w.restart_cause[b] = cause
        for i in range(P):
            w.run_until[b, i] = -1e9
        att = np.arange(team * N_ON, (team + 1) * N_ON)
        a = sq.attr
        if r == R_GOAL_KICK:
            taker = int(self._gk[team])
        elif r == R_PENALTY:
            out = att[~sq.is_gk[att]]
            taker = int(out[np.argmax(a["finishing"][out] + 0.5 * a["composure"][out])])
        elif r == R_CORNER:
            out = att[~sq.is_gk[att]]
            taker = int(out[np.argmax(a["crossing"][out] + 0.3 * a["curve"][out])])
        elif r == R_FK_DIRECT and np.hypot(HALF_LENGTH - pos[0] * self.sign[b, team], pos[1]) < 32.0:
            out = att[~sq.is_gk[att]]
            taker = int(out[np.argmax(a["curve"][out] + a["long_shots"][out] + 0.5 * a["finishing"][out])])
        else:
            cand = att[~sq.is_gk[att]]
            if r in (R_FK_DIRECT, R_FK_INDIRECT) and pos[0] * self.sign[b, team] < -30:
                cand = att
            d = np.linalg.norm(w.pos[b, cand] - pos, axis=1)
            taker = int(cand[np.argmin(d)])
        w.restart_taker[b] = taker
        self.restart_cache[b] = restart_targets(w, sq, self.sign, b, self.styles)

    def _take_restart(self, b: int) -> None:
        w = self.w
        r = int(w.restart[b])
        team = int(w.restart_team[b])
        taker = int(w.restart_taker[b])
        rp = w.restart_pos[b]
        self._new_possession(b, team, RESTART_PATTERN[r], turnover=False)
        w.last_restart_t[b] = w.t[b]
        w.last_restart_type[b] = r
        w.restart_poss_id[b] = w.possession_id[b]
        wall = None
        if r in (R_FK_DIRECT, R_FK_INDIRECT) and self.restart_cache[b] is not None:
            opp = np.arange((1 - team) * N_ON, (2 - team) * N_ON)
            d = np.linalg.norm(w.pos[b, opp] - rp, axis=1)
            wall = (
                int(((d > 8.0) & (d < 10.5)).sum())
                if np.hypot(HALF_LENGTH - rp[0] * self.sign[b, team], rp[1]) < 33
                else 0
            )
        q: dict[str, Any] = {"restart_type": RESTART_NAMES[r], "awarded_to": SIDE[team]}
        if int(w.restart_cause[b]) >= 0:
            q["cause_event_id"] = int(w.restart_cause[b])
        if wall is not None:
            q["wall_size"] = wall
        rev = self._ev(b, "restart", taker, team, (rp[0], rp[1], BALL_RADIUS), q)
        w.restart[b] = -1
        self.restart_cache[b] = None
        w.owner[b] = taker
        w.ball_state[b] = BS_OWNED
        w.last_touch[b] = taker
        w.last_touch_team[b] = team
        w.receive_t[b] = -1e9
        w.act_timer[b] = 0.0
        ctx = RESTART_CTX[r]
        self._decide([b], [ctx], restart_event={b: (rev, r)})

    # ------------------------------------------------------------------------------------------------------
    # decisions
    # ------------------------------------------------------------------------------------------------------
    def _forced_actions(self) -> None:
        for b in range(self.w.b):
            f = self.forced[b]
            if f is not None:
                self.forced[b] = None
                f(self, b)

    def _decisions(self) -> None:
        w = self.w
        stunned = w.stun[np.arange(w.b), np.maximum(w.owner, 0)] > 0
        due = (w.ball_state == BS_OWNED) & (w.act_timer <= 0) & ~self.done & ~stunned
        due &= (w.status == ST["first_half"]) | (w.status == ST["second_half"])
        if not due.any():
            w.act_timer -= DT
            return
        bs = [int(b) for b in np.nonzero(due)[0]]
        ctxs = [CTX_GK_HANDS if w.hands[b] else CTX_OPEN for b in bs]
        self._decide(bs, ctxs)
        w.act_timer -= DT

    def decider_input(self, bs: list[int], ctxs: list[int]) -> DeciderInput:
        w, sq = self.w, self.sq
        idx = np.array(bs)
        carriers = w.owner[idx]
        team = sq.team[carriers]
        sign = self.sign[idx, team]
        lines = offside_lines(w, sq, self.sign)[idx, team]
        last_def = last_defender_x(w, sq, self.sign)[idx, team]
        ctx = np.array(ctxs, dtype=np.int64)
        direct = w.restart[idx] < 0  # placeholder, refined below
        direct = np.array([w.last_restart_type[b] == R_FK_DIRECT for b in bs])
        counter = self.phase[idx] == PH["counter_attack"]
        setp = self.phase[idx] == PH["set_piece"]
        sit = situation_code(ctx, direct, counter, setp)
        return DeciderInput(
            carrier=carriers,
            sign=sign,
            ball=w.ball_pos[idx].copy(),
            pos=w.pos[idx],
            vel=w.vel[idx],
            run_target=w.run_target[idx],
            running=w.run_until[idx] > w.t[idx, None],
            ctx=ctx,
            situation=sit,
            last_def_x=last_def,
            offside_line=lines,
            t=w.t[idx],
        )

    def _decide(self, bs: list[int], ctxs: list[int], restart_event: dict[int, tuple[int, int]] | None = None) -> None:
        for b in bs:
            self._snapshot(b)
            self.keyframe[b] = True
        di = self.decider_input(bs, ctxs)
        table = build_options(self.sq, self.models, self.styles, di, self.rng)
        choice = choose(table, self.rng)
        self.tables_seen += len(bs)
        for k, b in enumerate(bs):
            slot = int(choice[k])
            if not table.valid[k, slot]:
                slot = S_HOLD if table.valid[k, S_HOLD] else int(np.argmax(table.valid[k]))
            self._record_decision(b, k, slot, table, di)
            self._execute_option(b, k, slot, table, di, ctxs[k], (restart_event or {}).get(b))

    def _record_decision(self, b: int, k: int, slot: int, table: OptionTable, di: DeciderInput) -> None:
        if b != 0 or not (self.take_snapshots or self.record_decisions):
            return
        w = self.w
        c = int(di.carrier[k])
        viable = table.valid[k] & (table.p[k] > 0.5) & (table.dv[k] > table.dv[k, S_HOLD] - 0.005)
        bx_t = w.ball_pos[b, :2] * di.sign[k]
        xt = float(np.asarray(self.models.xt(np.array([bx_t[0]]), np.array([bx_t[1]])))[0])
        intent = self._intent_dict(b, k, slot, table)
        self.decisions.append(
            DecisionRecord(
                b=b,
                tick=int(w.tick[b]),
                t=float(w.t[b]),
                carrier=c,
                slot=slot,
                kind=int(table.kind[k, slot]),
                n_viable=int(viable.sum()),
                entropy=float(option_entropy(table.prob[k : k + 1])[0]),
                xt=xt,
                best_dv=float(np.max(np.where(table.valid[k], table.dv[k], -1))),
                chosen_dv=float(table.dv[k, slot]),
                intent=intent,
            )
        )

    def _intent_dict(self, b: int, k: int, slot: int, table: OptionTable) -> dict[str, Any]:
        kind = int(table.kind[k, slot])
        names = ("pass_player", "pass_space", "shot", "carry", "clearance")
        rec = int(table.receiver[k, slot])
        tgt = table.target[k, slot]
        d: dict[str, Any] = {
            "kind": names[kind],
            "player": self.sq.ids[int(self.w.owner[b])],
            "power": float(np.clip(table.power[k, slot], 0, 1)),
            "curve": float(np.clip(table.curve[k, slot], -1, 1)),
        }
        if kind == I_PASS_PLAYER:
            d["target_player"] = self.sq.ids[rec]
        else:
            d["target"] = [float(tgt[0]), float(tgt[1]), float(tgt[2])]
        return d

    def _execute_option(
        self,
        b: int,
        k: int,
        slot: int,
        table: OptionTable,
        di: DeciderInput,
        ctx: int,
        restart: tuple[int, int] | None,
    ) -> None:
        o = table.kind.shape[1]
        row = k * o + slot
        kind = int(table.kind[k, slot])
        target = table.target[k, slot]
        receiver = int(table.receiver[k, slot])
        info = {
            "ctx": ctx,
            "situation": int(di.situation[k]),
            "restart": restart,
            "pressure": float(table.pressure[k]),
            "receiver_offside": bool(table.receiver_offside[k, slot]),
            "slot": slot,
            "power": float(table.power[k, slot]),
        }
        self.execute(b, kind, table.res, row, target, receiver, float(table.p[k, slot]), float(table.p0[k, slot]), info)

    def execute(
        self,
        b: int,
        kind: int,
        res: Resolved,
        row: int,
        target: F,
        receiver: int,
        p_success: float,
        p0: float,
        info: dict[str, Any],
    ) -> None:
        """Carry out a resolved action for the ball carrier of element b (AI or injected fan intent)."""
        w = self.w
        c = int(w.owner[b])
        if kind == I_CARRY:
            self._start_carry(b, c, target, info, p_success)
            return
        self._release(b, c, kind, res, row, target, receiver, p_success, p0, info)

    # ------------------------------------------------------------------------------------------------------
    # carries and take-ons
    # ------------------------------------------------------------------------------------------------------
    def _start_carry(self, b: int, c: int, target: F, info: dict[str, Any], p_success: float) -> None:
        w, sq = self.w, self.sq
        w.hands[b] = False
        slot = int(info.get("slot", -1))
        drib = sq.attr["dribbling"][c] / 99.0
        w.carry_target[b] = target[:2]
        w.carry_speed[b] = sq.max_speed[c] * (0.42 + 0.2 * drib) * (0.55 if slot == S_HOLD else 1.0)
        w.act_timer[b] = 0.8 + 1.2 * self.rng.random()
        if slot == S_DRIBBLE:
            self._take_on(b, c, p_success)
            return
        # the carry keeps the ball with the displayed retention probability: drawn now, played out by the physics
        kept = self.rng.random() < p_success
        if info.get("forced"):
            self.forced_carry_kept[b] = kept
        if kept:
            self._carry_safe_until[b] = w.t[b] + w.act_timer[b]
            self._carry_fail_t[b] = np.inf
        else:
            self._carry_safe_until[b] = -np.inf
            self._carry_fail_t[b] = w.t[b] + w.act_timer[b] * self.rng.uniform(0.3, 0.9)
        self._carry_who[b] = c

    def _carry_lost(self, b: int) -> None:
        """A carry drawn to fail: the nearest opponent nicks it, or a heavy touch runs away from the carrier."""
        w, sq = self.w, self.sq
        c = int(w.owner[b])
        self._carry_fail_t[b] = np.inf
        team = int(sq.team[c])
        opp = np.arange((1 - team) * N_ON, (2 - team) * N_ON)
        d = np.linalg.norm(w.pos[b, opp] - w.ball_pos[b, :2], axis=1)
        j = int(opp[int(np.argmin(d))])
        loc = w.ball_pos[b].copy()
        if float(d.min()) <= CONTROL_RADIUS and w.stun[b, j] <= 0:
            e1 = self._ev(
                b,
                "duel",
                j,
                1 - team,
                loc,
                {"opponent": sq.ids[c], "duel_type": "tackle", "outcome": "won", "possession_after": "tackler"},
            )
            e2 = self._ev(b, "dispossessed", c, team, loc, {"opponent": sq.ids[j]})
            self.logs[b].link(e1, e2)
            self._close_open_carry(b, lost=True)
            self._set_anim(b, j, "tackle")
            self._gain(b, j)
        else:
            self._ev(b, "miscontrol", c, team, loc, {"aerial_won": False})
            self._close_open_carry(b, lost=True)
            # the heavy touch runs toward the closest opponent, who is first to it
            gap = w.pos[b, j] - w.ball_pos[b, :2]
            self._deflect(b, c, scale=0.0, spread=0.25, toward=gap)
            w.ball_vel[b, :2] *= max(1.0, float(np.sqrt(2 * 0.8 * np.linalg.norm(gap))) / 2.0)

    def _take_on(self, b: int, c: int, p_success: float) -> None:
        w, sq = self.w, self.sq
        _, dist, j = self._pressure(b, c)
        if dist > 3.5:
            return
        team = int(sq.team[c])
        ok = self.rng.random() < p_success
        loc = w.ball_pos[b]
        ev = self._ev(
            b,
            "take_on",
            c,
            team,
            loc,
            {
                "opponent": sq.ids[j],
                "outcome": "success" if ok else "fail",
                "nutmeg": bool(ok and self.rng.random() < 0.08),
            },
            under_pressure=True,
        )
        w.last_dribble_t[b] = w.t[b]
        if ok:
            w.stun[b, j] = 0.9
            self._set_anim(b, j, "tackle" if self.rng.random() < 0.5 else "turn")
            w.carry_speed[b] = sq.max_speed[c] * 0.92
            w.act_timer[b] = 0.7 + 0.5 * self.rng.random()
        else:
            reach = float(np.linalg.norm(w.pos[b, j] - w.ball_pos[b, :2])) <= CONTROL_RADIUS
            dv = self._ev(
                b,
                "duel",
                j,
                None,
                loc,
                {
                    "opponent": sq.ids[c],
                    "duel_type": "tackle",
                    "outcome": "won",
                    "possession_after": "tackler" if reach else "loose",
                },
                under_pressure=True,
            )
            self.logs[b].link(ev, dv)
            self._close_open_carry(b, lost=True)
            self._set_anim(b, j, "tackle")
            w.stun[b, c] = 0.6
            if reach:
                self._gain(b, j)
            else:
                w.ball_vel[b] = 0.0
                self._deflect(b, j, scale=0.0, spread=2.0)
            self.keyframe[b] = True

    def _close_open_carry(self, b: int, lost: bool = False, end_of_period: bool = False) -> None:
        """Emit the carry of the current owner if it covered at least 3 m."""
        w, sq = self.w, self.sq
        c = int(w.owner[b])
        if c < 0 or w.ball_state[b] != BS_OWNED:
            return
        start = w.carry_start[b]
        end = w.pos[b, c]
        dist = float(np.linalg.norm(end - start))
        if w.carry_path[b] >= 3.0 and dist >= 1.0:
            team = int(sq.team[c])
            s = self.sign[b, team]
            d0 = np.hypot(HALF_LENGTH - start[0] * s, start[1] * s)
            d1 = np.hypot(HALF_LENGTH - end[0] * s, end[1] * s)
            prog = _progressive(start[0] * s, end[0] * s, d0, d1, end[1] * s) and end[0] * s > 0
            outcome = "end_of_period" if end_of_period else ("lost" if lost else "retained")
            self.logs[b].add(
                "carry",
                float(w.carry_start_t[b]),
                round(w.carry_start_t[b] * SIM_HZ),
                int(w.period[b]),
                SIDE[team],
                sq.ids[c],
                (float(start[0]), float(start[1]), BALL_RADIUS),
                {
                    "outcome": outcome,
                    "distance_m": dist,
                    "path_m": float(w.carry_path[b]),
                    "max_speed_mps": float(w.carry_max_speed[b]),
                    "is_progressive": bool(prog),
                },
                end_location=(float(end[0]), float(end[1]), BALL_RADIUS),
                duration_s=float(w.t[b] - w.carry_start_t[b]),
                possession_id=int(w.possession_id[b]),
                possession_team=SIDE[team],
                play_pattern=PP_NAMES[int(w.play_pattern[b])],
                phase_of_play=PHASES[int(self.phase[b])],
            )
        w.carry_path[b] = 0.0
        w.carry_start[b] = end

    # ------------------------------------------------------------------------------------------------------
    # kicks
    # ------------------------------------------------------------------------------------------------------
    def _release(
        self,
        b: int,
        c: int,
        kind: int,
        res: Resolved,
        row: int,
        target: F,
        receiver: int,
        p_success: float,
        p0: float,
        info: dict[str, Any],
    ) -> None:
        w, sq = self.w, self.sq
        team = int(sq.team[c])
        self._close_open_carry(b)
        kick = int(res.kick[row])
        origin = w.ball_pos[b : b + 1].copy()  # the flight starts where the ball is on the contact tick
        tgt = res.target[row : row + 1].copy()
        raw = np.array([float(np.clip(info.get("power", 0.5), 0, 1))])
        power = effective_power(sq, np.array([c]), np.array([kick]), raw)
        pre = info.pop("plan", None)
        if pre is not None:  # an injected (fan) action resolved once for every reality
            plan, needs_refine = K.KickPlan(origin, tgt, np.array([kick]), pre), False
        else:
            plan = K.plan_kicks(origin, tgt, np.array([kick]), power, res.curve_applied[row : row + 1])
            needs_refine = kick not in (K.K_GROUND,) or abs(float(res.curve_applied[row])) > 0.05
        w.ball_pos[b] = origin[0]
        w.ball_vel[b] = 0.0  # set in _plan_pending once all of this tick's kicks are refined together
        w.ball_state[b] = BS_FLIGHT
        w.owner[b] = -1
        w.hands[b] = False
        w.last_touch[b] = c
        w.last_touch_team[b] = team
        w.flight_passer[b] = c
        w.flight_receiver[b] = receiver
        w.flight_t0[b] = w.t[b]
        w.flight_origin[b] = origin[0]
        w.flight_peak[b] = origin[0, 2]
        w.collector[b] = -1
        w.flight_mode[b] = FM_FREE
        anim = KICK_ANIM.get(kick, "pass_short")
        if kind == I_CLEARANCE and kick != K.K_HEADER_PASS:
            anim = "pass_long"
        self._set_anim(b, c, anim, mirror=bool(res.body[row] == 1))
        if kick in (K.K_HEADER, K.K_HEADER_PASS) and w.ball_pos[b, 2] > 1.9:
            w.vz[b, c] = jump_speed(sq.attr["jumping"][c : c + 1])[0]
        flight = {I_SHOT: FLIGHT_SHOT, I_CLEARANCE: FLIGHT_CLEAR}.get(kind, FLIGHT_PASS)
        w.flight_kind[b] = flight
        info["_target"] = target
        info["_refine"] = needs_refine
        self.pending.append(Pending(b, c, kind, flight, res, row, plan, None, receiver, p_success, p0, -1, info))

    def _launch_pending(self, pend: list["Pending"]) -> None:
        """Refine this tick's kicks in one batch, add execution noise, launch the balls and log the events."""
        w = self.w
        ref = [p for p in pend if p.info.get("_refine")]
        if ref:
            plans = [p.plan for p in ref]
            batch = K.refine(
                K.KickPlan(
                    np.concatenate([q.origin for q in plans]),
                    np.concatenate([q.target for q in plans]),
                    np.concatenate([q.kick for q in plans]),
                    K.Launch(
                        np.concatenate([q.launch.speed for q in plans]),
                        np.concatenate([q.launch.elev for q in plans]),
                        np.concatenate([q.launch.azimuth for q in plans]),
                        np.concatenate([q.launch.spin for q in plans]),
                    ),
                )
            )
            for k, p in enumerate(ref):
                sel = np.array([k])
                p.plan = K.KickPlan(batch.origin[sel], batch.target[sel], batch.kick[sel], batch.launch.take(sel))
        for p in pend:
            b, row, res = p.b, p.row, p.res
            kick = int(res.kick[row])
            noisy = K.perturb(p.plan.launch, np.array([kick]), res.noise_mult[row : row + 1], self.rng)
            if kick == K.K_GROUND and abs(float(res.curve_applied[row])) < 0.05:
                noisy.elev[:] = 0.0
                noisy.spin[:] = 0.0
            w.ball_vel[b] = noisy.velocity()[0]
            w.ball_spin[b] = noisy.spin[0]
            p.noisy = noisy
            ev = self._kick_event(b, p.kicker, p.kind, res, row, p.info["_target"], p.receiver, p.p0, p.info, noisy)
            self.logs[b].records[ev]["_p"] = p.p_success  # displayed (player-adjusted) probability, engine-private
            w.flight_event[b] = ev
            p.event = ev

    def _kick_event(
        self,
        b: int,
        c: int,
        kind: int,
        res: Resolved,
        row: int,
        target: F,
        receiver: int,
        p0: float,
        info: dict[str, Any],
        noisy: K.Launch,
    ) -> int:
        w, sq = self.w, self.sq
        team = int(sq.team[c])
        loc = w.ball_pos[b].copy()
        first_time = bool(w.t[b] - w.receive_t[b] <= FIRST_TIME_S)
        body = BODY_PARTS[int(res.body[row])]
        restart = info.get("restart")
        if kind == I_SHOT:
            s = self.sign[b, team]
            x_t, y_t = loc[0] * s, loc[1] * s
            dist = float(np.hypot(HALF_LENGTH - x_t, y_t))
            a1 = np.arctan2(HALF_GOAL - y_t, HALF_LENGTH - x_t)
            a2 = np.arctan2(-HALF_GOAL - y_t, HALF_LENGTH - x_t)
            sit = _situation_name(
                int(info.get("situation", SIT_OPEN)),
                int(w.last_restart_type[b]),
                bool(w.t[b] - w.last_restart_t[b] <= SET_PIECE_WINDOW_S and w.restart_poss_id[b] == w.possession_id[b]),
                (restart is not None and restart[1] == R_KICK_OFF)
                or (
                    w.last_restart_type[b] == R_KICK_OFF
                    and w.t[b] - w.last_restart_t[b] <= 10
                    and w.restart_poss_id[b] == w.possession_id[b]
                ),
            )
            gk = int(self._gk[1 - team])
            gk_speed = float(np.linalg.norm(w.vel[b, gk]))
            gk_pos_name = (
                "prone" if (w.z[b, gk] > 0.05 or w.stun[b, gk] > 0) else ("moving" if gk_speed > 1.2 else "set")
            )
            opp = np.arange((1 - team) * N_ON, (2 - team) * N_ON)
            outfield = opp[~sq.is_gk[opp]]
            from sim.evaluate import cone_count

            cone = int(cone_count(np.array([[x_t, y_t]]), (w.pos[b, outfield] * s)[None])[0])
            gk_off = HALF_LENGTH - w.pos[b, gk, 0] * s
            q: dict[str, Any] = {
                "keeper": sq.ids[gk],
                "outcome": "off_target",
                "body_part": body if body in ("right_foot", "left_foot", "head") else "other",
                "technique": SHOT_TECHNIQUES[int(res.shot_tech[row])],
                **({"body_part": "other"} if _scramble(res, row, self.rng) else {}),
                "situation": sit,
                "ball_speed_mps": float(noisy.speed[0]),
                "distance_to_goal_m": dist,
                "angle_rad": float(abs(a1 - a2)),
                "on_target": False,
                "xg": float(np.clip(p0, 0, 1)),
                "first_time": first_time,
                "one_on_one": bool(cone == 0 and dist < 18 and gk_off < 8),
                "open_goal": bool(gk_off > 6 and dist < 25),
                "follows_dribble": bool(w.t[b] - w.last_dribble_t[b] < 2.5),
                "aerial_won": bool(w.t[b] - w.aerial_won_t[b] < 0.3),
                "n_defenders_in_cone": cone,
                "gk_position": gk_pos_name,
                "freeze_frame_ref": f"snap:{self.meta.match_id}:{int(w.tick[b])}",
            }
            lp = int(w.last_pass_ev[b])
            if lp >= 0 and w.last_pass_team[b] == team and w.t[b] - w.last_pass_t[b] <= KEY_PASS_WINDOW_S:
                lq = self.logs[b].q(lp)
                if lq.get("outcome") == "complete":
                    q["key_pass_id"] = lp
            ev = self._ev(b, "shot", c, team, loc, q)
            if "key_pass_id" in q:
                lq = self.logs[b].q(int(q["key_pass_id"]))
                lq["is_key_pass"] = True
                lq["assisted_shot_id"] = ev
                self.logs[b].link(ev, int(q["key_pass_id"]))
            w.shot_ev[b] = ev
            if restart is not None:
                self.logs[b].link(ev, restart[0])
            return ev
        if kind == I_CLEARANCE:
            q = {
                "body_part": body if body in ("right_foot", "left_foot", "head") else "other",
                "outcome": "in_play_safe",
                "aerial_won": bool(w.t[b] - w.aerial_won_t[b] < 0.3),
            }
            ev = self._ev(b, "clearance", c, team, loc, q)
            w.clear_ev[b] = ev
            return ev
        # pass
        s = self.sign[b, team]
        tgt = target[:2]
        ang = float(np.arctan2(noisy.velocity()[0, 1], noisy.velocity()[0, 0]))
        rtype = None if restart is None else RESTART_NAMES[restart[1]]
        gk_dist = None
        if sq.is_gk[c]:
            kk = int(res.kick[row])
            gk_dist = {K.K_GK_THROW: "throw", K.K_PUNT: "punt", K.K_DROP_KICK: "drop_kick"}.get(kk, "pass")
            if rtype == "goal_kick":
                gk_dist = "goal_kick"
        tech = PASS_TECHNIQUES[int(res.pass_tech[row])]
        ptype = PASS_TYPES[int(res.pass_type[row])]
        q = {
            "recipient": sq.ids[receiver] if receiver >= 0 else None,
            "outcome": "incomplete",
            "height": "ground",
            "length_class": "short",
            "body_part": body,
            "technique": tech,
            "pass_type": ptype,
            "restart": rtype,
            "gk_distribution": gk_dist,
            "length_m": float(np.linalg.norm(tgt - loc[:2])),
            "angle_rad": ang,
            "ball_speed_mps": float(noisy.speed[0]),
            "peak_height_m": float(loc[2]),
            "first_time": first_time,
            "recipient_offside": bool(info.get("receiver_offside", False)),
            "xpass": float(np.clip(p0, 0, 1)),
        }
        ev = self._ev(b, "pass", c, team, loc, q)
        if restart is not None:
            self.logs[b].link(ev, restart[0])
        w.last_pass_ev[b] = ev
        w.last_pass_t[b] = w.t[b]
        w.last_pass_team[b] = team
        _ = s
        return ev

    # ------------------------------------------------------------------------------------------------------
    # planning flights launched this tick
    # ------------------------------------------------------------------------------------------------------
    def _plan_pending(self) -> None:
        if not self.pending:
            return
        pend, self.pending = self.pending, []
        w = self.w
        self._launch_pending(pend)
        shots = [p for p in pend if p.flight == FLIGHT_SHOT]
        n_mc = 64  # enough samples that 1 / E[on x psxg] is not biased upward
        pos = np.stack([w.ball_pos[p.b] for p in pend])
        vel = np.stack([w.ball_vel[p.b] for p in pend])
        spin = np.stack([w.ball_spin[p.b] for p in pend])
        extra_pos, extra_vel, extra_spin = [], [], []
        for p in shots:
            for _ in range(1):
                lau = K.perturb(
                    K.Launch(
                        np.repeat(p.plan.launch.speed, n_mc),
                        np.repeat(p.plan.launch.elev, n_mc),
                        np.repeat(p.plan.launch.azimuth, n_mc),
                        np.repeat(p.plan.launch.spin, n_mc, axis=0),
                    ),
                    np.full(n_mc, int(p.res.kick[p.row])),
                    np.full(n_mc, float(p.res.noise_mult[p.row])),
                    self.rng,
                )
                extra_pos.append(np.repeat(p.plan.origin, n_mc, axis=0))
                extra_vel.append(lau.velocity())
                extra_spin.append(lau.spin)
        if extra_pos:
            pos = np.concatenate([pos, *extra_pos])
            vel = np.concatenate([vel, *extra_vel])
            spin = np.concatenate([spin, *extra_spin])
        steps = 100 if any(p.flight != FLIGHT_SHOT for p in pend) else 60
        paths = PL.predict_paths(pos, vel, spin, steps)
        for j, p in enumerate(pend):
            if p.flight == FLIGHT_SHOT:
                k0 = len(pend) + shots.index(p) * n_mc
                self._plan_shot(p, paths, j, np.arange(k0, k0 + n_mc))
            else:
                self._plan_pass(p, paths, j)

    def _reach(
        self, b: int, path: F, vpath: F, exclude: int = -1, exclude_until: int = 0, stop_k: int = PL.NEVER
    ) -> IA:
        w, sq = self.w, self.sq
        earliest = np.zeros(P, dtype=np.int64)
        if exclude >= 0:
            earliest[exclude] = exclude_until
        earliest = np.maximum(earliest, np.ceil(w.stun[b] / DT).astype(np.int64))
        pc = self.models.pitch_control_params
        react = np.full(P, pc.reaction_time_s)
        vmax = np.full(P, pc.max_player_speed_mps)
        # feet and chest up to 1.6 m; higher balls only with the head (jump reach) or a keeper's hands
        reach_z = sq.jump_reach.copy()
        k = PL.reach_ticks(path, vpath, w.pos[b], w.vel[b], react, vmax, None, reach_z, earliest, stop_k=stop_k)
        # outfield players can only head balls up to their jump reach; keepers only handle inside their box
        for t in (0, 1):
            g = int(self._gk[t])
            if k[g] < PL.NEVER:
                pk = path[min(int(k[g]), len(path) - 1)]
                x_t = pk[0] * self.sign[b, t]
                in_box = x_t <= -BOX_X and abs(pk[1]) <= BOX_HALF_WIDTH
                if not in_box and pk[2] > 1.9:
                    k[g] = PL.NEVER
        return k

    def _reach_physical(self, b: int, path: F, vpath: F, who: IA, stop_k: int) -> IA:
        """Earliest reachable path index for players ``who`` with their own speed, acceleration and a short
        reaction (they are expecting the ball)."""
        w, sq = self.w, self.sq
        vmax = effective_max_speed(sq.max_speed[who], w.stamina[b, who])
        earliest = np.ceil(w.stun[b, who] / DT).astype(np.int64)
        return PL.reach_ticks(
            path, vpath, w.pos[b, who], w.vel[b, who], np.full(len(who), 0.3), vmax, sq.max_accel[who],
            sq.jump_reach[who], earliest, stop_k=stop_k,
        )  # fmt: skip

    def _plan_pass(self, p: Pending, paths: PL.Paths, j: int) -> None:
        w, sq = self.w, self.sq
        b = p.b
        path, vpath = paths.xs[:, j], paths.vs[:, j]
        out_k = int(min(paths.out_k[j], paths.goal_k[j]))
        team = int(sq.team[p.kicker])
        k = self._reach(b, path, vpath, exclude=p.kicker, exclude_until=12, stop_k=out_k)
        mates = np.arange(team * N_ON, (team + 1) * N_ON)
        opps = np.arange((1 - team) * N_ON, (2 - team) * N_ON)
        k_m, k_o = k[mates], k[opps]
        # a receiver standing offside cannot complete the pass (displayed p = 0), but the ball can still reach him
        p_draw = 0.75 if p.info.get("receiver_offside") else p.p_success
        success = self.rng.random() < p_draw
        rec = p.receiver
        mode, coll = FM_FREE, -1
        sub = SUB_WON
        if p.flight == FLIGHT_CLEAR:
            success = False
        if p.flight == FLIGHT_PASS:
            self.diag["pass_success_drawn" if success else "pass_fail_drawn"] += 1
        if success:
            # interceptions follow the pitch-control time-to-intercept (0.7 s, 5 m/s); for the pass that does
            # arrive, a teammate's real kinematics decide where he meets it
            k_phys = self._reach_physical(b, path, vpath, mates, out_k)
            k[mates] = k_phys
            k_m = k_phys
            if not (k_m < PL.NEVER).any():
                self.diag["pass_success_unreachable"] += 1
        if success and (k_m < PL.NEVER).any():
            best = int(mates[int(np.argmin(k_m))])
            coll = rec if rec >= 0 and k[rec] < PL.NEVER and k[rec] <= k[best] + 15 else best
            mode = FM_RECEIVE
            # a teammate fumbles a ball nobody else wanted only via the miscontrol branch below
        else:
            sigma = PL.interception_time_sigma(self.models.pitch_control_params.tti_sigma)
            icp = PL.choose_interceptor(k_o, opps, self.rng, sigma)
            k_rec = int(k[rec]) if rec >= 0 else PL.NEVER
            # a body in the lane right in front of the passer blocks it (no reaction time needed)
            if p.flight == FLIGHT_PASS and self.rng.random() < 0.6:
                seg = path[2:9]
                dl = np.linalg.norm(seg[None, :, :2] - w.pos[b, opps][:, None, :], axis=2)
                kb = np.argmin(dl, axis=1)
                lat = dl[np.arange(len(opps)), kb]
                cand = (lat < 0.9) & (seg[kb, 2] < 1.6) & ~sq.is_gk[opps]
                if cand.any():
                    jb = int(np.argmin(np.where(cand, lat, 99.0)))
                    icp = int(opps[jb])
                    k[icp] = int(kb[jb]) + 2
            if p.flight == FLIGHT_CLEAR:
                # clearances: first to reach it, either side
                cand = int(np.argmin(k))
                if k[cand] < PL.NEVER:
                    coll, mode = cand, FM_LOOSE
            elif icp >= 0 and (k[icp] <= k_rec + 6 or k_rec >= PL.NEVER):
                coll, mode = icp, FM_INTERCEPT
                r = self.rng.random()
                sub = SUB_WON if r < 0.62 else (SUB_IN_PLAY if r < 0.8 else (SUB_DEFLECT if r < 0.95 else SUB_LOST))
                if sq.is_gk[icp] and path[min(int(k[icp]), len(path) - 1), 2] > 1.4:
                    mode = FM_CLAIM
                    r = self.rng.random()
                    sub = CLAIM_CATCH if r < 0.62 else (CLAIM_PUNCH if r < 0.94 else CLAIM_FAIL)
            elif k_rec < PL.NEVER:
                coll, mode = rec, FM_MISCONTROL
            elif (k_m < PL.NEVER).any():
                coll, mode = int(mates[int(np.argmin(k_m))]), FM_MISCONTROL
            elif (k_o < PL.NEVER).any():
                coll, mode = int(opps[int(np.argmin(k_o))]), FM_INTERCEPT
                sub = SUB_WON
        self._set_flight(b, coll, mode, int(k[coll]) if coll >= 0 else PL.NEVER, path, sub)
        w.flight_offside[b] = bool(p.info.get("receiver_offside", False)) and coll == rec and mode == FM_RECEIVE
        w.flight_aerial[b] = coll >= 0 and path[min(int(k[coll]), len(path) - 1), 2] > PASS_LOW_MAX_Z
        self.flight_res[b] = {"path": path, "res": p.res, "row": p.row, "info": p.info, "kind": p.kind}

    def _set_flight(self, b: int, coll: int, mode: int, k: int, path: F, sub: int) -> None:
        w = self.w
        w.collector[b] = coll
        w.flight_mode[b] = mode
        w.flight_sub[b] = sub
        if coll >= 0 and k < PL.NEVER:
            kk = min(k, len(path) - 1)
            w.collect_tick[b] = int(w.tick[b]) + kk
            w.collect_point[b] = path[kk]
        else:
            w.collect_tick[b] = -1
            w.collector[b] = -1
            if mode != FM_SAVE:
                w.flight_mode[b] = FM_FREE

    def _plan_shot(self, p: Pending, paths: PL.Paths, j: int, mc: IA) -> None:
        w, sq = self.w, self.sq
        b = p.b
        team = int(sq.team[p.kicker])
        s = self.sign[b, team]
        gk = int(self._gk[1 - team])
        path, vpath = paths.xs[:, j], paths.vs[:, j]
        # blockers: outfield defenders who can get to the shot's path in its first stretch
        opps = np.arange((1 - team) * N_ON, (2 - team) * N_ON)
        outfield = opps[~sq.is_gk[opps]]
        k_all = self._reach(b, path, vpath, exclude=p.kicker, exclude_until=99)
        # blockers: outfield defenders whose body is already (nearly) in the shot's path - no reaction needed
        seg = path[2:22]
        near = np.linalg.norm(seg[None, :, :2] - w.pos[b, outfield][:, None, :], axis=2)
        kk = np.argmin(near, axis=1)
        lateral = near[np.arange(len(outfield)), kk]
        low = seg[kk, 2] < 1.9
        p_each = np.where((lateral < 1.3) & low, 0.7 * np.exp(-lateral / 0.7), 0.0)
        p_block = 1.0 - float(np.prod(1.0 - p_each))
        k_all[outfield] = np.where(p_each > 0, kk + 2, k_all[outfield])
        # on-target / psxg calibration over the execution noise (Monte Carlo on the same physics)
        cross = self._goal_crossings(paths, np.concatenate([[j], mc]), s)
        on, ey, ez, spd = cross
        psx = self._psxg(b, team, ey, ez, spd, p)
        e_on = float(np.mean(on[1:] * psx[1:])) if len(mc) else float(on[0] * psx[0])
        target_p = p.p_success
        kfac = target_p / max((1.0 - p_block) * e_on, 1e-6)
        r = self.rng.random()
        info = p.info
        info["psxg"] = float(psx[0]) if on[0] else None
        info["p_block"] = p_block
        if r < p_block and p_each.max() > 0:
            blocker = int(outfield[int(self.rng.choice(len(outfield), p=p_each / p_each.sum()))])
            self._set_flight(b, blocker, FM_BLOCK, int(k_all[blocker]), path, 0)
            info["planned"] = "blocked"
        elif on[0]:
            p_goal = float(np.clip(psx[0] * kfac, 0.0, 0.985))
            if self.rng.random() < p_goal:
                self._set_flight(b, -1, FM_FREE, PL.NEVER, path, 0)
                info["planned"] = "goal"
                self._keeper_dive(b, gk, path, reach=False)
            else:
                ksave = self._save_tick(b, gk, path)
                self._set_flight(b, gk, FM_SAVE, ksave, path, self._save_kind(b, gk, vpath[ksave]))
                info["planned"] = "saved"
                self._keeper_dive(b, gk, path, reach=True, k=ksave)
        else:
            self._set_flight(b, -1, FM_FREE, PL.NEVER, path, 0)
            info["planned"] = "off_target"
            if abs(ey[0]) < HALF_GOAL + 1.5 and ez[0] < GOAL_HEIGHT + 0.6:
                self._keeper_dive(b, gk, path, reach=False)
        info["kfac"] = kfac
        self.flight_res[b] = {"path": path, "res": p.res, "row": p.row, "info": info, "kind": p.kind}

    def _goal_crossings(self, paths: PL.Paths, cols: IA, s: float) -> tuple[BA, F, F, F]:
        xs, vs = paths.xs[:, cols], paths.vs[:, cols]
        gx = HALF_LENGTH * s
        past = xs[..., 0] * s >= HALF_LENGTH
        has = past.any(axis=0)
        k = np.argmax(past, axis=0)
        c = np.arange(len(cols))
        y = xs[k, c, 1] * s
        z = xs[k, c, 2]
        spd = np.linalg.norm(vs[k, c], axis=1)
        # "on target" = the ball ends up in the net on its own (post-and-in included)
        on = has & (paths.goal_k[cols] < PL.NEVER) & (paths.goal_side[cols] * s > 0)
        y = np.clip(y, -(HALF_GOAL - 0.15), HALF_GOAL - 0.15)
        z = np.clip(z, BALL_RADIUS, GOAL_HEIGHT - 0.1)
        _ = gx
        return on, y, z, spd

    def _psxg(self, b: int, team: int, ey: F, ez: F, spd: F, p: Pending) -> F:
        w = self.w
        s = self.sign[b, team]
        gk = int(self._gk[1 - team])
        n = len(ey)
        origin = w.flight_origin[b]
        dist = float(np.hypot(HALF_LENGTH - origin[0] * s, origin[1] * s))
        f = PostShotFeatures(
            end_y=ey,
            end_z=ez,
            speed_mps=spd,
            distance_m=np.full(n, dist),
            gk_x=np.full(n, w.pos[b, gk, 0] * s),
            gk_y=np.full(n, w.pos[b, gk, 1] * s),
            gk_dive=np.full(n, 50.0),
        )
        return np.asarray(self.models.psxg(f), dtype=np.float64)

    def _save_tick(self, b: int, gk: int, path: F) -> int:
        """Path index where the keeper meets the shot: where it passes his body plane (or the goal line)."""
        w = self.w
        team = int(self.sq.team[gk])
        s = self.sign[b, team]
        gk_x_t = w.pos[b, gk, 0] * s
        xt = path[:, 0] * s
        reach = xt <= gk_x_t + 0.3
        k = int(np.argmax(reach)) if reach.any() else len(path) - 1
        return max(k, 2)

    def _save_kind(self, b: int, gk: int, v: F) -> int:
        sq = self.sq
        speed = float(np.linalg.norm(v))
        handling = sq.attr["gk_handling"][gk]
        p_catch = float(sigmoid(np.array(1.2 + 0.05 * (handling - 70) - 0.13 * (speed - 16)))) * 0.85
        r = self.rng.random()
        if r < p_catch:
            return SAVE_CATCH
        r = self.rng.random()
        if r < 0.04:
            return SAVE_TO_POST
        if r < 0.4:
            return SAVE_PARRY_OUT
        if r < 0.93:
            return SAVE_PARRY_SAFE
        return SAVE_PARRY_DANGER

    def _keeper_dive(self, b: int, gk: int, path: F, reach: bool, k: int | None = None) -> None:
        w, sq = self.w, self.sq
        if k is None:
            k = self._save_tick(b, gk, path)
        k = min(k, len(path) - 1)
        pt = path[k]
        lateral = pt[1] - w.pos[b, gk, 1]
        t_arr = max(k * DT, 0.15)
        if abs(lateral) < 0.9 and pt[2] < 1.9:
            return
        side_right = (lateral * self.sign[b, sq.team[gk]]) < 0
        reach_m = float(np.clip(abs(lateral) - (0.5 if reach else -0.3), 0.0, 3.0))
        dur = float(np.clip(t_arr, 0.3, 0.9))
        w.dive_vel[b, gk] = (0.0, np.sign(lateral) * reach_m / dur)
        w.dive_until[b, gk] = w.t[b] + dur
        w.vz[b, gk] = 1.6 if pt[2] > 0.6 else 0.9
        self._set_anim(b, gk, "gk_dive_right" if side_right else "gk_dive_left")
        w.stun[b, gk] = min(t_arr + 0.9, 1.6)

    # ------------------------------------------------------------------------------------------------------
    # movement
    # ------------------------------------------------------------------------------------------------------
    def _move_players(self) -> None:
        w, sq = self.w, self.sq
        if int(w.tick[0]) % 3 == 0 or self.keyframe.any():
            self._compute_targets()
        tgt, cap = self.targets, self.caps
        stunned = w.stun > 0
        cap = np.where(stunned, np.minimum(cap, 0.8), cap)
        vmax = effective_max_speed(sq.max_speed[None, :], w.stamina)
        # urgent movers (ball, pressing, runs) respond fast; shape players ease into position
        tau = np.where(cap >= 4.5, 0.3, 0.7)
        new_pos, new_vel = steer(w.pos, w.vel, tgt, cap, vmax, sq.max_accel[None, :], sq.max_decel[None, :], tau=tau)
        diving = w.dive_until > w.t[:, None]
        if diving.any():
            new_vel = np.where(diving[..., None], w.dive_vel, new_vel)
            new_pos = np.where(diving[..., None], w.pos + w.dive_vel * DT, new_pos)
        w.pos, w.vel = new_pos, new_vel
        speed = np.linalg.norm(w.vel, axis=-1)
        # facing: ball when slow, movement when fast; owners face where they carry
        to_ball = w.ball_pos[:, None, :2] - w.pos
        ball_ang = np.arctan2(to_ball[..., 1], to_ball[..., 0])
        mov_ang = np.arctan2(w.vel[..., 1], w.vel[..., 0])
        desired = np.where(speed > 3.0, mov_ang, ball_ang)
        owners = w.owner >= 0
        w.facing = turn_toward(w.facing, desired, speed)
        if owners.any():
            # the ball carrier faces where he carries the ball; standing, he shapes toward his next action
            bi = np.nonzero(owners)[0]
            o = w.owner[bi]
            sp = speed[bi, o]
            to_t = w.carry_target[bi] - w.pos[bi, o]
            intent = np.arctan2(to_t[:, 1], to_t[:, 0])
            f0 = w.facing[bi, o]
            diff = np.arctan2(np.sin(intent - f0), np.cos(intent - f0))
            shaped = f0 + np.clip(diff, -CARRIER_TURN_RATE * DT, CARRIER_TURN_RATE * DT)
            w.facing[bi, o] = np.where(sp >= 1.0, mov_ang[bi, o], shaped)
        w.stamina = drain_stamina(w.stamina, speed, sq.attr["stamina"][None, :])
        w.z, w.vz = step_jump(w.z, w.vz)
        w.stun = np.maximum(w.stun - DT, 0.0)
        # carry bookkeeping
        if owners.any():
            bi = np.nonzero(owners & (w.ball_state == BS_OWNED))[0]
            o = w.owner[bi]
            w.carry_path[bi] += speed[bi, o] * DT
            w.carry_max_speed[bi] = np.maximum(w.carry_max_speed[bi], speed[bi, o])
        # referee follows play at a diagonal offset
        ref_t = w.ball_pos[:, :2] + np.array([-9.0, 11.0]) * np.where(w.ball_pos[:, 1:2] > 0, -1.0, 1.0)
        ref_t[:, 1] = np.clip(ref_t[:, 1], -30, 30)
        rp, rv = steer(
            w.ref_pos, w.ref_vel, ref_t, np.full(w.b, 6.5), np.full(w.b, 7.5), np.full(w.b, 3.5), np.full(w.b, 5.0)
        )
        w.ref_pos, w.ref_vel = rp, rv

    def _compute_targets(self) -> None:
        w, sq = self.w, self.sq
        b_all = np.arange(w.b)
        ref = w.ball_pos[:, :2].copy()
        flying = (w.ball_state == BS_FLIGHT) & (w.collector >= 0)
        ref[flying] = 0.5 * ref[flying] + 0.5 * w.collect_point[flying, :2]
        poss = np.where(w.poss_team >= 0, w.poss_team, w.last_touch_team)
        tgt, cap = shape_targets(w, sq, self.sign, ref, poss, self.styles)
        # team shape reacts smoothly (players do not twitch with every ball movement)
        if self._shape is None or self._shape.shape != tgt.shape:
            self._shape = tgt.copy()
        else:
            self._shape += SHAPE_SMOOTHING * (tgt - self._shape)
        tgt = self._shape.copy()
        tgt, cap = goal_side(w, sq, self.sign, poss, tgt, cap)
        tgt, cap = box_marking(w, sq, self.sign, poss, tgt, cap, n_pairs=6)
        tgt, cap = lane_blocker(w, sq, self.sign, poss, tgt, cap)
        t_now = w.t[:, None]
        # runs
        running = w.run_until > t_now
        tgt = np.where(running[..., None], w.run_target, tgt)
        cap = np.where(running, 0.93 * sq.max_speed[None, :], cap)
        # owners carry
        owned = (w.ball_state == BS_OWNED) & (w.owner >= 0)
        if owned.any():
            bi = np.nonzero(owned)[0]
            o = w.owner[bi]
            tgt[bi, o] = w.carry_target[bi]
            cap[bi, o] = w.carry_speed[bi]
            hands = w.hands[bi]
            if hands.any():
                bh, oh = bi[hands], o[hands]
                s = self.sign[bh, sq.team[oh]]
                tgt[bh, oh, 0] = -(HALF_LENGTH - 12.0) * s
                tgt[bh, oh, 1] = w.pos[bh, oh, 1] * 0.5
                cap[bh, oh] = 2.0
            self._pressers(bi, o, tgt, cap)
        # collectors
        coll = (w.ball_state == BS_FLIGHT) & (w.collector >= 0)
        if coll.any():
            bi = np.nonzero(coll)[0]
            c = w.collector[bi]
            cp = w.collect_point[bi, :2]
            tgt[bi, c] = cp
            # tight on time: run through the meeting point instead of braking into it
            to_cp = cp - w.pos[bi, c]
            dist = np.linalg.norm(to_cp, axis=1)
            t_left = (w.collect_tick[bi] - w.tick[bi]) * DT
            tight = (dist / np.maximum(sq.max_speed[c], 1.0) + 0.4 > t_left) & (dist > 3.0)
            through = cp + 1.0 * to_cp / np.maximum(dist, 1e-6)[:, None]
            tgt[bi, c] = np.where(tight[:, None], through, tgt[bi, c])
            # once the meeting tick has come, chase the ball itself (with a little lead)
            overdue = w.tick[bi] >= w.collect_tick[bi]
            chase = w.ball_pos[bi, :2] + 0.25 * w.ball_vel[bi, :2]
            tgt[bi, c] = np.where(overdue[:, None], chase, tgt[bi, c])
            cap[bi, c] = sq.max_speed[c]
        # loose balls without a planned collector: nearest of each team chases
        loose = (w.ball_state == BS_FLIGHT) & (w.collector < 0)
        if loose.any():
            bi = np.nonzero(loose)[0]
            ahead = w.ball_pos[bi, :2] + 0.6 * w.ball_vel[bi, :2]
            for t in (0, 1):
                idx = np.arange(t * N_ON, (t + 1) * N_ON)
                d = np.linalg.norm(w.pos[bi][:, idx] - ahead[:, None, :], axis=2)
                d = np.where(sq.is_gk[idx][None, :], d + 15.0, d)
                j = idx[np.argmin(d, axis=1)]
                tgt[bi, j] = ahead
                cap[bi, j] = sq.max_speed[j]
        # dead balls: set-piece positions
        dead = w.ball_state == BS_DEAD
        if dead.any():
            for b in np.nonzero(dead)[0]:
                rc = self.restart_cache[b]
                if rc is not None:
                    tgt[b] = rc
                    d = np.linalg.norm(rc - w.pos[b], axis=1)
                    cap[b] = np.clip(0.8 + 0.35 * d, 1.0, 5.5)
                    taker = int(w.restart_taker[b])
                    if taker >= 0:
                        cap[b, taker] = min(6.5, float(sq.max_speed[taker]))
                else:
                    tgt[b] = w.pos[b]
                    cap[b] = 1.0
                if w.celebrate_until[b] > w.t[b]:
                    scorer = int(w.last_touch[b])
                    if scorer >= 0:
                        team = sq.team[scorer]
                        corner = np.array([HALF_LENGTH - 2.0, np.sign(w.pos[b, scorer, 1] + 1e-6) * (HALF_WIDTH - 2)])
                        corner[0] *= self.sign[b, team]
                        mates = np.arange(team * N_ON, (team + 1) * N_ON)
                        tgt[b, mates] = corner
                        cap[b, mates] = np.minimum(5.0, sq.max_speed[mates])
                        tgt[b, scorer] = corner
                        cap[b, scorer] = 6.5
        self.targets, self.caps = tgt, cap
        _ = b_all

    def _pressers(self, bi: IA, o: IA, tgt: F, cap: F) -> None:
        w, sq = self.w, self.sq
        team = sq.team[o]
        for t in (0, 1):
            m = team == 1 - t  # defending team t
            if not m.any():
                continue
            bb, oo = bi[m], o[m]
            idx = np.arange(t * N_ON, (t + 1) * N_ON)
            dpos = w.pos[bb][:, idx]
            opos = w.pos[bb, oo]
            d = np.linalg.norm(dpos - opos[:, None, :], axis=2)
            # the keeper rushes out only to an attacker through on goal
            s = self.sign[bb, t]
            o_x_t = opos[:, 0] * s
            gk_ok = (o_x_t < -36.0) & (np.abs(opos[:, 1]) < 16)
            # through on goal: the keeper comes out to meet the carrier whoever else is near
            gk = idx[sq.is_gk[idx]][0]
            rush = (o_x_t < -38.0) & (np.abs(opos[:, 1]) < 12.0)
            if rush.any():
                tgt[bb[rush], gk] = opos[rush] + 0.2 * w.vel[bb[rush], oo[rush]]
                cap[bb[rush], gk] = sq.max_speed[gk]
            d = np.where(sq.is_gk[idx][None, :] & ~gk_ok[:, None], 99.0, d)
            order = np.argsort(d, axis=1)
            first = idx[order[:, 0]]
            dist1 = d[np.arange(len(bb)), order[:, 0]]
            press = self.styles[t].press
            zone = (o_x_t < 10.0 + 25.0 * press) | (dist1 < 9.0)
            approach = opos + 0.35 * w.vel[bb, oo]
            dirv = approach - w.pos[bb, first]
            dn = np.maximum(np.linalg.norm(dirv, axis=1, keepdims=True), 1e-6)
            # close enough to challenge (keepers go straight for it)
            stand = approach - dirv / dn * np.where(sq.is_gk[first], 0.0, 0.45)[:, None]
            sel = zone & (dist1 < 22.0)
            tgt[bb[sel], first[sel]] = stand[sel]
            cap[bb[sel], first[sel]] = np.where(dist1[sel] > 4.0, sq.max_speed[first[sel]] * 0.95, 3.0)
            danger = o_x_t < -22.0  # near our goal everybody gets tight to the ball
            if press > 0.55 or danger.any():
                second = idx[order[:, 1]]
                dist2 = d[np.arange(len(bb)), order[:, 1]]
                sel2 = zone & (dist2 < 14.0) & ~sq.is_gk[second] & ((press > 0.55) | danger)
                lane = 0.6 * opos + 0.4 * w.pos[bb, second]
                tgt[bb[sel2], second[sel2]] = lane[sel2]
                cap[bb[sel2], second[sel2]] = sq.max_speed[second[sel2]] * 0.8

    # ------------------------------------------------------------------------------------------------------
    # ball
    # ------------------------------------------------------------------------------------------------------
    def _move_ball(self) -> None:
        w = self.w
        owned = (w.ball_state == BS_OWNED) & (w.owner >= 0)
        if owned.any():
            bi = np.nonzero(owned)[0]
            o = w.owner[bi]
            f = w.facing[bi, o]
            lead = np.where(w.hands[bi], 0.25, 0.45)
            want = np.column_stack(
                [
                    w.pos[bi, o, 0] + lead * np.cos(f),
                    w.pos[bi, o, 1] + lead * np.sin(f),
                    np.where(w.hands[bi], 1.0, BALL_RADIUS),
                ]
            )
            first_touch = w.t[bi] - w.receive_t[bi] < 0.5 * DT
            limit = (np.linalg.norm(w.vel[bi, o], axis=1) + np.where(first_touch, 20.0, 9.0)) * DT
            self._glide(bi, want, limit)
            poss = w.poss_team[bi]
            ok = poss >= 0
            w.poss_path[bi[ok]] += np.linalg.norm(w.vel[bi[ok], o[ok]], axis=1) * DT
        fl = w.ball_state == BS_FLIGHT
        self._wood = np.zeros(w.b, dtype=bool)
        if fl.any():
            bi = np.nonzero(fl)[0]
            pos, vel, spin = w.ball_pos[bi], w.ball_vel[bi], w.ball_spin[bi]
            flags = step_ball(pos, vel, spin)
            w.ball_pos[bi], w.ball_vel[bi], w.ball_spin[bi] = pos, vel, spin
            w.flight_peak[bi] = np.maximum(w.flight_peak[bi], pos[:, 2])
            self._wood[bi] = flags.woodwork
            poss = w.poss_team[bi]
            ok = poss >= 0
            w.poss_path[bi[ok]] += np.linalg.norm(vel[ok, :2], axis=1) * DT
        dead = w.ball_state == BS_DEAD
        if dead.any():
            # a dead ball rolls on, then is carried back to where play restarts (never teleported)
            rolling = dead & ((w.t - self._dead_since) < 1.2) & (np.linalg.norm(w.ball_vel, axis=1) > 0.3)
            if rolling.any():
                bi = np.nonzero(rolling)[0]
                pos, vel, spin = w.ball_pos[bi], w.ball_vel[bi], w.ball_spin[bi]
                step_ball(pos, vel, spin)
                vel *= 0.9
                w.ball_pos[bi], w.ball_vel[bi], w.ball_spin[bi] = pos, vel, spin
            carry = dead & ~rolling
            if carry.any():
                bi = np.nonzero(carry)[0]
                want = np.zeros((len(bi), 3))
                for k, b in enumerate(bi):
                    want[k] = self._dead_ball_spot(int(b))
                self._glide(bi, want, np.full(len(bi), 7.0 * DT))
                w.ball_spin[bi] = 0.0

    def _glide(self, bi: IA, want: F, limit: F) -> None:
        """Move held / retrieved balls toward ``want`` no faster than ``limit`` per tick; velocity = displacement."""
        w = self.w
        delta = want - w.ball_pos[bi]
        n = np.linalg.norm(delta, axis=1)
        scale = np.minimum(1.0, limit / np.maximum(n, 1e-9))
        step = delta * scale[:, None]
        w.ball_pos[bi] = w.ball_pos[bi] + step
        w.ball_vel[bi] = step / DT

    def _dead_ball_spot(self, b: int) -> F:
        w = self.w
        if w.restart[b] < 0:
            if w.status[b] == ST["half_time"]:
                return np.array([0.0, 0.0, BALL_RADIUS])
            return np.asarray(w.ball_pos[b], dtype=np.float64).copy()
        rp = w.restart_pos[b]
        taker = int(w.restart_taker[b])
        if w.restart[b] == R_THROW_IN and taker >= 0 and np.linalg.norm(w.pos[b, taker] - rp) < 1.5:
            f = w.facing[b, taker]
            return np.array([w.pos[b, taker, 0] + 0.3 * np.cos(f), w.pos[b, taker, 1] + 0.3 * np.sin(f), 2.0])
        return np.array([rp[0], rp[1], BALL_RADIUS])

    # ------------------------------------------------------------------------------------------------------
    # interactions
    # ------------------------------------------------------------------------------------------------------
    def _interactions(self) -> None:
        w = self.w
        sq = self.sq
        fl = (w.ball_state == BS_FLIGHT) & ~self.done
        if fl.any():
            # vectorized pre-filter: only elements where something can happen this tick go through Python
            bp = w.ball_pos
            out = (np.abs(bp[:, 0]) > HALF_LENGTH + BALL_RADIUS) | (np.abs(bp[:, 1]) > HALF_WIDTH + BALL_RADIUS)
            c = w.collector
            has_c = c >= 0
            rows = np.arange(w.b)
            d = np.linalg.norm(bp[:, :2] - w.pos[rows, np.maximum(c, 0)], axis=1)
            due = w.tick >= w.collect_tick - 2
            near = has_c & due & (d <= CONTROL_RADIUS + 1.3)
            late = has_c & (w.tick - w.collect_tick > 10)
            speed = np.linalg.norm(w.ball_vel, axis=1)
            since = w.t - w.flight_t0
            speed_all = np.linalg.norm(w.ball_vel, axis=1)
            shot_free = (w.flight_mode == FM_FREE) & (w.flight_kind == FLIGHT_SHOT) & (speed_all > 3.0)
            loose = ~has_c & ~shot_free & ((speed < 9.0) | ((since > 0.6) & (w.tick % 6 == 0)))
            loose &= w.t >= self._replan_after
            flagged = fl & (out | self._wood | near | late | loose)
            for b in np.nonzero(flagged)[0]:
                self._flight_events(int(b))
        owned = (w.ball_state == BS_OWNED) & ~self.done
        if owned.any():
            team = sq.team[np.maximum(w.owner, 0)]
            db = np.linalg.norm(w.pos - w.ball_pos[:, None, :2], axis=2)
            opp = sq.team[None, :] != team[:, None]
            fail_due = owned & (w.t >= self._carry_fail_t) & (w.owner == self._carry_who)
            for b in np.nonzero(fail_due)[0]:
                self._carry_lost(int(b))
            contest = owned & (opp & (db < CONTROL_RADIUS)).any(axis=1)
            contest &= ~((w.t < self._carry_safe_until) & (w.owner == self._carry_who))
            contest &= w.ball_state == BS_OWNED
            if contest.any():
                self._tackles(np.nonzero(contest)[0])
            for b in np.nonzero(owned & (w.ball_state == BS_OWNED))[0]:
                self._owned_bounds(int(b))
        if int(w.tick[0]) % 12 == 0:
            self._runs()

    def _flight_events(self, b: int) -> None:
        w, sq = self.w, self.sq
        bp = w.ball_pos[b]
        # goal
        if abs(bp[0]) > HALF_LENGTH + BALL_RADIUS:
            if abs(bp[1]) < HALF_GOAL and bp[2] < GOAL_HEIGHT:
                self._goal(b, 0 if bp[0] * self.sign[b, 0] > 0 else 1)
            else:
                self._out(b, "goal_line")
            return
        if abs(bp[1]) > HALF_WIDTH + BALL_RADIUS:
            self._out(b, "touchline")
            return
        if self._wood[b]:
            self._woodwork(b)
            return
        c = int(w.collector[b])
        tick = int(w.tick[b])
        if c >= 0:
            d = float(np.linalg.norm(bp[:2] - w.pos[b, c]))
            save = w.flight_mode[b] == FM_SAVE
            # a touch needs the ball within reach: 1 m on the ground (a keeper's dive reaches further for parries)
            reach = CONTROL_RADIUS + (1.2 if save else 0.0)
            reach_z = sq.jump_reach[c] + (0.3 if save else 0.0)
            due = tick >= int(w.collect_tick[b]) - 2
            # a running player takes the ball in front of him, not after overrunning it
            v = w.vel[b, c]
            sp = float(np.linalg.norm(v))
            # projection of the ball on his running direction: the first touch can take it across his body
            ahead = float(np.dot(bp[:2] - w.pos[b, c], v)) / max(sp, 1e-6) > -0.35
            in_front = ahead or sp < 1.5 or save
            if due and d <= reach and bp[2] <= reach_z and in_front:
                self._collect(b, c)
                return
            late = tick - int(w.collect_tick[b])
            if late > (10 if save else 25):
                self.diag[f"late_mode_{int(w.flight_mode[b])}"] += 1
                if save:
                    self._miss_save(b)
                else:
                    self._replan_loose(b, exclude=-1)
            return
        # nobody planned: replan when it slows down or periodically
        speed = float(np.linalg.norm(w.ball_vel[b]))
        since = w.t[b] - w.flight_t0[b]
        if w.flight_mode[b] == FM_FREE and w.flight_kind[b] == FLIGHT_SHOT:
            if speed > 3.0:
                return  # goal / out decides it
            # a shot that died before reaching the goal line: it missed, and the ball is there for anyone
            self._shot_end(b, "off_target", self._extrapolate_goal_plane(b))
            w.flight_kind[b] = FLIGHT_LOOSE
        if (speed < 9.0 or (since > 0.6 and tick % 6 == 0)) and w.t[b] >= self._replan_after[b]:
            self._replan_loose(b, exclude=int(w.last_touch[b]) if since < 0.4 else -1)

    def _miss_save(self, b: int) -> None:
        """The keeper could not get to a shot he was meant to save: it goes on under physics."""
        w = self.w
        w.collector[b] = -1
        w.flight_mode[b] = FM_FREE

    def _replan_loose(self, b: int, exclude: int) -> None:
        """Queue a loose-ball re-plan; all of a tick's re-plans share one batched path prediction."""
        self._replan_queue[b] = exclude

    def _flush_replans(self) -> None:
        if not self._replan_queue:
            return
        queue, self._replan_queue = self._replan_queue, {}
        w = self.w
        bs = np.array(list(queue), dtype=np.int64)
        paths = PL.predict_paths(w.ball_pos[bs], w.ball_vel[bs], w.ball_spin[bs], 60)
        for j, b in enumerate(bs):
            self._replan_one(int(b), queue[int(b)], paths, j)
            if w.collector[b] < 0:
                self._replan_after[b] = w.t[b] + 0.4  # nobody can get there yet: look again shortly

    def _replan_one(self, b: int, exclude: int, paths: PL.Paths, j: int) -> None:
        w = self.w
        if w.ball_state[b] != BS_FLIGHT:
            return
        path, vpath = paths.xs[:, j], paths.vs[:, j]
        out_k = int(min(paths.out_k[j], paths.goal_k[j]))
        k = self._reach(b, path, vpath, exclude=exclude, exclude_until=10 if exclude >= 0 else 0, stop_k=out_k)
        if (k < PL.NEVER).any():
            kmin = int(k.min())
            near = np.nonzero(k <= kmin + 3)[0]
            # in their own box defenders are first to a loose ball (they are goal-side and expecting it)
            bp = path[0]
            defend = np.array(
                [abs(bp[0] - (-HALF_LENGTH * self.sign[b, self.sq.team[i]])) < 18.0 and abs(bp[1]) < 22.0 for i in near]
            )
            if defend.any() and not defend.all() and self.rng.random() < 0.75:
                near = near[defend]
            c = int(self.rng.choice(near))
            self._set_flight(b, c, FM_LOOSE, int(k[c]), path, 0)
            others = np.nonzero((k <= kmin + 8) & (self.sq.team != self.sq.team[c]))[0]
            w.flight_sub[b] = int(others[0]) + 100 if len(others) else 0
        else:
            w.collector[b] = -1
            w.flight_mode[b] = FM_FREE
        if w.flight_kind[b] == FLIGHT_SHOT:
            w.flight_kind[b] = FLIGHT_LOOSE
        elif w.flight_kind[b] == FLIGHT_PASS and w.flight_mode[b] in (FM_LOOSE, FM_FREE):
            self._finish_pass(b, "incomplete", None)
            w.flight_kind[b] = FLIGHT_LOOSE

    def _woodwork(self, b: int) -> None:
        w = self.w
        fr = self.flight_res[b]
        if w.flight_kind[b] == FLIGHT_SHOT and fr is not None and fr["info"].get("planned") == "goal":
            return  # off the post and in: the physics already says so
        if w.flight_kind[b] == FLIGHT_SHOT and w.shot_ev[b] >= 0:
            self._wood_shot[b] = int(w.shot_ev[b])
            q = self.logs[b].q(int(w.shot_ev[b]))
            q["outcome"] = "woodwork"
            q["on_target"] = False
            q.pop("psxg", None)
            self.logs[b].records[int(w.shot_ev[b])]["end_location"] = tuple(float(v) for v in w.ball_pos[b])
            w.shot_ev[b] = -1
        self._replan_loose(b, exclude=-1)
        w.flight_kind[b] = FLIGHT_LOOSE

    def _finish_pass(self, b: int, outcome: str, end: Any, receiver: int | None = None) -> None:
        w = self.w
        ev = int(w.flight_event[b])
        if ev < 0 or w.flight_kind[b] != FLIGHT_PASS:
            return
        rec = self.logs[b].records[ev]
        if rec["type"] != "pass":
            return
        q = rec["q"]
        q["outcome"] = outcome
        e = w.ball_pos[b] if end is None else end
        start = np.array(rec["location"])
        rec["end_location"] = (float(e[0]), float(e[1]), float(e[2]))
        length = float(np.linalg.norm(np.array(e[:2]) - start[:2]))
        q["length_m"] = length
        q["length_class"] = "short" if length < PASS_SHORT_MAX else ("medium" if length <= PASS_MEDIUM_MAX else "long")
        peak = float(w.flight_peak[b])
        q["peak_height_m"] = max(peak, 0.0)
        q["height"] = "ground" if peak <= PASS_GROUND_MAX_Z else ("low" if peak <= PASS_LOW_MAX_Z else "high")
        q["duration_s"] = float(w.t[b] - w.flight_t0[b])
        team = SIDE.index(rec["team"])
        s = self.sign[b, team]
        sx, sy, ex, ey = start[0] * s, start[1] * s, e[0] * s, e[1] * s
        q["is_switch"] = bool(abs(ey - sy) >= 34.0)
        d0 = np.hypot(HALF_LENGTH - sx, sy)
        d1 = np.hypot(HALF_LENGTH - ex, ey)
        prog = _progressive(sx, ex, d0, d1, ey)
        if q.get("restart") in ("goal_kick", "kick_off"):
            prog = False
        q["is_progressive"] = bool(prog)
        q["lines_broken"] = self._lines_broken(b, team, sx, ex)
        q["is_line_breaking"] = q["lines_broken"] > 0 and outcome == "complete"
        if q.get("restart") == "corner":
            q["corner_delivery"] = self._corner_delivery(b, team, start)
            for ri in rec["related"]:
                if self.logs[b].records[ri]["type"] == "restart":
                    self.logs[b].q(ri)["corner_delivery"] = q["corner_delivery"]
        w.flight_event[b] = -1

    def _lines_broken(self, b: int, team: int, sx: float, ex: float) -> int:
        w, sq = self.w, self.sq
        s = self.sign[b, team]
        opp = np.arange((1 - team) * N_ON, (2 - team) * N_ON)
        n = 0
        for line in (1, 2, 3):
            m = opp[sq.line[opp] == line]
            if len(m) == 0:
                continue
            lx = float(np.mean(w.pos[b, m, 0]) * s)
            if sx < lx < ex:
                n += 1
        return n

    def _corner_delivery(self, b: int, team: int, start: F) -> str:
        fr = self.flight_res[b]
        if fr is None:
            return "straight"
        path = fr["path"]
        s = self.sign[b, team]
        k = min(len(path) - 1, max(int((self.w.t[b] - self.w.flight_t0[b]) / DT), 2))
        seg = path[: k + 1, :2] * s
        a, e = seg[0], seg[-1]
        mid = seg[len(seg) // 2]
        chord_mid = 0.5 * (a + e)
        dev = mid[0] - chord_mid[0]  # toward the goal line is +x in the attacking frame
        if dev >= 1.0:
            return "inswinging"
        if dev <= -1.0:
            return "outswinging"
        return "straight"

    # --- collection ---------------------------------------------------------------------------------------
    def _collect(self, b: int, c: int) -> None:
        w, sq = self.w, self.sq
        self._wood_shot[b] = -1
        mode = int(w.flight_mode[b])
        team = int(sq.team[c])
        bp = w.ball_pos[b].copy()
        kind = int(w.flight_kind[b])
        self.keyframe[b] = True
        passer = int(w.flight_passer[b])
        pass_ev = int(w.flight_event[b]) if kind == FLIGHT_PASS else -1
        aerial = bp[2] > 1.0
        if aerial and self._aerial_duel(b, c):
            return
        if mode == FM_RECEIVE:
            if w.flight_offside[b]:
                self._offside(b, c, passer, pass_ev)
                return
            self._finish_pass(b, "complete", bp)
            if self._first_time(b, c, bp, aerial, pass_ev):
                return
            self._receive(b, c, bp, aerial, pass_ev, passer, clean=None)
            return
        if mode == FM_MISCONTROL:
            self._finish_pass(b, "incomplete", bp)
            rq = self._receipt(b, c, bp, aerial, pass_ev, passer, "incomplete", "miscontrol")
            mev = self._ev(b, "miscontrol", c, team, bp, {"aerial_won": False})
            self.logs[b].link(rq, mev)
            self._deflect(b, c, scale=0.45, spread=1.1)
            return
        if mode == FM_INTERCEPT:
            self._intercept(b, c, bp, kind, pass_ev, passer)
            return
        if mode == FM_CLAIM:
            self._claim(b, c, bp, pass_ev)
            return
        if mode == FM_SAVE:
            self._save(b, c, bp)
            return
        if mode == FM_BLOCK:
            self._block(b, c, bp)
            return
        # loose ball
        self._loose_collect(b, c, bp, aerial)

    def _aerial_duel(self, b: int, c: int) -> bool:
        """Header contest around the collector. Returns True if it ended in a foul (play stops)."""
        w, sq = self.w, self.sq
        team = int(sq.team[c])
        opp = np.arange((1 - team) * N_ON, (2 - team) * N_ON)
        d = np.linalg.norm(w.pos[b, opp] - w.pos[b, c], axis=1)
        d = np.where(sq.is_gk[opp], 99.0, d)
        j = int(opp[int(np.argmin(d))])
        if d.min() > 2.2:
            return False
        loc = w.ball_pos[b]
        e1 = self._ev(b, "duel", c, team, loc, {"opponent": sq.ids[j], "duel_type": "aerial", "outcome": "won"})
        e2 = self._ev(b, "duel", j, 1 - team, loc, {"opponent": sq.ids[c], "duel_type": "aerial", "outcome": "lost"})
        self.logs[b].link(e1, e2)
        w.aerial_won_t[b] = w.t[b]
        for i in (c, j):
            if w.z[b, i] == 0:
                w.vz[b, i] = jump_speed(sq.attr["jumping"][i : i + 1])[0]
        self._set_anim(b, j, "header_jumping")
        if self.rng.random() < 0.025:
            self._foul(b, j, c, "foul")
            return True
        return False

    def _receipt(
        self, b: int, c: int, bp: F, aerial: bool, pass_ev: int, passer: int, outcome: str, quality: str
    ) -> int:
        w, sq = self.w, self.sq
        team = int(sq.team[c])
        s = self.sign[b, team]
        theta = np.arctan2(np.sin(w.facing[b, c]), np.cos(w.facing[b, c]))
        theta_t = np.arctan2(np.sin(theta + (0 if s > 0 else np.pi)), np.cos(theta + (0 if s > 0 else np.pi)))
        orient = (
            "open"
            if abs(theta_t) < np.radians(60)
            else ("half_turn" if abs(theta_t) <= np.radians(120) else "back_to_goal")
        )
        _, near, _ = self._pressure(b, c)
        part = (
            "head" if bp[2] > 1.55 else ("chest" if bp[2] > 0.9 else ("left_foot" if sq.left_foot[c] else "right_foot"))
        )
        q = {
            "passer": sq.ids[passer] if passer >= 0 else None,
            "outcome": outcome,
            "touch_quality": quality,
            "body_orientation": orient,
            "first_touch_body_part": part,
            "pressure_distance_m": float(near),
            "aerial": bool(aerial),
            "pass_id": pass_ev if pass_ev >= 0 else None,
        }
        ev = self._ev(b, "ball_receipt", c, team, bp, q)
        if pass_ev >= 0:
            self.logs[b].q(pass_ev)["receipt_id"] = ev
            self.logs[b].link(pass_ev, ev)
        return ev

    def _receive(self, b: int, c: int, bp: F, aerial: bool, pass_ev: int, passer: int, clean: bool | None) -> None:
        w, sq = self.w, self.sq
        speed = float(np.linalg.norm(w.ball_vel[b]))
        press, _, _ = self._pressure(b, c)
        tech = sq.attr["technique"][c]
        p_clean = float(
            sigmoid(np.array(2.3 + 0.05 * (tech - 70) - 0.09 * max(speed - 11.0, 0) - 0.9 * aerial - 0.9 * press))
        )
        quality = "clean" if (clean if clean is not None else self.rng.random() < p_clean) else "loose"
        ev = self._receipt(b, c, bp, aerial, pass_ev, passer, "complete", quality)
        self._gain(b, c)
        w.receipt_ev[b] = ev
        self._set_anim(b, c, "receive_aerial" if aerial else "receive_ground", mirror=bool(sq.left_foot[c]))
        if quality == "loose":
            w.act_timer[b] += 0.35
            w.carry_speed[b] = 1.5
        self._mark_run_targeted(b, c)

    def _first_time(self, b: int, c: int, bp: F, aerial: bool, pass_ev: int) -> bool:
        """One-touch actions on arrival: headers and volleys at goal, flick-ons, cutback finishes, clearances."""
        w, sq = self.w, self.sq
        team = int(sq.team[c])
        s = self.sign[b, team]
        x_t, y_t = bp[0] * s, bp[1] * s
        dist = np.hypot(HALF_LENGTH - x_t, y_t)
        in_box = x_t >= BOX_X and abs(y_t) <= BOX_HALF_WIDTH
        z = bp[2]
        r = self.rng.random()
        kind = None
        if z > 1.0 and in_box and dist < 18.0 and z <= sq.jump_reach[c] + 0.1:
            kind = I_SHOT if r < 0.8 else I_PASS_PLAYER  # attack the cross: head it at goal
        elif z > 1.25:
            if in_box and dist < 15.0:
                kind = I_SHOT if r < 0.78 else I_PASS_PLAYER
            elif x_t < -20.0:
                kind = I_CLEARANCE if r < 0.7 else I_PASS_PLAYER
            elif z > 1.55:
                kind = I_PASS_PLAYER if r < 0.6 else None
        elif z > 0.35:
            if in_box and dist < 16.0 and r < 0.45:
                kind = I_SHOT
        elif in_box and dist < 14.0:
            cut = pass_ev >= 0 and self.logs[b].q(pass_ev).get("pass_type") in ("cutback", "cross")
            if r < (0.55 if cut else 0.22):
                kind = I_SHOT
        if kind is None and z > HEAD_ONLY_Z and not sq.is_gk[c]:
            kind = I_CLEARANCE if x_t < -20.0 else I_PASS_PLAYER  # too high to control: head it on
        if kind is None:
            return False
        # gain control nominally, then act immediately with a one-touch technique
        pq = self.logs[b].q(pass_ev) if pass_ev >= 0 else {}
        rq = self._receipt(b, c, bp, aerial, pass_ev, int(w.flight_passer[b]), "complete", "clean")
        _ = pq
        self._gain(b, c)
        w.receive_t[b] = w.t[b]
        w.receipt_ev[b] = rq
        w.ball_pos[b] = bp
        bounced = bool(0.12 < bp[2] < 0.7 and w.ball_vel[b, 2] > 0.3)  # rising off a bounce
        self._one_touch(b, c, kind, bp, bounced)
        return True

    def _one_touch(self, b: int, c: int, kind: int, bp: F, bounced: bool) -> None:
        w = self.w
        di = self.decider_input([b], [CTX_OPEN])
        from sim.decision import S_CLEAR, S_PASS0, S_SHOT, S_THROUGH0, common, enumerate_slots, score_table

        cm = common(self.sq, di)
        sl = enumerate_slots(self.sq, di, cm, self.rng)
        valid = np.zeros_like(sl.valid)
        if kind == I_SHOT:
            valid[0, S_SHOT] = True
        elif kind == I_CLEARANCE:
            valid[0, S_CLEAR] = True
            sl.kind[0, S_CLEAR] = I_CLEARANCE
        else:
            near = np.linalg.norm(sl.target[0, S_PASS0:S_THROUGH0, :2] - w.pos[b, c], axis=1) < 22.0
            valid[0, S_PASS0:S_THROUGH0] = near & sl.valid[0, S_PASS0:S_THROUGH0]
            if not valid.any():
                valid[0, S_CLEAR] = True
                sl.kind[0, S_CLEAR] = I_CLEARANCE
        sl.valid = valid
        # score with the real ball height so headers / volleys resolve as such
        di.ball = bp[None, :].copy()
        di.first_time = np.array([True])
        di.bounced = np.array([bounced])
        di.ball_vel = w.ball_vel[b : b + 1].copy()
        table = score_table(self.sq, self.models, self.styles, di, cm, sl, self.rng)
        slot = int(choose(table, self.rng)[0])
        if not table.valid[0, slot]:
            slot = int(np.argmax(table.valid[0]))
        self._snapshot(b)
        self._record_decision(b, 0, slot, table, di)
        self._execute_option(b, 0, slot, table, di, CTX_OPEN, None)

    def _loose_collect(self, b: int, c: int, bp: F, aerial: bool) -> None:
        w, sq = self.w, self.sq
        team = int(sq.team[c])
        contest = int(w.flight_sub[b]) - 100
        if (
            contest >= 0
            and sq.team[contest] != team
            and np.linalg.norm(w.pos[b, contest] - w.pos[b, c]) < 3.0
            and not aerial
        ):
            e1 = self._ev(
                b,
                "duel",
                c,
                team,
                bp,
                {
                    "opponent": sq.ids[contest],
                    "duel_type": "fifty_fifty",
                    "outcome": "won",
                    "possession_after": "tackler",
                },
            )
            e2 = self._ev(
                b,
                "duel",
                contest,
                1 - team,
                bp,
                {"opponent": sq.ids[c], "duel_type": "fifty_fifty", "outcome": "lost", "possession_after": "opponent"},
            )
            self.logs[b].link(e1, e2)
        if w.flight_kind[b] == FLIGHT_PASS:
            self._finish_pass(b, "incomplete", bp)
        if aerial and self._first_time(b, c, bp, aerial, -1):
            return
        offensive = bool(w.last_touch_team[b] == team)
        self._ev(b, "ball_recovery", c, team, bp, {"offensive": offensive, "recovery_failure": False})
        in_own_box = bp[0] * self.sign[b, team] <= -BOX_X and abs(bp[1]) <= BOX_HALF_WIDTH
        last_by_mate = w.last_touch_team[b] == team and w.last_touch[b] != c
        self._gain(b, c)
        if sq.is_gk[c] and in_own_box and not last_by_mate:
            w.hands[b] = True
            w.act_timer[b] = 2.0 + 2.5 * self.rng.random()
            self._set_anim(b, c, "gk_scoop")
        else:
            self._set_anim(b, c, "receive_aerial" if aerial else "receive_ground")

    def _deflect(self, b: int, c: int, scale: float, spread: float, toward: F | None = None) -> None:
        w = self.w
        v = w.ball_vel[b].copy()
        speed = float(np.linalg.norm(v[:2])) * scale + 2.0
        base = np.arctan2(v[1], v[0]) if toward is None else np.arctan2(toward[1], toward[0])
        ang = base + self.rng.normal(0, spread)
        # a defender's touch near his own goal does not steer the ball into it (it goes wide or back out)
        team = int(self.sq.team[c])
        own_x = -HALF_LENGTH * self.sign[b, team]
        to_goal = np.array([own_x - w.ball_pos[b, 0], -w.ball_pos[b, 1]])
        dist_goal = float(np.linalg.norm(to_goal))
        u = np.array([np.cos(ang), np.sin(ang)])
        if dist_goal < 25.0 and float(u @ to_goal) / max(dist_goal, 1e-6) > 0.3:
            u[0] = -u[0]
            ang = float(np.arctan2(u[1], u[0]))
        w.ball_vel[b] = (speed * np.cos(ang), speed * np.sin(ang), abs(self.rng.normal(1.5, 1.5)))
        w.ball_spin[b] = 0.0
        w.ball_pos[b, 2] = max(w.ball_pos[b, 2], BALL_RADIUS + 0.05)
        w.ball_state[b] = BS_FLIGHT
        w.owner[b] = -1
        w.last_touch[b] = c
        w.last_touch_team[b] = self.sq.team[c]
        w.flight_t0[b] = w.t[b]
        w.flight_kind[b] = FLIGHT_LOOSE
        w.flight_event[b] = -1
        w.flight_peak[b] = w.ball_pos[b, 2]
        self._replan_loose(b, exclude=c)

    def _intercept(self, b: int, c: int, bp: F, kind: int, pass_ev: int, passer: int) -> None:
        w, sq = self.w, self.sq
        team = int(sq.team[c])
        sub = int(w.flight_sub[b])
        if kind == FLIGHT_CLEAR:
            self._loose_collect(b, c, bp, bp[2] > 1.0)
            return
        near_passer = (
            passer >= 0
            and float(np.linalg.norm(w.pos[b, passer] - w.pos[b, c])) < 4.0
            and (w.t[b] - w.flight_t0[b] < 0.6)
        )
        ptype = self.logs[b].q(pass_ev).get("pass_type") if pass_ev >= 0 else None
        self._finish_pass(b, "intercepted", bp)
        if pass_ev >= 0:
            self.logs[b].q(pass_ev)["deflected"] = sub == SUB_DEFLECT
        if sub == SUB_DEFLECT:
            ev = self._ev(b, "deflection", c, team, bp, {"deflection_target": "cross" if ptype == "cross" else "pass"})
            self.logs[b].link(ev, pass_ev)
            self._deflect(b, c, scale=0.7, spread=0.35)
            return
        if near_passer:
            ev = self._ev(
                b,
                "block",
                c,
                team,
                bp,
                {
                    "block_type": "cross_block" if ptype == "cross" else "pass_block",
                    "deflection": False,
                    "blocked_event_id": pass_ev,
                },
            )
            self.logs[b].link(ev, pass_ev)
            if self.rng.random() < 0.03:
                self._foul(b, c, passer, "handball")
                return
            self._deflect(b, c, scale=0.35, spread=1.4)
            return
        if bp[2] > 1.25 and not sq.is_gk[c]:
            # headed away
            s = self.sign[b, team]
            if bp[0] * s < -10:
                ev = self._ev(
                    b,
                    "clearance",
                    c,
                    team,
                    bp,
                    {
                        "body_part": "head",
                        "outcome": "in_play_safe",
                        "aerial_won": bool(w.t[b] - w.aerial_won_t[b] < 0.3),
                    },
                )
                w.clear_ev[b] = ev
                self.logs[b].link(ev, pass_ev)
                self._set_anim(b, c, "header_jumping")
                self._deflect(b, c, scale=0.9, spread=0.5, toward=np.array([s, 0.0]))
                return
        if bp[2] > HEAD_ONLY_Z and not sq.is_gk[c]:
            ev = self._ev(
                b,
                "interception",
                c,
                team,
                bp,
                {"passer": sq.ids[passer] if passer >= 0 else None, "outcome": "won", "pass_id": pass_ev},
            )
            self.logs[b].link(ev, pass_ev)
            self._gain(b, c)
            self._one_touch(b, c, I_PASS_PLAYER, bp, False)
            return
        out = {SUB_WON: "won", SUB_IN_PLAY: "success_in_play", SUB_LOST: "lost"}[min(sub, SUB_LOST)]
        ev = self._ev(
            b,
            "interception",
            c,
            team,
            bp,
            {"passer": sq.ids[passer] if passer >= 0 else None, "outcome": out, "pass_id": pass_ev},
        )
        self.logs[b].link(ev, pass_ev)
        if sq.is_gk[c]:
            x_t = bp[0] * self.sign[b, team]
            if x_t > -HALF_LENGTH + 16.5 or abs(bp[1]) > BOX_HALF_WIDTH:
                gk_ev = self._ev(
                    b,
                    "gk_action",
                    c,
                    team,
                    bp,
                    {"action": "sweeper", "outcome": "clear" if sub else "claim", "body_part": "right_foot"},
                )
                self.logs[b].link(gk_ev, ev)
        if sub == SUB_WON:
            self._gain(b, c)
            self._set_anim(b, c, "receive_ground")
        elif sub == SUB_IN_PLAY:
            mates = np.arange(team * N_ON, (team + 1) * N_ON)
            near = mates[np.argsort(np.linalg.norm(w.pos[b, mates] - w.pos[b, c], axis=1))[1]]
            self._deflect(b, c, scale=0.6, spread=0.3, toward=w.pos[b, near] - w.pos[b, c])
        else:
            self._deflect(b, c, scale=0.5, spread=1.0)

    def _claim(self, b: int, c: int, bp: F, pass_ev: int) -> None:
        w, sq = self.w, self.sq
        team = int(sq.team[c])
        sub = int(w.flight_sub[b])
        self._finish_pass(b, "intercepted", bp)
        if sub == CLAIM_CATCH:
            ev = self._ev(
                b,
                "gk_action",
                c,
                team,
                bp,
                {"action": "claim", "outcome": "success", "body_part": "both_hands", "technique": "standing"},
            )
            self.logs[b].link(ev, pass_ev)
            self._gain(b, c, "from_keeper")
            w.hands[b] = True
            w.act_timer[b] = 2.0 + 3.0 * self.rng.random()
            self._set_anim(b, c, "gk_catch_high")
            w.vz[b, c] = 1.5
        elif sub == CLAIM_PUNCH:
            r = self.rng.random()
            out = "in_play_safe" if r < 0.5 else ("in_play_danger" if r < 0.8 else "out_of_play")
            ev = self._ev(b, "gk_action", c, team, bp, {"action": "punch", "outcome": out, "body_part": "both_hands"})
            self.logs[b].link(ev, pass_ev)
            s = self.sign[b, team]
            if out == "out_of_play":
                toward = np.array([0.0, np.sign(bp[1] + 1e-6)])
                self._deflect(b, c, scale=1.0, spread=0.15, toward=toward)
                w.ball_vel[b, :2] = toward * 14.0
            else:
                self._deflect(b, c, scale=0.8 if out == "in_play_safe" else 0.4, spread=0.4, toward=np.array([s, 0.0]))
            self._set_anim(b, c, "gk_catch_high")
        else:
            ev = self._ev(
                b, "gk_action", c, team, bp, {"action": "claim", "outcome": "fail", "body_part": "both_hands"}
            )
            self.logs[b].link(ev, pass_ev)
            self._deflect(b, c, scale=0.3, spread=2.0)

    def _shot_end(self, b: int, outcome: str, end: Any) -> int:
        w = self.w
        ev = int(w.shot_ev[b])
        if ev < 0:
            return -1
        rec = self.logs[b].records[ev]
        q = rec["q"]
        q["outcome"] = outcome
        q["on_target"] = outcome in ("goal", "saved", "saved_to_post")
        fr = self.flight_res[b]
        psxg = None if fr is None else fr["info"].get("psxg")
        if q["on_target"] and psxg is not None:
            q["psxg"] = float(np.clip(psxg, 0, 1))
        else:
            q.pop("psxg", None)
        rec["end_location"] = (float(end[0]), float(end[1]), float(max(end[2], 0.0)))
        q["peak_height_m"] = float(w.flight_peak[b])
        w.shot_ev[b] = -1
        return ev

    def _save(self, b: int, c: int, bp: F) -> None:
        w, sq = self.w, self.sq
        team = int(sq.team[c])
        sub = int(w.flight_sub[b])
        if sub == SAVE_CATCH and float(np.linalg.norm(bp[:2] - w.pos[b, c])) > CONTROL_RADIUS:
            sub = SAVE_PARRY_SAFE  # only a ball the keeper actually reaches with his body can be held
        s = self.sign[b, 1 - team]
        cross_pt = (HALF_LENGTH * s, bp[1], bp[2])
        out = {SAVE_CATCH: "saved", SAVE_TO_POST: "saved_to_post"}.get(sub, "saved")
        shot_ev = self._shot_end(b, "saved" if out == "saved" else out, cross_pt)
        lateral = abs(bp[1] - w.pos[b, c, 1])
        technique = "diving" if lateral > 0.9 or w.dive_until[b, c] > w.t[b] - 0.5 else "standing"
        body = "both_hands"
        if technique == "standing" and bp[2] < 0.45 and self.rng.random() < 0.3:
            body = "right_foot" if self.rng.random() < 0.6 else "left_foot"
        elif technique == "standing" and 0.6 < bp[2] < 1.4 and self.rng.random() < 0.25:
            body = "chest"
        gk_out = {
            SAVE_CATCH: "catch",
            SAVE_PARRY_SAFE: "parry_safe",
            SAVE_PARRY_DANGER: "parry_danger",
            SAVE_PARRY_OUT: "parry_out",
            SAVE_TO_POST: "to_post",
        }[sub]
        pos_before = "moving" if np.linalg.norm(w.vel[b, c]) > 1.2 else "set"
        if shot_ev >= 0:
            pos_before = self.logs[b].q(shot_ev).get("gk_position") or pos_before
        q = {
            "action": "save",
            "outcome": gk_out,
            "technique": technique,
            "body_part": body,
            "shot_id": shot_ev if shot_ev >= 0 else None,
            "position_before": pos_before,
            "dive_distance_m": float(lateral),
        }
        ev = self._ev(b, "gk_action", c, team, bp, q)
        if shot_ev >= 0:
            self.logs[b].q(shot_ev)["gk_event_id"] = ev
            self.logs[b].link(ev, shot_ev)
        self.flight_res[b] = None
        own = -s  # direction away from the keeper's goal
        if sub == SAVE_CATCH:
            # the keeper gathers the ball and lands with it: the dive ends here
            w.dive_until[b, c] = w.t[b]
            w.dive_vel[b, c] = 0.0
            w.vel[b, c] = 0.0
            self._gain(b, c, "from_keeper")
            w.hands[b] = True
            w.act_timer[b] = 2.5 + 3.0 * self.rng.random()
            if technique == "standing":
                self._set_anim(b, c, "gk_catch_high" if bp[2] > 1.2 else "gk_catch_low")
        elif sub == SAVE_PARRY_OUT:
            toward = np.array([0.25 * own, np.sign(bp[1] + 1e-6)])
            self._deflect(b, c, scale=0.5, spread=0.15, toward=toward)
            w.ball_vel[b, 0] = abs(w.ball_vel[b, 0]) * s * 0.3 + s * 2.0  # over / round the post
            w.ball_vel[b, 2] = 3.0
        elif sub == SAVE_TO_POST:
            self._deflect(b, c, scale=0.4, spread=0.2, toward=np.array([own, 0.0]))
        elif sub == SAVE_PARRY_SAFE:
            toward = np.array([own * 0.6, np.sign(bp[1] + 1e-6)])
            self._deflect(b, c, scale=0.45, spread=0.3, toward=toward)
        else:
            # pushed out wide of the goal, not back into the middle
            wide = np.sign(bp[1] + (1e-6 if self.rng.random() < 0.5 else -1e-6))
            self._deflect(b, c, scale=0.35, spread=0.35, toward=np.array([own * 0.5, wide]))

    def _block(self, b: int, c: int, bp: F) -> None:
        w, sq = self.w, self.sq
        team = int(sq.team[c])
        fr = self.flight_res[b]
        on_target = bool(fr and fr["info"].get("psxg") is not None)
        shot_ev = self._shot_end(b, "blocked", bp)
        if self.rng.random() < 0.3:
            ev = self._ev(b, "deflection", c, team, bp, {"deflection_target": "shot"})
            if shot_ev >= 0:
                self.logs[b].q(shot_ev)["deflected"] = True
        else:
            ev = self._ev(
                b,
                "block",
                c,
                team,
                bp,
                {"block_type": "shot_block", "deflection": False, "save_block": on_target, "blocked_event_id": shot_ev},
            )
        if shot_ev >= 0:
            self.logs[b].q(shot_ev)["block_event_id"] = ev
            self.logs[b].link(ev, shot_ev)
        self.flight_res[b] = None
        if self.rng.random() < 0.035:
            self._foul(b, c, int(w.flight_passer[b]), "handball")
            return
        self._deflect(b, c, scale=0.4, spread=0.9, toward=-w.ball_vel[b, :2])  # off the body, back out

    # --- out of play, goals, offside, fouls ----------------------------------------------------------------
    def _out(self, b: int, line: str) -> None:
        w, sq = self.w, self.sq
        bp = w.ball_pos[b].copy()
        last = int(w.last_touch[b])
        lt = int(w.last_touch_team[b]) if last >= 0 else -1
        if w.flight_kind[b] == FLIGHT_SHOT:
            self._shot_end(b, "off_target", self._extrapolate_goal_plane(b))
        self._finish_pass(b, "out_of_play", bp)
        ce = int(w.clear_ev[b])
        if ce >= 0:
            self.logs[b].q(ce)["outcome"] = "out_of_play"
            w.clear_ev[b] = -1
        x = float(np.clip(bp[0], -HALF_LENGTH, HALF_LENGTH))
        y = float(np.clip(bp[1], -HALF_WIDTH, HALF_WIDTH))
        if line == "touchline":
            team = 1 - lt if lt >= 0 else 0
            r, pos, delay = R_THROW_IN, np.array([x, np.sign(y) * HALF_WIDTH]), 5.0 + 4.0 * self.rng.random()
        else:
            # which goal line, and who defends it
            defending = 0 if bp[0] * self.sign[b, 0] < 0 else 1
            if lt == defending:
                team = 1 - defending
                r = R_CORNER
                pos = np.array([np.sign(bp[0]) * (HALF_LENGTH - 0.3), np.sign(bp[1] + 1e-6) * (HALF_WIDTH - 0.3)])
                delay = 14.0 + 6.0 * self.rng.random()
            else:
                team = defending
                r = R_GOAL_KICK
                pos = np.array([np.sign(bp[0]) * (HALF_LENGTH - 5.5), np.sign(bp[1] + 1e-6) * 4.0])
                delay = 9.0 + 5.0 * self.rng.random()
        q = {
            "line": line,
            "last_touch_team": SIDE[lt] if lt >= 0 else None,
            "last_touch_player": sq.ids[last] if last >= 0 else None,
            "result_restart_type": RESTART_NAMES[r],
        }
        ev = self._ev(b, "ball_out_of_play", None, lt if lt >= 0 else None, (x, y, 0.0), q, under_pressure=False)
        self.flight_res[b] = None
        self._set_restart(b, r, team, pos, delay, cause=ev)

    def _extrapolate_goal_plane(self, b: int) -> tuple[float, float, float]:
        w = self.w
        bp, v = w.ball_pos[b], w.ball_vel[b]
        gx = np.sign(bp[0]) * HALF_LENGTH
        if abs(v[0]) < 1e-3:
            return (float(gx), float(bp[1]), float(bp[2]))
        t = (gx - bp[0]) / v[0]
        return (float(gx), float(bp[1] + v[1] * t), float(max(bp[2] + v[2] * t, 0.0)))

    def _goal(self, b: int, scoring_team: int) -> None:
        w, sq = self.w, self.sq
        last = int(w.last_touch[b])
        bp = w.ball_pos[b].copy()
        own_goal = last >= 0 and sq.team[last] != scoring_team
        if w.flight_kind[b] == FLIGHT_SHOT and w.shot_ev[b] >= 0 and not own_goal:
            shot_ev = self._shot_end(b, "goal", (np.sign(bp[0]) * HALF_LENGTH, bp[1], bp[2]))
            kp = self.logs[b].q(shot_ev).get("key_pass_id")
            if kp is not None:
                self.logs[b].q(int(kp))["is_assist"] = True
        elif self._wood_shot[b] >= 0 and not own_goal:
            # it came back off the woodwork and went in untouched: the shot scored
            q = self.logs[b].q(int(self._wood_shot[b]))
            q["outcome"] = "goal"
            q["on_target"] = True
        else:
            if w.shot_ev[b] >= 0:
                self._shot_end(b, "goal", (np.sign(bp[0]) * HALF_LENGTH, bp[1], bp[2]))
            self._finish_pass(b, "incomplete", bp)
            if own_goal:
                self._ev(b, "own_goal", last, None, bp)
        w.score[b, scoring_team] += 1
        self.flight_res[b] = None
        scorer = last
        if scorer >= 0 and not own_goal:
            self._set_anim(b, scorer, "celebrate", variant=int(self.rng.integers(0, 3)))
        for i in range(P):
            if sq.team[i] != scoring_team and not sq.is_gk[i] and self.rng.random() < 0.6:
                self._set_anim(b, i, "disappointed")
        w.celebrate_until[b] = w.t[b] + 9.0
        self._setup_kickoff(b, 1 - scoring_team, teleport=False, delay=42.0 + 10.0 * self.rng.random())

    def _offside(self, b: int, c: int, passer: int, pass_ev: int) -> None:
        w, sq = self.w, self.sq
        team = int(sq.team[c])
        self._finish_pass(b, "offside", w.ball_pos[b])
        lines = offside_lines(w, sq, self.sign)[b, team]
        q = {
            "passer": sq.ids[passer] if passer >= 0 else None,
            "offside_line_x": float(lines * self.sign[b, team]),
            "margin_m": float(max(w.pos[b, c, 0] * self.sign[b, team] - lines, 0.01)),
            "pass_id": pass_ev if pass_ev >= 0 else None,
        }
        ev = self._ev(b, "offside", c, team, (w.pos[b, c, 0], w.pos[b, c, 1], 0.0), q)
        self.logs[b].link(ev, pass_ev)
        self.flight_res[b] = None
        pos = np.clip(w.pos[b, c], [-HALF_LENGTH + 1, -HALF_WIDTH + 1], [HALF_LENGTH - 1, HALF_WIDTH - 1])
        self._set_restart(b, R_FK_INDIRECT, 1 - team, pos, 8.0 + 4.0 * self.rng.random(), cause=ev)

    def _foul(self, b: int, offender: int, victim: int, foul_type: str) -> None:
        w, sq = self.w, self.sq
        team = int(sq.team[offender])
        loc = w.pos[b, victim] if victim >= 0 else w.pos[b, offender]
        x_t = loc[0] * self.sign[b, 1 - team]
        in_box = x_t >= BOX_X and abs(loc[1]) <= BOX_HALF_WIDTH
        card = "yellow" if self.rng.random() < (0.13 if foul_type == "foul" else 0.3) else "none"
        offensive = w.poss_team[b] == team
        q = {
            "opponent": sq.ids[victim] if victim >= 0 else None,
            "foul_type": foul_type,
            "card": card,
            "penalty": bool(in_box and not offensive),
            "offensive": bool(offensive),
        }
        ev = self._ev(b, "foul_committed", offender, team, (loc[0], loc[1], 0.0), q)
        if victim >= 0:
            q2 = {"opponent": sq.ids[offender], "foul_type": foul_type, "card": card, "penalty": q["penalty"]}
            ev2 = self._ev(b, "foul_won", victim, 1 - team, (loc[0], loc[1], 0.0), q2)
            self.logs[b].link(ev, ev2)
            self._set_anim(b, victim, "fall")
            w.stun[b, victim] = 1.6
        if w.flight_kind[b] == FLIGHT_SHOT and w.shot_ev[b] >= 0:
            self._shot_end(b, "blocked", w.ball_pos[b])
        self._finish_pass(b, "incomplete", w.ball_pos[b])
        self.flight_res[b] = None
        if q["penalty"]:
            s = self.sign[b, 1 - team]
            self._set_restart(
                b, R_PENALTY, 1 - team, np.array([PENALTY_SPOT_X * s, 0.0]), 30.0 + 10 * self.rng.random(), ev
            )
        else:
            pos = np.clip(loc, [-HALF_LENGTH + 1, -HALF_WIDTH + 1], [HALF_LENGTH - 1, HALF_WIDTH - 1])
            # keep free kicks outside the penalty area of the defending side
            r = R_FK_INDIRECT if foul_type == "dangerous_play" else R_FK_DIRECT
            self._set_restart(b, r, 1 - team, pos, 12.0 + 10.0 * self.rng.random(), ev)

    # --- tackles --------------------------------------------------------------------------------------------
    def _tackles(self, bs: IA) -> None:
        w, sq = self.w, self.sq
        o = w.owner[bs]
        team = sq.team[o]
        for k, b in enumerate(bs):
            b = int(b)
            c = int(o[k])
            if w.hands[b] or w.ball_state[b] != BS_OWNED:
                continue
            opp = np.arange((1 - team[k]) * N_ON, (2 - team[k]) * N_ON)
            d = np.linalg.norm(w.pos[b, opp] - w.ball_pos[b, :2], axis=1)
            close = (d < CONTROL_RADIUS) & (w.stun[b, opp] <= 0)
            if not close.any():
                continue
            for j in opp[close]:
                j = int(j)
                rate = 0.5 + 0.8 * sq.attr["tackling"][j] / 99.0
                # near their own goal defenders commit
                if w.ball_pos[b, 0] * self.sign[b, 1 - team[k]] > 20.0:
                    rate *= 1.8
                # a carrier running straight at a defender gets challenged at once
                rel = w.pos[b, j] - w.pos[b, c]
                if float(np.dot(rel, w.vel[b, c])) > 0:
                    rate += 2.0
                if sq.is_gk[j]:
                    rate = 2.0
                if self.rng.random() < rate * DT:
                    self._tackle(b, j, c)
                    break

    def _tackle(self, b: int, j: int, c: int) -> None:
        w, sq = self.w, self.sq
        a = sq.attr
        loc = w.ball_pos[b].copy()
        tq = a["tackling"][j] / 99.0
        slide = self.rng.random() < 0.2 and np.linalg.norm(w.vel[b, j]) > 3.0
        p_foul = (0.07 + 0.18 * (1.0 - tq)) * (1.6 if slide else 1.0)
        if sq.is_gk[j]:
            p_foul = 0.06
        team_j = int(sq.team[j])
        x_t = w.pos[b, c, 0] * self.sign[b, 1 - team_j]
        if x_t >= BOX_X and abs(w.pos[b, c, 1]) <= BOX_HALF_WIDTH:
            p_foul *= 0.45  # defenders are careful in their own box
        if self.rng.random() < p_foul:
            self._set_anim(b, j, "slide_tackle" if slide else "tackle")
            self._close_open_carry(b, lost=True)
            self._foul(b, j, c, "foul")
            return
        z = 0.25 + 0.045 * (a["tackling"][j] - a["dribbling"][c]) + 0.02 * (a["strength"][j] - a["strength"][c])
        if sq.is_gk[j]:
            z = 0.4 + 0.04 * (a["gk_positioning"][j] - a["dribbling"][c])
        won = self.rng.random() < float(sigmoid(np.array(z)))
        self._set_anim(b, j, "slide_tackle" if slide else "tackle")
        w.stun[b, j] = 0.5 if won else 0.7
        if sq.is_gk[j]:
            out = "won" if won else "lost_in_play"
            self._ev(b, "gk_action", j, team_j, loc, {"action": "rush_out", "outcome": out, "body_part": "both_hands"})
            if won:
                self._close_open_carry(b, lost=True)
                self._ev(b, "dispossessed", c, None, loc, {"opponent": sq.ids[j]})
                self._gain(b, j, "from_keeper")
                w.hands[b] = True
                w.act_timer[b] = 2.0 + 2.0 * self.rng.random()
            return
        if not won:
            self._ev(b, "duel", j, team_j, loc, {"opponent": sq.ids[c], "duel_type": "tackle", "outcome": "lost"})
            return
        r = self.rng.random()
        after = "tackler" if r < 0.6 else ("loose" if r < 0.85 else ("teammate" if r < 0.95 else "out_of_play"))
        e1 = self._ev(
            b,
            "duel",
            j,
            team_j,
            loc,
            {"opponent": sq.ids[c], "duel_type": "tackle", "outcome": "won", "possession_after": after},
        )
        e2 = self._ev(b, "dispossessed", c, None, loc, {"opponent": sq.ids[j]})
        self.logs[b].link(e1, e2)
        self._close_open_carry(b, lost=True)
        self._set_anim(b, c, "fall" if self.rng.random() < 0.3 else "turn")
        w.stun[b, c] = 0.7
        if after == "tackler":
            self._gain(b, j)
        elif after == "teammate":
            mates = np.arange(team_j * N_ON, (team_j + 1) * N_ON)
            near = mates[np.argsort(np.linalg.norm(w.pos[b, mates] - w.pos[b, j], axis=1))[1]]
            w.ball_vel[b] = 0.0
            self._deflect(b, j, scale=0.0, spread=0.2, toward=w.pos[b, near] - w.pos[b, j])
        elif after == "out_of_play":
            w.ball_vel[b] = 0.0
            side = np.sign(w.ball_pos[b, 1] + 1e-6)
            self._deflect(b, j, scale=0.0, spread=0.05, toward=np.array([0.0, side]))
            w.ball_vel[b, :2] *= 4.0
        else:
            w.ball_vel[b] = 0.0
            self._deflect(b, j, scale=0.0, spread=3.0)

    def _owned_bounds(self, b: int) -> None:
        """A carrier who dribbles over a line loses the ball out of play."""
        w = self.w
        bp = w.ball_pos[b]
        if abs(bp[1]) > HALF_WIDTH + BALL_RADIUS or abs(bp[0]) > HALF_LENGTH + BALL_RADIUS:
            c = int(w.owner[b])
            self._close_open_carry(b, lost=True)
            w.ball_state[b] = BS_FLIGHT
            w.owner[b] = -1
            w.last_touch[b] = c
            w.last_touch_team[b] = self.sq.team[c]
            w.flight_kind[b] = FLIGHT_LOOSE
            w.flight_event[b] = -1
            self._out(b, "touchline" if abs(bp[1]) > HALF_WIDTH + BALL_RADIUS else "goal_line")

    # --- off-ball runs ----------------------------------------------------------------------------------------
    def _runs(self) -> None:
        w, sq = self.w, self.sq
        owned = (w.ball_state == BS_OWNED) & (w.owner >= 0) & ~self.done
        t_now = w.t
        # finish runs that ended
        ended = (w.run_type >= 0) & (w.run_until <= t_now[:, None])
        if ended.any():
            for b, i in zip(*np.nonzero(ended), strict=True):
                self._end_run(int(b), int(i))
        if not owned.any():
            return
        lines = offside_lines(w, sq, self.sign)
        for b in np.nonzero(owned)[0]:
            b = int(b)
            c = int(w.owner[b])
            team = int(sq.team[c])
            s = self.sign[b, team]
            cx_t = w.pos[b, c, 0] * s
            mates = [i for i in range(team * N_ON, (team + 1) * N_ON) if i != c and not sq.is_gk[i]]
            free = [i for i in mates if w.run_type[b, i] < 0 and w.stun[b, i] <= 0]
            if not free:
                continue
            r = self.rng.random()
            tempo = self.styles[team].tempo
            if r > 0.32 * tempo:
                continue
            i = int(self.rng.choice(free))
            px_t, py_t = w.pos[b, i, 0] * s, w.pos[b, i, 1] * s
            cy_t = w.pos[b, c, 1] * s
            line = float(lines[b, team])
            rtype = -1
            target_t = None
            group = sq.group[i]
            if group in ("ST", "W", "AM") and cx_t < 30 and px_t > line - 9 and line < 40:
                rtype = 0  # run in behind
                target_t = (min(line + 9 + 8 * self.rng.random(), 48.0), np.clip(py_t * 0.75, -25, 25))
            elif group == "FB" and abs(cy_t) > 15 and np.sign(cy_t) == np.sign(py_t) and px_t < cx_t + 2 and cx_t > -5:
                if abs(py_t) >= abs(cy_t) - 1:
                    rtype = 1  # overlap
                    target_t = (cx_t + 12.0, np.sign(py_t) * min(abs(cy_t) + 5.0, 32.0))
                else:
                    rtype = 2  # underlap
                    target_t = (cx_t + 11.0, cy_t * 0.45)
            elif group in ("ST", "AM", "CM", "W") and np.hypot(px_t - cx_t, py_t - cy_t) > 12:
                if self.rng.random() < 0.6:
                    rtype = 3  # checking run toward the ball
                    vx, vy = cx_t - px_t, cy_t - py_t
                    n = np.hypot(vx, vy)
                    target_t = (px_t + vx / n * min(n - 8, 10), py_t + vy / n * min(n - 8, 10))
                else:
                    rtype = 4  # decoy: drag a marker away
                    target_t = (min(px_t + 8, line + 2), np.clip(py_t + np.sign(py_t + 1e-6) * 9, -30, 30))
            elif np.hypot(px_t - cx_t, py_t - cy_t) < 22:
                rtype = 5  # support
                target_t = ((px_t + cx_t) / 2 + 4, (py_t + cy_t) / 2 + np.sign(py_t - cy_t + 1e-6) * 8)
            if rtype < 0 or target_t is None:
                continue
            tx = float(np.clip(target_t[0], -50, 50)) * s
            ty = float(np.clip(target_t[1], -32.5, 32.5)) * s
            w.run_target[b, i] = (tx, ty)
            w.run_until[b, i] = t_now[b] + 2.0 + 1.2 * self.rng.random()
            w.run_type[b, i] = rtype
            w.run_start[b, i] = w.pos[b, i]
            w.run_start_t[b, i] = t_now[b]
            w.run_carrier[b, i] = c
            w.mark[b, i] = -1 if rtype != 4 else w.mark[b, i]

    def _mark_run_targeted(self, b: int, c: int) -> None:
        w = self.w
        if w.run_type[b, c] >= 0:
            w.run_until[b, c] = w.t[b]
            self._end_run(b, c, targeted="targeted_received")

    def _end_run(self, b: int, i: int, targeted: str = "not_targeted") -> None:
        w, sq = self.w, self.sq
        rtype = int(w.run_type[b, i])
        w.run_type[b, i] = -1
        w.run_until[b, i] = -1e9
        start = w.run_start[b, i]
        end = w.pos[b, i]
        dur = float(w.t[b] - w.run_start_t[b, i])
        dist = float(np.linalg.norm(end - start))
        if dur < 1.0 or dist < 5.0:
            return
        avg = dist / max(dur, 1e-6)
        if avg < 3.2:
            return
        names = ("run_in_behind", "overlap", "underlap", "checking_run", "decoy", "support")
        team = int(sq.team[i])
        carrier = int(w.run_carrier[b, i])
        lp = int(w.last_pass_ev[b])
        if targeted == "not_targeted" and lp >= 0:
            rq = self.logs[b].q(lp)
            if rq.get("recipient") == sq.ids[i] and self.logs[b].records[lp]["t"] >= w.run_start_t[b, i]:
                targeted = "targeted_received" if rq.get("outcome") == "complete" else "targeted_missed"
        q = {
            "run_type": names[rtype],
            "outcome": targeted,
            "max_speed_mps": float(min(avg * 1.25, sq.max_speed[i])),
            "distance_m": dist,
            "carrier": sq.ids[carrier] if carrier >= 0 else None,
            "targeted": targeted != "not_targeted",
            "offside_at_release": False,
        }
        if rtype == 4:
            opp = np.arange((1 - team) * N_ON, (2 - team) * N_ON)
            dd = np.linalg.norm(w.pos[b, opp] - end, axis=1)
            q["dragged_defender"] = sq.ids[int(opp[int(np.argmin(dd))])]
        self.logs[b].add(
            "off_ball_run",
            float(w.run_start_t[b, i]),
            round(w.run_start_t[b, i] * SIM_HZ),
            int(w.period[b]),
            SIDE[team],
            sq.ids[i],
            (float(start[0]), float(start[1]), 0.0),
            q,
            end_location=(float(end[0]), float(end[1]), 0.0),
            duration_s=dur,
            possession_id=int(w.possession_id[b]),
            possession_team=SIDE[team],
            phase_of_play=PHASES[int(self.phase[b])],
            play_pattern=PP_NAMES[int(w.play_pattern[b])],
        )

    # ------------------------------------------------------------------------------------------------------
    # phase of play and animation
    # ------------------------------------------------------------------------------------------------------
    def _update_phase(self) -> None:
        w = self.w
        team = np.where(w.poss_team >= 0, w.poss_team, np.maximum(w.last_touch_team, 0))
        s = self.sign[np.arange(w.b), team]
        bx = w.ball_pos[:, 0] * s
        ph = np.where(
            bx >= FINAL_THIRD_X, PH["final_third"], np.where(bx >= -FINAL_THIRD_X, PH["progression"], PH["build_up"])
        )
        trans = (w.t - w.last_turnover_t) <= TRANSITION_WINDOW_S
        ph = np.where(trans, PH["attacking_transition"], ph)
        adv = bx - w.poss_start_x
        direct = adv / np.maximum(w.poss_path, 1e-6)
        counter = w.poss_counter_ok & ((w.t - w.poss_start_t) <= COUNTER_WINDOW_S) & (adv >= COUNTER_ADVANCE_M)
        counter &= direct >= COUNTER_DIRECTNESS
        ph = np.where(counter, PH["counter_attack"], ph)
        if counter.any():
            w.play_pattern[counter & (w.play_pattern == PP["regular_play"])] = PP["from_counter"]
        rt = w.last_restart_type
        set_type = np.isin(rt, [R_CORNER, R_FK_DIRECT, R_FK_INDIRECT, R_PENALTY]) | (rt == R_THROW_IN)
        setp = set_type & ((w.t - w.last_restart_t) <= SET_PIECE_WINDOW_S) & (w.restart_poss_id == w.possession_id)
        ph = np.where(setp, PH["set_piece"], ph)
        ph = np.where(w.ball_state == BS_DEAD, PH["dead_ball"], ph)
        self.phase = ph.astype(np.int64)

    def _animate(self) -> None:
        w, sq = self.w, self.sq
        w.anim_t += DT
        w.anim_lock -= DT
        free = w.anim_lock <= 0
        speed = np.linalg.norm(w.vel, axis=-1)
        mov = np.arctan2(w.vel[..., 1], w.vel[..., 0])
        rel = np.arctan2(np.sin(mov - w.facing), np.cos(mov - w.facing))
        loco = np.full(speed.shape, A["idle"])
        loco = np.where(speed > 0.5, A["jog"], loco)
        loco = np.where(speed > 4.2, A["run"], loco)
        loco = np.where(speed > 6.4, A["sprint"], loco)
        back = (np.abs(rel) > np.radians(125)) & (speed > 0.6) & (speed < 5.5)
        loco = np.where(back, A["jog_back"], loco)
        side = (np.abs(rel) > np.radians(55)) & (np.abs(rel) <= np.radians(125)) & (speed > 0.6) & (speed < 5.0)
        loco = np.where(side & (rel > 0), A["strafe_left"], loco)
        loco = np.where(side & (rel < 0), A["strafe_right"], loco)
        gk = sq.is_gk[None, :] & (speed < 0.6)
        loco = np.where(gk, A["gk_idle"], loco)
        owned = (w.ball_state == BS_OWNED) & (w.owner >= 0) & ~w.hands
        if owned.any():
            bi = np.nonzero(owned)[0]
            o = w.owner[bi]
            loco[bi, o] = np.where(speed[bi, o] > 0.5, A["dribble"], A["idle"])
        changed = free & (w.anim != loco)
        w.anim = np.where(free, loco, w.anim)
        w.anim_t = np.where(changed, 0.0, w.anim_t)
        w.anim_mirror = np.where(free, False, w.anim_mirror)
        w.anim_lock = np.where(free, 0.0, w.anim_lock)


def _scramble(res: Resolved, row: int, rng: np.random.Generator) -> bool:
    """A few volleys in a goalmouth scramble go in off a knee or the chest (body part "other")."""
    return int(res.shot_tech[row]) in (4, 5) and rng.random() < 0.1


def _progressive(sx: float, ex: float, d0: float, d1: float, ey: float) -> bool:
    gain = d0 - d1
    into_box = ex >= BOX_X and abs(ey) <= BOX_HALF_WIDTH and not (sx >= BOX_X)
    if sx < 0 and ex < 0:
        ok = gain >= 30.0
    elif sx < 0 <= ex:
        ok = gain >= 15.0
    else:
        ok = gain >= 10.0
    return bool(ok or into_box)


def _situation_name(sit: int, last_restart: int, in_set_window: bool, kick_off: bool) -> str:
    if sit == SIT_PENALTY:
        return "penalty"
    if sit == SIT_DIRECT_FK:
        return "direct_free_kick"
    if in_set_window:
        if last_restart == R_CORNER:
            return "corner_phase"
        if last_restart in (R_FK_DIRECT, R_FK_INDIRECT):
            return "free_kick_phase"
        if last_restart == R_THROW_IN:
            return "throw_in_phase"
    if kick_off:
        return "kick_off"
    if sit == SIT_COUNTER:
        return "counter_attack"
    _ = SIT_SET_PIECE
    return "open_play"


__all__ = ["Match"]
