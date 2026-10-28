"""Engine state: static squad arrays and the batched dynamic world (leading dimension B = batch of realities).

A full match runs with B = 1; "1,000 realities" run the same code with B = 1000. ``World.snapshot(b)`` and
``World.from_snapshot`` copy the complete state of one element, which is what the fork API stores on disk.
"""

from dataclasses import dataclass, fields
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from shared.schemas.meta import MatchMeta, PlayerMeta
from sim.constants import BALL_RADIUS
from sim.squads import FORMATIONS, POSITION_GROUP

F = npt.NDArray[np.float64]
IA = npt.NDArray[np.int64]
BA = npt.NDArray[np.bool_]

N_ON: Final = 11
P: Final = 22

# Ball states
BS_OWNED: Final = 0
BS_FLIGHT: Final = 1
BS_DEAD: Final = 2

# Flight modes (what the planner decided happens to the ball in flight)
FM_NONE: Final = 0
FM_RECEIVE: Final = 1  # teammate collects (pass completes)
FM_INTERCEPT: Final = 2  # opponent collects
FM_MISCONTROL: Final = 3  # intended receiver touches but loses it
FM_LOOSE: Final = 4  # nobody planned: first to reach it
FM_SAVE: Final = 5  # keeper saves a shot
FM_BLOCK: Final = 6  # outfield player blocks a shot
FM_FREE: Final = 7  # nobody can touch it (shot going in / wide, clearance into touch)
FM_CLAIM: Final = 8  # keeper claims / punches a cross

# Restart codes follow shared RESTARTS order: kick_off, throw_in, goal_kick, corner, free_kick_direct,
# free_kick_indirect, penalty, drop_ball
R_KICK_OFF, R_THROW_IN, R_GOAL_KICK, R_CORNER, R_FK_DIRECT, R_FK_INDIRECT, R_PENALTY, R_DROP = range(8)

ARCHETYPE_CODES: Final = (
    "advanced_playmaker",
    "deep_lying_playmaker",
    "box_to_box",
    "ball_winning_midfielder",
    "poacher",
    "target_forward",
    "inside_forward",
    "touchline_winger",
    "overlapping_fullback",
    "inverted_fullback",
    "ball_playing_centre_back",
    "stopper",
    "sweeper_keeper",
    "shot_stopper",
)


@dataclass
class Squad:
    """Static per-player data for the 22 starters (index 0-10 home, 11-21 away; goalkeepers at 0 and 11)."""

    meta: MatchMeta
    players: list[PlayerMeta]
    ids: list[str]
    rec_index: IA
    """Index of each on-pitch player in the recording's player order (home squad, then away squad)."""
    n_rec: int
    team: IA
    is_gk: BA
    line: IA
    group: list[str]
    out_xy: F
    in_xy: F
    attr: dict[str, F]
    max_speed: F
    max_accel: F
    max_decel: F
    reaction: F
    jump_reach: F
    height: F
    left_foot: BA
    weak_foot: F
    archetype: IA

    @classmethod
    def from_meta(cls, meta: MatchMeta) -> "Squad":
        players: list[PlayerMeta] = []
        rec_index: list[int] = []
        offset = 0
        for team in (meta.home, meta.away):
            starters = [p for p in team.players if p.starter][:N_ON]
            for p in starters:
                players.append(p)
                rec_index.append(offset + team.players.index(p))
            offset += len(team.players)
        n = len(players)
        if n != P:
            raise ValueError(f"need 11 starters per side, got {n}")
        team_idx = np.array([0] * N_ON + [1] * N_ON, dtype=np.int64)
        out_xy = np.zeros((P, 2))
        in_xy = np.zeros((P, 2))
        line = np.zeros(P, dtype=np.int64)
        for k, tm in enumerate((meta.home, meta.away)):
            slots = FORMATIONS[tm.formation]
            starters = [p for p in tm.players if p.starter][:N_ON]
            by_pos = {s.position: s for s in slots}
            for j, p in enumerate(starters):
                s = by_pos.get(p.position, slots[j])
                out_xy[k * N_ON + j] = s.out_xy
                in_xy[k * N_ON + j] = s.in_xy
                line[k * N_ON + j] = s.line
        attr = {
            a: np.array([float(getattr(p.attributes, a)) for p in players])
            for a in type(players[0].attributes).model_fields
        }
        return cls(
            meta=meta,
            players=players,
            ids=[p.id for p in players],
            rec_index=np.array(rec_index, dtype=np.int64),
            n_rec=offset,
            team=team_idx,
            is_gk=np.array([p.position == "GK" for p in players]),
            line=line,
            group=[POSITION_GROUP[p.position] for p in players],
            out_xy=out_xy,
            in_xy=in_xy,
            attr=attr,
            max_speed=np.array([p.physical.max_speed_mps for p in players]),
            max_accel=np.array([p.physical.max_accel_mps2 for p in players]),
            max_decel=np.array([p.physical.max_decel_mps2 for p in players]),
            reaction=np.array([p.physical.reaction_time_s for p in players]),
            jump_reach=np.array([p.physical.jump_reach_m for p in players]),
            height=np.array([p.height_cm / 100 for p in players]),
            left_foot=np.array([p.preferred_foot == "left" for p in players]),
            weak_foot=np.array([float(p.weak_foot) for p in players]),
            archetype=np.array([ARCHETYPE_CODES.index(p.archetype) for p in players], dtype=np.int64),
        )

    def q(self, name: str) -> F:
        """Attribute on a 0..1 scale."""
        return np.asarray(self.attr[name] / 99.0)

    def teammates(self, i: int) -> IA:
        t = self.team[i]
        return np.array([j for j in range(P) if self.team[j] == t and j != i], dtype=np.int64)


@dataclass
class World:
    """Batched dynamic state. Every field has leading dimension B."""

    t: F
    period: IA
    status: IA
    """Index into shared.recording.STATUSES."""
    period_end_t: F
    ball_pos: F
    ball_vel: F
    ball_spin: F
    ball_state: IA
    owner: IA
    last_touch: IA
    last_touch_team: IA
    pos: F
    vel: F
    facing: F
    z: F
    vz: F
    stamina: F
    anim: IA
    anim_t: F
    anim_lock: F
    anim_mirror: BA
    anim_variant: IA
    stun: F
    dive_vel: F
    """(B, P, 2) horizontal velocity imposed during a keeper dive / slide (bypasses steering)."""
    score: IA
    poss_team: IA
    possession_id: IA
    poss_start_t: F
    poss_start_x: F
    """Ball x in the possessing team's attacking frame when the possession started."""
    poss_path: F
    """Ball path length during this possession (for counter-attack directness)."""
    poss_counter_ok: BA
    """Possession began with an open-play turnover outside the attacking third."""
    play_pattern: IA
    last_turnover_t: F
    last_restart_t: F
    last_restart_type: IA
    restart: IA
    restart_team: IA
    restart_pos: F
    restart_timer: F
    restart_taker: IA
    restart_cause: IA
    """Event index of the cause (foul / out of play), -1 if none."""
    act_timer: F
    carry_target: F
    carry_speed: F
    receive_t: F
    carry_start: F
    carry_start_t: F
    carry_path: F
    carry_max_speed: F
    collector: IA
    flight_mode: IA
    collect_tick: IA
    collect_point: F
    flight_kind: IA
    """0 = pass, 1 = shot, 2 = clearance, 3 = loose / deflection, 4 = save rebound."""
    flight_passer: IA
    flight_receiver: IA
    flight_event: IA
    flight_t0: F
    flight_origin: F
    flight_peak: F
    flight_offside: BA
    flight_aerial: BA
    flight_sub: IA
    """Planner sub-outcome (catch / parry kind, interception kind, miscontrol)."""
    run_target: F
    run_until: F
    run_type: IA
    run_start: F
    run_start_t: F
    run_carrier: IA
    mark: IA
    tick: IA
    restart_poss_id: IA
    ref_pos: F
    ref_vel: F
    dive_until: F
    celebrate_until: F
    last_pass_ev: IA
    last_pass_t: F
    last_pass_team: IA
    shot_ev: IA
    clear_ev: IA
    receipt_ev: IA
    last_dribble_t: F
    aerial_won_t: F
    last_bounce_t: F
    """Clock time of the ball's last bounce (half-volleys are struck just after one)."""
    hands: BA
    """Keeper holding the ball in his hands."""

    @classmethod
    def empty(cls, b: int) -> "World":
        def f(*shape: int) -> F:
            return np.zeros((b, *shape))

        def i(*shape: int, fill: int = -1) -> IA:
            return np.full((b, *shape), fill, dtype=np.int64)

        def bo(*shape: int) -> BA:
            return np.zeros((b, *shape), dtype=bool)

        return cls(
            t=f(),
            period=i(fill=1),
            status=i(fill=1),
            period_end_t=f(),
            ball_pos=np.tile([0.0, 0.0, BALL_RADIUS], (b, 1)),
            ball_vel=f(3),
            ball_spin=f(3),
            ball_state=i(fill=BS_DEAD),
            owner=i(),
            last_touch=i(),
            last_touch_team=i(),
            pos=f(P, 2),
            vel=f(P, 2),
            facing=f(P),
            z=f(P),
            vz=f(P),
            stamina=np.ones((b, P)),
            anim=i(P, fill=0),
            anim_t=f(P),
            anim_lock=f(P),
            anim_mirror=bo(P),
            anim_variant=i(P, fill=0),
            stun=f(P),
            dive_vel=f(P, 2),
            score=i(2, fill=0),
            poss_team=i(),
            possession_id=i(fill=0),
            poss_start_t=f(),
            poss_start_x=f(),
            poss_path=f(),
            poss_counter_ok=bo(),
            play_pattern=i(fill=0),
            last_turnover_t=np.full(b, -1e9),
            last_restart_t=np.full(b, -1e9),
            last_restart_type=i(),
            restart=i(),
            restart_team=i(),
            restart_pos=f(2),
            restart_timer=f(),
            restart_taker=i(),
            restart_cause=i(),
            act_timer=f(),
            carry_target=f(2),
            carry_speed=f(),
            receive_t=np.full(b, -1e9),
            carry_start=f(2),
            carry_start_t=f(),
            carry_path=f(),
            carry_max_speed=f(),
            collector=i(),
            flight_mode=i(fill=FM_NONE),
            collect_tick=i(),
            collect_point=f(3),
            flight_kind=i(fill=0),
            flight_passer=i(),
            flight_receiver=i(),
            flight_event=i(),
            flight_t0=f(),
            flight_origin=f(3),
            flight_peak=f(),
            flight_offside=bo(),
            flight_aerial=bo(),
            flight_sub=i(fill=0),
            run_target=f(P, 2),
            run_until=np.full((b, P), -1e9),
            run_type=i(P),
            run_start=f(P, 2),
            run_start_t=f(P),
            run_carrier=i(P),
            mark=i(P),
            tick=i(fill=0),
            restart_poss_id=i(),
            ref_pos=np.tile([-8.0, 10.0], (b, 1)),
            ref_vel=f(2),
            dive_until=np.full((b, P), -1e9),
            celebrate_until=np.full(b, -1e9),
            last_pass_ev=i(),
            last_pass_t=np.full(b, -1e9),
            last_pass_team=i(),
            shot_ev=i(),
            clear_ev=i(),
            receipt_ev=i(),
            last_dribble_t=np.full(b, -1e9),
            aerial_won_t=np.full(b, -1e9),
            last_bounce_t=np.full(b, -1e9),
            hands=bo(),
        )

    @property
    def b(self) -> int:
        return len(self.t)

    def snapshot(self, b: int) -> dict[str, npt.NDArray[Any]]:
        return {f.name: np.array(getattr(self, f.name)[b]) for f in fields(self)}

    @classmethod
    def from_snapshot(cls, snap: dict[str, npt.NDArray[Any]], b: int) -> "World":
        missing = [f.name for f in fields(cls) if f.name not in snap]
        if missing:  # snapshots written before a field existed get its default
            blank = cls.empty(1)
            snap = {**snap, **{k: getattr(blank, k)[0] for k in missing}}
        kw = {f.name: np.repeat(np.asarray(snap[f.name])[None, ...], b, axis=0) for f in fields(cls)}
        return cls(**kw)


def world_field_names() -> list[str]:
    return [f.name for f in fields(World)]
