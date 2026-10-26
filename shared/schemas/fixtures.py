"""Example instances of every contract. Written to shared/fixtures/ and to typed TS fixtures by export.py.

They double as documentation and as the renderer/UI teams' first test data, so they describe one consistent
scene: 38:52 of the first half, 1-1, home (attacking +x) breaks. Marchetti (H10) carries at (18.4, 6.1) and plays
a curled through ball behind the away line for Demir (H09), whose shot 2.5 s later is saved by Ferrán (A01).
"""

import numpy as np
from pydantic import BaseModel

from shared.model_api import StubModels, shot_geometry
from shared.schemas.actions import ActionIntent, BallLaunch, ExecutionEllipse, ResolvedAction
from shared.schemas.events import (
    BlockQ,
    CarryQ,
    ClearanceQ,
    DuelQ,
    FoulQ,
    GkQ,
    InterceptionQ,
    MatchEvent,
    OffsideQ,
    OutOfPlayQ,
    PassQ,
    PhysicalQ,
    PressingSequenceQ,
    PressureQ,
    ReceiptQ,
    RecoveryQ,
    RestartQ,
    RunQ,
    ShotQ,
    StepUpQ,
    TakeOnQ,
    TouchLossQ,
)
from shared.schemas.labels import CameraCalibration, FrameLabels, ObjectLabel
from shared.schemas.meta import (
    Archetype,
    CrestSpec,
    KitColors,
    MatchMeta,
    PhysicalParams,
    PlayerAttributes,
    PlayerMeta,
    Position,
    TeamMeta,
)
from shared.schemas.models import ModelOutputs, OptionEvaluation, PitchControlGrid
from shared.schemas.moments import Consequence, DecisionMoment, RewindScores
from shared.schemas.narrative import Evidence, FactCheck, Narrative, ViewerProfile
from shared.schemas.overlay import (
    AccuracyPanelPayload,
    AccuracyRow,
    CaptionPayload,
    ControlChaosPayload,
    CueAnchor,
    HighlightPayload,
    MilestonePayload,
    MomentumChartPayload,
    NameTagPayload,
    OverlayCue,
    PassCardPayload,
    RewindPromptPayload,
    SeriesPoint,
    ShotCardPayload,
    ThresholdBadgePayload,
)
from shared.schemas.rollout import (
    Interval,
    OutcomeCount,
    RolloutDistribution,
    RolloutOutcome,
    RolloutRequest,
    RolloutResult,
    RolloutSample,
    Verdict,
)
from shared.schemas.state import BallState, PlayerState, Possession, RefereeState, Score, SimState
from shared.schemas.stats import PlayerStats, Recap, StatsSnapshot, TeamStats, WhatIf
from shared.schemas.stream import FrameMsg, SeekCmd, StreamProtocol

MATCH = "demo-0001"
T_PASS, TICK_PASS = 2332.4, 58_310
T_SHOT, TICK_SHOT = 2334.9, 58_372
BALL = (18.4, 6.1)
PASS_END = (38.6, -3.6)
SHOT_FROM = (39.1, -3.2)
SNAP = f"snap:{MATCH}:{TICK_PASS}"

_ATTRS = list(PlayerAttributes.model_fields)

_LINEUP: list[tuple[int, Position, Archetype]] = [
    (1, "GK", "sweeper_keeper"),
    (2, "RB", "overlapping_fullback"),
    (5, "RCB", "ball_playing_centre_back"),
    (4, "LCB", "stopper"),
    (3, "LB", "inverted_fullback"),
    (6, "DM", "deep_lying_playmaker"),
    (8, "RCM", "box_to_box"),
    (10, "LCM", "advanced_playmaker"),
    (7, "RW", "touchline_winger"),
    (11, "LW", "inside_forward"),
    (9, "ST", "poacher"),
]

_NAMES = {
    "home": [
        ("Tomás", "Varga"), ("Kofi", "Mensah"), ("Rhys", "Calloway"), ("Idris", "Okafor"), ("Lukas", "Brandt"),
        ("Mateo", "Quiroga"), ("Sami", "Haddad"), ("Elio", "Marchetti"), ("Dayo", "Adeyemi"), ("Niall", "Keane"),
        ("Yusuf", "Demir"),
    ],
    "away": [
        ("Pablo", "Ferrán"), ("Jonah", "Whitlock"), ("Emeka", "Obi"), ("Henrik", "Solberg"), ("Ravi", "Nair"),
        ("Andrés", "Molina"), ("Théo", "Garnier"), ("Kenji", "Arai"), ("Malik", "Diallo"), ("Callum", "Reid"),
        ("Bruno", "Saraiva"),
    ],
}  # fmt: skip

# World positions and velocities at T_PASS, in lineup order.
_HOME_POS = [(-45.0, 0.0), (14.0, -24.0), (-12.0, -9.0), (-13.0, 8.0), (2.0, 22.0), (5.0, 2.0), (12.0, -8.0),
             BALL, (28.0, -20.0), (27.0, 18.0), (31.0, -2.0)]  # fmt: skip
_HOME_VEL = [(0.5, 0.0), (6.8, 0.2), (2.0, 0.0), (2.0, 0.0), (3.0, -0.5), (3.5, 0.3), (5.0, 0.5),
             (6.2, 0.4), (5.5, 1.0), (5.0, -1.2), (6.5, -0.6)]  # fmt: skip
_AWAY_POS = [(50.5, 0.0), (34.0, 20.0), (33.0, 6.0), (32.5, -3.0), (31.0, -18.0), (26.0, 2.0), (22.0, 10.0),
             (20.0, -6.0), (10.0, 24.0), (12.0, -22.0), (2.0, 0.0)]  # fmt: skip
_AWAY_VEL = [(0.0, 0.0), (2.0, -0.5), (2.5, -0.3), (3.0, 0.0), (2.0, 0.8), (3.5, 0.0), (4.0, -1.0),
             (-1.5, 2.5), (5.5, 0.0), (5.0, 0.0), (3.0, 0.0)]  # fmt: skip


def _player(side: str, i: int) -> PlayerMeta:
    number, position, archetype = _LINEUP[i]
    first, last = _NAMES[side][i]
    base = 55 + (i * 7 + (3 if side == "home" else 0)) % 30
    attrs = {name: min(99, max(1, base + (j * 13 + i * 5) % 21 - 10)) for j, name in enumerate(_ATTRS)}
    if position != "GK":
        attrs.update({k: 20 + (i * 3) % 15 for k in _ATTRS if k.startswith("gk_")})
    if archetype == "advanced_playmaker":
        attrs.update(vision=91, curve=89, long_passing=88, technique=90, short_passing=87)
    return PlayerMeta.model_validate(
        {
            "id": f"{'H' if side == 'home' else 'A'}{number:02d}",
            "team": side,
            "first_name": first,
            "last_name": last,
            "display_name": last,
            "shirt_number": number,
            "position": position,
            "archetype": archetype,
            "preferred_foot": "left" if i in (4, 7, 9) else "right",
            "weak_foot": 1 + (i % 5),
            "height_cm": 172 + (i * 4) % 22,
            "nationality": ["NGA", "ESP", "FRA", "GHA", "BRA", "JPN", "IRL", "TUR", "SEN", "NOR", "IND"][i],
            "attributes": attrs,
            "physical": PhysicalParams(
                max_speed_mps=round(7.6 + attrs["pace"] / 99 * 2.4, 2),
                max_accel_mps2=round(2.8 + attrs["acceleration"] / 99 * 2.2, 2),
                max_decel_mps2=round(4.0 + attrs["acceleration"] / 99 * 2.0, 2),
                reaction_time_s=0.2,
            ),
        }
    )


def match_meta() -> MatchMeta:
    def team(side: str) -> TeamMeta:
        home = side == "home"
        return TeamMeta(
            side="home" if home else "away",
            name="Harbour City Mariners" if home else "Redcliffe Forge",
            short_name="HCM" if home else "RFG",
            city="Harbour City" if home else "Redcliffe",
            formation="4-3-3",
            kit=KitColors(
                primary="#0b3d91" if home else "#c8102e",
                secondary="#f2f4f8" if home else "#1a1a1a",
                accent="#4cc9f0" if home else "#f5b700",
                number="#ffffff",
                pattern="plain" if home else "chevron",
            ),
            gk_kit=KitColors(
                primary="#2dc653" if home else "#7b2cbf",
                secondary="#0d1b2a",
                accent="#ffffff",
                number="#0d1b2a" if home else "#ffffff",
            ),
            crest=CrestSpec(
                shape="shield" if home else "roundel",
                monogram="HC" if home else "RF",
                primary="#0b3d91" if home else "#c8102e",
                secondary="#4cc9f0" if home else "#f5b700",
                emblem="anchor" if home else "bolt",
            ),
            players=[_player(side, i) for i in range(11)],
        )

    return MatchMeta(match_id=MATCH, seed=20261026, home=team("home"), away=team("away"), venue="Lantern Park")


def _facing(v: tuple[float, float], default: float) -> float:
    return round(float(np.arctan2(v[1], v[0])), 3) if np.hypot(*v) > 0.5 else default


def sim_state() -> SimState:
    meta = match_meta()
    players: list[PlayerState] = []
    for team, pos, vel, toward in ((meta.home, _HOME_POS, _HOME_VEL, 0.0), (meta.away, _AWAY_POS, _AWAY_VEL, 3.142)):
        for i, p in enumerate(team.players):
            speed = float(np.hypot(*vel[i]))
            anim = "sprint" if speed > 6.0 else "run" if speed > 4.0 else "jog" if speed > 1.0 else "idle"
            if p.position == "GK":
                anim = "gk_idle"
            if p.id == "H10":
                anim = "dribble"
            players.append(
                PlayerState(
                    id=p.id,
                    pos=pos[i],
                    vel=vel[i],
                    facing=_facing(vel[i], toward),
                    anim=anim,
                    anim_t=0.6,
                    anim_mirror=p.preferred_foot == "left" and anim == "dribble",
                    stamina=0.86,
                )
            )
    return SimState(
        tick=TICK_PASS,
        t=T_PASS,
        period=1,
        status="first_half",
        ball=BallState(pos=(BALL[0] + 0.6, BALL[1] + 0.1, 0.11), vel=(6.2, 0.4, 0.0), last_touch="H10"),
        players=players,
        referee=RefereeState(pos=(10.0, 12.0), vel=(3.0, 0.0)),
        possession=Possession(team="home", player="H10", possession_id=214),
        phase="counter_attack",
        score=Score(home=1, away=1),
        attacking_positive_x="home",
        keyframe=True,
        snapshot_ref=SNAP,
    )


def _through_ball_intent() -> ActionIntent:
    return ActionIntent(
        kind="pass_space",
        player="H10",
        target=(38.0, -4.0, 0.0),
        path=[BALL, (27.0, 3.0), (33.0, -1.5), (38.0, -4.0)],
        curve=0.35,
        power=0.62,
        source="fan",
    )


def resolved_action() -> ResolvedAction:
    # Chord to the target points at -0.476 rad; with right-curl (negative z spin) the ball is aimed left of it.
    return ResolvedAction(
        intent=_through_ball_intent(),
        action_type="pass",
        body_part="left_foot",
        pass_technique="curled",
        pass_type="through_ball",
        pass_height="ground",
        intended_end=(38.0, -4.0, 0.0),
        launch=BallLaunch(speed_mps=17.5, elevation_rad=0.03, azimuth_rad=-0.40, spin=(0.0, 0.0, -8.0)),
        curve_applied=0.3,
        ellipse=ExecutionEllipse(
            plane="pitch", center=(38.0, -4.0, 0.0), semi_major_m=1.8, semi_minor_m=0.9, angle_rad=-0.476
        ),
        preview_arc=[(18.4, 6.1, 0.11), (24.6, 4.0, 0.11), (30.8, 1.0, 0.11), (35.0, -1.8, 0.11), (38.0, -4.0, 0.11)],
        execution_quality=0.84,
        available_pass_techniques=["standard", "driven", "lofted", "chipped", "curled", "outside_of_boot"],
        max_curve=0.85,
    )


def _shot_qualifiers() -> ShotQ:
    dist, angle = shot_geometry(np.array([SHOT_FROM[0]]), np.array([SHOT_FROM[1]]))
    return ShotQ(
        keeper="A01",
        outcome="saved",
        body_part="right_foot",
        technique="placed",
        situation="counter_attack",
        ball_speed_mps=24.1,
        distance_to_goal_m=round(float(dist[0]), 2),
        angle_rad=round(float(angle[0]), 3),
        peak_height_m=0.7,
        on_target=True,
        xg=0.31,
        psxg=0.42,
        n_defenders_in_cone=1,
        gk_position="moving",
        key_pass_id="ev_004209",
        gk_event_id="ev_004212",
        freeze_frame_ref=f"snap:{MATCH}:{TICK_SHOT}",
    )


def _event(id: str, index: int, type: str, t: float, **kw: object) -> MatchEvent:
    base: dict[str, object] = {
        "id": id,
        "index": index,
        "source": "ground_truth",
        "type": type,
        "period": 1,
        "t": t,
        "tick": round(t * 25),
        "team": "home",
        "player": "H10",
        "location": (BALL[0], BALL[1], 0.11),
        "possession_id": 214,
        "possession_team": "home",
        "play_pattern": "from_counter",
        "phase_of_play": "counter_attack",
    }
    base.update(kw)
    return MatchEvent.model_validate(base)


def match_event() -> MatchEvent:
    return _event(
        "ev_004211",
        4211,
        "shot",
        T_SHOT,
        player="H09",
        location=(SHOT_FROM[0], SHOT_FROM[1], 0.11),
        end_location=(52.5, 2.1, 0.6),
        under_pressure=True,
        related_events=["ev_004209", "ev_004212"],
        qualifiers=_shot_qualifiers(),
    )


def event_examples() -> list[MatchEvent]:
    """One event per qualifier kind, plus a no-qualifier event and a partially-detected one."""
    pass_q = PassQ(
        recipient="H09",
        outcome="complete",
        height="ground",
        length_class="medium",
        body_part="left_foot",
        technique="curled",
        pass_type="through_ball",
        length_m=round(float(np.hypot(PASS_END[0] - BALL[0], PASS_END[1] - BALL[1])), 2),
        angle_rad=round(float(np.arctan2(PASS_END[1] - BALL[1], PASS_END[0] - BALL[0])), 3),
        ball_speed_mps=17.2,
        peak_height_m=0.12,
        duration_s=1.7,
        is_progressive=True,
        is_line_breaking=True,
        lines_broken=1,
        is_key_pass=True,
        assisted_shot_id="ev_004211",
        receipt_id="ev_004210",
        xpass=0.58,
    )
    end3 = (PASS_END[0], PASS_END[1], 0.11)
    return [
        _event("ev_000001", 0, "period_start", 0.0, team=None, player=None, location=None, possession_id=None,
               possession_team=None, play_pattern="regular_play", phase_of_play=None),
        _event("ev_004209", 4209, "pass", T_PASS, end_location=end3, qualifiers=pass_q),
        _event("ev_004210", 4210, "ball_receipt", 2334.1, player="H09", location=end3,
               qualifiers=ReceiptQ(passer="H10", outcome="complete", touch_quality="clean",
                                   body_orientation="open", first_touch_body_part="right_foot",
                                   pressure_distance_m=1.9, pass_id="ev_004209")),
        _event("ev_004208", 4208, "carry", 2329.9, location=(9.0, 4.2, 0.11), end_location=(BALL[0], BALL[1], 0.11),
               duration_s=2.5, qualifiers=CarryQ(outcome="retained", distance_m=9.6, path_m=10.1,
                                                  max_speed_mps=6.4, is_progressive=False)),
        _event("ev_004205", 4205, "take_on", 2329.0, location=(8.0, 4.0, 0.11),
               qualifiers=TakeOnQ(opponent="A10", outcome="success", nutmeg=True)),
        match_event(),
        _event("ev_004212", 4212, "gk_action", 2335.3, team="away", player="A01", location=(51.8, 1.6, 0.0),
               possession_team="away", qualifiers=GkQ(action="save", outcome="parry_safe", technique="diving",
                                                       body_part="both_hands", shot_id="ev_004211",
                                                       position_before="moving", dive_distance_m=1.7)),
        _event("ev_004100", 4100, "duel", 2290.2, team="away", player="A06", location=(-3.0, 1.0, 0.0),
               possession_team="away", qualifiers=DuelQ(opponent="H06", duel_type="tackle", outcome="won",
                                                         possession_after="tackler")),
        _event("ev_004101", 4101, "interception", 2301.7, team="away", player="A04", location=(30.0, -2.0, 0.3),
               qualifiers=InterceptionQ(passer="H08", outcome="won")),
        _event("ev_004102", 4102, "block", 2310.0, team="away", player="A05", location=(36.0, 4.0, 0.4),
               qualifiers=BlockQ(block_type="shot_block", save_block=True)),
        _event("ev_004103", 4103, "clearance", 2311.4, team="away", player="A02", location=(45.0, 9.0, 1.6),
               qualifiers=ClearanceQ(body_part="head", outcome="in_play_safe", aerial_won=True)),
        _event("ev_004104", 4104, "ball_recovery", 2318.9, team="home", player="H06", location=(0.0, 3.0, 0.11),
               qualifiers=RecoveryQ()),
        _event("ev_004105", 4105, "dispossessed", 2320.0, team="away", player="A09", location=(-2.0, 1.0, 0.11),
               qualifiers=TouchLossQ(opponent="H06")),
        _event("ev_004106", 4106, "foul_committed", 2250.0, team="away", player="A08", location=(-20.0, 9.0, 0.0),
               qualifiers=FoulQ(opponent="H06", card="yellow")),
        _event("ev_004107", 4107, "restart", 2262.0, location=(-20.0, 9.0, 0.11), play_pattern="from_free_kick",
               phase_of_play="set_piece", qualifiers=RestartQ(restart_type="free_kick_indirect", awarded_to="home",
                                                               cause_event_id="ev_004106")),
        _event("ev_004108", 4108, "ball_out_of_play", 2270.0, team="home", player="H07",
               location=(30.0, -34.0, 0.2), qualifiers=OutOfPlayQ(line="touchline", last_touch_team="home",
                                                                   last_touch_player="H07",
                                                                   result_restart_type="throw_in")),
        _event("ev_004109", 4109, "offside", 2280.5, player="H09", location=(36.0, 1.0, 0.0),
               qualifiers=OffsideQ(passer="H10", offside_line_x=34.6, margin_m=1.4)),
        _event("ev_004110", 4110, "pressure", 2333.1, duration_s=1.2, team="away", player="A04",
               location=(33.0, -2.6, 0.0), possession_team="home",
               qualifiers=PressureQ(pressed_player="H09", pressed_role="receiver", min_distance_m=1.6,
                                    closing_speed_mps=2.4)),
        _event("ev_004111", 4111, "pressing_sequence", 2200.0, duration_s=6.2, team="home", player=None,
               location=(30.0, 0.0, 0.0), possession_team="away", phase_of_play="out_of_possession",
               qualifiers=PressingSequenceQ(n_pressures=3, pressing_players=["H09", "H10", "H08"],
                                            outcome="success", pressure_ids=["ev_004090", "ev_004091",
                                                                             "ev_004092"])),
        _event("ev_004112", 4112, "off_ball_run", 2331.0, duration_s=3.1, player="H09", location=(25.0, 0.0, 0.0),
               end_location=(38.6, -3.6, 0.0), qualifiers=RunQ(run_type="run_in_behind",
                                                                outcome="targeted_received", max_speed_mps=8.7,
                                                                distance_m=14.1, targeted=True)),
        _event("ev_004113", 4113, "sprint", 2331.4, duration_s=2.6, player="H09", location=(27.0, -1.0, 0.0),
               end_location=(38.6, -3.6, 0.0), qualifiers=PhysicalQ(distance_m=12.0, peak_speed_mps=9.1)),
        _event("ev_004114", 4114, "line_step_up", 2240.0, duration_s=1.6, team="away", player=None,
               location=(-20.0, 0.0, 0.0), possession_team="home", phase_of_play="out_of_possession",
               qualifiers=StepUpQ(line_players=["A02", "A05", "A04", "A03"], line_height_before=-24.0,
                                  line_height_after=-19.5, outcome="trap_success")),
        # Detected by the Spotter: track ids, a confidence, and the qualifiers it could not see left null.
        _event("ev_d_000877", 877, "pass", T_PASS + 0.04, source="detected", confidence=0.83, player=None,
               track_id=17, secondary_track_id=9, possession_id=None, phase_of_play=None, end_location=end3,
               qualifiers=pass_q.model_copy(update={"recipient": None, "technique": None, "body_part": None,
                                                    "xpass": None, "receipt_id": None, "assisted_shot_id": None})),
    ]  # fmt: skip


def _options() -> list[OptionEvaluation]:
    return [
        OptionEvaluation(
            option_id="opt_1",
            intent=_through_ball_intent(),
            label="Through ball to Demir",
            label_key="option.through_ball",
            label_args={"player": "H09"},
            p_success=0.58,
            p_success_baseline=0.49,
            value_if_success=0.21,
            xt_gain=0.176,
            decision_value=0.102,
            is_real=True,
        ),
        OptionEvaluation(
            option_id="opt_2",
            intent=ActionIntent(kind="pass_player", player="H10", target_player="H02", power=0.5),
            label="Pass to the overlapping Mensah",
            label_key="option.pass_to_player",
            label_args={"player": "H02"},
            p_success=0.86,
            p_success_baseline=0.84,
            value_if_success=0.09,
            xt_gain=0.056,
            decision_value=0.071,
        ),
        OptionEvaluation(
            option_id="opt_3",
            intent=ActionIntent(kind="shot", player="H10", target=(52.5, -2.9, 0.4), power=0.9),
            label="Shoot from distance",
            label_key="option.long_shot",
            p_success=0.03,
            value_if_success=1.0,
            xg=0.03,
            decision_value=0.03,
        ),
    ]


def model_outputs() -> ModelOutputs:
    models = StubModels()
    nx, ny = 21, 14
    dx, dy = 105.0 / nx, 68.0 / ny
    xs = -52.5 + dx / 2 + dx * np.arange(nx)
    ys = -34.0 + dy / 2 + dy * np.arange(ny)
    points = np.array([(x, y) for y in ys for x in xs])
    pc = models.pitch_control(
        np.array(_HOME_POS), np.array(_HOME_VEL), np.array(_AWAY_POS), np.array(_AWAY_VEL), np.array(BALL), points
    )
    xt = models.xt(np.array([BALL[0]]), np.array([BALL[1]]))
    return ModelOutputs(
        tick=TICK_PASS,
        t=T_PASS,
        team_in_possession="home",
        ball_carrier="H10",
        ball_xy=BALL,
        xt_at_ball=round(float(xt[0]), 4),
        shot_xg_now=0.03,
        pitch_control=PitchControlGrid(
            team="home",
            nx=nx,
            ny=ny,
            x0=round(float(xs[0]), 4),
            y0=round(float(ys[0]), 4),
            dx=dx,
            dy=round(dy, 6),
            values=[round(float(v), 3) for v in pc],
        ),
        options=_options(),
        model_versions=dict(models.versions),
    )


def decision_moment() -> DecisionMoment:
    return DecisionMoment(
        id="mom_017",
        tick=TICK_PASS,
        t=T_PASS,
        period=1,
        snapshot_ref=SNAP,
        team="home",
        ball_carrier="H10",
        moment_type="counter_attack",
        origin="scout",
        real_action=resolved_action(),
        real_event_id="ev_004209",
        consequence=Consequence(kind="big_chance", xt_swing=0.28, event_ids=["ev_004209", "ev_004211"]),
        options=_options(),
        scores=RewindScores(
            richness=0.74,
            stakes=0.66,
            consequence=0.81,
            salience=0.9,
            total=0.77,
            viable_options=3,
            option_entropy=1.02,
        ),
        flagged=True,
        hook="Marchetti had three options on the break. What would you have done?",
    )


def rollout_request() -> RolloutRequest:
    return RolloutRequest(snapshot_ref=SNAP, intent=_through_ball_intent(), n=1, seed=7)


def _rollout_outcome() -> RolloutOutcome:
    return RolloutOutcome(
        label="shot_saved",
        primary_success=True,
        terminal_t=2336.2,
        xg_generated=0.31,
        xt_delta=0.27,
        ended_by_event_id="ev_r_roll_9f2c_0002",
    )


def rollout_result() -> RolloutResult:
    shot = match_event().model_copy(update={"id": "ev_r_roll_9f2c_0002", "index": 2, "rollout_id": "roll_9f2c"})
    return RolloutResult(
        id="roll_9f2c",
        snapshot_ref=SNAP,
        seed=7,
        resolved=resolved_action(),
        evaluation=_options()[0],
        frames=[sim_state()],
        events=[MatchEvent.model_validate(shot.model_dump())],
        outcome=_rollout_outcome(),
    )


def rollout_distribution() -> RolloutDistribution:
    counts = {"goal": 182, "shot_saved": 151, "shot_off_target": 96, "turnover": 418, "possession_retained": 153}
    return RolloutDistribution(
        snapshot_ref=SNAP,
        intent=_through_ball_intent(),
        resolved=resolved_action(),
        n=1000,
        seed=7,
        outcome_counts=[OutcomeCount.model_validate({"label": k, "count": v}) for k, v in counts.items()],
        p_primary_success=0.574,
        p_primary_success_ci95=Interval(lo=0.543, hi=0.605),
        p_goal=0.182,
        p_goal_ci95=Interval(lo=0.159, hi=0.207),
        mean_xg=0.19,
        mean_xt_delta=0.11,
        model_p_success=0.58,
        elapsed_ms=2140.0,
        samples=[
            RolloutSample(label="goal", primary_success=True, end_xy=(38.4, -3.9), xg=0.33),
            RolloutSample(label="turnover", primary_success=False, end_xy=(35.2, -1.7), xg=0.0),
        ],
    )


def verdict() -> Verdict:
    options = _options()
    return Verdict(
        moment_id="mom_017",
        user_option=options[1],
        real_option=options[0],
        rollout_id="roll_9f2c",
        user_outcome=_rollout_outcome(),
        real_outcome_label="shot_saved",
        framing="as_expected",
        value_gap=round(options[1].decision_value - options[0].decision_value, 4),
        distribution=rollout_distribution(),
    )


def _cue(id: str, t0: float, t1: float, priority: int, anchor: CueAnchor, payload: object) -> OverlayCue:
    return OverlayCue.model_validate(
        {"id": id, "t_start": t0, "t_end": t1, "priority": priority, "anchor": anchor, "payload": payload}
    )


def overlay_cue() -> OverlayCue:
    return OverlayCue(
        id="cue_shot_004211",
        t_start=2335.0,
        t_end=2341.0,
        priority=80,
        anchor=CueAnchor(kind="screen", region="lower_third"),
        payload=ShotCardPayload(
            event_id="ev_004211",
            shooter="H09",
            xg=0.31,
            psxg=0.42,
            ball_speed_kmh=86.8,
            distance_m=_shot_qualifiers().distance_to_goal_m,
            technique="placed",
            outcome="saved",
        ),
        evidence=["ev_004211"],
    )


def cue_examples() -> list[OverlayCue]:
    """One cue per payload type."""
    player = CueAnchor(kind="player", player="H09")
    series = [SeriesPoint(t=2300.0, value=0.1), SeriesPoint(t=2330.0, value=0.4)]
    return [
        _cue("cue_tag_h09", 2331.0, 2336.0, 30, player, NameTagPayload(player="H09", emphasis="highlight")),
        _cue("cue_badge_h09", 2333.0, 2337.0, 60, player,
             ThresholdBadgePayload(player="H09", metric="top_speed", value=32.8, unit="km/h", threshold=32.0,
                                   record=True)),
        _cue("cue_pass_004209", 2334.1, 2339.0, 70, CueAnchor(kind="screen", region="bottom_left"),
             PassCardPayload(event_id="ev_004209", passer="H10", recipient="H09", length_m=22.41, ball_speed_kmh=61.9,
                             xpass=0.58, difficulty=5, technique="curled", pass_type="through_ball",
                             outcome="complete")),
        overlay_cue(),
        _cue("cue_momentum", 2340.0, 2352.0, 40, CueAnchor(kind="screen", region="top_right"),
             MomentumChartPayload(home=series, away=[SeriesPoint(t=2300.0, value=0.3)], window_s=300.0)),
        _cue("cue_chaos", 2340.0, 2352.0, 40, CueAnchor(kind="screen", region="top_left"),
             ControlChaosPayload(index=0.42, series=series, drivers={"turnover_rate": 0.4, "pressure": 0.3})),
        _cue("cue_accuracy", 2200.0, 2400.0, 20, CueAnchor(kind="screen", region="bottom_right"),
             AccuracyPanelPayload(rows=[
                 AccuracyRow(event_type="pass", precision=0.94, recall=0.91, f1=0.925, support=412),
                 AccuracyRow(event_type="pass", qualifier="pass.height=high", precision=0.8, recall=0.72, f1=0.758,
                             support=38),
                 AccuracyRow(event_type=None, qualifier="ball_speed", metric="mae", mae=3.1, unit="km/h",
                             support=1200),
             ])),
        _cue("cue_rewind_017", 2336.5, 2346.5, 90, CueAnchor(kind="screen", region="center"),
             RewindPromptPayload(moment_id="mom_017", narrative_id="nar_0411_en_casual")),
        _cue("cue_caption_0412", 2335.5, 2342.0, 50, CueAnchor(kind="screen", region="lower_third"),
             CaptionPayload(narrative_id="nar_0412_es_casual", speak=True)),
        _cue("cue_milestone", 2336.0, 2341.0, 55, CueAnchor(kind="screen", region="top_center"),
             MilestonePayload(narrative_id="nar_0413_en_analyst", milestone="first_shot_on_target", team="home",
                              player="H09")),
        _cue("cue_highlight", 2332.4, 2334.1, 65, CueAnchor(kind="pitch", pitch_xy=PASS_END),
             HighlightPayload(players=["H09", "A04"], style="trail")),
    ]  # fmt: skip


def narrative() -> Narrative:
    return Narrative(
        id="nar_0412_es_casual",
        kind="why_it_matters",
        t=T_SHOT,
        language="es",
        mode="casual",
        club_perspective="home",
        text="¡Qué contra! Marchetti la puso al hueco y Demir obligó al portero a lucirse: una ocasión de 0,31 xG.",
        evidence=[Evidence(kind="event", ref="ev_004211", field="qualifiers.xg", value=0.31)],
        fact_check=FactCheck(verified=True, attempts=1),
        source_id="nar_0412_en_casual",
        voice="es-ES-AlvaroNeural",
    )


def viewer_profile() -> ViewerProfile:
    return ViewerProfile(
        session_id="sess_a1b2", mode="analyst", favorite_club="home", language="fr", voice_enabled=True
    )


def frame_labels() -> FrameLabels:
    return FrameLabels(
        match_id=MATCH,
        frame=1234,
        t=T_PASS,
        tick=TICK_PASS,
        image="frames/001234.jpg",
        width=1920,
        height=1080,
        camera_mode="broadcast",
        camera=CameraCalibration(
            intrinsics=[1630.0, 0.0, 960.0, 0.0, 1630.0, 540.0, 0.0, 0.0, 1.0],
            extrinsics=[1.0, 0.0, 0.0, -18.0, 0.0, -0.34, -0.94, 16.0, 0.0, 0.94, -0.34, 58.0],
            fov_y_deg=36.6,
            position=(18.0, -55.0, 18.0),
        ),
        objects=[
            ObjectLabel(
                cls="player_home",
                track_id="H10",
                team="home",
                player="H10",
                box=(1002.0, 488.0, 1031.0, 571.0),
                visible_fraction=1.0,
                world=(BALL[0], BALL[1], 0.0),
                center_px=(1016.5, 571.0),
            ),
            ObjectLabel(
                cls="ball",
                track_id="ball",
                box=(1040.0, 562.0, 1047.0, 569.0),
                visible_fraction=0.9,
                world=(BALL[0] + 0.6, BALL[1] + 0.1, 0.11),
                center_px=(1043.5, 565.5),
            ),
        ],
        ball_shadow_px=(1043.6, 567.2),
    )


def stream_protocol() -> StreamProtocol:
    return StreamProtocol(server=FrameMsg(state=sim_state()), client=SeekCmd(t=2330.0))


def stats_snapshot() -> StatsSnapshot:
    return StatsSnapshot(
        t=T_SHOT,
        teams={
            "home": TeamStats(goals=1, shots=7, shots_on_target=3, xg=1.12, passes=241, passes_completed=203,
                              pass_accuracy=0.842, possession=0.54, pressures=61, ppda=9.8, field_tilt=0.58),
            "away": TeamStats(goals=1, shots=5, shots_on_target=2, xg=0.71, passes=199, passes_completed=158,
                              pass_accuracy=0.794, possession=0.46, pressures=70, ppda=11.2, field_tilt=0.42),
        },
        players={
            "H09": PlayerStats(team="home", touches=19, shots=3, goals=1, xg=0.62, distance_m=4210.0,
                               top_speed_kmh=32.8, sprints=14),
            "H10": PlayerStats(team="home", touches=48, passes=41, passes_completed=36, key_passes=3,
                               progressive_passes=7, distance_m=4580.0, top_speed_kmh=29.1, longest_pass_m=48.2),
        },
        control_chaos=0.42,
        momentum=0.35,
        milestones=["first_shot_on_target:home", "first_shot_on_target:away"],
    )  # fmt: skip


def recap() -> Recap:
    return Recap(
        match_id=MATCH,
        final_stats=stats_snapshot(),
        key_moment_ids=["mom_003", "mom_017"],
        what_ifs=[
            WhatIf(moment_id="mom_017", verdict_framing="as_expected", p_goal_alternative=0.09, p_goal_real=0.18)
        ],
        player_of_the_match="H10",
        narrative_ids=["nar_recap_en_casual"],
    )


FIXTURES: dict[str, BaseModel] = {
    "MatchMeta": match_meta(),
    "SimState": sim_state(),
    "MatchEvent": match_event(),
    "ModelOutputs": model_outputs(),
    "DecisionMoment": decision_moment(),
    "RolloutRequest": rollout_request(),
    "RolloutResult": rollout_result(),
    "RolloutDistribution": rollout_distribution(),
    "Verdict": verdict(),
    "OverlayCue": overlay_cue(),
    "Narrative": narrative(),
    "ViewerProfile": viewer_profile(),
    "FrameLabels": frame_labels(),
    "StreamProtocol": stream_protocol(),
    "StatsSnapshot": stats_snapshot(),
    "Recap": recap(),
}
"""One example per root model, keyed by class name."""

EXAMPLES: dict[str, list[BaseModel]] = {
    "MatchEvent": list(event_examples()),
    "OverlayCue": list(cue_examples()),
}
"""Variant coverage: every qualifier kind and every cue payload type."""
