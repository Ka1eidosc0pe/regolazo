"""Example instances of every root contract. Written to shared/fixtures/ and to typed TS fixtures by export.py.

They double as documentation: one realistic moment (a counter-attack ending in a through ball and a shot) is
followed through every contract.
"""

from pydantic import BaseModel

from shared.schemas.actions import ActionIntent, BallLaunch, ExecutionEllipse, ResolvedAction
from shared.schemas.events import MatchEvent, PassQ, PressureQ, ShotQ
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
from shared.schemas.overlay import CueAnchor, OverlayCue, ShotCardPayload
from shared.schemas.rollout import (
    Interval,
    RolloutDistribution,
    RolloutOutcome,
    RolloutRequest,
    RolloutResult,
)
from shared.schemas.state import BallState, PlayerState, Possession, RefereeState, Score, SimState

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
        ("Tomás", "Varga"),
        ("Kofi", "Mensah"),
        ("Rhys", "Calloway"),
        ("Idris", "Okafor"),
        ("Lukas", "Brandt"),
        ("Mateo", "Quiroga"),
        ("Sami", "Haddad"),
        ("Elio", "Marchetti"),
        ("Dayo", "Adeyemi"),
        ("Niall", "Keane"),
        ("Yusuf", "Demir"),
    ],
    "away": [
        ("Pablo", "Ferrán"),
        ("Jonah", "Whitlock"),
        ("Emeka", "Obi"),
        ("Henrik", "Solberg"),
        ("Ravi", "Nair"),
        ("Andrés", "Molina"),
        ("Théo", "Garnier"),
        ("Kenji", "Arai"),
        ("Malik", "Diallo"),
        ("Callum", "Reid"),
        ("Bruno", "Saraiva"),
    ],
}


def _player(side: str, i: int) -> PlayerMeta:
    number, position, archetype = _LINEUP[i]
    first, last = _NAMES[side][i]
    base = 55 + (i * 7 + (3 if side == "home" else 0)) % 30
    attrs = {name: min(99, max(1, base + (j * 13 + i * 5) % 21 - 10)) for j, name in enumerate(_ATTRS)}
    if position != "GK":
        attrs.update({k: 20 + (i * 3) % 15 for k in _ATTRS if k.startswith("gk_")})
    pace = attrs["pace"]
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
                max_speed_mps=round(7.6 + pace / 99 * 2.4, 2),
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
            gk_kit=KitColors(primary="#2dc653", secondary="#0d1b2a", accent="#ffffff", number="#0d1b2a"),
            crest=CrestSpec(
                shape="shield" if home else "roundel",
                monogram="HC" if home else "RF",
                primary="#0b3d91" if home else "#c8102e",
                secondary="#4cc9f0" if home else "#f5b700",
                emblem="anchor" if home else "bolt",
            ),
            players=[_player(side, i) for i in range(11)],
        )

    return MatchMeta(match_id="demo-0001", seed=20261026, home=team("home"), away=team("away"), venue="Lantern Park")


def sim_state() -> SimState:
    meta = match_meta()
    players: list[PlayerState] = []
    for k, team in enumerate((meta.home, meta.away)):
        sign = -1.0 if k == 0 else 1.0
        for i, p in enumerate(team.players):
            x = sign * (50.0 - 8.0 * (i // 3 + (0 if i == 0 else 1))) + (12.0 if k == 0 else -4.0)
            y = -25.0 + (i * 17.0) % 50.0
            players.append(
                PlayerState(
                    id=p.id,
                    pos=(round(x, 2), round(y, 2)),
                    vel=(4.1, 0.6) if k == 0 else (-2.0, 0.3),
                    facing=0.0 if k == 0 else 3.14159,
                    anim="sprint" if i in (9, 10) else "jog",
                    anim_t=0.8,
                    stamina=0.86,
                )
            )
    return SimState(
        tick=58_310,
        t=2332.4,
        period=1,
        status="first_half",
        ball=BallState(pos=(18.4, 6.1, 0.11), vel=(6.2, 0.4, 0.0), last_touch="H10"),
        players=players,
        referee=RefereeState(pos=(10.0, 12.0), vel=(3.0, 0.0)),
        possession=Possession(team="home", player="H10", sequence_id=214),
        phase="counter_attack",
        score=Score(home=1, away=1),
        attacking_positive_x="home",
        keyframe=True,
        snapshot_ref="snap:demo-0001:58310",
    )


def _through_ball_intent() -> ActionIntent:
    return ActionIntent(
        kind="pass_space",
        player="H10",
        target=(38.0, -4.0, 0.0),
        path=[(18.4, 6.1), (27.0, 3.0), (33.0, -1.5), (38.0, -4.0)],
        curve=0.35,
        power=0.62,
        source="fan",
    )


def resolved_action() -> ResolvedAction:
    return ResolvedAction(
        intent=_through_ball_intent(),
        action_type="pass",
        body_part="left_foot",
        pass_technique="curled",
        pass_type="through_ball",
        pass_height="ground",
        intended_end=(38.0, -4.0, 0.0),
        launch=BallLaunch(speed_mps=17.5, elevation_rad=0.03, azimuth_rad=-0.52, spin=(0.0, 0.0, -8.0)),
        curve_applied=0.3,
        ellipse=ExecutionEllipse(center=(38.0, -4.0, 0.0), semi_major_m=1.8, semi_minor_m=0.9, angle_rad=-0.52),
        preview_arc=[(18.4, 6.1, 0.11), (25.9, 3.4, 0.11), (32.4, -0.6, 0.11), (38.0, -4.0, 0.11)],
        execution_quality=0.84,
    )


def match_event() -> MatchEvent:
    return MatchEvent(
        id="ev_004211",
        index=4211,
        source="ground_truth",
        type="shot",
        period=1,
        t=2334.9,
        tick=58_372,
        team="home",
        player="H09",
        location=(39.1, -3.2, 0.11),
        end_location=(52.5, 2.1, 0.6),
        possession_id=214,
        possession_team="home",
        play_pattern="from_counter",
        phase_of_play="counter_attack",
        under_pressure=True,
        related_events=["ev_004209", "ev_004212"],
        qualifiers=ShotQ(
            keeper="A01",
            outcome="saved",
            body_part="right_foot",
            technique="placed",
            situation="counter_attack",
            ball_speed_mps=24.1,
            distance_to_goal_m=13.8,
            angle_rad=0.47,
            peak_height_m=0.7,
            on_target=True,
            xg=0.31,
            psxg=0.42,
            first_time=True,
            n_defenders_in_cone=1,
            gk_position="moving",
            key_pass_id="ev_004209",
            gk_event_id="ev_004212",
            freeze_frame_ref="snap:demo-0001:58372",
        ),
    )


def extra_events() -> list[MatchEvent]:
    """Other qualifier shapes, used by the round-trip tests."""
    return [
        MatchEvent(
            id="ev_004209",
            index=4209,
            source="ground_truth",
            type="pass",
            period=1,
            t=2332.4,
            tick=58_310,
            team="home",
            player="H10",
            location=(18.4, 6.1, 0.11),
            end_location=(38.6, -3.6, 0.11),
            possession_id=214,
            possession_team="home",
            play_pattern="from_counter",
            phase_of_play="counter_attack",
            qualifiers=PassQ(
                recipient="H09",
                outcome="complete",
                height="ground",
                length_class="medium",
                body_part="left_foot",
                technique="curled",
                pass_type="through_ball",
                length_m=22.4,
                angle_rad=-0.45,
                ball_speed_mps=17.2,
                peak_height_m=0.12,
                duration_s=1.7,
                is_progressive=True,
                is_line_breaking=True,
                lines_broken=1,
                is_key_pass=True,
                assisted_shot_id="ev_004211",
                xpass=0.58,
            ),
        ),
        MatchEvent(
            id="ev_d_000877",
            index=877,
            source="detected",
            confidence=0.81,
            type="pressure",
            period=1,
            t=2333.1,
            duration_s=1.2,
            team="away",
            player="A05",
            location=(30.2, -1.0, 0.0),
            qualifiers=PressureQ(
                pressed_player="H09", pressed_role="receiver", min_distance_m=1.6, closing_speed_mps=2.4
            ),
        ),
        MatchEvent(
            id="ev_000001",
            index=0,
            source="ground_truth",
            type="period_start",
            period=1,
            t=0.0,
            tick=0,
            team=None,
            player=None,
            location=None,
        ),
    ]


def model_outputs() -> ModelOutputs:
    nx, ny = 8, 5
    values = [
        round(min(1.0, max(0.0, 0.5 + 0.06 * (ix - 3) - 0.04 * abs(iy - 2))), 3) for iy in range(ny) for ix in range(nx)
    ]
    return ModelOutputs(
        tick=58_310,
        t=2332.4,
        team_in_possession="home",
        ball_carrier="H10",
        ball_xy=(18.4, 6.1),
        xt_at_ball=0.034,
        shot_xg_now=0.03,
        pitch_control=PitchControlGrid(team="home", nx=nx, ny=ny, x0=-49.0, y0=-27.2, cell_m=13.6, values=values),
        options=options(),
        model_versions={
            "xg": "xg-gbm-v1",
            "xpass": "xpass-gbm-v1",
            "pitch_control": "spearman2018",
            "xt": "xt-16x12-v1",
        },
    )


def options() -> list[OptionEvaluation]:
    return [
        OptionEvaluation(
            option_id="opt_1",
            intent=_through_ball_intent(),
            label="Through ball to Demir",
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
            p_success=0.03,
            value_if_success=1.0,
            xg=0.03,
            decision_value=0.03,
        ),
    ]


def decision_moment() -> DecisionMoment:
    return DecisionMoment(
        id="mom_017",
        tick=58_310,
        t=2332.4,
        period=1,
        snapshot_ref="snap:demo-0001:58310",
        team="home",
        ball_carrier="H10",
        moment_type="counter_attack",
        origin="scout",
        real_action=resolved_action(),
        real_event_id="ev_004209",
        consequence=Consequence(kind="big_chance", xt_swing=0.28, event_ids=["ev_004209", "ev_004211"]),
        options=options(),
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
    return RolloutRequest(snapshot_ref="snap:demo-0001:58310", intent=options()[1].intent, n=1000, seed=7)


def rollout_result() -> RolloutResult:
    return RolloutResult(
        id="roll_9f2c",
        snapshot_ref="snap:demo-0001:58310",
        seed=7,
        resolved=resolved_action(),
        frames=[sim_state()],
        events=[match_event()],
        outcome=RolloutOutcome(
            label="shot_saved",
            primary_success=True,
            terminal_t=2336.2,
            xg_generated=0.31,
            xt_delta=0.27,
            ended_by_event_id="ev_004212",
        ),
    )


def rollout_distribution() -> RolloutDistribution:
    return RolloutDistribution(
        snapshot_ref="snap:demo-0001:58310",
        intent=_through_ball_intent(),
        resolved=resolved_action(),
        n=1000,
        seed=7,
        outcome_counts={
            "goal": 182,
            "shot_saved": 151,
            "shot_off_target": 96,
            "turnover": 418,
            "possession_retained": 153,
        },
        p_primary_success=0.574,
        p_primary_success_ci95=Interval(lo=0.543, hi=0.605),
        p_goal=0.182,
        p_goal_ci95=Interval(lo=0.159, hi=0.207),
        mean_xg=0.19,
        mean_xt_delta=0.11,
        model_p_success=0.58,
        elapsed_ms=2140.0,
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
            distance_m=13.8,
            technique="placed",
            outcome="saved",
        ),
        evidence=["ev_004211"],
    )


def narrative() -> Narrative:
    return Narrative(
        id="nar_0412_es_casual",
        kind="why_it_matters",
        t=2334.9,
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


FIXTURES: dict[str, BaseModel] = {
    "MatchMeta": match_meta(),
    "SimState": sim_state(),
    "MatchEvent": match_event(),
    "ModelOutputs": model_outputs(),
    "DecisionMoment": decision_moment(),
    "RolloutRequest": rollout_request(),
    "RolloutResult": rollout_result(),
    "RolloutDistribution": rollout_distribution(),
    "OverlayCue": overlay_cue(),
    "Narrative": narrative(),
    "ViewerProfile": viewer_profile(),
}
"""One example per root model, keyed by class name."""
