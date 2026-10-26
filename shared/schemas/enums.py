"""Football vocabulary shared by simulator ground truth, the Spotter, models and the UI.

Definitions and thresholds live in docs/event_taxonomy.md (section 7 lists these enums). Values are snake_case
string literals so they read the same in Python, JSON and TypeScript.
"""

from typing import Literal

EventType = Literal[
    # Tier A: on-ball
    "pass",
    "ball_receipt",
    "carry",
    "take_on",
    "shot",
    "duel",
    "interception",
    "block",
    "clearance",
    "ball_recovery",
    "dispossessed",
    "miscontrol",
    "deflection",
    "foul_committed",
    "foul_won",
    "gk_action",
    "restart",
    "ball_out_of_play",
    "offside",
    "own_goal",
    "period_start",
    "period_end",
    # Tier B: off-ball, pressure and physical (from tracking)
    "pressure",
    "pressing_sequence",
    "off_ball_run",
    "sprint",
    "high_intensity_run",
    "acceleration",
    "line_step_up",
]

TIER_A_EVENT_TYPES: tuple[EventType, ...] = (
    "pass",
    "ball_receipt",
    "carry",
    "take_on",
    "shot",
    "duel",
    "interception",
    "block",
    "clearance",
    "ball_recovery",
    "dispossessed",
    "miscontrol",
    "deflection",
    "foul_committed",
    "foul_won",
    "gk_action",
    "restart",
    "ball_out_of_play",
    "offside",
    "own_goal",
    "period_start",
    "period_end",
)
TIER_B_EVENT_TYPES: tuple[EventType, ...] = (
    "pressure",
    "pressing_sequence",
    "off_ball_run",
    "sprint",
    "high_intensity_run",
    "acceleration",
    "line_step_up",
)

BodyPart = Literal[
    "right_foot",
    "left_foot",
    "head",
    "chest",
    "other",
    "hands",
    "no_touch",
    "left_hand",
    "right_hand",
    "both_hands",
]

PassHeight = Literal["ground", "low", "high"]
PassLength = Literal["short", "medium", "long"]
PassTechnique = Literal["standard", "driven", "lofted", "chipped", "curled", "outside_of_boot", "backheel"]
PassType = Literal["standard", "cross", "cutback", "through_ball", "layoff", "flick_on", "back_pass"]
CornerDelivery = Literal["inswinging", "outswinging", "straight"]
GkDistribution = Literal["throw", "punt", "drop_kick", "goal_kick", "pass"]
PassOutcome = Literal["complete", "incomplete", "intercepted", "out_of_play", "offside"]

ReceiptOutcome = Literal["complete", "incomplete"]
TouchQuality = Literal["clean", "loose", "miscontrol"]
BodyOrientation = Literal["open", "half_turn", "back_to_goal"]

CarryOutcome = Literal["retained", "lost", "end_of_period"]
TakeOnOutcome = Literal["success", "fail"]

ShotTechnique = Literal[
    "driven",
    "placed",
    "finesse",
    "chip",
    "volley",
    "half_volley",
    "header",
    "diving_header",
    "toe_poke",
    "backheel",
    "overhead_kick",
]
ShotSituation = Literal[
    "open_play",
    "counter_attack",
    "corner_phase",
    "free_kick_phase",
    "throw_in_phase",
    "direct_free_kick",
    "penalty",
    "kick_off",
]
ShotOutcome = Literal["goal", "saved", "saved_to_post", "off_target", "blocked", "woodwork", "wayward"]
GkPosition = Literal["set", "moving", "prone"]

DuelType = Literal["tackle", "aerial", "fifty_fifty"]
DuelOutcome = Literal["won", "lost"]
PossessionAfter = Literal["tackler", "teammate", "opponent", "out_of_play", "loose"]
InterceptionOutcome = Literal["won", "success_in_play", "success_out", "lost"]
BlockType = Literal["shot_block", "pass_block", "cross_block"]
ClearanceOutcome = Literal["out_of_play", "in_play_safe", "in_play_danger"]
DeflectionTarget = Literal["shot", "pass", "cross"]
FoulType = Literal["foul", "handball", "dangerous_play", "dive", "six_seconds", "backpass_pickup"]
Card = Literal["none", "yellow", "second_yellow", "red"]

GkActionType = Literal["save", "claim", "punch", "sweeper", "rush_out", "pick_up", "goal_conceded"]
GkOutcome = Literal[
    "catch",
    "parry_safe",
    "parry_danger",
    "parry_out",
    "to_post",
    "success",
    "fail",
    "collected_twice",
    "claim",
    "clear",
    "won",
    "lost_in_play",
    "lost_out",
    "in_play_safe",
    "in_play_danger",
    "out_of_play",
    "no_touch",
    "touched_in",
]
GkTechnique = Literal["standing", "diving"]

RestartType = Literal[
    "kick_off",
    "throw_in",
    "goal_kick",
    "corner",
    "free_kick_direct",
    "free_kick_indirect",
    "penalty",
    "drop_ball",
]
OutLine = Literal["touchline", "goal_line"]

PressedRole = Literal["carrier", "receiver", "releaser"]
PressureOutcome = Literal["forced_turnover", "forced_hurried_action", "none"]
PressSequenceOutcome = Literal["success", "broken", "out", "whistle"]
RunType = Literal["run_in_behind", "overlap", "underlap", "checking_run", "decoy", "support"]
RunOutcome = Literal["targeted_received", "targeted_missed", "not_targeted"]
AccelType = Literal["acceleration", "deceleration"]
StepUpOutcome = Literal["trap_success", "trap_beaten", "reset"]

PhaseOfPlay = Literal[
    "build_up",
    "progression",
    "final_third",
    "attacking_transition",
    "defensive_transition",
    "counter_attack",
    "set_piece",
    "out_of_possession",
    "dead_ball",
]
DefensiveBlock = Literal["high_press", "mid_block", "low_block"]
PlayPattern = Literal[
    "regular_play",
    "from_corner",
    "from_free_kick",
    "from_throw_in",
    "from_goal_kick",
    "from_keeper",
    "from_kick_off",
    "from_counter",
    "other",
]
