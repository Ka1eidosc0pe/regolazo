"""Physical constants, pitch landmarks and event thresholds (docs/event_taxonomy.md section 8) in one place.

Every threshold the simulator applies when labelling ground-truth events lives here, so tuning never means
hunting for inlined numbers. Values marked [INVENTED] in the taxonomy are tagged the same way below.
"""

from typing import Final

from shared.schemas.common import GOAL_HEIGHT_M, GOAL_WIDTH_M, PITCH_LENGTH_M, PITCH_WIDTH_M

SIM_HZ: Final = 25
DT: Final = 1.0 / SIM_HZ

# --- Pitch (world frame, centre origin) ---------------------------------------------------------------------
HALF_LENGTH: Final = PITCH_LENGTH_M / 2  # 52.5
HALF_WIDTH: Final = PITCH_WIDTH_M / 2  # 34.0
HALF_GOAL: Final = GOAL_WIDTH_M / 2  # 3.66
GOAL_HEIGHT: Final = GOAL_HEIGHT_M  # 2.44
GOAL_DEPTH: Final = 2.0
POST_RADIUS: Final = 0.06
BOX_X: Final = 36.0  # penalty area front edge in the attacking frame (x >= 36)
BOX_HALF_WIDTH: Final = 20.16
SIX_YARD_X: Final = 47.0
SIX_YARD_HALF_WIDTH: Final = 9.16
PENALTY_SPOT_X: Final = 41.5
CENTRE_CIRCLE_R: Final = 9.15
FINAL_THIRD_X: Final = 17.5

# --- Ball ---------------------------------------------------------------------------------------------------
BALL_RADIUS: Final = 0.11
BALL_MASS: Final = 0.43
GRAVITY: Final = 9.81
AIR_DENSITY: Final = 1.2
DRAG_COEFF: Final = 0.25
BALL_AREA: Final = 3.14159265 * BALL_RADIUS**2
DRAG_K: Final = 0.5 * AIR_DENSITY * DRAG_COEFF * BALL_AREA / BALL_MASS
"""Quadratic drag: a = -DRAG_K * |v| * v (1/m)."""
MAGNUS_K: Final = 0.005
"""Magnus acceleration a = MAGNUS_K * (omega x v)."""
RESTITUTION: Final = 0.62
"""Vertical coefficient of restitution on grass."""
BOUNCE_FRICTION: Final = 0.32
"""Fraction of the contact-point slip velocity removed per bounce."""
ROLL_DECEL: Final = 0.78
"""Rolling resistance on grass, m/s^2."""
SPIN_DECAY_AIR: Final = 0.08
SPIN_DECAY_GROUND: Final = 3.0
REST_SPEED: Final = 0.05
WOODWORK_RESTITUTION: Final = 0.6
NET_DAMPING: Final = 6.0

# --- Players ------------------------------------------------------------------------------------------------
PLAYER_RADIUS: Final = 0.35
CONTROL_RADIUS: Final = 1.0
"""Horizontal distance at which a player can play the ball with the feet."""
FOOT_REACH_Z: Final = 0.9
CHEST_REACH_Z: Final = 1.55
HEAD_STANDING_Z: Final = 1.85
GK_HAND_REACH_Z: Final = 2.5
MAX_TURN_RATE: Final = 9.0
"""rad/s at rest; decreases with speed."""
JUMP_SPEED_MIN: Final = 2.6
JUMP_SPEED_MAX: Final = 3.6

# --- Event thresholds (docs/event_taxonomy.md section 8) ---------------------------------------------------
PASS_GROUND_MAX_Z: Final = 0.30  # [INVENTED]
PASS_LOW_MAX_Z: Final = 1.5
PASS_SHORT_MAX: Final = 15.0
PASS_MEDIUM_MAX: Final = 30.0
SWITCH_LATERAL_M: Final = 34.0
BACK_PASS_ANGLE: Final = 135.0 * 3.14159265 / 180.0
BACK_PASS_MIN_LEN: Final = 3.0
CARRY_MIN_M: Final = 3.0
TAKE_ON_RADIUS: Final = 2.5  # [INVENTED]
KEY_PASS_WINDOW_S: Final = 5.0  # [INVENTED]
FIRST_TIME_S: Final = 0.4  # [INVENTED]
COUNTER_DIRECTNESS: Final = 0.75
COUNTER_ADVANCE_M: Final = 16.5
COUNTER_WINDOW_S: Final = 15.0  # [INVENTED]
COUNTERPRESS_WINDOW_S: Final = 5.0
SET_PIECE_WINDOW_S: Final = 10.0  # [INVENTED]
TRANSITION_WINDOW_S: Final = 8.0  # [INVENTED]
PRESSURE_RADIUS: Final = 5.0
PRESSURE_CLOSE_RADIUS: Final = 2.0
PRESSURE_MIN_S: Final = 0.4  # [INVENTED]
PRESSURE_CLOSING_MPS: Final = -0.5
HIR_MPS: Final = 5.5
SPRINT_MPS: Final = 7.0
SPEED_EVENT_MIN_S: Final = 1.0  # [INVENTED]
ACCEL_THRESHOLD: Final = 3.0
ACCEL_MIN_S: Final = 0.5  # [INVENTED]
ACCEL_MIN_DV: Final = 1.5
RUN_MIN_SPEED: Final = 4.0
RUN_MIN_S: Final = 1.0
RUN_MIN_DISPLACEMENT: Final = 5.0
STEP_UP_ADVANCE_M: Final = 4.0
STEP_UP_WINDOW_S: Final = 2.0
TOUCH_CLEAN_M: Final = 1.5
TOUCH_LOOSE_M: Final = 3.0
CORNER_BULGE_M: Final = 1.0
