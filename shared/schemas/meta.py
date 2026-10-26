"""MatchMeta: teams, players, attributes, kits and the seed that defines a synthetic match."""

from typing import Annotated, Literal

from pydantic import Field

from shared.schemas.common import (
    GOAL_HEIGHT_M,
    GOAL_WIDTH_M,
    PITCH_LENGTH_M,
    PITCH_WIDTH_M,
    SCHEMA_VERSION,
    Contract,
    PlayerId,
    TeamSide,
)

Rating = Annotated[int, Field(ge=1, le=99)]
HexColor = Annotated[str, Field(pattern=r"^#[0-9a-fA-F]{6}$", examples=["#0b3d91"])]

Position = Literal[
    "GK",
    "RB",
    "RCB",
    "CB",
    "LCB",
    "LB",
    "RWB",
    "LWB",
    "DM",
    "CM",
    "RCM",
    "LCM",
    "AM",
    "RM",
    "LM",
    "RW",
    "LW",
    "CF",
    "ST",
]

Archetype = Literal[
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
]

Foot = Literal["left", "right"]
KitPattern = Literal["plain", "stripes", "hoops", "sash", "halves", "chevron"]
CrestShape = Literal["shield", "circle", "roundel", "diamond", "hexagon"]
CrestEmblem = Literal["star", "wave", "bird", "lion", "tower", "bolt", "anchor", "oak"]
Weather = Literal["clear_night", "light_rain_night", "misty_night"]


class PlayerAttributes(Contract):
    """Synthetic 1-99 ratings (PLAN.md section 2.6). Goalkeeping ratings are present for every player."""

    pace: Rating
    acceleration: Rating
    stamina: Rating
    strength: Rating
    jumping: Rating
    short_passing: Rating
    long_passing: Rating
    vision: Rating
    crossing: Rating
    curve: Rating
    technique: Rating
    dribbling: Rating
    finishing: Rating
    shot_power: Rating
    long_shots: Rating
    volleys: Rating
    heading: Rating
    composure: Rating
    tackling: Rating
    interceptions: Rating
    marking: Rating
    positioning: Rating
    gk_reflexes: Rating
    gk_diving: Rating
    gk_handling: Rating
    gk_positioning: Rating
    gk_distribution: Rating


class PhysicalParams(Contract):
    """Kinematic limits used by the simulator, derived from attributes and real tracking distributions."""

    max_speed_mps: Annotated[float, Field(gt=0, le=12.5)]
    max_accel_mps2: Annotated[float, Field(gt=0, le=10.0)]
    max_decel_mps2: Annotated[float, Field(gt=0, le=12.0)]
    reaction_time_s: Annotated[float, Field(ge=0, le=1.0)]
    jump_reach_m: Annotated[float, Field(gt=0, le=3.6)] = 2.6
    """Highest point the head or hands can reach on a jump."""


class PlayerMeta(Contract):
    id: PlayerId
    team: TeamSide
    first_name: str
    last_name: str
    display_name: str
    """Name on the back of the shirt and on overlays."""
    shirt_number: Annotated[int, Field(ge=1, le=99)]
    position: Position
    archetype: Archetype
    preferred_foot: Foot
    weak_foot: Annotated[int, Field(ge=1, le=5)]
    height_cm: Annotated[int, Field(ge=150, le=210)]
    nationality: str
    """Fictional-world flavour only (ISO 3166 alpha-3 code); no real players."""
    attributes: PlayerAttributes
    physical: PhysicalParams
    starter: bool = True


class KitColors(Contract):
    primary: HexColor
    secondary: HexColor
    accent: HexColor
    number: HexColor
    """Colour of shirt numbers and names."""
    pattern: KitPattern = "plain"


class CrestSpec(Contract):
    """Parameters for the generated (original) crest."""

    shape: CrestShape
    monogram: Annotated[str, Field(min_length=1, max_length=3)]
    primary: HexColor
    secondary: HexColor
    emblem: CrestEmblem


class TeamMeta(Contract):
    side: TeamSide
    name: str
    short_name: Annotated[str, Field(min_length=3, max_length=3)]
    """Three-letter scoreboard code."""
    city: str
    formation: Annotated[str, Field(pattern=r"^\d(-\d){2,4}$", examples=["4-3-3"])]
    kit: KitColors
    gk_kit: KitColors
    crest: CrestSpec
    players: list[PlayerMeta]
    """Starters first (11, goalkeeper first), then substitutes."""


class PitchSpec(Contract):
    length_m: float = PITCH_LENGTH_M
    width_m: float = PITCH_WIDTH_M
    goal_width_m: float = GOAL_WIDTH_M
    goal_height_m: float = GOAL_HEIGHT_M


class MatchMeta(Contract):
    schema_version: str = SCHEMA_VERSION
    match_id: str
    seed: int
    sim_hz: Annotated[int, Field(gt=0)] = 25
    period_duration_s: Annotated[float, Field(gt=0)] = 2700.0
    pitch: PitchSpec = PitchSpec()
    home: TeamMeta
    away: TeamMeta
    home_attacks_positive_x_first_half: bool = True
    kickoff_team: TeamSide = "home"
    venue: str
    weather: Weather = "clear_night"
