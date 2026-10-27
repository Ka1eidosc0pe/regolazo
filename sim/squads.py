"""Generated squads: fictional clubs, kits, crests, players with archetypes and coherent attributes (MatchMeta).

Formations, slot shapes and shirt numbers are defined here and reused by the tactics module.
"""

from dataclasses import dataclass
from typing import Final, get_args

import numpy as np

from shared.model_api import PhysicalPriors
from shared.schemas.meta import (
    Archetype,
    CrestEmblem,
    CrestShape,
    CrestSpec,
    KitColors,
    KitPattern,
    MatchMeta,
    PhysicalParams,
    PlayerAttributes,
    PlayerMeta,
    Position,
    TeamMeta,
    Weather,
)
from sim.constants import GRAVITY
from sim.names import CITIES, CLUB_SUFFIXES, FAMOUS, GK_COLOURS, KIT_COLOURS, REGIONS, VENUE_WORDS


@dataclass(frozen=True)
class Slot:
    position: Position
    shirt: int
    out_xy: tuple[float, float]
    """Out-of-possession offset from the team anchor (team frame: attacking +x, +y = left)."""
    in_xy: tuple[float, float]
    """In-possession offset."""
    line: int
    """0 = goalkeeper, 1 = defence, 2 = midfield, 3 = attack."""


def _s(pos: Position, shirt: int, out_xy: tuple[float, float], in_xy: tuple[float, float], line: int) -> Slot:
    return Slot(pos, shirt, out_xy, in_xy, line)


FORMATIONS: Final[dict[str, tuple[Slot, ...]]] = {
    "4-3-3": (
        _s("GK", 1, (0, 0), (0, 0), 0),
        _s("RB", 2, (-14, -18), (-5, -27), 1),
        _s("RCB", 5, (-15, -6.5), (-16, -10), 1),
        _s("LCB", 4, (-15, 6.5), (-16, 10), 1),
        _s("LB", 3, (-14, 18), (-5, 27), 1),
        _s("DM", 6, (-7, 0), (-7, 0), 2),
        _s("RCM", 8, (-1, -10), (5, -13), 2),
        _s("LCM", 10, (-1, 10), (5, 13), 2),
        _s("RW", 7, (8, -20), (18, -26), 3),
        _s("ST", 9, (12, 0), (22, 0), 3),
        _s("LW", 11, (8, 20), (18, 26), 3),
    ),
    "4-2-3-1": (
        _s("GK", 1, (0, 0), (0, 0), 0),
        _s("RB", 2, (-14, -18), (-5, -27), 1),
        _s("RCB", 5, (-15, -6.5), (-16, -10), 1),
        _s("LCB", 4, (-15, 6.5), (-16, 10), 1),
        _s("LB", 3, (-14, 18), (-5, 27), 1),
        _s("DM", 6, (-6, -6), (-5, -7), 2),
        _s("CM", 8, (-6, 6), (-2, 8), 2),
        _s("RW", 7, (3, -19), (16, -25), 3),
        _s("AM", 10, (4, 0), (12, 0), 2),
        _s("LW", 11, (3, 19), (16, 25), 3),
        _s("ST", 9, (13, 0), (24, 0), 3),
    ),
    "4-4-2": (
        _s("GK", 1, (0, 0), (0, 0), 0),
        _s("RB", 2, (-14, -18), (-5, -27), 1),
        _s("RCB", 5, (-15, -6.5), (-16, -10), 1),
        _s("LCB", 4, (-15, 6.5), (-16, 10), 1),
        _s("LB", 3, (-14, 18), (-5, 27), 1),
        _s("RM", 7, (-3, -20), (12, -27), 2),
        _s("RCM", 8, (-4, -7), (0, -9), 2),
        _s("LCM", 6, (-4, 7), (2, 9), 2),
        _s("LM", 11, (-3, 20), (12, 27), 2),
        _s("ST", 9, (10, -4), (22, -5), 3),
        _s("CF", 10, (10, 4), (20, 6), 3),
    ),
}

ARCHETYPES_BY_POSITION: Final[dict[str, tuple[tuple[Archetype, float], ...]]] = {
    "GK": (("sweeper_keeper", 0.4), ("shot_stopper", 0.6)),
    "FB": (("overlapping_fullback", 0.65), ("inverted_fullback", 0.35)),
    "CB": (("ball_playing_centre_back", 0.45), ("stopper", 0.55)),
    "DM": (("deep_lying_playmaker", 0.5), ("ball_winning_midfielder", 0.5)),
    "CM": (
        ("box_to_box", 0.45),
        ("deep_lying_playmaker", 0.15),
        ("advanced_playmaker", 0.25),
        ("ball_winning_midfielder", 0.15),
    ),
    "AM": (("advanced_playmaker", 0.85), ("inside_forward", 0.15)),
    "W": (("touchline_winger", 0.5), ("inside_forward", 0.5)),
    "ST": (("poacher", 0.45), ("target_forward", 0.35), ("inside_forward", 0.2)),
}

POSITION_GROUP: Final[dict[str, str]] = {
    "GK": "GK",
    "RB": "FB",
    "LB": "FB",
    "RWB": "FB",
    "LWB": "FB",
    "RCB": "CB",
    "CB": "CB",
    "LCB": "CB",
    "DM": "DM",
    "CM": "CM",
    "RCM": "CM",
    "LCM": "CM",
    "AM": "AM",
    "RM": "W",
    "LM": "W",
    "RW": "W",
    "LW": "W",
    "ST": "ST",
    "CF": "ST",
}

ATTRS: Final = tuple(PlayerAttributes.model_fields)
OUTFIELD_ATTRS: Final = tuple(a for a in ATTRS if not a.startswith("gk_"))
GK_ATTRS: Final = tuple(a for a in ATTRS if a.startswith("gk_"))

PROFILE: Final[dict[str, dict[str, float]]] = {
    "advanced_playmaker": {
        "vision": 15,
        "curve": 13,
        "long_passing": 10,
        "short_passing": 10,
        "technique": 13,
        "dribbling": 6,
        "composure": 8,
        "crossing": 4,
        "long_shots": 6,
        "finishing": -2,
        "tackling": -25,
        "interceptions": -18,
        "marking": -25,
        "strength": -10,
        "heading": -15,
        "jumping": -10,
        "stamina": -3,
    },
    "deep_lying_playmaker": {
        "short_passing": 12,
        "long_passing": 14,
        "vision": 12,
        "composure": 8,
        "technique": 6,
        "curve": 4,
        "tackling": -6,
        "interceptions": 2,
        "marking": -5,
        "pace": -8,
        "acceleration": -6,
        "finishing": -18,
        "heading": -8,
    },
    "box_to_box": {
        "stamina": 14,
        "tackling": 2,
        "interceptions": 2,
        "short_passing": 3,
        "long_shots": 4,
        "finishing": -6,
        "strength": 4,
        "pace": 2,
        "jumping": 2,
        "heading": -2,
    },
    "ball_winning_midfielder": {
        "tackling": 14,
        "interceptions": 14,
        "marking": 10,
        "strength": 8,
        "stamina": 8,
        "vision": -8,
        "curve": -12,
        "finishing": -22,
        "dribbling": -8,
        "technique": -6,
        "positioning": 4,
    },
    "poacher": {
        "finishing": 16,
        "composure": 10,
        "positioning": 12,
        "acceleration": 8,
        "pace": 4,
        "volleys": 10,
        "heading": 2,
        "tackling": -35,
        "marking": -35,
        "interceptions": -30,
        "long_passing": -15,
        "vision": -8,
        "stamina": -6,
    },
    "target_forward": {
        "heading": 18,
        "jumping": 14,
        "strength": 16,
        "finishing": 8,
        "volleys": 4,
        "pace": -10,
        "acceleration": -10,
        "dribbling": -10,
        "tackling": -30,
        "marking": -30,
        "interceptions": -25,
    },
    "inside_forward": {
        "dribbling": 14,
        "finishing": 8,
        "curve": 10,
        "long_shots": 10,
        "acceleration": 10,
        "pace": 8,
        "technique": 8,
        "tackling": -30,
        "marking": -30,
        "heading": -12,
        "strength": -8,
    },
    "touchline_winger": {
        "pace": 14,
        "acceleration": 12,
        "crossing": 16,
        "dribbling": 10,
        "stamina": 4,
        "finishing": -8,
        "heading": -14,
        "tackling": -25,
        "marking": -25,
        "strength": -10,
    },
    "overlapping_fullback": {
        "pace": 10,
        "acceleration": 8,
        "stamina": 14,
        "crossing": 10,
        "tackling": 4,
        "marking": 2,
        "finishing": -25,
        "heading": -6,
        "vision": -6,
    },
    "inverted_fullback": {
        "short_passing": 8,
        "vision": 4,
        "technique": 4,
        "tackling": 4,
        "interceptions": 6,
        "positioning": 4,
        "crossing": -4,
        "pace": 2,
        "finishing": -25,
    },
    "ball_playing_centre_back": {
        "long_passing": 10,
        "short_passing": 6,
        "composure": 8,
        "vision": 2,
        "tackling": 8,
        "interceptions": 10,
        "marking": 8,
        "heading": 8,
        "strength": 6,
        "pace": -6,
        "dribbling": -10,
        "finishing": -30,
        "curve": -10,
    },
    "stopper": {
        "tackling": 14,
        "marking": 14,
        "strength": 14,
        "heading": 14,
        "jumping": 10,
        "interceptions": 8,
        "positioning": 8,
        "pace": -8,
        "dribbling": -20,
        "vision": -15,
        "technique": -14,
        "finishing": -35,
        "long_passing": -8,
    },
    "sweeper_keeper": {
        "gk_reflexes": 12,
        "gk_diving": 12,
        "gk_handling": 12,
        "gk_positioning": 14,
        "gk_distribution": 18,
    },
    "shot_stopper": {
        "gk_reflexes": 18,
        "gk_diving": 16,
        "gk_handling": 12,
        "gk_positioning": 12,
        "gk_distribution": -4,
    },
}

HEIGHT_RANGE: Final[dict[str, tuple[int, int]]] = {
    "GK": (184, 198),
    "CB": (182, 196),
    "FB": (170, 184),
    "DM": (174, 189),
    "CM": (170, 186),
    "AM": (167, 182),
    "W": (165, 181),
    "ST": (174, 193),
}


def _pick[T](rng: np.random.Generator, items: tuple[tuple[T, float], ...]) -> T:
    w = np.array([p for _, p in items])
    return items[int(rng.choice(len(items), p=w / w.sum()))][0]


def _rating(x: float) -> int:
    return int(np.clip(round(x), 1, 99))


def physical_params(
    attrs: PlayerAttributes, height_cm: int, is_gk: bool, priors: PhysicalPriors, rng: np.random.Generator
) -> PhysicalParams:
    max_speed = float(np.clip(priors.max_speed_mean + priors.max_speed_std * (attrs.pace - 70) / 10, 7.0, 10.2))
    accel = float(np.clip(priors.max_accel_mean + priors.max_accel_std * (attrs.acceleration - 70) / 10, 2.8, 5.4))
    decel = float(np.clip(accel * 1.35, 3.5, 7.5))
    calm = (attrs.composure + attrs.positioning) / 198
    reaction = float(np.clip(priors.reaction_time_s * (1.25 - 0.5 * calm) + rng.normal(0, 0.01), 0.12, 0.45))
    lift = (2.6 + attrs.jumping / 99) ** 2 / (2 * GRAVITY)
    base = height_cm / 100 * (1.25 if is_gk else 0.94)
    return PhysicalParams(
        max_speed_mps=round(max_speed, 2),
        max_accel_mps2=round(accel, 2),
        max_decel_mps2=round(decel, 2),
        reaction_time_s=round(reaction, 3),
        jump_reach_m=round(float(np.clip(base + lift, 1.8, 3.5)), 2),
    )


def _attributes(archetype: Archetype, overall: float, rng: np.random.Generator) -> PlayerAttributes:
    prof = PROFILE[archetype]
    is_gk = archetype in ("sweeper_keeper", "shot_stopper")
    vals: dict[str, int] = {}
    for a in ATTRS:
        if is_gk:
            if a.startswith("gk_"):
                vals[a] = _rating(overall + prof.get(a, 0) + rng.normal(0, 3))
            else:
                centre = 46 if a in ("short_passing", "long_passing", "composure", "jumping", "strength") else 30
                if archetype == "sweeper_keeper" and a in ("short_passing", "long_passing", "pace", "composure"):
                    centre += 14
                vals[a] = _rating(centre + rng.normal(0, 6))
        elif a.startswith("gk_"):
            vals[a] = _rating(rng.uniform(6, 18))
        else:
            vals[a] = _rating(overall + prof.get(a, 0) + rng.normal(0, 4))
    return PlayerAttributes(**vals)


def _name(rng: np.random.Generator, used: set[str]) -> tuple[str, str, str]:
    codes = sorted(REGIONS)
    for _ in range(200):
        code = codes[int(rng.integers(len(codes)))]
        firsts, lasts = REGIONS[code]
        first, last = firsts[int(rng.integers(len(firsts)))], lasts[int(rng.integers(len(lasts)))]
        if f"{first} {last}" in FAMOUS or last in used:
            continue
        used.add(last)
        return first, last, code
    raise RuntimeError("name pool exhausted")


def _hex_to_rgb(h: str) -> np.ndarray:
    return np.array([int(h[i : i + 2], 16) for i in (1, 3, 5)], dtype=float)


def _contrast(a: str, b: str) -> float:
    return float(np.linalg.norm(_hex_to_rgb(a) - _hex_to_rgb(b)))


def _team(
    side: str, city: str, suffix: str, formation: str, primary: str, rng: np.random.Generator, priors: PhysicalPriors
) -> TeamMeta:
    letter = "H" if side == "home" else "A"
    words = city.split()
    monogram = "".join(w[0] for w in words)[:2].upper() if len(words) > 1 else city[:2].upper()
    short = (monogram + suffix[0]).upper()[:3] if len(monogram) == 2 else city[:3].upper()
    secondary = str(rng.choice([c for c in KIT_COLOURS if _contrast(c, primary) > 200]))
    number = "#ffffff" if _contrast(primary, "#ffffff") > 260 else "#111111"
    pattern: KitPattern = rng.choice(get_args(KitPattern), p=[0.4, 0.15, 0.1, 0.1, 0.1, 0.15])
    kit = KitColors(primary=primary, secondary=secondary, accent=secondary, number=number, pattern=pattern)
    gk_primary = str(rng.choice(GK_COLOURS))
    gk_kit = KitColors(primary=gk_primary, secondary="#111111", accent="#ffffff", number="#111111")
    shape: CrestShape = rng.choice(get_args(CrestShape))
    emblem: CrestEmblem = rng.choice(get_args(CrestEmblem))
    crest = CrestSpec(shape=shape, monogram=monogram[:3], primary=primary, secondary=secondary, emblem=emblem)
    level = rng.uniform(68, 76)
    used_names: set[str] = set()
    slots = FORMATIONS[formation]
    archetypes: list[Archetype] = [_pick(rng, ARCHETYPES_BY_POSITION[POSITION_GROUP[s.position]]) for s in slots]
    if "advanced_playmaker" not in archetypes:  # every side gets one "sees and bends the impossible pass" player
        mids = [i for i, s in enumerate(slots) if POSITION_GROUP[s.position] in ("CM", "AM")]
        archetypes[mids[-1]] = "advanced_playmaker"
    players: list[PlayerMeta] = []
    shirts = {s.shirt for s in slots}
    sub_positions: list[Position] = ["GK", "CB", "RB", "CM", "DM", "LW", "ST"]
    free_numbers = [n for n in range(12, 31) if n not in shirts]
    rng.shuffle(free_numbers)
    entries: list[tuple[Position, int, Archetype, bool]] = [
        (s.position, s.shirt, a, True) for s, a in zip(slots, archetypes, strict=True)
    ]
    for k, pos in enumerate(sub_positions):
        entries.append((pos, free_numbers[k], _pick(rng, ARCHETYPES_BY_POSITION[POSITION_GROUP[pos]]), False))
    for pos, shirt, arch, starter in entries:
        group = POSITION_GROUP[pos]
        overall = level + rng.normal(0, 3.5) - (2.5 if not starter else 0.0)
        attrs = _attributes(arch, overall, rng)
        lo, hi = HEIGHT_RANGE[group]
        height = int(rng.integers(lo, hi + 1))
        if arch == "target_forward":
            height = max(height, 186)
        first, last, code = _name(rng, used_names)
        left_bias = 0.5 if pos in ("LB", "LW", "LM", "LCB") else 0.2
        if arch == "inside_forward" and pos in ("RW", "RM"):
            left_bias = 0.75
        if arch == "inside_forward" and pos in ("LW", "LM"):
            left_bias = 0.2
        foot = "left" if rng.random() < left_bias else "right"
        weak = int(rng.choice([1, 2, 3, 4, 5], p=[0.05, 0.25, 0.45, 0.2, 0.05]))
        if arch in ("advanced_playmaker", "inside_forward") and weak < 3:
            weak += 1
        players.append(
            PlayerMeta(
                id=f"{letter}{shirt:02d}",
                team=side,
                first_name=first,
                last_name=last,
                display_name=last,
                shirt_number=shirt,
                position=pos,
                archetype=arch,
                preferred_foot=foot,
                weak_foot=weak,
                height_cm=height,
                nationality=code,
                attributes=attrs,
                physical=physical_params(attrs, height, group == "GK", priors, rng),
                starter=starter,
            )
        )
    return TeamMeta(
        side=side,
        name=f"{city} {suffix}",
        short_name=short,
        city=city,
        formation=formation,
        kit=kit,
        gk_kit=gk_kit,
        crest=crest,
        players=players,
    )


def generate_meta(seed: int, match_id: str | None = None, priors: PhysicalPriors | None = None) -> MatchMeta:
    """A complete synthetic fixture for ``seed``: two original clubs and 18-player squads."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x5EED]))
    priors = priors or PhysicalPriors()
    cities = rng.choice(len(CITIES), 2, replace=False)
    suffixes = rng.choice(len(CLUB_SUFFIXES), 2, replace=False)
    formations = sorted(FORMATIONS)
    primaries = list(KIT_COLOURS)
    home_primary = str(rng.choice(primaries))
    away_primary = str(rng.choice([c for c in primaries if _contrast(c, home_primary) > 230]))
    home = _team(
        "home",
        CITIES[cities[0]],
        CLUB_SUFFIXES[suffixes[0]],
        formations[int(rng.integers(3))],
        home_primary,
        rng,
        priors,
    )
    away = _team(
        "away",
        CITIES[cities[1]],
        CLUB_SUFFIXES[suffixes[1]],
        formations[int(rng.integers(3))],
        away_primary,
        rng,
        priors,
    )
    weather: Weather = rng.choice(get_args(Weather), p=[0.6, 0.25, 0.15])
    return MatchMeta(
        match_id=match_id or f"m{seed:08d}",
        seed=seed,
        home=home,
        away=away,
        home_attacks_positive_x_first_half=bool(rng.random() < 0.5),
        kickoff_team="home" if rng.random() < 0.5 else "away",
        venue=str(rng.choice(VENUE_WORDS)),
        weather=weather,
    )
