"""Hand-built game situations for resolver / engine tests."""

from typing import Any

import numpy as np

from shared.schemas.meta import MatchMeta
from sim.constants import BALL_RADIUS
from sim.engine import SimEngine
from sim.match import Match
from sim.squads import generate_meta
from sim.world import BS_OWNED, N_ON

META_SEED = 21


def base_match(
    seed: int = META_SEED, attr_overrides: dict[str, dict[str, int]] | None = None
) -> tuple[Match, MatchMeta]:
    meta = generate_meta(seed, f"scene{seed}")
    if attr_overrides:
        home = meta.home
        players = []
        for p in home.players:
            upd = attr_overrides.get(p.id)
            players.append(p.model_copy(update={"attributes": p.attributes.model_copy(update=upd)}) if upd else p)
        meta = meta.model_copy(update={"home": home.model_copy(update={"players": players})})
    m = Match(meta, batch=1)
    m.start()
    m.step()
    return m, meta


def place(
    m: Match,
    carrier: str,
    at: tuple[float, float],
    mates: dict[str, tuple[float, float]] | None = None,
    opps: dict[str, tuple[float, float]] | None = None,
    ball_z: float = BALL_RADIUS,
    ball_vel: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    """Positions in the home team's attacking frame (home attacks +x): everybody else far away."""
    w, sq = m.w, m.sq
    s = 1.0 if m.sign[0, 0] > 0 else -1.0
    w.pos[0] = np.array([[-40.0 + 2 * (i % 11), -30.0 + 3 * (i // 11)] for i in range(22)]) * s
    w.pos[0, N_ON:, 0] = -45.0 * s  # opponents parked deep in the home half unless placed
    w.pos[0, N_ON:, 1] = np.linspace(-30, 30, N_ON) * s
    gk_away = int(m._gk[1])
    w.pos[0, gk_away] = (51.0 * s, 0.0)
    w.vel[0] = 0.0
    for pid, xy in (mates or {}).items():
        w.pos[0, sq.ids.index(pid)] = (xy[0] * s, xy[1] * s)
    for pid, xy in (opps or {}).items():
        w.pos[0, sq.ids.index(pid)] = (xy[0] * s, xy[1] * s)
    c = sq.ids.index(carrier)
    w.pos[0, c] = (at[0] * s, at[1] * s)
    w.ball_state[0] = BS_OWNED
    w.owner[0] = c
    w.restart[0] = -1
    w.ball_pos[0] = (at[0] * s, at[1] * s, ball_z)
    w.ball_vel[0] = ball_vel
    w.poss_team[0] = 0
    w.hands[0] = False
    w.last_restart_type[0] = -1
    w.stun[0] = 0.0


def engine_for(m: Match, models: Any = None) -> tuple[SimEngine, str]:
    eng = SimEngine(models=models or m.models, workers=1)
    tick = int(m.w.tick[0])
    eng.register(m.meta, [(tick, m.w.snapshot(0))])
    return eng, f"snap:{m.meta.match_id}:{tick}"


def home_ids(m: Match) -> list[str]:
    return [p.id for p in m.sq.players[:N_ON]]


def away_ids(m: Match) -> list[str]:
    return [p.id for p in m.sq.players[N_ON:]]
