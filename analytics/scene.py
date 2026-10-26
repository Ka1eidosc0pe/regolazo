"""Turn a ``SimState`` into model inputs (attacking frame, freeze-frame features).

Convenience for the simulator, agents and tests: the geometry (shot cone, nearest defender, passing lane) is the same
code that built the StatsBomb training features, so a scene is described to the models exactly as the data was.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from analytics.features import count_in_lane, count_in_shot_cone, nearest_distance
from analytics.pitch_control import defending_gk_index
from shared.model_api import (
    GOAL_X,
    PASS_HEIGHT,
    PASS_KIND,
    SHOT_BODY_PART,
    SHOT_SITUATION,
    SHOT_TECHNIQUE,
    AnalyticsModels,
    PassFeatures,
    ShotFeatures,
)
from shared.schemas.common import PITCH_LENGTH_M, PITCH_WIDTH_M, TeamSide
from shared.schemas.models import PitchControlGrid
from shared.schemas.state import SimState

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Scene:
    """One team's view of a frame: everything rotated so ``team`` attacks +x."""

    team: TeamSide
    att_ids: list[str]
    att_pos: FloatArray
    att_vel: FloatArray
    def_ids: list[str]
    def_pos: FloatArray
    def_vel: FloatArray
    ball: FloatArray
    """(x, y, z) in the attacking frame."""
    sign: float = 1.0
    """World -> attacking frame factor (+1 if ``team`` attacks +x, else -1; the rotation is its own inverse)."""

    @property
    def def_gk(self) -> int | None:
        return defending_gk_index(self.def_pos)

    def outfield_defenders(self) -> FloatArray:
        gk = self.def_gk
        keep = np.ones(len(self.def_pos), dtype=bool)
        if gk is not None:
            keep[gk] = False
        return self.def_pos[keep]

    def player_xy(self, player_id: str) -> FloatArray:
        return np.asarray(self.att_pos[self.att_ids.index(player_id)], dtype=np.float64)


def scene_from_state(state: SimState, team: TeamSide | None = None) -> Scene:
    """``team`` defaults to the team in possession (home if nobody has the ball)."""
    side: TeamSide = team or state.possession.team or "home"
    sign = 1.0 if state.attacking_positive_x == side else -1.0
    prefix = "H" if side == "home" else "A"
    att = [p for p in state.players if p.on_pitch and p.id.startswith(prefix)]
    dfn = [p for p in state.players if p.on_pitch and not p.id.startswith(prefix)]

    def arr(ps: list[tuple[float, float]]) -> FloatArray:
        return np.asarray(ps, dtype=np.float64).reshape(-1, 2) * sign

    bx, by, bz = state.ball.pos
    return Scene(
        team=side,
        att_ids=[p.id for p in att],
        att_pos=arr([p.pos for p in att]),
        att_vel=arr([p.vel for p in att]),
        def_ids=[p.id for p in dfn],
        def_pos=arr([p.pos for p in dfn]),
        def_vel=arr([p.vel for p in dfn]),
        ball=np.array([bx * sign, by * sign, bz]),
        sign=sign,
    )


def pitch_control_grid(models: AnalyticsModels, scene: Scene, nx: int = 21, ny: int = 14) -> PitchControlGrid:
    """``ModelOutputs.pitch_control`` for ``scene.team``: a world-frame grid tiling the pitch, row-major by y."""
    dx, dy = PITCH_LENGTH_M / nx, PITCH_WIDTH_M / ny
    x0, y0 = -PITCH_LENGTH_M / 2 + dx / 2, -PITCH_WIDTH_M / 2 + dy / 2
    gy, gx = np.meshgrid(y0 + dy * np.arange(ny), x0 + dx * np.arange(nx), indexing="ij")
    world = np.column_stack([gx.ravel(), gy.ravel()])  # iy-major, matching values[iy * nx + ix]
    pc = models.pitch_control(
        scene.att_pos, scene.att_vel, scene.def_pos, scene.def_vel, scene.ball[:2], world * scene.sign
    )
    values = [float(v) for v in np.clip(np.round(pc, 4), 0.0, 1.0)]
    return PitchControlGrid(team=scene.team, nx=nx, ny=ny, x0=x0, y0=y0, dx=dx, dy=dy, values=values)


def shot_features(
    scene: Scene,
    shooter: FloatArray,
    body_part: int = SHOT_BODY_PART["foot"],
    technique: int = SHOT_TECHNIQUE["placed"],
    situation: int = SHOT_SITUATION["open_play"],
    under_pressure: bool = False,
    first_time: bool = False,
) -> ShotFeatures:
    """xG features for a shot from ``shooter`` (attacking-frame (x, y)) against the scene's defence."""
    x = np.array([float(shooter[0])])
    y = np.array([float(shooter[1])])
    gk = scene.def_gk
    gk_xy = scene.def_pos[gk] if gk is not None else np.array([GOAL_X, 0.0])
    opp = scene.outfield_defenders()[None, :, :]
    return ShotFeatures(
        x=x,
        y=y,
        body_part=np.array([body_part], dtype=np.int64),
        technique=np.array([technique], dtype=np.int64),
        situation=np.array([situation], dtype=np.int64),
        gk_x=np.array([gk_xy[0]]),
        gk_y=np.array([gk_xy[1]]),
        n_defenders_in_cone=count_in_shot_cone(x, y, opp),
        nearest_defender_m=nearest_distance(x, y, opp),
        under_pressure=np.array([under_pressure]),
        first_time=np.array([first_time]),
    )


def pass_features(
    scene: Scene,
    passer: FloatArray,
    targets: FloatArray,
    height: int = PASS_HEIGHT["ground"],
    kind: int = PASS_KIND["standard"],
    under_pressure: bool = False,
) -> PassFeatures:
    """xPass features for passes from ``passer`` to each of ``targets`` (M x 2), all opponents counted."""
    targets = np.asarray(targets, dtype=np.float64).reshape(-1, 2)
    m = targets.shape[0]
    x = np.full(m, float(passer[0]))
    y = np.full(m, float(passer[1]))
    opp = np.broadcast_to(scene.def_pos, (m, *scene.def_pos.shape))
    return PassFeatures(
        x=x,
        y=y,
        end_x=targets[:, 0],
        end_y=targets[:, 1],
        height=np.full(m, height, dtype=np.int64),
        kind=np.full(m, kind, dtype=np.int64),
        nearest_defender_to_receiver_m=nearest_distance(targets[:, 0], targets[:, 1], opp),
        defenders_in_lane=count_in_lane(x, y, targets[:, 0], targets[:, 1], opp),
        under_pressure=np.full(m, under_pressure),
    )
