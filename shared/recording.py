"""Recorded match format: what the simulator writes and the server, renderer tooling and Spotter read.

A recording is a directory:

    manifest.json   {"schema_version", "match_id", "n_frames", "sim_hz", "duration_s", "player_ids", "created_by"}
    meta.json       MatchMeta
    frames.npz      columnar SimState arrays (see FRAME_ARRAYS), float32 / int16 / int8
    events.jsonl    one MatchEvent per line, ordered by t
    moments.jsonl   optional, one DecisionMoment per line
    snapshots/      optional, engine-private fork data (owned by sim/)

Players are stored in the fixed order ``manifest.player_ids`` (home starters, home subs, away starters, away
subs). ``state_at(i)`` rebuilds a full ``SimState`` for one frame, so a 90-minute match (~135k frames) never
has to be materialized as Python objects.
"""

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, get_args

import numpy as np
import numpy.typing as npt

from shared.schemas.common import SCHEMA_VERSION
from shared.schemas.enums import PhaseOfPlay, RestartType
from shared.schemas.events import MatchEvent
from shared.schemas.meta import MatchMeta
from shared.schemas.moments import DecisionMoment
from shared.schemas.state import (
    BallState,
    MatchStatus,
    PlayerAnim,
    PlayerState,
    Possession,
    RefereeState,
    Score,
    SimState,
)

ANIMS: Final = get_args(PlayerAnim)
PHASES: Final = get_args(PhaseOfPlay)
RESTARTS: Final = get_args(RestartType)
STATUSES: Final = get_args(MatchStatus)
SIDES: Final = ("home", "away")

FRAME_ARRAYS: Final = {
    # name: (shape suffix, dtype) — leading dim is the frame index T, P = number of players
    "tick": ((), np.int64),
    "t": ((), np.float64),
    "period": ((), np.int8),
    "status": ((), np.int8),
    "ball_pos": ((3,), np.float32),
    "ball_vel": ((3,), np.float32),
    "ball_spin": ((3,), np.float32),
    "ball_in_play": ((), np.bool_),
    "last_touch": ((), np.int16),
    "player_pos": (("P", 2), np.float32),
    "player_vel": (("P", 2), np.float32),
    "player_facing": (("P",), np.float32),
    "player_z": (("P",), np.float32),
    "player_anim": (("P",), np.int8),
    "player_anim_t": (("P",), np.float32),
    "player_anim_mirror": (("P",), np.bool_),
    "player_anim_variant": (("P",), np.int8),
    "player_stamina": (("P",), np.float32),
    "player_on_pitch": (("P",), np.bool_),
    "ref_pos": ((2,), np.float32),
    "ref_vel": ((2,), np.float32),
    "poss_team": ((), np.int8),
    "poss_player": ((), np.int16),
    "possession_id": ((), np.int32),
    "phase": ((), np.int8),
    "restart": ((), np.int8),
    "score": ((2,), np.int16),
    "attacking_positive_x": ((), np.int8),
    "keyframe": ((), np.bool_),
}
"""Indices into ANIMS / PHASES / RESTARTS / STATUSES / SIDES / player_ids; -1 means None."""


def player_order(meta: MatchMeta) -> list[str]:
    return [p.id for p in meta.home.players] + [p.id for p in meta.away.players]


def snapshot_ref(match_id: str, tick: int) -> str:
    return f"snap:{match_id}:{tick}"


def _f(x: float) -> float:
    return round(float(x), 4)


def states_to_arrays(states: Sequence[SimState], ids: Sequence[str]) -> dict[str, npt.NDArray[Any]]:
    """Pack SimStates into FRAME_ARRAYS. Every state must list players by id; missing players are off-pitch."""
    n, p = len(states), len(ids)
    index = {pid: i for i, pid in enumerate(ids)}
    out: dict[str, npt.NDArray[Any]] = {}
    for name, (shape, dtype) in FRAME_ARRAYS.items():
        dims = tuple(p if d == "P" else d for d in shape)
        out[name] = np.zeros((n, *dims), dtype=dtype)
    out["last_touch"][:] = -1
    out["poss_player"][:] = -1
    out["poss_team"][:] = -1
    out["restart"][:] = -1
    for k, s in enumerate(states):
        out["tick"][k] = s.tick
        out["t"][k] = s.t
        out["period"][k] = s.period
        out["status"][k] = STATUSES.index(s.status)
        out["ball_pos"][k] = s.ball.pos
        out["ball_vel"][k] = s.ball.vel
        out["ball_spin"][k] = s.ball.spin
        out["ball_in_play"][k] = s.ball.in_play
        if s.ball.last_touch is not None:
            out["last_touch"][k] = index[s.ball.last_touch]
        for ps in s.players:
            i = index[ps.id]
            out["player_pos"][k, i] = ps.pos
            out["player_vel"][k, i] = ps.vel
            out["player_facing"][k, i] = ps.facing
            out["player_z"][k, i] = ps.z
            out["player_anim"][k, i] = ANIMS.index(ps.anim)
            out["player_anim_t"][k, i] = ps.anim_t
            out["player_anim_mirror"][k, i] = ps.anim_mirror
            out["player_anim_variant"][k, i] = ps.anim_variant
            out["player_stamina"][k, i] = ps.stamina
            out["player_on_pitch"][k, i] = ps.on_pitch
        if s.referee is not None:
            out["ref_pos"][k] = s.referee.pos
            out["ref_vel"][k] = s.referee.vel
        if s.possession.team is not None:
            out["poss_team"][k] = SIDES.index(s.possession.team)
        if s.possession.player is not None:
            out["poss_player"][k] = index[s.possession.player]
        out["possession_id"][k] = s.possession.possession_id
        out["phase"][k] = PHASES.index(s.phase)
        if s.restart is not None:
            out["restart"][k] = RESTARTS.index(s.restart)
        out["score"][k] = (s.score.home, s.score.away)
        out["attacking_positive_x"][k] = SIDES.index(s.attacking_positive_x)
        out["keyframe"][k] = s.keyframe
    return out


def write_recording(
    directory: Path,
    meta: MatchMeta,
    arrays: dict[str, npt.NDArray[Any]],
    events: Iterable[MatchEvent],
    moments: Iterable[DecisionMoment] = (),
    created_by: str = "sim",
) -> None:
    """Write a recording. ``arrays`` must contain every FRAME_ARRAYS key (see ``states_to_arrays``)."""
    missing = set(FRAME_ARRAYS) - set(arrays)
    if missing:
        raise ValueError(f"missing frame arrays: {sorted(missing)}")
    directory.mkdir(parents=True, exist_ok=True)
    ids = player_order(meta)
    n = len(arrays["tick"])
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "match_id": meta.match_id,
        "n_frames": n,
        "sim_hz": meta.sim_hz,
        "duration_s": float(arrays["t"][-1]) if n else 0.0,
        "player_ids": ids,
        "created_by": created_by,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (directory / "meta.json").write_text(meta.model_dump_json(indent=2) + "\n")
    typed = {k: np.asarray(arrays[k], dtype=FRAME_ARRAYS[k][1]) for k in FRAME_ARRAYS}
    np.savez_compressed(directory / "frames.npz", **typed)  # type: ignore[arg-type]
    with (directory / "events.jsonl").open("w") as fh:
        fh.writelines(e.model_dump_json() + "\n" for e in events)
    moment_list = list(moments)
    if moment_list:
        with (directory / "moments.jsonl").open("w") as fh:
            fh.writelines(m.model_dump_json() + "\n" for m in moment_list)


@dataclass
class Recording:
    directory: Path
    meta: MatchMeta
    manifest: dict[str, Any]
    arrays: dict[str, npt.NDArray[Any]]

    @classmethod
    def open(cls, directory: Path) -> "Recording":
        manifest = json.loads((directory / "manifest.json").read_text())
        meta = MatchMeta.model_validate_json((directory / "meta.json").read_text())
        with np.load(directory / "frames.npz") as data:
            arrays = {k: data[k] for k in data.files}
        return cls(directory, meta, manifest, arrays)

    @property
    def n_frames(self) -> int:
        return int(self.manifest["n_frames"])

    @property
    def player_ids(self) -> list[str]:
        return list(self.manifest["player_ids"])

    def index_at(self, t: float) -> int:
        """Last frame with time <= t (clamped)."""
        i = int(np.searchsorted(self.arrays["t"], t, side="right")) - 1
        return min(max(i, 0), self.n_frames - 1)

    def events(self) -> list[MatchEvent]:
        path = self.directory / "events.jsonl"
        return [MatchEvent.model_validate_json(line) for line in path.read_text().splitlines() if line]

    def moments(self) -> list[DecisionMoment]:
        path = self.directory / "moments.jsonl"
        if not path.exists():
            return []
        return [DecisionMoment.model_validate_json(line) for line in path.read_text().splitlines() if line]

    def state_at(self, i: int) -> SimState:
        a, ids = self.arrays, self.player_ids

        def pid(idx: int) -> str | None:
            return None if idx < 0 else ids[idx]

        players = [
            PlayerState(
                id=ids[j],
                pos=(_f(a["player_pos"][i, j, 0]), _f(a["player_pos"][i, j, 1])),
                vel=(_f(a["player_vel"][i, j, 0]), _f(a["player_vel"][i, j, 1])),
                facing=_f(a["player_facing"][i, j]),
                z=max(0.0, _f(a["player_z"][i, j])),
                anim=ANIMS[int(a["player_anim"][i, j])],
                anim_t=max(0.0, _f(a["player_anim_t"][i, j])),
                anim_mirror=bool(a["player_anim_mirror"][i, j]),
                anim_variant=int(a["player_anim_variant"][i, j]),
                stamina=min(1.0, max(0.0, _f(a["player_stamina"][i, j]))),
                on_pitch=bool(a["player_on_pitch"][i, j]),
            )
            for j in range(len(ids))
            if a["player_on_pitch"][i, j]
        ]
        tick = int(a["tick"][i])
        keyframe = bool(a["keyframe"][i])
        poss_team = int(a["poss_team"][i])
        restart = int(a["restart"][i])
        return SimState(
            tick=tick,
            t=float(a["t"][i]),
            period=1 if int(a["period"][i]) == 1 else 2,
            status=STATUSES[int(a["status"][i])],
            ball=BallState(
                pos=(_f(a["ball_pos"][i, 0]), _f(a["ball_pos"][i, 1]), _f(a["ball_pos"][i, 2])),
                vel=(_f(a["ball_vel"][i, 0]), _f(a["ball_vel"][i, 1]), _f(a["ball_vel"][i, 2])),
                spin=(_f(a["ball_spin"][i, 0]), _f(a["ball_spin"][i, 1]), _f(a["ball_spin"][i, 2])),
                in_play=bool(a["ball_in_play"][i]),
                last_touch=pid(int(a["last_touch"][i])),
            ),
            players=players,
            referee=RefereeState(
                pos=(_f(a["ref_pos"][i, 0]), _f(a["ref_pos"][i, 1])),
                vel=(_f(a["ref_vel"][i, 0]), _f(a["ref_vel"][i, 1])),
            ),
            possession=Possession(
                team=None if poss_team < 0 else SIDES[poss_team],
                player=pid(int(a["poss_player"][i])),
                possession_id=int(a["possession_id"][i]),
            ),
            phase=PHASES[int(a["phase"][i])],
            restart=None if restart < 0 else RESTARTS[restart],
            score=Score(home=int(a["score"][i, 0]), away=int(a["score"][i, 1])),
            attacking_positive_x=SIDES[int(a["attacking_positive_x"][i])],
            keyframe=keyframe,
            snapshot_ref=snapshot_ref(self.meta.match_id, tick) if keyframe else None,
        )
