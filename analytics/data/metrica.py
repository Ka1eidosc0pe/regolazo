"""Metrica Sports sample tracking data (games 1 and 2, 25 Hz CSV) loader.

Metrica coordinates are normalised to [0, 1] with the origin at the top-left; they are converted to the world frame
(metres, centre origin, +y up the screen) on a 105 x 68 m pitch.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from analytics.paths import metrica_dir
from shared.schemas.common import PITCH_LENGTH_M, PITCH_WIDTH_M

FloatArray = npt.NDArray[np.float64]

FPS = 25.0
SMOOTH_WINDOW = 7
"""Frames in the centred moving average applied before differentiating (0.28 s)."""
MAX_PLAUSIBLE_SPEED_MPS = 12.0


@dataclass(frozen=True)
class TeamTracking:
    period: npt.NDArray[np.int64]
    frame: npt.NDArray[np.int64]
    time_s: FloatArray
    players: list[str]
    pos: FloatArray
    """(T, P, 2) metres, NaN when the player is off the pitch."""
    ball: FloatArray
    """(T, 2) metres."""


@dataclass(frozen=True)
class MetricaEvent:
    team: str
    type: str
    subtype: str
    period: int
    start_frame: int
    end_frame: int
    start_time_s: float
    end_time_s: float
    from_player: str
    to_player: str
    start: tuple[float, float]
    end: tuple[float, float]


def to_metres(x: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
    return (x - 0.5) * PITCH_LENGTH_M, -(y - 0.5) * PITCH_WIDTH_M


def load_tracking(game: int, team: str, root: Path | None = None) -> TeamTracking:
    """``team`` is "Home" or "Away"."""
    root = root or metrica_dir()
    path = root / f"Sample_Game_{game}" / f"Sample_Game_{game}_RawTrackingData_{team}_Team.csv"
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader)
        next(reader)
        header = next(reader)
    names = [h for h in header[3:] if h]
    data = np.loadtxt(path, delimiter=",", skiprows=3)
    xy = data[:, 3:]
    n_obj = xy.shape[1] // 2
    assert n_obj == len(names), (n_obj, names)
    x, y = to_metres(xy[:, 0::2], xy[:, 1::2])
    pos = np.stack([x, y], axis=2)
    ball_idx = names.index("Ball")
    players = [n for n in names if n != "Ball"]
    keep = [i for i, n in enumerate(names) if n != "Ball"]
    return TeamTracking(
        period=data[:, 0].astype(np.int64),
        frame=data[:, 1].astype(np.int64),
        time_s=data[:, 2],
        players=players,
        pos=pos[:, keep, :],
        ball=pos[:, ball_idx, :],
    )


def load_events(game: int, root: Path | None = None) -> list[MetricaEvent]:
    root = root or metrica_dir()
    path = root / f"Sample_Game_{game}" / f"Sample_Game_{game}_RawEventsData.csv"
    out: list[MetricaEvent] = []
    with path.open() as fh:
        for row in csv.DictReader(fh):
            sx, sy = to_metres(np.array(float(row["Start X"])), np.array(float(row["Start Y"])))
            ex, ey = to_metres(np.array(float(row["End X"])), np.array(float(row["End Y"])))
            out.append(
                MetricaEvent(
                    team=row["Team"],
                    type=row["Type"],
                    subtype=row["Subtype"],
                    period=int(row["Period"]),
                    start_frame=int(row["Start Frame"]),
                    end_frame=int(row["End Frame"]),
                    start_time_s=float(row["Start Time [s]"]),
                    end_time_s=float(row["End Time [s]"]),
                    from_player=row["From"],
                    to_player=row["To"],
                    start=(float(sx), float(sy)),
                    end=(float(ex), float(ey)),
                )
            )
    return out


def smooth(a: FloatArray, window: int = SMOOTH_WINDOW) -> FloatArray:
    """Centred moving average along axis 0, NaN-aware (a window with any NaN gives NaN)."""
    kernel = np.ones(window) / window
    flat = a.reshape(a.shape[0], -1)
    out = np.full_like(flat, np.nan)
    half = window // 2
    for j in range(flat.shape[1]):
        col = flat[:, j]
        conv = np.convolve(col, kernel, mode="valid")
        out[half : half + conv.size, j] = conv
    return out.reshape(a.shape)


def velocities(pos: FloatArray, period: npt.NDArray[np.int64], fps: float = FPS) -> FloatArray:
    """(T, P, 2) smoothed velocity in m/s via central differences, computed within each period."""
    vel = np.full_like(pos, np.nan)
    for per in np.unique(period):
        idx = np.flatnonzero(period == per)
        sp = smooth(pos[idx])
        v = np.gradient(sp, 1.0 / fps, axis=0)
        speed = np.linalg.norm(v, axis=-1)
        v[speed > MAX_PLAUSIBLE_SPEED_MPS] = np.nan
        vel[idx] = v
    return vel


def attacks_positive_x(track: TeamTracking, period: int, gk_index: int) -> bool:
    """A team attacks +x when its keeper's mean x in that period is negative."""
    sel = track.period == period
    return bool(np.nanmean(track.pos[sel, gk_index, 0]) < 0)


def goalkeeper_index(track: TeamTracking) -> int:
    """The player nearest a goal line at the first frame (Metrica lists keepers like any other player)."""
    x0 = np.abs(track.pos[0, :, 0])
    return int(np.argmax(np.where(np.isnan(x0), -1.0, x0)))
