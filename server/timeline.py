"""The recorded match as the server sees it: frames by index (with a shared serialized-frame cache), events, moments."""

import bisect
from collections import OrderedDict
from pathlib import Path

import numpy as np

from shared.recording import Recording
from shared.schemas.events import MatchEvent
from shared.schemas.moments import DecisionMoment
from shared.schemas.state import SimState
from shared.schemas.stream import FrameMsg


class Timeline:
    """Read-only view over one ``Recording``, shared by every session and REST handler."""

    def __init__(self, recording: Recording, frame_cache_size: int = 4096) -> None:
        self.recording = recording
        self.meta = recording.meta
        self.match_id = recording.meta.match_id
        self.times: np.ndarray = recording.arrays["t"]
        self.ticks: np.ndarray = recording.arrays["tick"]
        self.n_frames = recording.n_frames
        if self.n_frames == 0:
            raise ValueError(f"recording {recording.directory} has no frames")
        self.duration_s = float(self.times[-1])
        self.events: list[MatchEvent] = sorted(recording.events(), key=lambda e: (e.t, e.index))
        self.event_times: list[float] = [e.t for e in self.events]
        self.moments: list[DecisionMoment] = sorted(recording.moments(), key=lambda m: m.t)
        self.moment_times: list[float] = [m.t for m in self.moments]
        self._cache: OrderedDict[int, str] = OrderedDict()
        self._cache_size = frame_cache_size

    @classmethod
    def open(cls, directory: Path, frame_cache_size: int = 4096) -> "Timeline":
        return cls(Recording.open(directory), frame_cache_size)

    def index_at(self, t: float) -> int:
        return self.recording.index_at(t)

    def time_of(self, index: int) -> float:
        return float(self.times[index])

    def tick_of(self, index: int) -> int:
        return int(self.ticks[index])

    def state_at(self, index: int) -> SimState:
        return self.recording.state_at(index)

    def frame_json(self, index: int) -> str:
        """The ``frame`` message for one recorded frame, serialized once and shared across sessions (LRU)."""
        cached = self._cache.get(index)
        if cached is not None:
            self._cache.move_to_end(index)
            return cached
        text = FrameMsg(state=self.state_at(index)).model_dump_json()
        self._cache[index] = text
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return text

    def events_between(self, t_from: float, t_to: float) -> list[MatchEvent]:
        """Events with ``t_from <= t <= t_to``."""
        lo = bisect.bisect_left(self.event_times, t_from)
        hi = bisect.bisect_right(self.event_times, t_to)
        return self.events[lo:hi]
