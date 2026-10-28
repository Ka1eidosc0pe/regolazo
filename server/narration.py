"""Precomputed narratives (written by ``python -m agents.precompute``) as the server serves them.

``<recording>/narratives/<lang>_<mode>.jsonl`` holds one ``Narrative`` per line. ``NarrationStore`` loads them once
at startup; each session asks it for the narratives of its own ViewerProfile (language and mode) whose match time
``t`` the clock has just passed. ``Narrative.audio_url`` already points at ``/api/files/narratives/audio/<file>.mp3``.
A recording without a ``narratives/`` directory yields an empty store and nothing is delivered.
"""

import bisect
import logging
from pathlib import Path

from pydantic import ValidationError

from shared.schemas.narrative import Narrative

log = logging.getLogger("regolazo.narration")


class NarrationStore:
    def __init__(self, narratives: list[Narrative] | None = None) -> None:
        self._by_profile: dict[tuple[str, str], list[Narrative]] = {}
        self._times: dict[tuple[str, str], list[float]] = {}
        for n in sorted(narratives or [], key=lambda n: (n.t, n.id)):
            self._by_profile.setdefault((n.language, n.mode), []).append(n)
        for key, rows in self._by_profile.items():
            self._times[key] = [n.t for n in rows]

    @classmethod
    def load(cls, recording_dir: Path) -> "NarrationStore":
        directory = Path(recording_dir) / "narratives"
        rows: list[Narrative] = []
        for path in sorted(directory.glob("*.jsonl")) if directory.is_dir() else []:
            for number, line in enumerate(path.read_text().splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    rows.append(Narrative.model_validate_json(line))
                except ValidationError:
                    log.warning("skipping invalid narrative %s:%d", path.name, number)
        return cls(rows)

    def __len__(self) -> int:
        return sum(len(v) for v in self._by_profile.values())

    def profiles(self) -> list[tuple[str, str]]:
        return sorted(self._by_profile)

    def for_profile(self, language: str, mode: str) -> list[Narrative]:
        return list(self._by_profile.get((language, mode), []))

    def between(self, language: str, mode: str, t_after: float, t_upto: float) -> list[Narrative]:
        """Narratives for this profile with ``t_after < t <= t_upto``, in time order."""
        key = (language, mode)
        times = self._times.get(key)
        if not times:
            return []
        lo = bisect.bisect_right(times, t_after)
        hi = bisect.bisect_right(times, t_upto)
        return self._by_profile[key][lo:hi]
