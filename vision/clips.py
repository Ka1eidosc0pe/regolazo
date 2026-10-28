"""Export time windows of recorded matches as renderer clips ({meta, frames}), loaded with ?clip=/spotter/<name>."""

import json
from pathlib import Path

from shared.recording import Recording


def export_clip(data_dir: Path, match: str, start: float, end: float, out: Path) -> Path:
    rec = Recording.open(data_dir / "matches" / match)
    i0, i1 = rec.index_at(start), rec.index_at(end)
    frames = [rec.state_at(i).model_dump(mode="json") for i in range(i0, i1 + 1)]
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{match}-{int(start)}-{int(end)}.json"
    path.write_text(json.dumps({"meta": rec.meta.model_dump(mode="json"), "frames": frames}))
    return path
