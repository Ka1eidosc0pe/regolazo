"""Engine plumbing: snapshot refs and loading the ``RolloutEngine`` implementation named in configuration."""

import importlib

from shared.engine_api import RolloutEngine


class UnknownSnapshotError(KeyError):
    """The snapshot ref is malformed or does not exist (mapped to HTTP 404)."""


def parse_snapshot_ref(ref: str) -> tuple[str, int]:
    """``snap:<match_id>:<tick>`` -> (match_id, tick)."""
    parts = ref.split(":")
    if len(parts) != 3 or parts[0] != "snap" or not parts[1] or not parts[2].isdigit():
        raise UnknownSnapshotError(f"malformed snapshot ref {ref!r}")
    return parts[1], int(parts[2])


def load_engine(spec: str, match_id: str | None = None) -> RolloutEngine:
    """``fake`` -> ``FakeEngine``; ``package.module:factory`` -> ``factory(match_id)`` (e.g. the real simulator).

    The factory may be any callable returning an object that implements ``shared.engine_api.RolloutEngine``.
    """
    if spec == "fake":
        from server.fake_engine import FakeEngine

        return FakeEngine(match_id)
    module_name, _, attr = spec.partition(":")
    if not attr:
        raise ValueError(f"engine spec {spec!r} must be 'fake' or 'module:callable'")
    engine = getattr(importlib.import_module(module_name), attr)(match_id)
    if not isinstance(engine, RolloutEngine):
        raise TypeError(f"{spec} did not return a RolloutEngine")
    return engine
