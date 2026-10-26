"""A contract-valid stand-in for the simulator's ``RolloutEngine`` (tests and development).

Everything is derived from ``shared.schemas.fixtures``: one fixed resolved action, option list and outcome
distribution, re-labelled with the request's snapshot, intent, seed and sample size. No simulation happens.
"""

import hashlib

from server.engine import UnknownSnapshotError, parse_snapshot_ref
from server.fixture import fixture_states
from shared.schemas.actions import ActionIntent, ResolvedAction
from shared.schemas.fixtures import model_outputs, resolved_action, rollout_distribution, rollout_result
from shared.schemas.models import ModelOutputs
from shared.schemas.rollout import (
    OutcomeCount,
    RolloutDistribution,
    RolloutRequest,
    RolloutResult,
)

SIM_HZ = 25


def _derived_seed(request: RolloutRequest) -> int:
    if request.seed is not None:
        return request.seed
    digest = hashlib.sha256(request.snapshot_ref.encode()).digest()
    return int.from_bytes(digest[:4], "big")


class FakeEngine:
    """Implements ``shared.engine_api.RolloutEngine`` with canned, contract-valid answers."""

    def __init__(self, match_id: str | None = None) -> None:
        self.match_id = match_id
        self.calls: list[str] = []

    def _check(self, ref: str) -> int:
        match_id, tick = parse_snapshot_ref(ref)
        if self.match_id is not None and match_id != self.match_id:
            raise UnknownSnapshotError(f"snapshot {ref!r} belongs to another match")
        return tick

    def preview(self, snapshot_ref: str, intent: ActionIntent) -> ResolvedAction:
        self._check(snapshot_ref)
        self.calls.append("preview")
        return resolved_action().model_copy(update={"intent": intent})

    def options(self, snapshot_ref: str) -> ModelOutputs:
        tick = self._check(snapshot_ref)
        self.calls.append("options")
        return model_outputs().model_copy(update={"tick": tick, "t": tick / SIM_HZ})

    def rollout(self, request: RolloutRequest) -> RolloutResult:
        tick = self._check(request.snapshot_ref)
        self.calls.append("rollout")
        seed = _derived_seed(request)
        rid = f"roll_{seed:08x}"
        base = rollout_result()
        resolved = resolved_action().model_copy(update={"intent": request.intent})
        frames = []
        if request.return_frames and request.n == 1:
            n_frames = int(request.horizon_s * SIM_HZ)
            frames = [
                s.model_copy(update={"tick": tick + s.tick, "t": round((tick + s.tick) / SIM_HZ, 6)})
                for s in fixture_states(n_frames)[:: request.frame_stride]
            ]
        events = [
            e.model_copy(update={"id": f"ev_r_{rid}_{i:04d}", "index": i, "rollout_id": rid})
            for i, e in enumerate(base.events)
        ]
        return base.model_copy(
            update={
                "id": rid,
                "snapshot_ref": request.snapshot_ref,
                "seed": seed,
                "resolved": resolved,
                "evaluation": base.evaluation.model_copy(update={"intent": request.intent}),
                "frames": frames,
                "events": events,
            }
        )

    def distribution(self, request: RolloutRequest) -> RolloutDistribution:
        self._check(request.snapshot_ref)
        self.calls.append("distribution")
        base = rollout_distribution()
        total = sum(c.count for c in base.outcome_counts)
        counts = [OutcomeCount(label=c.label, count=round(c.count * request.n / total)) for c in base.outcome_counts]
        counts[0] = counts[0].model_copy(update={"count": counts[0].count + request.n - sum(c.count for c in counts)})
        return base.model_copy(
            update={
                "snapshot_ref": request.snapshot_ref,
                "intent": request.intent,
                "resolved": resolved_action().model_copy(update={"intent": request.intent}),
                "n": request.n,
                "seed": _derived_seed(request),
                "outcome_counts": counts,
            }
        )
