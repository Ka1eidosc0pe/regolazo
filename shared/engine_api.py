"""Interface the simulator exposes to the live server and the agents (fork, preview, rollouts, options).

Implemented by sim/ (Step 3). Consumers (server/, agents/) code against this protocol and use a fake engine in
tests until the real one lands.
"""

from typing import Protocol, runtime_checkable

from shared.schemas.actions import ActionIntent, ResolvedAction
from shared.schemas.models import ModelOutputs
from shared.schemas.rollout import RolloutDistribution, RolloutRequest, RolloutResult


@runtime_checkable
class RolloutEngine(Protocol):
    def preview(self, snapshot_ref: str, intent: ActionIntent) -> ResolvedAction:
        """Resolve an intent without executing it: technique, preview arc and execution ellipse. Target < 50 ms."""
        ...

    def options(self, snapshot_ref: str) -> ModelOutputs:
        """Options available to the ball carrier at a snapshot, scored by the analytics models."""
        ...

    def rollout(self, request: RolloutRequest) -> RolloutResult:
        """Fork at ``request.snapshot_ref``, inject the intent, simulate ``horizon_s``. Target < 1 s incl. frames."""
        ...

    def distribution(self, request: RolloutRequest) -> RolloutDistribution:
        """``request.n`` batched rollouts (frames not returned). Target: 1,000 x 10 s in a few seconds."""
        ...
