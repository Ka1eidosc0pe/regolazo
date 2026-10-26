"""ReGolazo contracts (PLAN.md section 4.2). Frozen at SCHEMA_VERSION; changes go through the lead engineer."""

from shared.schemas.actions import ActionIntent, BallLaunch, ExecutionEllipse, ResolvedAction
from shared.schemas.common import SCHEMA_VERSION
from shared.schemas.events import MatchEvent
from shared.schemas.meta import MatchMeta, PlayerAttributes, PlayerMeta, TeamMeta
from shared.schemas.models import ModelOutputs, OptionEvaluation, PitchControlGrid
from shared.schemas.moments import DecisionMoment, RewindScores
from shared.schemas.narrative import Narrative, ViewerProfile
from shared.schemas.overlay import OverlayCue
from shared.schemas.rollout import RolloutDistribution, RolloutRequest, RolloutResult
from shared.schemas.state import SimState

ROOT_MODELS = (
    MatchMeta,
    SimState,
    MatchEvent,
    ModelOutputs,
    DecisionMoment,
    RolloutRequest,
    RolloutResult,
    RolloutDistribution,
    OverlayCue,
    Narrative,
    ViewerProfile,
)
"""Top-level contracts: each gets a JSON Schema file, a TS type and an example fixture."""

__all__ = [
    "ROOT_MODELS",
    "SCHEMA_VERSION",
    "ActionIntent",
    "BallLaunch",
    "DecisionMoment",
    "ExecutionEllipse",
    "MatchEvent",
    "MatchMeta",
    "ModelOutputs",
    "Narrative",
    "OptionEvaluation",
    "OverlayCue",
    "PitchControlGrid",
    "PlayerAttributes",
    "PlayerMeta",
    "ResolvedAction",
    "RewindScores",
    "RolloutDistribution",
    "RolloutRequest",
    "RolloutResult",
    "SimState",
    "TeamMeta",
    "ViewerProfile",
]
