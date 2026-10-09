"""Minimal, environment-independent HPD algorithm components."""

from .harness_replay import (
    HarnessAdapter,
    ReplayError,
    build_three_view_examples,
    replay_trajectory,
)
from .tir_training import (
    TIRTrainer,
    ThreeStageSchedule,
    completion_only_nll,
    make_three_view_examples,
    validate_complete_triplets,
)
from .types import (
    ModelAction,
    ReferenceTurn,
    ReplayBoundary,
    TextExample,
    ToolCall,
    ToolObservation,
    Trajectory,
)

__all__ = [
    "HarnessAdapter",
    "ModelAction",
    "ReferenceTurn",
    "ReplayBoundary",
    "ReplayError",
    "TIRTrainer",
    "TextExample",
    "ThreeStageSchedule",
    "ToolCall",
    "ToolObservation",
    "Trajectory",
    "build_three_view_examples",
    "completion_only_nll",
    "make_three_view_examples",
    "replay_trajectory",
    "validate_complete_triplets",
]
