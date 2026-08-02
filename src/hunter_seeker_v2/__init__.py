"""Compact, transactional Hunter-Seeker runtime.

The package is intentionally side-by-side with :mod:`hunter_seeker_core`.
Nothing here imports the legacy mixin stack.
"""

from .contracts import (
    Action,
    AgentConfig,
    BoundaryKind,
    Decision,
    Observation,
    Outcome,
    RuntimeMode,
    Transition,
)
from .agent import CompactHunterSeeker
from .adapters import (
    ArcActionAdapter,
    ArcObservationAdapter,
    ArcOutcomeAdapter,
)
from .student import (
    StateConditionedStudentPolicy,
    StudentPolicyConfig,
    StudentRepresentationTrajectory,
    StudentTeacherSample,
)

__all__ = [
    "Action",
    "AgentConfig",
    "BoundaryKind",
    "CompactHunterSeeker",
    "Decision",
    "Observation",
    "Outcome",
    "RuntimeMode",
    "Transition",
    "ArcActionAdapter",
    "ArcObservationAdapter",
    "ArcOutcomeAdapter",
    "StateConditionedStudentPolicy",
    "StudentPolicyConfig",
    "StudentRepresentationTrajectory",
    "StudentTeacherSample",
]
