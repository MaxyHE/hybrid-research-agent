"""Evidence-first contracts and orchestration primitives for General V1.

This package is deliberately separate from the legacy Hybrid strategy.  It
provides the versioned research artifacts that a General runner must produce:
plan, fetched-source registry, snapshot-bound evidence cards, coverage state,
and citation audit.  Runtime graph integration is intentionally allowed only
through these contracts, never through a task-specific contract or gold URL.
"""

from .schemas import (
    CoverageDecision,
    CoverageState,
    CoverageStatus,
    CoverageVerdict,
    EvidenceCard,
    EvidenceStance,
    GeneralRunConfig,
    MemoFinding,
    PlanItem,
    ResearchBrief,
    ResearchRequirement,
    ResearchMemo,
    ResearchPlan,
    ResearchTask,
    SourceClass,
    SourceClassificationBasis,
    SourceChannel,
    SourceRecord,
    SupervisorDecision,
)

__all__ = [
    "CoverageDecision",
    "CoverageState",
    "CoverageStatus",
    "CoverageVerdict",
    "EvidenceCard",
    "EvidenceStance",
    "GeneralRunConfig",
    "MemoFinding",
    "PlanItem",
    "ResearchBrief",
    "ResearchRequirement",
    "ResearchMemo",
    "ResearchPlan",
    "ResearchTask",
    "SourceClass",
    "SourceClassificationBasis",
    "SourceChannel",
    "SourceRecord",
    "SupervisorDecision",
]
