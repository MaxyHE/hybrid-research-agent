"""A source-attributed, runnable Open Deep Research baseline adaptation."""

from .hybrid import (
    CollectionSourcePack,
    build_hybrid_odr_runner,
    odr_p1_deep_evidence_handoff_policy,
    odr_p1_deep_policy,
)
from .runtime import (
    OdrBaselineArtifacts,
    OdrBaselinePolicy,
    OdrBaselineResult,
    OdrBaselineRunner,
)

__all__ = [
    "CollectionSourcePack",
    "OdrBaselineArtifacts",
    "OdrBaselinePolicy",
    "OdrBaselineResult",
    "OdrBaselineRunner",
    "build_hybrid_odr_runner",
    "odr_p1_deep_evidence_handoff_policy",
    "odr_p1_deep_policy",
]
