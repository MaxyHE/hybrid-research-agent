"""Deterministic product routing for bounded Hybrid research tasks.

The router is deliberately contract-driven.  It does not ask an LLM to guess
whether a fixed workflow applies: callers must provide the source roles and
official-domain constraints that make a static route safe to use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


AUTO_ROUTING_MODE = "auto"
ROUTING_MODES = {
    AUTO_ROUTING_MODE,
    "adaptive_hybrid",
    "web_only",
    "collection_only",
    "static_dual",
    "static_dual_fetch",
    "static_top1",
    "static_topk",
    # Development-only candidate-selection arms.  They are explicit routes,
    # never selected by ``auto``.
    "static_ranked_fetch_2",
    "one_shot_llm_selector",
}


def _domains(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        candidates = [value]
    elif isinstance(value, (list, tuple, set)):
        candidates = list(value)
    else:
        candidates = []
    return tuple(
        sorted(
            {
                item.strip().lower().lstrip(".")
                for item in candidates
                if isinstance(item, str) and item.strip()
            }
        )
    )


@dataclass(frozen=True)
class RoutingDecision:
    """A traceable routing decision made before tool execution."""

    requested_mode: str
    selected_mode: str
    reason: str
    official_domain: str | None = None

    def metadata(self) -> dict[str, Any]:
        return {
            "requested_mode": self.requested_mode,
            "selected_mode": self.selected_mode,
            "reason": self.reason,
            "official_domain": self.official_domain,
        }


def choose_routing_mode(
    *,
    requested_mode: str | None,
    research_profile: str,
    task_contract: Mapping[str, Any] | None,
    static_official_domain: str | None = None,
) -> RoutingDecision:
    """Choose a product route without changing explicit experiment arms.

    ``auto`` selects ``static_dual_fetch`` only when the caller supplies a
    bounded two-source contract: a Collection snapshot, exactly one official
    domain, and no declared discovery, comparison, or recovery ambiguity.
    Every other Hybrid task keeps the learned adaptive controller available.
    """
    requested = (requested_mode or AUTO_ROUTING_MODE).strip().lower()
    if requested not in ROUTING_MODES:
        raise ValueError(
            f"Unknown agent routing mode {requested_mode!r}; expected one of "
            f"{sorted(ROUTING_MODES)}."
        )
    if requested != AUTO_ROUTING_MODE:
        explicit_domain = (static_official_domain or "").strip().lower()
        return RoutingDecision(
            requested_mode=requested,
            selected_mode=requested,
            reason="explicit_mode",
            official_domain=explicit_domain.lstrip(".") or None,
        )

    if research_profile.strip().lower() != "hybrid":
        return RoutingDecision(
            requested_mode=requested,
            selected_mode="adaptive_hybrid",
            reason="non_hybrid_profile",
        )
    if not isinstance(task_contract, Mapping):
        return RoutingDecision(
            requested_mode=requested,
            selected_mode="static_dual",
            reason="missing_task_contract_static_fallback",
        )

    collection_snapshot_required = bool(
        task_contract.get("collection_snapshot_required")
    )
    official_page_required = bool(task_contract.get("official_page_required"))
    domains = _domains(
        task_contract.get("official_domains")
        or task_contract.get("official_domain")
        or static_official_domain
    )
    adaptive_required = any(
        bool(task_contract.get(key))
        for key in (
            "source_discovery_required",
            "multiple_official_pages_required",
            "candidate_ambiguity",
            "fetch_recovery_required",
            "comparison_required",
        )
    )
    if (
        collection_snapshot_required
        and official_page_required
        and len(domains) == 1
        and not adaptive_required
    ):
        return RoutingDecision(
            requested_mode=requested,
            selected_mode="static_dual_fetch",
            reason="bounded_collection_plus_single_official_page",
            official_domain=domains[0],
        )
    if adaptive_required:
        return RoutingDecision(
            requested_mode=requested,
            selected_mode="adaptive_hybrid",
            reason="adaptive_contract_requirement",
        )
    if not collection_snapshot_required or not official_page_required:
        reason = "incomplete_contract_static_fallback"
    elif len(domains) != 1:
        reason = "official_domain_not_unique_static_fallback"
    else:
        reason = "static_fallback"
    return RoutingDecision(
        requested_mode=requested,
        selected_mode="static_dual",
        reason=reason,
    )
