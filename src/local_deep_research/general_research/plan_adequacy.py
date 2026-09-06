"""Deterministic check that a brief's required intent survives decomposition."""

from __future__ import annotations

from dataclasses import dataclass

from .schemas import ResearchBrief, ResearchPlan


PLAN_ADEQUACY_AUDIT_SCHEMA_VERSION = "plan-adequacy-audit/v1"


@dataclass(frozen=True, slots=True)
class RequirementPlanDecision:
    """One brief requirement and the plan items explicitly serving it."""

    requirement_id: str
    required: bool
    plan_item_ids: tuple[str, ...]

    @property
    def is_adequate(self) -> bool:
        return bool(self.plan_item_ids) or not self.required

    def to_dict(self) -> dict[str, object]:
        return {
            "requirement_id": self.requirement_id,
            "required": self.required,
            "plan_item_ids": list(self.plan_item_ids),
            "is_adequate": self.is_adequate,
        }


@dataclass(frozen=True, slots=True)
class PlanAdequacyAudit:
    """Replayable result of the Brief -> Plan mapping audit.

    It deliberately checks explicit links only.  Semantic equivalence between
    natural-language sentences is a model judgement; silently inferring it
    here would make the gate opaque and unstable.
    """

    brief_id: str
    plan_id: str
    decisions: tuple[RequirementPlanDecision, ...]
    schema_version: str = PLAN_ADEQUACY_AUDIT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PLAN_ADEQUACY_AUDIT_SCHEMA_VERSION:
            raise ValueError("unsupported plan adequacy audit schema")
        if not self.brief_id or not self.plan_id:
            raise ValueError("plan adequacy audit requires brief_id and plan_id")
        decisions = tuple(self.decisions)
        if not all(isinstance(item, RequirementPlanDecision) for item in decisions):
            raise TypeError("decisions must contain RequirementPlanDecision values")
        requirement_ids = [item.requirement_id for item in decisions]
        if len(requirement_ids) != len(set(requirement_ids)):
            raise ValueError("plan adequacy decisions must use unique requirement IDs")
        object.__setattr__(self, "decisions", decisions)

    @property
    def missing_required_requirement_ids(self) -> tuple[str, ...]:
        return tuple(
            decision.requirement_id
            for decision in self.decisions
            if decision.required and not decision.is_adequate
        )

    @property
    def is_adequate(self) -> bool:
        return not self.missing_required_requirement_ids

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "brief_id": self.brief_id,
            "plan_id": self.plan_id,
            "decisions": [decision.to_dict() for decision in self.decisions],
            "missing_required_requirement_ids": list(
                self.missing_required_requirement_ids
            ),
            "is_adequate": self.is_adequate,
        }


def audit_plan_adequacy(brief: ResearchBrief, plan: ResearchPlan) -> PlanAdequacyAudit:
    """Ensure all explicit required brief requirements appear in the plan."""

    if not isinstance(brief, ResearchBrief) or not isinstance(plan, ResearchPlan):
        raise TypeError("brief and plan must use General research contracts")
    if brief.run_id != plan.run_id or brief.user_query != plan.query:
        raise ValueError("brief and plan must belong to the same run and query")
    known_requirement_ids = {
        requirement.requirement_id for requirement in brief.requirements
    }
    for item in plan.items:
        if not item.requirement_ids:
            raise ValueError(
                f"agentic plan item {item.item_id!r} has no brief requirement link"
            )
        unknown = set(item.requirement_ids).difference(known_requirement_ids)
        if unknown:
            raise ValueError(
                f"agentic plan item {item.item_id!r} names unknown requirements: "
                + ", ".join(sorted(unknown))
            )
    decisions = tuple(
        RequirementPlanDecision(
            requirement_id=requirement.requirement_id,
            required=requirement.required,
            plan_item_ids=tuple(
                item.item_id
                for item in plan.items
                if requirement.requirement_id in item.requirement_ids
            ),
        )
        for requirement in brief.requirements
    )
    return PlanAdequacyAudit(
        brief_id=brief.brief_id,
        plan_id=plan.plan_id,
        decisions=decisions,
    )


__all__ = [
    "PLAN_ADEQUACY_AUDIT_SCHEMA_VERSION",
    "PlanAdequacyAudit",
    "RequirementPlanDecision",
    "audit_plan_adequacy",
]
