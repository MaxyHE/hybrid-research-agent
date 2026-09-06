"""Runtime-neutral evidence-policy decisions for research-agent actions.

The guard deliberately separates a model's *proposal* from the runtime's
decision to execute it.  It contains no model calls and no source content: it
only uses action history plus coarse evidence counters, making the same policy
auditable against frozen replay scenarios before it is enabled online.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
import json
from typing import Any, Iterable


class EvidencePolicyMode(StrEnum):
    """How a runtime should use the guard's decisions."""

    OFF = "off"
    SHADOW = "shadow"
    ENFORCE = "enforce"

    @classmethod
    def parse(cls, value: str | None) -> "EvidencePolicyMode":
        """Parse a user setting without making malformed config dangerous."""
        try:
            return cls((value or cls.OFF.value).strip().lower())
        except ValueError:
            return cls.OFF


class EvidencePolicyVerdict(StrEnum):
    ALLOW = "allow"
    ADVISE = "advise"
    BLOCK = "block"


@dataclass(frozen=True)
class EvidencePolicyDecision:
    """One explainable review of a proposed tool batch or STOP action."""

    verdict: EvidencePolicyVerdict
    reason: str
    mode: EvidencePolicyMode
    evidence_count: int
    search_calls: int
    fetch_calls: int
    proposed_tool_calls: int

    @property
    def is_blocking(self) -> bool:
        return self.verdict == EvidencePolicyVerdict.BLOCK

    def trace_metadata(self) -> dict[str, Any]:
        """Return bounded metadata safe to retain with a trace run summary."""
        return {
            "verdict": self.verdict.value,
            "reason": self.reason,
            "mode": self.mode.value,
            "evidence_count": self.evidence_count,
            "search_calls": self.search_calls,
            "fetch_calls": self.fetch_calls,
            "proposed_tool_calls": self.proposed_tool_calls,
        }


def _call_name_and_arguments(
    call: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Accept LangChain- and OpenAI-shaped tool calls without mutation."""
    function = call.get("function")
    if isinstance(function, dict):
        name = function.get("name")
        arguments = function.get("arguments", {})
    else:
        name = call.get("name")
        arguments = call.get("args", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {"_raw": arguments}
    if not isinstance(arguments, dict):
        arguments = {"_raw": str(arguments)}
    return str(name or ""), arguments


def _tool_signature(name: str, arguments: dict[str, Any]) -> str:
    return f"{name}:{json.dumps(arguments, ensure_ascii=False, sort_keys=True, default=str)}"


def _is_search_tool(name: str) -> bool:
    return name == "web_search" or name.startswith("search_")


@dataclass
class EvidencePolicyGuard:
    """Track coarse evidence state and review future Planner proposals.

    v1 intentionally has a small enforcement surface. After at least one
    evidence item exists, exact duplicate batches and tool-budget excess are
    safe to block because they cannot add new evidence. Fetch-before-STOP /
    Fetch-before-more-search are advisory until a fresh end-to-end evaluation
    establishes they do not reject valid short-path research tasks.
    """

    mode: EvidencePolicyMode = EvidencePolicyMode.OFF
    require_fetch_before_stop: bool = False
    _seen_signatures: set[str] = field(default_factory=set, init=False)
    _search_calls: int = field(default=0, init=False)
    _fetch_calls: int = field(default=0, init=False)
    _decision_counts: Counter[str] = field(default_factory=Counter, init=False)
    _non_allow_decisions: list[dict[str, Any]] = field(
        default_factory=list, init=False
    )

    @property
    def enabled(self) -> bool:
        return self.mode != EvidencePolicyMode.OFF

    @property
    def search_calls(self) -> int:
        return self._search_calls

    @property
    def fetch_calls(self) -> int:
        return self._fetch_calls

    def _make_decision(
        self,
        verdict: EvidencePolicyVerdict,
        reason: str,
        *,
        evidence_count: int,
        proposed_tool_calls: int,
    ) -> EvidencePolicyDecision:
        decision = EvidencePolicyDecision(
            verdict=verdict,
            reason=reason,
            mode=self.mode,
            evidence_count=max(0, evidence_count),
            search_calls=self.search_calls,
            fetch_calls=self.fetch_calls,
            proposed_tool_calls=max(0, proposed_tool_calls),
        )
        if self.enabled:
            self._decision_counts[f"{verdict.value}:{reason}"] += 1
            if verdict != EvidencePolicyVerdict.ALLOW:
                # Keep trace run metadata bounded even if an agent loops.
                if len(self._non_allow_decisions) < 32:
                    self._non_allow_decisions.append(decision.trace_metadata())
        return decision

    @staticmethod
    def _normalized_calls(
        tool_calls: Iterable[dict[str, Any]],
    ) -> list[tuple[str, dict[str, Any], str]]:
        normalized = []
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            name, arguments = _call_name_and_arguments(call)
            if not name:
                continue
            normalized.append(
                (name, arguments, _tool_signature(name, arguments))
            )
        return normalized

    def observe_calls(self, tool_calls: Iterable[dict[str, Any]]) -> None:
        """Commit calls that were accepted for execution to this state."""
        for name, _arguments, signature in self._normalized_calls(tool_calls):
            self._seen_signatures.add(signature)
            if name == "fetch_content":
                self._fetch_calls += 1
            elif _is_search_tool(name):
                self._search_calls += 1

    def review_tool_batch(
        self,
        tool_calls: Iterable[dict[str, Any]],
        *,
        evidence_count: int,
        scheduled_tool_calls: int | None = None,
        max_tool_calls: int | None = None,
    ) -> EvidencePolicyDecision | None:
        """Review a proposed tool batch without changing guard state."""
        if not self.enabled:
            return None
        calls = self._normalized_calls(tool_calls)
        if not calls:
            return self._make_decision(
                EvidencePolicyVerdict.ALLOW,
                "no_tool_calls",
                evidence_count=evidence_count,
                proposed_tool_calls=0,
            )
        if (
            evidence_count > 0
            and max_tool_calls is not None
            and scheduled_tool_calls is not None
        ):
            if scheduled_tool_calls + len(calls) > max_tool_calls:
                return self._make_decision(
                    EvidencePolicyVerdict.BLOCK,
                    "tool_call_budget",
                    evidence_count=evidence_count,
                    proposed_tool_calls=len(calls),
                )
        if evidence_count > 0 and all(
            signature in self._seen_signatures for _, _, signature in calls
        ):
            return self._make_decision(
                EvidencePolicyVerdict.BLOCK,
                "duplicate_tool_call_batch",
                evidence_count=evidence_count,
                proposed_tool_calls=len(calls),
            )
        all_search = all(_is_search_tool(name) for name, _, _ in calls)
        if (
            self.require_fetch_before_stop
            and evidence_count > 0
            and self.search_calls > 0
            and self.fetch_calls == 0
            and all_search
        ):
            return self._make_decision(
                EvidencePolicyVerdict.ADVISE,
                "fetch_recommended_before_additional_search",
                evidence_count=evidence_count,
                proposed_tool_calls=len(calls),
            )
        return self._make_decision(
            EvidencePolicyVerdict.ALLOW,
            "tool_batch_allowed",
            evidence_count=evidence_count,
            proposed_tool_calls=len(calls),
        )

    def review_stop(
        self, *, evidence_count: int
    ) -> EvidencePolicyDecision | None:
        """Flag a possibly premature final response; v1 does not block it."""
        if not self.enabled:
            return None
        if (
            self.require_fetch_before_stop
            and evidence_count > 0
            and self.search_calls > 0
            and self.fetch_calls == 0
        ):
            return self._make_decision(
                EvidencePolicyVerdict.ADVISE,
                "stop_without_fetch",
                evidence_count=evidence_count,
                proposed_tool_calls=0,
            )
        return self._make_decision(
            EvidencePolicyVerdict.ALLOW,
            "stop_allowed",
            evidence_count=evidence_count,
            proposed_tool_calls=0,
        )

    def trace_summary(self) -> dict[str, Any] | None:
        """Return run-level audit metadata, or ``None`` when disabled."""
        if not self.enabled:
            return None
        return {
            "mode": self.mode.value,
            "require_fetch_before_stop": self.require_fetch_before_stop,
            "search_calls": self.search_calls,
            "fetch_calls": self.fetch_calls,
            "decisions": dict(sorted(self._decision_counts.items())),
            "non_allow_decisions": list(self._non_allow_decisions),
        }
