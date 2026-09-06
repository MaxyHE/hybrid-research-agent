"""Observation-derived decision state for Hybrid research planners.

The tracker deliberately exposes only what the runtime has observed: the
user-visible task requirements, search candidates, fetch attempts, and the
evidence registry.  It is *not* an evaluator and must never receive gold
URLs, expected answers, or task-fixture annotations.  Its role is to make
otherwise implicit recovery and stopping choices legible to a small planner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
from urllib.parse import urlparse


PLANNER_UNCERTAINTY_STATE_PROTOCOL = "planner-uncertainty-state/v1"


def _clean_url(value: Any) -> str:
    return str(value or "").strip()


def _result_url(result: Mapping[str, Any]) -> str:
    return _clean_url(result.get("link") or result.get("url"))


def _result_family(result: Mapping[str, Any]) -> str:
    source_engine = str(result.get("source_engine") or "").lower()
    url = _result_url(result)
    if source_engine == "fetch":
        return "fetch"
    if source_engine.startswith("collection_") or url.startswith(
        "/library/document/"
    ):
        return "collection"
    return "web"


def _fetch_failure_reason(observation: str) -> str | None:
    text = observation.lower()
    if not text.strip():
        return "empty_body"
    # The Fetch runtime refuses URLs that are absent from the current
    # SearchResultsCollector.  Keep this distinct from ordinary HTTP/fetch
    # errors so a trace can prove that the policy guard, rather than network
    # behavior, blocked the attempt.
    if "cannot fetch an unobserved url" in text:
        return "unobserved_url"
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if "403" in text or "blocked" in text or "forbidden" in text:
        return "blocked"
    if "landing page" in text:
        return "landing_page"
    if "source role" in text and "mismatch" in text:
        return "source_role_mismatch"
    if "no extractable content" in text:
        return "empty_body"
    if "not relevant" in text or "irrelevant" in text:
        return "irrelevant_body"
    if (
        "cannot fetch" in text
        or "error fetching" in text
        or "failed to fetch" in text
    ):
        return "fetch_error"
    return None


def _keywords(value: str) -> set[str]:
    """Return modest lexical clues without pretending to understand a page."""
    return {
        word.lower()
        for word in value.replace("/", " ").replace("-", " ").split()
        if len(word.strip()) >= 4
    }


@dataclass(frozen=True)
class _Requirement:
    identifier: str
    family: str
    description: str
    min_count: int = 1
    match_terms: tuple[str, ...] = ()


class UncertaintyStateTracker:
    """Maintain a compact, prompt-safe view of uncertainty during a run."""

    def __init__(self, task_contract: Mapping[str, Any] | None = None) -> None:
        self._task_contract = dict(task_contract or {})
        self._requirements = self._derive_requirements(self._task_contract)
        self.reset()

    @staticmethod
    def _derive_requirements(
        task_contract: Mapping[str, Any],
    ) -> list[_Requirement]:
        """Read only agent-visible requirements; ignore evaluator-only fields."""
        requirements: list[_Requirement] = []
        raw_requirements = task_contract.get("agent_requirements")
        if isinstance(raw_requirements, list):
            for index, raw in enumerate(raw_requirements):
                if not isinstance(raw, Mapping):
                    continue
                family = str(raw.get("family") or "fetch").strip().lower()
                if family not in {"collection", "web", "fetch"}:
                    continue
                identifier = str(raw.get("id") or f"requirement_{index + 1}")
                description = str(raw.get("description") or identifier).strip()
                try:
                    min_count = max(1, int(raw.get("min_count", 1)))
                except (TypeError, ValueError):
                    min_count = 1
                raw_terms = raw.get("match_terms")
                if isinstance(raw_terms, str):
                    raw_terms = [raw_terms]
                match_terms = tuple(
                    term.strip().lower()
                    for term in raw_terms or []
                    if isinstance(term, str) and term.strip()
                )
                requirements.append(
                    _Requirement(
                        identifier,
                        family,
                        description,
                        min_count,
                        match_terms,
                    )
                )
        if requirements:
            return requirements

        # Backwards-compatible projection of the existing bounded contract.
        if task_contract.get("collection_snapshot_required"):
            requirements.append(
                _Requirement("collection_snapshot", "collection", "Collection evidence")
            )
        if task_contract.get("official_page_required"):
            requirements.append(
                _Requirement("current_web_evidence", "fetch", "Fetched current web evidence")
            )
        if task_contract.get("comparison_required"):
            requirements.extend(
                [
                    _Requirement("comparison_side_a", "fetch", "Comparison side A"),
                    _Requirement("comparison_side_b", "fetch", "Comparison side B"),
                ]
            )
        return requirements

    def reset(self, query: str = "") -> None:
        self.query = query
        self._calls: dict[str, dict[str, Any]] = {}
        self._fetch_attempts: dict[str, dict[str, Any]] = {}
        self._results: list[dict[str, Any]] = []
        self._collection_queries: list[str] = []
        self._web_queries: list[str] = []
        self._stop_attempts = 0
        self._legal_stop_accepted = 0
        self._illegal_stop_rejected = 0
        self._illegal_stop_recovered = 0
        self._tool_calls_after_sufficient = 0
        self._unobserved_url_attempts = 0

    def observe_tool_calls(self, tool_calls: list[Mapping[str, Any]] | Any) -> None:
        if not isinstance(tool_calls, list):
            return
        if self.snapshot()["evidence_sufficient"]:
            self._tool_calls_after_sufficient += len(tool_calls)
        for call in tool_calls:
            if not isinstance(call, Mapping):
                continue
            call_id = str(call.get("id") or "")
            if not call_id:
                continue
            self._calls[call_id] = {
                "name": str(call.get("name") or ""),
                "args": dict(call.get("args") or {}),
            }
            name = self._calls[call_id]["name"]
            query = str(self._calls[call_id]["args"].get("query") or "").strip()
            if query and (
                name == "search_library" or name.startswith("search_collection_")
            ):
                if query not in self._collection_queries:
                    self._collection_queries.append(query)
            if query and (name == "web_search" or name.startswith("search_web_")):
                if query not in self._web_queries:
                    self._web_queries.append(query)

    def observe_tool_observation(self, message: Any) -> None:
        call_id = str(getattr(message, "tool_call_id", "") or "")
        call = self._calls.get(call_id)
        if not call or call.get("name") != "fetch_content":
            return
        url = _clean_url(call.get("args", {}).get("url"))
        if not url:
            return
        observation = str(getattr(message, "content", "") or "")
        reason = _fetch_failure_reason(observation)
        if reason == "unobserved_url":
            self._unobserved_url_attempts += 1
        self._fetch_attempts[url] = {
            "url": url,
            "status": "failed" if reason else "pending",
            "reason": reason,
            "observation_excerpt": observation[:180],
        }

    def record_stop_attempt(self) -> None:
        self._stop_attempts += 1

    def review_stop_attempt(self) -> dict[str, Any]:
        """Record and classify one tentative planner STOP from visible state."""
        self._stop_attempts += 1
        state = self.snapshot()
        incomplete_stop_legal = "STOP_INCOMPLETE" in state[
            "legal_action_families"
        ]
        if state["stop_legal"]:
            self._legal_stop_accepted += 1
            if self._illegal_stop_rejected:
                self._illegal_stop_recovered += 1
        elif not incomplete_stop_legal:
            self._illegal_stop_rejected += 1
        return {
            "legal": bool(state["stop_legal"]),
            "incomplete_stop_legal": incomplete_stop_legal,
            "blockers": list(state["stop_blockers"]),
            "legal_action_families": list(state["legal_action_families"]),
            "preferred_action_families": list(state["preferred_action_families"]),
            "action_guidance": dict(state["action_guidance"]),
        }

    def recovery_action(self, tool_names: list[str]) -> dict[str, Any] | None:
        """Return one observation-derived action after repeated illegal STOPs."""
        state = self.snapshot()
        preferred = str(
            (state.get("action_guidance") or {}).get("preferred_action_family")
            or ""
        )

        def choose(predicate: Any) -> str | None:
            return next((name for name in tool_names if predicate(name)), None)

        if preferred == "Collection Search":
            name = choose(
                lambda value: value == "search_library"
                or value.startswith("search_collection_")
            )
            previous = {
                query.casefold()
                for query in state.get("collection_search_queries") or []
            }
            candidates: list[str] = []
            for item in state.get("missing_requirements") or []:
                if item.get("family") != "collection":
                    continue
                description = str(item.get("description") or "")
                cleaned = description
                for token in ("collection", "background", "evidence"):
                    cleaned = cleaned.replace(token, " ").replace(
                        token.title(), " "
                    )
                candidates.append(" ".join(cleaned.split()))
            candidates.extend(
                [
                    str(self._task_contract.get("static_collection_query") or ""),
                    self.query,
                ]
            )
            query = next(
                (
                    candidate
                    for candidate in candidates
                    if candidate and candidate.casefold() not in previous
                ),
                "",
            )
            if name and query:
                return {
                    "family": preferred,
                    "tool_name": name,
                    "tool_input": {"query": query},
                }

        if preferred == "Fetch observed candidate":
            name = choose(lambda value: value == "fetch_content")
            candidates = [
                item
                for item in state.get("candidate_ledger") or []
                if item.get("fetch_status") == "unfetched"
            ]
            matched = [item for item in candidates if item.get("may_address")]
            chosen = (matched or candidates or [None])[0]
            url = str((chosen or {}).get("url") or "")
            if name and url:
                return {
                    "family": preferred,
                    "tool_name": name,
                    "tool_input": {"url": url},
                }

        if preferred == "Web Search":
            name = choose(
                lambda value: value == "web_search" or value.startswith("search_web_")
            )
            missing = state.get("missing_requirements") or []
            query = str((missing[0] if missing else {}).get("description") or self.query)
            if name and query:
                return {
                    "family": preferred,
                    "tool_name": name,
                    "tool_input": {"query": query},
                }

        return None

    def record_legal_stop_accepted(self) -> None:
        self._legal_stop_accepted += 1

    def record_illegal_stop_rejected(self) -> None:
        self._illegal_stop_rejected += 1

    def record_illegal_stop_recovered(self) -> None:
        self._illegal_stop_recovered += 1

    def sync_results(self, results: list[Mapping[str, Any]] | Any) -> None:
        if not isinstance(results, list):
            return
        self._results = [dict(result) for result in results if isinstance(result, Mapping)]
        for result in self._results:
            if _result_family(result) != "fetch":
                continue
            url = _result_url(result)
            if not url:
                continue
            content = str(result.get("full_content") or result.get("snippet") or "")
            if content.strip():
                self._fetch_attempts[url] = {
                    "url": url,
                    "status": "succeeded",
                    "reason": None,
                    "observation_excerpt": content[:180],
                }

    def _satisfied_for(self, requirement: _Requirement) -> list[dict[str, Any]]:
        matches: list[dict[str, Any]] = []
        for result in self._results:
            family = _result_family(result)
            # A fetched page is valid web evidence as well as fetch evidence.
            family_matches = family == requirement.family or (
                requirement.family == "web" and family == "fetch"
            )
            if not family_matches:
                continue
            if requirement.family == "fetch" and not str(
                result.get("full_content") or result.get("snippet") or ""
            ).strip():
                continue
            if requirement.match_terms:
                haystack = " ".join(
                    (
                        _result_url(result),
                        str(result.get("title") or ""),
                        str(result.get("snippet") or ""),
                        str(result.get("full_content") or "")[:1000],
                    )
                ).lower()
                if not any(term in haystack for term in requirement.match_terms):
                    continue
            matches.append(result)
        return matches

    def _candidate_ledger(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        requirement_terms = _keywords(" ".join(r.description for r in self._requirements))
        query_terms = _keywords(self.query)
        for result in self._results:
            if _result_family(result) != "web":
                continue
            url = _result_url(result)
            if not url or url in seen:
                continue
            seen.add(url)
            text = f"{result.get('title', '')} {result.get('snippet', '')}"
            overlap = len(_keywords(text) & (requirement_terms | query_terms))
            candidate_haystack = f"{url} {text}".lower()
            attempt = self._fetch_attempts.get(url)
            try:
                domain = (urlparse(url).hostname or "").lower()
            except ValueError:
                domain = ""
            rows.append(
                {
                    "url": url,
                    "title": str(result.get("title") or "")[:180],
                    "snippet": str(result.get("snippet") or "")[:260],
                    "domain": domain,
                    "source_type": "web_search_candidate",
                    "fetch_status": (attempt or {}).get("status", "unfetched"),
                    "fetch_failure_reason": (attempt or {}).get("reason"),
                    "relevance_hint": (
                        "lexical_match" if overlap else "needs_planner_review"
                    ),
                    "may_address": [
                        requirement.identifier
                        for requirement in self._requirements
                        if requirement.family in {"web", "fetch"}
                        and (
                            not requirement.match_terms
                            or any(
                                term in candidate_haystack
                                for term in requirement.match_terms
                            )
                        )
                    ],
                }
            )
        return rows[:12]

    def snapshot(self) -> dict[str, Any]:
        satisfied: list[dict[str, Any]] = []
        missing: list[dict[str, Any]] = []
        for requirement in self._requirements:
            matches = self._satisfied_for(requirement)
            entry = {
                "id": requirement.identifier,
                "family": requirement.family,
                "description": requirement.description,
                "required_count": requirement.min_count,
                "observed_count": len(matches),
                "match_terms": list(requirement.match_terms),
            }
            if len(matches) >= requirement.min_count:
                satisfied.append(entry)
            else:
                missing.append(entry)
        candidate_ledger = self._candidate_ledger()
        failed_fetches = [
            attempt
            for attempt in self._fetch_attempts.values()
            if attempt.get("status") == "failed"
        ]
        untried = [
            row["url"]
            for row in candidate_ledger
            if row.get("fetch_status") == "unfetched"
        ]
        evidence_sufficient = bool(self._requirements) and not missing
        stop_blockers = [
            f"{item['id']}: {item['observed_count']}/{item['required_count']} "
            f"{item['family']} evidence"
            for item in missing
        ]
        missing_families = {str(item["family"]) for item in missing}
        recovery_exhausted = bool(candidate_ledger) and all(
            row.get("fetch_status") == "failed" for row in candidate_ledger
        )
        if not missing:
            legal_action_families = ["STOP_SUCCESS"]
            preferred_action_families = ["STOP_SUCCESS"]
            action_guidance = {
                "preferred_action_family": "STOP_SUCCESS",
                "instruction": "The visible evidence contract is complete; STOP_SUCCESS is legal.",
            }
        elif "collection" in missing_families:
            legal_action_families = ["Collection Search"]
            preferred_action_families = ["Collection Search"]
            action_guidance = {
                "preferred_action_family": "Collection Search",
                "instruction": (
                    "Reformulate the Collection query using the task terms; do not "
                    "repeat the exact prior Collection query."
                ),
            }
        elif ({"fetch", "web"} & missing_families) and untried:
            legal_action_families = ["Fetch observed candidate"]
            preferred_action_families = ["Fetch observed candidate"]
            action_guidance = {
                "preferred_action_family": "Fetch observed candidate",
                "instruction": (
                    "Fetch one untried observed candidate that may address a visible "
                    "missing requirement."
                ),
            }
        elif ({"fetch", "web"} & missing_families) and recovery_exhausted:
            legal_action_families = ["STOP_INCOMPLETE"]
            preferred_action_families = ["STOP_INCOMPLETE"]
            action_guidance = {
                "preferred_action_family": "STOP_INCOMPLETE",
                "instruction": (
                    "All observed candidates have failed. Stop with an explicit "
                    "evidence gap rather than claiming the contract is complete."
                ),
            }
        else:
            legal_action_families = ["Web Search"]
            preferred_action_families = ["Web Search"]
            action_guidance = {
                "preferred_action_family": "Web Search",
                "instruction": (
                    "Search for an observed public candidate that can address the "
                    "visible missing fetched evidence."
                ),
            }
        return {
            "protocol": PLANNER_UNCERTAINTY_STATE_PROTOCOL,
            "missing_requirements": missing,
            "satisfied_evidence": satisfied,
            "evidence_contract_status": {
                "complete": evidence_sufficient,
                "required": len(self._requirements),
                "satisfied": len(satisfied),
                "missing": len(missing),
            },
            "candidate_ledger": candidate_ledger,
            "failed_fetches": failed_fetches,
            "fetch_failure_reason": (
                failed_fetches[-1].get("reason") if failed_fetches else None
            ),
            "untried_candidates": untried,
            "collection_search_queries": list(self._collection_queries[-4:]),
            "web_search_queries": list(self._web_queries[-4:]),
            "legal_action_families": legal_action_families,
            "preferred_action_families": preferred_action_families,
            "action_guidance": action_guidance,
            "recovery_exhausted": recovery_exhausted,
            "stop_legal": evidence_sufficient,
            "stop_blockers": stop_blockers,
            "evidence_sufficient": evidence_sufficient,
            "unnecessary_exploration": self._tool_calls_after_sufficient,
            "post_contract_extra_calls": self._tool_calls_after_sufficient,
            "stop_attempts": self._stop_attempts,
            "legal_stop_accepted": self._legal_stop_accepted,
            "illegal_stop_rejected": self._illegal_stop_rejected,
            "illegal_stop_recovered": self._illegal_stop_recovered,
            "unobserved_url_attempts": self._unobserved_url_attempts,
        }

    def render_prompt(self) -> str:
        """Render a bounded state block for the Planner's dynamic prompt."""
        state = self.snapshot()
        lines = [
            "PLANNER UNCERTAINTY STATE (observation-derived; not a hidden answer):",
            f"- evidence contract: {state['evidence_contract_status']}",
            f"- missing requirements: {state['missing_requirements']}",
            f"- satisfied evidence: {state['satisfied_evidence']}",
            f"- failed fetches: {state['failed_fetches']}",
            f"- untried candidates: {state['untried_candidates']}",
            f"- stop legal: {state['stop_legal']}",
            f"- stop blockers: {state['stop_blockers']}",
            f"- legal action families: {state['legal_action_families']}",
            f"- preferred action families: {state['preferred_action_families']}",
            f"- action guidance: {state['action_guidance']}",
            f"- previous Collection queries: {state['collection_search_queries']}",
            "Use the ledger to choose an observed URL, recover after a failed "
            "fetch, and avoid treating a search snippet as fetched page evidence.",
        ]
        for index, candidate in enumerate(state["candidate_ledger"], start=1):
            lines.append(
                "- candidate "
                f"{index}: {candidate['url']} | {candidate['title']} | "
                f"status={candidate['fetch_status']} | "
                f"hint={candidate['relevance_hint']}"
            )
        return "\n".join(lines)

    def trace_summary(self) -> dict[str, Any]:
        return self.snapshot()


__all__ = ["PLANNER_UNCERTAINTY_STATE_PROTOCOL", "UncertaintyStateTracker"]
