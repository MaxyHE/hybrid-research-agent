"""Evidence-only citation synthesis for planner/writer separation."""

from __future__ import annotations

import re
from typing import Any

from langchain_core.documents import Document

from local_deep_research.citation_handlers.standard_citation_handler import (
    StandardCitationHandler,
)


def evidence_gap_report(
    search_results: list[dict[str, Any]], *, reason: str
) -> str:
    """Return a non-empty, provenance-only report when synthesis is unavailable.

    This is deliberately deterministic.  A writer outage or an empty writer
    response must not turn collected evidence into a blank user-facing answer,
    and it must not be replaced by an ungrounded planner draft.
    """
    lines = [
        "Research status: a complete evidence-grounded report could not be "
        f"generated ({reason}).",
    ]
    if not search_results:
        lines.append("No usable evidence was collected, so no factual claim is made.")
        return "\n\n".join(lines)

    lines.append("Verified evidence collected in this run:")
    for fallback_index, result in enumerate(search_results, start=1):
        if not isinstance(result, dict):
            continue
        index = result.get("index", fallback_index)
        title = str(result.get("title") or "Untitled source").strip()
        source = str(result.get("link") or result.get("url") or "").strip()
        route = EvidenceOnlyCitationHandler._evidence_route(result)
        suffix = f" — {source}" if source else ""
        lines.append(f"- [{index}] {route}: {title}{suffix}")
    lines.append(
        "No additional claim is made because the requested synthesis or "
        "evidence contract is incomplete."
    )
    return "\n".join(lines)


class EvidenceOnlyCitationHandler(StandardCitationHandler):
    """Write the final report from tool evidence, never planner prose."""

    @staticmethod
    def _evidence_route(result: dict[str, Any]) -> str:
        """Return the user-visible route that produced an evidence item."""
        engine = str(result.get("source_engine") or "").lower()
        source = str(result.get("link") or result.get("url") or "")
        if engine.startswith("collection_") or source.startswith(
            "/library/document/"
        ):
            return "Collection document"
        if engine == "fetch":
            return "Fetched Web page (content verified)"
        return "Web search result (snippet only; page not fetched)"

    def _create_documents(
        self,
        search_results: str | list[dict[str, Any]],
        nr_of_links: int = 0,
    ) -> list[Document]:
        """Preserve route provenance that the base citation handler drops."""
        documents = super()._create_documents(search_results, nr_of_links)
        if isinstance(search_results, str):
            return documents

        results_by_index = {
            int(result["index"]): result
            for result in search_results
            if isinstance(result, dict) and "index" in result
        }
        for document in documents:
            result = results_by_index.get(int(document.metadata["index"]), {})
            document.metadata["evidence_route"] = self._evidence_route(result)
        return documents

    @staticmethod
    def _format_sources(documents: list[Document]) -> str:
        """Expose citation identity and route provenance to the writer."""
        sources = []
        for document in documents:
            metadata = document.metadata
            sources.append(
                "\n".join(
                    (
                        f"[{metadata['index']}]",
                        f"Evidence route: {metadata['evidence_route']}",
                        f"Source title: {metadata['title']}",
                        f"Source URL: {metadata['source']}",
                        f"Evidence text: {document.page_content}",
                    )
                )
            )
        return "\n\n".join(sources)

    def analyze_followup(
        self,
        question: str,
        search_results: str | list[dict[str, Any]],
        previous_knowledge: str,
        nr_of_links: int,
    ) -> dict[str, Any]:
        del previous_knowledge
        documents = self._create_documents(
            search_results, nr_of_links=nr_of_links
        )
        if not documents:
            return self._no_sources_response(question)
        formatted_sources = self._format_sources(documents)
        length_match = re.search(
            r"(?:限\s*)?(\d+)\s*字(?:以内)?", question
        )
        length_requirement = ""
        if length_match is not None:
            hard_limit = int(length_match.group(1))
            target_limit = max(1, int(hard_limit * 0.9))
            length_requirement = (
                "- Hard length limit: the complete answer must contain no "
                f"more than {hard_limit} Unicode characters. Aim for at "
                f"most {target_limit} characters so citations fit.\n"
            )
        prompt = f"""Write the final answer to the question using ONLY the evidence below.

Question:
{question}

Evidence:
{formatted_sources}

Requirements:
- Treat every claim from an earlier planner as untrusted; only the evidence above is authoritative.
- Treat each Evidence route and Source URL as authoritative provenance. Call only a `Collection document` a Collection source; `Web search result` and `Fetched Web page` are Web evidence. Never swap these roles.
- A `Web search result (snippet only; page not fetched)` is not a read or verified webpage, even when its URL is from an official domain. Never call it a current official webpage, say that you read it, or use it to satisfy a request for a verified official-page claim.
- If the question requires a current official-page fact but no `Fetched Web page (content verified)` supports it, state that a verifiable official page was not obtained and omit that fact rather than relying on a search snippet.
- If the evidence is irrelevant or insufficient for a requested point, say so explicitly instead of answering from memory.
- Cite factual claims with the provided source numbers in square brackets, such as [1]. Never invent a source or URL.
- Distinguish internal project evidence from external public evidence when the question asks for that distinction.
- Obey every requested format and length limit. Prefer concise decision-oriented prose and do not append a bibliography.
{length_requirement}- Count the answer body and citations together when enforcing a length limit.
- Do not repeat raw source snippets or expose tool-observation formatting.
"""
        response = self._invoke_with_streaming(prompt)
        if not isinstance(response, str) or not response.strip():
            return {
                "content": evidence_gap_report(
                    search_results
                    if isinstance(search_results, list)
                    else [],
                    reason="writer returned an empty response",
                ),
                "documents": documents,
                "synthesis_status": "writer_empty_fallback",
            }
        return {
            "content": response,
            "documents": documents,
            "synthesis_status": "complete",
        }
