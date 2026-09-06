"""Small source boundary for the Hybrid-ODR runtime.

This module deliberately models only what the research loop needs: discover a
source, fetch its text, and retain which channel supplied it.  It does not
carry General V1's source classification, evidence cards, coverage state, or
publication policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class DiscoveredResource:
    resource_locator: str
    title: str
    snippet: str
    channel: str


@dataclass(frozen=True, slots=True)
class FetchedResource:
    content: str
    retrieved_at: str
    title: str | None = None
    citation_aliases: tuple[str, ...] = ()
    retrieval_method: str | None = None


@runtime_checkable
class SourceConnector(Protocol):
    """One source channel exposed to a focused researcher."""

    def search(self, query: str) -> Iterable[DiscoveredResource]:
        """Return discovery results for one query."""

    def fetch(self, resource: DiscoveredResource) -> FetchedResource:
        """Return the full text of a discovered result."""


__all__ = ["DiscoveredResource", "FetchedResource", "SourceConnector"]
