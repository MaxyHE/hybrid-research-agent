"""Typed source-connector boundary for the General V1 controller.

Connectors own network or local-collection access.  The controller only sees
their discovered candidates through :class:`SearchCandidate.model_view`; it
never receives a callable URL tool.  A connector's returned text becomes
evidence-capable only after this module persists it through the content
artifact store and derives a ``SourceRecord`` from the configured channel.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Iterable, Protocol, runtime_checkable

from .actions import SearchCandidate
from .artifact_store import ContentArtifactStore
from .config import SourceConnectorConfig
from .schemas import SourceChannel, SourceClass, SourceClassificationBasis
from .source_policy import (
    register_full_fetch,
    register_local_collection_fetch,
)
from .workflow import FetchedPage


class ConnectorRuntimeError(RuntimeError):
    """A connector failed without exposing provider internals to the controller."""


def _text(value: object, *, field_name: str, maximum: int = 8_000) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    normalized = value.strip()
    if len(normalized) > maximum:
        raise ValueError(f"{field_name} exceeds {maximum} characters")
    return normalized


@dataclass(frozen=True, slots=True)
class DiscoveredResource:
    """One connector-owned result before a controller creates a candidate ID."""

    resource_locator: str
    title: str
    snippet: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "resource_locator",
            _text(self.resource_locator, field_name="resource_locator", maximum=4_096),
        )
        object.__setattr__(self, "title", _text(self.title, field_name="title", maximum=1_000))
        object.__setattr__(
            self,
            "snippet",
            _text(self.snippet, field_name="snippet", maximum=1_500),
        )


@dataclass(frozen=True, slots=True)
class FetchedResource:
    """A connector-owned full-text result for an already observed resource."""

    content: str
    retrieved_at: str
    title: str | None = None
    source_class: SourceClass = SourceClass.UNKNOWN
    source_classification_basis: SourceClassificationBasis = (
        SourceClassificationBasis.UNVERIFIED
    )
    publisher_id: str | None = None
    source_group: str | None = None
    published_on: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "content", _text(self.content, field_name="content", maximum=20_000_000))
        object.__setattr__(
            self, "retrieved_at", _text(self.retrieved_at, field_name="retrieved_at", maximum=64)
        )
        if self.title is not None:
            object.__setattr__(self, "title", _text(self.title, field_name="title", maximum=1_000))
        object.__setattr__(self, "source_class", SourceClass(self.source_class))
        object.__setattr__(
            self,
            "source_classification_basis",
            SourceClassificationBasis(self.source_classification_basis),
        )
        if (
            self.source_class != SourceClass.UNKNOWN
            and self.source_classification_basis
            == SourceClassificationBasis.UNVERIFIED
        ):
            raise ValueError(
                "a non-unknown source_class requires a verified classification basis"
            )


@runtime_checkable
class SourceConnector(Protocol):
    """One bounded search/fetch provider; never a model-visible tool object."""

    def search(self, query: str) -> Iterable[DiscoveredResource]:
        """Return discovered resources for one legal search action."""

    def fetch(self, resource: DiscoveredResource) -> FetchedResource:
        """Return full text for an already observed resource only."""


@dataclass(frozen=True, slots=True)
class ConnectorBinding:
    """Bind one implementation to one immutable run connector declaration."""

    config: SourceConnectorConfig
    connector: SourceConnector

    def __post_init__(self) -> None:
        if not isinstance(self.config, SourceConnectorConfig):
            raise TypeError("config must be SourceConnectorConfig")
        if not isinstance(self.connector, SourceConnector):
            raise TypeError("connector must implement search() and fetch()")


class ConnectorRegistry:
    """Runtime-owned mapping from declared connector IDs to implementations."""

    def __init__(self, bindings: Iterable[ConnectorBinding]) -> None:
        indexed: dict[str, ConnectorBinding] = {}
        for binding in bindings:
            if not isinstance(binding, ConnectorBinding):
                raise TypeError("bindings must contain ConnectorBinding values")
            connector_id = binding.config.connector_id
            if connector_id in indexed:
                raise ValueError("connector bindings must use unique connector_id values")
            indexed[connector_id] = binding
        if not indexed:
            raise ValueError("ConnectorRegistry requires at least one binding")
        self._bindings = indexed

    def get(self, connector_id: str) -> ConnectorBinding:
        binding = self._bindings.get(connector_id)
        if binding is None:
            raise ConnectorRuntimeError("connector_not_bound")
        return binding

    @property
    def connector_ids(self) -> frozenset[str]:
        return frozenset(self._bindings)


def make_candidate(
    *,
    candidate_id: str,
    binding: ConnectorBinding,
    resource: DiscoveredResource,
    observed_at: str,
) -> SearchCandidate:
    """Create the only model-selectable handle for a discovered resource."""

    return SearchCandidate(
        candidate_id=candidate_id,
        connector_id=binding.config.connector_id,
        source_channel=binding.config.source_channel,
        resource_locator=resource.resource_locator,
        title=resource.title,
        snippet=resource.snippet,
        observed_at=observed_at,
    )


def source_id_for_candidate(candidate: SearchCandidate) -> str:
    """Derive an opaque, stable source ID without placing the locator in model state."""

    material = f"{candidate.connector_id}\n{candidate.resource_locator}"
    return "source-" + sha256(material.encode("utf-8")).hexdigest()[:24]


def materialize_fetched_page(
    *,
    artifact_store: ContentArtifactStore,
    candidate: SearchCandidate,
    fetched: FetchedResource,
) -> FetchedPage:
    """Persist full text and construct a verified, channel-correct page artifact."""

    if not isinstance(artifact_store, ContentArtifactStore):
        raise TypeError("artifact_store must be ContentArtifactStore")
    if not isinstance(candidate, SearchCandidate):
        raise TypeError("candidate must be SearchCandidate")
    if not isinstance(fetched, FetchedResource):
        raise TypeError("fetched must be FetchedResource")
    common = {
        "artifact_store": artifact_store,
        "source_id": source_id_for_candidate(candidate),
        "title": fetched.title or candidate.title,
        "content": fetched.content,
        "retrieved_at": fetched.retrieved_at,
        "source_connector_id": candidate.connector_id,
        "source_class": fetched.source_class,
        "source_classification_basis": fetched.source_classification_basis,
        "publisher_id": fetched.publisher_id,
        "source_group": fetched.source_group,
        "published_on": fetched.published_on,
    }
    if candidate.source_channel == SourceChannel.PUBLIC_WEB:
        source = register_full_fetch(url=candidate.resource_locator, **common)
    elif candidate.source_channel == SourceChannel.LOCAL_COLLECTION:
        source = register_local_collection_fetch(
            resource_locator=candidate.resource_locator, **common
        )
    else:  # defensive for future SourceChannel values
        raise ConnectorRuntimeError("unsupported_source_channel")
    return FetchedPage(source=source, content=fetched.content)


__all__ = [
    "ConnectorBinding",
    "ConnectorRegistry",
    "ConnectorRuntimeError",
    "DiscoveredResource",
    "FetchedResource",
    "SourceConnector",
    "make_candidate",
    "materialize_fetched_page",
    "source_id_for_candidate",
]
