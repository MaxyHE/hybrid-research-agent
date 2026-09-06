"""Authorized local-collection connector for the General V1 protocol.

The existing Collection RAG engine is reused only for discovery.  It cannot
itself become evidence because its output is an indexed snippet.  This adapter
therefore resolves every selected document again through the user's encrypted
document store, verifies current collection membership, and lets the common
connector layer snapshot the full text before evidence extraction.

This module intentionally does not decide whether a collection's data may be
shown to a particular model.  The product boundary must combine
``CollectionDescriptor.is_public`` with the frozen egress/model policy before
it exposes this connector to the controller.  ``requires_user_authorization``
in the execution config remains a second, independent capability gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Protocol, runtime_checkable
from urllib.parse import urlparse
from uuid import UUID

from .connectors import DiscoveredResource, FetchedResource
from .schemas import SourceClass, SourceClassificationBasis


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _uuid(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a UUID string")
    try:
        return str(UUID(value.strip()))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a UUID string") from exc


def _text(value: object, *, field_name: str, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    normalized = value.strip()
    if len(normalized) > maximum:
        raise ValueError(f"{field_name} exceeds {maximum} characters")
    return normalized


@dataclass(frozen=True, slots=True)
class CollectionDescriptor:
    """A user-authorized collection resolved by the application, not the model."""

    collection_id: str
    collection_name: str
    is_public: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "collection_id", _uuid(self.collection_id, field_name="collection_id")
        )
        object.__setattr__(
            self,
            "collection_name",
            _text(self.collection_name, field_name="collection_name", maximum=255),
        )
        if not isinstance(self.is_public, bool):
            raise TypeError("is_public must be a boolean")

    @property
    def source_group(self) -> str:
        """One collection is one provenance group; it is not pseudo-diversity."""

        return f"collection:{self.collection_id}"

    def resource_locator(self, document_id: str) -> str:
        return f"local://{self.collection_id}/{_uuid(document_id, field_name='document_id')}"


@dataclass(frozen=True, slots=True)
class CollectionSearchHit:
    """A discovery-only local search result; it contains no full document text."""

    document_id: str
    title: str
    snippet: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "document_id", _uuid(self.document_id, field_name="document_id"))
        object.__setattr__(self, "title", _text(self.title, field_name="title", maximum=1_000))
        object.__setattr__(self, "snippet", _text(self.snippet, field_name="snippet", maximum=8_000))


@dataclass(frozen=True, slots=True)
class CollectionDocument:
    """A full document returned only after membership is re-checked."""

    document_id: str
    title: str
    content: str
    published_on: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "document_id", _uuid(self.document_id, field_name="document_id"))
        object.__setattr__(self, "title", _text(self.title, field_name="title", maximum=1_000))
        object.__setattr__(self, "content", _text(self.content, field_name="content", maximum=20_000_000))
        if self.published_on is not None:
            object.__setattr__(
                self,
                "published_on",
                _text(self.published_on, field_name="published_on", maximum=32),
            )


@runtime_checkable
class LocalCollectionBackend(Protocol):
    """Private backend interface; it receives only application-owned IDs."""

    def search(
        self, descriptor: CollectionDescriptor, query: str
    ) -> Iterable[CollectionSearchHit]:
        """Return verified members of exactly ``descriptor.collection_id``."""

    def fetch(
        self, descriptor: CollectionDescriptor, document_id: str
    ) -> CollectionDocument:
        """Return the full text only if it remains a collection member."""


def _locator_document_id(locator: str, descriptor: CollectionDescriptor) -> str:
    try:
        parsed = urlparse(locator)
    except ValueError as exc:
        raise ValueError("local collection locator is malformed") from exc
    document_id = parsed.path.strip("/")
    if (
        parsed.scheme != "local"
        or parsed.netloc.casefold() != descriptor.collection_id
        or not document_id
        or parsed.params
        or parsed.query
        or parsed.fragment
        or "/" in document_id
    ):
        raise ValueError("candidate does not belong to this authorized collection")
    return _uuid(document_id, field_name="document_id")


class AuthorizedLocalCollectionConnector:
    """Expose one explicit collection through the generic Search/Fetch contract."""

    def __init__(
        self,
        *,
        descriptor: CollectionDescriptor,
        backend: LocalCollectionBackend,
        clock: Callable[[], str] = _timestamp,
    ) -> None:
        if not isinstance(descriptor, CollectionDescriptor):
            raise TypeError("descriptor must be CollectionDescriptor")
        if not isinstance(backend, LocalCollectionBackend):
            raise TypeError("backend must implement LocalCollectionBackend")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self.descriptor = descriptor
        self._backend = backend
        self._clock = clock

    def search(self, query: str) -> Iterable[DiscoveredResource]:
        hits = tuple(self._backend.search(self.descriptor, query))
        if not all(isinstance(hit, CollectionSearchHit) for hit in hits):
            raise TypeError("local collection backend must return CollectionSearchHit values")
        seen_document_ids: set[str] = set()
        discovered: list[DiscoveredResource] = []
        for hit in hits:
            if hit.document_id in seen_document_ids:
                continue
            seen_document_ids.add(hit.document_id)
            discovered.append(
                DiscoveredResource(
                    resource_locator=self.descriptor.resource_locator(hit.document_id),
                    title=hit.title,
                    snippet=hit.snippet,
                )
            )
        return tuple(discovered)

    def fetch(self, resource: DiscoveredResource) -> FetchedResource:
        if not isinstance(resource, DiscoveredResource):
            raise TypeError("resource must be DiscoveredResource")
        document_id = _locator_document_id(resource.resource_locator, self.descriptor)
        document = self._backend.fetch(self.descriptor, document_id)
        if not isinstance(document, CollectionDocument):
            raise TypeError("local collection backend must return CollectionDocument")
        if document.document_id != document_id:
            raise ValueError("local collection backend returned a different document")
        return FetchedResource(
            content=document.content,
            retrieved_at=self._clock(),
            title=document.title,
            # A collection membership label does not establish the document's
            # publisher or authority.  Preserve unknown rather than inventing
            # a source class from a local path or a collection name.
            source_class=SourceClass.UNKNOWN,
            source_classification_basis=SourceClassificationBasis.UNVERIFIED,
            source_group=self.descriptor.source_group,
            published_on=document.published_on,
        )


class ProjectLocalCollectionBackend:
    """Adapter over the existing encrypted document DB and Collection RAG index."""

    def __init__(
        self,
        *,
        username: str,
        settings_snapshot: dict[str, Any],
        max_results: int = 8,
    ) -> None:
        self.username = _text(username, field_name="username", maximum=255)
        if not isinstance(settings_snapshot, dict):
            raise TypeError("settings_snapshot must be a dict")
        if isinstance(max_results, bool) or not isinstance(max_results, int) or not 1 <= max_results <= 50:
            raise ValueError("max_results must be between 1 and 50")
        self._settings_snapshot = dict(settings_snapshot)
        self._max_results = max_results

    def _document(self, descriptor: CollectionDescriptor, document_id: str):
        from local_deep_research.database.models.library import Document, DocumentCollection
        from local_deep_research.database.session_context import get_user_db_session

        with get_user_db_session(self.username) as session:
            return (
                session.query(Document)
                .join(
                    DocumentCollection,
                    DocumentCollection.document_id == Document.id,
                )
                .filter(
                    DocumentCollection.collection_id == descriptor.collection_id,
                    Document.id == document_id,
                )
                .first()
            )

    @staticmethod
    def _document_id_from_result(result: dict[str, Any]) -> str | None:
        metadata = result.get("metadata")
        if isinstance(metadata, dict):
            for key in ("document_id", "source_id"):
                value = metadata.get(key)
                if isinstance(value, str):
                    try:
                        return _uuid(value, field_name="document_id")
                    except ValueError:
                        pass
        url = result.get("url") or result.get("link")
        if isinstance(url, str):
            prefix = "/library/document/"
            if url.startswith(prefix):
                candidate = url[len(prefix) :].split("/", 1)[0]
                try:
                    return _uuid(candidate, field_name="document_id")
                except ValueError:
                    pass
        return None

    @staticmethod
    def _title(document: Any, fallback: str) -> str:
        for value in (getattr(document, "title", None), getattr(document, "filename", None), fallback):
            if isinstance(value, str) and value.strip():
                return value.strip()
        return "Collection document"

    def search(
        self, descriptor: CollectionDescriptor, query: str
    ) -> Iterable[CollectionSearchHit]:
        from local_deep_research.web_search_engines.engines.search_engine_collection import (
            CollectionSearchEngine,
        )

        snapshot = {**self._settings_snapshot, "_username": self.username}
        engine = CollectionSearchEngine(
            collection_id=descriptor.collection_id,
            collection_name=descriptor.collection_name,
            llm=None,
            max_results=self._max_results,
            settings_snapshot=snapshot,
        )
        try:
            raw_results = engine.search(query, limit=self._max_results)
        finally:
            close = getattr(engine, "close", None)
            if callable(close):
                close()
        if not isinstance(raw_results, list):
            raise TypeError("CollectionSearchEngine.search must return a list")
        hits: list[CollectionSearchHit] = []
        for raw in raw_results:
            if not isinstance(raw, dict):
                continue
            document_id = self._document_id_from_result(raw)
            if document_id is None:
                continue
            # Re-check membership before the result's snippet reaches the
            # controller/model. This makes a stale or faulty vector index a
            # safe omission, never a cross-collection disclosure.
            document = self._document(descriptor, document_id)
            if document is None:
                continue
            try:
                hits.append(
                    CollectionSearchHit(
                        document_id=document_id,
                        title=self._title(document, str(raw.get("title") or "")),
                        snippet=_text(
                            raw.get("snippet") or raw.get("content") or "",
                            field_name="collection search snippet",
                            maximum=8_000,
                        ),
                    )
                )
            except ValueError:
                continue
        return tuple(hits)

    def fetch(
        self, descriptor: CollectionDescriptor, document_id: str
    ) -> CollectionDocument:
        document = self._document(descriptor, document_id)
        if document is None:
            raise PermissionError("collection document is absent or no longer authorized")
        content = getattr(document, "text_content", None)
        if not isinstance(content, str) or not content.strip():
            raise ValueError("collection document has no extracted text")
        published = getattr(document, "published_date", None)
        return CollectionDocument(
            document_id=document_id,
            title=self._title(document, "Collection document"),
            content=content,
            published_on=published.isoformat() if published is not None else None,
        )


def resolve_project_collection_descriptor(
    *, collection_id: str, username: str
) -> CollectionDescriptor:
    """Resolve one user-owned, agent-enabled collection before model exposure."""

    normalized_id = _uuid(collection_id, field_name="collection_id")
    normalized_username = _text(username, field_name="username", maximum=255)
    from local_deep_research.database.models.library import Collection
    from local_deep_research.database.session_context import get_user_db_session

    with get_user_db_session(normalized_username) as session:
        collection = session.query(Collection).filter_by(id=normalized_id).first()
        if collection is None:
            raise PermissionError("collection is absent or not accessible to this user")
        if collection.agent_enabled is not True:
            raise PermissionError("collection is not authorized for agent research")
        return CollectionDescriptor(
            collection_id=collection.id,
            collection_name=collection.name,
            is_public=collection.is_public is True,
        )


def project_local_collection_connector(
    *,
    descriptor: CollectionDescriptor,
    username: str,
    settings_snapshot: dict[str, Any],
    max_results: int = 8,
) -> AuthorizedLocalCollectionConnector:
    """Build the concrete connector after the application resolved authorization."""

    return AuthorizedLocalCollectionConnector(
        descriptor=descriptor,
        backend=ProjectLocalCollectionBackend(
            username=username,
            settings_snapshot=settings_snapshot,
            max_results=max_results,
        ),
    )


__all__ = [
    "AuthorizedLocalCollectionConnector",
    "CollectionDescriptor",
    "CollectionDocument",
    "CollectionSearchHit",
    "LocalCollectionBackend",
    "ProjectLocalCollectionBackend",
    "project_local_collection_connector",
    "resolve_project_collection_descriptor",
]
