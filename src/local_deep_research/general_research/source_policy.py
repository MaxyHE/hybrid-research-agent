"""Deterministic URL and source-quality policy for General Research v1.

This module deliberately does *not* infer authority from a domain suffix.
``.gov``, ``.edu``, country TLDs, and a registrable-domain heuristic are all
too weak to establish who published a page or whether it is fit to support a
claim.  ``SourceClass`` is explicit provenance supplied by a resolver or
reviewer, and quality decisions retain the signals that produced the score.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

from .schemas import (
    SourceClass,
    SourceClassificationBasis,
    SourceChannel,
    SourceRecord,
)


SOURCE_POLICY_VERSION = "source-policy/v1"
_DEFAULT_PORTS = {"http": 80, "https": 443}
_TRACKING_QUERY_PARAMETERS = frozenset({"fbclid", "gclid", "mc_cid", "mc_eid"})
_BASE_QUALITY_SCORES = {
    SourceClass.PRIMARY: 95,
    SourceClass.GOVERNMENT: 90,
    SourceClass.REGULATOR: 90,
    SourceClass.STANDARDS_BODY: 88,
    SourceClass.ACADEMIC: 85,
    SourceClass.OFFICIAL: 80,
    SourceClass.INDUSTRY: 65,
    SourceClass.NEWS: 60,
    SourceClass.SECONDARY: 50,
    SourceClass.COMMUNITY: 30,
    SourceClass.SEARCH_SNIPPET: 10,
    SourceClass.UNKNOWN: 35,
}


def _required_text(value: str, field_name: str) -> str:
    cleaned = str(value or "").strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be non-empty")
    return cleaned


def _normalize_host(host: str) -> str:
    try:
        normalized = host.strip().rstrip(".").encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise ValueError("URL hostname cannot be IDNA-normalized") from exc
    normalized = normalized.casefold()
    if not normalized:
        raise ValueError("URL must include a hostname")
    return normalized


def _normalized_path(path: str) -> str:
    # Percent encoding is normalized without resolving dot segments: URL path
    # normalization can change a server's routing semantics, so it is not a
    # safe identity transform for evidence provenance.
    return quote(path or "/", safe="/%:@!$&'()*+,;=-._~")


def _normalized_query(query: str) -> str:
    parameters = parse_qsl(query, keep_blank_values=True)
    retained = [
        (name, value)
        for name, value in parameters
        if not name.casefold().startswith("utm_")
        and name.casefold() not in _TRACKING_QUERY_PARAMETERS
    ]
    return urlencode(sorted(retained), doseq=True, quote_via=quote, safe="~")


@dataclass(frozen=True)
class CanonicalSource:
    """Canonical URL identity for deduplication, never an authority verdict."""

    schema_version: str = "general-canonical-source/v1"
    original_url: str = ""
    canonical_url: str = ""
    scheme: str = ""
    host: str = ""

    def __post_init__(self) -> None:
        if not self.original_url or not self.canonical_url or not self.host:
            raise ValueError("CanonicalSource fields must be non-empty")
        if self.scheme not in {"http", "https"}:
            raise ValueError("CanonicalSource supports only http and https")

    def to_dict(self) -> dict[str, str]:
        return {
            "schema_version": self.schema_version,
            "original_url": self.original_url,
            "canonical_url": self.canonical_url,
            "scheme": self.scheme,
            "host": self.host,
        }


def canonicalize_url(url: str, *, allowed_schemes: Iterable[str] = ("https", "http")) -> CanonicalSource:
    """Return a stable, privacy-safe URL identity without network access.

    Fragments and known marketing tracking parameters do not identify page
    content, so they are removed.  Authentication-bearing URLs are rejected
    rather than silently preserving credentials in a trace artifact.
    """

    original_url = _required_text(url, "url")
    allowed = {str(value).casefold().rstrip(":") for value in allowed_schemes}
    try:
        parsed = urlsplit(original_url)
    except ValueError as exc:
        raise ValueError("URL could not be parsed") from exc
    scheme = parsed.scheme.casefold()
    if scheme not in allowed or scheme not in {"http", "https"}:
        raise ValueError("URL scheme is not allowed")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("URLs containing credentials are not accepted")
    host = _normalize_host(parsed.hostname or "")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("URL port is invalid") from exc
    netloc = host if port in (None, _DEFAULT_PORTS[scheme]) else f"{host}:{port}"
    canonical_url = urlunsplit(
        (
            scheme,
            netloc,
            _normalized_path(parsed.path),
            _normalized_query(parsed.query),
            "",
        )
    )
    return CanonicalSource(
        original_url=original_url,
        canonical_url=canonical_url,
        scheme=scheme,
        host=host,
    )


@dataclass(frozen=True)
class SourceQuality:
    """An explainable source-quality score; not a factuality or entailment score."""

    schema_version: str = "general-source-quality/v1"
    source_id: str = ""
    score: int = 0
    signals: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.source_id:
            raise ValueError("source_id must be non-empty")
        if not 0 <= self.score <= 100:
            raise ValueError("score must be between 0 and 100")
        signals = tuple(dict.fromkeys(str(value) for value in self.signals if str(value)))
        object.__setattr__(self, "signals", signals)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "source_id": self.source_id,
            "score": self.score,
            "signals": list(self.signals),
        }


def assess_source_quality(source: SourceRecord) -> SourceQuality:
    """Score declared provenance and retrieval integrity deterministically.

    The class is an explicit input.  In particular, no branch examines a
    TLD, so a convincing-looking hostname cannot acquire official status by
    itself.  A future provenance resolver may change ``source_class`` before
    this function runs, with the resolver's evidence recorded separately.
    """

    effective_class = source.source_class
    if (
        effective_class != SourceClass.UNKNOWN
        and source.source_classification_basis
        == SourceClassificationBasis.UNVERIFIED
    ):
        # This is unreachable for current SourceRecord instances, but retains
        # a safe downgrade for a legacy artifact reader.
        effective_class = SourceClass.UNKNOWN
    score = _BASE_QUALITY_SCORES[effective_class]
    signals = [f"declared_source_class:{source.source_class.value}"]
    signals.append(
        "source_classification_basis:"
        f"{source.source_classification_basis.value}"
    )
    signals.append(f"source_channel:{source.source_channel.value}")
    if source.source_channel == SourceChannel.PUBLIC_WEB:
        try:
            canonical = canonicalize_url(source.canonical_url)
        except ValueError:
            canonical = None
            score -= 20
            signals.append("invalid_canonical_url")
        if canonical and canonical.scheme == "https":
            score += 2
            signals.append("https_transport")
    else:
        # A ``local://`` locator is deliberately not an HTTP URL. Penalizing
        # it for lacking web transport would blur source provenance with the
        # fact that the user authorized a local collection.
        signals.append("local_collection_locator")
    if source.content_verified:
        score += 5
        signals.append("fetched_content_verified")
    else:
        score = min(score, 55)
        signals.append("content_not_verified")
    if source.content_hash:
        signals.append("content_hash_recorded")
    else:
        signals.append("content_hash_missing")
    if source.publisher_id:
        score += 2
        signals.append("publisher_id_declared")
    if source.published_on:
        signals.append("publication_date_declared")
    else:
        signals.append("publication_date_unknown")
    if source.is_snippet or source.source_class == SourceClass.SEARCH_SNIPPET:
        score = min(score, 25)
        signals.append("snippet_only")
    return SourceQuality(source_id=source.source_id, score=max(0, min(100, score)), signals=tuple(signals))


def make_source_record(
    *,
    source_id: str,
    url: str,
    title: str,
    retrieved_at: str,
    source_class: SourceClass | str = SourceClass.UNKNOWN,
    source_channel: SourceChannel | str = SourceChannel.PUBLIC_WEB,
    source_connector_id: str = "public_web",
    source_classification_basis: SourceClassificationBasis | str = (
        SourceClassificationBasis.UNVERIFIED
    ),
    publisher_id: str | None = None,
    source_group: str | None = None,
    published_on: str | None = None,
    content_hash: str | None = None,
    content_artifact_id: str | None = None,
    is_snippet: bool = False,
    content_verified: bool = False,
    allowed_schemes: Iterable[str] = ("https", "http"),
) -> SourceRecord:
    """Build a source record using canonical URL identity and explicit provenance.

    ``source_group`` should be a verified publisher identity when one is
    available.  It deliberately falls back to the exact host rather than an
    eTLD+1 heuristic; a suffix is not evidence that unrelated subdomains have
    independent editorial control.
    """

    resolved_channel = SourceChannel(source_channel)
    if resolved_channel != SourceChannel.PUBLIC_WEB:
        raise ValueError(
            "make_source_record is for public_web; use "
            "make_local_collection_source_record for local collection data"
        )
    canonical = canonicalize_url(url, allowed_schemes=allowed_schemes)
    return SourceRecord(
        source_id=source_id,
        url=canonical.original_url,
        canonical_url=canonical.canonical_url,
        source_host=canonical.host,
        source_group=source_group or publisher_id or canonical.host,
        source_channel=resolved_channel,
        source_connector_id=source_connector_id,
        source_class=SourceClass(source_class),
        source_classification_basis=SourceClassificationBasis(
            source_classification_basis
        ),
        title=title,
        publisher_id=publisher_id,
        published_on=published_on,
        retrieved_at=retrieved_at,
        content_hash=content_hash,
        content_artifact_id=content_artifact_id,
        is_snippet=is_snippet,
        content_verified=content_verified,
    )


def make_local_collection_source_record(
    *,
    source_id: str,
    resource_locator: str,
    title: str,
    retrieved_at: str,
    source_connector_id: str = "local_collection",
    source_class: SourceClass | str = SourceClass.UNKNOWN,
    source_classification_basis: SourceClassificationBasis | str = (
        SourceClassificationBasis.UNVERIFIED
    ),
    publisher_id: str | None = None,
    source_group: str | None = None,
    published_on: str | None = None,
    content_hash: str | None = None,
    content_artifact_id: str | None = None,
    is_snippet: bool = False,
    content_verified: bool = False,
) -> SourceRecord:
    """Build a SourceRecord for an authorized local collection document.

    A local document uses an opaque ``local://collection/<document-id>``
    resource identity. It is intentionally not run through web URL
    canonicalization or egress policy, and it remains marked as
    ``local_collection`` throughout trace, evidence, and presentation layers.
    """

    locator = _required_text(resource_locator, "resource_locator")
    parsed = urlsplit(locator)
    if (
        parsed.scheme != "local"
        or not parsed.netloc
        or not parsed.path.strip("/")
        or parsed.query
        or parsed.fragment
        or any(segment in {".", ".."} for segment in parsed.path.split("/"))
    ):
        raise ValueError(
            "local collection locator must be local://<collection>/<document-id>"
        )
    canonical_locator = urlunsplit(
        (
            "local",
            parsed.netloc.casefold(),
            quote(parsed.path, safe="/%:@!$&'()*+,;=-._~"),
            "",
            "",
        )
    )
    collection_host = parsed.netloc.casefold()
    return SourceRecord(
        source_id=source_id,
        url=canonical_locator,
        canonical_url=canonical_locator,
        source_host=collection_host,
        source_group=source_group or publisher_id or collection_host,
        source_channel=SourceChannel.LOCAL_COLLECTION,
        source_connector_id=source_connector_id,
        source_class=SourceClass(source_class),
        source_classification_basis=SourceClassificationBasis(
            source_classification_basis
        ),
        title=title,
        publisher_id=publisher_id,
        published_on=published_on,
        retrieved_at=retrieved_at,
        content_hash=content_hash,
        content_artifact_id=content_artifact_id,
        is_snippet=is_snippet,
        content_verified=content_verified,
    )


def register_full_fetch(
    *,
    artifact_store: object,
    source_id: str,
    url: str,
    title: str,
    content: str,
    retrieved_at: str,
    source_class: SourceClass | str = SourceClass.UNKNOWN,
    source_connector_id: str = "public_web",
    source_classification_basis: SourceClassificationBasis | str = (
        SourceClassificationBasis.UNVERIFIED
    ),
    publisher_id: str | None = None,
    source_group: str | None = None,
    published_on: str | None = None,
    allowed_schemes: Iterable[str] = ("https", "http"),
) -> SourceRecord:
    """Persist one full page snapshot and return its auditable SourceRecord.

    The function is the only intended bridge from a raw fetch response to the
    General evidence ledger.  It deliberately does not accept a caller-made
    ``content_hash`` or ``content_artifact_id``: those are derived from the
    stored bytes, so a tool adapter cannot accidentally label a summary or a
    different page version as the fetched snapshot.
    """

    from .artifact_store import ContentArtifactStore

    if not isinstance(artifact_store, ContentArtifactStore):
        raise TypeError("artifact_store must be ContentArtifactStore")
    artifact = artifact_store.store(content)
    return make_source_record(
        source_id=source_id,
        url=url,
        title=title,
        retrieved_at=retrieved_at,
        source_class=source_class,
        source_connector_id=source_connector_id,
        source_classification_basis=source_classification_basis,
        publisher_id=publisher_id,
        source_group=source_group,
        published_on=published_on,
        content_hash=artifact.content_hash,
        content_artifact_id=artifact.artifact_id,
        content_verified=True,
        allowed_schemes=allowed_schemes,
    )


def register_local_collection_fetch(
    *,
    artifact_store: object,
    source_id: str,
    resource_locator: str,
    title: str,
    content: str,
    retrieved_at: str,
    source_connector_id: str = "local_collection",
    source_class: SourceClass | str = SourceClass.UNKNOWN,
    source_classification_basis: SourceClassificationBasis | str = (
        SourceClassificationBasis.UNVERIFIED
    ),
    publisher_id: str | None = None,
    source_group: str | None = None,
    published_on: str | None = None,
) -> SourceRecord:
    """Persist an authorized local document and derive its snapshot identity."""

    from .artifact_store import ContentArtifactStore

    if not isinstance(artifact_store, ContentArtifactStore):
        raise TypeError("artifact_store must be ContentArtifactStore")
    artifact = artifact_store.store(content)
    return make_local_collection_source_record(
        source_id=source_id,
        resource_locator=resource_locator,
        title=title,
        retrieved_at=retrieved_at,
        source_connector_id=source_connector_id,
        source_class=source_class,
        source_classification_basis=source_classification_basis,
        publisher_id=publisher_id,
        source_group=source_group,
        published_on=published_on,
        content_hash=artifact.content_hash,
        content_artifact_id=artifact.artifact_id,
        content_verified=True,
    )


def source_age_days(source: SourceRecord, *, as_of: date) -> int | None:
    """Return age from declared publication date; unknown dates stay unknown."""

    if source.published_on is None:
        return None
    return (as_of - date.fromisoformat(source.published_on)).days
