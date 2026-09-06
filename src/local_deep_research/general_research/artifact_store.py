"""Content-addressed storage for replayable General Research page snapshots.

The General evidence ledger stores only a source reference, a content hash,
and a short quote.  Full fetched text is persisted separately under the run's
operator-selected artifact directory.  This avoids putting arbitrary web
content into trace metadata while preserving enough material to verify a quote
and replay a run later.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import tempfile


CONTENT_ARTIFACT_SCHEMA_VERSION = "general-content-artifact/v1"
_ARTIFACT_ID_RE = re.compile(r"^sha256-[0-9a-f]{64}$")


class ContentArtifactError(ValueError):
    """The on-disk snapshot is missing, malformed, or does not verify."""


def _content_hash(content: str) -> str:
    if not isinstance(content, str) or not content:
        raise ContentArtifactError("content must be a non-empty string")
    return "sha256:" + sha256(content.encode("utf-8")).hexdigest()


def _artifact_id(content_hash: str) -> str:
    if not content_hash.startswith("sha256:"):
        raise ContentArtifactError("content_hash must use the sha256: prefix")
    digest = content_hash.removeprefix("sha256:")
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ContentArtifactError("content_hash must contain a lowercase SHA-256 digest")
    return f"sha256-{digest}"


@dataclass(frozen=True, slots=True)
class ContentArtifact:
    """Stable reference to an immutable fetched-text snapshot."""

    artifact_id: str
    content_hash: str
    byte_length: int
    schema_version: str = CONTENT_ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CONTENT_ARTIFACT_SCHEMA_VERSION:
            raise ContentArtifactError("unsupported content artifact schema")
        if not _ARTIFACT_ID_RE.fullmatch(self.artifact_id):
            raise ContentArtifactError("artifact_id is malformed")
        if _artifact_id(self.content_hash) != self.artifact_id:
            raise ContentArtifactError("artifact_id must agree with content_hash")
        if isinstance(self.byte_length, bool) or not isinstance(self.byte_length, int):
            raise ContentArtifactError("byte_length must be an integer")
        if self.byte_length <= 0:
            raise ContentArtifactError("byte_length must be positive")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "artifact_id": self.artifact_id,
            "content_hash": self.content_hash,
            "byte_length": self.byte_length,
        }


class ContentArtifactStore:
    """Append-only, content-addressed text snapshots below one explicit root."""

    def __init__(self, root_dir: str | Path, *, max_content_bytes: int = 5_000_000):
        root = Path(root_dir).expanduser()
        if not root.is_absolute():
            raise ContentArtifactError("artifact root must be an absolute path")
        if (
            isinstance(max_content_bytes, bool)
            or not isinstance(max_content_bytes, int)
            or max_content_bytes < 1
        ):
            raise ValueError("max_content_bytes must be a positive integer")
        self._root = root.resolve()
        self._snapshots_dir = self._root / "source_snapshots"
        self._max_content_bytes = max_content_bytes

    @property
    def root_dir(self) -> Path:
        return self._root

    @staticmethod
    def _metadata_path(snapshot_path: Path) -> Path:
        return snapshot_path.with_suffix(".json")

    def _paths(self, artifact_id: str) -> tuple[Path, Path]:
        if not _ARTIFACT_ID_RE.fullmatch(artifact_id):
            raise ContentArtifactError("artifact_id is malformed")
        snapshot_path = self._snapshots_dir / f"{artifact_id}.txt"
        # artifact_id is strictly matched, but retain this guard so future
        # refactors cannot turn a trace value into a path traversal sink.
        if snapshot_path.parent.resolve() != self._snapshots_dir.resolve():
            raise ContentArtifactError("artifact path escaped snapshot directory")
        return snapshot_path, self._metadata_path(snapshot_path)

    @staticmethod
    def _atomic_write(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", dir=path.parent
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, path)
        except Exception:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise

    def store(self, content: str) -> ContentArtifact:
        """Persist content once and return a replayable content-addressed ID."""

        payload = content.encode("utf-8") if isinstance(content, str) else b""
        if not payload:
            raise ContentArtifactError("content must be a non-empty string")
        if len(payload) > self._max_content_bytes:
            raise ContentArtifactError(
                f"content exceeds max_content_bytes={self._max_content_bytes}"
            )
        content_hash = _content_hash(content)
        artifact = ContentArtifact(
            artifact_id=_artifact_id(content_hash),
            content_hash=content_hash,
            byte_length=len(payload),
        )
        snapshot_path, metadata_path = self._paths(artifact.artifact_id)
        if snapshot_path.exists():
            existing = snapshot_path.read_bytes()
            if existing != payload:
                raise ContentArtifactError("content-addressed snapshot collision")
        else:
            self._atomic_write(snapshot_path, payload)
        metadata = json.dumps(
            artifact.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        if metadata_path.exists():
            try:
                existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ContentArtifactError("snapshot metadata cannot be read") from exc
            if existing_metadata != artifact.to_dict():
                raise ContentArtifactError("snapshot metadata does not match content")
        else:
            self._atomic_write(metadata_path, metadata)
        return artifact

    def load(self, artifact_id: str, *, expected_content_hash: str | None = None) -> str:
        """Load and verify a stored snapshot before evidence extraction/replay."""

        snapshot_path, metadata_path = self._paths(artifact_id)
        try:
            payload = snapshot_path.read_bytes()
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ContentArtifactError("snapshot or metadata is missing") from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise ContentArtifactError("snapshot or metadata cannot be read") from exc
        try:
            artifact = ContentArtifact(
                artifact_id=str(metadata["artifact_id"]),
                content_hash=str(metadata["content_hash"]),
                byte_length=int(metadata["byte_length"]),
                schema_version=str(metadata["schema_version"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ContentArtifactError("snapshot metadata is malformed") from exc
        actual_hash = "sha256:" + sha256(payload).hexdigest()
        if artifact.artifact_id != artifact_id or artifact.content_hash != actual_hash:
            raise ContentArtifactError("snapshot content does not match its metadata")
        if artifact.byte_length != len(payload):
            raise ContentArtifactError("snapshot byte length does not match metadata")
        if expected_content_hash is not None and expected_content_hash != actual_hash:
            raise ContentArtifactError("snapshot content does not match expected hash")
        try:
            return payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ContentArtifactError("snapshot is not UTF-8 text") from exc


__all__ = [
    "CONTENT_ARTIFACT_SCHEMA_VERSION",
    "ContentArtifact",
    "ContentArtifactError",
    "ContentArtifactStore",
]
