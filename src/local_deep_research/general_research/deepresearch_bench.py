"""Strict DeepResearch Bench task selection and raw-report export.

This module does not implement a second evaluator.  It reads the public
benchmark's query JSONL supplied by the operator, freezes the documented
General V1 development selection, and emits the exact raw-data shape consumed
by the benchmark's official RACE/FACT pipeline.  General's own artifact audit
remains a complementary integrity check, not a substitute for that pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .artifact_audit import audit_general_run_artifact
from .config import GeneralExecutionConfig
from .run_manifest import GeneralRunManifest
from .runner import GeneralRunArtifacts
from .workflow import GeneralWorkflowResult


# Legacy protocol: see docs/backup/GENERAL_V1_BENCHMARK_PROTOCOL.zh.md. These
# IDs are intentionally public-benchmark identifiers, not locally authored
# test questions.
GENERAL_V1_DEEPRESEARCH_BENCH_DEVELOPMENT_IDS = (
    8,
    19,
    20,
    31,
    42,
    44,
    62,
    66,
    68,
    69,
    72,
    81,
)
DEEPRESEARCH_BENCH_BATCH_MANIFEST_SCHEMA_VERSION = "deepresearch-bench-batch/v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _task_id(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("DeepResearch Bench task id must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class DeepResearchBenchTask:
    """One public query, retaining metadata only for selection/audit."""

    task_id: int
    prompt: str
    language: str
    topic: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _task_id(self.task_id))
        object.__setattr__(self, "prompt", _text(self.prompt, field_name="prompt"))
        object.__setattr__(self, "language", _text(self.language, field_name="language"))
        object.__setattr__(self, "topic", _text(self.topic, field_name="topic"))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DeepResearchBenchTask":
        if not isinstance(value, Mapping):
            raise TypeError("DeepResearch Bench task must be a mapping")
        required = {"id", "prompt", "language", "topic"}
        missing = required.difference(value)
        if missing:
            raise ValueError(
                "DeepResearch Bench task is missing fields: " + ", ".join(sorted(missing))
            )
        return cls(
            task_id=_task_id(value["id"]),
            prompt=_text(value["prompt"], field_name="prompt"),
            language=_text(value["language"], field_name="language"),
            topic=_text(value["topic"], field_name="topic"),
        )


@dataclass(frozen=True, slots=True)
class DeepResearchBenchRecord:
    """Exact three-field raw report contract accepted by the official tools."""

    task_id: int
    prompt: str
    article: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _task_id(self.task_id))
        object.__setattr__(self, "prompt", _text(self.prompt, field_name="prompt"))
        object.__setattr__(self, "article", _text(self.article, field_name="article"))

    def to_dict(self) -> dict[str, object]:
        return {"id": self.task_id, "prompt": self.prompt, "article": self.article}


@dataclass(frozen=True, slots=True)
class DeepResearchBenchTaskExecution:
    """A General run bound to one official benchmark task.

    The batch layer reads the saved General artifacts rather than trusting a
    callback's narration. This prevents a report produced for a different
    prompt, config, or code revision from entering the evaluator's raw JSONL.
    """

    result: GeneralWorkflowResult
    artifacts: GeneralRunArtifacts

    def __post_init__(self) -> None:
        if not isinstance(self.result, GeneralWorkflowResult):
            raise TypeError("result must be GeneralWorkflowResult")
        if not isinstance(self.artifacts, GeneralRunArtifacts):
            raise TypeError("artifacts must be GeneralRunArtifacts")


@dataclass(frozen=True, slots=True)
class DeepResearchBenchBatch:
    """The batch's immutable index beside official-format raw reports."""

    batch_dir: Path
    raw_records_path: Path
    manifest_path: Path
    task_ids: tuple[int, ...]


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read required General artifact: {path.name}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"required General artifact must be a JSON object: {path.name}")
    return value


def _task_input_fingerprint(tasks: tuple[DeepResearchBenchTask, ...]) -> str:
    payload = [
        {
            "id": task.task_id,
            "prompt": task.prompt,
            "language": task.language,
            "topic": task.topic,
        }
        for task in tasks
    ]
    return sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _validated_task_run(
    task: DeepResearchBenchTask, execution: DeepResearchBenchTaskExecution
) -> dict[str, object]:
    artifacts = execution.artifacts
    config = GeneralExecutionConfig.from_mapping(_load_json_object(artifacts.config_path))
    manifest = GeneralRunManifest.from_mapping(_load_json_object(artifacts.manifest_path))
    manifest.validate_for_config(config)
    if config.run.query != task.prompt:
        raise ValueError("benchmark task prompt differs from its General execution config")
    if manifest.execution_purpose != "benchmark":
        raise ValueError("DeepResearch Bench runs require execution_purpose='benchmark'")
    if manifest.code_revision == "unrecorded":
        raise ValueError("DeepResearch Bench runs require an immutable code_revision")
    artifact_audit = audit_general_run_artifact(artifacts.run_dir)
    if not artifact_audit.artifact_integrity_passed:
        raise ValueError("General artifact integrity audit failed before benchmark export")
    return {
        "task_id": task.task_id,
        "run_id": config.run.run_id,
        "workflow_status": execution.result.status.value,
        "artifact_run_dir": str(artifacts.run_dir),
        "execution_config_digest_sha256": config.digest(),
        "run_manifest_digest_sha256": manifest.digest(),
        "code_revision": manifest.code_revision,
        "artifact_integrity_passed": artifact_audit.artifact_integrity_passed,
        "publishable": artifact_audit.publishable,
    }


def run_deepresearch_bench_batch(
    *,
    tasks: Iterable[DeepResearchBenchTask],
    task_runner: Callable[[DeepResearchBenchTask], DeepResearchBenchTaskExecution],
    output_root: str | Path,
    batch_id: str,
    clock: Callable[[], str] = _now,
) -> DeepResearchBenchBatch:
    """Run a selected public subset and export its official raw JSONL safely.

    Tasks run in fixed order and one at a time. General already has bounded
    worker concurrency inside each task; cross-task concurrency needs actual
    provider-rate calibration and is intentionally outside this evaluator
    adapter. The function accepts any frozen subset, not just the default 12.
    """

    task_list = tuple(tasks)
    if not task_list or not all(isinstance(task, DeepResearchBenchTask) for task in task_list):
        raise ValueError("tasks must contain at least one DeepResearchBenchTask")
    task_ids = [task.task_id for task in task_list]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("benchmark batch task IDs must be unique")
    if not callable(task_runner) or not callable(clock):
        raise TypeError("task_runner and clock must be callable")
    batch_name = _text(batch_id, field_name="batch_id")
    if any(character.isspace() for character in batch_name):
        raise ValueError("batch_id must not contain whitespace")
    root = Path(output_root).expanduser()
    if not root.is_absolute():
        raise ValueError("benchmark output_root must be absolute")
    batch_dir = root.resolve() / f"general-deepresearch-bench-{batch_name}"
    if batch_dir.exists():
        raise ValueError("benchmark batch directory already exists; refuse to overwrite")
    batch_dir.mkdir(parents=True, exist_ok=False)
    raw_records_path = batch_dir / "raw_reports.jsonl"
    manifest_path = batch_dir / "batch_manifest.json"
    records: list[DeepResearchBenchRecord] = []
    task_runs: list[dict[str, object]] = []
    try:
        for task in task_list:
            execution = task_runner(task)
            if not isinstance(execution, DeepResearchBenchTaskExecution):
                raise TypeError("task_runner must return DeepResearchBenchTaskExecution")
            task_runs.append(_validated_task_run(task, execution))
            records.append(record_from_general_result(task, execution.result))
    except Exception as exc:
        # Preserve the partial index, but never create an evaluator input that
        # could be mistaken for a complete benchmark subset.
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": DEEPRESEARCH_BENCH_BATCH_MANIFEST_SCHEMA_VERSION,
                    "benchmark": "Ayanami0730/deep_research_bench",
                    "created_at": _text(clock(), field_name="clock result"),
                    "task_ids": task_ids,
                    "task_input_fingerprint_sha256": _task_input_fingerprint(task_list),
                    "batch_state": "failed_before_raw_export",
                    "completed_task_runs": task_runs,
                    "failure_type": type(exc).__name__,
                    "execution_order": "serial_fixed_order",
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            encoding="utf-8",
            newline="\n",
        )
        raise
    write_deepresearch_bench_records(raw_records_path, records)
    manifest_payload = {
        "schema_version": DEEPRESEARCH_BENCH_BATCH_MANIFEST_SCHEMA_VERSION,
        "benchmark": "Ayanami0730/deep_research_bench",
        "created_at": _text(clock(), field_name="clock result"),
        "task_ids": task_ids,
        "task_input_fingerprint_sha256": _task_input_fingerprint(task_list),
        "batch_state": "complete",
        "raw_records_path": raw_records_path.name,
        "task_runs": task_runs,
        "execution_order": "serial_fixed_order",
    }
    manifest_path.write_text(
        json.dumps(manifest_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
        newline="\n",
    )
    return DeepResearchBenchBatch(
        batch_dir=batch_dir,
        raw_records_path=raw_records_path,
        manifest_path=manifest_path,
        task_ids=tuple(task_ids),
    )


def load_deepresearch_bench_tasks(path: str | Path) -> tuple[DeepResearchBenchTask, ...]:
    """Read official query JSONL without copying its questions into the repo."""

    task_path = Path(path).expanduser()
    try:
        lines = task_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError("DeepResearch Bench query file cannot be read") from exc
    if not lines:
        raise ValueError("DeepResearch Bench query file must not be empty")
    tasks: list[DeepResearchBenchTask] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"DeepResearch Bench query line {line_number} is invalid JSON"
            ) from exc
        try:
            tasks.append(DeepResearchBenchTask.from_mapping(raw))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"DeepResearch Bench query line {line_number} is invalid: {exc}"
            ) from exc
    task_ids = [task.task_id for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("DeepResearch Bench query file uses duplicate task IDs")
    return tuple(tasks)


def select_general_v1_development_tasks(
    tasks: Iterable[DeepResearchBenchTask],
) -> tuple[DeepResearchBenchTask, ...]:
    """Select exactly the frozen public 12-task development subset."""

    task_list = tuple(tasks)
    if not all(isinstance(task, DeepResearchBenchTask) for task in task_list):
        raise TypeError("tasks must contain DeepResearchBenchTask values")
    by_id = {task.task_id: task for task in task_list}
    missing = sorted(set(GENERAL_V1_DEEPRESEARCH_BENCH_DEVELOPMENT_IDS) - set(by_id))
    if missing:
        raise ValueError(
            "DeepResearch Bench query file lacks frozen General V1 development IDs: "
            + ", ".join(str(task_id) for task_id in missing)
        )
    return tuple(by_id[task_id] for task_id in GENERAL_V1_DEEPRESEARCH_BENCH_DEVELOPMENT_IDS)


def record_from_general_result(
    task: DeepResearchBenchTask, result: GeneralWorkflowResult
) -> DeepResearchBenchRecord:
    """Export the stored General report, including a safe incomplete result."""

    if not isinstance(task, DeepResearchBenchTask):
        raise TypeError("task must be DeepResearchBenchTask")
    if not isinstance(result, GeneralWorkflowResult):
        raise TypeError("result must be GeneralWorkflowResult")
    return DeepResearchBenchRecord(
        task_id=task.task_id,
        prompt=task.prompt,
        article=result.report_markdown,
    )


def write_deepresearch_bench_records(
    path: str | Path, records: Iterable[DeepResearchBenchRecord]
) -> None:
    """Create a deterministic official-format JSONL file without overwriting."""

    output_path = Path(path).expanduser()
    if output_path.exists():
        raise ValueError("DeepResearch Bench output already exists; refuse to overwrite")
    record_list = tuple(records)
    if not record_list or not all(
        isinstance(record, DeepResearchBenchRecord) for record in record_list
    ):
        raise ValueError("records must contain at least one DeepResearchBenchRecord")
    ids = [record.task_id for record in record_list]
    if len(ids) != len(set(ids)):
        raise ValueError("DeepResearch Bench output must use unique task IDs")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
        for record in record_list
    )
    try:
        output_path.write_text(payload, encoding="utf-8", newline="\n")
    except OSError as exc:
        raise ValueError("DeepResearch Bench output cannot be written") from exc


__all__ = [
    "DEEPRESEARCH_BENCH_BATCH_MANIFEST_SCHEMA_VERSION",
    "DeepResearchBenchBatch",
    "DeepResearchBenchRecord",
    "DeepResearchBenchTask",
    "DeepResearchBenchTaskExecution",
    "GENERAL_V1_DEEPRESEARCH_BENCH_DEVELOPMENT_IDS",
    "load_deepresearch_bench_tasks",
    "record_from_general_result",
    "run_deepresearch_bench_batch",
    "select_general_v1_development_tasks",
    "write_deepresearch_bench_records",
]
