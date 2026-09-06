"""Versioned, runtime-neutral schema for executable agent trajectories."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


SCHEMA_VERSION = "agent-trace/v1"


class EventKind(StrEnum):
    RUN_START = "run_start"
    MESSAGE = "message"
    TOOL_CALL = "tool_call"
    TOOL_OBSERVATION = "tool_observation"
    FINAL_ANSWER = "final_answer"
    ERROR = "error"
    RUN_END = "run_end"


class RunOutcome(StrEnum):
    SUCCESS = "success"
    ERROR = "error"
    STOPPED = "stopped"


class TraceEvent(BaseModel):
    """One ordered event in an agent run.

    The schema deliberately records observations separately from tool calls.
    This makes cached replay possible and prevents SFT data from silently
    replacing real tool output with model-authored text.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["agent-trace/v1"] = SCHEMA_VERSION
    run_id: str = Field(min_length=1)
    event_id: int = Field(ge=0)
    turn: int = Field(ge=0)
    timestamp: datetime
    kind: EventKind

    role: Literal["system", "user", "assistant", "tool"] | None = None
    content: str | list[Any] | dict[str, Any] | None = None

    tool_call_id: str | None = None
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None

    outcome: RunOutcome | None = None
    error_type: str | None = None
    latency_ms: float | None = Field(default=None, ge=0)
    token_usage: dict[str, int] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_kind_specific_fields(self) -> "TraceEvent":
        if self.kind == EventKind.TOOL_CALL:
            if not self.tool_call_id or not self.tool_name:
                raise ValueError(
                    "tool_call requires tool_call_id and tool_name"
                )
            if self.tool_input is None:
                raise ValueError("tool_call requires tool_input")
        elif self.kind == EventKind.TOOL_OBSERVATION:
            if not self.tool_call_id or not self.tool_name:
                raise ValueError(
                    "tool_observation requires tool_call_id and tool_name"
                )
            if self.role != "tool":
                raise ValueError("tool_observation role must be 'tool'")
        elif self.kind == EventKind.FINAL_ANSWER:
            if self.role != "assistant" or self.content is None:
                raise ValueError(
                    "final_answer requires assistant role and non-null content"
                )
        elif self.kind == EventKind.ERROR:
            if not self.error_type:
                raise ValueError("error requires error_type")
        elif self.kind == EventKind.RUN_END:
            if self.outcome is None:
                raise ValueError("run_end requires outcome")
        return self
