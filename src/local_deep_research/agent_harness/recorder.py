"""In-memory recorder and thin adapters for LangChain messages."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any
from uuid import uuid4

from .schema import EventKind, RunOutcome, TraceEvent


class TraceRecorder:
    """Build an ordered trajectory without coupling the schema to LangGraph."""

    def __init__(
        self,
        *,
        run_id: str | None = None,
        runtime: str,
        model: str,
        strategy: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.run_id = run_id or str(uuid4())
        self._next_event_id = 0
        self._turn = 0
        self._events: list[TraceEvent] = []
        self._tool_names_by_call: dict[str, str] = {}
        # LangGraph's ``updates`` stream can surface the same message again
        # when state is merged between nodes. A trajectory represents tool
        # execution, not graph-state delivery, so record each call/observation
        # once by its stable call id.
        self._recorded_tool_calls: set[tuple[str, str, str]] = set()
        self._recorded_tool_observations: set[tuple[str, str, str]] = set()
        self._append(
            EventKind.RUN_START,
            metadata={
                "runtime": runtime,
                "model": model,
                "strategy": strategy,
                **(metadata or {}),
            },
        )

    @property
    def events(self) -> tuple[TraceEvent, ...]:
        return tuple(self._events)

    def _append(self, kind: EventKind, **fields: Any) -> TraceEvent:
        event = TraceEvent(
            run_id=self.run_id,
            event_id=self._next_event_id,
            turn=self._turn,
            timestamp=datetime.now(timezone.utc),
            kind=kind,
            **fields,
        )
        self._events.append(event)
        self._next_event_id += 1
        return event

    def record_message(
        self,
        role: str,
        content: str | list[Any] | dict[str, Any],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> TraceEvent:
        if role == "user":
            self._turn += 1
        return self._append(
            EventKind.MESSAGE,
            role=role,
            content=content,
            metadata=metadata or {},
        )

    def record_assistant_message(self, message: Any) -> list[TraceEvent]:
        """Record content and structured calls from a LangChain AIMessage.

        Only the small public message surface is used, so tests and other
        runtimes can supply compatible objects without importing LangChain.
        """

        recorded: list[TraceEvent] = []
        content = getattr(message, "content", None)
        additional = getattr(message, "additional_kwargs", None) or {}
        response_metadata = getattr(message, "response_metadata", None) or {}
        planner_budget = additional.get("planner_budget")
        if content:
            recorded.append(
                self._append(
                    EventKind.MESSAGE,
                    role="assistant",
                    content=content,
                    metadata={
                        "reasoning_content": additional.get(
                            "reasoning_content"
                        ),
                        "response_metadata": response_metadata,
                        "planner_budget": planner_budget,
                    },
                )
            )

        for call in getattr(message, "tool_calls", None) or []:
            call_id = str(call.get("id") or call.get("tool_call_id") or "")
            tool_name = str(call.get("name") or "")
            tool_input = dict(call.get("args") or {})
            call_signature = (
                call_id,
                tool_name,
                json.dumps(tool_input, ensure_ascii=False, sort_keys=True),
            )
            if call_signature in self._recorded_tool_calls:
                continue
            if call_id and tool_name:
                self._tool_names_by_call[call_id] = tool_name
                self._recorded_tool_calls.add(call_signature)
            recorded.append(
                self._append(
                    EventKind.TOOL_CALL,
                    role="assistant",
                    tool_call_id=call_id,
                    tool_name=tool_name,
                    tool_input=tool_input,
                    metadata={"planner_budget": planner_budget}
                    if planner_budget
                    else {},
                )
            )
        return recorded

    def record_tool_message(self, message: Any) -> TraceEvent | None:
        call_id = str(getattr(message, "tool_call_id", "") or "")
        tool_name = str(getattr(message, "name", "") or "")
        tool_name = tool_name or self._tool_names_by_call.get(call_id, "")
        content = getattr(message, "content", None)
        observation_signature = (
            call_id,
            tool_name,
            json.dumps(content, ensure_ascii=False, sort_keys=True, default=str),
        )
        if observation_signature in self._recorded_tool_observations:
            return None
        status = getattr(message, "status", None)
        event = self._append(
            EventKind.TOOL_OBSERVATION,
            role="tool",
            content=content,
            tool_call_id=call_id,
            tool_name=tool_name,
            metadata={"status": status} if status is not None else {},
        )
        if call_id:
            self._recorded_tool_observations.add(observation_signature)
        return event

    def record_final_answer(self, content: str) -> TraceEvent:
        return self._append(
            EventKind.FINAL_ANSWER,
            role="assistant",
            content=content,
        )

    def record_error(self, error: BaseException) -> TraceEvent:
        return self._append(
            EventKind.ERROR,
            content=str(error),
            error_type=type(error).__name__,
        )

    def finish(
        self,
        outcome: RunOutcome,
        *,
        latency_ms: float | None = None,
        token_usage: dict[str, int] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TraceEvent:
        return self._append(
            EventKind.RUN_END,
            outcome=outcome,
            latency_ms=latency_ms,
            token_usage=token_usage,
            metadata=metadata or {},
        )
