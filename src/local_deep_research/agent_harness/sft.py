"""Convert validated runtime traces into tool-aware SFT conversations."""

from __future__ import annotations

from typing import Any, Iterable

from .schema import EventKind, RunOutcome, TraceEvent
from .store import JsonlTraceStore


WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the selected primary source and return grounded result "
            "snippets with source indices."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The focused search query.",
                }
            },
            "required": ["query"],
        },
    },
}


def _search_tool_schema(name: str) -> dict[str, Any]:
    if name == "web_search":
        return WEB_SEARCH_TOOL
    description = (
        "Search the named specialized evidence source and return grounded "
        "result snippets with citation indices."
    )
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The focused search query.",
                    }
                },
                "required": ["query"],
            },
        },
    }


def _fetch_tool_schema(fetch_mode: str | None) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "url": {
            "type": "string",
            "description": "A URL or local library document path to read.",
        }
    }
    required = ["url"]
    if fetch_mode in {"summary_focus", "summary_focus_query"}:
        properties["focus"] = {
            "type": "string",
            "description": "The specific claim or question to extract evidence for.",
        }
        required.append("focus")
    return {
        "type": "function",
        "function": {
            "name": "fetch_content",
            "description": "Read a source page and return evidence under a stable citation.",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def _subtopic_tool_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "research_subtopic",
            "description": "Delegate two to five focused, non-overlapping research questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subtopics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 2,
                        "maxItems": 5,
                    }
                },
                "required": ["subtopics"],
            },
        },
    }


def infer_tool_schemas(
    events: list[TraceEvent],
    fetch_mode: str | None,
) -> list[dict[str, Any]]:
    """Derive the runtime tool surface used by a trajectory.

    This is public because replay scenarios and SFT examples must describe the
    same tools instead of maintaining two nearly-identical schema registries.
    """
    names = list(
        dict.fromkeys(
            event.tool_name
            for event in events
            if event.kind == EventKind.TOOL_CALL and event.tool_name
        )
    )
    schemas = []
    for name in names:
        if name == "web_search" or name.startswith("search_"):
            schemas.append(_search_tool_schema(name))
        elif name == "fetch_content":
            schemas.append(_fetch_tool_schema(fetch_mode))
        elif name == "research_subtopic":
            schemas.append(_subtopic_tool_schema())
        else:
            raise SftConversionError(
                f"cannot infer a training schema for tool {name!r}"
            )
    if not schemas:
        raise SftConversionError("trace contains no tool calls")
    return schemas


class SftConversionError(ValueError):
    """Raised when a trace is valid for diagnostics but unsafe for SFT."""


def compact_tool_observation(content: str, max_chars: int) -> str:
    """Keep a bounded head/tail view while marking the original length."""
    marker = f"\n...[tool observation compacted from {len(content)} chars]...\n"
    available = max_chars - len(marker)
    tail_chars = max(40, available // 4)
    head_chars = available - tail_chars
    return f"{content[:head_chars]}{marker}{content[-tail_chars:]}"


def trace_to_sft_example(
    events: Iterable[TraceEvent],
    *,
    tools: list[dict[str, Any]] | None = None,
    max_tool_observation_chars: int | None = None,
) -> dict[str, Any]:
    """Build one OpenAI/Qwen-style conversation from a successful trace.

    The actual agent final message is preferred over the later LDR
    post-processing final_answer event so one assistant turn is not
    duplicated in the training target.
    """
    validated = JsonlTraceStore.validate(events)
    if validated[-1].outcome != RunOutcome.SUCCESS:
        raise SftConversionError(
            f"only successful traces are SFT eligible: {validated[-1].outcome}"
        )

    messages: list[dict[str, Any]] = []
    postprocessed_final: str | None = None
    index = 1
    while index < len(validated) - 1:
        event = validated[index]

        if event.kind == EventKind.ERROR:
            raise SftConversionError("trace contains an error event")

        if event.kind == EventKind.FINAL_ANSWER:
            if isinstance(event.content, str):
                postprocessed_final = event.content
            index += 1
            continue

        if event.kind == EventKind.MESSAGE:
            if event.role == "assistant":
                next_kind = (
                    validated[index + 1].kind
                    if index + 1 < len(validated)
                    else None
                )
                if next_kind == EventKind.TOOL_CALL:
                    index += 1
                    continue
            if event.role not in {"system", "user", "assistant"}:
                raise SftConversionError(
                    f"unsupported message role: {event.role}"
                )
            if not isinstance(event.content, str):
                raise SftConversionError(
                    f"{event.role} message content must be text"
                )
            messages.append({"role": event.role, "content": event.content})
            index += 1
            continue

        if event.kind == EventKind.TOOL_CALL:
            calls = []
            while (
                index < len(validated) - 1
                and validated[index].kind == EventKind.TOOL_CALL
            ):
                call = validated[index]
                calls.append(
                    {
                        "id": call.tool_call_id,
                        "type": "function",
                        "function": {
                            "name": call.tool_name,
                            "arguments": call.tool_input or {},
                        },
                    }
                )
                index += 1
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": calls,
                }
            )
            continue

        if event.kind == EventKind.TOOL_OBSERVATION:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": event.tool_call_id,
                    "name": event.tool_name,
                    "content": event.content,
                }
            )
            index += 1
            continue

        index += 1

    if not messages or messages[0].get("role") != "system":
        raise SftConversionError("trace is missing the runtime system prompt")
    if not any(message.get("role") == "user" for message in messages):
        raise SftConversionError("trace is missing a user message")
    if messages[-1].get("role") != "assistant":
        if postprocessed_final is None:
            raise SftConversionError("trace is missing a final answer")
        messages.append({"role": "assistant", "content": postprocessed_final})

    compacted_observations = 0
    original_observation_chars = 0
    compacted_observation_chars = 0
    if max_tool_observation_chars is not None:
        if max_tool_observation_chars < 160:
            raise SftConversionError(
                "max_tool_observation_chars must be at least 160"
            )
        for message in messages:
            if message.get("role") != "tool":
                continue
            content = message.get("content")
            if not isinstance(content, str):
                continue
            original_observation_chars += len(content)
            if len(content) > max_tool_observation_chars:
                message["content"] = compact_tool_observation(
                    content,
                    max_tool_observation_chars,
                )
                compacted_observations += 1
            compacted_observation_chars += len(message["content"])

    start = validated[0]
    resolved_tools = (
        tools
        if tools is not None
        else infer_tool_schemas(
            validated,
            start.metadata.get("fetch_mode"),
        )
    )
    return {
        "messages": messages,
        "tools": resolved_tools,
        "metadata": {
            "schema_version": start.schema_version,
            "run_id": start.run_id,
            "runtime": start.metadata.get("runtime"),
            "model": start.metadata.get("model"),
            "strategy": start.metadata.get("strategy"),
            "tool_observation_compaction": (
                {
                    "enabled": True,
                    "max_chars": max_tool_observation_chars,
                    "compacted_messages": compacted_observations,
                    "original_chars": original_observation_chars,
                    "compacted_chars": compacted_observation_chars,
                    "method": "head_tail_with_explicit_marker",
                }
                if max_tool_observation_chars is not None
                else {"enabled": False}
            ),
        },
    }
