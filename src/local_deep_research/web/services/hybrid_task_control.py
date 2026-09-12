"""Bridge Hybrid runtime events to the existing research-task UI."""

import json
from pathlib import Path
from threading import RLock


class HybridTaskProgress:
    def __init__(self, callback):
        self.callback = callback
        self.lock = RLock()
        self.sequence = 0
        self.sources = set()
        self.tasks = set()
        self.model_calls = 0

    def __call__(self, event):
        if self.callback is None or event.kind not in {
            "run_started", "supervisor_delegated", "workflow_one_pass_plan",
            "workflow_requirement_plan", "model_request_started",
            "fetch_completed", "search_completed", "source_range_read",
        }:
            return
        with self.lock:
            data = event.data
            if event.kind == "fetch_completed":
                self.sources.add(data.get("source_id"))
            if event.kind in {"supervisor_delegated", "workflow_one_pass_plan"}:
                self.tasks.update(data["tasks"])
            if event.kind == "workflow_requirement_plan":
                self.tasks.update(data["requirements"])
            if event.kind == "model_request_started":
                self.model_calls = max(self.model_calls, data.get("call_index", 0))
            if event.sequence <= self.sequence:
                return
            self.sequence = event.sequence
            phase = "search"
            if event.kind == "run_started":
                message, phase = "正在规划研究任务", "search_planning"
            elif event.kind in {"supervisor_delegated", "workflow_one_pass_plan", "workflow_requirement_plan"}:
                message = f"已分配 {len(self.tasks)} 个研究子任务"
            elif event.kind == "model_request_started":
                role = data.get("role")
                if role == "writer":
                    message, phase = "正在综合证据并撰写报告", "output_generation"
                elif role == "compression":
                    message = "正在整理原文证据与研究结论"
                elif role == "supervisor":
                    message, phase = "正在规划或核对剩余研究工作", "search_planning"
                else:
                    message = f"正在研究；已发起 {self.model_calls} 次模型调用"
            elif event.kind == "fetch_completed":
                message = f"已读取 {len(self.sources)} 份来源，继续核对证据"
            elif event.kind == "search_completed":
                message = "检索完成，正在选择和阅读资料"
            elif event.kind == "source_range_read":
                message = "正在展开原文段落，核对上下文"
            else:
                return
            # Phase milestones and counts, not an estimate of time remaining.
            self.callback(message, None, {
                "phase": phase, "hybrid_event": event.kind,
                "event_sequence": event.sequence, "updated_at": event.timestamp,
                "model_calls_started": self.model_calls,
                "sources_read": len(self.sources), "research_tasks": len(self.tasks),
            })


def save_cancelled_research(runner, output_root):
    """Retain completed work without synthesizing a report or claiming resume."""
    folder = Path(output_root) / runner.run_id
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "trace.jsonl").write_text(
        "".join(json.dumps(e.to_dict(), ensure_ascii=False) + "\n" for e in runner._trace),
        encoding="utf-8",
    )
    sources = list(runner._sources_by_id.values())
    (folder / "sources.json").write_text(
        json.dumps([s.public_view() for s in sources], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    snapshots = folder / "source_snapshots"
    snapshots.mkdir(exist_ok=True)
    for source in sources:
        if source.content is not None:
            (snapshots / f"{source.source_id}.txt").write_text(source.content, encoding="utf-8")
    (folder / "run.json").write_text(json.dumps({
        "run_id": runner.run_id, "status": "suspended", "terminal_reason": "user_cancelled",
        "model_calls_used": runner._model_calls_used, "tool_calls_used": runner._tool_calls_used,
        "research_tasks": list(runner._research_state.research_tasks),
        "research_notes": list(runner._research_state.research_notes),
        "evidence_packets": getattr(runner, "_located_packets", {}),
        "resumable": False,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
