"""Deterministic, provenance-preserving supervisor handoff memos.

The supervisor needs compact state from parallel workers, but a second model
"compression" call would create an untraceable factual channel and spend a
meaningful part of a small run budget.  General V1 therefore builds its first
memo form deterministically from the authoritative evidence ledger.  The
supervisor sees evidence-card claims plus IDs; the writer still reads the
full cards and citations remain verified against the fetched snapshots.
"""

from __future__ import annotations

from typing import Iterable

from .schemas import EvidenceCard, MemoFinding, ResearchMemo, ResearchTask


def build_research_memos(
    *,
    tasks: Iterable[ResearchTask],
    evidence_cards: Iterable[EvidenceCard],
    created_at: str,
) -> tuple[ResearchMemo, ...]:
    """Create one compact, evidence-linked memo for every dispatched task.

    A task without linked evidence is represented explicitly as unresolved.
    This lets a supervisor try a different source strategy in a later round;
    it is not silently mistaken for successful research.
    """

    task_list = tuple(tasks)
    cards = tuple(evidence_cards)
    if not all(isinstance(task, ResearchTask) for task in task_list):
        raise TypeError("tasks must contain ResearchTask values")
    if not all(isinstance(card, EvidenceCard) for card in cards):
        raise TypeError("evidence_cards must contain EvidenceCard values")

    memos: list[ResearchMemo] = []
    for task in task_list:
        task_items = set(task.plan_item_ids)
        linked_cards = tuple(
            card
            for card in cards
            if task_items.intersection(card.plan_item_ids)
        )
        findings = tuple(
            MemoFinding(
                finding_id=f"{task.task_id}-finding-{index:03d}",
                text=card.claim,
                evidence_ids=(card.evidence_id,),
            )
            for index, card in enumerate(linked_cards, start=1)
        )
        unresolved = ()
        if not findings:
            unresolved = (
                "No accepted evidence card yet for the assigned plan item(s).",
            )
        memos.append(
            ResearchMemo(
                task_id=task.task_id,
                plan_item_ids=task.plan_item_ids,
                findings=findings,
                unresolved_questions=unresolved,
                created_at=created_at,
            )
        )
    return tuple(memos)


__all__ = ["build_research_memos"]
