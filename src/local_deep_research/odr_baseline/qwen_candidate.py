"""Opt-in Qwen planning improvements; hosted/default runtime stays unchanged."""
import json
import re
from dataclasses import replace
from typing import Literal, Mapping
from urllib.parse import urlparse

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from .runtime import OdrBaselineRunner, ResearchBudgetExhausted, _EvidenceCandidate, _extractive_evidence_span
from .qwen_writer import QwenMultiPassageWriter
from .sources import DiscoveredResource


class QwenRequirement(BaseModel):
    requirement: str = Field(description="One independently answerable user requirement, not a paper name or broad topic.")
    retrieval_query: str = Field(description="Focused discovery query using the source document's language where useful.")
    source_channel: Literal["web", "collection"] = Field(description="Which permitted source channel can establish this claim.")
    source_document_id: str | None = Field(default=None, description="For a requirement about one named local document, copy its exact ID from the selected Collection catalogue. Otherwise null.")
    answer_focus: Literal["mechanism", "tradeoff", "fact"] = Field(description="Mechanism includes retrieval/generation/quality control; tradeoff asks only a method-specific cost, constraint or design choice.")
    evidence_query: str = Field(description="Short source-language query for the needed passage INSIDE the document: specific mechanism or tradeoff terms, not its full title or author list.")


class QwenResearchPlan(BaseModel):
    requirements: list[QwenRequirement]


class QwenPlanCompletion(BaseModel):
    missing_requirements: list[QwenRequirement] = Field(
        description="ALL user-requested research requirements absent from the draft; empty only if nothing is missing."
    )


class QwenCandidateRunner(QwenMultiPassageWriter, OdrBaselineRunner):
    candidate_implementation = "open-dev-v9-functions-and-file-links"
    evidence_windows_per_source = 3

    def __init__(self, *, source_mode: str, allowed_web_hosts=(), collection_catalogue=(), **kwargs):
        channels = {"web_only": ("web",), "collection_only": ("collection",),
                    "hybrid": ("collection", "web")}
        if source_mode not in channels:
            raise ValueError("unknown Qwen source mode")
        self._planning_channels = channels[source_mode]
        self._planning_web_hosts = tuple(allowed_web_hosts)
        self._catalogue = {item["document_id"]: item["title"] for item in collection_catalogue}
        self._target_documents = {}
        self._evidence_queries = {}
        super().__init__(**kwargs)
        if "collection" in self._planning_channels and self.collection_connector is None:
            raise ValueError("selected source mode requires a Collection connector")

    def _plan_evidence_requirements(self, *, requirement_limit):
        if not 1 <= requirement_limit <= self.policy.breadth_budget:
            raise ValueError("requirement_limit must be within the configured breadth budget")
        # Do not merge independent requirements to reserve a speculative repair.
        # The existing workflow already disables repair when all slots are used.
        requirement_limit = self.policy.breadth_budget
        prompt = f"""Plan a research answer to the WHOLE user request before seeing search results.
Return one QwenResearchPlan tool call with at most {requirement_limit} requirements.
Each requirement must be independently answerable from a focused evidence passage.
Keep every requested comparison member, condition, and requested dimension in the plan.
For a methods comparison, separate HOW each method works from its tradeoffs when these
need different passages. A motivation/background paragraph does not explain a mechanism.
When comparing three methods on mechanisms AND tradeoffs, use SIX requirements:
one mechanism requirement and one tradeoff requirement for EACH method. Do not put
both into one row merely because they refer to the same paper.
The mechanism row includes retrieval, generation and quality control. The tradeoff
row asks ONLY about the named method's own design choice, cost or constraint, not
limitations of earlier models that motivated the work.
Use retrieval_query to find the document; use evidence_query to locate the actual
answer passage inside it. The latter should omit the full paper title and instead
use precise technical terms for the mechanism or tradeoff if known. For tradeoffs,
look for computational cost, accuracy/efficiency choices, assumptions or limitations.
For dates, preserve the ordinary deadline, extension deadline, and exceptional situations;
answering an application method alone does not answer a deadline question.
For evaluation tools, distinguish task/deliverable definitions, evaluation criteria, and
actual available code entry points. Do not invent specific scripts or facts in the plan.
Prefer concise requirements with one main question, not bundles of unrelated clauses.
Use focused retrieval queries; the final writer will supply the requested language.
Language, formatting and clickable citations are output constraints, not separate
research questions. Do not allocate a requirement merely to proving a website is official.

Available source channels: {', '.join(self._planning_channels)}.
Selected Collection: {self.collection_context or '(none)'}.
Document catalogue (identity metadata, not evidence): {json.dumps(self._catalogue, ensure_ascii=False)}.
Allowed Web host suffixes: {', '.join(self._planning_web_hosts) or '(no additional host restriction)'}.
Respect the user's source requirements: local paper definitions use collection;
current repository or release information uses web. Never choose an unavailable channel.
When the user asks for a local paper and its implementation, include both channels.
Resolve acronyms using the catalogue's full paper title and topic before seeking code.
Keep that project identity in every Web query; similarly named repositories are not
interchangeable. For formats and entry points, seek the actual loader, script or example,
without assuming a file format or API that has not been read.
For a question about a named paper, bind source_document_id to that exact catalogue
document and use its full title in the retrieval query. Similar titles are different
documents. Catalogue titles establish identity only, never the paper's findings.
"""
        model = self.llm.bind_tools([QwenResearchPlan], parallel_tool_calls=False)
        values = []
        reason = "planner_returned_no_valid_plan"
        try:
            response = self._invoke(
                role="evidence_planner", model=model,
                messages=[SystemMessage(content=prompt), HumanMessage(content=self.query)],
                tool_definitions=(QwenResearchPlan,),
                tool_binding_options={"parallel_tool_calls": False},
                decision_stream_id="evidence_planner",
            )
            for call in getattr(response, "tool_calls", None) or []:
                if isinstance(call, Mapping) and call.get("name") == "QwenResearchPlan":
                    args = call.get("args") or {}
                    values = args.get("requirements", []) if isinstance(args, Mapping) else []
                    break
        except ResearchBudgetExhausted:
            reason = "research_model_budget_exhausted"
        except Exception as exc:
            reason = f"planner_failure:{type(exc).__name__}"
        values = values if isinstance(values, list) else []
        values = self._complete_requirement_plan(values, planning_context=prompt)
        self._writer_requirements = [dict(item) for item in values if isinstance(item, Mapping)]
        values = self._group_requirement_budget(values, requirement_limit)
        planned = []
        for item in values if isinstance(values, list) else []:
            if not isinstance(item, Mapping):
                continue
            requirement = str(item.get("requirement") or "").strip()
            query = str(item.get("retrieval_query") or "").strip()
            channel = str(item.get("source_channel") or "").strip()
            document_id = item.get("source_document_id")
            if not requirement or not query:
                continue
            if channel not in self._planning_channels:
                self._research_state.record_unresolved([requirement])
                self._event("qwen_plan_channel_rejected", requirement=requirement, source_channel=channel)
                continue
            if document_id and (channel != "collection" or document_id not in self._catalogue):
                self._research_state.record_unresolved([requirement])
                self._event("qwen_plan_document_rejected", requirement=requirement)
                continue
            if requirement not in {r[0] for r in planned}:
                planned.append((requirement, query, channel, None))
                evidence_query = str(item.get("evidence_query") or query).strip()
                self._evidence_queries[requirement] = evidence_query
                self._event("qwen_evidence_query_planned", requirement=requirement,
                            evidence_query=evidence_query, answer_focus=item.get("answer_focus"))
                if document_id:
                    self._target_documents[requirement] = document_id
                    self._event("qwen_plan_document_bound", requirement=requirement, document_id=document_id)
        if not planned:
            planned = [(self.query, self.query, channel, None) for channel in self._planning_channels]
            self._event("workflow_requirement_plan_fallback", reason=reason, requirement=self.query)
        return self._build_evidence_requirement_entries(
            accepted=planned[:requirement_limit], overflow=planned[requirement_limit:],
            plan_origin="model",
        )

    def _group_requirement_budget(self, values, limit):
        if len(values) <= limit:
            return values
        grouped = []
        mechanism_groups = {}
        for item in values:
            if not isinstance(item, Mapping):
                continue
            item = dict(item)
            document_id = item.get("source_document_id")
            key = (item.get("source_channel"), document_id)
            # Only known local identities establish that two rows concern the
            # same subject. Never merge mechanisms with goals or tradeoffs.
            mergeable = (key[0] == "collection" and document_id in self._catalogue
                         and item.get("answer_focus") == "mechanism")
            if mergeable and key in mechanism_groups:
                target = mechanism_groups[key]
                for field in ("requirement", "evidence_query"):
                    if item.get(field) and item[field] != target.get(field):
                        target[field] = str(target.get(field) or "") + "；" + item[field]
            else:
                grouped.append(item)
                if mergeable:
                    mechanism_groups[key] = item
        # If even grouped work exceeds the budget, distribute slots across
        # subjects before giving any subject its second slot.
        if len(grouped) > limit:
            subjects = {}
            for index, item in enumerate(grouped):
                key = (item.get("source_channel"), item.get("source_document_id") or f"unbound-{index}")
                subjects.setdefault(key, []).append(item)
            grouped = [rows[depth] for depth in range(max(map(len, subjects.values()), default=0))
                       for rows in subjects.values() if depth < len(rows)]
        self._event("qwen_requirement_budget_grouped", initial_count=len(values),
                    grouped_count=len(grouped), execution_limit=limit,
                    requirements=[item.get("requirement") for item in grouped])
        return grouped

    def _complete_requirement_plan(self, values, *, planning_context):
        """One pre-retrieval completion pass; never discard the original plan."""
        model = self.llm.bind_tools([QwenPlanCompletion], parallel_tool_calls=False)
        prompt = """Read the ORIGINAL user request independently, then compare it to the draft plan.
Return one QwenPlanCompletion call listing ALL missing research requirements.
Enumerate every explicitly requested subject, comparison member, dimension and condition.
A row about one subject does not cover the other named subjects. A mechanism row
does not cover a separately requested tradeoff. Output-language and citation-format
instructions are not extra research questions.
Keep existing requirements: return only additions, not a rewritten or shortened plan.
Use the permitted channels and exact catalogue document IDs from the planning context.
Do not answer the question or invent findings. No source evidence has been retrieved.
Return all omissions even if the total exceeds the execution budget; the runner will
retain overflow as unfinished work, rather than silently drop part of the user's request.
An empty list means the draft already covers the entire request.
"""
        try:
            response = self._invoke(
                role="requirement_plan_completion", model=model,
                messages=[SystemMessage(content=prompt + "\nPlanning context:\n" + planning_context),
                          HumanMessage(content="Original request:\n" + self.query
                                       + "\nDraft plan:\n" + json.dumps(values, ensure_ascii=False))],
                tool_definitions=(QwenPlanCompletion,),
                tool_binding_options={"parallel_tool_calls": False},
                decision_stream_id="requirement_plan_completion",
            )
            for call in getattr(response, "tool_calls", None) or []:
                if isinstance(call, Mapping) and call.get("name") == "QwenPlanCompletion":
                    completion = QwenPlanCompletion.model_validate(call.get("args") or {})
                    additions = [item.model_dump() for item in completion.missing_requirements]
                    self._event("qwen_requirement_plan_completed", initial_count=len(values),
                                added_count=len(additions), additions=additions)
                    return [*values, *additions]
            reason = "missing_completion_tool_call"
        except Exception as exc:
            reason = type(exc).__name__
        # Preserve useful work, but a failed completion pass cannot certify that
        # the original request was fully represented (the S2 v3 regression).
        self._research_state.record_unresolved(["Original request coverage not established: " + self.query])
        self._event("qwen_requirement_plan_completion_failed", reason=reason)
        return values

    def _fetch_ranked_source_batch(self, *, query, task, task_budget, source_channel=None):
        implementation = re.search(r"格式|字段|输入|脚本|入口|编码|format|schema|input|script|encod", task, re.I)
        if source_channel != "web" or not implementation:
            return super()._fetch_ranked_source_batch(
                query=query, task=task, task_budget=task_budget, source_channel=source_channel)
        ranked = self._static_search_source_ids(
            query=query, task=task, task_budget=task_budget, connector=self.connector, channel="web")
        selected, fetched = [], []
        limit = self.policy.max_fetches_per_research_unit

        def read(source_id):
            selected.append(source_id)
            self._read_source(source_id=source_id, task=task, task_budget=task_budget)
            if self._sources_by_id[source_id].content:
                fetched.append(source_id)

        if ranked and limit:
            read(ranked[0])
        primary = self._sources_by_id[fetched[0]] if fetched else None
        parsed = urlparse(primary.url) if primary else None
        parts = parsed.path.strip("/").split("/") if parsed else []
        repo = "/".join(parts[:2]) if len(parts) >= 2 else ""
        if parsed and parsed.hostname == "github.com" and repo and len(selected) < limit:
            # Search only within the discovered repository. Fetch returned URLs,
            # never fabricate a branch or a source-file path from a paper acronym.
            focus = self._evidence_queries.get(task, query)
            follow_query = f"site:github.com/{repo}/blob/ {focus}"
            deeper = self._static_search_source_ids(
                query=follow_query, task=task, task_budget=task_budget,
                connector=self.connector, channel="web")
            candidates = list(dict.fromkeys([*deeper, *ranked[1:]]))
            candidates = [sid for sid in candidates if (
                urlparse(self._sources_by_id[sid].url).hostname == "github.com"
                and urlparse(self._sources_by_id[sid].url).path.lower().startswith("/" + repo.lower() + "/blob/")
                and sid not in selected)]
            candidates.sort(key=lambda sid: bool(re.search(r"readme(?:\.|$)", self._sources_by_id[sid].url, re.I)))
            for sid in candidates[:limit - len(selected)]:
                read(sid)
            # Search may not index a repository's code. Continue through file
            # links preserved from pages already fetched, using the same budget.
            intent = task + " " + focus
            terms = set(re.findall(r"[a-z]{3,}", intent.lower()))
            for pattern, additions in (
                (r"格式|字段|输入|format|schema|input", ("data", "dataset", "loader", "config", "json", "tsv")),
                (r"编码|encod|embedding", ("encode", "embedding", "generate")),
                (r"评价|评估|eval", ("eval", "evaluation", "score")),
            ):
                if re.search(pattern, intent, re.I):
                    terms.update(additions)
            while len(selected) < limit:
                links = {}
                for page in list(self._sources_by_id.values()):
                    if not page.content or not page.url.lower().startswith(f"https://github.com/{repo.lower()}"):
                        continue
                    for link in re.findall(r"https://github\.com/[^\s<>\"']+", page.content):
                        path = urlparse(link).path
                        if not any(path.lower().startswith("/" + repo.lower() + "/" + kind + "/") for kind in ("blob", "tree")):
                            continue
                        name = path.rsplit("/", 1)[-1].lower()
                        score = sum(term in name for term in terms)
                        if not score or re.search(r"readme|license|changelog", name):
                            continue
                        source = self._register_discovery(DiscoveredResource(
                            resource_locator=link, title=path, snippet="File link observed on " + page.url, channel="web"), task=task)
                        if source.source_id not in selected and not source.content and not source.fetch_error:
                            links[source.source_id] = (score, link)
                        elif source.content and source.source_id not in fetched and "/blob/" in path:
                            self._read_source(source_id=source.source_id, task=task, task_budget=task_budget)
                            fetched.append(source.source_id)
                if not links:
                    break
                sid = min(links, key=lambda key: (-links[key][0], links[key][1]))
                self._event("qwen_repository_link_followed", task=task, source_id=sid, url=links[sid][1])
                read(sid)
            self._event("qwen_repository_implementation_lookup", task=task, repository=repo,
                        query=follow_query, discovered_source_ids=deeper, fetched_source_ids=fetched)
        else:
            for sid in ranked[1:limit]:
                read(sid)
        self._event("workflow_fixed_retrieval", task=task, channels=["web"],
                    required_source_channel=source_channel, selection_policy="repository_implementation",
                    selected_source_ids=selected, fetched_source_ids=fetched)
        return fetched

    def _format_evidence_candidates(self, entries, *, candidate_source_ids_by_requirement):
        entries = [replace(entry, retrieval_query=self._evidence_queries[entry.requirement])
                   if entry.requirement in self._evidence_queries else entry for entry in entries]
        filtered = {}
        for entry in entries:
            ids = candidate_source_ids_by_requirement.get(entry.requirement_id, [])
            document_id = self._target_documents.get(entry.requirement)
            if document_id:
                ids = [source_id for source_id in ids
                       if self._sources_by_id[source_id].url == f"/library/document/{document_id}"]
            filtered[entry.requirement_id] = ids
        candidates, source_ids, _ = super()._format_evidence_candidates(entries, candidate_source_ids_by_requirement=filtered)
        # A single top lexical window missed mechanism/metric details in S2/H2.
        # Add at most one disjoint candidate on each side of that window, from
        # the SAME fetched document. No refetch, concatenated quote or extra LLM.
        for entry in entries:
            initial = list(candidates[entry.requirement_id])
            for candidate in initial:
                content = self._sources_by_id[candidate.source_id].content
                for offset, segment in ((0, content[:candidate.source_start]),
                                        (candidate.source_end, content[candidate.source_end:])):
                    if not segment.strip():
                        continue
                    excerpt, start, end = _extractive_evidence_span(
                        segment, requirement=entry.requirement, retrieval_query=entry.retrieval_query,
                        maximum=self.policy.evidence_excerpt_max_chars,
                    )
                    if not excerpt.strip():
                        continue
                    group = candidates[entry.requirement_id]
                    group.append(_EvidenceCandidate(
                        candidate_key=f"{entry.requirement_id}-candidate-{len(group)+1:02d}",
                        source_id=candidate.source_id, excerpt=excerpt,
                        source_start=offset+start, source_end=offset+end,
                    ))
        sections = []
        for entry in entries:
            blocks = []
            for candidate in candidates[entry.requirement_id]:
                source = self._sources_by_id[candidate.source_id]
                blocks.append(f"- candidate key {candidate.candidate_key} [{source.source_id}] "
                              f"channel={source.channel}; title={source.title}\n"
                              f"snapshot characters {candidate.source_start}:{candidate.source_end}:\n{candidate.excerpt}")
            if blocks:
                sections.append(f"{entry.requirement_id}: {entry.requirement}\n" + "\n".join(blocks))
        return candidates, source_ids, "\n\n".join(sections) or "(none)"

    def _invoke(self, *, role, model, messages, **kwargs):
        if role == "coverage_reviewer":
            messages = [SystemMessage(content=str(messages[0].content) + """
For each requirement, first identify the concrete answer it asks for. Select a quote
that states that answer, not merely a sentence about the same topic. A method name,
motivation, or goal does not explain its operations. A benchmark's task count does
not establish its inputs and deliverables. Naming two evaluation frameworks does
not explain their criteria. A claimed advantage does not establish a design tradeoff.
For named-paper comparisons, check the candidate title belongs to that paper.
Copy enough continuous text to substantiate the answer; if none does, leave it unresolved.
"""), *messages[1:]]
        return super()._invoke(role=role, model=model, messages=messages, **kwargs)
