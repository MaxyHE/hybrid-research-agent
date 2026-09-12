"""Opt-in located-evidence handoff; research policy and default writer unchanged."""
import json
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import create_model

from .runtime import OdrBaselineRunner

WRITER_GUIDANCE = '''Use the original evidence to check the findings, rather than treating
the findings as verified facts. Preserve the source's conditions and scope. Separate
documented behavior from your proposed implementation: a citation for an API's existence
does not establish that your combination of APIs is valid. For recommendations, give the
smallest useful design and explain the relevant trade-off. Use concrete code or commands
when their usage is supported by the supplied evidence; otherwise explain the algorithm
in prose or clearly identified pseudocode instead of inventing executable syntax. Do not
claim code was tested unless execution evidence is supplied. Missing information in these
excerpts does not establish that the source lacks it. Keep the requested length and focus;
avoid an extra disclaimer section or unrelated implementation options.
When asked for a starting point, recommend one minimal path using existing components.
Add an alternative only when it addresses a distinct requirement in the question.
Keep optional optimizers, persistence layers and extra review loops out of the initial
implementation unless the user's requirements make them necessary.
Keep the subject and scope of comparisons explicit: related-work limitations describe
the earlier methods, not automatically the method introduced by the source. Preserve
contrast clauses and exceptions when shortening a finding; do not turn a historical
baseline's weakness into the new method's conclusion. Reconcile mixed findings against
their original passages before placing them under a method's limitations.
'''


class LocatedHandoffRunner(OdrBaselineRunner):
    """Select evidence at the existing compression step, not in another agent."""

    def __init__(self, writer_guidance=False, source_fact_handoff=False, attributed_handoff=False, **kwargs):
        super().__init__(**kwargs)
        self._located_packets = {}
        self._writer_guidance = writer_guidance
        self._source_fact_handoff = source_fact_handoff
        self._writer_source_handles = writer_guidance
        self._attributed_handoff = attributed_handoff

    def _invoke(self, *, role, model, messages, **kwargs):
        if role == 'writer' and self._writer_guidance:
            original = getattr(self, 'original_user_query', None)
            language_instruction = ''
            if original is not None:
                language_instruction = '\nThe original user request below controls output language and length. Honor any explicit requested language; otherwise use its language. Internal source-scope instructions and research notes do not choose the output language.\nOriginal user request:\n' + original
            if self._attributed_handoff:
                language_instruction += '\nFindings may carry subject, claim_role and conditions. These preserve attribution, not a quality score. Organize claims under their named subject. Prior-work or general-background claims are context, not limitations or results of the focal method. Retain the conditions when stating a comparison. Check these labels against the attached original passages; unlabelled older findings still require the same attribution care.'
            messages = [messages[0].model_copy(update={
                'content': messages[0].content + '\n' + WRITER_GUIDANCE + language_instruction
            }), *messages[1:]]
        return super()._invoke(role=role, model=model, messages=messages, **kwargs)

    def _task_passages(self, task):
        passages = {}
        for source in list(self._sources_by_id.values()):
            if source.content is None or task not in source.fetched_for:
                continue
            text = source.content
            maximum = self.policy.source_view_max_chars
            if len(text) <= maximum:
                spans = [(0, len(text))]
            else:
                width = maximum // 3
                middle = len(text) // 2 - width // 2
                spans = [(0, width), (middle, middle + width), (len(text)-width, len(text))]
            for event in list(self._trace):
                d = event.data
                if event.kind == 'source_range_read' and d['task'] == task and d['source_id'] == source.source_id:
                    spans.append((d['start'], d['end']))
            merged = []
            for start, end in sorted(spans):
                if merged and start <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
                else:
                    merged.append((start, end))
            for left, right in merged:
                for start in range(left, right, 1200):
                    end = min(start + 1200, right)
                    key = f'{source.source_id}:{start}:{end}'
                    passages[key] = {'source_id': source.source_id, 'url': source.url,
                                     'channel': source.channel, 'start': start, 'end': end,
                                     'text': text[start:end]}
        return passages

    def _compress_research_trail(self, *, task, working_trail):
        passages = self._task_passages(task)
        if not passages:
            self._located_packets[task] = {'findings': [], 'evidence': {}}
            return 'No fetched evidence available for this task.'
        # Expose the actual selection set in the tool schema, not a free-form ID.
        fields = {'finding': (str, ...), 'passage_ids': (list[Literal[tuple(passages)]], ...)}
        if self._source_fact_handoff:
            fields['kind'] = (Literal['source_fact', 'proposal'], ...)
        if self._attributed_handoff:
            fields.update(subject=(str, ...),
                          claim_role=(Literal['method_description', 'method_result', 'method_limitation', 'prior_work', 'general_background'], ...),
                          conditions=(str, ...))
        finding_type = create_model('LocatedFindingChoice', **fields)
        result_type = create_model('LocatedFindings', findings=(list[finding_type], ...))
        model = self.llm.bind_tools([result_type], tool_choice='auto')
        prompt = 'Prepare concise findings answering the ORIGINAL question and focused task, using supplied passages. Preserve logical conditions, exceptions, dates and the source/object concerned. Select passage IDs supporting each finding; select adjacent passages when needed. The program attaches the original text and position, so do not write quotations or offsets yourself. Separate recommendations from source facts. Use LocatedFindings to return your findings.'
        if self._source_fact_handoff:
            prompt += ' Mark each finding kind=source_fact only when the original passages support the statement, including its conditions and scope. A documented recommendation is a source_fact when attributed to its source. Your own suggested design, inferred API combination or unsupported extrapolation is kind=proposal; proposals are archived separately and not passed on as evidence-backed facts. Prioritize a compact set of useful source facts; do not generate proposals merely to fill a category. Missing information in excerpts does not establish that the source lacks it. The writer will formulate the final recommendation from source facts and original passages.'
            prompt += ' Separate historical/related-work claims from the source method\'s own findings and limitations. Split statements about different methods into separate findings, naming the subject and retaining the comparison direction, conditions and contrast. Do not group a prior approach\'s weakness under the current method\'s limitations.'
        if self._attributed_handoff:
            prompt += ' For each finding, give subject (the actual method/system or group being described, not merely the paper containing the sentence), claim_role, and conditions (dataset, time, comparison baseline, assumption or exception necessary for the statement; empty if none). Use prior_work for earlier approaches discussed by the source, general_background for broad contextual statements, method_result for the focal method\'s demonstrated result, and method_limitation only for a limitation of that named focal method. Split claims with different subjects or roles instead of attaching one label to a mixed paragraph. The finding text itself must retain the subject and conditions, not rely only on metadata.'
        response = self._invoke(
            role='compression', model=model,
            messages=[SystemMessage(content=prompt),
                      HumanMessage(content=json.dumps({'question': self.query, 'task': task,
                                                       'passages': passages}, ensure_ascii=False))],
            tool_definitions=[result_type],
            tool_binding_options={'tool_choice': 'auto'},
        )
        call = next(c for c in response.tool_calls if c['name'] == 'LocatedFindings')
        args = self._expand_adjacent_choices(call['args'], passages)
        selected = result_type.model_validate(args)
        findings, evidence = [], {}
        for finding in selected.findings:
            if self._source_fact_handoff and finding.kind == 'proposal':
                self._event('located_proposal_archived', task=task, proposal=finding.finding,
                            passage_ids=finding.passage_ids)
                continue
            for key in finding.passage_ids:
                evidence[key] = passages[key]
            item = {'finding': finding.finding, 'evidence_ids': finding.passage_ids}
            if self._attributed_handoff:
                item.update(subject=finding.subject, claim_role=finding.claim_role,
                            conditions=finding.conditions)
            findings.append(item)
        self._located_packets[task] = {'findings': findings, 'evidence': evidence}
        self._event('located_handoff_selected', task=task, findings=findings,
                    evidence_ids=list(evidence), evidence_characters=sum(len(e['text']) for e in evidence.values()))
        # Supervisor schedules remaining work; Writer receives original evidence below.
        return json.dumps({'findings': findings}, ensure_ascii=False)

    def _expand_adjacent_choices(self, args, passages):
        """A combined range may name several complete adjacent supplied pieces."""
        findings = []
        for finding in args['findings']:
            ids = []
            for key in finding['passage_ids']:
                if key in passages:
                    ids.append(key)
                    continue
                source_id, left, right = key.split(':')
                left, right = int(left), int(right)
                pieces = sorted(((p['start'], p['end'], pid) for pid, p in passages.items()
                                 if p['source_id'] == source_id and left <= p['start'] < p['end'] <= right))
                cursor, expanded = left, []
                for start, end, pid in pieces:
                    if start != cursor:
                        break
                    expanded.append(pid)
                    cursor = end
                if not expanded or cursor != right:
                    raise ValueError(f'Chosen range is not a union of supplied adjacent passages: {key}')
                ids.extend(expanded)
                self._event('located_adjacent_passages_resolved', requested=key, passage_ids=expanded)
            findings.append({**finding, 'passage_ids': ids})
        return {'findings': findings}

    def _write_report(self, *, tasks, notes, unresolved_tasks, writer_source_ids=None):
        tasks = list(tasks)
        findings, evidence = [], {}
        for task in tasks:
            packet = self._located_packets.get(task, {'findings': [], 'evidence': {}})
            findings.extend(packet['findings'])
            evidence.update(packet['evidence'])
        self._event('located_writer_packet_created', finding_count=len(findings),
                    evidence_ids=list(evidence), evidence_characters=sum(len(e['text']) for e in evidence.values()))
        return super()._write_report(tasks=tasks,
            notes=[json.dumps({'findings': findings, 'evidence': evidence}, ensure_ascii=False)],
            unresolved_tasks=unresolved_tasks, writer_source_ids=writer_source_ids)
