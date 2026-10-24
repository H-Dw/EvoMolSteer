from agenticgen.runtime.paths import skill as packaged_skill
"""Four private population analysts, then one separately audited synthesis session.

Isolation is enforced by constructing independent data/tool scopes before any
request. No analyst receives the council input, sibling outputs or filesystem tools.
"""
import concurrent.futures
import copy
import threading
import time
from pathlib import Path

import jsonschema

from agenticgen.providers.deepseek_api import call_json
from .group_agent import plan_population
from .group_tools import PopulationTools, score_summary, paired
from .contracts import validate_ticks
from .io import canonical, read, write, value_sha, sha, repository
from .learning import enabled, learning_schema, validate_memory
from .group_protocol import input_limit, budget_context
from .search_policy import upper_tick, for_config, validate_probe_quota


class BoundedCaller:
    """Share this instance across teachers: at most four simultaneous HTTP calls."""
    def __init__(self, caller=call_json, concurrency=4):
        if concurrency != 4:
            raise ValueError('The four-analyst protocol has a global HTTP limit of four')
        self.caller = caller
        self.semaphore = threading.BoundedSemaphore(concurrency)

    def __call__(self, *args, **kwargs):
        with self.semaphore:
            return self.caller(*args, **kwargs)


def private_group(spec, expected_teacher, observed_round, tick_upper=10):
    """Whitelist the evidence surface; do not expose sibling metadata in prompts/tools."""
    if spec['group_id'] not in {'G1', 'G2', 'G3', 'G4'}:
        raise ValueError('Four explicit group IDs G1..G4 are required')
    data = copy.deepcopy(spec['data'])
    if data['teacher']['teacher_id'] != expected_teacher or data['observed_round'] != observed_round:
        raise ValueError('All group results must belong to one teacher and the same completed round')
    if data.get('group_id', spec['group_id']) != spec['group_id']:
        raise ValueError('Wrong evidence group identity')
    data = {k: data[k] for k in ['teacher', 'observed_round', 'round_ids', 'chains',
        'version', 'controller_protocol', 'private_learning_history'] if k in data}
    data['group_id'] = spec['group_id']
    tools = PopulationTools(data, Path('.'))  # validate complete unique candidate identities without writing
    ticks = validate_ticks(list(spec['anchor_ticks']), tick_upper)
    if any(tools.trial(c, 'latest')['ticks'] != ticks for c in data['chains']):
        raise ValueError('One group must have one shared schedule over all its latest results')
    return data


def synthesis_schema(proposal_ids, evidence_ids, controller=False):
    text = {'type': 'string', 'minLength': 1}
    rationale = {'type': 'object', 'additionalProperties': False,
        'properties': {'proposal_id': {'enum': proposal_ids}, 'mean_reason': text, 'best_reason': text,
                       'risk_and_uncertainty': text,
                       'evidence_ids': {'type': 'array', 'minItems': 1, 'uniqueItems': True,
                                        'items': {'enum': evidence_ids}}},
        'required': ['proposal_id', 'mean_reason', 'best_reason', 'risk_and_uncertainty', 'evidence_ids']}
    props = {'comparison': text, 'failure_directions': text, 'public_memory': dict(text, maxLength=6000),
        'ranked_proposal_ids': {'type': 'array', 'minItems': len(proposal_ids), 'maxItems': len(proposal_ids),
                                'uniqueItems': True, 'items': {'enum': proposal_ids}},
        'selected': {'type': 'array', 'minItems': 4, 'maxItems': 4, 'items': rationale}}
    if controller:
        props['learning_update'] = learning_schema(evidence_ids)
        rationale['properties']['condition_reason'] = text
        rationale['properties']['functional_reason'] = text
        rationale['required'] += ['condition_reason','functional_reason']
    return {'type': 'object', 'additionalProperties': False, 'properties': props, 'required': list(props)}


def validate_synthesis(result, proposals, already_executed=(), policy=None):
    ranked = result['ranked_proposal_ids']
    selected = [x['proposal_id'] for x in result['selected']]
    if len(ranked) != len(proposals) or set(ranked) != set(proposals):
        raise ValueError('Rank every supplied proposal exactly once')
    if len(selected) != 4 or len(set(selected)) != 4:
        raise ValueError('Select four different proposals')
    if any(p not in proposals for p in selected):
        raise ValueError('A synthesizer may select only analyst proposals')
    if selected != sorted(selected, key=ranked.index):
        raise ValueError('Selected proposals must preserve the priority ranking')
    if len({tuple(proposals[p]['ticks']) for p in selected}) != 4:
        raise ValueError('Selected shared schedules must be distinct')
    cached = {tuple(t) for t in already_executed}
    if any(tuple(proposals[p]['ticks']) in cached for p in selected):
        raise ValueError('Selected schedule already executed; use cached evidence and choose an unexecuted supplied proposal')
    validate_probe_quota(proposals, selected, policy)


def compact_evidence(value):
    """Projection for synthesis only; immutable full tool records remain on disk.

    Every candidate/trial is retained. Strip molecule file/SMILES metadata from
    candidate histories, not failures or scores. Display numeric diagnostics to
    six decimals; source evidence IDs and hashes bind the original precision.
    """
    result = copy.deepcopy(value)
    if result.get('tool') == 'geometry_diagnostics':
        # These identical prose/metadata fields repeat for every example;
        # correspondence is already stated once on the enclosing tool result.
        for candidate in result['result'].get('largest_error_candidates', []):
            candidate.pop('alignment', None)
    if result.get('tool') == 'candidate_history':
        for chain in result['result']:
            for trial in chain['trials']:
                record = trial.get('record', {})
                trial['record'] = {k: v for k, v in record.items() if k in {
                    'pic50_on_rescore', 'build_success', 'connected', 'failure_reason',
                    'posebusters_pass', 'severe_clash', 'plip_recovery', 'local_rms_A'}}
    def rounded(x):
        if isinstance(x, float):
            return round(x, 6)
        if isinstance(x, list):
            return [rounded(v) for v in x]
        if isinstance(x, dict):
            return {k: rounded(v) for k, v in x.items()}
        return x
    return rounded(result)


def analyze_four_groups(specs, output, config, *, published_reflections=None,
                        caller=None, analyst_skill=None, synthesis_skill=None):
    """No GPU execution. Fail closed if any analyst fails; preserve all successes."""
    output = Path(output)
    if len(specs) != 4 or [s['group_id'] for s in specs] != ['G1', 'G2', 'G3', 'G4']:
        raise ValueError('Provide four ordered group inputs G1..G4')
    teacher = specs[0]['data']['teacher']['teacher_id']
    round_no = specs[0]['data']['observed_round']
    private = [private_group(s, teacher, round_no, upper_tick(config)) for s in specs]
    population = len(private[0]['chains'])
    identities = [[c['candidate_id'] for c in d['chains']] for d in private]
    if any(ids != identities[0] for ids in identities[1:]):
        raise ValueError('Group comparisons require the same ordered original candidate identities')
    published_reflections = copy.deepcopy(published_reflections or [])
    for memory in published_reflections:
        if memory['teacher_id'] != teacher or memory['published_after_round'] >= round_no:
            raise ValueError('Same-round or future synthesis cannot leak into isolated analysts')
    control = enabled(config)
    if control:
        validate_memory(published_reflections, teacher, private[0]['version'], round_no)
    from .committed_wave import load_commitment
    commitment = load_commitment(output, specs, private[0]['version'], upper_tick(config)) if control else None
    analyst_skill = Path(analyst_skill or packaged_skill('plip-group-analyst'))
    synthesis_skill = Path(synthesis_skill or packaged_skill('plip-group-synthesizer'))
    synthesis_output_limit = config.get('council', {}).get('max_output_tokens', 32768)
    if type(synthesis_output_limit) is not int or not 16384 <= synthesis_output_limit <= 65536:
        raise ValueError('Synthesis output budget must be an explicit integer within 16K..64K')
    binding = {'schema': 'plip-four-plus-one-v1', 'teacher_id': teacher, 'observed_round': round_no,
        'private_population_hashes': {s['group_id']: value_sha(d) for s, d in zip(specs, private)},
        'anchors': [s['anchor_ticks'] for s in specs],
        'group_history_hashes': [value_sha(s.get('group_history', [])) for s in specs],
        'published_reflections_sha256': value_sha(published_reflections),
        'analyst_skill_sha256': sha(analyst_skill), 'synthesis_skill_sha256': sha(synthesis_skill),
        'llm': config['llm'], 'synthesis_max_output_tokens': synthesis_output_limit,
        'analyst_sessions': 4, 'proposals_per_analyst': 2, 'selected_proposals': 4}
    if control:
        binding['committed_wave_sha256'] = value_sha(commitment)
    output.mkdir(parents=True, exist_ok=True)
    if (output/'binding.json').exists() and read(output/'binding.json') != binding:
        raise ValueError('Do not overwrite a council with changed private inputs, history, skills or configuration')
    if (output/'COMPLETE.json').exists():
        completed = read(output/'COMPLETE.json')
        if (completed['selection_sha256'] != sha(output/'selection.json')
                or completed['review_inputs_sha256'] != sha(output/'MASTER_REVIEW_INPUTS.json')):
            raise ValueError('Completed council checksums changed')
        for name, digest in read(output/'MASTER_REVIEW_INPUTS.json')['files_sha256'].items():
            path = (output/name).resolve()
            if not path.is_relative_to(output.resolve()) or sha(path) != digest:
                raise ValueError('Completed council evidence changed')
        return read(output/'selection.json')
    write(output/'binding.json', binding)
    caller = caller if isinstance(caller, BoundedCaller) else BoundedCaller(caller) if caller else BoundedCaller()
    reports, timings, failures = {}, {}, {}
    def worker(spec, data):
        started = time.time()
        report = plan_population(data, output/'analysts'/spec['group_id'], config,
            hypotheses=2, anchor=spec['anchor_ticks'], group_history=spec.get('group_history', []),
            published_reflections=published_reflections, skill=analyst_skill, caller=caller,
            committed_wave=commitment)
        return report, {'start': started, 'end': time.time()}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(worker, spec, data): spec['group_id'] for spec, data in zip(specs, private)}
        for future in concurrent.futures.as_completed(futures):
            gid = futures[future]
            try:
                reports[gid], timings[gid] = future.result()
            except Exception as exc:
                # Error type/message only: API layer already excludes credentials.
                failures[gid] = {'error_type': type(exc).__name__, 'message': str(exc)}
    write(output/'analyst_timing.json', timings)
    if failures:
        write(output/'analyst_failures.json', failures)
        raise RuntimeError('An isolated analyst failed; synthesis was not run; completed requests are retained')
    evidence, proposals, groups = {}, {}, []
    group_records = []
    for spec, data in zip(specs, private):
        gid = spec['group_id']
        report = reports[gid]
        tools = PopulationTools(data, output/'analysts'/gid/'tools')
        records = tools.records('latest')
        group_records.append(records)
        report_id = gid+':analysis'
        evidence[report_id] = {'population_diagnosis': report['population_diagnosis'],
                              'memory_update': report['memory_update'], 'observed_results': score_summary(records),
                              'learning_update': report.get('learning_update', [])}
        from .objective import condition_summary
        condition = condition_summary([tools.trial(c,'latest') for c in data['chains']],data['teacher']) if control else None
        evidence[report_id]['condition_quality'] = condition
        groups.append({'group_id': gid, 'anchor_ticks': spec['anchor_ticks'], 'report_evidence_id': report_id,
                       'observed_results': score_summary(records),
                       'local_proposal_ids': [gid+':'+j['id'] for j in report['jobs']]})
        for j in report['jobs']:
            key = gid+':'+j['id']
            proposals[key] = {k: j[k] for k in ['ticks', 'direction', 'exploration_degree', 'hypothesis',
                'expected_mean_effect', 'expected_best_effect', 'falsifier', 'evidence_ids']}
            proposals[key]['group_id'] = gid
            for field in ['diagnostic_alternative', 'expected_subgroup_response', 'revision_from_previous_result', 'reference_ticks', 'experiment_purpose', 'expected_condition_effect', 'expected_functional_effect', 'search_mode', 'scale_reason']:
                if field in j:
                    proposals[key][field] = j[field]
            for eid in j['evidence_ids']:
                f = output/'analysts'/gid/'tools'/(eid.removeprefix('tool:')+'.json')
                value = read(f)
                if value['evidence_id'] != eid:
                    raise ValueError('Analyst cited a mismatched tool record')
                evidence[gid+':'+eid] = value
            proposals[key]['evidence_ids'] = [gid+':'+e for e in j['evidence_ids']]
    if len({tuple(p['ticks']) for p in proposals.values()}) < 4:
        write(output/'insufficient_distinct_proposals.json', proposals)
        raise ValueError('Fewer than four distinct schedules; keep private proposals, do not invent extra hypotheses')
    for a in range(4):
        for b in range(a+1, 4):
            evidence[f'paired:G{a+1}:G{b+1}'] = paired(group_records[a], group_records[b])
    # Short aliases remove repeated 64-character hashes from schemas and every
    # proposal citation; each alias still links to the complete source evidence.
    aliases = {eid: f'E{i+1:03d}' for i, eid in enumerate(evidence)}
    full_evidence = evidence
    evidence = {aliases[eid]: {'source_id': eid, 'source_sha256': value_sha(value),
                              'value': compact_evidence(value)} for eid, value in full_evidence.items()}
    for group in groups:
        group['report_evidence_id'] = aliases[group['report_evidence_id']]
    for proposal in proposals.values():
        proposal['evidence_ids'] = [aliases[eid] for eid in proposal['evidence_ids']]
    write(output/'synthesis.full_evidence.json', full_evidence)
    payload = {'teacher_id': teacher, 'observed_round': round_no, 'groups': groups, 'proposals': proposals,
        'search_budget': budget_context(config, round_no),
        'evidence': evidence, 'earlier_published_reflections': published_reflections,
        'evidence_projection': 'All referenced tools and candidate/trial entries retained; candidate file/SMILES metadata omitted; diagnostics displayed to 6 decimals. Full precision sources archived by ID/hash.',
        'priority': config.get('agent_objective',{}).get('priority',['mean_pic50','best_pic50']), 'next_group_count': 4, 'candidates_per_group': population,
        'parent_policy': 'Selected schedules apply to the same next committed teacher TOP1 population. Analyst source branches are evidence, not restart permission.'}
    if control:
        payload['search_policy'] = for_config(config)
        payload['condition_comparison'] = config.get('condition_comparison', {})
        payload.update(version=private[0]['version'], parent_policy='Fixed new R0 before update 10 for every schedule; no incumbent restart.')
        from .learning import executed_schedules
        executed = {tuple(row['ticks']) for d in private for row in executed_schedules(d, published_reflections)}
        payload['already_executed_schedules'] = [list(t) for t in sorted(executed)]
        payload['eligible_unexecuted_proposal_ids'] = [p for p,v in proposals.items() if tuple(v['ticks']) not in executed]
        if len({tuple(proposals[p]['ticks']) for p in payload['eligible_unexecuted_proposal_ids']}) < 4:
            write(output/'insufficient_unexecuted_proposals.json', payload)
            raise ValueError('Fewer than four unexecuted proposed schedules; preserve evidence, require Agent proposal repair without inventing a scientific action')
    from .evidence_pack import pack
    expanded_evidence_sha = value_sha(payload['evidence'])
    payload['evidence'], payload['shared_evidence_blocks'] = pack(payload['evidence'])
    payload['evidence_encoding'] = {
        'protocol': 'lossless_shared_blocks_v1', 'expanded_evidence_sha256': expanded_evidence_sha,
        'instruction': 'Recursively replace each object containing only $evidence_ref with its named shared_evidence_blocks value. All referenced data is present in this request; no observation, candidate, failure or source is omitted.'}
    messages = [{'role': 'system', 'content': synthesis_skill.read_text(encoding='utf-8')},
                {'role': 'user', 'content': canonical(payload)}]
    schema = synthesis_schema(list(proposals), list(evidence), controller=control)
    size = len(canonical(messages).encode('utf-8')) + len(canonical(schema).encode('utf-8'))
    write(output/'synthesis.evidence.json', payload)
    write(output/'synthesis.budget.json', {'conservative_bytes': size, 'limit': input_limit(config),
        'source_bytes': len(canonical(full_evidence).encode('utf-8')), 'projected_bytes': len(canonical(evidence).encode('utf-8')),
        'shared_blocks':len(payload['shared_evidence_blocks']), 'expanded_evidence_sha256':expanded_evidence_sha})
    if size > input_limit(config):
        raise ValueError('Synthesis evidence exceeds the explicit input budget; evidence preserved without silent truncation')
    def validate(result):
        jsonschema.validate(result, schema)
        try:
            validate_synthesis(result, proposals, payload.get('already_executed_schedules', []), for_config(config))
        except ValueError as exc:
            raise jsonschema.ValidationError(str(exc)) from exc
    result = caller(messages, schema, output/'synthesis.json', model=config['llm']['model'],
        timeout=config['llm']['timeout'], attempts=config['llm']['attempts'],
        max_output_tokens=synthesis_output_limit, thinking=config['llm']['thinking'],
        reasoning_effort=config['llm']['reasoning_effort'], input_store=output/'synthesis_inputs', response_validator=validate)
    validate(result)
    # A separate publication document is the only cross-group memory fed back later.
    write(output/'reflection.json', result)
    memory = {'teacher_id': teacher, 'published_after_round': round_no,
              'text': result['public_memory'], 'source_reflection_sha256': sha(output/'reflection.json')}
    if control:
        def shared_condition_summary(data):
            tools=PopulationTools(data,Path('.'))
            summary=condition_summary([tools.trial(c,'latest') for c in data['chains']],data['teacher'])
            # Raw candidate evidence is archived in the council; publish population
            # statistics once rather than duplicate every coordinate error each round.
            summary.pop('individual_condition_errors')
            return summary
        memory.update(version=private[0]['version'], learning_update=result['learning_update'],
            executed=[{'ticks': s['anchor_ticks'], 'round': round_no, 'group': s['group_id'],
                       'observed': score_summary(records), 'source_sha256': value_sha(records),
                       'condition_function':shared_condition_summary(d)}
                      for s, records, d in zip(specs, group_records, private)])
        write(output/'task_knowledge.json', memory)
    write(output/'published_memory.json', memory)
    markdown = [f'# {teacher} / round {round_no}: synthesis', '', result['comparison'], '',
                '## Failed directions and uncertainty', '', result['failure_directions'], '',
                '## Ranked selections', '']
    for rank, selection in enumerate(result['selected'], 1):
        p = proposals[selection['proposal_id']]
        markdown.extend([f"{rank}. {selection['proposal_id']} — amplitudes {[x/10 for x in p['ticks']]}",
                         f"   Mean: {selection['mean_reason']} Best: {selection['best_reason']}",
                         f"   Uncertainty: {selection['risk_and_uncertainty']}", ''])
    markdown.extend(['## Memory for later rounds', '', result['public_memory'], ''])
    temp = output/'reflection.md.tmp'
    temp.write_text('\n'.join(markdown), encoding='utf-8')
    temp.replace(output/'reflection.md')
    selection = {'schema': 'plip-council-selection-v1', 'teacher_id': teacher, 'observed_round': round_no,
        'candidates_per_group': population, 'logical_llm_sessions': 5,
        'jobs': [{'priority': i+1, 'source_proposal_id': row['proposal_id'],
                  'ticks': proposals[row['proposal_id']]['ticks'],
                  'proposal': proposals[row['proposal_id']]} for i, row in enumerate(result['selected'])],
        'parent_policy': payload['parent_policy'], 'published_memory_sha256': sha(output/'published_memory.json'),
        'binding_sha256': sha(output/'binding.json'), 'formal_gpu_executed': False}
    write(output/'selection.json', selection)
    # Master GPT can enumerate every actual prompt, receipt, tool and reflection;
    # this index is evidence for review, never a fabricated review approval.
    inventory = {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob('*'))
                 if p.is_file() and p.name not in {'MASTER_REVIEW_INPUTS.json', 'COMPLETE.json'}}
    write(output/'MASTER_REVIEW_INPUTS.json', {'teacher_id': teacher, 'observed_round': round_no,
        'groups': 4, 'proposals': 8, 'selected': 4, 'files_sha256': inventory, 'reviewed_by_master': False})
    write(output/'COMPLETE.json', {'selection_sha256': sha(output/'selection.json'),
        'review_inputs_sha256': sha(output/'MASTER_REVIEW_INPUTS.json'), 'time': time.time()})
    return selection
