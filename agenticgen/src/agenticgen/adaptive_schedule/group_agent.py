from agenticgen.runtime.paths import skill as packaged_skill
"""One teacher-level LLM session with deterministic tools and multiple hypotheses.

Application-level JSON tool calls reuse the audited DeepSeek transport, request
deduplication and receipts. This planner does not launch or resume GPU sampling.
"""
from pathlib import Path

import jsonschema

from agenticgen.providers.deepseek_api import call_json
from .contracts import validate_ticks, earliest
from .group_tools import PopulationTools
from .io import canonical, read, write, value_sha, repository
from .learning import (enabled, learning_schema, verified_large_steps, validate_memory,
                       executed_schedules, validate_controller_hypotheses)
from .grid import description
from .group_protocol import input_limit, budget_context
from .search_policy import for_config, upper_tick


def capacity_count(path, expected_teachers):
    folder = Path(path)
    complete, manifest = read(folder/'COMPLETE.json'), read(folder/'manifest.json')
    n = complete.get('selected_hypotheses_per_wave')
    if n not in {8, 4, 3, 2} or not complete.get('both_teachers_passed'):
        raise ValueError('No measured common hypothesis capacity')
    if set(manifest['teacher_ids']) != set(expected_teachers):
        raise ValueError('Capacity evidence belongs to a different teacher cohort')
    rows = [r for r in read(folder/'results.json') if r['batch_size'] == n*100]
    if (set(r['teacher'] for r in rows) != set(expected_teachers)
            or any(r['status'] != 'ok' or r['attempts'] != n*100 for r in rows)):
        raise ValueError('Incomplete two-teacher capacity evidence')
    return n


def response_schema(tools, count, evidence_ids, final_only=False, decision_audit=False, controller=False, tick_upper=10,
                    action_anchor=None, local_domain=None):
    text = {'type': 'string', 'minLength': 1}
    hypothesis = {'type': 'object', 'additionalProperties': False,
        'properties': {'id': {'type': 'string', 'pattern': '^H[1-8]$'},
            'ticks': {'type': 'array', 'minItems': 4, 'maxItems': 4,
                      'items': {'type': 'integer', 'minimum': 0, 'maximum': tick_upper}},
            'direction': text, 'exploration_degree': text, 'hypothesis': text,
            'expected_mean_effect': text, 'expected_best_effect': text, 'falsifier': text,
            'evidence_ids': {'type': 'array', 'minItems': 1, 'uniqueItems': True, 'items': {'enum': sorted(evidence_ids)}}},
        'required': ['id', 'ticks', 'direction', 'exploration_degree', 'hypothesis',
                     'expected_mean_effect', 'expected_best_effect', 'falsifier', 'evidence_ids']}
    final = {'type': 'object', 'additionalProperties': False,
        'properties': {'kind': {'const': 'hypotheses'}, 'population_diagnosis': text, 'memory_update': text,
            'hypotheses': {'type': 'array', 'minItems': count, 'maxItems': count, 'items': hypothesis}},
        'required': ['kind', 'population_diagnosis', 'memory_update', 'hypotheses']}
    if decision_audit:
        for key in ['diagnostic_alternative', 'expected_subgroup_response', 'revision_from_previous_result']:
            hypothesis['properties'][key] = text
            hypothesis['required'].append(key)
    if controller:
        final['properties']['learning_update'] = learning_schema(evidence_ids)
        final['required'].append('learning_update')
        hypothesis['properties']['reference_ticks'] = ({'const': list(action_anchor)} if action_anchor is not None
                                                       else hypothesis['properties']['ticks'])
        if local_domain is not None and tick_upper <= 10:
            hypothesis['properties']['ticks'] = {'type':'array', 'minItems':4, 'maxItems':4,
                'prefixItems':[{'type':'integer','enum':values} for values in local_domain]}
        hypothesis['properties']['experiment_purpose'] = {'enum': ['direction_probe','exploit','bracket_peak','interaction_probe','conflict_diagnosis']}
        hypothesis['properties']['expected_condition_effect'] = text
        hypothesis['properties']['expected_functional_effect'] = text
        hypothesis['properties']['search_mode'] = {'enum': ['local','range_probe']}
        hypothesis['properties']['scale_reason'] = text
        hypothesis['required'] += ['reference_ticks','experiment_purpose','expected_condition_effect','expected_functional_effect','search_mode','scale_reason']
    if final_only:
        return final
    calls = [{'type': 'object', 'additionalProperties': False,
              'properties': {'name': {'const': name}, 'arguments': schema},
              'required': ['name', 'arguments']} for name, (_, schema) in tools.registry.items()]
    return {'oneOf': [final, {'type': 'object', 'additionalProperties': False,
        'properties': {'kind': {'const': 'tools'}, 'reason': text,
                      'calls': {'type': 'array', 'minItems': 1, 'maxItems': 4, 'items': {'oneOf': calls}}},
        'required': ['kind', 'reason', 'calls']}]}


def validate_hypotheses(message, anchor, count, evidence_ids, double_directions=None):
    validate_ticks(anchor)
    double_directions = double_directions or {}
    hypotheses = message['hypotheses']
    if len(hypotheses) != count or [h['id'] for h in hypotheses] != [f'H{i+1}' for i in range(count)]:
        raise ValueError('Ordered, unique hypothesis IDs and the configured count are required')
    seen = set()
    for h in hypotheses:
        ticks = validate_ticks(h['ticks'])
        earliest(anchor, ticks)
        if tuple(ticks) in seen:
            raise ValueError('Duplicate schedules do not create independent hypotheses')
        seen.add(tuple(ticks))
        delta = [(i, b-a) for i, (a, b) in enumerate(zip(anchor, ticks)) if a != b]
        if len(delta) > 2:
            raise ValueError('Change at most two intervals relative to the common anchor')
        for i, d in delta:
            if abs(d) > 2 or (abs(d) == 2 and double_directions.get(i) != (1 if d > 0 else -1)):
                raise ValueError('One tick by default; two require verified consecutive group-mean improvement')
        if not h['evidence_ids'] or not set(h['evidence_ids']).issubset(evidence_ids):
            raise ValueError('Each hypothesis must cite numerical evidence actually supplied')


def verified_double_directions(group_history, anchor):
    """Two successive complete, common-valid-mean improving anchor edits; no LLM self-certification."""
    if len(group_history) < 3 or group_history[-1]['ticks'] != anchor:
        return {}
    out = {}
    recent = group_history[-3:]
    for i in range(4):
        changes = [recent[j+1]['ticks'][i]-recent[j]['ticks'][i] for j in range(2)]
        if (all(changes) and changes[0]*changes[1] > 0 and all(
            x.get('attempts') == 100 and x.get('paired_common_valid', 0) > 0
            and x.get('paired_mean_delta', 0) > 0 and x.get('validity_delta', -1) >= 0
            for x in recent[1:])):
            out[i] = 1 if changes[0] > 0 else -1
    return out


def plan_population(data, output, config, *, hypotheses=4, anchor=(5, 5, 5, 5),
                    group_history=None, skill=None, caller=call_json, max_tool_turns=4,
                    published_reflections=None):
    if hypotheses not in {8, 4, 3, 2} or not 1 <= max_tool_turns <= 6:
        raise ValueError('Unsupported hypothesis count or tool-turn budget')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    population = len(data['chains'])
    anchor = validate_ticks(list(anchor), upper_tick(config))
    group_history = group_history or []
    published_reflections = published_reflections or []
    for memory in published_reflections:
        if (memory['teacher_id'] != data['teacher']['teacher_id']
                or memory['published_after_round'] >= data['observed_round']):
            raise ValueError('Only already-published earlier-round reflection is visible to a group analyst')
    control = enabled(config)
    if control:
        validate_memory(published_reflections, data['teacher']['teacher_id'], data['version'], data['observed_round'])
        data = dict(data, published_reflections=published_reflections)
    tools = PopulationTools(data, output/'tools')
    skill = Path(skill or packaged_skill('plip-group-analyst' if control else 'plip-population-optimizer'))
    instructions = skill.read_text(encoding='utf-8')
    double = verified_double_directions(group_history, anchor)
    initial = [tools.call('history_summary', {}),
               tools.call('paired_comparison', {'a': 'latest', 'b': 'incumbent'})]
    large = verified_large_steps(group_history, anchor) if control else {}
    from .learning import action_domain
    local_domain = action_domain(anchor, large, for_config(config)) if control else None
    executed = executed_schedules(data, published_reflections) if control else []
    if control:
        initial += [tools.call('teacher_condition', {}), tools.call('task_knowledge', {}),
                    tools.call('execution_history', {}), tools.call('condition_quality', {'selection':'latest'})]
    evidence = {x['evidence_id'] for x in initial}
    common = {'teacher_id': data['teacher']['teacher_id'], 'population_sha256': tools.digest,
        'group_id': data.get('group_id'),
        'teacher_pic50': data['teacher']['source_record']['pic50_on_rescore'],
        'core_atoms': len(data['teacher']['core_atoms']), 'candidates': population,
        'observed_round': data['observed_round'], 'hypotheses_required': hypotheses,
        'search_budget': budget_context(config, data['observed_round']),
        'common_anchor_ticks': anchor, 'verified_double_tick_directions': double,
        'action_contract': {'tick_to_amplitude': 0.1, 'tick_bounds_inclusive': [0, 10],
            'max_changed_intervals': 2, 'default_max_absolute_tick_change': 1,
            'double_tick_rule': 'A two-tick change is allowed only in a listed verified_double_tick_directions interval and direction.',
            'all_hypotheses_distinct': True, 'no_unchanged_anchor': True,
            'ids_in_order': [f'H{i+1}' for i in range(hypotheses)]},
        'group_history': group_history, 'published_reflections': published_reflections,
        'visibility': 'This group only, its own history and earlier published reflections; no sibling current-round data',
        'intervals': description(), 'tools': tools.catalog(),
        'evidence': initial, 'reward_frozen': True, 'max_tool_turns': max_tool_turns,
        'parent_policy': 'All hypotheses use the same pre-round per-candidate incumbent snapshot; all 100 identities retained',
        'objective_priority': config.get('agent_objective',{}).get('priority',['mean_pic50','best_pic50']),
        'protocol': 'Application-level registered JSON tools; request kind=tools or return kind=hypotheses'}
    if control:
        common.update(version=data['version'], controller_role='adaptive conditional learning controller',
            parent_policy='Every proposal restarts the same new R0 before update 10. Incumbent is output only.',
            executed_schedules=executed, verified_large_step_directions=large)
        common['continuous_objectives'] = config['agent_objective']
        common['action_contract'].update(default_max_absolute_tick_change=2,
            double_tick_rule='Two ticks allowed by default; three only in verified_large_step_directions.',
            no_executed_schedule=True, initial_step_ticks=2)
        common['action_contract'].update(tick_bounds_inclusive=[0,upper_tick(config)],
            search_policy=for_config(config),
            range_probe_rule='Only after range expansion; at most one of final four. Max two changed intervals. Fine grid remains 0.1. Explain actual jump vs coarse default; all other proposals are local.')
        common['condition_comparison'] = config.get('condition_comparison', {})
        common['action_contract'].update(action_anchor_ticks=anchor,
            allowed_local_ticks_by_interval=local_domain,
            reference_ticks_must_equal=anchor,
            reference_semantics='Action deltas use this current group anchor. Fixed R0 before update 10 is the physical restart only. Historical causal comparisons do not replace the action anchor.')
    messages = [{'role': 'system', 'content': instructions}, {'role': 'user', 'content': canonical(common)}]
    binding = {'population_sha256': tools.digest, 'common_sha256': value_sha(common),
               'skill_sha256': value_sha(instructions), 'llm': config['llm']}
    if (output/'binding.json').exists() and read(output/'binding.json') != binding:
        raise ValueError('Do not reuse an LLM session after its input, skill or config changed')
    write(output/'binding.json', binding)
    for turn in range(max_tool_turns+1):
        schema = response_schema(tools, hypotheses, evidence, final_only=turn == max_tool_turns,
                                 decision_audit=config.get('council', {}).get('decision_audit', False), controller=control, tick_upper=upper_tick(config),
                                 action_anchor=anchor if control else None, local_domain=local_domain)
        estimated_bytes = len(canonical(messages).encode('utf-8')) + len(canonical(schema).encode('utf-8'))
        limit = input_limit(config)
        if estimated_bytes > limit:
            raise ValueError(f'Group input/schema exceeds conservative byte budget: {estimated_bytes}>{limit}; request fewer detailed candidates')
        def validate(message):
            jsonschema.validate(message, schema)
            if message['kind'] == 'hypotheses':
                try:
                    if control:
                        validate_controller_hypotheses(message, anchor, hypotheses, evidence, large, executed, for_config(config))
                    else:
                        validate_hypotheses(message, anchor, hypotheses, evidence, double)
                except ValueError as exc:
                    raise jsonschema.ValidationError(str(exc)) from exc
        response = caller(messages, schema, output/f'turn_{turn:02d}.json',
            model=config['llm']['model'], timeout=config['llm']['timeout'], attempts=config['llm']['attempts'],
            max_output_tokens=config['llm']['max_output_tokens'], input_store=output/'llm_inputs',
            thinking=config['llm']['thinking'], reasoning_effort=config['llm']['reasoning_effort'], response_validator=validate)
        validate(response)
        write(output/f'turn_{turn:02d}.budget.json', {'conservative_bytes': estimated_bytes, 'limit': limit})
        if response['kind'] == 'hypotheses':
            parents = [{"candidate_id": c['candidate_id'], 'parent_id': c['incumbent_id'],
                        'source_branch': c.get('restart_branch', tools.trial(c, 'incumbent').get('source_branch')),
                        'restartable_at_export': c.get('restartable', False),
                        'source_slot': c.get('slot'), 'source_batch': c.get('batch'),
                        'parent_ticks': tools.trial(c, 'incumbent')['ticks']} for c in data['chains']]
            jobs = []
            for h in response['hypotheses']:
                starts = [10]*population if control else [earliest(p['parent_ticks'], h['ticks']) if p['parent_ticks'] != h['ticks'] else 100 for p in parents]
                jobs.append({**h, 'candidates': population, 'ticks_for_all_candidates': h['ticks'],
                             'earliest_common_restart': min(starts), 'per_candidate_restart_steps': starts,
                             'parent_population_sha256': value_sha(parents)})
            result = {'schema': 'plip-group-hypotheses-v1', **binding,
                'teacher_id': common['teacher_id'], 'observed_round': data['observed_round'],
                'common_anchor_ticks': anchor, 'population_diagnosis': response['population_diagnosis'],
                'memory_update': response['memory_update'], 'parents': parents, 'jobs': jobs,
                'group_id': data.get('group_id'),
                'candidate_attempts_if_executed': population*hypotheses, 'llm_turns': turn+1,
                'planning_only': True, 'gpu_executed': False,
                'execution_requires': 'validated 100-slot mixed-parent recovery, frozen RNG/wave mapping, quality-before-prune transaction'}
            if control:
                result.update(learning_update=response['learning_update'], version=data['version'],
                    parent_policy=common['parent_policy'], execution_requires='fixed new R0 step10, validated common-random physical batch, quality-before-prune')
                write(output/'learning_update.json', {'version': data['version'], 'teacher_id': common['teacher_id'],
                    'observed_round': data['observed_round'], 'group_id': data.get('group_id'),
                    'learning_update': response['learning_update'], 'evidence_population_sha256': tools.digest})
            write(output/'hypotheses.json', result)
            return result
        results = [tools.call(c['name'], c['arguments']) for c in response['calls']]
        evidence.update(r['evidence_id'] for r in results)
        messages = messages + [{'role': 'assistant', 'content': canonical(response)},
                               {'role': 'user', 'content': canonical({'tool_results': results})}]
    raise AssertionError('The last permitted turn must provide hypotheses')
