import copy
import json
import random

import jsonschema
import pytest

from evomolsteer.adaptive_schedule.group_tools import (
    distribution, score_summary, paired, local_geometry, interval_summary, PopulationTools, export_legacy_population)
from evomolsteer.adaptive_schedule.group_agent import (
    plan_population, capacity_count, validate_hypotheses, verified_double_directions)
from evomolsteer.adaptive_schedule.io import defaults, read, write
from evomolsteer.adaptive_schedule.grid import CHECKPOINT_STEPS


def record(score, valid=True):
    return {'pic50_on_rescore': score, 'build_success': valid, 'connected': valid, 'failure_reason': ''}


def population():
    steps = [{'step': s, 'loss': 1., 'clip_factor': .5, 'max_atom_guidance_A': .025} for s in range(100)]
    nodes = {str(s): {'local_endpoint_A': [[1., 0., 0.]], 'local_current_A': [[2., 0., 0.]]} for s in CHECKPOINT_STEPS}
    return {'teacher': {'teacher_id': 't', 'source_record': {'pic50_on_rescore': 7.8},
                        'coords_A': [[0., 0., 0.]], 'core_atoms': [0]},
        'observed_round': 1, 'round_ids': ['r00', 'r01'],
        'chains': [{'candidate_id': f'c{i:03d}', 'batch': i//20, 'slot': i%20,
                    'incumbent_id': 'r01', 'latest_id': 'r01', 'trials': [
            {'id': f'r{r:02d}', 'ticks': [5]*4, 'record': record(3+i/100+r/10),
             'nodes': nodes, 'intervals': interval_summary(steps), 'source_branch': f'r{r:02d}'} for r in range(2)]}
                   for i in range(100)]}


def hypotheses(n, evidence):
    ticks = [[5 + (d if j == interval else 0) for j in range(4)] for interval in range(4) for d in [-1, 1]]
    return {'kind': 'hypotheses', 'population_diagnosis': 'test', 'memory_update': 'test',
            'hypotheses': [dict(id=f'H{i+1}', ticks=ticks[i], evidence_ids=[evidence],
                **{k: 'test' for k in ['direction', 'exploration_degree', 'hypothesis',
                     'expected_mean_effect', 'expected_best_effect', 'falsifier']}) for i in range(n)]}


def test_numerical_tools_keep_failures_and_fixed_top_ten():
    assert score_summary([record(5)]*9)['top10_mean'] is None
    result = score_summary([record(5)]*10+[record(None, False)])
    assert result['attempts'] == 11 and result['valid'] == 10 and result['top10_mean'] == 5
    change = paired([record(3), record(7), record(None, False)], [record(4), record(None, False), record(5)])
    assert change['common_valid'] == 1 and change['delta']['mean'] == 1
    assert change['valid_to_invalid'] == change['invalid_to_valid'] == 1
    assert distribution([1, 2, 3, 4])['median'] == 2.5


def test_geometry_uses_angstroms_and_does_not_align_away_translation():
    g = local_geometry([[3, 4, 0]], [[0, 0, 0]])
    assert g['rms_A'] == 5
    assert g['pseudo_huber_delta1'] == pytest.approx(26**.5-1)
    with pytest.raises(ValueError):
        local_geometry([[float('nan'), 0, 0]], [[0, 0, 0]])


def test_registry_is_readonly_bounded_and_reproducible(tmp_path):
    tools = PopulationTools(population(), tmp_path)
    before = random.getstate()
    a = tools.call('geometry_diagnostics', {'selection': 'latest', 'step': 90, 'state': 'endpoint'})
    b = tools.call('geometry_diagnostics', {'selection': 'latest', 'step': 90, 'state': 'endpoint'})
    assert a == b and a['result']['rms_A']['mean'] == 1
    assert random.getstate() == before
    with pytest.raises(ValueError, match='Unregistered'):
        tools.call('shell', {'command': 'echo unsafe'})
    with pytest.raises(jsonschema.ValidationError):
        tools.call('population_summary', {'selection': 'latest', 'path': '/tmp/forged.json'})
    result = tools.call('subgroup_response', {'a': 'r00', 'b': 'r01'})['result']
    assert [g['common_valid'] for g in result['groups']] == [25]*4
    assert tools.call('interval_diagnostics', {'selection': 'latest'})['result'][1]['clipped_fraction'] == 1


def test_one_teacher_session_requests_tools_then_proposes_four_shared_schedules(tmp_path):
    calls = []
    def caller(messages, schema, output, **kwargs):
        calls.append(messages)
        if len(calls) == 1:
            response = {'kind': 'tools', 'reason': 'inspect late geometry and early clipping', 'calls': [
                {'name': 'geometry_diagnostics', 'arguments': {'selection': 'latest', 'step': 90, 'state': 'endpoint'}},
                {'name': 'interval_diagnostics', 'arguments': {'selection': 'latest'}}]}
        else:
            supplied = json.loads(messages[-1]['content'])['tool_results']
            response = hypotheses(4, supplied[0]['evidence_id'])
        kwargs['response_validator'](response)
        return response
    result = plan_population(population(), tmp_path, defaults(), caller=caller)
    assert len(calls) == 2  # no per-candidate API calls
    contract = json.loads(calls[0][1]['content'])['action_contract']
    assert contract['max_changed_intervals'] == 2
    assert contract['default_max_absolute_tick_change'] == 1
    assert contract['tick_to_amplitude'] == .1
    assert contract['ids_in_order'] == ['H1', 'H2', 'H3', 'H4']
    assert len(result['jobs']) == 4 and result['candidate_attempts_if_executed'] == 400
    assert result['planning_only'] and not result['gpu_executed']
    assert len(result['parents']) == 100
    assert len({j['parent_population_sha256'] for j in result['jobs']}) == 1
    assert all(j['candidates'] == 100 for j in result['jobs'])
    assert result['jobs'][0]['earliest_common_restart'] == 10
    assert result['jobs'][2]['earliest_common_restart'] == 26
    assert read(tmp_path/'hypotheses.json') == result


def test_duplicate_hypotheses_and_large_jumps_rejected():
    h = hypotheses(4, 'e')
    h['hypotheses'][1]['ticks'] = h['hypotheses'][0]['ticks']
    with pytest.raises(ValueError, match='Duplicate'):
        validate_hypotheses(h, [5]*4, 4, {'e'})
    h = hypotheses(4, 'e')
    h['hypotheses'][0]['ticks'] = [3, 5, 5, 5]
    with pytest.raises(ValueError, match='One tick'):
        validate_hypotheses(h, [5]*4, 4, {'e'})
    validate_hypotheses(h, [5]*4, 4, {'e'}, {0: -1})


def test_two_ticks_need_verified_group_improvements_not_llm_claim():
    history = [{'ticks': [3, 5, 5, 5]}, {'ticks': [4, 5, 5, 5], 'attempts': 100,
               'paired_common_valid': 100, 'paired_mean_delta': .1, 'validity_delta': 0},
               {'ticks': [5]*4, 'attempts': 100, 'paired_common_valid': 100, 'paired_mean_delta': .1, 'validity_delta': 0}]
    assert verified_double_directions(history, [5]*4) == {0: 1}
    history[-1]['validity_delta'] = -1
    assert verified_double_directions(history, [5]*4) == {}


def test_complete_capacity_receipt_must_match_both_teachers(tmp_path):
    write(tmp_path/'COMPLETE.json', {'selected_hypotheses_per_wave': 4, 'both_teachers_passed': True})
    write(tmp_path/'manifest.json', {'teacher_ids': ['t1', 't2']})
    write(tmp_path/'results.json', [{'teacher': 't1', 'batch_size': 400, 'attempts': 400, 'status': 'ok'}])
    with pytest.raises(ValueError, match='Incomplete'):
        capacity_count(tmp_path, ['t1', 't2'])
    rows = read(tmp_path/'results.json')
    rows.append(dict(rows[0], teacher='t2'))
    write(tmp_path/'results.json', rows)
    assert capacity_count(tmp_path, ['t1', 't2']) == 4


def test_input_change_cannot_reuse_an_existing_group_session(tmp_path):
    def caller(messages, schema, output, **kwargs):
        e = json.loads(messages[1]['content'])['evidence'][0]['evidence_id']
        return hypotheses(4, e)
    data = population()
    plan_population(data, tmp_path, defaults(), caller=caller)
    data = copy.deepcopy(data)
    data['chains'][0]['trials'][0]['record']['pic50_on_rescore'] = 8
    with pytest.raises(ValueError, match='Do not reuse'):
        plan_population(data, tmp_path, defaults(), caller=caller)


def test_export_uses_retained_restart_sources_and_rejects_incomplete_population(tmp_path):
    root = tmp_path/'source'
    t = population()['teacher']
    write(root/'teachers.json', [t])
    for b in range(5):
        branch = f'branches/b{b}'
        write(root/branch/'manifest.json', {'schedules': [[5]*4]*20, 'parents': [None]*20, 'start_step': 0})
        write(root/branch/'nodes.json', {str(s): {'step': s, 'local_endpoint_A': [[[1, 0, 0]]]*20} for s in CHECKPOINT_STEPS})
        write(root/branch/'steps.json', [{'step': s, 'clip_factor': [1]*20} for s in range(100)])
        folder = root/'versions/v1/t'/f'batch_{b:03d}'
        write(folder/'histories.json', [{'slot': i, 'best_id': 'r00', 'history': [
            {'id': 'r00', 'branch': branch, 'ticks': [5]*4, 'record': record(4)}]} for i in range(20)])
        retained = f'retained/b{b}'
        write(folder/'retention.json', {'best_ids': ['r00']*20, 'restart_refs': [retained]*20})
        (root/retained).mkdir(parents=True)
        for s in [10, 26, 51, 76]:
            (root/retained/f'checkpoint_{s:03d}.pt.gz').write_bytes(b'placeholder; existence-only export')
    data = export_legacy_population(root, 't', 'v1', 0, tmp_path/'population.json')
    assert len(data['chains']) == 100
    assert all(c['restartable'] and c['restart_branch'].startswith('retained/') for c in data['chains'])
    with pytest.raises(ValueError, match='completed'):
        export_legacy_population(root, 't', 'v1', 1, tmp_path/'incomplete.json')
