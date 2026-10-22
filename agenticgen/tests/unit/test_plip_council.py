import copy
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from test_plip_group_agent import population, hypotheses
from agenticgen.adaptive_schedule.council import (
    analyze_four_groups, private_group, validate_synthesis, BoundedCaller, compact_evidence)
from agenticgen.adaptive_schedule.io import defaults, read
from agenticgen.adaptive_schedule.group_tools import PopulationTools


def groups():
    result = []
    for n in range(4):
        data = population()
        data['group_id'] = f'G{n+1}'
        anchor = [5]*4
        anchor[n] = 4
        for c in data['chains']:
            c['trials'][-1]['ticks'] = anchor
            c['trials'][-1]['record']['pic50_on_rescore'] += n/10
            c['trials'][-1]['record']['private_canary'] = f'private_group_{n+1}'
        data['untrusted_sibling_container'] = 'must_never_reach_tools'
        result.append({'group_id': data['group_id'], 'data': data, 'anchor_ticks': anchor,
                       'group_history': [{'private_canary': f'private_group_{n+1}', 'ticks': anchor}]})
    return result


def test_four_private_analysts_overlap_then_one_synthesis_selects_four(tmp_path):
    barrier = threading.Barrier(4)
    calls, finished = [], set()
    lock = threading.Lock()
    def caller(messages, schema, output, **kwargs):
        payload = json.loads(messages[-1]['content'])
        if 'groups' not in payload:
            gid = payload['group_id']
            serialized = json.dumps(messages)
            assert f'private_group_{gid[-1]}' in serialized
            assert all(f'private_group_{x}' not in serialized for x in range(1, 5) if str(x) != gid[-1])
            assert 'must_never_reach_tools' not in serialized
            barrier.wait(timeout=10)
            response = hypotheses(2, payload['evidence'][0]['evidence_id'])
            for h, direction in zip(response['hypotheses'], [-1, 1]):
                h['ticks'] = list(payload['common_anchor_ticks'])
                h['ticks'][(int(gid[-1])) % 4] += direction
            with lock:
                finished.add(gid)
        else:
            assert finished == {'G1', 'G2', 'G3', 'G4'}
            assert kwargs['max_output_tokens'] == 32768
            ids = list(payload['proposals'])
            selected = ids[::2]
            response = {'comparison': 'compare all four', 'failure_directions': 'test',
                'public_memory': 'published after private analysis', 'ranked_proposal_ids': ids,
                'selected': [{'proposal_id': p, 'mean_reason': 'mean', 'best_reason': 'best',
                    'risk_and_uncertainty': 'test', 'evidence_ids': [next(k for k, v in payload['evidence'].items()
                        if v['source_id'] == p.split(':')[0]+':analysis')]} for p in selected]}
        kwargs['response_validator'](response)
        with lock:
            calls.append(str(output))
        return response
    result = analyze_four_groups(groups(), tmp_path, defaults(), caller=caller)
    assert len(calls) == 5
    assert len(result['jobs']) == 4 and result['logical_llm_sessions'] == 5
    assert result['formal_gpu_executed'] is False
    assert (tmp_path/'reflection.md').exists()
    assert read(tmp_path/'published_memory.json')['published_after_round'] == 1
    index = read(tmp_path/'MASTER_REVIEW_INPUTS.json')
    assert not index['reviewed_by_master']
    assert any('tools/' in name for name in index['files_sha256'])
    assert analyze_four_groups(groups(), tmp_path, defaults(), caller=caller) == result
    assert len(calls) == 5  # Complete and hash-verified councils do not call again.
    (tmp_path/'reflection.md').write_text('tampered', encoding='utf-8')
    with pytest.raises(ValueError, match='evidence changed'):
        analyze_four_groups(groups(), tmp_path, defaults(), caller=caller)


def test_private_scope_and_tool_registry_cannot_read_siblings(tmp_path):
    spec = groups()[0]
    private = private_group(spec, 't', 1)
    assert 'untrusted_sibling_container' not in private
    tools = PopulationTools(private, tmp_path)
    history = tools.call('candidate_history', {'candidate_ids': ['c000']})
    serialized = json.dumps(history)
    assert 'private_group_1' in serialized and 'private_group_2' not in serialized
    spec['data']['chains'][2]['trials'][-1]['ticks'] = [5]*4
    with pytest.raises(ValueError, match='shared schedule'):
        private_group(spec, 't', 1)


def test_same_round_reflection_and_wrong_identity_rejected_before_api(tmp_path):
    with pytest.raises(ValueError, match='leak'):
        analyze_four_groups(groups(), tmp_path, defaults(), published_reflections=[
            {'teacher_id': 't', 'published_after_round': 1, 'text': 'forbidden'}])
    specs = groups()
    specs[3]['data']['chains'].reverse()
    with pytest.raises(ValueError, match='ordered original'):
        analyze_four_groups(specs, tmp_path, defaults())


def test_one_failed_analyst_prevents_synthesis_and_keeps_successes(tmp_path):
    def caller(messages, schema, output, **kwargs):
        payload = json.loads(messages[-1]['content'])
        assert 'groups' not in payload
        if payload['group_id'] == 'G4':
            raise RuntimeError('diagnostic failure')
        response = hypotheses(2, payload['evidence'][0]['evidence_id'])
        for h, direction in zip(response['hypotheses'], [-1, 1]):
            h['ticks'] = list(payload['common_anchor_ticks'])
            h['ticks'][3] += direction
        return response
    with pytest.raises(RuntimeError, match='synthesis was not run'):
        analyze_four_groups(groups(), tmp_path, defaults(), caller=caller)
    assert (tmp_path/'analysts/G1/hypotheses.json').exists()
    assert 'G4' in read(tmp_path/'analyst_failures.json')
    assert not (tmp_path/'synthesis.json').exists()


def test_synthesis_rejects_duplicates_invented_or_misordered_proposals():
    proposals = {str(i): {'ticks': [i, 5, 5, 5]} for i in range(8)}
    result = {'ranked_proposal_ids': list(proposals), 'selected': [{'proposal_id': str(i)} for i in range(4)]}
    validate_synthesis(result, proposals)
    bad = copy.deepcopy(result)
    bad['selected'].reverse()
    with pytest.raises(ValueError, match='ranking'):
        validate_synthesis(bad, proposals)
    bad['selected'][0]['proposal_id'] = 'invented'
    with pytest.raises(ValueError, match='only analyst'):
        validate_synthesis(bad, proposals)
    proposals['3']['ticks'] = proposals['0']['ticks']
    with pytest.raises(ValueError, match='distinct'):
        validate_synthesis(result, proposals)


def test_missing_final_geometry_is_not_advertised_as_available(tmp_path):
    data = population()
    for c in data['chains']:
        for t in c['trials']:
            t['nodes'].pop('100', None)
    tools = PopulationTools(data, tmp_path)
    assert 100 not in tools.registry['geometry_diagnostics'][1]['properties']['step']['enum']


def test_global_http_limit_is_shared_between_teachers():
    active = 0
    peak = 0
    lock = threading.Lock()
    gate = threading.Barrier(4)
    def call():
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        gate.wait(timeout=10)
        with lock:
            active -= 1
    bounded = BoundedCaller(call)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(bounded) for _ in range(8)]
        for future in futures:
            future.result()
    assert peak == 4


def test_synthesis_projection_retains_every_failure_and_trial_without_mutating_sources():
    source = {'tool': 'candidate_history', 'result': [{'candidate_id': 'c0', 'trials': [
        {'id': 'r00', 'ticks': [5]*4, 'record': {'pic50_on_rescore': 4.123456789,
            'build_success': True, 'connected': True, 'smiles': 'CC', 'failure_reason': ''}},
        {'id': 'r01', 'ticks': [4, 5, 5, 5], 'record': {'pic50_on_rescore': None,
            'build_success': False, 'connected': False, 'failure_reason': 'invalid'}}]}]}
    result = compact_evidence(source)
    assert len(result['result'][0]['trials']) == 2
    assert result['result'][0]['trials'][1]['record']['failure_reason'] == 'invalid'
    assert result['result'][0]['trials'][0]['record']['pic50_on_rescore'] == 4.123457
    assert source['result'][0]['trials'][0]['record']['smiles'] == 'CC'
    assert 'smiles' not in result['result'][0]['trials'][0]['record']
