import copy
import json

from evomolsteer.lineage_retrieval.audit_packets import selector_audit_node
from evomolsteer.lineage_retrieval import orchestrate as o


def example():
    mapping = {f'C{i:012x}': f'private_{i}' for i in range(7)}
    cards = [{'key': key, 'now': {'step': 10, 'current_radius_gyration_A': i + 10.,
        'online_pic50': 8. + i / 10, 'forecast_residue_contact_footprint': ['VAL:53', 'VAL:66']},
        'history': [{'step': s, 'current_radius_gyration_A': i + float(s),
            'online_pic50': 7. + s / 10, 'forecast_residue_contact_footprint': ['VAL:66', 'VAL:53']}
            for s in range(11)]} for i, key in enumerate(mapping.values())]
    answer = {'ranking': [{'candidate_id': cid, 'rank': i + 1,
        'evidence': 'Compare C000000000005 online_pic50 and current geometry.', 'counterevidence': '',
        'observations': [{'feature': 'current_radius_gyration_A', 'step': 10}]}
        for i, cid in enumerate(list(mapping)[:5])], 'applied_clauses': []}
    calls = [{'tool': 'get_candidate', 'args': {'candidate_id': 'C000000000006'}}]
    return {'step': 10, 'cards': cards}, mapping, answer, calls


def test_named_competitors_and_unlisted_geometry_have_full_past():
    pool, mapping, answer, calls = example()
    node = selector_audit_node(pool, mapping, answer, calls)
    choice = node['choices'][0]
    assert 'online_pic50' in choice['history_columns']
    assert 'forecast_residue_contact_footprint' in choice['history_columns']
    assert len(choice['history_rows']) == 11
    compared = {r['candidate_id']: r for r in node['compared_candidates']}
    assert len(compared['C000000000005']['history_rows']) == 11
    assert 'history_rows' not in compared['C000000000006']  # only queried current; no prose history claim
    ranges = node['available_population_current_scalar_ranges']['current_radius_gyration_A']
    assert ranges['count'] == 7 and ranges['min'] == 10. and ranges['max'] == 16.
    assert 'private_' not in json.dumps(node)
    changed = copy.deepcopy(pool)
    changed['cards'][0]['history'].append({'step': 11, 'current_radius_gyration_A': 99999.})
    assert selector_audit_node(changed, mapping, answer, calls) == node


def test_large_audit_pairs_split_without_omitting_any_node(tmp_path, monkeypatch):
    packet = {'scientist_descriptions': [{'interval': i, 'text': 'x' * 440_000} for i in range(1, 11)],
        'scientist_background_evidence': [], 'selector_nodes': [{'step': s} for s in o.STEPS],
        'same_time_population_quantiles': [{'step': s} for s in range(51)]}
    from evomolsteer.lineage_retrieval import runner
    calls = []
    def fake(job, role, prompt, part, schema, model, effort):
        calls.append(part['audit_slice'])
        assert schema['properties']['selector_assessments']['maxItems'] == 1
        result = {'passes': True, 'scientist_evidence_truth': True, 'scientist_clause_applied': True,
            'selector_assessments': [{'step': v['step'], 'evidence_truth': True, 'clause_applied': True}
                for v in part['selector_nodes']], 'unsupported_claims': [], 'scientist_assessment': 'Reviewed',
            'instruction_effect_assessment': 'Source-grounded', 'failure_analysis': 'None'}
        o.write(job / 'answer.json', result)
        return result
    monkeypatch.setattr(runner, 'run_job', fake)
    result = o.run_audit_chunks(tmp_path, 'Audit source.', packet)
    assert o.audit_passes(result)
    assert len(calls) == 10
    assert sorted(s for c in calls for s in c['selector_steps']) == o.STEPS
    assert len(o.read(tmp_path / 'aggregation.json')['slice_response_sha256']) == 10
