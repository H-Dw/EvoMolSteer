"""Locality, evidence provenance and real reward-binding regression tests."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from evomolsteer.io import read_json, write_json
from evomolsteer.local_support import bounds, candidate_sizes, nested_patches, validate_slots
from evomolsteer.ancestral_modes.reward import AncestorReward
from evomolsteer.ancestral_modes.support_mining import (
    compile_program, grouped_contrast, load_registry, prepare, propose, teacher_trends,
)
from test_ancestral_modes import fixture


def policy():
    return {'min_points': 2, 'max_fraction': .7, 'graph_depths': [1, 2],
            'spatial_radius_bond_multiples': [1.1], 'interface_fractions': [.4, .6],
            'pocket_sites': 2, 'band_bond_fraction': .25}


def source():
    p, bank = fixture()
    mode = bank['modes']['A']
    mode.update(candidate_id='F_example', template_id='A', evidence_origin='teacher', root_id='root_teacher',
                terminal_score=8., full_adjacency={str(i): [j for j in (i-1, i+1) if 0 <= j < 5] for i in range(5)})
    mode['path']['ancestor_slots'] = [0] * len(mode['path']['times'])
    pocket = pd.DataFrame({'x_A': [0., 3.], 'y_A': [-2., -1.], 'z_A': [0., 1.],
                           'residue': ['A:R:1', 'A:R:2'], 'atom_name': ['N', 'O'], 'pocket_index': [0, 1]})
    return p, bank, pocket


def test_fractional_budget_scales_without_a_twenty_atom_cap():
    assert bounds(25, {'max_fraction': .32}) == (2, 8)
    assert bounds(100, {'max_fraction': .32}) == (2, 32)
    assert bounds(25, {'max_fraction': .32, 'max_points': 5}) == (2, 5)
    assert candidate_sizes(10, 3, {'max_fraction': .7}) == [3, 4, 5, 6, 7]
    with pytest.raises(ValueError):
        bounds(5, {'max_fraction': .2})
    with pytest.raises(ValueError):
        bounds(25, {})
    with pytest.raises(ValueError):
        validate_slots([0, 1, 25], 25, {'max_fraction': .32})


def test_nested_means_subset_not_just_matching_a_cardinality():
    parent = {'template_id': 'A', 'atom_slots': list(range(15))}
    patches = [{'patch_id': str(n), 'template_id': 'A', 'atom_slots': list(range(n))} for n in [3, 7, 13]]
    patches += [{'patch_id': 'outside', 'template_id': 'A', 'atom_slots': [0, 1, 19]},
                {'patch_id': 'other_teacher', 'template_id': 'B', 'atom_slots': [0, 1, 2]}]
    assert nested_patches(patches, parent) == ['3', '7', '13']
    assert nested_patches(patches, parent, {'max_fraction': .25}, {'A': 25}) == ['3']
    with pytest.raises(ValueError, match='teacher atom count'):
        nested_patches(patches, parent, {'max_fraction': .25})


def test_proposals_have_variable_geometry_and_universal_slot_semantics():
    _, bank, pocket = source()
    regions = propose(bank['modes']['A'], pocket, policy())
    assert {len(r['atom_slots']) for r in regions} == {2, 3}
    assert all(r['locality']['fraction'] <= .7 for r in regions)
    assert len({tuple(r['atom_slots']) for r in regions}) == len(regions)
    assert regions == propose(bank['modes']['A'], pocket, policy())


def test_cluster_means_do_not_count_a_repeated_root_as_independent_evidence():
    values = np.array([[1.] * 7, [3.] * 7, [2.] * 7, [4.] * 7])
    result = grouped_contrast(values, [True, False, True, False], ['a', 'a', 'b', 'b'], 100, 42)
    repeated = grouped_contrast(np.vstack([values, np.repeat(values[:2], 30, axis=0)]),
        [True, False, True, False] + [True] * 30 + [False] * 30,
        ['a', 'a', 'b', 'b'] + ['a'] * 60, 100, 42)
    assert result['features'] == repeated['features']
    assert result['roots'] == repeated['roots'] == 2
    assert result['features']['point_error_A']['other_minus_high'] == 2.


def test_no_terminal_success_contrast_is_not_reported_as_zero_effect():
    result = grouped_contrast(np.ones((3, 7)), [False] * 3, ['a', 'b', 'c'], 10, 42)
    assert result['features']['point_error_A']['other_minus_high'] is None
    assert result['features']['point_error_A']['cluster_interval_95'] is None


def test_source_batch_is_the_bootstrap_unit_and_one_positive_batch_is_insufficient():
    result = grouped_contrast(np.arange(42).reshape(6, 7), [True, True, False, False, False, False],
        ['a', 'b', 'c', 'd', 'e', 'f'], 50, 42, [0, 0, 1, 1, 2, 2])
    assert result['bootstrap_unit'] == 'source_batch'
    assert result['high_roots'] == 2 and result['high_clusters'] == 1
    assert result['features']['point_error_A']['cluster_interval_95'] is None


def test_whole_window_teacher_trends_are_not_self_centered_constant_features():
    _, bank, pocket = source()
    region = propose(bank['modes']['A'], pocket, policy())[0]
    trend = teacher_trends(bank['modes']['A'], region, [.2, .6], 2)
    assert trend['n_times'] == 3 and trend['domain'] == [.2, .6]
    c = np.asarray(trend['fits']['actual_current']['time_derivative_coefficients'])
    assert c[0, 1] == pytest.approx(2.)
    assert not np.allclose(c, 0)


@pytest.mark.parametrize('operator', ['pointwise', 'pair_distance', 'pocket_distance'])
def test_compiled_local_features_change_real_coordinate_gradient(operator):
    p, bank, pocket = source()
    p['clock_policy'] = 'endpoint_prediction'
    p.update(analyst_sha256='old_analyst', designer_sha256='old_designer', evidence_bundle_sha256='old_evidence')
    region = propose(bank['modes']['A'], pocket, policy())[0]
    region['local_support_evidence'] = 'evidence/measured.json'
    program = compile_program(p, region, bank, policy(), operator)
    assert program['clock_policy'] == p['clock_policy'] and program['dose'] == p['dose']
    assert program['mode_specs'][0]['atom_slots'] == region['atom_slots']
    assert program['local_feature_binding']['active_terms'] == ['registered_' + operator]
    assert program['typed_weight'] == program['charge_weight'] == 0
    assert not {'analyst_sha256', 'designer_sha256', 'evidence_bundle_sha256'}.intersection(program)
    reward = AncestorReward(program, bank)
    torch.manual_seed(42)
    x = torch.randn(1, 5, 3, dtype=torch.double, requires_grad=True)
    mask = torch.ones(1, 5, dtype=torch.bool)
    logits = torch.zeros(1, 5, 3, dtype=torch.double)

    def value(z):
        # A coupled endpoint map exercises the actual coordinate pullback, not a detached target gradient.
        endpoint = z * .7 + z.mean(1, keepdim=True) * .2
        return reward.evaluate(z, endpoint, logits, logits, mask, .3)[0].sum()

    g = torch.autograd.grad(value(x), x)[0]
    a, b = x.detach().clone(), x.detach().clone()
    a[0, 0, 0] += 1e-6
    b[0, 0, 0] -= 1e-6
    assert g.norm() > 0
    assert float(g[0, 0, 0]) == pytest.approx(float((value(a) - value(b)) / 2e-6), abs=1e-6)
    wide = copy.deepcopy(program)
    wide['mode_specs'][0]['atom_slots'] = [0, 1, 2, 3]
    with pytest.raises(ValueError, match='policy'):
        AncestorReward(wide, bank)


def test_reward_has_no_implicit_twenty_point_limit():
    p, bank, _ = source()
    mode = bank['modes']['A']
    coords = [[float(i), .1 * i, 0.] for i in range(70)]
    mode.update(coords_A=coords, terminal_coords_A=coords, native_elements=['C'] * 70)
    mode['path']['coords_A'] = [coords] * 3
    p['mode_specs'][0]['atom_slots'] = list(range(21))
    p['local_support_policy'] = {'max_fraction': .32}
    AncestorReward(p, bank)


def test_research_catalog_contains_measured_small_supports_and_bound_operators(tmp_path):
    from evomolsteer.ancestral_modes.programs import actions
    p, bank, pocket = source()
    p.update(program_id='parent', clock_policy='endpoint_prediction')
    regions = propose(bank['modes']['A'], pocket, policy())
    for r in regions:
        r.update(shortlisted=True, local_support_evidence='evidence/' + r['region_id'] + '.json')
    state = {'best': {'program': p}, 'reference': {'program': p}, 'rounds': [], 'dose_reference_steps': []}
    master = {'learning_window': [.2, .6], 'local_support_registry': 'registry', 'local_support_policy': policy(),
              'local_support_phases': ['endpoint_clock'], 'local_support_operators': ['pointwise', 'pair_distance']}
    catalog = actions(tmp_path, master, bank, regions, state, 2)
    bound = [a for a in catalog if 'local_feature_binding' in a['program']]
    assert {len(a['program']['mode_specs'][0]['atom_slots']) for a in bound} == {2, 3}
    assert {a['program']['operator'] for a in bound} == {'pointwise', 'pair_distance'}
    assert all(a['program']['clock_policy'] == 'endpoint_prediction' for a in bound)
    assert all(a['program']['local_feature_binding']['region_id'] == a['region_ids'][0] for a in bound)
    from evomolsteer.ancestral_modes.programs import validate
    tampered = copy.deepcopy(bound[0]['program'])
    tampered['local_feature_binding']['atom_slots'] = [3, 4]
    with pytest.raises(ValueError, match='binding'):
        validate(tampered, bank)


def test_compact_registry_roundtrip_and_stale_input_rejection(tmp_path):
    p, bank, pocket = source()
    paths = {name: tmp_path / name for name in ('bank.json', 'atoms.csv', 'membership.csv', 'pocket.csv', 'config.json')}
    write_json(paths['bank.json'], bank)
    pocket.to_csv(paths['pocket.csv'], index=False)
    rows, membership = [], []
    for i in range(7):
        node = 'teacher' if i == 0 else 'node_' + str(i)
        root = 'root_teacher' if i < 2 else 'root_' + str(i // 2)
        membership.append({'node_id': node, 'root_id': root, 'quality_tail': i % 2 == 0})
        for j, xyz in enumerate(bank['modes']['A']['terminal_coords_A']):
            rows.append(dict(node_id=node, atom_index=j, x_A=xyz[0] + .1 * i,
                             y_A=xyz[1], z_A=xyz[2]))
    pd.DataFrame(rows).to_csv(paths['atoms.csv'], index=False)
    pd.DataFrame(membership).to_csv(paths['membership.csv'], index=False)
    config = {'local_support_policy': policy(), 'positive_label': 'quality_tail',
              'exclude_teacher_root': True, 'bootstrap_repeats': 30, 'seed': 42,
              'learning_window': [.2, .6], 'trend_degree': 2, 'max_regions_per_template': 3}
    write_json(paths['config.json'], config)
    args = [paths[n] for n in ('bank.json', 'atoms.csv', 'membership.csv', 'pocket.csv', 'config.json')]
    manifest = prepare(*args, tmp_path / 'out')
    again = prepare(*args, tmp_path / 'out')
    assert manifest == again
    stored, regions = load_registry(tmp_path / 'out', bank)
    assert len(stored['shortlisted_regions']) == 3
    ev = read_json(tmp_path / 'out' / regions[0]['local_support_evidence'])
    assert ev['terminal_contrast']['n_high'] + ev['terminal_contrast']['n_other'] == 5
    assert not list((tmp_path / 'out').rglob('*.parquet'))
    config['local_support_policy']['max_fraction'] = .5
    write_json(paths['config.json'], config)
    with pytest.raises(ValueError, match='changed'):
        prepare(*args, tmp_path / 'out')


def test_evidence_cache_binds_support_window_and_geometry_parameters(tmp_path):
    from evomolsteer.ancestral_modes.evidence_cache import cache_path
    _, bank, pocket = source()
    mode = bank['modes']['A']
    mode.update(batch=0, terminal_slot=0)
    base = tmp_path / 'dataset'
    source_path = base / 'single' / 'batch_000' / 'trajectory.h5'
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(b'fixture input fingerprint')
    write_json(base / 'frame_batch_000.json', {'target_com': [0, 0, 0]})
    write_json(base / 'config.json', {'coord_scale': 1.})
    region = propose(mode, pocket, policy())[0]
    master = {'campaign_study': 'study', 'learning_window': [.2, .6], 'field_sigma_A': 1.2}
    _, a = cache_path(tmp_path, master, mode, region, source_path, 'current', Path(__file__))
    changed = copy.deepcopy(region)
    changed['atom_slots'] = list(reversed(changed['atom_slots']))
    _, b = cache_path(tmp_path, master, mode, changed, source_path, 'current', Path(__file__))
    master['field_sigma_A'] = 2.
    _, c = cache_path(tmp_path, master, mode, region, source_path, 'current', Path(__file__))
    master['learning_window'] = [.3, .6]
    _, d = cache_path(tmp_path, master, mode, region, source_path, 'current', Path(__file__))
    assert len({a, b, c, d}) == 4


@pytest.mark.parametrize('strict',[False,True])
@pytest.mark.parametrize('alias',[False,True])
def test_simulated_api_transport_delivers_local_evidence_and_compiles_same_support(tmp_path, monkeypatch,strict,alias):
    """Transport/schema test only; no claim about actual LLM reasoning or quality."""
    import evomolsteer.ancestral_modes.agents as agents
    from evomolsteer.ancestral_modes.programs import program_delta, numerical_identity
    parent, bank, pocket = source()
    parent.update(program_id='R02_test', clock_policy='endpoint_prediction')
    bank['limits'] = ['Synthetic transport fixture; not real API evidence']
    region = next(r for r in propose(bank['modes']['A'], pocket, policy()) if len(r['atom_slots']) == 3)
    region['local_support_evidence'] = 'evidence/local.json'
    write_json(tmp_path / region['local_support_evidence'], {'region_id': region['region_id'], 'fixture': True})
    program = compile_program(parent, region, bank, policy(), 'pair_distance')
    action = {'action_id': 'A0', 'parent_id': 'R02_test', 'region_ids': [region['region_id']],
              'phase': 'endpoint_clock', 'reason': 'local support test', 'program': program,
              'numeric_identity': numerical_identity(program, bank), 'authoritative_program_delta': program_delta(parent, program)}
    record = {'program': parent, 'metrics': {}, 'response': {}}
    state = {'reference': record, 'best': record, 'rounds': [], 'original': {'best': 8.}}
    repo = Path(__file__).resolve().parents[1]
    master = {'campaign_study': str(tmp_path / 'study'), 'learning_window': [.2, .6],
              'analyst_skill': 'skills/ancestral-mode-analyst/SKILL.md',
              'designer_skill': 'skills/ancestral-mode-designer/SKILL.md',
              'local_support_registry': str(tmp_path), 'local_support_policy': policy()}
    if strict:master.update(strict_feature_handoff=True,parent_feedback={'fixture_transport_only':True})
    calls = []

    def fake_call(messages, schema, output, **kwargs):
        output = Path(output)
        calls.append((output.name, messages, schema))
        if output.name.startswith('Analyst.query'):
            tool_schema = schema['properties']['requests']['items']['properties']['tool']
            tool = tool_schema.get('const', 'ancestor' if output.name == 'Analyst.query0.json' else 'controller_response')
            response = {'requests': [{'tool': tool, 'region_id': region['region_id']}], 'purpose': 'fixture request'}
        elif output.name == 'Analyst.json':
            response = {k: 'fixture explanation' for k in schema['properties']}
            response.update(group_id='G000', parent_program_id='R02_test', primary_operator='pair_distance',
                evidence_refs=[region['region_id'] + '/local_support_evidence', 'parents/R02_test/outcomes'],
                counterevidence_refs=['source/limits'], active_chemistry_modules=[],
                observable_contract=agents.CLOCKS['endpoint_prediction'])
            if strict:
                response.update(controlled_feature='pair_error_A')
                response['evidence_refs'].append('parents/R02_test/response')
            if alias:response['evidence_refs']=[region['region_id']+'/local_support_evidence',region['region_id']+'/controller_response']
        else:
            response = {'option_id': 'A0', 'hypothesis_group': 'G000', 'parent_program_id': 'R02_test',
                        'evidence_refs': [region['region_id'] + '/local_support_evidence'],
                        'design_grounds': 'fixture binding only', 'falsifier': 'fixture falsifier'}
            if strict:response['evidence_refs']+=['parents/R02_test/outcomes','parents/R02_test/response']
            if alias:response['evidence_refs']=[region['region_id']+'/local_support_evidence',region['region_id']+'/controller_response']
        import jsonschema
        jsonschema.validate(response, schema)
        if kwargs.get('response_validator'):
            if strict and output.name=='Designer.response.json':
                with pytest.raises(jsonschema.ValidationError,match='actual parent evidence'):
                    kwargs['response_validator'](dict(response,evidence_refs=[region['region_id']+'/local_support_evidence']))
            kwargs['response_validator'](response)
        write_json(output, response)
        write_json(output.with_suffix('.request.json'), {'fixture_transport_only': True})
        return response

    monkeypatch.setattr(agents, 'call_json', fake_call)
    monkeypatch.setattr(agents, 'factual_review', lambda *args, **kwargs: {'violations': []})
    selected, _, _ = agents.ask(repo, master, tmp_path / 'calls', bank, [region], state, [action], 'endpoint_clock')
    assert selected['mode_specs'][0]['atom_slots'] == region['atom_slots']
    assert selected['operator'] == 'pair_distance'
    queries = [c for c in calls if c[0].startswith('Analyst.query')]
    assert len(queries) == 3
    assert queries[-1][2]['properties']['requests']['items']['properties']['tool']['const'] == 'local_support_evidence'
    designer = next(c for c in calls if c[0] == 'Designer.response.json')
    delivered = json.loads(designer[1][1]['content'])
    assert region['region_id'] + '/local_support_evidence' in delivered['shared_evidence']
    assert set(designer[2]['properties']['evidence_refs']['items']['enum'])==set(delivered['shared_evidence'])
