"""Common ancestry-clock selection and immutable-reference data contracts."""
import copy
import gzip
import json

import numpy as np
import pandas as pd
import pytest

from evomolsteer.continuous.regional_common_depth import (
    declared_depths, learn_common_depth_rules, prepare_inputs,
)
from evomolsteer.continuous.regional_reference import (
    REQUIRED_FILES, evaluate_xyz, synthesize_reference,
)
from evomolsteer.io import digest, write_json


def statistics(spec=None, negative_batches=None):
    spec = spec or {2: [0], 3: [0, 1], 5: [0]}
    negative_batches = negative_batches or {}
    effects, batches, fitted = [], [], {'depths': {}}
    for depth, regions in spec.items():
        functions = {}
        for region in regions:
            for axis, value in zip('xyz', (1., .4, -.2)):
                feature = f'region_{region:02d}_internal_displacement_{axis}'
                effects.append({'depth': depth, 'metric': 'adjusted_covariance', 'feature': feature,
                                'whole_window_mean': value, 'CI_low': value - .05, 'CI_high': value + .05,
                                'q': .01 if axis == 'x' else .2, 'independent_batches_available': 14})
                functions['adjusted_covariance/' + feature] = {
                    'degree': 0, 'coefficients': [value],
                    'observed_time_start': depth / 100., 'observed_time_end': .49}
                for batch in range(14):
                    factor = -.1 if batch < negative_batches.get(depth, 0) else 1.
                    batches.append({'batch': batch, 'depth': depth, 'metric': 'adjusted_covariance',
                                    'feature': feature, 'whole_window_value': value * factor})
        fitted['depths'][str(depth)] = {'functions': functions}
    catalog = {'landmarks_A': [[0., 0., 0.], [2., 1., 0.]], 'region_width_A': 1.}
    return pd.DataFrame(effects), pd.DataFrame(batches), fitted, catalog


def reference():
    times = [0., .01, .02, .03, .1, .2, .49]
    return {'window': [0., .5], 'times': times, 'schema_version': 'affinity-endpoint-library-1.0',
            'branch_mutation': {'observation': 'preserve'}, 'original_priors': [.3, .7],
            'frames': [
                {'time': time, 'teacher_endpoint_A': [[[0., 0., 0.], [2., 0., 0.]]],
                 'teacher_scores': [8.], 'teacher_batches': [0],
                 'teacher_contrast_direction_unit': [[[0., 1., 0.], [0., -1., 0.]]],
                 'teacher_contrast_confidence': [.02], 'teacher_contrast_atom_weight': [[.8, 1.2]],
                 'teacher_contrast_provenance': [{'raw_direction_RMS_A': .09,
                                                 'lower_immediate_parent_slots': [2], 'lag2_observed': True}]}
                for time in times]}


def test_one_global_depth_uses_eligible_count_and_records_every_rejected_depth():
    rules = learn_common_depth_rules(*statistics(), [0., .5], depths=(2, 3, 5, 8, 13))
    assert rules['common_depth'] == 3
    assert {region['depth'] for region in rules['selected_regions']} == {3}
    assert [region['region'] for region in rules['selected_regions']] == [0, 1]
    assert [row['eligible_region_count'] for row in rules['depth_evaluations']] == [1, 2, 1, 0, 0]
    assert [row['status'] for row in rules['depth_evaluations']] == [
        'eligible_not_selected', 'selected', 'eligible_not_selected', 'no_observations', 'no_observations']
    assert not rules['new_significance_tests'] and not rules['selection_uses_final_generated_labels']


def test_equal_counts_use_mean_loo_positive_fraction_then_shallower_depth():
    args = statistics({2: [0], 3: [0]}, negative_batches={2: 4})
    rules = learn_common_depth_rules(*args, [0., .5], depths=(2, 3))
    assert rules['common_depth'] == 3
    assert rules['depth_evaluations'][0]['mean_loo_positive_fraction'] == pytest.approx(10 / 14)
    tie = learn_common_depth_rules(*statistics({5: [0], 3: [0]}), [0., .5], depths=(5, 3))
    assert tie['common_depth'] == 3


def test_complete_xyz_frozen_q_and_independent_batch_units_are_retained():
    effects, batches, fitted, catalog = statistics()
    snapshot = effects.copy(deep=True)
    rules = learn_common_depth_rules(effects, batches, fitted, catalog, [0., .5], depths=(2, 3, 5))
    pd.testing.assert_frame_equal(effects, snapshot)
    for region in rules['selected_regions']:
        assert region['independent_batches'] == region['loo_observed_batches'] == 14
        assert region['component_q'] == [.01, .2, .2]
        assert set(region['functions_xyz']) == set('xyz')
        assert region['functions_xyz']['z']['coefficients'] == [-.2]
        assert evaluate_xyz(region['functions_xyz'], .02) is None
        np.testing.assert_allclose(evaluate_xyz(region['functions_xyz'], .03), [1., .4, -.2])


def test_no_missing_initial_fields_and_no_original_teacher_facts_change():
    rules = learn_common_depth_rules(*statistics(), [0., .5], depths=(2, 3, 5))
    original = reference(); snapshot = copy.deepcopy(original)
    transformed, coverage = synthesize_reference(original, rules)
    assert original == snapshot
    for before, after in zip(original['frames'], transformed['frames']):
        for key, value in before.items():
            if key not in {'teacher_contrast_direction_unit', 'teacher_contrast_confidence'}:
                assert after[key] == value
        assert after['teacher_original_branch_direction_unit'] == before['teacher_contrast_direction_unit']
        assert after['teacher_original_branch_confidence'] == before['teacher_contrast_confidence']
        if before['time'] < .03:
            assert not np.asarray(after['teacher_contrast_direction_unit']).any()
            assert after['teacher_contrast_confidence'] == [0.]
        else:
            direction = np.asarray(after['teacher_contrast_direction_unit'])
            np.testing.assert_allclose(direction.mean(1), 0, atol=1e-15)
    assert transformed['branch_mutation'] == original['branch_mutation']
    assert transformed['original_priors'] == original['original_priors']
    assert coverage['new_active_teachers'] == 4
    assert 'rotational components are not projected out' in rules['pose_policy']


@pytest.mark.parametrize('empty_mode', ['no_significance', 'no_observations'])
def test_no_qualifying_common_depth_produces_exact_zero_fields(empty_mode):
    effects, batches, fitted, catalog = statistics()
    if empty_mode == 'no_significance': effects['q'] = 1.
    else: effects, batches = effects.iloc[:0], batches.iloc[:0]
    rules = learn_common_depth_rules(effects, batches, fitted, catalog, [0., .5], depths=(2, 3, 5))
    assert rules['common_depth'] is None and rules['selected_regions'] == []
    transformed, coverage = synthesize_reference(reference(), rules)
    assert coverage['new_active_teachers'] == 0
    assert not np.asarray([f['teacher_contrast_direction_unit'] for f in transformed['frames']]).any()


def test_insufficient_batches_and_absent_xyz_are_not_silently_filled():
    effects, batches, fitted, catalog = statistics({2: [0]})
    effects['independent_batches_available'] = 13
    rules = learn_common_depth_rules(effects, batches, fitted, catalog, [0., .5], depths=(2,))
    assert rules['common_depth'] is None
    effects['independent_batches_available'] = 14
    fitted['depths']['2']['functions'].pop('adjusted_covariance/region_00_internal_displacement_z')
    with pytest.raises(ValueError, match='complete XYZ'):
        learn_common_depth_rules(effects, batches, fitted, catalog, [0., .5], depths=(2,))


@pytest.mark.parametrize('depths', [(), (2, 2), (False,), (2.5,), (-1,)])
def test_depth_declaration_is_explicit_positive_and_unique(depths):
    with pytest.raises(ValueError): declared_depths(depths)


def test_source_receipts_bind_full_xyz_statistics_and_original_teacher_parent(tmp_path):
    """Synthetic file fixtures test verification; they are not claimed live runs."""
    mining = tmp_path / 'mining'; mining.mkdir()
    script = tmp_path / 'source.py'; script.write_text('source')
    source_input = tmp_path / 'source_input.json'; source_input.write_text('{}')
    for name in REQUIRED_FILES:
        if name != 'manifest.json': (mining / name).write_text('{}')
    manifest = {'window': [0., .5], 'reference_sha256': digest(source_input),
                'files': {name: {'sha256': digest(mining / name)} for name in REQUIRED_FILES if name != 'manifest.json'}}
    write_json(mining / 'manifest.json', manifest)
    source = {'schema_version': 'multi-depth-live-execution-receipt-1.0', 'tool_id': 'multi_depth_mutation',
              'command': ['python', str(script)], 'returncode': 0, 'error': None,
              'inputs_and_code_unchanged': True, 'complete_outputs': True,
              'input_files': {str(source_input): digest(source_input)}, 'code_files': {str(script): digest(script)},
              'output_files': {str(mining / name): digest(mining / name) for name in REQUIRED_FILES}}
    source_path = tmp_path / 'multi.receipt.json'; write_json(source_path, source)
    ref = tmp_path / 'branch.json.gz'
    ref.write_bytes(gzip.compress(json.dumps({'window': [0., .5], 'branch_mutation': {'parent_reference_sha256': digest(source_input)}}).encode()))
    branch = copy.deepcopy(source)
    branch.update(schema_version='flowcompat-supplemental-tool-receipt-2.0', tool_id='branch_mutation',
                  inputs_unchanged=True, source_unchanged=True, output_files={str(ref): digest(ref)})
    branch_path = tmp_path / 'branch.receipt.json'; write_json(branch_path, branch)
    bound = prepare_inputs(mining, ref, source_path, branch_path)
    assert str(ref.resolve()) in bound and str((mining / 'fitted_trends.json').resolve()) in bound
    (mining / 'fitted_trends.json').write_text('changed')
    with pytest.raises(ValueError, match='changed'):
        prepare_inputs(mining, ref, source_path, branch_path)
