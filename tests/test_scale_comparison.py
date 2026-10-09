import numpy as np
import pandas as pd
import pytest

from evomolsteer.generation.scale_comparison import (
    compare_completed_panel, paired_summary, terminal_summary, validate_candidates, terminal_scope,
)


@pytest.fixture
def panel():
    native = pd.DataFrame({'batch': [1, 1, 2, 2], 'slot': [0, 1, 0, 1],
        'pic50_on_rescore': [7., 7.1, 7.2, 7.3], 'valid_connected': [True, False, True, True],
        'pb_fast_pass': [True, False, True, True], 'smiles': ['C', None, 'CC', 'CCC'],
        'energy_status': ['converged', None, 'not_converged', 'converged'],
        'mmff_relief_per_heavy': [.1, np.nan, 100., .3]})
    r26 = native.copy()
    r26.pic50_on_rescore += .1
    r11 = pd.concat([r26, native.iloc[:2].assign(batch=3)], ignore_index=True)
    r11.pic50_on_rescore += .1
    scope = {'included_batches': [1, 2], 'snapshot_n_per_arm': 4,
        'original_expected_n_per_arm': 6, 'source_campaign_complete_at_capture': False}
    r11_audit = {'code_commit': 'fixed', 'initial_state_signatures':
        {f'gradient/{i}': str(i) for i in [1, 2, 3]}}
    control_audit = {'code_commit': 'fixed', 'initial_state_signatures':
        {f'{arm}/{i}': str(i) for arm in ['gradient', 'unguided'] for i in [1, 2]}}
    return r11, r26, native, native.copy(), scope, r11_audit, control_audit


def test_partial_controls_do_not_expand_to_full_candidate(panel):
    result = compare_completed_panel(*panel[:4], 7.25, *panel[4:])
    assert result['paired_panel']['R11']['n'] == 4
    assert result['full_unpaired']['R11']['n'] == 6
    assert result['paired_effects']['R11_vs_unguided']['paired_mean_pic50'] == pytest.approx(.2)
    assert not result['snapshot_scope']['source_campaign_complete_at_capture']


def test_uses_all_predictions_but_only_valid_converged_energy(panel):
    summary = terminal_summary(panel[2], 7.25)
    assert summary['n'] == 4 and summary['valid_n'] == 3
    assert summary['all_mean_pic50'] == pytest.approx(7.15)
    assert summary['strain_converged_n'] == 2
    assert summary['strain_median_per_heavy'] == pytest.approx(.2)
    assert summary['elite_valid_n'] == 1


def test_cross_cohort_initial_state_mismatch_is_rejected(panel):
    panel[-1]['initial_state_signatures']['unguided/2'] = 'other prior'
    with pytest.raises(ValueError, match='initial prior differs'):
        compare_completed_panel(*panel[:4], 7.25, *panel[4:])


def test_unpaired_and_duplicate_slots_are_rejected(panel):
    with pytest.raises(ValueError, match='coverage differs'):
        paired_summary(panel[1], panel[2].iloc[:-1])
    with pytest.raises(ValueError, match='Duplicate'):
        validate_candidates(pd.concat([panel[2], panel[2].iloc[[0]]]))


def test_missing_score_and_truthy_strings_are_rejected(panel):
    frame = panel[2].copy()
    frame.loc[0, 'pic50_on_rescore'] = np.nan
    with pytest.raises(ValueError, match='Every attempted'):
        validate_candidates(frame)
    frame = panel[2].copy()
    frame['valid_connected'] = frame.valid_connected.astype(str)
    with pytest.raises(ValueError, match='Boolean'):
        validate_candidates(frame)


def test_batch_bootstrap_is_deterministic_and_tracks_heterogeneity(panel):
    one = paired_summary(panel[1], panel[2])
    two = paired_summary(panel[1], panel[2])
    assert one == two
    assert one['n_batches'] == 2 and one['positive_batch_n'] == 2
    assert one['batch_bootstrap_CI95'] == pytest.approx([.1, .1])


def test_full_scope_requires_exact_completed_counts_and_batches():
    config = {'experiment': {'n': 4, 'arms': 'gradient,unguided', 'batch_indices': [1, 2]}}
    complete = {'status': 'complete', 'records': 8, 'gradient': {'n': 4}, 'unguided': {'n': 4}}
    scope = terminal_scope(config, complete, 4, [1, 2], 100.)
    assert scope['source_campaign_complete_at_capture'] and scope['snapshot_n_per_arm'] == 4
    assert scope['excluded_declared_batches'] == []
    complete['unguided']['n'] = 2
    with pytest.raises(ValueError, match='Actual completed arm counts'):
        terminal_scope(config, complete, 4, [1, 2], 100.)
    with pytest.raises(ValueError, match='registered size / batches'):
        terminal_scope(config, complete, 4, [1, 2, 3], 100.)


def test_completed_snapshot_does_not_imply_whole_campaign_completion():
    config = {'experiment': {'n': 2, 'arms': 'gradient,unguided', 'batch_indices': [1]}}
    snapshot = {'snapshot_complete': True, 'original_expected_n_per_arm': 4,
                'snapshot_n_per_arm': 2, 'included_batches': [1],
                'source_campaign_complete_at_capture': False}
    scope = terminal_scope(config, {'status': 'complete'}, 4, [1, 2], 100., snapshot)
    assert not scope['source_campaign_complete_at_capture']
    snapshot['snapshot_n_per_arm'] = 3
    with pytest.raises(ValueError, match='derived configuration coverage'):
        terminal_scope(config, {'status': 'complete'}, 4, [1, 2], 100., snapshot)


def test_full_panel_effect_is_not_labelled_as_running(panel):
    r11, r26, native, steer, scope, r11_audit, control_audit = panel
    scope.update(source_campaign_complete_at_capture=True, original_expected_n_per_arm=4)
    result = compare_completed_panel(r11[r11.batch.le(2)], r26, native, steer, 7.25,
                                     scope, r11_audit, control_audit)
    assert 'all registered generation batches completed' in result['paired_effects']['R11_vs_R26']['limitation']
    assert 'still-running' not in result['paired_effects']['R11_vs_R26']['limitation']
