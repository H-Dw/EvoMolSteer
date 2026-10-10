import numpy as np
import pandas as pd
import pytest
from evomolsteer.continuous.terminal_outcome import outcome_labels, observed_support, choose_teachers, utility


def test_final_labels_censoring_and_duplicate_graph_mean():
    selected = np.array([[0, 0, 2], [0, 1, 1]], dtype=int)
    metrics = pd.DataFrame({'slot': [0, 1, 2], 'valid_connected': [True]*3,
        'pic50_on_rescore': [9., 7., 5.], 'smiles': ['same', 'same', 'other']})
    labels = outcome_labels(selected, metrics, 8.)
    root = labels[0]
    assert root.loc[0, 'observed_n'] == 3
    assert root.loc[0, 'terminal_mean'] == 6.5  # equal graphs: mean(9,7) vs 5
    assert root.loc[0, 'tail_fraction'] == 1/3
    assert root.loc[1, 'future_observed'] == False
    assert np.isnan(root.loc[1, 'terminal_mean'])
    assert np.isnan(root.loc[1, 'tail_fraction'])


def test_dynamic_clock_inclusive_last_selection_and_state_boundary():
    clock = np.array([0., .1, .2, .3, .4])
    state = clock+.1
    steps, window = observed_support(clock, state, np.array([True]*4+[False]), [.1, .3])
    np.testing.assert_array_equal(steps, [1, 2, 3])
    assert window == [.1, .4]
    with pytest.raises(ValueError):
        observed_support(clock, state, np.array([True]*4+[False]), [.1, .4])


def test_distribution_does_not_rank_a_single_best_child():
    labels = pd.DataFrame({'terminal_mean': [7.5, 7.8, np.nan],
                           'terminal_p75': [8., 7.9, np.nan], 'tail_fraction': [.1, .2, np.nan]})
    np.testing.assert_allclose(utility(labels, 'distribution', .5)[:2], [7.55, 7.9])
    xyz = np.array([[[0., 0., 0.]], [[1., 0., 0.]], [[2., 0., 0.]]])
    assert choose_teachers(xyz, utility(labels, 'mean'), 2) == [1, 0]


def test_parent_distribution_shrinkage_preserves_censoring_and_dampens_singletons():
    labels = pd.DataFrame({'terminal_mean': [9., 9., np.nan], 'unique_graph_n': [1, 10, 0]})
    result = utility(labels, 'hierarchical', parent_quality=np.array([7., 7., np.nan]), shrinkage=2.)
    assert result[0] < result[1] < 9.
    assert np.isnan(result[2])
    np.testing.assert_allclose(utility(labels, 'hierarchical', parent_quality=np.array([7., 7., np.nan]), shrinkage=0.)[:2], [9., 9.])


def test_invalid_high_head_is_observed_failure_not_quality_or_censoring():
    selected = np.array([[0, 0, 2]])
    metrics = pd.DataFrame({'slot': [0, 1, 2], 'valid_connected': [True, False, True],
        'pic50_on_rescore': [7., 10., 8.], 'smiles': ['valid', 'invalid', 'other']})
    root = outcome_labels(selected, metrics, 8.5)[0]
    assert root.loc[0, 'terminal_mean'] == 7.
    assert root.loc[0, 'observed_n'] == 2
    assert root.loc[0, 'valid_fraction'] == .5
    assert root.loc[0, 'tail_fraction'] == 0.
    assert root.loc[0, 'future_observed']
    assert not root.loc[1, 'future_observed']
