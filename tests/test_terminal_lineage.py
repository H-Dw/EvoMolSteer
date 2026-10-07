import numpy as np
import pytest
from evomolsteer.continuous.terminal_lineage import descendant_credit, diverse_survivors


def test_credit_backward_mapping_and_terminal_selection_exclusion():
    selected = np.array([[0, 0, 2, 2], [1, 1, 2, 3], [3, 3, 3, 3]])
    counts, labels = descendant_credit(selected, [1., 3., 5., 7.])
    np.testing.assert_array_equal(counts, [[2, 0, 2, 0], [0, 2, 1, 1], [1, 1, 1, 1]])
    np.testing.assert_allclose(labels[0, [0, 2]], [2., 6.])
    assert np.isnan(labels[0, [1, 3]]).all()
    np.testing.assert_array_equal(labels[2], [1., 3., 5., 7.])
    # A late resampling edge must participate; final-row resampling must not.
    selected[-1] = 0
    np.testing.assert_array_equal(descendant_credit(selected, [1., 3., 5., 7.])[0], counts)
    selected[1] = 2
    assert descendant_credit(selected, [1., 3., 5., 7.])[0][0, 2] == 4


def test_credit_rejects_invalid_parent_or_unobserved_scores():
    with pytest.raises(ValueError): descendant_credit(np.array([[0, 2]]), [1., 2.])
    with pytest.raises(ValueError): descendant_credit(np.array([[0., 1.]]), [1., 2.])
    with pytest.raises(ValueError): descendant_credit(np.array([[0, 1]]), [np.nan, 2.])


def test_teacher_ranking_uses_credit_not_descendant_multiplicity():
    x = np.array([[[0., 0., 0.], [1., 0., 0.]], [[2., 0., 0.], [3., 0., 0.]], [[4., 0., 0.], [5., 0., 0.]]])
    assert diverse_survivors(x, [6., 8., np.nan], [49, 1, 0], [[0., 2., 0.]], [0., 0., 0.], 1) == [1]
    # Coincident prototypes stay one coherent mode even with different labels.
    x[0] = x[1]
    assert diverse_survivors(x, [6., 8., np.nan], [49, 1, 0], [[0., 2., 0.]], [0., 0., 0.], 3) == [1]
