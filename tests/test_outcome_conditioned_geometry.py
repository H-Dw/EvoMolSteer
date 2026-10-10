import numpy as np
from evomolsteer.continuous.outcome_conditioned_geometry import matched_branch_effect


def test_duplicates_do_not_add_weight_and_instant_score_is_condition():
    x = np.array([[2.], [2.], [0.], [100.]])
    online = np.array([7., 7., 7.1, 9.])
    parents = np.array([0, 0, 1, 2])
    grandparent = np.array([4, 4, 4])
    quality = np.array([8., 7.5, 9.])
    effect, count = matched_branch_effect(x, online, parents, grandparent, quality, .25)
    assert effect.tolist() == [2.]
    assert count['matched_pairs'] == count['matched_ancestors'] == 1
    # The higher final quality of a dissimilar online-score branch is excluded.
    assert count['mean_online_gap'] < .25


def test_identical_coordinates_are_control_not_advantage():
    effect, count = matched_branch_effect(np.zeros((2, 3)), np.array([7., 7.]),
        np.array([0, 1]), np.array([2, 2]), np.array([7., 8.]), .25)
    assert np.array_equal(effect, np.zeros(3))
    assert count['zero_coordinate_feature_pairs'] == 1
