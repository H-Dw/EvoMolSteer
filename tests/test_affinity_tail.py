import numpy as np
import pandas as pd
import pytest

from evomolsteer.continuous.affinity_tail import (
    bh_adjust, comparison_metrics, matched_contrasts, shape_features, stage_trends,
)


def test_shape_rigid_motion_and_padding():
    xyz = np.array([[[0., 0., 0.], [2., 0., 0.], [0., 1., 0.], [np.nan]*3]])
    mask = np.array([[True, True, True, False]])
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    original = shape_features(xyz, mask)
    transformed = shape_features(xyz@rotation+7., mask)
    for name in original:
        np.testing.assert_allclose(original[name], transformed[name], atol=1e-12)
    assert original['shape::thickness_A'][0] == 0
    with pytest.raises(ValueError):
        shape_features(xyz, np.ones_like(mask))


def test_bh_controls_multiple_comparisons():
    np.testing.assert_allclose(bh_adjust([.5, .001, .04]), [.5, .003, .06])
    with pytest.raises(ValueError):
        bh_adjust([np.nan])


def test_clones_do_not_multiply_contrast_evidence():
    rows = []
    for batch in range(6):
        for smiles, elite, value in [('a', True, 2.), ('b', True, 4.), ('c', False, 0.)]:
            rows.append({'batch': batch, 'smiles': smiles, 'elite': elite, 'shape::x': value+batch})
    base = pd.DataFrame(rows)
    clones = pd.concat([base, base[base.smiles == 'a']]*3, ignore_index=True)
    effects, summary = matched_contrasts(base, ['shape::x'], bootstrap=100)
    other_effects, other_summary = matched_contrasts(clones, ['shape::x'], bootstrap=100)
    pd.testing.assert_frame_equal(effects, other_effects)
    pd.testing.assert_frame_equal(summary, other_summary)
    assert summary.mean_effect.iloc[0] == 3.


def test_multi_arm_table_requires_explicit_arm(tmp_path):
    path = tmp_path/'mixed.csv'
    pd.DataFrame({'arm': ['native', 'gradient'], 'batch': [0, 0], 'slot': [0, 0],
                  'valid_connected': [True, True], 'pb_fast_pass': [True, True]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match='Multi-arm'):
        comparison_metrics(path)
    native, _, arm = comparison_metrics(str(path)+'#native')
    assert len(native) == 1 and arm == 'native'
    with pytest.raises(ValueError, match='Unknown'):
        comparison_metrics(str(path)+'#other')


def test_trends_fit_actual_events_without_bins():
    rows = [{'group': 'elite_descendant', 'step': i, 'score_time': i/100, 'n_nodes': 2,
             'online_score': 7+2*i/100, 'score_rank_fraction': .7, 'shape::x': 1+i/100}
            for i in range(8) for _ in range(4)]
    curves, fits = stage_trends(pd.DataFrame(rows))
    assert curves.step.nunique() == 8
    score = next(f for f in fits if f['feature'] == 'online_score')
    assert score['end'] == .07 and score['r_squared'] > .999999
    np.testing.assert_allclose(score['derivative_coefficients_descending'], [0, 2], atol=1e-9)
