import gzip
import json

import pytest
from evomolsteer.continuous.cohort_target_functions import fit_targets


def test_cohort_motion_is_not_the_motion_of_the_contrast(tmp_path):
    times = [.2, .3, .4, .5, .6]
    frames = [{'time': t, 'dynamic_region': {'observed_batches': [0, 1, 2, 3],
              'positive_centers': [[2*t+1.]]*4, 'negative_centers': [[2*t]]*4}} for t in times]
    reference = tmp_path / 'ref.json.gz'
    reference.write_bytes(gzip.compress(json.dumps({'times': times, 'window': [.2, .7],
             'features': ['landmark_00_softmin'], 'frames': frames,
             'dynamic_cohort': {'selected_feature_indices': [0]}}).encode()))
    result = fit_targets(reference, tmp_path / 'functions.json')
    assert result['learned_support'] == [.2, .7] and result['observed_score_support'] == [.2, .6]
    for role in ['positive', 'negative']:
        f = result['functions'][role]['landmark_00_softmin']
        assert f['degree'] == 1 and f['fitted_rate_start'] == pytest.approx(2.)
        assert f['observed_end_minus_start'] == pytest.approx(.8)
    # The positive-minus-negative difference remains one at every time.
    p = result['functions']['positive']['landmark_00_softmin']['legendre_coefficients']
    n = result['functions']['negative']['landmark_00_softmin']['legendre_coefficients']
    assert p[1]-n[1] == pytest.approx(0., abs=1e-12)
