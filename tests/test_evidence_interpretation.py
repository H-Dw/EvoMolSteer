import pytest

from evomolsteer.ancestral_modes.evidence_interpretation import contrast_reading


def test_local_positive_contrasts_do_not_establish_beyond_global_increment():
    evidence = {'terminal_contrast': {'features': {
        'pair_error_A': {'cluster_interval_95': [.067, .324]},
        'point_error_A': {'cluster_interval_95': [.034, .168]},
        'beyond_global_error_A': {'cluster_interval_95': [-.045, .174]},
    }}}
    result = contrast_reading(evidence)
    assert result['features']['pair_error_A']['positive_error_contrast_supported']
    assert not result['positive_beyond_global_contrast_supported']
    assert result['features']['beyond_global_error_A']['interval_status'] == 'includes_zero'
    # Tool augmentation must not mutate frozen evidence files or statistics.
    assert 'deterministic_contrast_reading' not in evidence


@pytest.mark.parametrize('interval,status,supported', [
    (None, 'unavailable', False), ([0., .2], 'includes_zero', False),
    ([-.2, 0.], 'includes_zero', False), ([-.2, -.1], 'negative', False),
    ([.01, .2], 'positive', True),
])
def test_increment_requires_a_strictly_positive_interval(interval, status, supported):
    result = contrast_reading({'terminal_contrast': {'features': {
        'beyond_global_error_A': {'cluster_interval_95': interval}}}})
    assert result['features']['beyond_global_error_A']['interval_status'] == status
    assert result['positive_beyond_global_contrast_supported'] is supported


def test_missing_increment_cannot_inherit_a_positive_signed_axis():
    result = contrast_reading({'terminal_contrast': {'features': {
        'signed_dx_A': {'cluster_interval_95': [.1, .2]}}}})
    assert not result['features']['signed_dx_A']['positive_error_contrast_supported']
    assert not result['positive_beyond_global_contrast_supported']


@pytest.mark.parametrize('interval', [[2., 1.], [0., float('nan')], [1.]])
def test_malformed_intervals_are_not_presented_to_agents_as_evidence(interval):
    with pytest.raises(ValueError, match='Invalid exploratory contrast interval'):
        contrast_reading({'terminal_contrast': {'features': {
            'beyond_global_error_A': {'cluster_interval_95': interval}}}})
