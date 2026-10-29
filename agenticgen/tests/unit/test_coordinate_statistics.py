"""Only extreme upper RMSD observations may leave population statistics."""
import copy
import math
import pytest

from agenticgen.uniguide.coordinate_statistics import policy, summarize, DEFAULTS


def test_only_high_outlier_is_omitted_and_exceptionally_good_sample_stays():
    values = [.0001] + [1.] * 48 + [100.]
    before = copy.deepcopy(values)
    result = summarize(values)
    assert result['excluded_high_indices'] == [49]
    assert result['retained_count'] == 49 and result['lower_tail_excluded_count'] == 0
    assert result['upper_tail_filtered_mean'] == pytest.approx(48.0001 / 49)
    assert result['arithmetic_mean'] == pytest.approx(148.0001 / 50)
    assert result['upper_tail_filtered_mean_coordinate_loss_angstrom_squared'] == pytest.approx((48.+.0001**2)/98)
    assert result['ranking_eligible'] and values == before
    assert result['gradient_modified'] is False and result['source_samples_deleted'] is False


def test_no_fixed_fraction_or_lower_tail_trimming():
    values = [.01, .2, .3, .4, .5, .6, .7, .8, .9, 1.]
    result = summarize(values)
    assert result['excluded_high_count'] == 0
    assert result['upper_tail_filtered_mean'] == pytest.approx(sum(values)/len(values))


def test_degenerate_quartiles_have_an_absolute_excess_floor():
    result = summarize([0.] * 49 + [.5])
    assert result['threshold_angstrom'] == 1.
    assert result['excluded_high_count'] == 0 and result['upper_tail_filtered_mean'] == .01
    perfect = summarize([0.] * 50)
    assert perfect['upper_tail_filtered_mean_coordinate_loss_angstrom_squared'] == 0


def test_cutoff_is_strict_and_not_a_winsorized_replacement():
    assert summarize([.5]*49+[1.5])['excluded_high_count'] == 0
    result = summarize([.5]*49+[1.500001])
    assert result['excluded_high_count'] == 1 and result['upper_tail_filtered_mean'] == .5
    assert result['maximum'] == 1.500001


def test_excessive_bad_population_cannot_hide_behind_filtered_mean():
    result = summarize([.01]*40+[100.]*10)
    assert result['excluded_high_count'] == 10 and not result['ranking_eligible']
    assert result['status'] == 'too_many_high_outliers'
    assert result['arithmetic_mean'] > 20


def test_uniformly_bad_configuration_is_not_removed_as_outliers():
    result = summarize([100.]*50)
    assert result['excluded_high_count'] == 0
    assert result['upper_tail_filtered_mean'] == 100.


def test_small_population_is_not_silently_trimmed():
    result = summarize([0.,0.,100.])
    assert result['retained_count'] == 3 and result['excluded_high_count'] == 0


def test_promotion_recomputes_over_entire_population_not_shard_means():
    shard = [.5]*9+[100.]
    first = summarize(shard)
    entire = summarize(shard + [.5]*31+[100.]*9)
    assert first['ranking_eligible'] and first['excluded_high_count'] == 1
    assert entire['expected'] == 50 and not entire['ranking_eligible']
    assert entire['excluded_high_count'] == 10


@pytest.mark.parametrize('bad',[None, math.nan, math.inf, -math.inf, -1., True])
def test_missing_and_nonfinite_are_not_trimmed_into_success(bad):
    with pytest.raises(FloatingPointError):
        summarize([.5]*49+[bad])


def test_config_is_explicit_and_finite_large_outcomes_remain_in_audit():
    settings = dict(DEFAULTS, minimum_upper_excess_angstrom=2.)
    result = summarize([.1]*49+[1e12], settings, identities=list(range(50)))
    assert result['threshold_angstrom'] == 2.1
    assert result['excluded_high_samples'][0]['identity'] == 49
    assert result['raw_mean_coordinate_loss_angstrom_squared'] > 1e21
    assert result['upper_tail_filtered_mean_coordinate_loss_angstrom_squared'] == pytest.approx(.005)


@pytest.mark.parametrize('change',[
    {'interquartile_multiplier':0}, {'minimum_upper_excess_angstrom':float('nan')},
    {'maximum_excluded_fraction':.25}, {'maximum_excluded_fraction':True},
    {'minimum_population':1}, {'method':'two_sided'}, {'gradient_mask':True},
])
def test_invalid_policy_cannot_silently_change_scope(change):
    with pytest.raises(ValueError):policy({'rmsd_outlier_policy':change})
