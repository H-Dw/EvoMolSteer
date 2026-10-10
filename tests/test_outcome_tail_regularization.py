import pytest
from evomolsteer.continuous.outcome_tail_regularization import regularized_fraction


def test_single_lucky_child_gets_less_reliable_tail_credit():
    assert regularized_fraction(1., 1, .02, 2.) < regularized_fraction(1., 10, .02, 2.) < 1.
    assert regularized_fraction(1., 1, .02, 0.) == 1.
    assert regularized_fraction(0., 1, 0., 2.) == 0.


def test_unknown_future_is_never_regularized_to_a_failure_or_success():
    with pytest.raises(ValueError, match='extinct futures'):
        regularized_fraction(0., 0, .1, 2.)
    for raw, count, baseline, strength in [(1.1, 1, .1, 2), (.1, 1.5, .1, 2), (.1, 1, .1, float('nan'))]:
        with pytest.raises(ValueError):
            regularized_fraction(raw, count, baseline, strength)


def test_conversion_keeps_geometry_clocks_and_mean_and_counts_invalid_baseline(tmp_path):
    import gzip
    import json
    import pandas as pd
    from evomolsteer.continuous.outcome_tail_regularization import build
    from evomolsteer.io import digest, write_json, read_json
    metrics = tmp_path/'metrics.csv'
    pd.DataFrame({'batch': [0]*20, 'slot': range(20), 'valid_connected': [True, False]+[True]*18,
                  'pic50_on_rescore': [9., 9.]+[7.]*18}).to_csv(metrics, index=False)
    clouds = [[[1., 2., 3.]], [[4., 5., 6.]]]
    ref = {'window': [.2, .31], 'score_window': [.2, .3],
        'teacher_selection': {'tail_weight': .25, 'prior_mode': 'copy_family_mean_plus_tail'},
        'frames': [{'time': .2, 'teacher_batches': [0, 0], 'teacher_endpoint_A': clouds,
            'teacher_scores': [9.25, 8.025], 'teacher_outcomes': [
                {'alias_family_credit': {'terminal_mean': 9., 'tail_fraction': 1., 'observed_n': 1, 'valid_fraction': 1.}},
                {'alias_family_credit': {'terminal_mean': 8., 'tail_fraction': .1, 'observed_n': 10, 'valid_fraction': .9}}]}]}
    source = tmp_path/'reference.json.gz'
    source.write_bytes(gzip.compress(json.dumps(ref).encode()))
    evidence = tmp_path/'evidence.json'
    write_json(evidence, {'reference_path': str(source), 'reference_sha256': digest(source),
        'metrics_sha256': digest(metrics), 'donor_batches': [0], 'threshold_pic50': 8.5,
        'evidence_items': [], 'window': [.2, .31], 'score_window': [.2, .3]})
    before = digest(source)
    build(evidence, metrics, tmp_path/'out', 2.)
    after = json.loads(gzip.decompress((tmp_path/'out/reference.json.gz').read_bytes()))
    assert digest(source) == before
    assert after['window'] == ref['window']
    assert after['frames'][0]['teacher_endpoint_A'] == clouds
    table = pd.read_csv(tmp_path/'out/support_credit.csv')
    assert table.batch_fraction.tolist() == [.05, .05]
    assert table.mean_component_change.eq(0).all()
    assert after['frames'][0]['teacher_scores'][0] == pytest.approx(9.+.25*(1.+2*.05)/3.)
    assert read_json(tmp_path/'out/evidence.json')['evidence_items'][-1]['point_selection_unchanged']
