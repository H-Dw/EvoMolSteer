import pytest
from evomolsteer.io import read_json, write_json
from evomolsteer.continuous.outcome_evidence_scope import summarize


def test_conditional_packet_does_not_reuse_inherited_unconditional_significance(tmp_path):
    packet = tmp_path/'packet.json'
    write_json(packet, {'evidence_items': [
        {'id': 'outcome/geometry/occupancy', 'p': .001, 'q': .01, 'effect': 1e-8},
        {'id': 'outcome/matched/occupancy', 'p': .9, 'q': .98, 'effect': .1},
        {'id': 'outcome/matched_support', 'total_pairs': 2, 'supported_batch_events': 1}]})
    out = tmp_path/'matched.json'
    value = summarize(packet, 'matched_geometry', out)
    assert value['feature_count'] == 1 and value['corrected_significant_n'] == 0
    assert value['minimum_q'] == .98 and value['support']['total_pairs'] == 2
    assert read_json(out)['prefix'] == 'outcome/matched/'


def test_missing_scope_fails_instead_of_falling_back_to_a_different_analysis(tmp_path):
    packet = tmp_path/'packet.json'
    write_json(packet, {'evidence_items': [{'id': 'outcome/geometry/x', 'p': .1, 'q': .2}]})
    with pytest.raises(ValueError, match='nonempty'):
        summarize(packet, 'matched_geometry', tmp_path/'missing.json')
