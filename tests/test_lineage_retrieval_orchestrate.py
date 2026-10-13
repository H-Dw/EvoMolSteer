"""Decision packets and audit contracts must not expose oracle identities."""
import pytest

from evomolsteer.lineage_retrieval.orchestrate import anonymize, audit_passes, validate_selector


def pool():
    return {'step': 5, 'time': .05, 'cards': [
        {'key': f'private_{i}', 'now': {'step': 5, 'online_pic50': 8 + i / 10, 'current_Rg': .2},
         'history': [{'step': 0, 'online_pic50': 7., 'current_Rg': 1.},
                     {'step': 5, 'online_pic50': 8 + i / 10, 'current_Rg': .2}]}
        for i in range(6)]}


def answer(packet):
    return {'step': 5, 'ranking': [
        {'rank': i + 1, 'candidate_id': c['candidate_id'], 'confidence': 1 - i / 10,
         'evidence': 'A measured history observation supports the pocket pattern.', 'counterevidence': 'Other modes remain plausible.',
         'observations': [{'feature': 'current_Rg', 'step': 0, 'direction_or_value': '1.0'},
                          {'feature': 'current_Rg', 'step': 5, 'direction_or_value': '.2'}]}
        for i, c in enumerate(packet['cards'][:5])],
        'applied_clauses': [{'clause_id': 'Q.TEST', 'evidence': 'Compared the actual historical cloud change with multiple forecast modes.'}]}


def test_reanonymous_packet_hides_private_keys_and_affinity_in_control():
    source = pool()
    packet, mapping = anonymize(source, structural_only=True)
    assert len(packet['cards']) == 6
    assert len(mapping['anonymous_to_key']) == 6
    assert all('key' not in r for r in packet['cards'])
    assert all('online_pic50' not in st for r in packet['cards'] for st in [r['now'], *r['history']])
    assert 'online_pic50' in source['cards'][0]['now']
    again, same_mapping = anonymize(source, mapping, structural_only=True)
    assert again == packet and same_mapping == mapping


def test_explicit_history_citations_and_future_rejection():
    packet, _ = anonymize(pool())
    a = answer(packet)
    assert validate_selector(a, packet, 'Q.TEST')['clause_id'] == 'Q.TEST'
    a['ranking'][0]['observations'][0]['step'] = 6
    with pytest.raises(ValueError, match='unavailable'):
        validate_selector(a, packet, 'Q.TEST')


def test_confidence_rank_and_nonexistent_evidence_rejected():
    packet, _ = anonymize(pool())
    a = answer(packet)
    a['ranking'][1]['confidence'] = 1.1
    with pytest.raises(ValueError, match='confidence'):
        validate_selector(a, packet, 'Q.TEST')
    a = answer(packet)
    a['ranking'][0]['observations'][0]['feature'] = 'terminal_affinity'
    with pytest.raises(ValueError, match='unavailable'):
        validate_selector(a, packet, 'Q.TEST')


def test_joint_keys_get_actionable_feedback_without_changing_the_ranking():
    packet, _ = anonymize(pool())
    a = answer(packet)
    ids = [r['candidate_id'] for r in a['ranking']]
    a['ranking'][0]['observations'][0]['feature'] = 'current_Rg / online_pic50'
    with pytest.raises(ValueError, match='exactly ONE literal feature key'):
        validate_selector(a, packet, 'Q.TEST')
    assert [r['candidate_id'] for r in a['ranking']] == ids


def test_semantic_pass_requires_both_roles_and_all_ten_nodes():
    audit = {'passes': True, 'scientist_evidence_truth': True, 'scientist_clause_applied': True,
             'selector_assessments': [{'step': s, 'evidence_truth': True, 'clause_applied': True} for s in range(5, 51, 5)]}
    assert audit_passes(audit)
    audit['scientist_clause_applied'] = False
    assert not audit_passes(audit)
    audit['scientist_clause_applied'] = True
    audit['selector_assessments'].pop()
    assert not audit_passes(audit)
