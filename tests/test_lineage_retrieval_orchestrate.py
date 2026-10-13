"""Decision packets and audit contracts must not expose oracle identities."""
import pytest

from evomolsteer.lineage_retrieval.orchestrate import anonymize, audit_passes, selector_output_schema, validate_selector


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


def test_citation_schema_preserves_all_candidates_and_hides_control_affinity():
    import jsonschema
    packet, _ = anonymize(pool(), structural_only=True)
    schema = selector_output_schema(packet)
    props = schema['properties']['ranking']['items']['properties']
    assert set(props['candidate_id']['enum']) == {c['candidate_id'] for c in packet['cards']}
    assert 'online_pic50' not in props['observations']['items']['properties']['feature']['enum']
    a = answer(packet)
    for row, card in zip(a['ranking'], packet['cards']):
        for obs in row['observations']:
            obs['observed_value'] = next(s[obs['feature']] for s in card['history'] if s['step'] == obs['step'])
    jsonschema.validate(a, schema)
    assert validate_selector(a, packet, 'Q.TEST')
    a['ranking'][0]['observations'][0]['observed_value'] = 7.5
    with pytest.raises(ValueError, match='Observed value mismatch'):
        validate_selector(a, packet, 'Q.TEST')
    a['ranking'][0]['observations'][0]['observed_value'] = 1.0
    a['ranking'][0]['observations'][0]['feature'] = 'current_Rg / online_pic50'
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(a, schema)


def test_quality_retry_preserves_unaffected_outputs_and_never_forwards_the_audit(tmp_path, monkeypatch):
    from evomolsteer.lineage_retrieval import orchestrate as o
    study = tmp_path / 'study'
    folder = study / 'rounds' / 'R00'
    folder.mkdir(parents=True)
    o.init_versions(study, tmp_path)
    meta = {'scientist_version': 'S00', 'selector_version': 'Q00'}
    o.write(folder / '_version.json', meta)
    o.write(folder / 'hypotheses.json', [{'end_step': s} for s in o.STEPS])
    o.write(folder / 'metrics.json', {'complete': True})
    for job in [folder / 'scientist', *[folder / 'selectors' / f't{s:03d}' for s in o.STEPS]]:
        o.write(job / 'answer.json', {'untouched': job.name})
        o.write(job / 'controller_binding.json', {'unchanged': job.name})
    counter = {'audits': 0}
    def audit(_study, _folder, _meta, enforce=False):
        counter['audits'] += 1
        result = {'passes': counter['audits'] > 1, 'scientist_evidence_truth': True,
            'scientist_clause_applied': True, 'scientist_assessment': 'Accurate', 'unsupported_claims': [],
            'selector_assessments': [{'step': s, 'evidence_truth': s != 5 or counter['audits'] > 1,
                'clause_applied': True} for s in o.STEPS]}
        o.write(_folder / 'semantic_audit.json', result)
        return result
    def selectors(_study, _folder, _code, hypotheses, workers):
        recovery = o.read(_folder / 'quality_retry.json')
        assert recovery['selector_steps'] == [5]
        assert o.read(_folder / 'scientist' / 'answer.json') == {'untouched': 'scientist'}
        assert o.read(_folder / 'selectors' / 't010' / 'answer.json') == {'untouched': 't010'}
        assert not (_folder / 'selectors' / 't005' / 'answer.json').exists()
        assert 'unsupported_claims' not in recovery
        assert 'online_pic50' not in recovery
        o.write(_folder / 'selectors' / 't005' / 'answer.json', {'corrected': True})
    monkeypatch.setattr(o, 'semantic_audit', audit)
    monkeypatch.setattr(o, 'run_selectors', selectors)
    monkeypatch.setattr(o, 'run_scientist', lambda *_: pytest.fail('Unaffected Scientist was rerun'))
    monkeypatch.setattr(o, 'seal_metrics', lambda _s, _f, _m: o.write(_f / 'metrics.json', {'complete': True}))
    monkeypatch.setattr(o, 'commit_versions', lambda *_: 'test-only')
    assert o.audit_and_repair(study, folder, meta, 3)['passes']
    assert (folder / 'rejected' / 'T01' / 'selectors' / 't005' / 'answer.json').exists()
    assert (study / 'versions' / 'selector' / 'Q00' / 'repairs' / 'R00_T01' / 'workflow.md').exists()
