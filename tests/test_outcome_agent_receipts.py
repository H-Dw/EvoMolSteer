"""A correctly formatted LLM response cannot hide changed calculator outputs."""
import pytest
from evomolsteer.io import digest
from evomolsteer.continuous.outcome_agents import verify_tool_artifacts


def test_instruction_revision_keeps_historical_response_valid(tmp_path, monkeypatch):
    from evomolsteer.continuous import outcome_agents as agent
    from evomolsteer.io import read_json, write_json
    monkeypatch.setattr(agent, 'ROOT', tmp_path)
    monkeypatch.setattr(agent, 'outcome_capabilities', lambda: {})
    for name in ('analyst', 'designer', 'terminal-outcome', 'extension'):
        skill = tmp_path/'skills'/name/'SKILL.md'
        skill.parent.mkdir(parents=True)
        skill.write_text('Literal instruction '+name, encoding='utf-8')
    evidence = tmp_path/'evidence.json'
    registry = tmp_path/'registry.json'
    receipt = tmp_path/'receipt.json'
    raw = tmp_path/'raw.json'
    write_json(raw, {'actual': 1})
    write_json(evidence, {'window': [0, .51], 'score_window': [0, .5],
        'mode': 'mean', 'evidence_items': [{'id': 'actual'}]})
    write_json(registry, {'task': 'Test', 'instruction_modules': ['extension']})
    write_json(receipt, {'returncode': 0, 'inputs_unchanged': True,
        'input_files': {str(raw.resolve()): digest(raw)},
        'output_files': {str(evidence.resolve()): digest(evidence)}})
    request = agent.export_request('Analyst', evidence, registry, receipt, tmp_path/'r1')
    packet = read_json(request)
    response = tmp_path/'r1/Analyst.response.json'
    write_json(response, {'schema_version': agent.VERSION, 'agent': 'Analyst',
        'request_sha256': digest(request), 'instruction_sha256': packet['instruction_sha256'],
        'evidence_sha256': digest(evidence), 'tool_receipt_sha256': digest(receipt),
        'window': [0, .51], 'score_window': [0, .5], 'evidence_ids': ['actual'],
        'rationale': ['Observed'], 'counterevidence': ['Uncertain'],
        'label_source': 'decoded_final', 'findings': ['Actual'], 'extensions_ready': False})
    # Later iterative edits must not rewrite or invalidate the literal prior call.
    (tmp_path/'skills/extension/SKILL.md').write_text('New instruction', encoding='utf-8')
    agent.validate_response(request, response)
    # A mixed revision inside a single Analyst/Designer pair must be rejected.
    with pytest.raises(ValueError, match='Skill changed during one role pair'):
        agent.export_request('Designer', evidence, registry, receipt, tmp_path/'r1', response)
    request2 = agent.export_request('Analyst', evidence, registry, receipt, tmp_path/'r2')
    assert read_json(request2)['instruction_sha256'] != packet['instruction_sha256']


def test_receipt_requires_unchanged_computation(tmp_path):
    raw = tmp_path/'input.json'
    evidence = tmp_path/'evidence.json'
    raw.write_text('{"x":1}')
    evidence.write_text('{"effect":2}')
    receipt = {'returncode': 0, 'inputs_unchanged': True,
               'input_files': {str(raw): digest(raw)},
               'output_files': {str(evidence): digest(evidence)}}
    verify_tool_artifacts(receipt)
    evidence.write_text('{"effect":200}')
    with pytest.raises(ValueError, match='artifact changed'):
        verify_tool_artifacts(receipt)


def test_failed_or_empty_receipt_is_not_evidence():
    with pytest.raises(ValueError, match='Successful'):
        verify_tool_artifacts({'returncode': 1, 'inputs_unchanged': True})
    with pytest.raises(ValueError, match='inventory'):
        verify_tool_artifacts({'returncode': 0, 'inputs_unchanged': True})
