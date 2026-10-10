"""A correctly formatted LLM response cannot hide changed calculator outputs."""
import pytest
from evomolsteer.io import digest
from evomolsteer.continuous.outcome_agents import verify_tool_artifacts


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
