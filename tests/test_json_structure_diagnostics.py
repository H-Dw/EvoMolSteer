"""Replay provider syntax defects, distinguishing stop from output truncation."""
import json
from pathlib import Path

import httpx
import pytest

from evomolsteer.agent_workflow.contracts import canonical, format_contract, strict_json
from evomolsteer.deepseek_api import call_json, LLMInterfaceError, validation_diagnostics
from evomolsteer.io import read_json
from evomolsteer.json_syntax import container_diagnostic

ROOT = Path(__file__).resolve().parents[1]
CALLS = ROOT / 'docs/experiments/temporal_guidance_evidence_fix_20261013/calls'
INVALID = sorted(CALLS.glob('W*/chain_*/Designer/*.invalid.txt'))


def test_all_four_real_failures_identify_unclosed_failure_modes_array():
    assert len(INVALID) == 4
    for path in INVALID:
        original = path.read_bytes()
        with pytest.raises(json.JSONDecodeError) as exc:
            strict_json(original.decode('utf-8'))
        diagnostic = validation_diagnostics(exc.value)['structure']
        assert diagnostic['container_pointer'] == '/result/failure_modes'
        assert diagnostic['expected_closing'] == ']'
        assert diagnostic['actual_closing'] == '}'
        assert diagnostic['kind'] == 'mismatched_closer'
        assert path.read_bytes() == original
        assert read_json(path.with_suffix('.receipt.json'))['usage']['completion_tokens'] < 8192


@pytest.mark.parametrize('text,kind,path', [
    ('{"r":{"a":["escaped \\\" ] }",1}', 'mismatched_closer', '/r/a'),
    ('{"r":{"a":["valid"]', 'unclosed_container', '/r'),
    ('{"r":"no ending', 'unterminated_or_invalid_string', '/r'),
    ('{"r":[]} }', 'unexpected_closer', ''),
])
def test_string_escape_and_nested_paths(text, kind, path):
    result = container_diagnostic(text)
    assert result['kind'] == kind and result['container_pointer'] == path


def test_correct_json_has_no_structure_diagnostic():
    assert container_diagnostic(canonical({'a': [1, {'b': '[] {} "'}]})) is None


def test_two_neutral_repairs_include_exact_diagnostic_and_preserve_invalid_raw(tmp_path, monkeypatch):
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'test-placeholder')
    broken = '{"failure_modes":["A"]}'[:-2] + '}'
    success = {'failure_modes': ['A']}
    schema = {'type': 'object', 'properties': {'failure_modes': {'type': 'array', 'items': {'type': 'string'}}},
              'required': ['failure_modes'], 'additionalProperties': False}
    requests = []
    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) > 1:
            contract = json.loads(body['messages'][-1]['content'].split('\n', 1)[1])
            detail = contract['format_errors']['structure']
            assert detail['container_pointer'] == '/failure_modes'
            assert detail['expected_closing'] == ']'
        content = broken if len(requests) < 3 else canonical(success)
        return httpx.Response(200, json={'model': 'deepseek-flash', 'id': 'fixture',
            'usage': {'completion_tokens': 12}, 'choices': [{'finish_reason': 'stop', 'message': {'content': content}}]})
    output = tmp_path / 'result.json'
    assert call_json([{'role': 'user', 'content': 'Return JSON'}], schema, output,
        transport=httpx.MockTransport(handler), response_parser=strict_json,
        format_policy='evidence-tools-json-v2', format_builder=format_contract, max_format_repairs=2) == success
    assert len(requests) == 3 and len(requests[2]['messages']) > len(requests[1]['messages'])
    invalid = list(tmp_path.glob('*.invalid.txt'))
    assert len(invalid) == 2 and all(p.read_text() == broken for p in invalid)
    receipt = read_json(output.with_suffix('.receipt.json'))
    assert len(receipt['interface_repairs']) == 2 and receipt['finish_reason'] == 'stop'
    assert all(read_json(p.with_suffix('.receipt.json'))['finish_reason'] == 'stop' for p in invalid)


def test_length_is_not_treated_as_normal_syntax_repair(tmp_path, monkeypatch):
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'test-placeholder')
    requests = []
    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={'model': 'deepseek-flash', 'id': 'fixture', 'usage': {},
            'choices': [{'finish_reason': 'length', 'message': {'content': '{'}}]})
    output = tmp_path / 'result.json'
    with pytest.raises(ValueError, match='Incomplete'):
        call_json([], {'type': 'object'}, output, transport=httpx.MockTransport(handler), max_format_repairs=2)
    assert len(requests) == 1 and not output.exists()
    assert read_json(next(tmp_path.glob('*.incomplete_*.receipt.json')))['finish_reason'] == 'length'


def test_repair_budget_exhaustion_never_creates_success(tmp_path, monkeypatch):
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'test-placeholder')
    def handler(request):
        return httpx.Response(200, json={'model': 'deepseek-flash', 'id': 'fixture', 'usage': {},
            'choices': [{'finish_reason': 'stop', 'message': {'content': '{"a":[}'}}]})
    with pytest.raises(LLMInterfaceError):
        call_json([], {'type': 'object'}, tmp_path / 'result.json', transport=httpx.MockTransport(handler), max_format_repairs=2)
    assert len(list(tmp_path.glob('*.invalid.txt'))) == 3 and not (tmp_path / 'result.json').exists()
