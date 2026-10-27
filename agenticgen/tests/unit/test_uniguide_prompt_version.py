"""New-version integration retains real provider validation and deterministic caches."""
import importlib.util
import json
import sys

import httpx
import pytest

from test_uniguide_prompt_policy import ROOT, SKILLS, sample_context, schema, answer

sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('prompt_version', ROOT / 'scripts/run_uniguide_prompt_version.py')
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


def setup(tmp_path, monkeypatch, bad=False):
    from agenticgen.providers.deepseek_api import call_json
    from agenticgen.uniguide.protocol import validate_action
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'test-only')
    out = tmp_path / 'versions/UG05/teacher/advice_R01/redistribute.json'
    out.parent.mkdir(parents=True)
    calls, validations = [], []
    def http(request):
        calls.append(json.loads(request.content))
        value = answer()
        if bad:
            value['proposals'][0]['anchor_id'] = 'r80'
        return httpx.Response(200, json=dict(model='deepseek-flash', usage={}, choices=[dict(
            finish_reason='stop', message=dict(content=json.dumps(value)))]))
    def validate(result):
        for action in result['proposals']:
            validate_action(action, {'anchor':sample_context()['history'][0]},
                dict(min_index=11, max_index=19), [])
        validations.append(result)
    options = dict(model='deepseek-flash', thinking=True, reasoning_effort='high', max_output_tokens=16384,
        transport=httpx.MockTransport(http), response_validator=validate)
    binding = runtime.runtime_binding(SKILLS)
    wrapper = runtime.wrap_provider(call_json, tmp_path, 'UG05', SKILLS, binding)
    messages = [dict(role='system', content='original'), dict(role='user', content=json.dumps(sample_context()))]
    return wrapper, out, messages, options, calls, validations


def test_roundtrip_routes_roles_and_reuses_cache_with_reordered_object_keys(tmp_path, monkeypatch):
    wrapper, out, messages, options, calls, validations = setup(tmp_path, monkeypatch)
    assert wrapper(messages, schema(), out, **options) == answer()
    messages[1]['content'] = json.dumps(json.loads(messages[1]['content']), sort_keys=True)
    assert wrapper(messages, schema(), out, **options) == answer()
    assert len(calls) == 1 and len(validations) == 2
    assert 'select supplied proposals' not in calls[0]['messages'][0]['content']
    packed = json.loads(calls[0]['messages'][1]['content'])
    assert packed['execution_contract']['allowed_anchor_ids'] == ['anchor']
    expanded = runtime.policy.legacy.unpack_context(packed)
    assert expanded.pop('completed_action_evidence') == []
    assert expanded == sample_context()
    assert calls[0]['max_tokens'] == 16384


def test_failure_claim_prevents_automatic_third_http(tmp_path, monkeypatch):
    from agenticgen.providers.deepseek_api import LLMInterfaceError
    wrapper, out, messages, options, calls, _ = setup(tmp_path, monkeypatch, bad=True)
    with pytest.raises(LLMInterfaceError): wrapper(messages, schema(), out, **options)
    with pytest.raises(FileExistsError): wrapper(messages, schema(), out, **options)
    assert len(calls) == 2 and not out.exists()


def test_completed_proposals_are_explicit_and_bound_without_future_records(tmp_path, monkeypatch):
    wrapper, out, messages, options, calls, _ = setup(tmp_path, monkeypatch)
    group = tmp_path / 'versions/UG05/teacher/R01/increase'
    group.mkdir(parents=True)
    (group / 'COMPLETE.json').write_text('{}')
    proposal = dict(anchor_id='anchor', prediction='C falls', falsifier='C rises')
    (group / 'proposal.json').write_text(json.dumps(proposal))
    context = sample_context()
    context['history'].append(dict(context['history'][0], id='versions/UG05/teacher/R01/increase'))
    messages[1]['content'] = json.dumps(context)
    wrapper(messages, schema(), out, **options)
    expanded = runtime.policy.legacy.unpack_context(json.loads(calls[0]['messages'][1]['content']))
    assert expanded['completed_action_evidence'][0]['proposal'] == proposal
    assert expanded['completed_action_evidence'][0]['source_group'] == context['history'][1]['id']
    (group / 'proposal.json').write_text('{}')
    with pytest.raises(ValueError, match='Frozen prompt evidence changed'):
        wrapper(messages, schema(), out, **options)
    assert len(calls) == 1


def test_protocol_and_scope_drift_rejected_before_http(tmp_path, monkeypatch):
    wrapper, out, messages, options, calls, _ = setup(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='protocol'):
        wrapper(messages, schema(), out, **dict(options, max_output_tokens=24576))
    with pytest.raises(ValueError, match='outside'):
        wrapper(messages, schema(), tmp_path / 'versions/UG04/a.json', **options)
    assert calls == []


def test_runtime_frozen_binding_rejects_asset_drift(tmp_path):
    path = tmp_path / 'PROMPT_POLICY.json'
    runtime.frozen_json(path, {'policy':'v2'})
    runtime.frozen_json(path, {'policy':'v2'})
    with pytest.raises(ValueError): runtime.frozen_json(path, {'policy':'v1'})
