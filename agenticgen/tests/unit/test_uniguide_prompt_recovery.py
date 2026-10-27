"""Explicit recovery keeps real provider checks and never loops after failure."""
import hashlib
import json

import httpx
import pytest

from test_uniguide_prompt_version import setup, runtime, SKILLS, schema, answer
from agenticgen.providers.deepseek_api import call_json
import resume_uniguide_prompt_call as recovery


def fixture(tmp_path, monkeypatch):
    wrapper, output, messages, opts, calls, validations = setup(tmp_path, monkeypatch)
    def length(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json=dict(model='deepseek-flash', usage={}, choices=[dict(
            finish_reason='length', message=dict(content=''))]))
    with pytest.raises(ValueError, match='Incomplete'):
        wrapper(messages, schema(), output, **dict(opts, transport=httpx.MockTransport(length)))
    protected = {p:recovery.sha(p) for p in output.parent.iterdir() if p.is_file()}
    auth = tmp_path / 'authorization.json'
    auth.write_text(json.dumps(dict(output_relative_path=output.relative_to(tmp_path).as_posix(), successful_outputs={})))
    events = []
    def make(execute=False):
        return recovery.recovery_call(runtime, call_json, tmp_path, 'UG05', SKILLS,
            runtime.runtime_binding(SKILLS), auth, events.append, execute)
    return make, output, messages, opts, calls, validations, protected, events, length, auth


def test_preflight_validates_exact_request_without_http_or_claim(tmp_path, monkeypatch):
    make, out, messages, opts, calls, _, protected, events, _, _ = fixture(tmp_path, monkeypatch)
    with pytest.raises(recovery.TargetReached):make()(messages, schema(), out, **opts)
    assert len(calls) == 1 and events[-1]['event'] == 'exact_target_verified'
    assert not out.with_suffix('.explicit_recovery_claim.json').exists()
    assert all(recovery.sha(p) == value for p, value in protected.items())


def test_one_recovery_preserves_original_claim_and_identical_wire(tmp_path, monkeypatch):
    make, out, messages, opts, calls, validations, protected, _, _, _ = fixture(tmp_path, monkeypatch)
    with pytest.raises(recovery.TargetReached):make(True)(messages, schema(), out, **opts)
    assert len(calls) == 2 and len(validations) == 1 and json.loads(out.read_text()) == answer()
    assert out.with_suffix('.wire_1.request.json').read_bytes() == out.with_suffix('.wire_2.request.json').read_bytes()
    assert all(recovery.sha(p) == value for p, value in protected.items())
    with pytest.raises(ValueError, match='already succeeded'):make(True)(messages, schema(), out, **opts)
    assert len(calls) == 2
    native = runtime.wrap_provider(call_json, tmp_path, 'UG05', SKILLS, runtime.runtime_binding(SKILLS))
    assert native(messages, schema(), out, **opts) == answer()
    assert len(calls) == 2 and len(validations) == 2


def test_failed_recovery_cannot_loop_or_delete_claim(tmp_path, monkeypatch):
    make, out, messages, opts, calls, _, protected, _, length, _ = fixture(tmp_path, monkeypatch)
    opts = dict(opts, transport=httpx.MockTransport(length))
    with pytest.raises(ValueError, match='Incomplete'):make(True)(messages, schema(), out, **opts)
    with pytest.raises(ValueError, match='exactly one'):make(True)(messages, schema(), out, **opts)
    assert len(calls) == 2 and out.with_suffix('.explicit_recovery_claim.json').exists()
    assert all(recovery.sha(p) == value for p, value in protected.items())


@pytest.mark.parametrize('change', ['context', 'wire', 'protocol', 'claim'])
def test_drift_is_rejected_before_http(tmp_path, monkeypatch, change):
    make, out, messages, opts, calls, _, _, _, _, _ = fixture(tmp_path, monkeypatch)
    if change == 'context':
        ctx = json.loads(messages[1]['content']);ctx['physical_start'] = 'changed'
        messages[1]['content'] = json.dumps(ctx)
    elif change in ['wire', 'claim']:
        path = out.with_suffix('.wire_1.request.json' if change == 'wire' else '.execution_claim.json')
        value = json.loads(path.read_text());value['extra'] = True;path.write_text(json.dumps(value))
    else:opts = dict(opts, max_output_tokens=24576)
    with pytest.raises(ValueError):make(True)(messages, schema(), out, **opts)
    assert len(calls) == 1 and not out.with_suffix('.explicit_recovery_claim.json').exists()


def test_existing_exclusive_recovery_claim_prevents_resend(tmp_path, monkeypatch):
    make, out, messages, opts, calls, _, _, _, _, _ = fixture(tmp_path, monkeypatch)
    out.with_suffix('.explicit_recovery_claim.json').write_text('{}')
    with pytest.raises(FileExistsError):make(True)(messages, schema(), out, **opts)
    assert len(calls) == 1


def test_preceding_cache_retains_application_validation(tmp_path, monkeypatch):
    wrapper, out, messages, opts, calls, validations = setup(tmp_path, monkeypatch)
    wrapper(messages, schema(), out, **opts)
    auth = tmp_path / 'authorization.json'
    auth.write_text(json.dumps(dict(output_relative_path='versions/UG05/teacher/advice_R02/increase.json',
        successful_outputs={out.relative_to(tmp_path).as_posix():dict(output=recovery.sha(out), receipt=recovery.sha(out.with_suffix('.receipt.json')))})))
    recovered = recovery.recovery_call(runtime, call_json, tmp_path, 'UG05', SKILLS,
        runtime.runtime_binding(SKILLS), auth, lambda event:None)
    assert recovered(messages, schema(), out, **opts) == answer()
    assert len(calls) == 1 and len(validations) == 2
    opts['response_validator'] = lambda result: (_ for _ in ()).throw(ValueError('application check'))
    with pytest.raises(ValueError, match='application check'):recovered(messages, schema(), out, **opts)
    assert len(calls) == 1
