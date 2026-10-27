"""Recovery must preserve the original wire bytes and all scientific guards."""
import importlib.util
import json
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[2] / "scripts/resume_uniguide_cached_requests.py"
spec = importlib.util.spec_from_file_location("ug_request_recovery", path)
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


def message(content, role="user"):
    return {"role": role, "content": content}


def test_only_object_order_uses_exact_original():
    original = [message("system", "system"), message('{"b":{"z":2,"a":1},"a":[3,4]}')]
    rebuilt = [message("system", "system"), message('{"a":[3,4],"b":{"a":1,"z":2}}')]
    assert recovery.original_messages(rebuilt, original) is original


@pytest.mark.parametrize("new,old", [
    ('{"a":[4,3]}', '{"a":[3,4]}'),
    ('{"a":true}', '{"a":1}'),
    ('{"a":1.0}', '{"a":1}'),
    ('{"a":2}', '{"a":1}'),
    ('{"a":1,"a":1}', '{"a":1}'),
    ('{"b":NaN,"a":1}', '{"a":1,"b":NaN}'),
    ('{"a":"new"}', '{"a":"old"}'),
    ('[1,2]', '[1, 2]'),
])
def test_reject_changed_inputs(new, old):
    with pytest.raises(ValueError):
        recovery.original_messages([message(new)], [message(old)])


def test_system_text_and_message_count_remain_exact():
    with pytest.raises(ValueError):
        recovery.original_messages([message('{"b":2,"a":1}', "system")],
                                   [message('{"a":1,"b":2}', "system")])
    with pytest.raises(ValueError):
        recovery.original_messages([], [message("content")])


def test_provider_receives_original_options_and_checks_cache(tmp_path, monkeypatch):
    from agenticgen.providers.deepseek_api import call_json
    from agenticgen.runtime.jsonio import write_json, digest
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    output = tmp_path / "response.json"
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
    original = [message('{"z":2,"a":1}')]
    request = output.with_suffix(".request.json")
    write_json(request, {"provider": "DeepSeek", "response_schema": schema,
        "body": {"model": "deepseek-flash", "messages": original, "thinking": {"type": "disabled"},
                 "temperature": 0, "max_tokens": 8192, "response_format": {"type": "json_object"}}})
    write_json(output, {"ok": True})
    write_json(output.with_suffix(".receipt.json"), {"request_sha256": digest(request),
                "response_sha256": digest(output)})
    before = {str(p): p.read_bytes() for p in tmp_path.iterdir()}
    events = []
    wrapped = recovery.wrap_call(call_json, events.append)
    rebuilt = [message('{"a":1,"z":2}')]
    assert wrapped(rebuilt, schema, output) == {"ok": True}
    assert events[0]["event"] == "reuse_original_serialization"
    assert before == {str(p): p.read_bytes() for p in tmp_path.iterdir()}
    with pytest.raises(ValueError, match="different instructions"):
        wrapped(rebuilt, schema, output, max_output_tokens=16384)
    with pytest.raises(ValueError, match="Scientific request content"):
        wrapped([message('{"a":2,"z":2}')], schema, output)
    write_json(output, {"ok": False})
    with pytest.raises(ValueError, match="checksum"):
        wrapped(rebuilt, schema, output)


def test_fresh_call_untouched_and_errors_not_retried(tmp_path):
    count = []
    messages = [message('{"b":2,"a":1}')]
    def provider(received, schema, output, **kwargs):
        count.append(1)
        assert received is messages
        assert kwargs["max_output_tokens"] == 16384
        raise ValueError("Incomplete JSON answer")
    with pytest.raises(ValueError, match="Incomplete JSON"):
        recovery.wrap_call(provider, lambda _: None)(messages, {}, tmp_path / "fresh.json", max_output_tokens=16384)
    assert len(count) == 1


@pytest.fixture
def token_case(tmp_path, monkeypatch):
    import httpx
    from agenticgen.providers.deepseek_api import call_json
    from agenticgen.runtime.jsonio import write_json
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-only")
    output = tmp_path / "versions/UG04/teacher/advice_R03/redistribute.json"
    output.parent.mkdir(parents=True)
    messages = [message("contract", "system"), message('{"b":2,"a":1}')]
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
    body = dict(model="deepseek-flash", messages=messages, thinking={"type": "enabled"},
                reasoning_effort="high", max_tokens=16384, response_format={"type": "json_object"})
    request = output.with_suffix(".request.json")
    write_json(request, dict(provider="DeepSeek", body=body, response_schema=schema))
    output.with_suffix(".incomplete_1.txt").write_text("")
    auth = tmp_path / "authorization.json"
    write_json(auth, dict(version="UG04", output_relative_path=str(output.relative_to(tmp_path)),
        original_request_sha256=recovery.sha(request), old_max_output_tokens=16384,
        new_max_output_tokens=24576, user_instruction="Increase output cap by 50%"))
    events, calls = [], []
    validated = []
    def http(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json=dict(model="deepseek-flash", id="test-call", usage={},
            choices=[dict(finish_reason="stop", message=dict(content='{"ok":true}'))]))
    wrapped = recovery.wrap_call(recovery.wrap_token_override(call_json, events.append,
        tmp_path, auth, "UG04"), events.append)
    options = dict(thinking=True, reasoning_effort="high", max_output_tokens=16384,
        transport=httpx.MockTransport(http), response_validator=lambda x: validated.append(x))
    return output, messages, schema, auth, wrapped, options, calls, validated, events


def test_authorized_call_only_changes_cap_preserves_source_and_replays(token_case):
    output, messages, schema, auth, wrapped, options, calls, validated, events = token_case
    before = {p: p.read_bytes() for p in output.parent.iterdir()}
    assert wrapped(messages, schema, output, **options) == {"ok": True}
    assert len(calls) == 1 and calls[0]["max_tokens"] == 24576
    old = recovery.read(output.with_suffix(".request.json"))["body"]
    assert {k: v for k, v in calls[0].items() if k not in ("max_tokens", "messages")} == {
        k: v for k, v in old.items() if k not in ("max_tokens", "messages")}
    assert calls[0]["messages"][:-1] == messages
    assert all(p.read_bytes() == data for p, data in before.items())
    receipt = recovery.read(output.with_suffix(".receipt.json"))
    assert receipt["max_output_tokens"] == 24576
    assert receipt["request_sha256"] != recovery.sha(output.with_suffix(".request.json"))
    assert receipt["protocol_deviation"]["original_request_sha256"] == recovery.sha(output.with_suffix(".request.json"))
    assert recovery.sha(output.parent / receipt["wire_request_path"]) == receipt["wire_request_sha256"]
    rebuilt = [messages[0], message('{"a":1,"b":2}')]
    assert wrapped(rebuilt, schema, output, **options) == {"ok": True}
    assert len(calls) == 1 and len(validated) == 2
    # A different call stays at the frozen limit.
    assert wrapped(messages, schema, output.parent / "synthesis.json", **options) == {"ok": True}
    assert calls[-1]["max_tokens"] == 16384


@pytest.mark.parametrize("change", ["message", "schema", "model", "thinking", "effort", "cap", "request", "authorization"])
def test_override_rejects_unapproved_changes(token_case, change):
    output, messages, schema, auth, wrapped, options, calls, _, _ = token_case
    if change == "message": messages = [messages[0], message('{"b":3,"a":1}')]
    if change == "schema": schema = dict(schema, additionalProperties=False)
    if change == "model": options["model"] = "different"
    if change == "thinking": options["thinking"] = False
    if change == "effort": options["reasoning_effort"] = "low"
    if change == "cap": options["max_output_tokens"] = 24576
    if change == "request": output.with_suffix(".request.json").write_text("{}")
    if change == "authorization": auth.write_text("{}")
    with pytest.raises(ValueError): wrapped(messages, schema, output, **options)
    assert calls == []


def test_enlarged_length_failure_is_not_automatically_retried(token_case):
    import httpx
    output, messages, schema, _, wrapped, options, calls, _, _ = token_case
    def fail(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json=dict(model="deepseek-flash", usage={}, choices=[
            dict(finish_reason="length", message=dict(content=""))]))
    options["transport"] = httpx.MockTransport(fail)
    with pytest.raises(ValueError, match="Incomplete"):
        wrapped(messages, schema, output, **options)
    with pytest.raises(FileExistsError):
        wrapped(messages, schema, output, **options)
    assert len(calls) == 1 and not output.exists()


@pytest.mark.parametrize("corrupt", ["alias", "receipt", "effective"])
def test_override_cache_and_alias_tampering_is_rejected(token_case, corrupt):
    output, messages, schema, _, wrapped, options, calls, _, _ = token_case
    wrapped(messages, schema, output, **options)
    path = {"alias": output, "receipt": output.with_suffix(".receipt.json"),
            "effective": output.parent / "redistribute.token24576/response.json"}[corrupt]
    path.write_text('{"ok":false}')
    with pytest.raises(ValueError): wrapped(messages, schema, output, **options)
    assert len(calls) == 1
