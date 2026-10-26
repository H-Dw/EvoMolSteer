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
