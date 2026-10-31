import json
from pathlib import Path

import httpx
import jsonschema
import pytest

from agenticgen.providers.deepseek_tools import call_with_tools, RoundBudget, SessionStopped
from agenticgen.providers.strict_tools import FINAL_TOOL, ENDPOINT, definitions, schema_for_wire
from agenticgen.uniguide.io import read


SCHEMA = dict(type="object", properties=dict(
    decision=dict(type="string", enum=["propose", "hold"]),
    selected=dict(type="array", maxItems=3, uniqueItems=True,
                  items=dict(type="string", enum=["candidate_A", "candidate_B"])),
    reason=dict(type="string", minLength=1, maxLength=3000),
), required=["decision", "selected", "reason"], additionalProperties=False)
ANSWER = dict(decision="propose", selected=["candidate_B"], reason="A potentially better unmeasured path.")


class Registry:
    definitions = [dict(type="function", function=dict(name="measure", description="Existing evidence only",
        parameters=dict(type="object", properties={}, required=[], additionalProperties=False)))]
    binding = {"fixture": True}

    def __init__(self):
        self.receipt_ids = set()
        self.executed = []

    def execute(self, name, args):
        self.executed.append(name)
        self.receipt_ids.add("ev_existing_fixture")
        return dict(evidence_id="ev_existing_fixture", status="complete")


def call(name=FINAL_TOOL, args=None, identifier="submission"):
    return dict(id=identifier, type="function", function=dict(
        name=name, arguments=json.dumps(ANSWER if args is None else args)))


def wire(calls=None, content=None, finish=None):
    return dict(model="deepseek-flash", id="mock-strict", usage=dict(completion_tokens=20),
        choices=[dict(finish_reason=finish or ("tool_calls" if calls else "stop"),
            message=dict(role="assistant", content=content, reasoning_content="opaque-kept",
                         **({"tool_calls": calls} if calls else {})))])


def validate(answer):
    jsonschema.validate(answer, SCHEMA)
    if (answer["decision"] == "hold") != (not answer["selected"]):
        raise jsonschema.ValidationError("hold must explicitly select []")


def run(tmp_path, monkeypatch, responses, *, registry=None, **kwargs):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fixture-not-a-key")
    seen = []

    def handle(request):
        assert str(request.url) == ENDPOINT
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=responses.pop(0))

    answer = call_with_tools([dict(role="user", content="Fixture, not a scientific measurement")],
        SCHEMA, tmp_path/"answer.json", registry=registry or Registry(),
        budget=RoundBudget(tmp_path/"budget", kwargs.pop("max_calls", 30)),
        response_validator=validate, final_response_mode="strict_tool_v1",
        transport=httpx.MockTransport(handle), **kwargs)
    return answer, seen


def test_strict_wire_requires_selected_and_native_submission_is_not_evidence(tmp_path, monkeypatch):
    registry = Registry()
    answer, seen = run(tmp_path, monkeypatch, [wire([call()])], registry=registry, max_calls=1)
    assert answer == ANSWER and registry.executed == [] and not registry.receipt_ids
    request = seen[0]
    assert request["thinking"] == {"type": "enabled"} and request["reasoning_effort"] == "high"
    assert request["max_tokens"] == 16384 and request["tool_choice"] == "auto"
    assert all(t["function"]["strict"] for t in request["tools"])
    final = next(t["function"] for t in request["tools"] if t["function"]["name"] == FINAL_TOOL)
    assert final["parameters"]["required"] == ["decision", "selected", "reason"]
    assert final["parameters"]["properties"]["selected"]["items"]["enum"] == ["candidate_A", "candidate_B"]
    receipt = read(tmp_path/"answer.receipt.json")
    assert receipt["final_tool_call_id"] == "submission" and receipt["interface_repairs"] == 0
    assert receipt["evidence_ids"] == []
    binding = read(tmp_path/"answer.session/BINDING.json")
    assert binding["response_schema"]["properties"]["reason"]["maxLength"] == 3000
    assert binding["final_response_transport"]["client_only_checks"]


def test_evidence_roundtrip_then_final_submission_and_durable_replay(tmp_path, monkeypatch):
    answer, seen = run(tmp_path, monkeypatch, [wire([call("measure", {}, "read")]), wire([call()])])
    assert answer == ANSWER and len(seen) == 2
    assert seen[1]["messages"][-2]["reasoning_content"] == "opaque-kept"
    assert seen[1]["messages"][-1]["tool_call_id"] == "read"
    replayed, replay_calls = run(tmp_path, monkeypatch, [])
    assert replayed == answer and replay_calls == []
    assert len(list((tmp_path/"budget").glob("call_*.json"))) == 2


@pytest.mark.parametrize("invalid", [
    {"decision": "propose", "reason": "The explanation mentions candidate_B but selects nothing."},
    dict(ANSWER, selected=None), dict(ANSWER, selected="candidate_B"),
    dict(ANSWER, selected=["unknown"]), dict(ANSWER, selected=["candidate_B"]*2),
    dict(ANSWER, decision="hold"), dict(ANSWER, selected=[]),
    dict(ANSWER, reason="字"*3001), dict(ANSWER, extra="not allowed"),
])
def test_bad_native_arguments_need_explicit_model_replacement_not_inference(tmp_path, monkeypatch, invalid):
    result, requests = run(tmp_path, monkeypatch, [wire([call(args=invalid)]), wire([call()])])
    assert result == ANSWER and len(requests) == 2
    assert read(tmp_path/"answer.receipt.json")["interface_repairs"] == 1
    # Native rejected calls must receive real tool responses before a repair.
    assert requests[1]["messages"][-2]["role"] == "tool"
    assert requests[1]["messages"][-3]["reasoning_content"] == "opaque-kept"
    assert read(tmp_path/"answer.session/step_0001/TOOL_00.json")["result"]["status"] == "final_submission_rejected"


@pytest.mark.parametrize("text", [json.dumps(ANSWER), '{"decision":"hold","reason":"Nothing selected"}'])
def test_plain_content_cannot_silently_become_a_selection(tmp_path, monkeypatch, text):
    result, requests = run(tmp_path, monkeypatch, [wire(content=text), wire([call()])])
    assert result == ANSWER and len(requests) == 2
    assert FINAL_TOOL in requests[1]["messages"][-1]["content"]


def test_repeated_missing_selected_preserves_failure_and_does_not_replay(tmp_path, monkeypatch):
    invalid = dict(decision="propose", reason="candidate_B is a promising path.")
    with pytest.raises(SessionStopped, match="allowance exhausted"):
        run(tmp_path, monkeypatch, [wire([call(args=invalid)]), wire([call(args=invalid)])])
    assert not (tmp_path/"answer.json").exists()
    assert (tmp_path/"answer.session/FAILED.json").exists()
    assert len(list((tmp_path/"budget").glob("call_*.json"))) == 2
    with pytest.raises(SessionStopped, match="Archived"):
        run(tmp_path, monkeypatch, [])


def test_3000_character_reason_and_explicit_empty_hold_are_valid(tmp_path, monkeypatch):
    answer = dict(decision="hold", selected=[], reason="理"*3000)
    result, requests = run(tmp_path, monkeypatch, [wire([call(args=answer)])])
    assert result == answer and len(requests) == 1


def test_mixed_final_and_evidence_batch_is_not_executed(tmp_path, monkeypatch):
    registry = Registry()
    answer, requests = run(tmp_path, monkeypatch,
        [wire([call("measure", {}, "evidence"), call()]), wire([call()])], registry=registry)
    assert answer == ANSWER and registry.executed == []
    assert [m["tool_call_id"] for m in requests[1]["messages"] if m["role"] == "tool"] == ["evidence", "submission"]


def test_truncation_is_terminal_without_format_or_token_increase(tmp_path, monkeypatch):
    with pytest.raises(SessionStopped, match="incomplete"):
        run(tmp_path, monkeypatch, [wire(content="{", finish="length")])
    assert len(list((tmp_path/"budget").glob("call_*.json"))) == 1


def test_compiler_preserves_source_schema_and_fails_unknown_constraints():
    before = json.dumps(SCHEMA)
    tools, checks = definitions(Registry.definitions, SCHEMA)
    assert json.dumps(SCHEMA) == before and any(c["keyword"] == "maxLength" for c in checks)
    assert tools[-1]["function"]["parameters"]["required"] == SCHEMA["required"]
    with pytest.raises(ValueError, match="Unsupported"):
        schema_for_wire(dict(SCHEMA, oneOf=[{}]))
    with pytest.raises(ValueError, match="all declared fields"):
        schema_for_wire(dict(SCHEMA, required=["decision", "reason"]))


@pytest.mark.parametrize("job", ["03_prefix_joint_6pyy", "06_five_windows_3w54"])
def test_archived_missing_selected_payload_is_never_implicitly_completed(tmp_path, monkeypatch, job):
    evidence = Path(__file__).resolve().parents[2]/"docs/experiments/20261031_guidance_comparison5/monitoring/20261031_2017/llm_failure_contract_validation.json"
    failure = next(row for row in read(evidence) if row["job"] == job)
    original = failure["responses"][0]["parsed_response"]
    assert "selected" not in original and len(original["reason"]) < 3000
    result, requests = run(tmp_path, monkeypatch,
        [wire([call(args=original)]), wire([call()])])
    assert result["selected"] == ["candidate_B"] and len(requests) == 2
    rejected = read(tmp_path/"answer.session/step_0001/TOOL_00.json")
    assert rejected["call"]["function"]["arguments"] == json.dumps(original)
    assert "selected" in str(rejected["result"]["validation"])


def test_both_actual_council_roles_use_native_strict_schema_and_role_specific_payload(tmp_path, monkeypatch):
    from test_linear_guidance import prepared, action
    from agenticgen.uniguide import linear_council, linear_scale as scale
    from agenticgen.uniguide.absolute_coordinate import POLICY
    from agenticgen.uniguide.tool_runtime import configure, assets
    c, _, domain, base, _, _ = prepared(tmp_path, monkeypatch)
    c = configure(dict(c, protocol=POLICY, agent_tools=dict(c["agent_tools"], skill_release="2.8.2")))
    domain = scale.seal({**{k: v for k, v in domain.items() if k != "domain_sha256"},
                         "exploration_policy": c["guidance_exploration"]})
    skills = tmp_path/"skills"
    skills.mkdir()
    for name, path in assets(c).items():
        (skills/name).write_bytes(path.read_bytes())
    proposed = dict(action(base), exploration_tier="focused", improvement_reason="潜"*1000,
                    uncertainty="仍需完整人口测量。")
    ledger_before = (tmp_path/"attempts.json").read_bytes()
    seen = []

    def handle(request):
        body = json.loads(request.content)
        seen.append(body)
        context = json.loads(body["messages"][1]["content"])
        contract = context["final_submission_contract"]
        if contract["payload_field"] == "proposals":
            assert "candidate_ids" not in contract and contract["required_fields"] == ["decision", "proposals", "reason"]
            value = dict(decision="propose", proposals=[proposed], reason="理"*1000)
        else:
            assert contract["candidate_ids"] == [proposed["id"]]
            assert contract["required_fields"] == ["decision", "selected", "reason"]
            value = dict(decision="propose", selected=[proposed["id"]], reason="由"*1000)
        assert all(t["function"]["strict"] for t in body["tools"])
        return httpx.Response(200, json=wire([call(args=value)]))

    monkeypatch.setenv("DEEPSEEK_API_KEY", "fixture-not-a-key")
    result = linear_council.advise(c, [base], domain, tmp_path/"advice", skills,
        provider=lambda *a, **kw: call_with_tools(*a, **kw, transport=httpx.MockTransport(handle)))
    assert result["selected"] == [proposed["id"]] and len(result["actions"]) == 1 and len(seen) == 2
    assert (tmp_path/"attempts.json").read_bytes() == ledger_before


def test_release_282_is_immutable_and_registers_strict_transport(tmp_path, monkeypatch):
    from agenticgen.uniguide.tool_runtime import configure, assets
    from agenticgen.uniguide.absolute_coordinate import POLICY
    from agenticgen.uniguide.io import sha
    root = Path(__file__).resolve().parents[2]/"src/agenticgen/resources/skill_releases/guidance"
    release = read(root/"2.8.2/RELEASE.json")
    for name, digest in release["files_sha256"].items():
        assert sha(root/"2.8.2"/name) == digest
    for name, digest in release["old_files_sha256"].items():
        assert sha(root/"2.8.1"/name) == digest
    c = configure(dict(root=str(tmp_path), protocol=POLICY, agent_tools=dict(skill_release="2.8.2")))
    assert c["llm"]["final_response_mode"] == "strict_tool_v1" and len(assets(c)) == 4
    with pytest.raises(ValueError, match="strict final"):
        configure(dict(c, llm=dict(c["llm"], final_response_mode="text")))


@pytest.mark.parametrize("prefix,window_bounds", [
    (True, [[10, 25], [26, 50], [51, 75], [76, 99]]),
    (False, [[10, 20], [21, 30], [31, 40], [41, 50], [51, 99]]),
])
def test_new_release_validates_both_rerun_interfaces(tmp_path, prefix, window_bounds):
    from agenticgen.uniguide.absolute_runtime import validate_config
    root = Path(__file__).resolve().parents[2]/"configs"
    config = read(root/"uniguide_absolute_coordinate.json")
    config.update(read(root/"uniguide_linear_control.json"))
    overlay = read(root/"uniguide_selected_contract_retest.json")
    llm = dict(config["llm"], **overlay["llm"])
    config.update(overlay)
    config.update(root=str(tmp_path/"new"), generation_context_root=str(tmp_path/"frozen"),
        llm=llm, windows=window_bounds, prefix_inpainting=dict(enabled=prefix))
    configured = validate_config(config)
    assert configured["llm"]["final_response_mode"] == "strict_tool_v1"
    assert configured["windows"] == window_bounds and configured["prefix_inpainting"]["enabled"] is prefix
    assert configured["population"] == 50 and configured["rounds"] == 5
