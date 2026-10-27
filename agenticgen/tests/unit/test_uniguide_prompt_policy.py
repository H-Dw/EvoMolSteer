"""Prompt revision keeps evidence and scientific validation, isolates new calls."""
import importlib.util
import json
from pathlib import Path

import httpx
import jsonschema
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("policy", ROOT / "scripts/uniguide_prompt_policy.py")
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)
SKILLS = ROOT / "src/agenticgen/uniguide/skills"


def sample_context():
    window = dict(zip(policy.WINDOW_COLUMNS, [[10, 25], .1, .2, .3, .4, .5, .6, .7, -.8, -.9]))
    row = dict(id="anchor", schedule=[128., 128., 128., 128.], complete=True, C=.5,
               gap=1., attempts=50, window_diagnostics=[window] * 4)
    return dict(branch="redistribute", round=4, teacher=dict(target="teacher"), history=[row],
        default_anchor=row, comparison_best=row, calibration_knowledge=dict(failed=[dict(
            complete=False, attempts=25, error="not hidden")]), no_future_execution=False)


def schema():
    from agenticgen.uniguide.council import object_schema, TEXT, ACTION
    return object_schema(dict(analysis=TEXT, mechanism_uncertainty=TEXT,
        proposals=dict(type="array", items=ACTION, minItems=2, maxItems=2)))


def answer():
    return dict(analysis="A concise comparison", mechanism_uncertainty="Unresolved trajectory effect",
        proposals=[dict(id="p"+str(i), branch="redistribute", anchor_id="anchor", anchor_reason="W1 up; W2 down",
            mode="coarse", schedule=s, evidence="anchor measured", prediction="C lower", falsifier="C higher")
            for i, s in enumerate([[256., 64., 128., 128.], [64., 256., 128., 128.]])])


def test_lossless_factorization_preserves_failures_types_values_and_order():
    context = sample_context()
    context["edge"] = [1, 1., True, None, "1", [False, 0], {"x": -1e-30}]
    packed = policy.pack_context(context)
    assert policy.canonical(policy.unpack_context(packed)) == policy.canonical(context)
    assert len(json.dumps(packed)) < len(json.dumps(context))
    assert packed["shared"]
    with pytest.raises(ValueError):
        policy.pack_context({"$ref": "fake"})


def test_schema_bounds_do_not_mutate_shared_text_or_action_contract():
    original = schema()
    before = policy.canonical(original)
    messages = [dict(role="system", content="old"), dict(role="user", content=json.dumps(sample_context()))]
    new, revised, metrics = policy.transform(messages, original, SKILLS)
    assert policy.canonical(original) == before
    assert revised["properties"]["analysis"]["maxLength"] == 1600
    assert revised["properties"]["mechanism_uncertainty"]["maxLength"] == 500
    props = revised["properties"]["proposals"]["items"]["properties"]
    assert "maxLength" not in props["id"]
    assert props["falsifier"]["maxLength"] == 280
    assert props["schedule"] == original["properties"]["proposals"]["items"]["properties"]["schedule"]
    assert "select supplied proposals" not in new[0]["content"]
    assert metrics["lossless_context_verified"]
    jsonschema.validate(answer(), revised)
    bad = answer(); bad["analysis"] = "x" * 1601
    with pytest.raises(jsonschema.ValidationError): jsonschema.validate(bad, revised)


@pytest.mark.parametrize("text", ['{"a":1,"a":2}', '{"a":NaN}', '{}{}', '```json\n{}\n```'])
def test_output_rejects_duplicate_nonfinite_appended_and_markdown(text):
    with pytest.raises(json.JSONDecodeError): policy.parse_output(text)


@pytest.fixture
def case(tmp_path, monkeypatch):
    from agenticgen.providers.deepseek_api import call_json
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-only")
    output = tmp_path / "versions/UG04/teacher/advice_R04/redistribute.json"
    output.parent.mkdir(parents=True)
    messages = [dict(role="system", content="original system"), dict(role="user", content=json.dumps(sample_context()))]
    original = dict(provider="DeepSeek", response_schema=schema(), body=dict(model="deepseek-flash",
        messages=messages, thinking=dict(type="enabled"), reasoning_effort="high", max_tokens=16384,
        response_format=dict(type="json_object")))
    output.with_suffix(".request.json").write_text(json.dumps(original))
    output.with_suffix(".incomplete_1.txt").write_text("old failure")
    auth = tmp_path / "authorization.json"
    authorization = dict(version="UG04", policy=policy.POLICY, scope="unfinished_calls_of_frozen_version",
        user_instruction="Inspect redundant prompts and fix guidance/output constraints", teachers=["teacher"],
        policy_sha256=policy.file_sha(policy.__file__), skill_sha256={p.name:policy.file_sha(p) for p in SKILLS.glob("*.md")},
        successful_outputs={}, first_output=str(output.relative_to(tmp_path)).replace("\\", "/"),
        first_request_sha256=policy.file_sha(output.with_suffix(".request.json")))
    auth.write_text(json.dumps(authorization))
    events, calls, validated = [], [], []
    def http(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json=dict(model="deepseek-flash", id="mock", usage={},
            choices=[dict(finish_reason="stop", message=dict(content=json.dumps(answer())))]))
    def validator(result):
        from agenticgen.uniguide.protocol import validate_action
        for item in result["proposals"]:
            try:
                validate_action(item, {"anchor":sample_context()["history"][0]}, dict(min_index=11,max_index=19), [[128.]*4])
            except ValueError as error:
                raise jsonschema.ValidationError(str(error)) from error
        validated.append(result)
    options = dict(thinking=True, reasoning_effort="high", max_output_tokens=16384,
        transport=httpx.MockTransport(http), response_validator=validator)
    wrapped = policy.wrap_prompt_revision(call_json,call_json,events.append,tmp_path,auth,"UG04",SKILLS)
    return output,messages,schema(),auth,wrapped,options,calls,validated,events


def test_effective_request_cache_alias_and_original_evidence(case):
    output,messages,schema,auth,wrapped,options,calls,validated,events=case
    before={p:p.read_bytes() for p in output.parent.iterdir()}
    assert wrapped(messages,schema,output,**options)==answer()
    assert len(calls)==1 and calls[0]["max_tokens"]==16384
    assert calls[0]["thinking"]=={"type":"enabled"} and calls[0]["reasoning_effort"]=="high"
    assert all(p.read_bytes()==raw for p,raw in before.items())
    assert policy.unpack_context(json.loads(calls[0]["messages"][1]["content"]))==sample_context()
    receipt=policy.read_json(output.with_suffix(".receipt.json"))
    assert receipt["protocol_deviation"]["kind"]=="prompt_revision"
    assert policy.file_sha(output.parent/receipt["raw_response_path"])==receipt["raw_response_sha256"]
    assert wrapped(messages,schema,output,**options)==answer()
    assert len(calls)==1 and len(validated)==2


def test_failed_new_prompt_cannot_automatically_retry(case):
    output,messages,schema,auth,wrapped,options,calls,_,_=case
    def failure(request):
        calls.append(1)
        return httpx.Response(200,json=dict(model="deepseek-flash",usage={},choices=[dict(finish_reason="length",message=dict(content=""))]))
    options["transport"]=httpx.MockTransport(failure)
    with pytest.raises(ValueError,match="Incomplete"):wrapped(messages,schema,output,**options)
    with pytest.raises(FileExistsError):wrapped(messages,schema,output,**options)
    assert len(calls)==1 and not output.exists()


@pytest.mark.parametrize("kind",["original","cap","thinking","effort","authorization","schema","scope"])
def test_unapproved_input_change_rejected_before_http(case,kind):
    output,messages,schema,auth,wrapped,options,calls,_,_=case
    if kind=="original":messages[1]["content"]='{"changed":true}'
    if kind=="cap":options["max_output_tokens"]=24576
    if kind=="thinking":options["thinking"]=False
    if kind=="effort":options["reasoning_effort"]="low"
    if kind=="authorization":auth.write_text("{}")
    if kind=="schema":schema["additionalProperties"]=True
    if kind=="scope":output=output.parents[3]/"UG05/teacher/advice_R04/redistribute.json"
    with pytest.raises(ValueError):wrapped(messages,schema,output,**options)
    assert calls==[]


@pytest.mark.parametrize("kind",["alias","receipt","raw"])
def test_effective_cache_tampering_rejected(case,kind):
    output,messages,schema,auth,wrapped,options,calls,_,_=case
    wrapped(messages,schema,output,**options)
    receipt=policy.read_json(output.with_suffix(".receipt.json"))
    path={"alias":output,"receipt":output.with_suffix(".receipt.json"),"raw":output.parent/receipt["raw_response_path"]}[kind]
    path.write_text("{}")
    with pytest.raises(ValueError):wrapped(messages,schema,output,**options)
    assert len(calls)==1


@pytest.mark.parametrize("kind",["duplicate","overlength","illegal_action"])
def test_existing_single_repair_enforces_new_format_and_original_action_guard(case,kind):
    output,messages,schema,auth,wrapped,options,calls,validated,_=case
    bad=answer()
    if kind=="overlength":bad["analysis"]="x"*1601
    if kind=="illegal_action":bad["proposals"][0]["schedule"]=[128.]*4
    invalid=json.dumps(bad)
    if kind=="duplicate":invalid=invalid[:-1]+',"analysis":"duplicate"}'
    def http(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200,json=dict(model="deepseek-flash",usage={},choices=[dict(
            finish_reason="stop",message=dict(content=invalid if len(calls)==1 else json.dumps(answer())))]))
    options["transport"]=httpx.MockTransport(http)
    assert wrapped(messages,schema,output,**options)==answer()
    assert len(calls)==2 and len(validated)==1
    assert len(policy.read_json(output.with_suffix(".receipt.json"))["interface_repairs"])==1


def test_existing_success_keeps_original_prompt_and_schema(case):
    from agenticgen.providers.deepseek_api import call_json
    output,messages,schema,auth,_,options,calls,validated,events=case
    call_json(messages,schema,output,**options)
    data=policy.read_json(auth)
    data["successful_outputs"][str(output.relative_to(auth.parent))] = dict(
        output_sha256=policy.file_sha(output),receipt_sha256=policy.file_sha(output.with_suffix(".receipt.json")))
    auth.write_text(json.dumps(data))
    wrapped=policy.wrap_prompt_revision(call_json,call_json,events.append,auth.parent,auth,"UG04",SKILLS)
    assert wrapped(messages,schema,output,**options)==answer()
    assert len(calls)==1
    assert not (output.parent/"redistribute.promptcompact_v1").exists()


def test_new_later_call_uses_revision_without_overwriting_existing_success(case):
    output,messages,schema,auth,wrapped,options,calls,_,_=case
    later=output.parent/"decrease.json"
    messages[1]["content"]=json.dumps(dict(sample_context(),branch="decrease"))
    assert wrapped(messages,schema,later,**options)==answer()
    assert len(calls)==1 and later.with_suffix(".request.json").exists()
