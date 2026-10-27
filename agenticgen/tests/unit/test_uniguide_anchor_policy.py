"""Executable identities are distinct from compression and calibration labels."""
import importlib.util
import json

import httpx
import jsonschema
import pytest

from test_uniguide_prompt_policy import ROOT, SKILLS, sample_context, schema, answer, case

spec = importlib.util.spec_from_file_location("anchor_policy", ROOT / "scripts/uniguide_anchor_policy.py")
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)


def authorize(case):
    output, messages, contract, old_auth, old_wrapper, options, calls, validated, events = case
    path = old_auth.with_name("anchor_authorization.json")
    data = policy.read_json(old_auth)
    data.update(policy=policy.POLICY, policy_sha256=policy.file_sha(policy.__file__),
        legacy_policy_sha256=policy.file_sha(policy.legacy.__file__),
        legacy_authorization_path=str(old_auth), legacy_authorization_sha256=policy.file_sha(old_auth))
    path.write_text(json.dumps(data))
    from agenticgen.providers.deepseek_api import call_json
    wrapped = policy.wrap_anchor_revision(old_wrapper, call_json, events.append, old_auth.parent, path, "UG04", SKILLS)
    return path, wrapped


def test_compact_history_still_exposes_actual_ids_and_unmodified_evidence():
    context = sample_context()
    context["history"][0]["id"] = "R0/3w54_RNB_A_504"
    context["calibration_knowledge"]["same_schedule"] = dict(context["history"][0], id="calibration/full/3w54_RNB_A_504/p000")
    messages = [dict(role="system", content="old"), dict(role="user", content=json.dumps(context))]
    original = schema()
    before = policy.canonical(original)
    rendered, revised, metrics = policy.transform(messages, original, SKILLS)
    packed = json.loads(rendered[1]["content"])
    assert "$ref" in packed["context"]["history"][0]
    assert policy.canonical(policy.legacy.unpack_context(packed)) == policy.canonical(context)
    ids = ["R0/3w54_RNB_A_504"]
    assert packed["execution_contract"]["allowed_anchor_ids"] == ids
    assert packed["execution_contract"]["measured_anchors"][0]["schedule"] == context["history"][0]["schedule"]
    assert revised["properties"]["proposals"]["items"]["properties"]["anchor_id"]["enum"] == ids
    assert policy.canonical(original) == before
    for wrong in ["r80", "calibration/full/3w54_RNB_A_504/p000"]:
        bad = answer()
        for item in bad["proposals"]: item["anchor_id"] = wrong
        with pytest.raises(jsonschema.ValidationError) as error: jsonschema.validate(bad, revised)
        assert list(error.value.path)[-1] == "anchor_id" and error.value.validator == "enum"


@pytest.mark.parametrize("bad_id", ["r80", "calibration/full/3w54_RNB_A_504/p000"])
def test_one_repair_has_exact_anchor_enum_and_original_action_validation(case, bad_id):
    output, messages, contract, _, _, options, calls, validated, _ = case
    _, wrapped = authorize(case)
    bad = answer()
    for item in bad["proposals"]: item["anchor_id"] = bad_id
    def http(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json=dict(model="deepseek-flash", usage={}, choices=[dict(
            finish_reason="stop", message=dict(content=json.dumps(bad if len(calls) == 1 else answer())))]))
    options["transport"] = httpx.MockTransport(http)
    assert wrapped(messages, contract, output, **options) == answer()
    assert len(calls) == 2 and len(validated) == 1
    repair = calls[1]["messages"][-1]["content"]
    assert '"enum":["anchor"]' in repair and "anchor_id" in repair
    assert all(call["max_tokens"] == 16384 for call in calls)
    # A success must validate the original effective cache, without another HTTP.
    assert wrapped(messages, contract, output, **options) == answer()
    assert len(calls) == 2 and len(validated) == 2


def test_failed_v1_claim_is_preserved_and_v2_failure_cannot_retry(case):
    from agenticgen.providers.deepseek_api import LLMInterfaceError
    output, messages, contract, _, _, options, calls, _, _ = case
    old_folder = output.parent / (output.stem + ".promptcompact_v1")
    old_folder.mkdir()
    old_claim = old_folder / "authorization_claim.json"
    old_claim.write_text('{"preserve":"failed-v1"}')
    before = old_claim.read_bytes()
    _, wrapped = authorize(case)
    def http(request):
        calls.append(json.loads(request.content))
        bad = answer()
        for item in bad["proposals"]: item["anchor_id"] = "r80"
        return httpx.Response(200, json=dict(model="deepseek-flash", usage={}, choices=[dict(
            finish_reason="stop", message=dict(content=json.dumps(bad)))]))
    options["transport"] = httpx.MockTransport(http)
    with pytest.raises(LLMInterfaceError): wrapped(messages, contract, output, **options)
    with pytest.raises(FileExistsError): wrapped(messages, contract, output, **options)
    assert len(calls) == 2 and old_claim.read_bytes() == before and not output.exists()


def test_successful_v1_uses_unchanged_original_provider_cache(case):
    output, messages, contract, old_auth, old_wrapper, options, calls, validated, events = case
    old_wrapper(messages, contract, output, **options)
    before = {p: p.read_bytes() for p in output.parent.rglob("*") if p.is_file()}
    auth, _ = authorize(case)
    data = policy.read_json(auth)
    data["successful_outputs"] = {output.relative_to(old_auth.parent).as_posix(): dict(
        output_sha256=policy.file_sha(output), receipt_sha256=policy.file_sha(output.with_suffix(".receipt.json")))}
    auth.write_text(json.dumps(data))
    from agenticgen.providers.deepseek_api import call_json
    wrapped = policy.wrap_anchor_revision(old_wrapper, call_json, events.append, old_auth.parent, auth, "UG04", SKILLS)
    assert wrapped(messages, contract, output, **options) == answer()
    assert len(calls) == 1 and len(validated) == 2
    assert all(p.read_bytes() == raw for p, raw in before.items())
    assert not (output.parent / (output.stem + ".anchorids_v2")).exists()


@pytest.mark.parametrize("kind", ["cap", "schema", "original", "legacy_authorization", "authorization", "scope"])
def test_binding_changes_fail_before_network(case, kind):
    output, messages, contract, old_auth, _, options, calls, _, _ = case
    auth, wrapped = authorize(case)
    if kind == "cap": options["max_output_tokens"] = 24576
    if kind == "schema": contract["additionalProperties"] = True
    if kind == "original": messages[1]["content"] = '{}'
    if kind == "legacy_authorization": old_auth.write_text('{}')
    if kind == "authorization": auth.write_text('{}')
    if kind == "scope": output = output.parents[3] / "UG05/teacher/advice_R04/redistribute.json"
    with pytest.raises(ValueError): wrapped(messages, contract, output, **options)
    assert calls == []


def test_synthesis_selection_uses_candidate_keys_not_compression_labels():
    from agenticgen.uniguide.council import object_schema, TEXT
    context = sample_context()
    context.pop("branch")
    context["candidates"] = {branch + ":0": answer()["proposals"][0]
                             for branch in ("increase", "decrease", "redistribute")}
    selected = {branch: {"enum": [branch + ":0"]} for branch in ("increase", "decrease", "redistribute")}
    contract = object_schema(dict(analysis=TEXT, choice_evidence=TEXT, selected=object_schema(selected)))
    rendered, revised, _ = policy.transform([dict(role="system", content="old"),
        dict(role="user", content=json.dumps(context))], contract, SKILLS)
    assert revised["properties"]["selected"] == contract["properties"]["selected"]
    index = json.loads(rendered[1]["content"])["execution_contract"]
    assert index["candidate_ids_by_branch"] == {k: v["enum"] for k, v in selected.items()}
    assert policy.legacy.unpack_context(json.loads(rendered[1]["content"])) == context
