import copy
import json
import shutil
from pathlib import Path

import pytest

from evomolsteer.continuous import m2_skill_api as api
from evomolsteer.io import read_json, write_json

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def prepared(tmp_path):
    repo = tmp_path / "repo"
    for name in (api.BASE, api.REF, api.SOURCE):
        target = repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / name, target)
    shutil.copytree(REPO / api.CURRENT_PLAN, repo / api.CURRENT_PLAN)
    # The parent protocol is bound for provenance, not used as the new plan.
    target = repo / api.PLAN / "protocol.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REPO / api.PLAN / "protocol.json", target)
    study = repo / "study"
    packet = api.prepare(repo, study)
    return repo, study, packet


def test_plan_activation_preserves_matrix_and_selection_gates(prepared):
    repo, study, packet = prepared
    activation = api.validate_activation_policy(repo, study)
    assert activation["condition_matrix_rows"] == 46
    assert len(activation["conditions"]) == 17
    assert len(activation["deferred_conditions"]) == 29
    assert "A_full_G" in {r["condition_id"] for r in activation["conditions"]}
    assert "U_scalar_noop" in {r["condition_id"] for r in activation["deferred_conditions"]}
    assert {r["phase"] for r in activation["conditions"]} == {"P0", "P1"}
    fixed = {k: read_json(repo / api.BASE)[k] for k in api.TEMPERATURE_KEYS}
    assert packet["fixed_parameters"] == fixed
    assert all(k not in packet["allowed_G"] and k not in packet["allowed_U"] for k in fixed)


def test_real_role_assembler_delivers_fixed_policy_and_revised_D2(prepared, monkeypatch):
    repo, study, packet = prepared
    row = next(r for r in read_json(study / "activation.json")["conditions"] if r["condition_id"] == "P_struct")
    captured = []
    def mock_call(messages, schema, output, **kwargs):
        assert not any(k in schema["properties"]["edit"]["properties"]["parameter"]["enum"] for k in api.TEMPERATURE_KEYS)
        captured.append(messages)
        response = {"role": "Designer", "decision": "retain", "edit": None,
                    "evidence_ids": [], "rationale": "Keep the incumbent in this interface fixture."}
        write_json(output, response)
        return response
    monkeypatch.setattr(api, "call_json", mock_call)
    api.role_call(repo, study, row, 0, "Designer", analyst={"role": "Analyst", "findings": []})
    assert len(captured) == 1
    delivered = captured[0]
    assert "Temperatures are immutable task parameters" in delivered[0]["content"]
    d2 = next(m for m in read_json(study / "module_registry.json")["modules"] if m["id"] == "D2")
    assert d2["instruction"] in delivered[0]["content"]
    assert "mixture temperature" not in d2["instruction"]
    assert "temperature" not in d2["alternative_instruction"]
    assert json.loads(delivered[1]["content"])["fixed_parameters"] == packet["fixed_parameters"]
    assert not any(k in json.loads(delivered[1]["content"])["allowed_G"] for k in api.TEMPERATURE_KEYS)


@pytest.mark.parametrize("lane", ["G", "U"])
@pytest.mark.parametrize("parameter,value", [
    ("mixture_temperature", .25), ("mixture_temperature", .5),
    ("teacher_endpoint_temperature_A2", 2.), ("teacher_endpoint_temperature_A2", 4.)])
def test_temperature_edits_rejected_in_every_lane_before_program_write(prepared, lane, parameter, value):
    repo, study, _ = prepared
    response = {"decision": "edit", "edit": {"parameter": parameter, "value": value}, "evidence_ids": []}
    with pytest.raises(ValueError, match="Temperature edits are disabled"):
        api.compile_reward(repo, study, {"condition_id": "fixture", "lane": lane}, 0, response)
    assert not (study / "programs").exists()


def test_non_temperature_geometry_edit_retains_default_temperatures(prepared):
    repo, study, _ = prepared
    row = {"condition_id": "fixture", "lane": "G"}
    folder = study / "calls/fixture/chain_00"
    response = {"decision": "edit", "edit": {"parameter": "pointcloud_delta_A", "value": 2.}, "evidence_ids": []}
    write_json(folder / "Designer.json", response)
    (folder / "Designer.instructions.md").write_text("test fixture", encoding="utf-8")
    path, signature = api.compile_reward(repo, study, row, 0, response)
    program = read_json(path)
    assert program["pointcloud_delta_A"] == 2.
    assert api.validate_fixed_temperatures(repo, program) == read_json(study / "evidence.json")["fixed_parameters"]
    assert signature == api.effective_signature(program)


@pytest.mark.parametrize("parameter,value", [("mixture_temperature", .25), ("teacher_endpoint_temperature_A2", 2.)])
def test_direct_program_temperature_changes_are_rejected(parameter, value):
    program = copy.deepcopy(read_json(REPO / api.BASE))
    program[parameter] = value
    with pytest.raises(ValueError, match="frozen M2 defaults"):
        api.validate_fixed_temperatures(REPO, program)


def test_legacy_queue_cannot_resume_but_historical_signature_is_preserved():
    study = REPO / "docs/experiments/steer_dependency_20261009/deepseek_flash_m2_20261011"
    with pytest.raises(ValueError, match="Legacy study is archival"):
        api.validate_activation_policy(REPO, study)
    compiler = read_json(study / "calls/P_struct/chain_01/compiler.json")
    assert api.effective_signature(read_json(REPO / compiler["program"])) == compiler["effective_sha256"]


def test_p2_injection_fails_before_role_api_call(prepared, monkeypatch):
    repo, study, _ = prepared
    path = study / "activation.json"
    activation = read_json(path)
    activation["conditions"].append(next(r for r in activation["deferred_conditions"] if r["phase"] == "P2"))
    write_json(path, activation)
    def forbidden_call(*args, **kwargs):
        raise AssertionError("No API call may occur before the selection gate")
    monkeypatch.setattr(api, "call_json", forbidden_call)
    with pytest.raises(ValueError, match="Initial executor stops after P1"):
        api.role_call(repo, study, activation["conditions"][-1], 0, "Designer")


def test_unrecorded_module_change_cannot_keep_its_original_condition_label(prepared):
    repo, study, _ = prepared
    path = study / "activation.json"
    activation = read_json(path)
    row = next(r for r in activation["conditions"] if r["condition_id"] == "P_struct")
    row["designer_modules"] = "D1,D3,D4"
    write_json(path, activation)
    with pytest.raises(ValueError, match="master matrix"):
        api.validate_activation_policy(repo, study)


def test_temperature_permission_cannot_be_added_even_with_new_evidence_checksum(prepared):
    from evomolsteer.io import digest
    repo, study, _ = prepared
    evidence = read_json(study / "evidence.json")
    evidence["allowed_G"]["mixture_temperature"] = [.25, .5, 1.]
    write_json(study / "evidence.json", evidence)
    activation = read_json(study / "activation.json")
    activation["evidence_sha256"] = digest(study / "evidence.json")
    write_json(study / "activation.json", activation)
    with pytest.raises(ValueError, match="contradicts fixed-temperature"):
        api.validate_activation_policy(repo, study)
