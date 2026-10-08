import copy
import gzip
import json
from pathlib import Path
import subprocess

import httpx
import pytest

from evomolsteer.continuous.flowcompat_agents import (
    audit_flow_configuration, call_api, compile_design, export_request,
    import_response, payload, run_registered_tool, validate_response,
    verify_execution_response,
)
from evomolsteer.io import digest, read_json, write_json


@pytest.fixture
def bound(tmp_path):
    """Real deterministic subprocess receipts with tiny stand-in tool data."""
    root = tmp_path
    for role in ("analyst", "designer", "flow-compatibility"):
        path = root / "skills" / role / "SKILL.md"
        path.parent.mkdir(parents=True); path.write_text("generic " + role + " skill", encoding="utf-8")
    for script, module in (("mine_selection_innovation", "selection_innovation"), ("flowcompat_agent", "flowcompat_agents")):
        path = root / "scripts" / (script + ".py"); path.parent.mkdir(exist_ok=True)
        path.write_text("import sys,pathlib,json\na=sys.argv[1:];p=pathlib.Path(a[a.index('--output')+1]);p.parent.mkdir(parents=True,exist_ok=True)\np.write_text(json.dumps({'scope':{'window':[0.07,0.43]},'evidence_items':[{'id':'E1','q':0.4}]}))\n", encoding="utf-8")
        path = root / "src/evomolsteer/continuous" / (module + ".py")
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text("# registered deterministic tool\n")
    reference = root / "reference.json.gz"
    reference.write_bytes(gzip.compress(json.dumps({"window": [.07, .43]}).encode(), mtime=0))
    program = root / "program.json"
    write_json(program, {"window": [.07, .43], "reference_sha256": digest(reference),
                         "reward_view": "endpoint_pointcloud", "derivative_path": "flowr_endpoint_vjp",
                         "affinity_head_gradient": False, "additional_per_step_affinity_calls": 0,
                         "coordinate_representation": "predicted_endpoint_world_A", "native_rms_ratio": .33})
    receipts = []
    for tool, output, args in (
        ("selection_innovation", root / "evidence.json", ["--reference", str(reference)]),
        ("flow_compatibility", root / "audit.json", ["--action", "audit-flow", "--program", str(program), "--reference", str(reference)]),
    ):
        plan = root / (tool + ".plan.json")
        write_json(plan, {"tool_id": tool, "arguments": args + ["--output", str(output)],
                          "input_files": [str(reference), str(program)], "output_files": [str(output)]})
        receipt = root / (tool + ".receipt.json")
        run_registered_tool(plan, receipt, root)
        receipts.append(receipt)
    numerical = root / "numerical.py"; numerical.write_text("# baseline with no-op delegation")
    registry = root / "registry.json"
    write_json(registry, {"programs": {"baseline": {"program_path": "program.json", "program_sha256": digest(program),
                      "reference_path": "reference.json.gz", "reference_sha256": digest(reference)}},
        "formulas": {"joint_innovation": {"reward_view": "endpoint_innovation", "allowed_updates": ["innovation.field_strength_A", "innovation.reliability_power", "innovation.region_weight_mix"],
                                          "code_files": {"numerical.py": digest(numerical)}},
                     "native_control": {"reward_view": "endpoint_pointcloud", "allowed_updates": ["flow_control.time_envelope_power", "flow_control.jacobian_gain_saturation"],
                                        "code_files": {"numerical.py": digest(numerical)}}}})
    folder = root / "roles"
    request = export_request(root / "evidence.json", registry, receipts, folder, "Analyst", root=root)
    req = read_json(request)
    response = {"schema_version": "flowcompat-agent-1.0", "agent": "Analyst", "request_sha256": digest(request),
                "instruction_sha256": req["instruction_sha256"], "evidence_sha256": req["evidence_sha256"],
                "tool_receipt_sha256": sorted(req["bindings"]["tool_receipts"].values()), "window": [.07, .43],
                "observations": [{"evidence_ids": ["E1"], "finding": "Contrast is uncertain", "status": "exploratory"}],
                "hypotheses": [], "counterevidence": [{"evidence_ids": ["E1"], "finding": "Multiplicity adjusted association absent", "status": "unsupported"}],
                "limitations": ["Rejected futures unavailable"]}
    raw = folder / "analyst.raw.json"; write_json(raw, response)
    import_response(request, raw)
    return root, folder, request, raw, registry, receipts


def designer(bound, update=None, formula="joint_innovation"):
    root, folder, _, _, registry, receipts = bound
    analyst = folder / "Analyst.flowcompat.response.json"
    request = export_request(root / "evidence.json", registry, receipts, folder, "Designer", analyst=analyst, root=root)
    req = read_json(request)
    data = {"schema_version": "flowcompat-agent-1.0", "agent": "Designer", "request_sha256": digest(request),
            "instruction_sha256": req["instruction_sha256"], "evidence_sha256": req["evidence_sha256"],
            "tool_receipt_sha256": sorted(req["bindings"]["tool_receipts"].values()), "window": [.07, .43],
            "analyst_response_sha256": digest(analyst), "decision": "trial", "base_program_id": "baseline",
            "formula_id": formula, "updates": update or {"innovation.field_strength_A": .1},
            "evidence_ids": ["E1"], "rationale": "Small exploratory coordinate contrast", "failure_modes": ["Local contrast can extrapolate incorrectly"]}
    response = folder / "designer.raw.json"; write_json(response, data)
    return request, response


def test_real_tool_receipts_and_literal_instructions_are_bound(bound):
    _, _, request, raw, _, _ = bound
    assert validate_response(request, raw)["agent"] == "Analyst"
    req = read_json(request); hydrated = payload(req)
    assert len(hydrated["tool_receipts"]) == 2
    assert all(r["returncode"] == 0 for r in hydrated["tool_receipts"])
    assert hydrated["evidence"]["scope"]["window"] == [.07, .43]


@pytest.mark.parametrize("change", ["instructions", "evidence", "tool_output", "tool_plan", "skill"])
def test_mutation_of_bound_artifacts_is_rejected(bound, change):
    root, _, request, raw, _, _ = bound
    req = read_json(request)
    if change == "instructions":
        req["system_instruction"] += " altered"; write_json(request, req)
    elif change == "evidence":
        write_json(root / "evidence.json", {"scope": {"window": [.07, .43]}, "evidence_items": [{"id": "changed"}]})
    elif change == "tool_output":
        (root / "audit.json").write_text("changed")
    elif change == "tool_plan":
        (root / "selection_innovation.plan.json").write_text("changed")
    else:
        (root / "skills/flow-compatibility/SKILL.md").write_text("changed")
    with pytest.raises(ValueError):
        validate_response(request, raw)


def test_unknown_evidence_and_scope_change_cannot_import(bound):
    _, _, request, raw, _, _ = bound
    data = read_json(raw); data["window"] = [0, .5]; write_json(raw, data)
    with pytest.raises(ValueError, match="support"):
        validate_response(request, raw)
    data["window"] = [.07, .43]; data["observations"][0]["evidence_ids"] = ["fabricated"]; write_json(raw, data)
    with pytest.raises(ValueError, match="citation"):
        validate_response(request, raw)


def test_compiler_preserves_literal_single_axis_and_reference(bound):
    root, folder, *_ = bound
    req, response = designer(bound)
    p = compile_design(req, response, folder / "compiled.json", 7, root)
    assert p["window"] == [.07, .43]
    assert p["innovation"] == {"field_strength_A": .1}
    assert "flow_control" not in p
    assert p["reward_view"] == "endpoint_innovation"
    assert p["reference_sha256"] == digest(root / "reference.json.gz")
    assert p["flowcompat_provenance"]["implementation_gate_required"] is True


def test_multiple_axes_and_inactive_formula_rejected(bound):
    req, response = designer(bound, {"innovation.field_strength_A": .2, "innovation.region_weight_mix": .3})
    with pytest.raises(ValueError, match="single-axis"):
        validate_response(req, response)
    data = read_json(response); data["updates"] = {"flow_control.time_envelope_power": 1}; write_json(response, data)
    with pytest.raises(ValueError, match="Inactive"):
        validate_response(req, response)


def test_unknown_executable_update_rejected(bound):
    req, response = designer(bound)
    data = read_json(response); data["updates"] = {"python_code": "import os"}; write_json(response, data)
    with pytest.raises(Exception):
        validate_response(req, response)


def test_compilation_requires_current_numerical_code(bound):
    root, folder, *_ = bound
    req, response = designer(bound)
    (root / "numerical.py").write_text("changed numerical algorithm")
    with pytest.raises(ValueError, match="numerical module changed"):
        compile_design(req, response, folder / "compiled.json", 1, root)


def execution_fixture(bound):
    root, folder, *_ = bound
    req, response = designer(bound)
    program = folder / "compiled.json"; p = compile_design(req, response, program, 1, root)
    execution = folder / "execution.json"
    report = {"program_sha256": digest(program), "request_sha256": digest(req),
              "code_files": p["flowcompat_provenance"]["code_files"],
              "execution_checks": {k: True for k in ("actual_flowr_model", "actual_endpoint_vjp", "no_resampling", "no_head_gradient", "no_extra_production_forward", "window_matches", "initial_state_pair_matches")},
              "no_op_control": {"exact_baseline_arithmetic": True, "max_gradient_relative_change": 0},
              "diagnostics": {"paired_window_coordinate_rms_A": .01, "contrast_gradient_relative_change": .002, "contrast_teacher_shift_rms_A": .01},
              "affinity_delta": -100}
    write_json(execution, report)
    return program, execution, report


def test_actual_response_gate_does_not_consume_affinity(bound):
    program, execution, _ = execution_fixture(bound)
    out = execution.parent / "gate.json"
    result = verify_execution_response(program, execution, out)
    assert result["implementation_passed"] and result["efficacy_assessed"] is False


@pytest.mark.parametrize("key", ["flow_control.time_envelope_power", "flow_control.jacobian_gain_saturation"])
def test_amplitude_gate_requires_dose_response_not_direction_change(bound, key):
    root, folder, *_ = bound
    req, response = designer(bound, {key: .5}, "native_control")
    program = folder / "compiled.json"; p = compile_design(req, response, program, 1, root)
    assert p["agent_request_sha256"] == digest(req)
    report = {"program_sha256": digest(program), "request_sha256": digest(req),
              "code_files": p["flowcompat_provenance"]["code_files"],
              "execution_checks": {k: True for k in ("actual_flowr_model", "actual_endpoint_vjp", "no_resampling", "no_head_gradient", "no_extra_production_forward", "window_matches", "initial_state_pair_matches")},
              "no_op_control": {"exact_baseline_arithmetic": True, "max_gradient_relative_change": 0},
              "diagnostics": {"paired_window_coordinate_rms_A": .01, "flowcompat_schedule_absolute_change": .2,
                              "flowcompat_gradient_adjustment_relative_rms": 0}}
    execution = folder / "execution.json"; write_json(execution, report)
    assert verify_execution_response(program, execution, folder / "gate.json")["implementation_passed"]


@pytest.mark.parametrize("change", ["old_gradient_only", "no_trajectory", "no_noop", "wrong_code", "resampling"])
def test_response_requires_new_mechanism_actual_trajectory_and_noop(bound, change):
    program, execution, report = execution_fixture(bound)
    if change == "old_gradient_only":
        report["diagnostics"] = {"gradient_norm": 100, "paired_window_coordinate_rms_A": .2}
    elif change == "no_trajectory":
        report["diagnostics"]["paired_window_coordinate_rms_A"] = 0
    elif change == "no_noop":
        report["no_op_control"]["max_gradient_relative_change"] = 1e-7
    elif change == "wrong_code":
        report["code_files"] = {}
    else:
        report["execution_checks"]["no_resampling"] = False
    write_json(execution, report)
    with pytest.raises(ValueError):
        verify_execution_response(program, execution, execution.parent / "gate.json")


def test_configuration_audit_explicit_scope_and_no_model_attribution(bound):
    root, _, *_ = bound
    result = audit_flow_configuration(root / "program.json", root / "reference.json.gz", root / "real_audit.json")
    assert result["passed"] and result["window"] == [.07, .43]
    assert "no live Jacobian" in result["attribution_scope"]
    p = read_json(root / "program.json"); p["affinity_head_gradient"] = True; write_json(root / "program.json", p)
    with pytest.raises(ValueError, match="configuration failed"):
        audit_flow_configuration(root / "program.json", root / "reference.json.gz", root / "failed_audit.json")


def test_tool_cannot_execute_arbitrary_action_or_unbound_reference(bound):
    root, _, *_ = bound
    path = root / "bad.plan.json"
    write_json(path, {"tool_id": "flow_compatibility", "arguments": ["--action", "api", "--output", str(root / "not-created.json")],
                      "input_files": [str(root / "program.json")], "output_files": [str(root / "not-created.json")]})
    with pytest.raises(ValueError, match="read-only"):
        run_registered_tool(path, root / "bad.receipt.json", root)
    value = read_json(path); value["arguments"] = ["--action", "audit-flow", "--reference", str(root / "reference.json.gz"), "--output", str(root / "not-created.json")]; write_json(path, value)
    with pytest.raises(ValueError, match="Unbound tool input"):
        run_registered_tool(path, root / "bad.receipt.json", root)


def test_api_uses_same_validator_and_explicit_endpoint(bound, monkeypatch):
    _, _, request, raw, _, _ = bound
    for key in ("EVOMOLSTEER_BASE_URL", "EVOMOLSTEER_MODEL", "EVOMOLSTEER_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(ValueError, match="Explicit"):
        call_api(request)
    monkeypatch.setenv("EVOMOLSTEER_BASE_URL", "https://provider.example/v1")
    monkeypatch.setenv("EVOMOLSTEER_MODEL", "explicit-model")
    monkeypatch.setenv("EVOMOLSTEER_API_KEY", "secret-not-written")
    def handle(req):
        assert str(req.url) == "https://provider.example/v1/chat/completions"
        body = json.loads(req.content)
        assert body["model"] == "explicit-model" and "request_sha256" in body["messages"][1]["content"]
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(read_json(raw))}}]})
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        assert call_api(request, client=client)["agent"] == "Analyst"
