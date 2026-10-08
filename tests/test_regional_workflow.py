import copy
import gzip
import json
from pathlib import Path

import jsonschema
import pytest

from evomolsteer.continuous import regional_workflow as workflow
from evomolsteer.continuous.regional_reference import synthesize_reference
from evomolsteer.io import digest, read_json, write_json


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    program = tmp_path / "incumbent.json"
    write_json(program, {"window": [.1, .6], "reward_view": "endpoint_pointcloud", "seed": 42,
                         "derivative_path": "flowr_endpoint_vjp", "affinity_head_gradient": False,
                         "additional_per_step_affinity_calls": 0, "native_rms_ratio": .6})
    original = tmp_path / "original.gz"; original.write_bytes(gzip.compress(b'{}', mtime=0))
    reference = tmp_path / "regional.gz"; reference.write_bytes(gzip.compress(b'{"new":true}', mtime=0))
    numerical = tmp_path / "numeric.py"; numerical.write_text("frozen numerical source\n")
    receipt = tmp_path / "receipt.json"; write_json(receipt, {"tool_id": "regional_reference"})
    source_evidence = tmp_path / "input.evidence.json"; write_json(source_evidence, {"observed": True})
    old_receipt = tmp_path / "old.receipt.json"; write_json(old_receipt, {"tool_id": "branch_mutation"})
    extra_receipt = tmp_path / "extra.receipt.json"; write_json(extra_receipt, {"tool_id": "multi_depth_mutation"})
    functions = {axis: {"degree": 0, "coefficients": [value], "observed_time_start": .2, "observed_time_end": .59}
                 for axis, value in zip("xyz", [1., .2, -.3])}
    rules = {"window": [.1, .6], "source_reference_sha256": digest(original), "region_width_A": 1.,
             "absolute_covariance_floor": 1e-12, "selected_regions": [
                 {"region": 0, "depth": 2, "whole_window_xyz": [1., .2, -.3], "component_q": [.01, .2, .3],
                  "component_CI_low": [.8, -.1, -.6], "component_CI_high": [1.2, .5, .0],
                  "absolute_covariance_norm": 1.1, "independent_batches": 14, "loo_positive_fraction": .85,
                  "landmark_A": [0., 0., 0.], "functions_xyz": functions}]}
    rules_path = tmp_path / "rules.json"; write_json(rules_path, rules)
    manifest_path = tmp_path / "manifest.json"
    write_json(manifest_path, {"coverage": {"teachers": 14, "original_eligible_teachers": 3, "new_active_teachers": 3}})
    regional = {"rules": rules, "reference_path": str(reference), "reference_sha256": digest(reference),
                "receipt_sha256": digest(receipt), "rules_path": str(rules_path), "manifest_path": str(manifest_path),
                "manifest": read_json(manifest_path), "invariants": {"original_teacher_truth_unchanged": True},
                "receipt": {"tool_id": "regional_reference", "code_files": {"data_tool.py": "a"*64}}}
    evidence = {"window": [.1, .6], "evidence_items": [{"id": identifier} for identifier in
                ["branch_mutation/absolute", "multi_depth/2/adjusted_covariance/region_00_internal_displacement_x", "selection_innovation/contrast"]]}
    registry = {"programs": {"parent": {"program_path": str(program), "program_sha256": digest(program),
                                          "reference_path": str(original), "reference_sha256": digest(original)}},
                "formulas": {"branch": {"reward_view": "endpoint_branch_mixture", "reference_kind": "branch_mutation",
                                         "allowed_updates": ["branch_mixture.virtual_mass", "branch_mixture.direction_sign"],
                                         "formula": "frozen joint scalar", "code_files": {str(numerical): digest(numerical)}}},
                "context_files": {}}
    base = tmp_path / "base.request.json"
    write_json(base, {"schema_version": workflow.supplemental.VERSION, "role": "Analyst", "window": [.1, .6],
        "bindings": {"source_evidence": {str(source_evidence): digest(source_evidence)},
                     "tool_receipts": {str(old_receipt): digest(old_receipt)},
                     "extra_receipts": {str(extra_receipt): digest(extra_receipt)}}})
    values = {"evidence": evidence, "registry": registry, "tool_receipts": [], "extra_tool_receipts": []}
    monkeypatch.setattr(workflow.supplemental, "payload", lambda request: copy.deepcopy(values))
    monkeypatch.setattr(workflow, "check_regional_receipt", lambda *args: copy.deepcopy(regional))
    return {"tmp": tmp_path, "base": base, "regional": regional, "receipt": receipt, "reference": reference,
            "program": program, "numeric": numerical, "values": values, "rules": rules}


def export(prepared, role="Analyst", **kwargs):
    return workflow.export_request(prepared["base"], prepared["receipt"], prepared["reference"],
                                   prepared["tmp"] / (role.lower()+str(len(list(prepared['tmp'].glob(role.lower()+'*'))))), role, **kwargs)


def response_for(request_path, role="Analyst"):
    request = read_json(request_path)
    values = workflow.payload(request)
    ids = ["regional_reference/region_00/depth_2", "branch_mutation/absolute", "multi_depth/2/adjusted_covariance/region_00_internal_displacement_x"]
    data = {"schema_version": workflow.VERSION, "agent": role, "request_sha256": digest(request_path),
            "window": request["window"], **values["request_bindings"]}
    if role == "Analyst":
        data.update(observations=[{"evidence_ids": ids, "finding": "Independent batches support a coordinate association", "status": "supported"}],
                    hypotheses=[{"evidence_ids": ids, "finding": "A joint coordinate intervention requires positive/reverse/null tests", "status": "exploratory"}],
                    counterevidence=[{"evidence_ids": ids, "finding": "Covariance is observational and mixed depth can bias amplitude", "status": "supported"}],
                    limitations=["Whole-window association is not stepwise affinity gradient"])
    else:
        data.update(analyst_response_sha256=values["analyst_response_sha256"], decision="trial", base_program_id="parent_regional",
                    formula_id="branch_regional", updates={"branch_mixture.virtual_mass": .1}, evidence_ids=ids,
                    rationale="Bounded joint intervention over the exact reference; test opposite and null directions", failure_modes=["Observation does not establish a causal gradient"])
    return data


def imported_roles(prepared):
    analyst_request = export(prepared)
    raw = prepared["tmp"] / "analyst.raw.json"; write_json(raw, response_for(analyst_request))
    workflow.import_response(analyst_request, raw)
    analyst = analyst_request.parent / "Analyst.flowcompat-regional.response.json"
    designer_request = export(prepared, "Designer", analyst=analyst, analyst_request=analyst_request)
    raw_designer = prepared["tmp"] / "designer.raw.json"; write_json(raw_designer, response_for(designer_request, "Designer"))
    workflow.import_response(designer_request, raw_designer)
    return analyst_request, designer_request, designer_request.parent / "Designer.flowcompat-regional.response.json"


def test_actual_live_receipt_keeps_origin_and_teacher_truth():
    folder = workflow.PROJECT / "docs/experiments/flowcompat30_20261009/mining/regional_reference_v1"
    reference = workflow.PROJECT / "configs/experiments/flowcompat30_v1/regional_reference.json.gz"
    if not (folder / "execution_receipt.json").is_file():
        pytest.skip("Actual research fixture not present in clean checkout")
    result = workflow.check_regional_receipt(folder / "execution_receipt.json", reference)
    assert result["receipt"]["tool_id"] == "regional_reference"
    assert result["invariants"]["original_teacher_truth_unchanged"]
    assert result["invariants"]["registered_joint_recipe_matches"]
    assert result["manifest"]["coverage"]["new_active_teachers"] <= result["manifest"]["coverage"]["original_eligible_teachers"]


def test_actual_common_depth_receipt_and_full_export_profile(tmp_path):
    folder = workflow.PROJECT / "docs/experiments/flowcompat30_20261009/mining/regional_reference_v2"
    reference = workflow.PROJECT / "configs/experiments/flowcompat30_v1/regional_common_depth_reference.json.gz"
    base = workflow.PROJECT / "docs/experiments/flowcompat30_20261009/agents/branch_phase_v2/Analyst.flowcompat-supplemental.request.json"
    audit = workflow.PROJECT / "docs/experiments/flowcompat30_20261009/feature_to_reward_map.zh-CN.md"
    receipt = folder / "execution_receipt.json"
    if not all(path.is_file() for path in (reference, base, audit, receipt)):
        pytest.skip("Actual research fixtures not present in clean checkout")
    checked = workflow.check_regional_receipt(receipt, reference)
    assert checked["receipt"]["tool_id"] == "regional_common_depth"
    assert len({r["depth"] for r in checked["rules"]["selected_regions"]}) == 1
    assert checked["invariants"]["registered_joint_recipe_matches"]
    request_path = workflow.export_request(base, receipt, reference, tmp_path / "actual_profile", "Analyst", context_files=[audit])
    request = read_json(request_path); values = workflow.payload(request)
    assert values["regional_tool_receipt"]["tool_id"] == "regional_common_depth"
    assert values["registry"]["reference_origin"]["tool_id"] == "regional_common_depth"
    assert values["request_bindings"]["regional_reference_sha256"] == digest(reference)
    assert values["strategy_context"][str(audit)]["text"] == audit.read_text(encoding="utf-8")
    report = str(folder / "report.zh-CN.md")
    assert values["strategy_context"][report]["provenance"].startswith("Actual regional tool output")
    assert len(request["bindings"]["skill_files"]) == 4


def test_export_binds_literal_skills_context_registry_and_dynamic_scope(prepared):
    audit = prepared["tmp"] / "audit.md"; audit.write_text("Mixed depth is descriptive; not model causality.")
    request_path = export(prepared, context_files=[audit])
    request = read_json(request_path); values = workflow.payload(request)
    assert request["window"] == [.1, .6]
    assert "flow-compatibility-regional" in request["system_instruction"]
    assert len(request["bindings"]["skill_files"]) == 4
    assert values["strategy_context"][str(audit)]["text"] == audit.read_text()
    assert digest(audit) in values["request_bindings"]["source_evidence_sha256"]
    assert values["registry"]["programs"]["parent_regional"]["reference_path"] == str(prepared["reference"])
    assert values["registry"]["formulas"]["branch_regional"]["reference_kind"] == "regional_reference"
    assert values["regional_tool_receipt"]["code_files"] == {"data_tool.py": "a"*64}
    audit.write_text("changed")
    with pytest.raises(ValueError, match="artifact changed"):
        workflow.payload(request)


def test_roles_must_cite_actual_new_and_old_evidence_and_bind_receipt(prepared):
    request = export(prepared)
    raw = prepared["tmp"] / "raw.json"; data = response_for(request)
    write_json(raw, data)
    assert workflow.validate_response(request, raw)["agent"] == "Analyst"
    data["regional_tool_receipt_sha256"] = "b"*64; write_json(raw, data)
    with pytest.raises(ValueError, match="bind actual"):
        workflow.validate_response(request, raw)
    data = response_for(request)
    for key in ("observations", "hypotheses", "counterevidence"):
        data[key][0]["evidence_ids"] = ["branch_mutation/absolute"]
    write_json(raw, data)
    with pytest.raises(ValueError, match="actually interpret"):
        workflow.validate_response(request, raw)


def test_schema_rejects_second_axis_executable_text_and_unbounded_mass(prepared):
    _, request, response = imported_roles(prepared)
    original = read_json(response); raw = prepared["tmp"] / "invalid.json"
    for updates in ({"branch_mixture.virtual_mass": .1, "branch_mixture.direction_sign": -1},
                    {"branch_mixture.virtual_mass": .51}, {"branch_mixture.virtual_mass": "exec(1)"}):
        data = copy.deepcopy(original); data["updates"] = updates; write_json(raw, data)
        with pytest.raises(jsonschema.ValidationError):
            workflow.validate_response(request, raw)


def test_compile_requires_imported_response_and_preserves_original_controller(prepared):
    _, request, response = imported_roles(prepared)
    raw = prepared["tmp"] / "unimported.json"; write_json(raw, read_json(response))
    with pytest.raises(ValueError, match="canonical imported"):
        workflow.compile_design(request, raw, prepared["tmp"] / "invalid.program.json", 22)
    compiled_path = prepared["tmp"] / "compiled.json"
    compiled = workflow.compile_design(request, response, compiled_path, 22)
    assert compiled["window"] == [.1, .6] and compiled["seed"] == 42
    assert compiled["native_rms_ratio"] == .6
    assert compiled["branch_mixture"] == {"virtual_mass": .1, "direction_sign": 1., "region_weight_mix": 0.}
    assert compiled["generation_interface"] == "flowcompat_v2"
    assert compiled["reference_sha256"] == digest(prepared["reference"])
    assert compiled["flowcompat_provenance"]["reference_kind"] == "regional_reference"
    assert compiled["flowcompat_provenance"]["workflow_version"] == workflow.VERSION
    assert compiled["flowcompat_provenance"]["regional_recipe"]["receipt_sha256"] == digest(prepared["receipt"])
    prepared["numeric"].write_text("changed after import")
    with pytest.raises(ValueError, match="formula source changed"):
        workflow.compile_design(request, response, prepared["tmp"] / "changed.json", 24)


def test_validation_cannot_be_replaced_or_bypassed(prepared):
    _, request, response = imported_roles(prepared)
    validation = response.parent / "Designer.flowcompat-regional.validation.json"
    value = read_json(validation); value["response_sha256"] = "0"*64; write_json(validation, value)
    with pytest.raises(ValueError, match="validation/provenance"):
        workflow.compile_design(request, response, prepared["tmp"] / "no.json", 22)


def test_identical_reimport_preserves_original_raw_response_provenance(prepared):
    request = export(prepared)
    raw = prepared["tmp"] / "compact.raw.json"
    raw.write_text(json.dumps(response_for(request), separators=(",", ":")), encoding="utf-8")
    workflow.import_response(request, raw)
    validation = request.parent / "Analyst.flowcompat-regional.validation.json"
    canonical = request.parent / "Analyst.flowcompat-regional.response.json"
    initial = digest(validation)
    assert read_json(validation)["source_response_sha256"] == digest(raw)
    assert digest(raw) != digest(canonical)
    workflow.import_response(request, canonical)
    assert digest(validation) == initial


def test_api_and_subagent_share_same_schema_import(prepared, monkeypatch):
    request = export(prepared)
    data = response_for(request)
    class Result:
        def raise_for_status(self): pass
        def json(self): return {"choices": [{"message": {"content": json.dumps(data)}}]}
    class Client:
        def post(self, url, headers, json):
            assert url == "https://explicit.invalid/v1/chat/completions"
            assert headers["Authorization"] == "Bearer explicit-test-key"
            assert json["model"] == "explicit-model"
            user = __import__('json').loads(json["messages"][1]["content"])
            assert user["request_sha256"] == digest(request)
            assert user["request_bindings"]["regional_reference_sha256"] == digest(prepared["reference"])
            return Result()
    for name, value in {"EVOMOLSTEER_BASE_URL": "https://explicit.invalid/v1", "EVOMOLSTEER_MODEL": "explicit-model", "EVOMOLSTEER_API_KEY": "explicit-test-key"}.items():
        monkeypatch.setenv(name, value)
    assert workflow.call_api(request, client=Client()) == data
    assert (request.parent / "Analyst.flowcompat-regional.validation.json").is_file()
    monkeypatch.delenv("EVOMOLSTEER_API_KEY")
    with pytest.raises(ValueError, match="Explicit API"):
        workflow.call_api(request, client=Client())


def test_teacher_invariant_rejects_score_or_provenance_mutation():
    original = {"window": [.1, .6], "branch_mutation": {"scope": [.1, .6]}, "priors": [1.], "frames": [
        {"time": .3, "teacher_endpoint_A": [[[0., 0., 0.], [2., 0., 0.]]], "teacher_scores": [8.],
         "teacher_contrast_direction_unit": [[[0., 1., 0.], [0., -1., 0.]]], "teacher_contrast_confidence": [.01],
         "teacher_contrast_atom_weight": [[1., 1.]], "teacher_contrast_provenance": [{"lower_immediate_parent_slots": [2], "raw_direction_RMS_A": .1}]}]}
    functions = {axis: {"degree": 0, "coefficients": [value], "observed_time_start": .2, "observed_time_end": .5} for axis, value in zip("xyz", (1., 0., 0.))}
    rules = {"window": [.1, .6], "region_width_A": 1., "absolute_covariance_floor": 1e-12,
             "selected_regions": [{"region": 0, "landmark_A": [0., 0., 0.], "functions_xyz": functions}]}
    transformed, coverage = synthesize_reference(original, rules); rules["teacher_coverage"] = coverage
    assert workflow._teacher_invariants(original, transformed, rules)["unit_atom_RMS"]
    changed = copy.deepcopy(transformed); changed["frames"][0]["teacher_scores"][0] = 99.
    with pytest.raises(ValueError, match="Immutable teacher"):
        workflow._teacher_invariants(original, changed, rules)
    changed = copy.deepcopy(transformed); changed["frames"][0]["teacher_contrast_provenance"][0]["raw_direction_RMS_A"] = 2.
    with pytest.raises(ValueError, match="Immutable teacher"):
        workflow._teacher_invariants(original, changed, rules)


def test_implementation_gate_accepts_new_reward_with_zero_preconditioner(prepared):
    _, request, response = imported_roles(prepared)
    program = prepared["tmp"] / "trial.json"; p = workflow.compile_design(request, response, program, 22)
    report = {"program_sha256": digest(program), "request_sha256": p["flowcompat_provenance"]["request_sha256"],
              "code_files": p["flowcompat_provenance"]["code_files"],
              "execution_checks": {key: True for key in ("actual_flowr_model", "actual_endpoint_vjp", "no_resampling", "no_head_gradient", "no_extra_production_forward", "window_matches", "initial_state_pair_matches")},
              "no_op_control": {"exact_baseline_arithmetic": True, "max_gradient_relative_change": 0},
              "diagnostics": {"branch_virtual_mass_mean": .02, "branch_virtual_teacher_shift_rms_A": .01,
                              "branch_gradient_relative_change": .02, "paired_window_coordinate_rms_A": .03,
                              "flowcompat_gradient_adjustment_relative_rms": 0}}
    execution = prepared["tmp"] / "execution.json"; write_json(execution, report)
    result = workflow.verify_execution_response(program, execution, prepared["tmp"] / "gate.json")
    assert result["implementation_passed"] and not result["efficacy_assessed"]
    report["diagnostics"]["branch_gradient_relative_change"] = 0; write_json(execution, report)
    with pytest.raises(ValueError, match="failed to respond"):
        workflow.verify_execution_response(program, execution, prepared["tmp"] / "rejected.json")


def test_null_regional_mechanism_requires_exact_baseline(prepared):
    _, request, response = imported_roles(prepared)
    data = read_json(response); data["updates"]["branch_mixture.virtual_mass"] = 0
    # Re-import a fresh experiment response rather than overwriting validation.
    request_folder = prepared["tmp"] / "null_designer"
    analyst_path = Path(read_json(request)["bindings"]["analyst_path"])
    null_request = workflow.export_request(prepared["base"], prepared["receipt"], prepared["reference"], request_folder, "Designer", analyst=analyst_path,
                                           analyst_request=read_json(request)["bindings"]["analyst_request_path"])
    data = response_for(null_request, "Designer"); data["updates"]["branch_mixture.virtual_mass"] = 0
    raw = prepared["tmp"] / "null.raw.json"; write_json(raw, data); workflow.import_response(null_request, raw)
    imported = request_folder / "Designer.flowcompat-regional.response.json"
    program = prepared["tmp"] / "null.json"; p = workflow.compile_design(null_request, imported, program, 24)
    report = {"program_sha256": digest(program), "request_sha256": p["flowcompat_provenance"]["request_sha256"], "code_files": p["flowcompat_provenance"]["code_files"],
              "execution_checks": {key: True for key in ("actual_flowr_model", "actual_endpoint_vjp", "no_resampling", "no_head_gradient", "no_extra_production_forward", "window_matches", "initial_state_pair_matches")},
              "no_op_control": {"exact_baseline_arithmetic": True, "max_gradient_relative_change": 0},
              "diagnostics": {key: 0 for key in ("branch_virtual_mass_mean", "branch_virtual_teacher_shift_rms_A", "branch_gradient_relative_change", "paired_window_coordinate_rms_A")}}
    execution = prepared["tmp"] / "null.execution.json"; write_json(execution, report)
    assert workflow.verify_execution_response(program, execution, prepared["tmp"] / "null.gate.json")["implementation_passed"]
    report["diagnostics"]["paired_window_coordinate_rms_A"] = .001; write_json(execution, report)
    with pytest.raises(ValueError, match="exact baseline"):
        workflow.verify_execution_response(program, execution, prepared["tmp"] / "badnull.json")
