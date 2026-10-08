import gzip
import json
from pathlib import Path

import httpx
import pandas as pd
import pytest

from evomolsteer.continuous import flowcompat_agents as old
from evomolsteer.continuous.flowcompat_supplemental import (
    DESIGNER, FROZEN_ORIGINALS, VERSION, assert_frozen_original, call_api, check_receipt,
    check_extra_receipt, _source_closure, _consumed_branch_inputs,
    compile_design, export_request, import_response, payload, run_branch_tool,
    validate_response, verify_execution_response,
)
from evomolsteer.io import digest, read_json, write_json


@pytest.fixture
def bound(tmp_path):
    root = tmp_path
    for role in ("analyst", "designer", "flow-compatibility", "flow-compatibility-supplemental"):
        skill = root / "skills" / role / "SKILL.md"
        skill.parent.mkdir(parents=True); skill.write_text("Generic role advice: " + role)
    package = root / "src/evomolsteer/continuous"; package.mkdir(parents=True)
    for module in ("selection_innovation", "flowcompat_agents"):
        (package / (module + ".py")).write_text("# frozen test tool source")
    (package / "branch_mutation.py").write_text("from . import branch_helper\n")
    (package / "branch_helper.py").write_text("# local transitive branch source")
    (package / "__init__.py").write_text("# initializer must be hash bound")
    scripts = root / "scripts"; scripts.mkdir()
    (scripts / "flowcompat_supplemental_agent.py").write_text("# versioned CLI source")
    reference = root / "reference.json.gz"
    reference.write_bytes(gzip.compress(json.dumps({"window": [.04, .46], "discovery_batches": [0, 1, 2], "frames": []}).encode(), mtime=0))
    program = root / "program.json"
    write_json(program, {"window": [.04, .46], "reference_sha256": digest(reference), "reward_view": "endpoint_pointcloud",
                         "derivative_path": "flowr_endpoint_vjp", "affinity_head_gradient": False,
                         "additional_per_step_affinity_calls": 0, "coordinate_representation": "predicted_endpoint_world_A"})
    dataset = root / "data"
    campaign = dataset / "results/observed"
    write_json(campaign / "config.json", {"campaign": "observed"})
    for batch in range(3):
        file = campaign / "single" / f"batch_{batch:03d}" / "trajectory.npz"
        file.parent.mkdir(parents=True); file.write_bytes(b"tiny test input whose identity is hashed")
        write_json(campaign / f"frame_batch_{batch:03d}.json", {"target_com": [0, 0, 0]})
    (scripts / "mine_selection_innovation.py").write_text(
        "import sys,pathlib,json,hashlib\na=sys.argv[1:];p=pathlib.Path(a[a.index('--output')+1]);p.parent.mkdir(parents=True,exist_ok=True)\n"
        "ref=pathlib.Path(a[a.index('--reference')+1]);p.write_text(json.dumps({'scope':{'window':[0.04,0.46]},'reference_sha256':hashlib.sha256(ref.read_bytes()).hexdigest(),'independent_batches':3,'evidence_items':[{'id':'selection_innovation/small','metric':'contrast','q':0.8}]}))\n")
    (scripts / "flowcompat_agent.py").write_text("import sys,pathlib,json\na=sys.argv[1:];p=pathlib.Path(a[a.index('--output')+1]);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps({'scope':'configuration_only'}))\n")
    (scripts / "mine_branch_mutation.py").write_text(
        "import sys,pathlib,json,gzip,pandas as pd\na=sys.argv[1:];d=dict(zip(a[::2],a[1::2]));out=pathlib.Path(d['--output']);out.mkdir(parents=True)\n"
        "ref=json.loads(gzip.decompress(pathlib.Path(d['--reference']).read_bytes()));ref['branch_mutation']={'scope':ref['window']}\n"
        "for name,data in {'evidence.json':{'window':ref['window'],'independent_batches':3,'evidence_items':[{'id':'branch_mutation/covariance/motion','metric':'covariance','feature':'motion','whole_window_mean':-0.01,'q':0.03}]},'manifest.json':{'schema_version':'test-branch-mining'},'feature_catalog.json':{'features':['motion']},'fitted_trends.json':{'covariance/motion':{'coefficients':[-0.01,0.02]}},'teacher_field_summary.json':{'mean_confidence':0.001}}.items():(out/name).write_text(json.dumps(data))\n"
        "pd.DataFrame([{'id':'branch_mutation/'+m+'/motion','metric':m,'feature':'motion','whole_window_mean':-0.00001 if 'covariance' in m else -0.1,'q':0.3} for m in ['raw_covariance','adjusted_covariance','raw_correlation']]).to_parquet(out/'whole_window_effects.parquet',compression=None,index=False)\n"
        "if '--augmented-reference' in d:pathlib.Path(d['--augmented-reference']).write_bytes(gzip.compress(json.dumps(ref).encode(),mtime=0))\n")
    receipts = []
    for tool, script, output, args in (
        ("selection_innovation", "mine_selection_innovation", root / "old_evidence.json", ["--reference", str(reference)]),
        ("flow_compatibility", "flowcompat_agent", root / "old_audit.json", ["--action", "audit-flow", "--program", str(program), "--reference", str(reference)]),
    ):
        plan = root / (tool + ".plan.json")
        write_json(plan, {"tool_id": tool, "arguments": args + ["--output", str(output)],
                          "input_files": [str(reference), str(program)], "output_files": [str(output)]})
        receipt = root / (tool + ".receipt.json"); old.run_registered_tool(plan, receipt, root)
        receipts.append(receipt)
    destination, augmented = root / "branch_new", root / "branch_reference.json.gz"
    plan = root / "branch.plan.json"
    write_json(plan, {"tool_id": "branch_mutation", "arguments": ["--dataset", str(dataset), "--campaign", "observed", "--reference", str(reference), "--output", str(destination), "--augmented-reference", str(augmented), "--seed", "123"],
                      "input_files": [str(reference)], "output_files": [str(destination / "evidence.json"), str(destination / "manifest.json"), str(augmented)]})
    receipt = root / "branch.receipt.json"; run_branch_tool(plan, receipt, root=root)
    receipts.append(receipt)
    numerical = root / "numerical.py"; numerical.write_text("# unchanged registered numerical module")
    registry = root / "registry.json"
    context = root / "numerical_strategy_v2.md"; context.write_text("Frozen exploratory mixture: compare positive, reversed and empty directions")
    write_json(registry, {"programs": {"branch": {"program_path": str(program), "program_sha256": digest(program), "reference_path": str(augmented), "reference_sha256": digest(augmented)}},
                         "context_files": {str(context): digest(context)},
                         "formulas": {"branch_joint": {"reward_view": "endpoint_innovation", "reference_kind": "branch_mutation", "allowed_updates": ["innovation.field_strength_A", "innovation.reliability_power"], "code_files": {str(numerical): digest(numerical)}},
                                      "branch_mixture": {"reward_view": "endpoint_branch_mixture", "reference_kind": "branch_mutation", "allowed_updates": ["branch_mixture.virtual_mass", "branch_mixture.direction_sign", "branch_mixture.region_weight_mix"], "code_files": {str(numerical): digest(numerical)}}}})
    folder = root / "new_roles"
    request = export_request(root / "old_evidence.json", destination / "evidence.json", registry, receipts, folder, "Analyst", root=root)
    req = read_json(request)
    response = {"schema_version": VERSION, "agent": "Analyst", "request_sha256": digest(request), "instruction_sha256": req["instruction_sha256"],
                "evidence_sha256": req["evidence_sha256"], "source_evidence_sha256": sorted(req["bindings"]["source_evidence"].values()),
                "tool_receipt_sha256": sorted(req["bindings"]["tool_receipts"].values()), "extra_tool_receipt_sha256": [], "window": [.04, .46],
                "observations": [{"evidence_ids": ["branch_mutation/covariance/motion"], "finding": "Conditional absolute movement is small", "status": "exploratory"}],
                "hypotheses": [], "counterevidence": [{"evidence_ids": ["selection_innovation/small"], "finding": "Root contrast weak", "status": "unsupported"}],
                "limitations": ["Conditional on past survival"]}
    raw = folder / "analyst.raw.json"; write_json(raw, response); import_response(request, raw)
    return root, folder, request, raw, registry, receipts


def designer(bound, updates=None, formula="branch_joint"):
    root, folder, _, _, registry, receipts = bound
    request = export_request(root / "old_evidence.json", root / "branch_new/evidence.json", registry, receipts, folder, "Designer",
                             analyst=folder / "Analyst.flowcompat-supplemental.response.json", root=root)
    req = read_json(request)
    raw = folder / "designer.raw.json"
    write_json(raw, {"schema_version": VERSION, "agent": "Designer", "request_sha256": digest(request),
                     "instruction_sha256": req["instruction_sha256"], "evidence_sha256": req["evidence_sha256"],
                     "source_evidence_sha256": sorted(req["bindings"]["source_evidence"].values()), "tool_receipt_sha256": sorted(req["bindings"]["tool_receipts"].values()), "extra_tool_receipt_sha256": [],
                     "window": [.04, .46], "analyst_response_sha256": req["bindings"]["analyst_response_sha256"], "decision": "trial",
                     "base_program_id": "branch", "formula_id": formula, "updates": updates or {"innovation.field_strength_A": .1},
                     "evidence_ids": ["branch_mutation/covariance/motion"], "rationale": "Small branch-local extrapolation", "failure_modes": ["Survival conditioning, weak reliability"]})
    return request, raw


def test_third_tool_really_executed_and_binds_transitive_sources_inputs_and_all_outputs(bound):
    root, _, *_ = bound
    receipt = check_receipt(root / "branch.receipt.json")
    assert receipt["returncode"] == 0 and receipt["source_unchanged"]
    assert any(path.endswith("branch_helper.py") for path in receipt["code_files"])
    assert any(path.endswith("__init__.py") for path in receipt["code_files"])
    assert len(receipt["input_files"]) == 8  # ref + config + three trajectories + three frames
    assert len(receipt["output_files"]) == 7
    assert not (root / "Analyst.flowcompat.request.json").exists()


def test_new_packet_combines_evidence_and_selected_global_fits(bound):
    _, _, request, raw, *_ = bound
    values = payload(read_json(request))
    assert len(values["tool_receipts"]) == 3
    assert values["evidence"]["window"] == [.04, .46]
    assert values["evidence"]["branch_functions"] == {"covariance/motion": {"coefficients": [-.01, .02]}}
    assert values["evidence"]["branch_companion_moments"]["added_rows"] == 3
    assert any(v["metric"] == "raw_covariance" for v in values["evidence"]["evidence_items"])
    assert "Frozen exploratory mixture" in next(iter(values["strategy_context"].values()))["text"]
    assert validate_response(request, raw)["agent"] == "Analyst"


@pytest.mark.parametrize("target", ["helper", "trajectory", "frame", "branch_evidence", "supplemental_skill", "old_tool_output"])
def test_bound_mutations_rejected_without_rewriting_old_contract(bound, target):
    root, _, request, raw, *_ = bound
    paths = {"helper": root / "src/evomolsteer/continuous/branch_helper.py", "trajectory": root / "data/results/observed/single/batch_001/trajectory.npz",
             "frame": root / "data/results/observed/frame_batch_002.json", "branch_evidence": root / "branch_new/evidence.json",
             "supplemental_skill": root / "skills/flow-compatibility-supplemental/SKILL.md", "old_tool_output": root / "old_audit.json"}
    paths[target].write_text("changed")
    with pytest.raises(ValueError): validate_response(request, raw)


def test_no_retroactive_receipt_for_existing_outputs(bound):
    root, *_ = bound
    with pytest.raises(FileExistsError, match="fresh actual"):
        run_branch_tool(root / "branch.plan.json", root / "not_created_receipt.json", root=root)
    assert not (root / "not_created_receipt.json").exists()


def test_analyst_must_interpret_new_evidence_and_response_binds_three_receipts(bound):
    _, _, request, raw, *_ = bound
    data = read_json(raw); data["observations"][0]["evidence_ids"] = ["selection_innovation/small"]; write_json(raw, data)
    with pytest.raises(ValueError, match="interpret new branch"):
        validate_response(request, raw)
    data["observations"][0]["evidence_ids"] = ["branch_mutation/covariance/motion"]
    data["tool_receipt_sha256"].pop(); write_json(raw, data)
    with pytest.raises(ValueError, match="bind all"):
        validate_response(request, raw)


def test_branch_design_compilation_uses_actual_emitted_reference(bound):
    root, folder, *_ = bound
    request, raw = designer(bound)
    program = compile_design(request, raw, folder / "compiled.json", 8, root=root)
    assert program["innovation"] == {"field_strength_A": .1}
    assert program["reference_sha256"] == digest(root / "branch_reference.json.gz")
    assert program["window"] == [.04, .46]
    assert program["flowcompat_provenance"]["workflow_version"] == VERSION
    assert program["flowcompat_provenance"]["reference_kind"] == "branch_mutation"
    assert program["agent_request_sha256"] == digest(request)
    assert "flow_control" not in program


def test_independent_branch_mixture_compiles_literal_registered_mass(bound):
    root, folder, *_ = bound
    request, raw = designer(bound, {"branch_mixture.virtual_mass": .2}, "branch_mixture")
    p = compile_design(request, raw, folder / "mixture.json", 9, root=root)
    assert p["reward_view"] == "endpoint_branch_mixture"
    assert p["branch_mixture"] == {"virtual_mass": .2}
    assert "innovation" not in p
    assert p["flowcompat_provenance"]["context_files"]


def test_branch_mixture_plan_cannot_combine_mass_and_direction_axes(bound):
    request, raw = designer(bound, {"branch_mixture.virtual_mass": .2, "branch_mixture.direction_sign": -1}, "branch_mixture")
    with pytest.raises(ValueError, match="single-axis"):
        validate_response(request, raw)


@pytest.mark.parametrize("key,value", [("branch_mixture.virtual_mass", -.01), ("branch_mixture.virtual_mass", .51),
                                      ("branch_mixture.direction_sign", -1.01), ("branch_mixture.direction_sign", 1.01),
                                      ("branch_mixture.region_weight_mix", -.01), ("branch_mixture.region_weight_mix", 1.01)])
def test_branch_numeric_budget_is_bounded_by_schema(key, value):
    import jsonschema
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({key: value}, DESIGNER["properties"]["updates"])


def test_region_axis_compiles_as_independent_mechanism_without_virtual_mass(bound):
    root, folder, *_ = bound
    request, raw = designer(bound, {"branch_mixture.region_weight_mix": .2}, "branch_mixture")
    p = compile_design(request, raw, folder / "mixture.json", 9, root=root)
    assert p["branch_mixture"] == {"region_weight_mix": .2}


def test_strategy_source_context_tampering_invalidates_response(bound):
    root, _, request, raw, *_ = bound
    (root / "numerical_strategy_v2.md").write_text("changed strategy")
    with pytest.raises(ValueError, match="artifact changed"):
        validate_response(request, raw)


def branch_gate_fixture(tmp_path, mass=.2, sign=1, region_mix=0.):
    program = tmp_path / "branch.json"
    sha = "1" * 64
    write_json(program, {"reward_view": "endpoint_branch_mixture", "branch_mixture": {"virtual_mass": mass, "direction_sign": sign, "region_weight_mix": region_mix},
        "flowcompat_provenance": {"workflow_version": VERSION, "request_sha256": sha, "code_files": {"branch.py": sha},
                                  "parameter_updates": {"branch_mixture.virtual_mass": mass}}})
    field_active = mass > 0 and sign != 0; active = field_active or region_mix > 0
    report = {"program_sha256": digest(program), "request_sha256": sha, "code_files": {"branch.py": sha},
        "execution_checks": {key: True for key in ("actual_flowr_model", "actual_endpoint_vjp", "no_resampling", "no_head_gradient", "no_extra_production_forward", "window_matches", "initial_state_pair_matches")},
        "no_op_control": {"exact_baseline_arithmetic": True, "max_gradient_relative_change": 0},
        "diagnostics": {"branch_virtual_mass_mean": .05 if field_active else 0., "branch_virtual_teacher_shift_rms_A": .01 if field_active else 0.,
                        "branch_gradient_relative_change": .002 if active else 0., "paired_window_coordinate_rms_A": .001 if active else 0.,
                        "branch_region_weight_rms_change": .00001 if region_mix > 0 else 0.,
                        "flowcompat_gradient_adjustment_relative_rms": 0},
        "affinity_delta": -100}
    execution = tmp_path / "execution.json"; write_json(execution, report)
    return program, execution, report


@pytest.mark.parametrize("mass,sign", [(.2, 1), (.2, -1), (.2, 0), (0., 1)])
def test_positive_reverse_and_null_implementation_gates_are_distinct_from_efficacy(tmp_path, mass, sign):
    program, execution, _ = branch_gate_fixture(tmp_path, mass, sign)
    result = verify_execution_response(program, execution, tmp_path / "gate.json")
    assert result["implementation_passed"] and result["efficacy_assessed"] is False


@pytest.mark.parametrize("change", ["old_gradient_only", "no_trajectory", "excess_mass", "not_exact_noop"])
def test_branch_gate_rejects_missing_new_response_or_budget_violation(tmp_path, change):
    program, execution, report = branch_gate_fixture(tmp_path)
    if change == "old_gradient_only": report["diagnostics"] = {"gradient_norm": 100.}
    elif change == "no_trajectory": report["diagnostics"]["paired_window_coordinate_rms_A"] = 0
    elif change == "excess_mass": report["diagnostics"]["branch_virtual_mass_mean"] = .3
    else: report["no_op_control"]["max_gradient_relative_change"] = 1e-7
    write_json(execution, report)
    with pytest.raises(ValueError): verify_execution_response(program, execution, tmp_path / "gate.json")


def test_null_direction_requires_exact_actual_baseline_response(tmp_path):
    program, execution, report = branch_gate_fixture(tmp_path, .2, 0)
    report["diagnostics"]["paired_window_coordinate_rms_A"] = 1e-7; write_json(execution, report)
    with pytest.raises(ValueError, match="changed baseline"):
        verify_execution_response(program, execution, tmp_path / "gate.json")


def test_region_only_reward_response_does_not_require_virtual_mass_or_preconditioner_change(tmp_path):
    program, execution, report = branch_gate_fixture(tmp_path, mass=0, region_mix=.25)
    assert report["diagnostics"]["flowcompat_gradient_adjustment_relative_rms"] == 0
    assert report["diagnostics"]["branch_virtual_mass_mean"] == 0
    assert verify_execution_response(program, execution, tmp_path / "gate.json")["implementation_passed"]
    report["diagnostics"]["branch_region_weight_rms_change"] = 0; write_json(execution, report)
    with pytest.raises(ValueError, match="regional branch weights"):
        verify_execution_response(program, execution, tmp_path / "bad_gate.json")


def test_single_axis_and_unbound_numerical_program_rejected(bound):
    root, folder, *_ = bound
    request, raw = designer(bound, {"innovation.field_strength_A": .1, "innovation.reliability_power": .5})
    with pytest.raises(ValueError, match="single-axis"):
        compile_design(request, raw, folder / "bad.json", 8, root=root)
    data = read_json(raw); data["updates"] = {"innovation.field_strength_A": .1}; write_json(raw, data)
    (root / "numerical.py").write_text("changed")
    with pytest.raises(ValueError, match="numerical module changed"):
        compile_design(request, raw, folder / "bad.json", 8, root=root)


def test_api_and_subagent_import_share_exact_schema_and_validator(bound, monkeypatch):
    _, _, request, raw, *_ = bound
    monkeypatch.setenv("EVOMOLSTEER_BASE_URL", "https://provider.example/v1")
    monkeypatch.setenv("EVOMOLSTEER_MODEL", "explicit")
    monkeypatch.setenv("EVOMOLSTEER_API_KEY", "not-saved")
    def serve(req):
        body = json.loads(req.content)
        assert VERSION in body["messages"][0]["content"]
        assert "branch_mutation/covariance/motion" in body["messages"][1]["content"]
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(read_json(raw))}}]})
    with httpx.Client(transport=httpx.MockTransport(serve)) as client:
        assert call_api(request, client=client) == validate_response(request, raw)


def test_frozen_originals_unchanged_and_legacy_contract_not_extended():
    assert assert_frozen_original() == FROZEN_ORIGINALS
    assert set(old.TOOL_SPECS) == {"selection_innovation", "flow_compatibility"}
    assert old.ANALYST["properties"]["schema_version"]["const"] == "flowcompat-agent-1.0"
    assert "source_evidence_sha256" not in old.ANALYST["properties"]
    assert "branch_mixture.virtual_mass" not in old.DESIGNER["properties"]["updates"]["properties"]


def extra_fixture(bound):
    """Synthetic verifier fixture; the production live receipt is read separately."""
    root, *_ = bound
    script = root / "scripts/mine_multi_depth_mutation.py"; script.write_text("# synthetic verifier script identity\n")
    module = root / "src/evomolsteer/continuous/multi_depth_mutation.py"; module.write_text("# synthetic verifier module identity\n")
    folder = root / "extra"; folder.mkdir()
    rows = [{"id": f"multi_depth/3/{metric}/region_00_internal_displacement_{axis}", "depth": 3, "metric": metric,
             "feature": f"region_00_internal_displacement_{axis}", "whole_window_mean": 1e-6 if "covariance" in metric else .1, "q": .03}
            for metric in ("raw_covariance", "adjusted_covariance", "raw_correlation", "adjusted_correlation") for axis in "xyz"]
    write_json(folder / "evidence.json", {"schema_version": "multi-depth-mutation-evidence-1.0", "window": [.04, .46], "independent_batches": 3, "evidence_items": [rows[-2]]})
    pd.DataFrame(rows).to_parquet(folder / "whole_window_effects.parquet", compression=None, index=False)
    write_json(folder / "feature_catalog.json", {"features": sorted({v["feature"] for v in rows})})
    write_json(folder / "fitted_trends.json", {"schema_version": "multi-depth-global-functions-1.0", "statistics_unit": "Independent batches",
        "depths": {"3": {"times": [.07, .45], "first_missing_times": [.04, .05, .06],
        "functions": {v["metric"] + "/" + v["feature"]: {"coefficients": [.01]} for v in rows}}}})
    write_json(folder / "regional_direction_coverage.json", {"region_landmarks_A": [[0, 0, 0]], "units": "coordinate covariance",
        "rows": [{"depth": 3, "region": 0, "observed_pair_count": 3, "observed_pair_mean_cosine": .1}]})
    write_json(folder / "manifest.json", {"schema_version": "multi-depth-mutation-mining-1.0", "depths": [3], "nuisance": "chemistry_pose",
        "reference_sha256": digest(root / "reference.json.gz"),
        "files": {p.name: {"sha256": digest(p)} for p in folder.iterdir() if p.is_file()}})
    arguments = {"--dataset": str(root / "data"), "--campaign": "observed", "--reference": str(root / "reference.json.gz"), "--output": str(folder), "--depths": "3", "--nuisance": "chemistry_pose"}
    command = ["python", str(script), *[item for pair in arguments.items() for item in pair]]
    receipt = folder / "execution_receipt.json"
    write_json(receipt, {"schema_version": "multi-depth-live-execution-receipt-1.0", "tool_id": "multi_depth_mutation",
        "command": command, "returncode": 0, "error": None, "inputs_and_code_unchanged": True, "complete_outputs": True,
        "input_files": {str(p): digest(p) for p in _consumed_branch_inputs(root, arguments)},
        "code_files": _source_closure(root, [str(script), str(module)], eager_only=True),
        "output_files": {str(p): digest(p) for p in folder.iterdir() if p.is_file()}})
    return folder / "evidence.json", receipt


def test_optional_extra_packet_binds_separate_receipt_and_all_source_evidence(bound):
    root, _, _, raw, registry, receipts = bound
    evidence, receipt = extra_fixture(bound)
    assert check_extra_receipt(receipt)["tool_id"] == "multi_depth_mutation"
    request = export_request(root / "old_evidence.json", root / "branch_new/evidence.json", registry, receipts, root / "extra_roles", "Analyst",
                             extra_evidence=evidence, extra_receipt=receipt, root=root)
    req = read_json(request); values = payload(req)
    assert len(values["tool_receipts"]) == 3 and len(values["extra_tool_receipts"]) == 1
    assert len(req["bindings"]["source_evidence"]) == 3
    assert values["evidence"]["multi_depth_companion_moments"]["added_rows"] == 11
    assert len(values["evidence"]["multi_depth_feature_catalog"]["features"]) == 3
    assert len(values["evidence"]["multi_depth_functions"]["depths"]["3"]["functions"]) == 12
    assert "dependent" in values["evidence"]["multi_depth_regional_directions"]["interpretation"]
    response = read_json(raw)
    response.update(request_sha256=digest(request), instruction_sha256=req["instruction_sha256"], evidence_sha256=req["evidence_sha256"],
                    source_evidence_sha256=sorted(req["bindings"]["source_evidence"].values()),
                    extra_tool_receipt_sha256=sorted(req["bindings"]["extra_receipts"].values()))
    response["observations"].append({"evidence_ids": ["multi_depth/3/adjusted_correlation/region_00_internal_displacement_y"],
                                     "finding": "Window association is not stepwise vector agreement", "status": "exploratory"})
    path = root / "extra_roles/Analyst.raw.json"; write_json(path, response)
    assert validate_response(request, path)["agent"] == "Analyst"
    response["extra_tool_receipt_sha256"] = []; write_json(path, response)
    with pytest.raises(ValueError, match="bind all"):
        validate_response(request, path)


@pytest.mark.parametrize("change", ["failure", "wrong_tool", "missing_input", "missing_code", "wrong_script", "wrong_depth"])
def test_extra_live_receipt_rejects_incomplete_or_wrong_execution(bound, change):
    _, receipt = extra_fixture(bound); value = read_json(receipt)
    if change == "failure": value["inputs_and_code_unchanged"] = False
    elif change == "wrong_tool": value["tool_id"] = "unknown_tool"
    elif change == "missing_input": value["input_files"].pop(next(iter(value["input_files"])))
    elif change == "missing_code": value["code_files"].pop(next(iter(value["code_files"])))
    elif change == "wrong_depth": value["command"][value["command"].index("--depths") + 1] = "5"
    else: value["command"][1] = value["command"][1].replace("mine_multi_depth_mutation.py", "unknown.py")
    write_json(receipt, value)
    with pytest.raises(ValueError): check_extra_receipt(receipt)


def test_extra_window_mismatch_rejected_without_modifying_legacy_receipts(bound):
    root, _, _, _, registry, receipts = bound
    evidence, receipt = extra_fixture(bound); data = read_json(evidence); data["window"] = [0, .5]; write_json(evidence, data)
    # The synthetic fixture remains internally bound, but its support differs.
    value = read_json(receipt); value["output_files"][str(evidence)] = digest(evidence)
    manifest = evidence.parent / "manifest.json"; m = read_json(manifest); m["files"]["evidence.json"]["sha256"] = digest(evidence); write_json(manifest, m)
    value["output_files"][str(manifest)] = digest(manifest); write_json(receipt, value)
    with pytest.raises(ValueError, match="window"):
        export_request(root / "old_evidence.json", root / "branch_new/evidence.json", registry, receipts, root / "bad_extra_roles", "Analyst",
                       extra_evidence=evidence, extra_receipt=receipt, root=root)
