"""Hash-bound role transport and conservative compilation for flow-compatible trials.

The API backend and local subagent simulation share the same import validator.
Tool receipts attest execution, not causal attribution or performance.  The
pre-efficacy gate deliberately does not inspect affinity labels.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import httpx
import jsonschema

from ..io import digest, read_json, write_json

PROJECT = Path(__file__).resolve().parents[3]
VERSION = "flowcompat-agent-1.0"
SHA = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
IDS = {"type": "array", "minItems": 1, "uniqueItems": True,
       "items": {"type": "string", "minLength": 1}}
WINDOW = {"type": "array", "minItems": 2, "maxItems": 2,
          "items": {"type": "number"}}
STATEMENT = {
    "type": "object", "additionalProperties": False,
    "required": ["evidence_ids", "finding", "status"],
    "properties": {"evidence_ids": IDS, "finding": {"type": "string", "minLength": 1},
                   "status": {"enum": ["supported", "exploratory", "unsupported"]}},
}
COMMON = {"schema_version": {"const": VERSION}, "request_sha256": SHA,
          "instruction_sha256": SHA, "evidence_sha256": SHA,
          "tool_receipt_sha256": IDS, "window": WINDOW}
ANALYST = {
    "type": "object", "additionalProperties": False,
    "required": list(COMMON) + ["agent", "observations", "hypotheses", "counterevidence", "limitations"],
    "properties": {**COMMON, "agent": {"const": "Analyst"},
                   "observations": {"type": "array", "minItems": 1, "items": STATEMENT},
                   "hypotheses": {"type": "array", "items": STATEMENT},
                   "counterevidence": {"type": "array", "minItems": 1, "items": STATEMENT},
                   "limitations": {"type": "array", "minItems": 1, "items": {"type": "string"}}},
}
# Only these numeric keys can cross from model text to an executable program.
UPDATE_RULES = {
    "innovation.field_strength_A": (0., 1.),
    "innovation.reliability_power": (0., 2.),
    "innovation.region_weight_mix": (0., 1.),
    "flow_control.time_envelope_power": (0., 3.),
    "flow_control.parallel_component_scale": (0., 2.),
    "flow_control.gradient_norm_saturation": (0., 10.),
    "flow_control.jacobian_gain_saturation": (0., 10.),
    "native_rms_ratio": (.05, 1.),
}
DESIGNER = {
    "type": "object", "additionalProperties": False,
    "required": list(COMMON) + ["agent", "analyst_response_sha256", "decision", "base_program_id",
                                "formula_id", "updates", "evidence_ids", "rationale", "failure_modes"],
    "properties": {**COMMON, "agent": {"const": "Designer"}, "analyst_response_sha256": SHA,
                   "decision": {"enum": ["retain", "trial", "defer"]},
                   "base_program_id": {"type": ["string", "null"]},
                   "formula_id": {"type": ["string", "null"]},
                   "updates": {"type": "object", "additionalProperties": False,
                               "properties": {key: {"type": "number", "minimum": lo, "maximum": hi}
                                              for key, (lo, hi) in UPDATE_RULES.items()}},
                   "evidence_ids": IDS, "rationale": {"type": "string", "minLength": 1},
                   "failure_modes": {"type": "array", "minItems": 1, "items": {"type": "string"}}},
}
TOOL_SPECS = {
    "selection_innovation": {
        "script": "scripts/mine_selection_innovation.py",
        "module": "src/evomolsteer/continuous/selection_innovation.py",
        "flags": {"--dataset", "--campaign", "--reference", "--output", "--seed", "--neighbours"},
    },
    "flow_compatibility": {
        "script": "scripts/flowcompat_agent.py",
        "module": "src/evomolsteer/continuous/flowcompat_agents.py",
        "flags": {"--action", "--program", "--reference", "--output", "--source-module", "--model-audit"},
    },
}


def _text_sha(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _path(root, value):
    path = Path(value)
    return (path if path.is_absolute() else Path(root) / path).resolve()


def _finite_window(window):
    if len(window) != 2 or any(isinstance(v, bool) or not isinstance(v, (float, int)) or
                               not math.isfinite(v) for v in window) or not 0 <= window[0] < window[1] <= 1:
        raise ValueError("Finite learned support within model time required")
    return list(window)


def audit_flow_configuration(program, reference, output, source_modules=(), model_audit=None):
    """Read-only configuration audit; does not claim a measured model attribution."""
    from ..generation.window_reference import load_reference
    p, ref = read_json(program), load_reference(reference)
    window = _finite_window(p["window"])
    checks = {
        "window_matches_reference": window == ref["window"],
        "reference_digest_matches": p["reference_sha256"] == digest(reference),
        "actual_endpoint_vjp": p.get("derivative_path") == "flowr_endpoint_vjp",
        "no_affinity_head_gradient": p.get("affinity_head_gradient") is False,
        "no_extra_production_head_calls": p.get("additional_per_step_affinity_calls") == 0,
        "coordinate_endpoint_representation": p.get("coordinate_representation") == "predicted_endpoint_world_A",
    }
    result = {"schema_version": "flowcompat-configuration-audit-1.0", "window": window,
              "program_sha256": digest(program), "reference_sha256": digest(reference),
              "checks": checks, "passed": all(checks.values()),
              "source_modules": {str(Path(v).resolve()): digest(v) for v in source_modules},
              "attribution_scope": "Configuration/source checks only; no live Jacobian or neural-module causality measurement"}
    if model_audit:
        result["model_source_audit"] = {"path": str(Path(model_audit).resolve()), "sha256": digest(model_audit)}
    write_json(output, result)
    if not result["passed"]:
        raise ValueError("Flow compatibility configuration failed")
    return result


def run_registered_tool(plan_path, receipt_path, root=PROJECT, python=sys.executable):
    """Execute a literal argv plan without a shell; record inputs/code/outputs."""
    plan = read_json(plan_path)
    if set(plan) != {"tool_id", "arguments", "input_files", "output_files"}:
        raise ValueError("Exact tool plan keys required")
    spec = TOOL_SPECS.get(plan["tool_id"])
    if spec is None:
        raise ValueError("Unregistered tool")
    args = plan["arguments"]
    if not isinstance(args, list) or not args or len(args) % 2 or not all(isinstance(v, str) for v in args):
        raise ValueError("Flag/value argument pairs required")
    if set(args[::2]) - spec["flags"]:
        raise ValueError("Unregistered tool argument")
    flags = args[::2]
    if any(flags.count(v) > 1 for v in set(flags) - {"--source-module"}):
        raise ValueError("Duplicate tool flag")
    if "--output" not in flags:
        raise ValueError("Explicit tool output required")
    if plan["tool_id"] == "flow_compatibility" and dict(zip(args[::2], args[1::2])).get("--action") != "audit-flow":
        raise ValueError("Only read-only audit action is registered")
    if not plan["input_files"] or not plan["output_files"]:
        raise ValueError("Declared input and output artifacts required")
    inputs = {_path(root, v).as_posix(): digest(_path(root, v)) for v in plan["input_files"]}
    output_paths = [_path(root, v) for v in plan["output_files"]]
    if len(set(output_paths)) != len(output_paths) or any(v.as_posix() in inputs for v in output_paths):
        raise ValueError("Outputs must not overwrite inputs")
    if any(v.exists() for v in output_paths):
        raise FileExistsError("Tool output already exists; use a fresh immutable directory")
    destination = _path(root, args[flags.index("--output") * 2 + 1])
    if any(v != destination and not v.is_relative_to(destination) for v in output_paths):
        raise ValueError("Declared output outside actual tool destination")
    for flag, value in zip(args[::2], args[1::2]):
        if flag in {"--reference", "--program", "--source-module", "--model-audit"} and _path(root, value).as_posix() not in inputs:
            raise ValueError("Unbound tool input: " + flag)
    script, module = _path(root, spec["script"]), _path(root, spec["module"])
    codes = {script.as_posix(): digest(script), module.as_posix(): digest(module)}
    command = [str(python), str(script), *args]
    completed = subprocess.run(command, cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace")
    changed = [path for path, sha in inputs.items() if digest(path) != sha]
    outputs = {v.as_posix(): digest(v) for v in output_paths if v.is_file()}
    receipt = {"schema_version": "flowcompat-tool-receipt-1.0", "tool_id": plan["tool_id"],
               "plan_path": str(Path(plan_path).resolve()), "plan_sha256": digest(plan_path),
               "command": command, "returncode": completed.returncode, "input_files": inputs,
               "output_files": outputs, "code_files": codes, "inputs_unchanged": not changed,
               "complete_outputs": len(outputs) == len(output_paths),
               "stdout_tail": completed.stdout[-4000:], "stderr_tail": completed.stderr[-4000:]}
    write_json(receipt_path, receipt)
    if completed.returncode or changed or not receipt["complete_outputs"]:
        raise ValueError("Tool failed or artifacts incomplete; inspect preserved receipt")
    return receipt


def _check_receipt(path):
    r = read_json(path)
    if r.get("schema_version") != "flowcompat-tool-receipt-1.0" or r.get("tool_id") not in TOOL_SPECS:
        raise ValueError("Registered execution receipt required")
    if r.get("returncode") != 0 or not r.get("inputs_unchanged") or not r.get("complete_outputs"):
        raise ValueError("Unsuccessful tool receipt")
    if digest(r["plan_path"]) != r["plan_sha256"]:
        raise ValueError("Tool plan changed")
    plan = read_json(r["plan_path"])
    if plan["tool_id"] != r["tool_id"] or r["command"][2:] != plan["arguments"]:
        raise ValueError("Executed argv differs from bound tool plan")
    suffix = TOOL_SPECS[r["tool_id"]]["script"]
    if not Path(r["command"][1]).as_posix().endswith("/" + suffix):
        raise ValueError("Unregistered executed script")
    for group in ("input_files", "output_files", "code_files"):
        if not r.get(group):
            raise ValueError("Empty artifact binding")
        for file, sha in r[group].items():
            if digest(file) != sha:
                raise ValueError("Tool artifact changed: " + file)
    return r


def _evidence_ids(packet):
    rows = packet.get("evidence_items", packet.get("evidence", []))
    result = {row.get("id", row.get("evidence_id")) for row in rows}
    if not rows or None in result or len(result) != len(rows):
        raise ValueError("Unique supplied evidence identifiers required")
    return result


def _packet_window(packet):
    return _finite_window(packet.get("window", packet.get("scope", {}).get("window", [])))


def export_request(evidence, registry, receipts, output, role, *, analyst=None, root=PROJECT):
    """Store literal role/module text and bind compact evidence without duplicating it."""
    if role not in ("Analyst", "Designer"):
        raise ValueError("Unknown role")
    packet = read_json(evidence); _evidence_ids(packet)
    window = _packet_window(packet)
    rs = [_check_receipt(v) for v in receipts]
    if {r["tool_id"] for r in rs} != set(TOOL_SPECS):
        raise ValueError("Both actual innovation mining and flow compatibility tool receipts required")
    if not any(Path(evidence).resolve().as_posix() in r["output_files"] for r in rs):
        raise ValueError("Evidence must be an executed tool output")
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    role_path = Path(root) / "skills" / role.lower() / "SKILL.md"
    module_path = Path(root) / "skills/flow-compatibility/SKILL.md"
    text = role_path.read_text(encoding="utf-8").rstrip() + "\n\n" + module_path.read_text(encoding="utf-8").rstrip() + "\n"
    bindings = {"evidence_path": str(Path(evidence).resolve()), "registry_path": str(Path(registry).resolve()),
                "registry_sha256": digest(registry),
                "skill_files": {str(p.resolve()): digest(p) for p in (role_path, module_path)},
                "tool_receipts": {str(Path(v).resolve()): digest(v) for v in receipts}}
    if role == "Designer":
        if analyst is None:
            raise ValueError("Validated actual Analyst response required")
        ap = Path(analyst)
        validate_response(ap.parent / "Analyst.flowcompat.request.json", ap)
        bindings.update(analyst_path=str(ap.resolve()), analyst_response_sha256=digest(ap))
    request = {"schema_version": VERSION, "role": role, "window": window,
               "evidence_sha256": digest(evidence), "instruction_sha256": _text_sha(text),
               "system_instruction": text, "bindings": bindings,
               "response_schema": ANALYST if role == "Analyst" else DESIGNER}
    path = output / (role + ".flowcompat.request.json")
    if path.exists():
        raise FileExistsError("Role request is immutable")
    write_json(path, request)
    (output / (role + ".flowcompat.instructions.md")).write_text(text, encoding="utf-8")
    return path


def payload(request):
    b = request["bindings"]
    if digest(b["evidence_path"]) != request["evidence_sha256"] or digest(b["registry_path"]) != b["registry_sha256"]:
        raise ValueError("Evidence or formula registry changed")
    if _text_sha(request["system_instruction"]) != request["instruction_sha256"]:
        raise ValueError("Literal instructions changed")
    for path, sha in b["skill_files"].items():
        if digest(path) != sha:
            raise ValueError("Bound skill changed")
    rs = []
    for path, sha in b["tool_receipts"].items():
        if digest(path) != sha:
            raise ValueError("Tool receipt changed")
        rs.append(_check_receipt(path))
    result = {"evidence": read_json(b["evidence_path"]), "registry": read_json(b["registry_path"]),
              "tool_receipts": rs, "request_bindings": {"instruction_sha256": request["instruction_sha256"],
              "evidence_sha256": request["evidence_sha256"], "tool_receipt_sha256": sorted(b["tool_receipts"].values())}}
    if _packet_window(result["evidence"]) != request["window"]:
        raise ValueError("Bound evidence/request support mismatch")
    if "analyst_path" in b:
        if digest(b["analyst_path"]) != b["analyst_response_sha256"]:
            raise ValueError("Analyst response changed")
        result["analyst_document"] = read_json(b["analyst_path"])
        result["analyst_response_sha256"] = b["analyst_response_sha256"]
    return result


def validate_response(request_path, response_path):
    request, response = read_json(request_path), read_json(response_path)
    jsonschema.validate(response, request["response_schema"])
    values = payload(request)
    expected = {"request_sha256": digest(request_path), "instruction_sha256": request["instruction_sha256"],
                "evidence_sha256": request["evidence_sha256"], "window": request["window"]}
    if any(response[k] != value for k, value in expected.items()):
        raise ValueError("Response provenance or learned support mismatch")
    if sorted(response["tool_receipt_sha256"]) != sorted(request["bindings"]["tool_receipts"].values()):
        raise ValueError("Response did not bind executed tools")
    known = _evidence_ids(values["evidence"])
    entries = response["observations"] + response["hypotheses"] + response["counterevidence"] if request["role"] == "Analyst" else [response]
    if any(set(entry["evidence_ids"]) - known for entry in entries):
        raise ValueError("Unprovided evidence citation")
    if request["role"] == "Designer":
        if response["analyst_response_sha256"] != values["analyst_response_sha256"]:
            raise ValueError("Designer did not consume bound Analyst response")
        updates = response["updates"]
        if any(isinstance(v, bool) or not math.isfinite(v) for v in updates.values()):
            raise ValueError("Finite registered numeric updates required")
        registry = values["registry"]
        if response["decision"] == "defer":
            if updates or response["base_program_id"] is not None or response["formula_id"] is not None:
                raise ValueError("Deferred response cannot compile")
        else:
            if response["base_program_id"] not in registry["programs"] or response["formula_id"] not in registry["formulas"]:
                raise ValueError("Unknown registered program or formula")
            formula = registry["formulas"][response["formula_id"]]
            if set(updates) - set(formula["allowed_updates"]):
                raise ValueError("Inactive or unregistered formula parameter")
            if response["decision"] == "retain" and updates:
                raise ValueError("Retain cannot change parameters")
            if response["decision"] == "trial" and not updates:
                raise ValueError("Trial needs an explicit update")
            if len(updates) > 1:
                raise ValueError("Sequential single-axis exploration required")
    return response


def import_response(request_path, response_path, output=None):
    response = validate_response(request_path, response_path)
    request = read_json(request_path)
    folder = Path(output) if output else Path(request_path).parent
    target = folder / (request["role"] + ".flowcompat.response.json")
    source_digest = digest(response_path)
    if target.exists() and target.resolve() != Path(response_path).resolve() and read_json(target) != response:
        raise FileExistsError("Imported role response is immutable")
    write_json(target, response)
    write_json(folder / (request["role"] + ".flowcompat.validation.json"), {
        "schema_valid": True, "grounding_valid": True, "passed": True, "request_sha256": digest(request_path),
        "response_sha256": digest(target), "source_response_sha256": source_digest,
        "instruction_sha256": request["instruction_sha256"], "evidence_sha256": request["evidence_sha256"],
        "tool_receipt_sha256": sorted(request["bindings"]["tool_receipts"].values()),
        "efficacy_assessed": False})
    return response


def compile_design(request_path, response_path, output, round_number, root=PROJECT):
    response = validate_response(request_path, response_path)
    if response["agent"] != "Designer":
        raise ValueError("Designer response required")
    if isinstance(round_number, bool) or not isinstance(round_number, int) or not 1 <= round_number <= 30:
        raise ValueError("Round outside authorized thirty-round bound")
    if response["decision"] == "defer":
        return None
    request = read_json(request_path); registry = payload(request)["registry"]
    entry, formula = registry["programs"][response["base_program_id"]], registry["formulas"][response["formula_id"]]
    source, ref = _path(root, entry["program_path"]), _path(root, entry["reference_path"])
    if digest(source) != entry["program_sha256"] or digest(ref) != entry["reference_sha256"]:
        raise ValueError("Frozen registered program or reference changed")
    from ..generation.window_reference import load_reference
    program = copy.deepcopy(read_json(source))
    reference = load_reference(ref)
    if program["window"] != response["window"] or reference["window"] != response["window"]:
        raise ValueError("Design/reference learned support mismatch")
    if program.get("derivative_path") != "flowr_endpoint_vjp" or program.get("affinity_head_gradient") is not False or program.get("additional_per_step_affinity_calls") != 0:
        raise ValueError("Actual endpoint VJP without head gradient required")
    if formula["reward_view"] not in ("endpoint_pointcloud", "endpoint_innovation"):
        raise ValueError("Unimplemented formula family")
    if response["decision"] == "retain" and program["reward_view"] != formula["reward_view"]:
        raise ValueError("Retain must preserve the registered formula")
    if formula["reward_view"] == "endpoint_pointcloud" and any(k.startswith("innovation.") for k in response["updates"]):
        raise ValueError("Innovation parameter requires its registered reward family")
    program["reward_view"] = formula["reward_view"]
    program["reference_sha256"] = digest(ref)
    for key, value in response["updates"].items():
        if "." in key:
            parent, field = key.split("."); program.setdefault(parent, {})[field] = value
        else:
            program[key] = value
    # Do not silently add defaults: no-op candidates must retain baseline arithmetic.
    program["round"] = round_number
    program["agent_request_sha256"] = digest(request_path)
    program["flowcompat_provenance"] = {"request_sha256": digest(request_path), "designer_response_sha256": digest(response_path),
        "analyst_response_sha256": response["analyst_response_sha256"], "instruction_sha256": request["instruction_sha256"],
        "evidence_sha256": request["evidence_sha256"], "registry_sha256": request["bindings"]["registry_sha256"],
        "tool_receipt_sha256": response["tool_receipt_sha256"], "formula_id": response["formula_id"],
        "parameter_updates": response["updates"], "implementation_gate_required": True}
    modules = formula.get("code_files", {})
    if not modules:
        raise ValueError("Registered numerical module digests required")
    for file, sha in modules.items():
        if digest(_path(root, file)) != sha:
            raise ValueError("Registered numerical module changed")
    program["flowcompat_provenance"]["code_files"] = modules
    write_json(output, program)
    return program


def verify_execution_response(program_path, execution_path, output):
    """Verify mechanism-specific actual response; deliberately ignore efficacy."""
    program, execution = read_json(program_path), read_json(execution_path)
    provenance = program["flowcompat_provenance"]
    if execution.get("program_sha256") != digest(program_path) or execution.get("request_sha256") != provenance["request_sha256"]:
        raise ValueError("Actual execution program/request mismatch")
    if execution.get("code_files") != provenance["code_files"]:
        raise ValueError("Actual numerical module digest mismatch")
    checks = execution.get("execution_checks", {})
    required = ("actual_flowr_model", "actual_endpoint_vjp", "no_resampling", "no_head_gradient",
                "no_extra_production_forward", "window_matches", "initial_state_pair_matches")
    if any(checks.get(k) is not True for k in required):
        raise ValueError("Actual inference contract incomplete")
    controls = execution.get("no_op_control", {})
    if controls.get("exact_baseline_arithmetic") is not True or controls.get("max_gradient_relative_change") != 0:
        raise ValueError("Exact new-mechanism no-op control required")
    changes = provenance["parameter_updates"]
    diag = execution.get("diagnostics", {})
    def positive(key):
        value = diag.get(key)
        return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value) and value > 1e-8
    if changes:
        key, value = next(iter(changes.items()))
        nonzero = value != (1. if key == "flow_control.parallel_component_scale" else 0.)
        if nonzero:
            if not positive("paired_window_coordinate_rms_A"):
                raise ValueError("Actual trajectory did not respond to new mechanism")
            needed = ["contrast_gradient_relative_change"] if key.startswith("innovation.") else ["flowcompat_gradient_adjustment_relative_rms"]
            if key == "innovation.field_strength_A":
                needed.append("contrast_teacher_shift_rms_A")
            if key == "innovation.reliability_power" and program.get("innovation", {}).get("field_strength_A", 0) == 0:
                raise ValueError("Reliability update inactive without a teacher shift")
            if key in {"flow_control.time_envelope_power", "flow_control.jacobian_gain_saturation", "flow_control.gradient_norm_saturation"}:
                # These gates change the dose rather than the unit direction.
                needed = ["flowcompat_schedule_absolute_change"]
            if key == "native_rms_ratio":
                needed = ["guidance_dose_relative_change"]
            if any(not positive(k) for k in needed):
                raise ValueError("Expected new mechanism diagnostic did not respond")
    result = {"schema_version": "flowcompat-implementation-gate-1.0", "implementation_passed": True,
              "program_sha256": digest(program_path), "execution_sha256": digest(execution_path),
              "request_sha256": provenance["request_sha256"], "parameter_updates": changes,
              "diagnostics": diag, "efficacy_assessed": False,
              "interpretation": "Bound tool/instruction/program and mechanism response; no affinity improvement claim"}
    write_json(output, result)
    return result


def call_api(request_path, *, client=None):
    """Explicit OpenAI-compatible transport; no implicit endpoint/model/key."""
    base, model, key = (os.environ.get("EVOMOLSTEER_BASE_URL"), os.environ.get("EVOMOLSTEER_MODEL"), os.environ.get("EVOMOLSTEER_API_KEY"))
    if not all((base, model, key)):
        raise ValueError("Explicit EVOMOLSTEER_BASE_URL, EVOMOLSTEER_MODEL and EVOMOLSTEER_API_KEY required")
    request = read_json(request_path)
    values = payload(request); values["request_sha256"] = digest(request_path)
    body = {"model": model, "temperature": 0,
            "messages": [{"role": "system", "content": request["system_instruction"] + "\nReturn JSON matching the supplied schema:\n" + json.dumps(request["response_schema"])},
                         {"role": "user", "content": json.dumps(values, ensure_ascii=False, allow_nan=False)}],
            "response_format": {"type": "json_object"}}
    owned = client is None
    client = client or httpx.Client(timeout=180)
    try:
        r = client.post(base.rstrip("/") + "/chat/completions", headers={"Authorization": "Bearer " + key}, json=body)
        r.raise_for_status()
        data = json.loads(r.json()["choices"][0]["message"]["content"])
    finally:
        if owned:
            client.close()
    raw = Path(request_path).parent / (request["role"] + ".flowcompat.raw_api.json")
    write_json(raw, data)
    return import_response(request_path, raw)
