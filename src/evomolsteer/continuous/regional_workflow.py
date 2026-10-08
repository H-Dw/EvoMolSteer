"""Independent hash-bound regional knowledge profile over the frozen v2 reward.

The supplemental evidence transport is reused read-only. Its compiler is NOT
called: that compiler correctly requires references emitted by branch_mutation,
whereas this profile requires a separately attested regional_reference output.
No model-supplied executable text crosses the declarative compile boundary.
"""
from __future__ import annotations

import copy
import gzip
import json
import math
import os
from pathlib import Path

import httpx
import jsonschema
import numpy as np

from ..io import digest, read_json, write_json
from . import flowcompat_supplemental as supplemental
from . import regional_reference as synthesis

PROJECT = supplemental.PROJECT
VERSION = "flowcompat-regional-agent-1.0"
SYNTHESIS_SCRIPT = "scripts/synthesize_regional_reference.py"
SYNTHESIS_MODULE = "src/evomolsteer/continuous/regional_reference.py"
FROZEN_HELPERS = {
    "src/evomolsteer/continuous/flowcompat_supplemental.py": "53ab060aeb0949bb136cbd82a5421fdc31a6de83ad8f0071c424e4cb4174879f",
    "scripts/flowcompat_supplemental_agent.py": "254b99b3b9ecc28e4abd0c1f902912e505776ad02e81f61b78c8d02909597aea",
    "skills/flow-compatibility-supplemental/SKILL.md": "ea991816cfc49e79ca1d0d58833cb760464c3b37691abd786a89bc2e959e9af3",
    SYNTHESIS_MODULE: "3083769f31b2b2e68721e955e9254031bb16181c2002613485e30a54566692ab",
    SYNTHESIS_SCRIPT: "60da4956ab00ca8968a42fe315ed5bec561a97af646e163bf8efe036822be32d",
}
SYNTHESIS_FLAGS = {
    "--mining", "--reference", "--source-receipt", "--branch-receipt", "--output",
    "--output-reference", "--receipt", "--q-threshold", "--minimum-batches",
    "--minimum-loo-support", "--absolute-floor",
}
REGIONAL_TOOLS = {
    "regional_reference": {"schema": "regional-reference-live-execution-receipt-1.0",
                           "script": SYNTHESIS_SCRIPT, "module": SYNTHESIS_MODULE,
                           "flags": SYNTHESIS_FLAGS, "summary_outputs": {"regional_rules.json", "manifest.json"}},
    "regional_common_depth": {"schema": "regional-common-depth-live-execution-receipt-1.0",
                              "script": "scripts/synthesize_common_depth_reference.py",
                              "module": "src/evomolsteer/continuous/regional_common_depth.py",
                              "flags": SYNTHESIS_FLAGS | {"--depths"},
                              "summary_outputs": {"regional_rules.json", "manifest.json", "report.zh-CN.md"}},
}
ANALYST, DESIGNER = copy.deepcopy(supplemental.ANALYST), copy.deepcopy(supplemental.DESIGNER)
for schema in (ANALYST, DESIGNER):
    schema["properties"]["schema_version"] = {"const": VERSION}
    for field in ("regional_tool_receipt_sha256", "regional_reference_sha256", "registry_sha256"):
        schema["properties"][field] = supplemental.original.SHA
        schema["required"].append(field)
DESIGNER["properties"]["updates"]["properties"] = {
    "branch_mixture.virtual_mass": {"type": "number", "minimum": 0., "maximum": .5}
}


def assert_frozen_helpers():
    supplemental.assert_frozen_original()
    for relative, sha in FROZEN_HELPERS.items():
        if digest(PROJECT / relative) != sha:
            raise ValueError("Frozen helper changed: " + relative)
    return {**supplemental.FROZEN_ORIGINALS, **FROZEN_HELPERS}


def _path(root, value):
    return supplemental.original._path(root, value)


def _teacher_invariants(original, transformed, rules):
    """Independently compare immutable truth and reconstructed joint mode."""
    if original["window"] != transformed["window"] or transformed["window"] != rules["window"]:
        raise ValueError("Regional teacher scope differs from learned scope")
    if len(original["frames"]) != len(transformed["frames"]):
        raise ValueError("Regional teacher frame count changed")
    for key, value in original.items():
        if key != "frames" and transformed.get(key) != value:
            raise ValueError("Immutable original reference field changed: " + key)
    for before, after in zip(original["frames"], transformed["frames"]):
        for key, value in before.items():
            if key not in ("teacher_contrast_direction_unit", "teacher_contrast_confidence") and after.get(key) != value:
                raise ValueError("Immutable teacher structure/score/prior/provenance changed: " + key)
        if after.get("teacher_original_branch_direction_unit") != before["teacher_contrast_direction_unit"] or after.get("teacher_original_branch_confidence") != before["teacher_contrast_confidence"]:
            raise ValueError("Original branch direction/confidence reconstruction changed")
    reconstructed, coverage = synthesis.synthesize_reference(original, rules,
        source_reference_sha256=transformed["regional_reference"]["source_reference_sha256"])
    for expected, actual in zip(reconstructed["frames"], transformed["frames"]):
        if expected != actual:
            raise ValueError("Actual regional direction is not the registered full-joint recipe")
    if coverage != rules["teacher_coverage"]:
        raise ValueError("Regional teacher coverage no longer matches the live synthesis")
    directions = np.asarray([f["teacher_contrast_direction_unit"] for f in transformed["frames"]], float)
    rms = np.sqrt(np.sum(directions ** 2, axis=-1).mean(axis=-1))
    active = rms > 0
    if not np.isfinite(directions).all() or (active.any() and not np.allclose(rms[active], 1., atol=1e-12, rtol=0)):
        raise ValueError("Regional joint teacher must have finite unit atom RMS")
    if np.max(np.abs(directions.mean(axis=-2))) > 1e-12:
        raise ValueError("Regional teacher mode introduces global translation")
    return {"original_teacher_truth_unchanged": True, "registered_joint_recipe_matches": True,
            "unit_atom_RMS": True, "zero_translation": True, "coverage": coverage}


def check_regional_receipt(receipt_path, reference_path):
    """Verify the real new tool separately from all old branch tool receipts."""
    assert_frozen_helpers()
    receipt_path, reference_path = Path(receipt_path).resolve(), Path(reference_path).resolve()
    tool_id = read_json(receipt_path).get("tool_id")
    spec = REGIONAL_TOOLS.get(tool_id)
    if spec is None:
        raise ValueError("Unregistered independent regional data tool")
    receipt, bound = synthesis.validate_live_receipt(receipt_path, tool_id)
    if receipt.get("schema_version") != spec["schema"]:
        raise ValueError("Actual regional-reference live receipt required")
    command = receipt["command"]
    script = Path(command[1]).resolve()
    if not script.as_posix().endswith("/" + spec["script"]):
        raise ValueError("Unregistered regional synthesis script")
    root = script.parents[1]
    args = command[2:]
    if len(args) % 2 or not all(isinstance(v, str) for v in args) or len(set(args[::2])) != len(args[::2]) or set(args[::2]) - spec["flags"]:
        raise ValueError("Unknown or duplicate regional synthesis argv")
    parsed = dict(zip(args[::2], args[1::2]))
    required = {"--mining", "--reference", "--source-receipt", "--branch-receipt", "--output", "--output-reference"}
    if not required <= set(parsed):
        raise ValueError("Regional source/output paths must be explicit in actual argv")
    if _path(root, parsed["--output-reference"]) != reference_path:
        raise ValueError("New reference is not the actual synthesis output")
    folder = _path(root, parsed["--output"])
    expected_receipt = _path(root, parsed["--receipt"]) if "--receipt" in parsed else folder / "execution_receipt.json"
    if expected_receipt != receipt_path:
        raise ValueError("Regional live receipt location differs from captured argv")
    expected_outputs = {str(folder / name) for name in spec["summary_outputs"]} | {str(reference_path)}
    if set(receipt["output_files"]) != expected_outputs:
        raise ValueError("Actual regional output set differs from registered recipe")
    expected_codes = supplemental._source_closure(root, (spec["script"], spec["module"]), eager_only=True)
    if {Path(k).as_posix(): v for k, v in receipt["code_files"].items()} != expected_codes:
        raise ValueError("Regional actual source closure incomplete or changed")
    upstream = synthesis.prepare_inputs(_path(root, parsed["--mining"]), _path(root, parsed["--reference"]),
                                       _path(root, parsed["--source-receipt"]), _path(root, parsed["--branch-receipt"]))
    if receipt["input_files"] != upstream:
        raise ValueError("Regional live receipt omitted or substituted upstream artifacts")
    rules_path, manifest_path = folder / "regional_rules.json", folder / "manifest.json"
    rules, manifest = read_json(rules_path), read_json(manifest_path)
    reference = json.loads(gzip.decompress(reference_path.read_bytes()))
    source_reference = _path(root, parsed["--reference"])
    original = json.loads(gzip.decompress(source_reference.read_bytes()))
    if rules["source_reference_sha256"] != digest(source_reference) or reference["regional_reference"]["source_reference_sha256"] != digest(source_reference):
        raise ValueError("Regional source teacher library changed")
    if reference["regional_reference"]["rules_sha256"] != digest(rules_path) or manifest["rules_sha256"] != digest(rules_path) or manifest["reference_sha256"] != digest(reference_path):
        raise ValueError("Regional rules/reference manifest hashes differ")
    if manifest["generation_performed"] is not False or manifest["new_model_or_affinity_head"] is not False:
        raise ValueError("Pure-data regional synthesis required")
    for region in rules["selected_regions"]:
        if region["independent_batches"] < rules["minimum_independent_batches"] or len(region["batch_ids"]) != region["independent_batches"] or len(set(region["batch_ids"])) != region["independent_batches"]:
            raise ValueError("Regional support must count independent batches")
        if region["absolute_covariance_norm"] <= rules["absolute_covariance_floor"] or region["loo_positive_fraction"] < rules["minimum_loo_positive_fraction"]:
            raise ValueError("Regional absolute effect/direction support below registered qualification")
        if not any(q < rules["q_threshold"] for q in region["component_q"]):
            raise ValueError("Regional frozen covariance significance is missing")
    invariants = _teacher_invariants(original, reference, rules)
    if tool_id == "regional_common_depth":
        selected_depths = {region["depth"] for region in rules["selected_regions"]}
        if len(selected_depths) > 1:
            raise ValueError("Common-depth regional tool cannot mix ancestry horizons")
    return {"receipt": receipt, "rules": rules, "manifest": manifest, "invariants": invariants,
            "source_reference_path": str(source_reference), "rules_path": str(rules_path),
            "manifest_path": str(manifest_path), "reference_path": str(reference_path),
            "reference_sha256": digest(reference_path), "receipt_sha256": digest(receipt_path), "tool_id": tool_id,
            "report_path": str(folder / "report.zh-CN.md") if "report.zh-CN.md" in spec["summary_outputs"] else None,
            "artifact_hashes": bound}


def regional_profile(source_registry, regional, *, root=PROJECT):
    """Explicit new registry; no hidden replacement in the old v2 registry."""
    eligible = {key: value for key, value in source_registry["formulas"].items()
                if value.get("reward_view") == "endpoint_branch_mixture" and value.get("reference_kind") == "branch_mutation"}
    if not eligible:
        raise ValueError("Frozen BranchMixtureReward formula is missing")
    formulas = {}
    for key, source in eligible.items():
        for name, sha in source.get("code_files", {}).items():
            if digest(_path(root, name)) != sha:
                raise ValueError("Frozen numerical formula source changed")
        if not source.get("code_files") or "branch_mixture.virtual_mass" not in source["allowed_updates"]:
            raise ValueError("Bound virtual-mass formula registration required")
        formula = copy.deepcopy(source)
        formula.update(reference_kind="regional_reference", allowed_updates=["branch_mixture.virtual_mass"], source_formula_id=key)
        formulas[key + "_regional"] = formula
    programs = {}
    for key, source in source_registry["programs"].items():
        if source["reference_sha256"] != regional["rules"]["source_reference_sha256"]:
            continue
        if digest(_path(root, source["program_path"])) != source["program_sha256"] or digest(_path(root, source["reference_path"])) != source["reference_sha256"]:
            raise ValueError("Frozen source program or original branch reference changed")
        programs[key + "_regional"] = {
            **copy.deepcopy(source), "reference_path": regional["reference_path"],
            "reference_sha256": regional["reference_sha256"], "reference_kind": "regional_reference",
            "source_program_id": key, "source_reference_path": source["reference_path"],
            "source_reference_sha256": source["reference_sha256"],
        }
    if not programs:
        raise ValueError("No registered parent program uses the original branch teacher library")
    return {"schema_version": "flowcompat-regional-registry-1.0", "programs": programs, "formulas": formulas,
            "context": copy.deepcopy(source_registry.get("context", {})),
            "context_files": copy.deepcopy(source_registry.get("context_files", {})),
            "reference_origin": {"tool_id": regional["receipt"]["tool_id"], "receipt_sha256": regional["receipt_sha256"],
                                 "reference_sha256": regional["reference_sha256"], "rules_sha256": digest(regional["rules_path"])},
            "compiler": "Independent regional compiler; frozen supplemental compiler intentionally not used"}


def export_request(base_request, regional_receipt, reference, output, role, *, analyst=None, analyst_request=None, context_files=(), root=PROJECT):
    assert_frozen_helpers()
    if role not in ("Analyst", "Designer"):
        raise ValueError("Analyst or Designer role required")
    base_request = Path(base_request).resolve()
    base = read_json(base_request)
    if base.get("schema_version") != supplemental.VERSION or base.get("role") != "Analyst":
        raise ValueError("Frozen supplemental Analyst evidence request required as read-only foundation")
    values = supplemental.payload(base)
    regional = check_regional_receipt(regional_receipt, reference)
    window = supplemental.original._finite_window(regional["rules"]["window"])
    if base["window"] != window:
        raise ValueError("Base evidence and regional learned window differ")
    folder = Path(output).resolve()
    if folder.exists():
        raise FileExistsError("Fresh role export directory required")
    folder.mkdir(parents=True)
    combined = copy.deepcopy(values["evidence"])
    rules = regional["rules"]
    for region in rules["selected_regions"]:
        combined["evidence_items"].append({"id": f"regional_reference/region_{region['region']:02d}/depth_{region['depth']}",
            "metric": "adjusted_covariance_joint_XYZ", "whole_window_xyz": region["whole_window_xyz"],
            "absolute_covariance_norm": region["absolute_covariance_norm"], "component_q": region["component_q"],
            "component_CI_low": region["component_CI_low"], "component_CI_high": region["component_CI_high"],
            "independent_batches": region["independent_batches"], "loo_positive_fraction": region["loo_positive_fraction"],
            "interpretation": "Discovery-selected coordinate direction hypothesis; not a causal affinity gradient"})
    combined["evidence_items"].append({"id": "regional_reference/teacher_coverage", **regional["manifest"]["coverage"],
                                      "interpretation": "Original local mutation eligibility retained; positive/negative/null direction tests pending"})
    combined["regional_joint_rules"] = rules
    combined["regional_teacher_invariants"] = regional["invariants"]
    supplemental.original._evidence_ids(combined)
    evidence_path, registry_path = folder / "regional.evidence.json", folder / "regional.registry.json"
    write_json(evidence_path, combined)
    write_json(registry_path, regional_profile(values["registry"], regional, root=root))
    skills = [Path(root) / f"skills/{role.lower()}/SKILL.md", Path(root) / "skills/flow-compatibility/SKILL.md",
              Path(root) / "skills/flow-compatibility-supplemental/SKILL.md", Path(root) / "skills/flow-compatibility-regional/SKILL.md"]
    text = "\n\n".join(path.read_text(encoding="utf-8").rstrip() for path in skills) + "\n"
    source_evidence = {**base["bindings"]["source_evidence"], regional["rules_path"]: digest(regional["rules_path"]),
                       regional["manifest_path"]: digest(regional["manifest_path"])}
    additional_context = {}
    for name in context_files:
        path = _path(root, name)
        if path.suffix.lower() not in (".md", ".txt", ".json") or path.stat().st_size > 256 * 1024:
            raise ValueError("Bounded literal regional context required")
        additional_context[str(path)] = digest(path)
        source_evidence[str(path)] = digest(path)
    if regional.get("report_path"):
        path = Path(regional["report_path"])
        if path.stat().st_size > 256 * 1024:
            raise ValueError("Bounded actual regional tool report required")
        additional_context[str(path)] = digest(path)
        source_evidence[str(path)] = digest(path)
    bindings = {"base_request_path": str(base_request), "base_request_sha256": digest(base_request),
        "evidence_path": str(evidence_path), "registry_path": str(registry_path), "registry_sha256": digest(registry_path),
        "source_evidence": source_evidence, "skill_files": {str(path.resolve()): digest(path) for path in skills},
        "tool_receipts": copy.deepcopy(base["bindings"]["tool_receipts"]),
        "extra_receipts": copy.deepcopy(base["bindings"]["extra_receipts"]),
        "regional_receipt_path": str(Path(regional_receipt).resolve()), "regional_receipt_sha256": digest(regional_receipt),
        "regional_reference_path": regional["reference_path"], "regional_reference_sha256": regional["reference_sha256"],
        "rules_path": regional["rules_path"], "rules_sha256": digest(regional["rules_path"]),
        "additional_context_files": additional_context,
        "workflow_files": {str(Path(__file__).resolve()): digest(__file__),
                           str((Path(root) / "scripts/flowcompat_regional_agent.py").resolve()): digest(Path(root) / "scripts/flowcompat_regional_agent.py")},
        "frozen_helpers": assert_frozen_helpers()}
    if role == "Designer":
        if analyst is None:
            raise ValueError("Imported regional Analyst response required")
        analyst_path = Path(analyst).resolve()
        analyst_request = Path(analyst_request).resolve() if analyst_request else analyst_path.parent / "Analyst.flowcompat-regional.request.json"
        _require_import(analyst_request, analyst_path, "Analyst")
        bindings.update(analyst_path=str(analyst_path), analyst_response_sha256=digest(analyst_path), analyst_request_path=str(analyst_request), analyst_request_sha256=digest(analyst_request))
    request = {"schema_version": VERSION, "role": role, "window": window, "evidence_sha256": digest(evidence_path),
               "instruction_sha256": supplemental.original._text_sha(text), "system_instruction": text,
               "bindings": bindings, "response_schema": ANALYST if role == "Analyst" else DESIGNER}
    request_path = folder / (role + ".flowcompat-regional.request.json")
    write_json(request_path, request)
    (folder / (role + ".flowcompat-regional.instructions.md")).write_text(text, encoding="utf-8")
    return request_path


def payload(request):
    assert_frozen_helpers()
    if request.get("schema_version") != VERSION or request.get("role") not in ("Analyst", "Designer"):
        raise ValueError("Independent regional role contract required")
    bindings = request["bindings"]
    if bindings["frozen_helpers"] != assert_frozen_helpers():
        raise ValueError("Frozen helper bindings changed")
    for group in ("skill_files", "source_evidence", "workflow_files", "tool_receipts", "extra_receipts", "additional_context_files"):
        for file, sha in bindings[group].items():
            if digest(file) != sha:
                raise ValueError("Bound regional artifact changed: " + file)
    for path_key, sha_key in (("base_request_path", "base_request_sha256"), ("evidence_path", None),
                               ("registry_path", "registry_sha256"), ("regional_reference_path", "regional_reference_sha256"),
                               ("regional_receipt_path", "regional_receipt_sha256"), ("rules_path", "rules_sha256")):
        expected = bindings[sha_key] if sha_key else request["evidence_sha256"]
        if digest(bindings[path_key]) != expected:
            raise ValueError("Regional data/request/profile hash changed")
    if supplemental.original._text_sha(request["system_instruction"]) != request["instruction_sha256"]:
        raise ValueError("Literal regional instruction changed")
    # write_json sorts dictionary keys; the contractual Skill order is explicit.
    skill_order = ["skills/" + request["role"].lower() + "/SKILL.md", "skills/flow-compatibility/SKILL.md",
                   "skills/flow-compatibility-supplemental/SKILL.md", "skills/flow-compatibility-regional/SKILL.md"]
    ordered = []
    for suffix in skill_order:
        matches = [file for file in bindings["skill_files"] if Path(file).as_posix().endswith("/" + suffix)]
        if len(matches) != 1:
            raise ValueError("Complete literal role and module Skills required")
        ordered.append(Path(matches[0]).read_text(encoding="utf-8").rstrip())
    if "\n\n".join(ordered) + "\n" != request["system_instruction"]:
        raise ValueError("Exported instructions omit or alter actual Skills")
    base = read_json(bindings["base_request_path"])
    base_values = supplemental.payload(base)
    regional = check_regional_receipt(bindings["regional_receipt_path"], bindings["regional_reference_path"])
    evidence, registry = read_json(bindings["evidence_path"]), read_json(bindings["registry_path"])
    if base["window"] != request["window"] or evidence["window"] != request["window"] or regional["rules"]["window"] != request["window"]:
        raise ValueError("Regional evidence/teacher learning windows differ")
    if registry != regional_profile(base_values["registry"], regional):
        raise ValueError("Regional registry no longer equals the explicit frozen-formula profile")
    result = {**base_values, "evidence": evidence, "registry": registry,
              "regional_tool_receipt": regional["receipt"], "regional_recipe": regional["rules"],
              "regional_teacher_invariants": regional["invariants"],
              "request_bindings": {
                  "instruction_sha256": request["instruction_sha256"], "evidence_sha256": request["evidence_sha256"],
                  "registry_sha256": bindings["registry_sha256"], "source_evidence_sha256": sorted(bindings["source_evidence"].values()),
                  "tool_receipt_sha256": sorted(bindings["tool_receipts"].values()),
                  "extra_tool_receipt_sha256": sorted(bindings["extra_receipts"].values()),
                  "regional_tool_receipt_sha256": bindings["regional_receipt_sha256"],
                  "regional_reference_sha256": bindings["regional_reference_sha256"],
              }}
    if bindings["additional_context_files"]:
        result["strategy_context"] = copy.deepcopy(base_values.get("strategy_context", {}))
        for file, sha in bindings["additional_context_files"].items():
            origin = "Actual regional tool output bound by its own live receipt" if file == regional.get("report_path") else "Independent read-only audit context; not a tool output receipt"
            result["strategy_context"][file] = {"sha256": sha, "text": Path(file).read_text(encoding="utf-8"),
                                               "provenance": origin}
    if "analyst_path" in bindings:
        if digest(bindings["analyst_request_path"]) != bindings["analyst_request_sha256"] or digest(bindings["analyst_path"]) != bindings["analyst_response_sha256"]:
            raise ValueError("Regional Analyst response/request changed")
        _require_import(bindings["analyst_request_path"], bindings["analyst_path"], "Analyst")
        result.update(analyst_document=read_json(bindings["analyst_path"]), analyst_response_sha256=bindings["analyst_response_sha256"])
    return result


def validate_response(request_path, response_path):
    request, response = read_json(request_path), read_json(response_path)
    jsonschema.validate(response, ANALYST if request["role"] == "Analyst" else DESIGNER)
    values = payload(request)
    expected = {"request_sha256": digest(request_path), "instruction_sha256": request["instruction_sha256"],
                "evidence_sha256": request["evidence_sha256"], "window": request["window"],
                **values["request_bindings"]}
    for key, value in expected.items():
        actual = sorted(response[key]) if isinstance(value, list) and key.endswith("sha256") else response[key]
        if actual != value:
            raise ValueError("Regional response did not bind actual instructions/data/receipt/profile: " + key)
    entries = response["observations"] + response["hypotheses"] + response["counterevidence"] if request["role"] == "Analyst" else [response]
    known = supplemental.original._evidence_ids(values["evidence"])
    if any(set(entry["evidence_ids"]) - known for entry in entries):
        raise ValueError("Unprovided regional evidence citation")
    for prefix in ("regional_reference/", "multi_depth/", "branch_mutation/"):
        if not any(any(identifier.startswith(prefix) for identifier in entry["evidence_ids"]) for entry in entries):
            raise ValueError("Regional roles must actually interpret new/ancestry/multidepth evidence: " + prefix)
    if request["role"] == "Designer":
        if response["analyst_response_sha256"] != values["analyst_response_sha256"]:
            raise ValueError("Regional Designer did not consume the imported Analyst")
        updates = response["updates"]
        if any(isinstance(v, bool) or not math.isfinite(v) for v in updates.values()):
            raise ValueError("Finite registered mass required")
        if response["decision"] == "defer":
            if updates or response["base_program_id"] is not None or response["formula_id"] is not None:
                raise ValueError("Deferred regional design cannot compile")
        else:
            registry = values["registry"]
            if response["base_program_id"] not in registry["programs"] or response["formula_id"] not in registry["formulas"]:
                raise ValueError("Unknown independent regional program/formula")
            if response["decision"] == "trial" and set(updates) != {"branch_mixture.virtual_mass"} or response["decision"] == "retain" and updates:
                raise ValueError("Fresh regional exploration changes only the registered mass axis")
    return response


def import_response(request_path, response_path, output=None):
    data = validate_response(request_path, response_path)
    request = read_json(request_path)
    folder = Path(output).resolve() if output else Path(request_path).resolve().parent
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (request["role"] + ".flowcompat-regional.response.json")
    validation_path = folder / (request["role"] + ".flowcompat-regional.validation.json")
    if path.exists() and read_json(path) != data:
        raise FileExistsError("Imported regional response is immutable")
    if validation_path.exists():
        previous = read_json(validation_path)
        if previous["request_sha256"] != digest(request_path) or previous["response_sha256"] != digest(path):
            raise FileExistsError("Regional validation record is immutable")
        # Re-importing identical validated content must not replace the original
        # raw-response provenance with a later serialization of the same JSON.
        return data
    source_sha = digest(response_path)
    write_json(path, data)
    write_json(validation_path, {"schema_version": VERSION, "passed": True, "schema_valid": True, "grounding_valid": True,
        "request_sha256": digest(request_path), "response_sha256": digest(path), "source_response_sha256": source_sha,
        "instruction_sha256": request["instruction_sha256"], "regional_reference_sha256": data["regional_reference_sha256"],
        "regional_tool_receipt_sha256": data["regional_tool_receipt_sha256"], "registry_sha256": data["registry_sha256"],
        "efficacy_assessed": False})
    return data


def _require_import(request_path, response_path, role):
    request_path, response_path = Path(request_path).resolve(), Path(response_path).resolve()
    if response_path.name != role + ".flowcompat-regional.response.json":
        raise ValueError("Actual canonical imported regional response required")
    validation_path = response_path.parent / (role + ".flowcompat-regional.validation.json")
    if not validation_path.is_file():
        raise ValueError("Actual regional response import validation is missing")
    validation = read_json(validation_path)
    if validation.get("schema_version") != VERSION or validation.get("passed") is not True or validation.get("schema_valid") is not True or validation.get("grounding_valid") is not True or validation.get("request_sha256") != digest(request_path) or validation.get("response_sha256") != digest(response_path):
        raise ValueError("Imported regional response validation/provenance changed")
    response = validate_response(request_path, response_path)
    if response["agent"] != role:
        raise ValueError("Wrong imported regional role")
    return response


def compile_design(request_path, response_path, output, round_number, *, root=PROJECT):
    assert_frozen_helpers()
    if isinstance(round_number, bool) or not isinstance(round_number, int) or not 1 <= round_number <= 30:
        raise ValueError("Round outside current registered budget")
    data = _require_import(request_path, response_path, "Designer")
    if data["decision"] == "defer":
        return None
    request = read_json(request_path)
    values = payload(request)
    registry = values["registry"]
    entry = registry["programs"][data["base_program_id"]]
    formula = registry["formulas"][data["formula_id"]]
    reference = _path(root, entry["reference_path"])
    source = _path(root, entry["program_path"])
    if digest(source) != entry["program_sha256"] or digest(reference) != entry["reference_sha256"]:
        raise ValueError("Regional program/reference changed")
    if formula["reference_kind"] != "regional_reference" or formula["reward_view"] != "endpoint_branch_mixture" or formula["allowed_updates"] != ["branch_mixture.virtual_mass"]:
        raise ValueError("Independent registered regional branch mixture required")
    regional = check_regional_receipt(request["bindings"]["regional_receipt_path"], reference)
    program = copy.deepcopy(read_json(source))
    if program["window"] != data["window"] or regional["rules"]["window"] != data["window"]:
        raise ValueError("Regional compiler cannot change input learning window")
    if program.get("derivative_path") != "flowr_endpoint_vjp" or program.get("affinity_head_gradient") is not False or program.get("additional_per_step_affinity_calls") != 0:
        raise ValueError("Actual conditional endpoint VJP without head gradient/extra forward required")
    for file, sha in formula["code_files"].items():
        if digest(_path(root, file)) != sha:
            raise ValueError("Frozen numerical reward/controller changed")
    if data["decision"] == "retain" and program.get("reward_view") != "endpoint_branch_mixture":
        raise ValueError("Retain cannot introduce a regional formula")
    # Explicit formula defaults, rather than an unregistered latent controller
    # change. Direction controls remain outside this fresh Agent design axis.
    parameters = copy.deepcopy(program.get("branch_mixture", {}))
    parameters.setdefault("virtual_mass", 0.)
    parameters.setdefault("direction_sign", 1.)
    parameters.setdefault("region_weight_mix", 0.)
    if data["updates"]:
        parameters["virtual_mass"] = data["updates"]["branch_mixture.virtual_mass"]
    if parameters["region_weight_mix"] != 0 or parameters["direction_sign"] != 1:
        raise ValueError("Fresh regional Agent profile must use positive joint direction and unchanged regional cost")
    program.update(reward_view="endpoint_branch_mixture", branch_mixture=parameters, reference_sha256=digest(reference),
                   round=round_number, agent_request_sha256=digest(request_path), generation_interface="flowcompat_v2")
    b = request["bindings"]
    program["flowcompat_provenance"] = {
        "workflow_version": VERSION, "compiler_version": VERSION, "request_sha256": digest(request_path),
        "designer_response_sha256": digest(response_path), "analyst_response_sha256": data["analyst_response_sha256"],
        "instruction_sha256": request["instruction_sha256"], "evidence_sha256": request["evidence_sha256"],
        "source_evidence_sha256": data["source_evidence_sha256"], "tool_receipt_sha256": data["tool_receipt_sha256"],
        "extra_tool_receipt_sha256": data["extra_tool_receipt_sha256"], "registry_sha256": b["registry_sha256"],
        "formula_id": data["formula_id"], "reference_kind": "regional_reference", "parameter_updates": data["updates"],
        "code_files": formula["code_files"], "context_files": {**registry.get("context_files", {}), **b["additional_context_files"]},
        "regional_recipe": {"receipt_path": b["regional_receipt_path"], "receipt_sha256": b["regional_receipt_sha256"],
                            "reference_path": str(reference), "reference_sha256": digest(reference),
                            "rules_path": b["rules_path"], "rules_sha256": b["rules_sha256"],
                            "original_branch_reference_sha256": regional["rules"]["source_reference_sha256"],
                            "synthesis_code_files": regional["receipt"]["code_files"],
                            "tool_id": regional["receipt"]["tool_id"]},
        "workflow_files": b["workflow_files"], "frozen_helpers": assert_frozen_helpers(), "implementation_gate_required": True,
        "paired_direction_controls_required": True,
    }
    if Path(output).exists():
        raise FileExistsError("Fresh compiled regional program destination required")
    write_json(output, program)
    return program


def verify_execution_response(program, execution, output):
    """Actual implementation gate, independent from all affinity efficacy tests."""
    assert_frozen_helpers()
    p, report = read_json(program), read_json(execution)
    provenance = p.get("flowcompat_provenance", {})
    if provenance.get("workflow_version") != VERSION or p.get("reward_view") != "endpoint_branch_mixture":
        raise ValueError("Independently compiled regional program required")
    recipe = provenance["regional_recipe"]
    if digest(recipe["receipt_path"]) != recipe["receipt_sha256"] or digest(recipe["reference_path"]) != recipe["reference_sha256"] or digest(recipe["rules_path"]) != recipe["rules_sha256"]:
        raise ValueError("Regional execution recipe artifact changed")
    check_regional_receipt(recipe["receipt_path"], recipe["reference_path"])
    for group in ("workflow_files", "code_files", "context_files"):
        for file, sha in provenance[group].items():
            if digest(_path(PROJECT, file)) != sha:
                raise ValueError("Regional execution source changed")
    if report.get("program_sha256") != digest(program) or report.get("request_sha256") != provenance["request_sha256"] or report.get("code_files") != provenance["code_files"]:
        raise ValueError("Actual regional program/request/module binding mismatch")
    checks = ("actual_flowr_model", "actual_endpoint_vjp", "no_resampling", "no_head_gradient",
              "no_extra_production_forward", "window_matches", "initial_state_pair_matches")
    if any(report.get("execution_checks", {}).get(key) is not True for key in checks):
        raise ValueError("Actual regional inference contract incomplete")
    no_op = report.get("no_op_control", {})
    if no_op.get("exact_baseline_arithmetic") is not True or no_op.get("max_gradient_relative_change") != 0:
        raise ValueError("Exact regional zero-mass baseline control required")
    diagnostics = report.get("diagnostics", {})
    keys = ("branch_virtual_mass_mean", "branch_virtual_teacher_shift_rms_A", "branch_gradient_relative_change", "paired_window_coordinate_rms_A")
    for key in keys:
        value = diagnostics.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("Actual finite regional mechanism diagnostic required: " + key)
    params = p.get("branch_mixture", {})
    for field, lo, hi in (("virtual_mass", 0., .5), ("direction_sign", -1., 1.)):
        value = params.get(field, 0. if field == "virtual_mass" else 1.)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not lo <= value <= hi:
            raise ValueError("Regional paired control parameter outside registered bounds")
    active = params.get("virtual_mass", 0) > 0 and params.get("direction_sign", 1) != 0
    if params.get("region_weight_mix", 0) != 0:
        raise ValueError("Regional fresh profile does not register region-cost changes")
    if active:
        if any(diagnostics[key] <= 1e-8 for key in keys):
            raise ValueError("New regional reward/trajectory failed to respond")
        mass = params["virtual_mass"]
        if diagnostics["branch_virtual_mass_mean"] > mass + max(1e-8, mass * 1e-6):
            raise ValueError("Regional virtual mass exceeded registered budget")
    elif any(diagnostics[key] != 0 for key in keys):
        raise ValueError("Zero regional mass/direction changed exact baseline arithmetic")
    result = {"schema_version": "flowcompat-implementation-gate-1.0", "workflow_version": VERSION,
              "implementation_passed": True, "program_sha256": digest(program), "execution_sha256": digest(execution),
              "request_sha256": provenance["request_sha256"], "parameter_updates": provenance["parameter_updates"],
              "diagnostics": diagnostics, "regional_reference_sha256": recipe["reference_sha256"],
              "regional_recipe_sha256": recipe["rules_sha256"], "efficacy_assessed": False,
              "interpretation": "Actual new joint-coordinate response or exact null; positive/reverse/null paired efficacy must be evaluated separately"}
    write_json(output, result)
    return result


def call_api(request_path, *, client=None):
    """Same explicit OpenAI-compatible transport, regional import validation."""
    base, model, key = [os.environ.get(name) for name in ("EVOMOLSTEER_BASE_URL", "EVOMOLSTEER_MODEL", "EVOMOLSTEER_API_KEY")]
    if not all((base, model, key)):
        raise ValueError("Explicit API base, model and key required")
    request = read_json(request_path)
    values = payload(request)
    values["request_sha256"] = digest(request_path)
    body = {"model": model, "temperature": 0, "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": request["system_instruction"] + "\nReturn JSON matching supplied schema:\n" + json.dumps(request["response_schema"])},
                         {"role": "user", "content": json.dumps(values, ensure_ascii=False, allow_nan=False)}]}
    owned = client is None
    client = client or httpx.Client(timeout=180)
    try:
        result = client.post(base.rstrip("/") + "/chat/completions", headers={"Authorization": "Bearer " + key}, json=body)
        result.raise_for_status()
        data = json.loads(result.json()["choices"][0]["message"]["content"])
    finally:
        if owned:
            client.close()
    raw = Path(request_path).parent / (request["role"] + ".flowcompat-regional.raw_api.json")
    if raw.exists():
        raise FileExistsError("Fresh regional API response destination required")
    write_json(raw, data)
    return import_response(request_path, raw)
