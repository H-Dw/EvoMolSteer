"""Versioned third-tool extension without mutating a frozen running contract.

Only numeric, registered single-axis designs compile.  Both API and subagent
simulation use the same validator.  A receipt is produced by real subprocess
execution; historical output directories cannot be retroactively attested.
"""
from __future__ import annotations

import ast
import copy
import gzip
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import httpx
import jsonschema
import pandas as pd

from ..io import digest, read_json, write_json
from . import flowcompat_agents as original

PROJECT = original.PROJECT
VERSION = "flowcompat-supplemental-agent-2.0"
FROZEN_ORIGINALS = {
    "src/evomolsteer/continuous/flowcompat_agents.py": "0251a4fa97b18f4358327424c2f351c6680dc3b65de0f7f4ed2fb02c70842c02",
    "scripts/flowcompat_agent.py": "0e569369b7f6995b5e7c430e64885110d3deaf64b9e2d9e8f260f41f8f64d80e",
    "skills/flow-compatibility/SKILL.md": "abdd4ecc19f6a7c5816ce7e33e3a59f3369cf4bbc4577b1adb258f154d9990c8",
}
ANALYST, DESIGNER = copy.deepcopy(original.ANALYST), copy.deepcopy(original.DESIGNER)
for schema in (ANALYST, DESIGNER):
    schema["properties"]["schema_version"] = {"const": VERSION}
    schema["properties"]["source_evidence_sha256"] = original.IDS
    schema["required"].append("source_evidence_sha256")
    schema["properties"]["extra_tool_receipt_sha256"] = {"type": "array", "uniqueItems": True, "items": original.SHA}
    schema["required"].append("extra_tool_receipt_sha256")
BRANCH_UPDATE_RULES = {
    "branch_mixture.virtual_mass": (0., .5),
    "branch_mixture.direction_sign": (-1., 1.),
    "branch_mixture.region_weight_mix": (0., 1.),
}
DESIGNER["properties"]["updates"]["properties"].update({
    key: {"type": "number", "minimum": lo, "maximum": hi}
    for key, (lo, hi) in BRANCH_UPDATE_RULES.items()})
BRANCH_SCRIPT = "scripts/mine_branch_mutation.py"
BRANCH_MODULE = "src/evomolsteer/continuous/branch_mutation.py"
BRANCH_FLAGS = {"--dataset", "--campaign", "--reference", "--output", "--seed", "--augmented-reference"}
EXTRA_SCRIPT = "scripts/mine_multi_depth_mutation.py"
EXTRA_MODULE = "src/evomolsteer/continuous/multi_depth_mutation.py"
EXTRA_FLAGS = {"--dataset", "--campaign", "--reference", "--output", "--depths", "--seed", "--region-width", "--nuisance", "--receipt"}


def assert_frozen_original():
    """Pin the imported helper implementation, old CLI and Skill by bytes."""
    for relative, sha in FROZEN_ORIGINALS.items():
        if digest(PROJECT / relative) != sha:
            raise ValueError("Frozen original interface changed: " + relative)
    return dict(FROZEN_ORIGINALS)


def _resolve(root, value):
    return original._path(root, value)


def _source_closure(root, starts, *, eager_only=False):
    """Hash the complete statically imported local Python source closure."""
    root = Path(root).resolve(); package = root / "src/evomolsteer"
    queue = [_resolve(root, v) for v in starts]; seen = {}
    def resolve_module(parts):
        if not parts or parts[0] != "evomolsteer": return None
        base = package.joinpath(*parts[1:])
        return base.with_suffix(".py") if base.with_suffix(".py").is_file() else base / "__init__.py" if (base / "__init__.py").is_file() else None
    while queue:
        path = queue.pop()
        if path in seen: continue
        seen[path] = digest(path)
        if path.is_relative_to(package):
            for folder in path.parents:
                if not folder.is_relative_to(package): break
                initializer = folder / "__init__.py"
                if initializer.is_file() and initializer not in seen: queue.append(initializer)
        local_package = ["evomolsteer", *path.relative_to(package).parts[:-1]] if path.is_relative_to(package) else []
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        if eager_only:
            nodes = []; pending = [tree]
            while pending:
                node = pending.pop(); nodes.append(node)
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                    pending.extend(ast.iter_child_nodes(node))
        else: nodes = ast.walk(tree)
        for node in nodes:
            candidates = []
            if isinstance(node, ast.Import):
                candidates.extend(alias.name.split(".") for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                prefix = local_package[:len(local_package) - node.level + 1] if node.level else []
                parts = prefix + (node.module.split(".") if node.module else [])
                candidates.append(parts)
                candidates.extend(parts + [alias.name] for alias in node.names)
            for parts in candidates:
                target = resolve_module(parts)
                if target is not None and target not in seen: queue.append(target)
    return {p.as_posix(): sha for p, sha in sorted(seen.items())}


def _branch_arguments(plan):
    if set(plan) != {"tool_id", "arguments", "input_files", "output_files"} or plan["tool_id"] != "branch_mutation":
        raise ValueError("Exact registered branch tool plan required")
    args = plan["arguments"]
    if not isinstance(args, list) or not args or len(args) % 2 or not all(isinstance(v, str) for v in args):
        raise ValueError("Literal flag/value pairs required")
    flags = args[::2]
    if len(set(flags)) != len(flags) or set(flags) - BRANCH_FLAGS:
        raise ValueError("Unknown or duplicate branch tool flag")
    parsed = dict(zip(flags, args[1::2]))
    if not {"--dataset", "--campaign", "--reference", "--output"} <= set(parsed):
        raise ValueError("Explicit dataset/campaign/reference/output required")
    if "--seed" in parsed and (not parsed["--seed"].isdigit() or int(parsed["--seed"]) < 0):
        raise ValueError("Nonnegative deterministic seed required")
    return parsed


def _consumed_branch_inputs(root, parsed):
    reference = _resolve(root, parsed["--reference"])
    ref = json.loads(gzip.decompress(reference.read_bytes()))
    dataset = _resolve(root, parsed["--dataset"])
    source = (dataset / "results" / parsed["--campaign"]).resolve()
    if not source.is_relative_to(dataset): raise ValueError("Campaign outside dataset")
    paths = [reference, source / "config.json"]
    for batch in ref["discovery_batches"]:
        if isinstance(batch, bool) or not isinstance(batch, int) or batch < 0: raise ValueError("Integer discovery batches required")
        folder = source / "single" / f"batch_{batch:03d}"
        options = [v for v in (folder / "trajectory.npz", folder / "trajectory.h5") if v.is_file()]
        if len(options) != 1: raise ValueError("Exactly one observed trajectory per discovery batch required")
        paths.extend([options[0], source / f"frame_batch_{batch:03d}.json"])
    return paths


def run_branch_tool(plan_path, receipt_path, *, root=PROJECT, python=sys.executable):
    """Execute a fresh third tool and bind all consumed/generated local data."""
    assert_frozen_original()
    plan = read_json(plan_path); parsed = _branch_arguments(plan)
    destination = _resolve(root, parsed["--output"])
    augmented = _resolve(root, parsed["--augmented-reference"]) if "--augmented-reference" in parsed else None
    if destination.exists() or (augmented is not None and augmented.exists()):
        raise FileExistsError("Third-tool receipt requires fresh actual execution, not historical outputs")
    declared = {_resolve(root, v) for v in plan["output_files"]}
    required = {destination / "evidence.json", destination / "manifest.json"}
    if augmented is not None: required.add(augmented)
    if not required <= declared: raise ValueError("Declare branch evidence/manifest and requested full reference outputs")
    if any(not v.is_relative_to(destination) and v != augmented for v in declared):
        raise ValueError("Declared output outside branch destinations")
    inputs = set(_consumed_branch_inputs(root, parsed)) | {_resolve(root, v) for v in plan["input_files"]}
    if not inputs or any(v in inputs for v in declared) or destination in inputs:
        raise ValueError("Outputs cannot overwrite consumed inputs")
    before = {p.as_posix(): digest(p) for p in sorted(inputs)}
    source = _source_closure(root, (BRANCH_SCRIPT, BRANCH_MODULE))
    command = [str(python), str(_resolve(root, BRANCH_SCRIPT)), *plan["arguments"]]
    completed = subprocess.run(command, cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace")
    found = set(destination.rglob("*")) if destination.exists() else set()
    if augmented is not None and augmented.exists(): found.add(augmented)
    outputs = {p.as_posix(): digest(p) for p in sorted(found) if p.is_file()}
    unchanged = all(Path(path).is_file() and digest(path) == sha for path, sha in before.items())
    unchanged_source = all(digest(path) == sha for path, sha in source.items())
    versions = {}
    for name in ("numpy", "scipy", "pandas", "pyarrow", "torch", "h5py"):
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = None
    receipt = {"schema_version": "flowcompat-supplemental-tool-receipt-2.0", "tool_id": "branch_mutation",
               "plan_path": str(Path(plan_path).resolve()), "plan_sha256": digest(plan_path), "command": command,
               "returncode": completed.returncode, "input_files": before, "output_files": outputs, "code_files": source,
               "python_version": sys.version.split()[0],
               "frozen_originals": assert_frozen_original(), "package_versions": versions,
               "inputs_unchanged": unchanged, "source_unchanged": unchanged_source,
               "complete_outputs": all(p.as_posix() in outputs for p in declared),
               "stdout_tail": completed.stdout[-4000:], "stderr_tail": completed.stderr[-4000:]}
    write_json(receipt_path, receipt)
    if completed.returncode or not unchanged or not unchanged_source or not receipt["complete_outputs"]:
        raise ValueError("Third tool failed; preserved receipt records actual execution")
    return receipt


def check_receipt(path):
    assert_frozen_original()
    receipt = read_json(path)
    if receipt.get("tool_id") != "branch_mutation": return original._check_receipt(path)
    if receipt.get("schema_version") != "flowcompat-supplemental-tool-receipt-2.0": raise ValueError("Versioned third-tool receipt required")
    if receipt.get("returncode") != 0 or any(receipt.get(k) is not True for k in ("inputs_unchanged", "source_unchanged", "complete_outputs")):
        raise ValueError("Unsuccessful third-tool execution")
    if digest(receipt["plan_path"]) != receipt["plan_sha256"]: raise ValueError("Third-tool plan changed")
    plan = read_json(receipt["plan_path"]); parsed = _branch_arguments(plan)
    if receipt["command"][2:] != plan["arguments"] or not Path(receipt["command"][1]).as_posix().endswith("/" + BRANCH_SCRIPT):
        raise ValueError("Third-tool executed argv mismatch")
    if receipt["frozen_originals"] != FROZEN_ORIGINALS: raise ValueError("Third-tool original contract binding changed")
    for group in ("input_files", "output_files", "code_files"):
        if not receipt.get(group): raise ValueError("Missing complete third-tool artifact bindings")
        for file, sha in receipt[group].items():
            if digest(file) != sha: raise ValueError("Third-tool artifact changed: " + file)
    root = Path(receipt["command"][1]).resolve().parents[1]
    if _source_closure(root, (BRANCH_SCRIPT, BRANCH_MODULE)) != receipt["code_files"]:
        raise ValueError("Third-tool source closure incomplete or changed")
    required = {_resolve(root, v).as_posix() for v in _consumed_branch_inputs(root, parsed)}
    if not required <= set(receipt["input_files"]): raise ValueError("Third-tool consumed input binding incomplete")
    return receipt


def check_extra_receipt(path):
    """Read an actual live selfreceipt; never reconstruct or modify it."""
    assert_frozen_original(); value = read_json(path)
    if value.get("schema_version") != "multi-depth-live-execution-receipt-1.0" or value.get("tool_id") != "multi_depth_mutation":
        raise ValueError("Only the registered multi-depth live execution receipt is accepted")
    if value.get("returncode") != 0 or value.get("error") is not None or value.get("inputs_and_code_unchanged") is not True or value.get("complete_outputs") is not True:
        raise ValueError("Successful live extra-tool execution required")
    command = value.get("command", [])
    if len(command) < 10 or not all(isinstance(v, str) for v in command): raise ValueError("Literal extra-tool argv required")
    script = Path(command[1]).resolve()
    if not script.as_posix().endswith("/" + EXTRA_SCRIPT): raise ValueError("Unregistered extra-tool script")
    root = script.parents[1]; args = command[2:]
    if len(args) % 2 or len(set(args[::2])) != len(args[::2]) or set(args[::2]) - EXTRA_FLAGS:
        raise ValueError("Unregistered or duplicate extra-tool arguments")
    parsed = dict(zip(args[::2], args[1::2]))
    if not {"--dataset", "--campaign", "--reference", "--output"} <= set(parsed): raise ValueError("Explicit extra-tool inputs/output required")
    output = _resolve(root, parsed["--output"])
    expected_receipt = _resolve(root, parsed["--receipt"]) if "--receipt" in parsed else output / "execution_receipt.json"
    if Path(path).resolve() != expected_receipt: raise ValueError("Live receipt location differs from captured argv")
    normalized = copy.deepcopy(value)
    for group in ("input_files", "code_files", "output_files"):
        if not isinstance(value.get(group), dict) or not value[group]: raise ValueError("Complete extra-tool artifact bindings required")
        normalized[group] = {Path(file).resolve().as_posix(): sha for file, sha in value[group].items()}
        if len(normalized[group]) != len(value[group]): raise ValueError("Ambiguous extra-tool path aliases")
        for file, sha in normalized[group].items():
            if digest(file) != sha: raise ValueError("Live extra-tool artifact changed: " + file)
    eager = _source_closure(root, (EXTRA_SCRIPT, EXTRA_MODULE), eager_only=True)
    if any(normalized["code_files"].get(file) != sha for file, sha in eager.items()):
        raise ValueError("Live extra-tool eager source closure incomplete")
    consumed = {_resolve(root, v).as_posix() for v in _consumed_branch_inputs(root, parsed)}
    if not consumed <= set(normalized["input_files"]): raise ValueError("Live extra-tool consumed input binding incomplete")
    required = {output / "evidence.json", output / "manifest.json", output / "whole_window_effects.parquet", output / "fitted_trends.json", output / "feature_catalog.json", output / "regional_direction_coverage.json"}
    if not {p.as_posix() for p in required} <= set(normalized["output_files"]): raise ValueError("Live extra-tool outputs incomplete")
    manifest = read_json(output / "manifest.json")
    if manifest.get("schema_version") != "multi-depth-mutation-mining-1.0": raise ValueError("Registered extra-tool manifest required")
    reference = _resolve(root, parsed["--reference"])
    if manifest.get("reference_sha256") != digest(reference): raise ValueError("Extra-tool manifest/reference argv mismatch")
    try:
        checks = {}
        if "--depths" in parsed:
            depths = [int(v) for v in parsed["--depths"].split(",")]
            if not depths or len(set(depths)) != len(depths) or any(v < 2 for v in depths): raise ValueError("Invalid ancestry depths")
            checks["depths"] = sorted(depths)
        if "--seed" in parsed: checks["seed"] = int(parsed["--seed"])
        if "--region-width" in parsed:
            width = float(parsed["--region-width"])
            if not math.isfinite(width) or width <= 0: raise ValueError("Positive region width required")
            checks["region_width_A"] = width
        if "--nuisance" in parsed:
            if parsed["--nuisance"] not in ("chemistry", "chemistry_pose"): raise ValueError("Unknown nuisance convention")
            checks["nuisance"] = parsed["--nuisance"]
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid literal extra-tool parameters") from exc
    if any(manifest.get(key) != val for key, val in checks.items()): raise ValueError("Extra-tool manifest differs from captured argv")
    for file, record in manifest["files"].items():
        target = (output / file).resolve()
        if not target.is_relative_to(output) or normalized["output_files"].get(target.as_posix()) != record["sha256"]:
            raise ValueError("Extra-tool manifest/output binding mismatch")
    for record in manifest.get("sources", []):
        source = (_resolve(root, parsed["--dataset"]) / record["path"]).resolve()
        if normalized["input_files"].get(source.as_posix()) != record["sha256"]:
            raise ValueError("Extra-tool manifest/source input binding mismatch")
    return normalized


def _bound_output(receipts, path, tool):
    resolved = Path(path).resolve().as_posix()
    if not any(r["tool_id"] == tool and r["output_files"].get(resolved) == digest(path) for r in receipts):
        raise ValueError("Evidence is not a bound output of its actual registered tool")


def export_request(evidence, branch_evidence, registry, receipts, output, role, *, analyst=None, extra_evidence=None, extra_receipt=None, root=PROJECT):
    assert_frozen_original()
    if role not in ("Analyst", "Designer"): raise ValueError("Unknown role")
    checked = [check_receipt(v) for v in receipts]
    if len(checked) != 3 or {r["tool_id"] for r in checked} != {"selection_innovation", "flow_compatibility", "branch_mutation"}:
        raise ValueError("Exactly the two frozen receipts and a new actual third-tool receipt required")
    _bound_output(checked, evidence, "selection_innovation"); _bound_output(checked, branch_evidence, "branch_mutation")
    first, branch = read_json(evidence), read_json(branch_evidence)
    window = original._packet_window(first)
    if window != original._packet_window(branch): raise ValueError("Supplemental evidence learned support mismatch")
    branch_receipt = next(r for r in checked if r["tool_id"] == "branch_mutation")
    reference_argument = branch_receipt["command"][branch_receipt["command"].index("--reference") + 1]
    tool_root = Path(branch_receipt["command"][1]).resolve().parents[1]
    if first.get("reference_sha256") != branch_receipt["input_files"].get(_resolve(tool_root, reference_argument).as_posix()):
        raise ValueError("Supplemental reference identity mismatch")
    if first.get("independent_batches") != branch.get("independent_batches"):
        raise ValueError("Supplemental independent-batch support mismatch")
    combined = {"schema_version": "flowcompat-supplemental-evidence-2.0", "window": window,
                "evidence_items": first["evidence_items"] + branch["evidence_items"],
                "sources": {"selection_innovation": first, "branch_mutation": branch}}
    original._evidence_ids(combined)
    branch_folder = Path(branch_evidence).parent
    effects = branch_folder / "whole_window_effects.parquet"
    if effects.is_file():
        _bound_output(checked, effects, "branch_mutation")
        # A correlation-only shortlist would not let the Agent obey the Skill's
        # raw/adjusted and absolute-effect comparison. Supply only companion
        # moments for its selected features, not the full feature table.
        table = pd.read_parquet(effects)
        features = {row["feature"] for row in branch["evidence_items"]}
        companion = table[table.feature.isin(features)].sort_values(["feature", "metric"])
        existing = original._evidence_ids(combined)
        rows = [row for row in companion.to_dict("records") if row["id"] not in existing]
        combined["evidence_items"].extend(rows)
        combined["branch_companion_moments"] = {
            "selected_features": sorted(features), "added_rows": len(rows),
            "source_sha256": digest(effects),
            "interpretation": "Raw/adjusted covariance and correlation on the same salient features; shortlist does not replace the full multiplicity family"}
    fitted = branch_folder / "fitted_trends.json"
    if fitted.is_file():
        _bound_output(checked, fitted, "branch_mutation")
        functions = read_json(fitted)
        wanted = {v["metric"] + "/" + v["feature"] for v in combined["evidence_items"] if v["id"].startswith("branch_mutation/")}
        combined["branch_functions"] = {k: functions[k] for k in sorted(wanted) if k in functions}
    for name, key in (("feature_catalog.json", "branch_feature_catalog"), ("teacher_field_summary.json", "branch_teacher_field_summary")):
        path = branch_folder / name
        if path.is_file():
            _bound_output(checked, path, "branch_mutation"); combined[key] = read_json(path)
    if bool(extra_evidence) != bool(extra_receipt): raise ValueError("Extra evidence and actual live receipt must be supplied together")
    if extra_evidence:
        extra = check_extra_receipt(extra_receipt); additional = read_json(extra_evidence)
        _bound_output([extra], extra_evidence, "multi_depth_mutation")
        if original._packet_window(additional) != window or additional.get("independent_batches") != first.get("independent_batches"):
            raise ValueError("Extra-tool learned window or independent-batch support mismatch")
        captured = dict(zip(extra["command"][2::2], extra["command"][3::2])); extra_root = Path(extra["command"][1]).resolve().parents[1]
        if extra["input_files"].get(_resolve(extra_root, captured["--reference"]).as_posix()) != first.get("reference_sha256"):
            raise ValueError("Extra-tool frozen input reference mismatch")
        combined["sources"]["multi_depth_mutation"] = additional
        combined["evidence_items"].extend(additional["evidence_items"])
        folder = Path(extra_evidence).parent
        table_path = folder / "whole_window_effects.parquet"; _bound_output([extra], table_path, "multi_depth_mutation")
        table = pd.read_parquet(table_path)
        pairs = {(int(row["depth"]), row["feature"]) for row in additional["evidence_items"]}
        regions = set()
        for depth, feature in list(pairs):
            if feature.startswith("region_") and "_internal_displacement_" in feature:
                region = int(feature.split("_")[1]); regions.add(region)
                pairs.update((depth, f"region_{region:02d}_internal_displacement_{axis}") for axis in "xyz")
        selected = table[[ (int(row.depth), row.feature) in pairs for row in table.itertuples() ]].sort_values(["depth", "feature", "metric"])
        known = original._evidence_ids(combined)
        companion = [row for row in selected.to_dict("records") if row["id"] not in known]
        combined["evidence_items"].extend(companion)
        combined["multi_depth_companion_moments"] = {"added_rows": len(companion), "source_sha256": digest(table_path),
            "interpretation": "Companion raw/adjusted absolute covariance and correlation; selected regional XYZ completed at each cited depth. All tests retain original shared BH family."}
        catalog_path = folder / "feature_catalog.json"; _bound_output([extra], catalog_path, "multi_depth_mutation")
        catalog = read_json(catalog_path)
        combined["multi_depth_feature_catalog"] = {**catalog, "features": [feature for feature in catalog["features"] if any(feature == p[1] for p in pairs)]}
        fits_path = folder / "fitted_trends.json"; _bound_output([extra], fits_path, "multi_depth_mutation")
        fits = read_json(fits_path); selected_ids = {row["id"] for row in selected.to_dict("records")}
        combined["multi_depth_functions"] = {"schema_version": fits["schema_version"], "statistics_unit": fits.get("statistics_unit"),
            "derivative_semantics": fits.get("derivative_semantics"), "depths": {}}
        for depth, functions in fits["depths"].items():
            chosen = {key: item for key, item in functions["functions"].items() if f"multi_depth/{depth}/{key}" in selected_ids}
            if chosen: combined["multi_depth_functions"]["depths"][depth] = {"times": functions["times"], "first_missing_times": functions["first_missing_times"], "functions": chosen}
        directions_path = folder / "regional_direction_coverage.json"; _bound_output([extra], directions_path, "multi_depth_mutation")
        directions = read_json(directions_path)
        combined["multi_depth_regional_directions"] = {"units": directions.get("units"), "region_landmarks_A": directions["region_landmarks_A"],
            "rows": [row for row in directions["rows"] if int(row["region"]) in regions],
            "interpretation": "Whole-window mean-direction support differs from stepwise vector consistency. Pairwise batch comparisons are dependent descriptive pairs, not independent observations."}
        original._evidence_ids(combined)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    bundle = output / "flowcompat-supplemental.evidence.json"
    if bundle.exists() and read_json(bundle) != combined: raise FileExistsError("Combined evidence packet is immutable")
    write_json(bundle, combined)
    skills = [Path(root) / "skills" / role.lower() / "SKILL.md", Path(root) / "skills/flow-compatibility/SKILL.md",
              Path(root) / "skills/flow-compatibility-supplemental/SKILL.md"]
    text = "\n\n".join(path.read_text(encoding="utf-8").rstrip() for path in skills) + "\n"
    bindings = {"evidence_path": str(bundle.resolve()), "registry_path": str(Path(registry).resolve()), "registry_sha256": digest(registry),
                "source_evidence": {str(Path(v).resolve()): digest(v) for v in (evidence, branch_evidence)},
                "skill_files": {str(v.resolve()): digest(v) for v in skills},
                "tool_receipts": {str(Path(v).resolve()): digest(v) for v in receipts},
                "frozen_originals": assert_frozen_original(),
                "workflow_files": {str(Path(__file__).resolve()): digest(__file__),
                                   str((Path(root) / "scripts/flowcompat_supplemental_agent.py").resolve()): digest(Path(root) / "scripts/flowcompat_supplemental_agent.py")}}
    bindings["extra_receipts"] = {}
    if extra_evidence:
        bindings["source_evidence"][str(Path(extra_evidence).resolve())] = digest(extra_evidence)
        bindings["extra_receipts"][str(Path(extra_receipt).resolve())] = digest(extra_receipt)
    context = read_json(registry).get("context_files", {})
    bindings["context_files"] = {}
    for file, sha in context.items():
        path = _resolve(root, file)
        if digest(path) != sha: raise ValueError("Registry strategy context changed")
        bindings["context_files"][str(path)] = sha
    if role == "Designer":
        if analyst is None: raise ValueError("Validated supplemental Analyst response required")
        analyst = Path(analyst)
        validate_response(analyst.parent / "Analyst.flowcompat-supplemental.request.json", analyst)
        bindings.update(analyst_path=str(analyst.resolve()), analyst_response_sha256=digest(analyst))
    request = {"schema_version": VERSION, "role": role, "window": window, "evidence_sha256": digest(bundle),
               "instruction_sha256": original._text_sha(text), "system_instruction": text, "bindings": bindings,
               "response_schema": ANALYST if role == "Analyst" else DESIGNER}
    path = output / (role + ".flowcompat-supplemental.request.json")
    if path.exists(): raise FileExistsError("Supplemental request is immutable")
    write_json(path, request)
    (output / (role + ".flowcompat-supplemental.instructions.md")).write_text(text, encoding="utf-8")
    return path


def payload(request):
    assert_frozen_original()
    if request.get("schema_version") != VERSION: raise ValueError("Supplemental workflow version required")
    bindings = request["bindings"]
    if bindings["frozen_originals"] != FROZEN_ORIGINALS: raise ValueError("Frozen original binding mismatch")
    for group in ("source_evidence", "skill_files", "workflow_files", "tool_receipts", "extra_receipts", "context_files"):
        for file, sha in bindings[group].items():
            if digest(file) != sha: raise ValueError("Supplemental bound artifact changed: " + file)
    if digest(bindings["evidence_path"]) != request["evidence_sha256"] or digest(bindings["registry_path"]) != bindings["registry_sha256"]:
        raise ValueError("Supplemental evidence or registry changed")
    if original._text_sha(request["system_instruction"]) != request["instruction_sha256"]: raise ValueError("Literal supplemental instruction changed")
    checked = [check_receipt(v) for v in bindings["tool_receipts"]]
    combined = read_json(bindings["evidence_path"])
    if original._packet_window(combined) != request["window"]: raise ValueError("Supplemental support changed")
    result = {"evidence": combined, "registry": read_json(bindings["registry_path"]), "tool_receipts": checked,
              "request_bindings": {"instruction_sha256": request["instruction_sha256"], "evidence_sha256": request["evidence_sha256"],
                                   "source_evidence_sha256": sorted(bindings["source_evidence"].values()),
                                   "extra_tool_receipt_sha256": sorted(bindings["extra_receipts"].values()),
                                   "tool_receipt_sha256": sorted(bindings["tool_receipts"].values())}}
    result["extra_tool_receipts"] = [check_extra_receipt(path) for path in bindings["extra_receipts"]]
    if bindings["context_files"]:
        result["strategy_context"] = {}
        for file, sha in bindings["context_files"].items():
            path = Path(file)
            if path.suffix.lower() not in (".md", ".txt", ".json") or path.stat().st_size > 256 * 1024:
                raise ValueError("Bounded textual strategy context required")
            result["strategy_context"][file] = {"sha256": sha, "text": path.read_text(encoding="utf-8")}
    if "analyst_path" in bindings:
        if digest(bindings["analyst_path"]) != bindings["analyst_response_sha256"]: raise ValueError("Supplemental Analyst response changed")
        result.update(analyst_document=read_json(bindings["analyst_path"]), analyst_response_sha256=bindings["analyst_response_sha256"])
    return result


def validate_response(request_path, response_path):
    request, response = read_json(request_path), read_json(response_path)
    jsonschema.validate(response, ANALYST if request["role"] == "Analyst" else DESIGNER)
    values = payload(request)
    expected = {"request_sha256": digest(request_path), "instruction_sha256": request["instruction_sha256"],
                "evidence_sha256": request["evidence_sha256"], "window": request["window"]}
    if any(response[k] != v for k, v in expected.items()): raise ValueError("Supplemental response provenance/support mismatch")
    for key, group in (("tool_receipt_sha256", "tool_receipts"), ("extra_tool_receipt_sha256", "extra_receipts"), ("source_evidence_sha256", "source_evidence")):
        if sorted(response[key]) != sorted(request["bindings"][group].values()): raise ValueError("Supplemental response did not bind all executed inputs")
    entries = response["observations"] + response["hypotheses"] + response["counterevidence"] if request["role"] == "Analyst" else [response]
    known = original._evidence_ids(values["evidence"])
    if any(set(v["evidence_ids"]) - known for v in entries): raise ValueError("Unprovided supplemental evidence citation")
    if request["role"] == "Analyst" and not any(any(e.startswith("branch_mutation/") for e in v["evidence_ids"]) for v in entries):
        raise ValueError("Supplemental Analyst must interpret new branch evidence")
    if request["bindings"]["extra_receipts"] and not any(any(e.startswith("multi_depth/") for e in v["evidence_ids"]) for v in entries):
        raise ValueError("Supplied multi-depth evidence must be considered in the role response")
    if request["role"] == "Designer":
        if response["analyst_response_sha256"] != values["analyst_response_sha256"]: raise ValueError("Supplemental Designer did not consume Analyst")
        if not any(v.startswith("branch_mutation/") for v in response["evidence_ids"]): raise ValueError("Supplemental Designer must cite newly measured branch evidence")
        updates = response["updates"]
        if any(isinstance(v, bool) or not math.isfinite(v) for v in updates.values()): raise ValueError("Finite registered updates required")
        if response["decision"] == "defer":
            if updates or response["base_program_id"] is not None or response["formula_id"] is not None: raise ValueError("Deferred supplemental design cannot compile")
        else:
            registry = values["registry"]
            if response["base_program_id"] not in registry["programs"] or response["formula_id"] not in registry["formulas"]: raise ValueError("Unknown supplemental registered program/formula")
            formula = registry["formulas"][response["formula_id"]]
            if formula.get("reference_kind") not in ("incumbent", "branch_mutation"): raise ValueError("Explicit registered reference kind required")
            if formula.get("reward_view") not in ("endpoint_pointcloud", "endpoint_innovation", "endpoint_branch_mixture"):
                raise ValueError("Unimplemented supplemental formula")
            if set(updates) - set(formula["allowed_updates"]): raise ValueError("Inactive supplemental parameter")
            if any(k.startswith("branch_mixture.") for k in updates) and formula["reward_view"] != "endpoint_branch_mixture":
                raise ValueError("Branch mixture update requires its independent registered formula")
            if response["decision"] == "retain" and updates or response["decision"] == "trial" and not updates: raise ValueError("Decision/update mismatch")
            if len(updates) > 1: raise ValueError("Sequential single-axis supplemental trial required")
    return response


def import_response(request_path, response_path, output=None):
    data = validate_response(request_path, response_path); request = read_json(request_path)
    folder = Path(output) if output else Path(request_path).parent
    path = folder / (request["role"] + ".flowcompat-supplemental.response.json")
    source_sha = digest(response_path)
    if path.exists() and path.resolve() != Path(response_path).resolve() and read_json(path) != data: raise FileExistsError("Supplemental imported response is immutable")
    write_json(path, data)
    write_json(folder / (request["role"] + ".flowcompat-supplemental.validation.json"), {
        "schema_version": VERSION, "passed": True, "schema_valid": True, "grounding_valid": True,
        "request_sha256": digest(request_path), "response_sha256": digest(path), "source_response_sha256": source_sha,
        "instruction_sha256": request["instruction_sha256"], "evidence_sha256": request["evidence_sha256"],
        "source_evidence_sha256": data["source_evidence_sha256"], "tool_receipt_sha256": data["tool_receipt_sha256"],
        "extra_tool_receipt_sha256": data["extra_tool_receipt_sha256"], "efficacy_assessed": False})
    return data


def compile_design(request_path, response_path, output, round_number, *, root=PROJECT):
    data = validate_response(request_path, response_path)
    if data["agent"] != "Designer": raise ValueError("Supplemental Designer response required")
    if isinstance(round_number, bool) or not isinstance(round_number, int) or not 1 <= round_number <= 30: raise ValueError("Round outside registered budget")
    if data["decision"] == "defer": return None
    request = read_json(request_path); values = payload(request); registry = values["registry"]
    entry, formula = registry["programs"][data["base_program_id"]], registry["formulas"][data["formula_id"]]
    source, reference = _resolve(root, entry["program_path"]), _resolve(root, entry["reference_path"])
    if digest(source) != entry["program_sha256"] or digest(reference) != entry["reference_sha256"]: raise ValueError("Frozen supplemental program/reference changed")
    ref = json.loads(gzip.decompress(reference.read_bytes())); program = copy.deepcopy(read_json(source))
    if program["window"] != data["window"] or ref["window"] != data["window"]: raise ValueError("Supplemental design/reference scope mismatch")
    if formula["reference_kind"] == "branch_mutation":
        _bound_output(values["tool_receipts"], reference, "branch_mutation")
        if "branch_mutation" not in ref or ref["branch_mutation"]["scope"] != data["window"]: raise ValueError("Measured branch reference required")
    if program.get("derivative_path") != "flowr_endpoint_vjp" or program.get("affinity_head_gradient") is not False or program.get("additional_per_step_affinity_calls") != 0: raise ValueError("Actual conditional endpoint VJP without affinity gradient required")
    if formula["reward_view"] not in ("endpoint_pointcloud", "endpoint_innovation", "endpoint_branch_mixture"): raise ValueError("Unimplemented supplemental formula")
    if data["decision"] == "retain" and program["reward_view"] != formula["reward_view"]: raise ValueError("Retain cannot change registered formula")
    if formula["reward_view"] == "endpoint_pointcloud" and any(k.startswith("innovation.") for k in data["updates"]): raise ValueError("Innovation update requires its actual formula")
    if formula["reward_view"] == "endpoint_innovation" and formula["reference_kind"] != "branch_mutation": raise ValueError("Supplemental innovation must use measured branch reference")
    if formula["reward_view"] == "endpoint_branch_mixture" and formula["reference_kind"] != "branch_mutation": raise ValueError("Branch mixture must use its measured conditional mutation reference")
    if formula["reward_view"] == "endpoint_branch_mixture" and any(k.startswith("innovation.") for k in data["updates"]):
        raise ValueError("Legacy innovation parameter cannot silently control a branch mixture")
    modules = formula.get("code_files", {})
    if not modules: raise ValueError("Bound numerical modules required")
    for path, sha in modules.items():
        if digest(_resolve(root, path)) != sha: raise ValueError("Supplemental numerical module changed")
    program.update(reward_view=formula["reward_view"], reference_sha256=digest(reference), round=round_number,
                   agent_request_sha256=digest(request_path))
    for key, value in data["updates"].items():
        if "." in key:
            parent, field = key.split("."); program.setdefault(parent, {})[field] = value
        else: program[key] = value
    if program["reward_view"] == "endpoint_branch_mixture":
        parameters = program.get("branch_mixture", {})
        if parameters.get("virtual_mass", 0) <= 0 and any(
            key == "branch_mixture.direction_sign" and value != 0 for key, value in data["updates"].items()):
            raise ValueError("Branch direction update inactive without a virtual mixture mass")
    program["flowcompat_provenance"] = {"workflow_version": VERSION, "request_sha256": digest(request_path),
        "designer_response_sha256": digest(response_path), "analyst_response_sha256": data["analyst_response_sha256"],
        "instruction_sha256": request["instruction_sha256"], "evidence_sha256": request["evidence_sha256"],
        "source_evidence_sha256": data["source_evidence_sha256"], "tool_receipt_sha256": data["tool_receipt_sha256"],
        "extra_tool_receipt_sha256": data["extra_tool_receipt_sha256"],
        "registry_sha256": request["bindings"]["registry_sha256"], "formula_id": data["formula_id"],
        "reference_kind": formula["reference_kind"], "parameter_updates": data["updates"], "code_files": modules,
        "context_files": request["bindings"]["context_files"],
        "frozen_originals": assert_frozen_original(), "implementation_gate_required": True}
    write_json(output, program)
    return program


def verify_execution_response(program, execution, output):
    assert_frozen_original()
    p = read_json(program)
    if p.get("flowcompat_provenance", {}).get("workflow_version") != VERSION: raise ValueError("Supplemental compiled program required")
    if p["reward_view"] == "endpoint_branch_mixture":
        report = read_json(execution); provenance = p["flowcompat_provenance"]
        if report.get("program_sha256") != digest(program) or report.get("request_sha256") != provenance["request_sha256"] or report.get("code_files") != provenance["code_files"]:
            raise ValueError("Actual branch program/request/numerical module binding mismatch")
        required = ("actual_flowr_model", "actual_endpoint_vjp", "no_resampling", "no_head_gradient", "no_extra_production_forward", "window_matches", "initial_state_pair_matches")
        if any(report.get("execution_checks", {}).get(k) is not True for k in required): raise ValueError("Actual branch inference contract incomplete")
        control = report.get("no_op_control", {})
        if control.get("exact_baseline_arithmetic") is not True or control.get("max_gradient_relative_change") != 0:
            raise ValueError("Exact branch zero-mass baseline arithmetic required")
        diagnostics = report.get("diagnostics", {})
        def finite(key):
            value = diagnostics.get(key)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError("Measured finite branch mechanism diagnostic required: " + key)
            return value
        parameters = p.get("branch_mixture", {})
        field_active = parameters.get("virtual_mass", 0) > 0 and parameters.get("direction_sign", 1) != 0
        region_active = parameters.get("region_weight_mix", 0) > 0
        spatial_keys = ("branch_gradient_relative_change", "paired_window_coordinate_rms_A")
        virtual_keys = ("branch_virtual_mass_mean", "branch_virtual_teacher_shift_rms_A")
        if field_active or region_active:
            if any(finite(key) <= 1e-8 for key in spatial_keys): raise ValueError("Actual new branch reward/trajectory did not respond")
            if field_active:
                if any(finite(key) <= 1e-8 for key in virtual_keys): raise ValueError("Actual virtual branch field did not respond")
                mass = parameters["virtual_mass"]
                if finite("branch_virtual_mass_mean") > mass + max(1e-8, mass * 1e-6):
                    raise ValueError("Actual virtual-teacher mass exceeded its registered budget")
            elif any(finite(key) != 0 for key in virtual_keys):
                raise ValueError("Inactive virtual-field diagnostics must be zero")
            if region_active and finite("branch_region_weight_rms_change") <= 1e-8:
                raise ValueError("Actual regional branch weights did not respond")
        elif any(finite(key) != 0 for key in spatial_keys + virtual_keys):
            raise ValueError("Empty branch mechanisms changed baseline arithmetic")
        result = {"schema_version": "flowcompat-implementation-gate-1.0", "implementation_passed": True,
                  "program_sha256": digest(program), "execution_sha256": digest(execution), "request_sha256": provenance["request_sha256"],
                  "parameter_updates": provenance["parameter_updates"], "diagnostics": diagnostics, "efficacy_assessed": False,
                  "interpretation": "Measured branch-mixture response or exact null control; paired efficacy remains separate"}
    else:
        result = original.verify_execution_response(program, execution, output)
    result.update(workflow_version=VERSION, frozen_originals=assert_frozen_original())
    write_json(output, result)
    return result


def call_api(request_path, *, client=None):
    base, model, key = (os.environ.get("EVOMOLSTEER_BASE_URL"), os.environ.get("EVOMOLSTEER_MODEL"), os.environ.get("EVOMOLSTEER_API_KEY"))
    if not all((base, model, key)): raise ValueError("Explicit API base, model and key required")
    request = read_json(request_path); values = payload(request); values["request_sha256"] = digest(request_path)
    body = {"model": model, "temperature": 0, "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": request["system_instruction"] + "\nReturn JSON matching supplied schema:\n" + json.dumps(request["response_schema"])},
                         {"role": "user", "content": json.dumps(values, ensure_ascii=False, allow_nan=False)}]}
    owned = client is None; client = client or httpx.Client(timeout=180)
    try:
        result = client.post(base.rstrip("/") + "/chat/completions", headers={"Authorization": "Bearer " + key}, json=body)
        result.raise_for_status(); data = json.loads(result.json()["choices"][0]["message"]["content"])
    finally:
        if owned: client.close()
    raw = Path(request_path).parent / (request["role"] + ".flowcompat-supplemental.raw_api.json")
    write_json(raw, data)
    return import_response(request_path, raw)
