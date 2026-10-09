"""Fail-closed integrity audit of the frozen FLOWR confirmation panel.

This tool reads retained evidence and local Git objects only. It does not run a
model, tune a reward, or adopt a candidate. A passing integrity report permits a
separate efficacy decision; it is not evidence of improved affinity.
"""
import argparse
import ast
import csv
import hashlib
import json
import math
import re
import subprocess
from datetime import datetime
from pathlib import Path, PurePosixPath


VERSION = "flowcompat-confirmation-integrity-1.0"
CONTROL_ROUNDS = (25, 27, 29)
CANDIDATE_ROUNDS = (26, 28, 30)
BATCHES = tuple(range(49, 55))
SLOTS = tuple(range(50))
METADATA_KEYS = {"round", "program_id", "derivation"}
SCORE_COLUMNS = ("pic50_on_rescore", "pic50_off_rescore", "pic50_on_upstream",
                 "pic50_off_upstream", "gap_rescore")
UPSTREAM_FILES = {"flowr/models/fm_pocket.py", "flowr/models/integrator.py",
                  "flowr/models/pocket.py", "flowr/models/pocket_util.py"}
EXECUTION_CHECKS = ("actual_flowr_model", "actual_endpoint_vjp", "no_resampling",
                    "no_head_gradient", "no_extra_production_forward",
                    "window_matches", "initial_state_pair_matches")
WORKFLOWS = {"flowcompat-agent-1.0", "flowcompat-supplemental-agent-2.0",
             "flowcompat-regional-agent-1.0"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        return json.load(handle, parse_constant=lambda v: (_ for _ in ()).throw(
            ValueError("Nonfinite JSON constant: " + v)))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def semantic_program(program):
    """Ignore only the campaign's declared top-level administrative metadata."""
    return {key: value for key, value in program.items() if key not in METADATA_KEYS}


def recorded_json_sha(data):
    """Reproduce the explicit Linux io.write_json serialization of attestations."""
    body = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def verify_generation_config_binding(config):
    """Verify parsed fields without pretending to reconstruct controller bytes.

    The controller's unsorted serialization and the metadata binder's canonical
    serialization differ. The raw digest is retained as a byte identity record;
    only the separately attested canonical digest is reproducible from fields.
    """
    bound = config["upstream_source_attestation"]
    raw_sha = bound.get("generation_config_sha256_before_binding")
    require(valid_sha(raw_sha), "Missing or malformed original raw generation configuration SHA")
    canonical_sha = bound.get("generation_config_canonical_sha256_before_binding")
    require(valid_sha(canonical_sha),
            "Legacy upstream binding lacks a canonical pre-binding configuration SHA: evidence insufficient; "
            "the recorded raw SHA cannot establish whether parsed numerical fields changed")
    original = {key: value for key, value in config.items() if key != "upstream_source_attestation"}
    require(canonical_sha == recorded_json_sha(original),
            "Canonical pre-binding generation configuration fields mismatch")
    return {"raw_generation_config_sha256": raw_sha,
            "canonical_generation_config_sha256": canonical_sha,
            "canonical_fields_match": True, "raw_original_bytes_reconstructed": False}


def loading_upper_limit(config):
    """Validate the dataloader enumeration bound, not a generated population.

    The frozen controller loads through the largest declared batch index and
    skips undeclared batches before model generation. This single varying
    argument is therefore compared through its exact batch-derived contract.
    """
    exp = config["experiment"]
    indices, size = exp["batch_indices"], exp["batch"]
    require(isinstance(indices, list) and indices and len(indices) == len(set(indices))
            and all(type(value) is int and value >= 0 for value in indices)
            and type(size) is int and size > 0, "Invalid batch declaration for loader enumeration")
    expected = (max(indices) + 1) * size
    actual = config["flowr_args"].get("sample_n_molecules_per_target")
    require(type(actual) is int and actual == expected,
            "FLOWR loader enumeration limit must equal (max(batch_indices)+1)*batch")
    return actual


class GitSources:
    """Resolve the complete static local Python import closure at an actual commit."""
    def __init__(self, repo):
        self.repo = Path(repo).resolve()
        self.trees, self.blobs = {}, {}

    def command(self, *args):
        return subprocess.check_output(["git", "-C", str(self.repo), *args], stderr=subprocess.PIPE)

    def files(self, commit):
        require(isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40,64}", commit),
                "Actual full Git commit required")
        if commit not in self.trees:
            self.trees[commit] = set(self.command("ls-tree", "-r", "--name-only", commit).decode().splitlines())
        return self.trees[commit]

    def blob(self, commit, name):
        key = commit, name
        require(name in self.files(commit), "Missing executed Git source: " + name)
        if key not in self.blobs:
            self.blobs[key] = self.command("show", commit + ":" + name)
        return self.blobs[key]

    def hashes(self, commit, names):
        return {name: hashlib.sha256(self.blob(commit, name)).hexdigest() for name in sorted(names)}

    def closure(self, commit, entrypoint):
        files, pending, seen = self.files(commit), [entrypoint], set()
        def resolve(parts):
            if not parts or parts[0] != "evomolsteer":
                return None
            base = "src/" + "/".join(parts)
            return next((p for p in (base + ".py", base + "/__init__.py") if p in files), None)
        while pending:
            name = pending.pop()
            if name in seen:
                continue
            body = self.blob(commit, name)
            seen.add(name)
            parts = PurePosixPath(name).parts
            package = list(parts[1:-1]) if name.startswith("src/evomolsteer/") else []
            if package:
                for length in range(1, len(package) + 1):
                    init = "src/" + "/".join(package[:length]) + "/__init__.py"
                    if init in files and init not in seen:
                        pending.append(init)
            for node in ast.walk(ast.parse(body.decode("utf-8-sig"), filename=name)):
                targets = []
                if isinstance(node, ast.Import):
                    targets = [alias.name.split(".") for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    prefix = package[:len(package) - node.level + 1] if node.level else []
                    module = prefix + (node.module.split(".") if node.module else [])
                    targets = [module, *[module + alias.name.split(".") for alias in node.names]]
                pending.extend(target for target in map(resolve, targets) if target and target not in seen)
        return self.hashes(commit, seen)


class ConfirmationAudit:
    def __init__(self, repo, dataset=None, configuration_dataset=None):
        self.repo = Path(repo).resolve()
        self.docs = Path(dataset).resolve() if dataset else self.repo / "docs/experiments/flowcompat30_20261009"
        self.cfg = Path(configuration_dataset).resolve() if configuration_dataset else self.repo / "configs/experiments/flowcompat30_v1"
        self.git = GitSources(self.repo)
        self.checks, self.inputs, self.rounds = [], {}, {}

    def load(self, path):
        path = Path(path)
        data = read_json(path)
        self.inputs[str(path.resolve())] = digest(path)
        return data

    def check(self, name, action):
        try:
            detail = action()
            self.checks.append({"check": name, "passed": True, "detail": detail})
            return detail
        except Exception as error:
            self.checks.append({"check": name, "passed": False,
                                "error": type(error).__name__ + ": " + str(error)})
            return None

    def retention(self, number):
        folder = self.docs / f"round{number:02d}"
        record = self.load(folder / "retention.json")
        require(record.get("status") == "complete" and record.get("campaign") == f"flowcompat_r{number:02d}",
                "Complete matching retention manifest required")
        listed = set()
        for entry in record["files"]:
            name = entry["path"]
            path = (folder / name).resolve()
            require(isinstance(name, str) and not Path(name).is_absolute() and path.is_relative_to(folder.resolve()),
                    "Retention path escapes round directory")
            require(name not in listed, "Duplicate retention file: " + name)
            listed.add(name)
            require(valid_sha(entry["sha256"]) and digest(path) == entry["sha256"],
                    "Retention checksum mismatch: " + name)
            require(path.stat().st_size == entry["bytes"], "Retention byte count mismatch: " + name)
            self.inputs[str(path)] = entry["sha256"]
        required = {"candidate_metrics.csv", "execution_report.json", "implementation_feedback.json",
                    "inference_config/config.json",
                    "inference_config/reward_program.json", "inference_config/COMPLETE.json",
                    "inference_config/EVALUATION_VIEW.json"}
        if number != 1:
            required.add("conditional_formula_audit.json")
        if number in CANDIDATE_ROUNDS or number == getattr(self, "selected", None):
            required |= {"implementation_gate_input.json", "implementation_gate.json"}
        require(required <= listed, "Required evidence is not checksum-retained: " + str(sorted(required - listed)))
        return record

    def round_data(self, number, batches, arms, expected_program):
        folder = self.docs / f"round{number:02d}"
        retention = self.retention(number)
        program_path = self.cfg / f"round{number:02d}.json"
        program = self.load(program_path)
        require(semantic_program(program) == semantic_program(expected_program),
                "Reward/controller differs from frozen program beyond round/program_id/derivation")
        require(program.get("round") == number and program.get("seed") == 42, "Round/seed program binding mismatch")
        saved = self.load(folder / "inference_config/reward_program.json")
        # Retention deliberately normalizes CRLF to LF: compare JSON semantics;
        # exact executable bytes remain bound to cfg/roundNN.json and extension.
        require(saved == program, "Retained program does not match executed config program")
        cfg = self.load(folder / "inference_config/config.json")
        execution = self.load(folder / "execution_report.json")
        feedback = self.load(folder / "implementation_feedback.json")
        formula = self.load(folder / "conditional_formula_audit.json") if number != 1 else None
        complete = self.load(folder / "inference_config/COMPLETE.json")
        view = self.load(folder / "inference_config/EVALUATION_VIEW.json")
        plan = self.load(self.docs / f"round{number:02d}_plan.json")
        exp, ext = cfg["experiment"], cfg["extension"]
        require(ext["program_sha256"] == digest(program_path) == plan["program_sha256"],
                "Executable program SHA binding mismatch")
        require(ext["reference_sha256"] == program["reference_sha256"] == plan["reference_sha256"],
                "Teacher reference SHA binding mismatch")
        ref_value = exp["reference"]
        ref_path = Path(ref_value)
        if not ref_path.is_file() and "/configs/" in ref_value.replace("\\", "/"):
            ref_path = self.repo / "configs" / ref_value.replace("\\", "/").split("/configs/", 1)[1]
        require(digest(ref_path) == program["reference_sha256"], "Actual immutable teacher reference changed")
        self.inputs[str(ref_path.resolve())] = digest(ref_path)
        require(exp["campaign"] == execution["campaign"] == retention["campaign"] == f"flowcompat_r{number:02d}",
                "Campaign binding mismatch")
        require(exp["seed"] == 42 and exp["steps"] == execution["steps"] == 100 and exp["batch"] == 50,
                "Master seed/integration steps/batch size mismatch")
        require(exp["n"] == 50 * len(batches) and exp["batch_indices"] == list(batches), "Configured batch coverage mismatch")
        require(set(exp["arms"].split(",")) == set(arms) and plan["arms"] == exp["arms"] and plan["batches"] == list(batches),
                "Configured arm/plan mismatch")
        window = program["window"]
        require(window == execution["window"] == ext["control_domain"] == ext["evidence_domain"] == plan["learned_window"],
                "Learned/control/execution windows differ")
        require([exp["window_start"], exp["window"]] == window and len(window) == 2 and
                all(finite(v) for v in window) and 0 <= window[0] < window[1] <= 1, "Invalid dynamic window")
        require(complete.get("status") == "complete" and complete.get("records") == 50 * len(batches) * len(arms),
                "Incomplete full inference record")
        require(ext["code_commit"] == execution["code_commit"] == retention["inference_commit"], "Inference commit mismatch")
        if program_path.resolve().is_relative_to(self.repo):
            git_program = program_path.resolve().relative_to(self.repo).as_posix()
        else:
            git_program = "configs/experiments/flowcompat30_v1/" + program_path.name
        require(hashlib.sha256(self.git.blob(execution["code_commit"], git_program)).hexdigest() == digest(program_path),
                "Executable program bytes differ from actually executed Git commit")
        require(valid_sha(ext["checkpoint_sha256"]), "Actual checkpoint SHA required")
        require(execution.get("no_particle_resampling") is True and execution.get("outside_window_injection") is False
                and execution.get("affinity_head_gradient") is False and execution.get("additional_production_calls_per_step") == 0,
                "Native inference/no-head-gradient contract violated")
        require(ext.get("particle_resampling") is False and ext.get("affinity_head_gradient") is False
                and ext.get("additional_production_forward_calls_per_step") == 0 and ext.get("apply_guidance") is False
                and ext.get("post_window_injection") is False and ext.get("derivative_path") == "flowr_endpoint_vjp",
                "Actual controller contract violated")
        pre = execution["preflight"]
        require(pre.get("passed") is True and pre.get("affinity_head_gradient") is False
                and pre.get("production_extra_forward_calls_per_step") == 0
                and pre.get("derivative_path") == "flowr_endpoint_vjp"
                and pre.get("frozen_self_condition_and_assignment") is True, "Actual model VJP preflight failed")
        require(pre.get("checks") and all(all(finite(row[k]) for k in ("analytic", "numerical", "epsilon", "relative_error"))
                and row["epsilon"] > 0 and row["relative_error"] >= 0 for row in pre["checks"]), "Invalid VJP preflight measurements")
        expected_keys = {f"{arm}/{batch}" for arm in arms for batch in batches}
        signatures = execution["initial_state_signatures"]
        require(set(signatures) == expected_keys and all(valid_sha(v) for v in signatures.values()),
                "Initial-state signature coverage mismatch")
        results = execution["batch_results"]
        keys = [f'{row["arm"]}/{row["batch"]}' for row in results]
        require(len(keys) == len(set(keys)) and set(keys) == expected_keys and all(row["n"] == 50 for row in results),
                "Execution batch/arm/slot-count mismatch")
        controlled_steps = sum(window[0] - 2e-6 <= step / 100 and (step + 1) / 100 <= window[1] + 2e-6 for step in range(100))
        require(all(row["controlled_steps"] == (controlled_steps if row["arm"] == "gradient" else 0) for row in results),
                "Missing eligible gradient steps or nonzero native control window")
        require(execution["sources"] == retention["raw_trajectory_hashes"], "Retired trajectory signature binding mismatch")
        sources = {f'{row["arm"]}/{row["batch"]}': row["sha256"] for row in execution["sources"]}
        require(len(sources) == len(execution["sources"]) and set(sources) == expected_keys and all(valid_sha(v) for v in sources.values()),
                "Raw trajectory SHA coverage mismatch")
        require(view.get("complete") is True and view.get("campaign") == execution["campaign"], "Missing complete evaluation snapshot view")
        view_keys = []
        for row in view["batches"]:
            match = re.fullmatch(r"([^/]+)/batch_(\d+)", row["batch_path"])
            require(match is not None, "Invalid snapshot batch path")
            key = match[1] + "/" + str(int(match[2]))
            view_keys.append(key)
            require(row["steps"] == 100 and row["initial_state_signature"] == signatures.get(key)
                    and row["source_trajectory_sha256"] == sources.get(key) and valid_sha(row["snapshot_sha256"]),
                    "Evaluation snapshot initial/source/steps mismatch")
        require(len(view_keys) == len(set(view_keys)) and set(view_keys) == expected_keys, "Snapshot view coverage mismatch")
        if number != 1:
            require(formula.get("null_exact") is True and finite(formula.get("maximum_gradient_relative_change"))
                and formula["maximum_gradient_relative_change"] >= 0 and formula.get("samples")
                and all(row.get("null_scalar_exact") is True and row.get("null_gradient_exact") is True
                        and finite(row.get("gradient_relative_change")) for row in formula["samples"]),
                "Conditional reward null audit failed")
            expected_times = [step / 100 for step in range(100) if window[0] - 2e-6 <= step / 100 < window[1] - 2e-6]
            if number == 2:
                # This immutable first-interface control predates full-grid
                # formula auditing. Its full window files and terminal outputs
                # are independently required to equal the original R26 below.
                expected_times = [window[0], sum(window) / 2, expected_times[-1]]
            times = [row.get("time") for row in formula["samples"]]
            require(len(times) == len(expected_times) and all(finite(time) and abs(time - expected) < 2e-6
                    for time, expected in zip(times, expected_times)), "Formula audit does not cover the complete learned score-time window")
        with (folder / "candidate_metrics.csv").open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        keys = []
        for row in rows:
            require(re.fullmatch(r"\d+", row["batch"]) and re.fullmatch(r"\d+", row["slot"]), "Noninteger batch/slot identifier")
            key = row["arm"], int(row["batch"]), int(row["slot"])
            keys.append(key)
            require(all(math.isfinite(float(row[column])) for column in SCORE_COLUMNS), "Nonfinite or missing attempted affinity score")
            if row.get("energy_status") == "converged":
                require(math.isfinite(float(row["mmff_relief_per_heavy"])), "Converged strain score is nonfinite")
        expected_rows = {(arm, batch, slot) for arm in arms for batch in batches for slot in SLOTS}
        require(len(keys) == len(set(keys)) and set(keys) == expected_rows, "Final attempt panel is missing, duplicated, or mismatched")
        require(retention["candidate_rows"] == len(rows), "Retained row count mismatch")
        data = {"program": program, "config": cfg, "execution": execution, "feedback": feedback,
                "formula": formula, "metrics": rows, "path": folder, "program_path": program_path}
        self.rounds[number] = data
        return {"attempts": len(rows), "batches": list(batches), "arms": list(arms), "inference_commit": execution["code_commit"]}

    def native_settings(self, data, baseline):
        left, right = data["config"], baseline["config"]
        for name in ("checkpoint", "input_dataset", "seed", "steps", "batch", "verify_passive", "export_terminal"):
            require(left["experiment"].get(name) == right["experiment"].get(name), "Native input/runtime setting differs: " + name)
        for name in ("checkpoint_sha256", "native_integrator_parameters", "native_sde", "native_coord_noise_level", "native_categorical_sampling"):
            require(left["extension"].get(name) == right["extension"].get(name), "Actual native/checkpoint setting differs: " + name)
        require(left["runtime_integrator"] == right["runtime_integrator"], "Actual runtime integrator changed")
        loading_upper_limit(left)
        loading_upper_limit(right)
        # Only a validated enumeration bound and the output path vary. Every
        # other FLOWR argument retains exact comparison, including new fields.
        clean = lambda args: {key: value for key, value in args.items()
                             if key not in {"save_dir", "sample_n_molecules_per_target"}}
        require(clean(left["flowr_args"]) == clean(right["flowr_args"]), "Underlying FLOWR arguments changed")

    def code_closure(self, number, expected=None):
        data = self.rounds[number]
        v2 = data["program"].get("generation_interface") == "flowcompat_v2"
        entry = ("scripts/generate_affinity_endpoint_flowr.py" if number == 1 else
                 "scripts/generate_flowcompat_v2_flowr.py" if v2 else "scripts/generate_flowcompat_flowr.py")
        actual = self.git.closure(data["execution"]["code_commit"], entry)
        if expected is not None:
            require(actual == expected, "Executed numerical import closure differs from frozen source")
        for name, sha in actual.items():
            require(digest(self.repo / name) == sha, "Live numerical source differs from executed/frozen source: " + name)
        provenance = data["program"].get("flowcompat_provenance")
        if provenance:
            require(provenance["code_files"] == actual, "Declared numerical closure omits/adds/mismatches actual static imports")
        ext = data["config"]["extension"]
        for key, file in (("flowcompat_module_sha256", "src/evomolsteer/generation/flowcompat_controller.py"),
                          ("v2_controller_sha256", "src/evomolsteer/generation/flowcompat_v2_controller.py")):
            if key in ext:
                require(ext[key] == actual.get(file), "Runtime controller SHA differs from executed module")
        data["code_files"] = actual
        return {"entrypoint": entry, "source_count": len(actual), "code_files": actual}

    def upstream(self, number):
        folder = self.rounds[number]["path"]
        before_path = folder / f"round{number:02d}.upstream.before.json"
        after_path = folder / f"round{number:02d}.upstream.after.json"
        cfg = self.rounds[number]["config"]
        bound = cfg.get("upstream_source_attestation", {})
        require(bound.get("schema_version") == "bound-flowr-upstream-source-attestation-1.0", "Missing config-bound upstream source attestation")
        before, after = bound["before"], bound["after"]
        before_sha, after_sha = recorded_json_sha(before), recorded_json_sha(after)
        require(bound.get("before_record_sha256") == before_sha and bound.get("after_record_sha256") == after_sha,
                "Embedded before/after original byte checksums cannot be reproduced")
        config_binding = verify_generation_config_binding(cfg)
        if before_path.exists() or after_path.exists():
            require(before_path.is_file() and after_path.is_file(), "Only one redundant upstream side record exists")
            require(self.load(before_path) == before and self.load(after_path) == after
                    and digest(before_path) == before_sha and digest(after_path) == after_sha,
                    "Redundant upstream side records differ from config-bound source evidence")
        require(before.get("schema_version") == after.get("schema_version") == "flowr-upstream-source-attestation-1.0", "Unknown upstream attestation schema")
        require(set(before["files"]) == set(after["files"]) == UPSTREAM_FILES, "Upstream attestation must cover exactly the declared four files")
        require(before["files"] == after["files"] and before["input_dataset"] == after["input_dataset"], "Upstream source changed during inference")
        require(after.get("source_unchanged") is True and after.get("reference_sha256") == before_sha, "Upstream before/after checksum binding mismatch")
        require(all(valid_sha(row["sha256"]) and isinstance(row["bytes"], int) and row["bytes"] > 0 for row in before["files"].values()), "Invalid upstream source hash/size")
        require(datetime.fromisoformat(before["captured_UTC"]) <= datetime.fromisoformat(after["captured_UTC"]), "Upstream after timestamp precedes before")
        commit = self.rounds[number]["execution"]["code_commit"]
        script_sha = self.git.hashes(commit, ["scripts/attest_flowr_sources.py"])["scripts/attest_flowr_sources.py"]
        require(before.get("script_sha256") == after.get("script_sha256") == script_sha, "Upstream attester does not match executed Git source")
        binder_sha = self.git.hashes(commit, ["scripts/bind_flowr_source_attestation.py"])["scripts/bind_flowr_source_attestation.py"]
        require(bound.get("binder_sha256") == binder_sha, "Upstream binder does not match executed Git source")
        self.rounds[number]["upstream"] = before
        return {"files": before["files"], "configuration_binding": config_binding,
                "scope": "Exactly four declared upstream files; not all dependencies"}

    def gate(self, number, null_round):
        data = self.rounds[number]
        p, provenance = data["program"], data["program"]["flowcompat_provenance"]
        workflow = provenance.get("workflow_version")
        require(workflow in WORKFLOWS, "Unsupported implementation workflow version")
        source = self.load(data["path"] / "implementation_gate_input.json")
        gate = self.load(data["path"] / "implementation_gate.json")
        require(source.get("program_sha256") == gate.get("program_sha256") == digest(data["program_path"])
                and source.get("request_sha256") == gate.get("request_sha256") == provenance["request_sha256"], "Implementation request/program binding mismatch")
        require(gate.get("workflow_version", workflow) == workflow, "Recorded gate belongs to a different workflow")
        require(gate.get("execution_sha256") == digest(data["path"] / "implementation_gate_input.json"), "Implementation gate input checksum mismatch")
        require(source.get("code_files") == data["code_files"] == provenance["code_files"], "Implementation gate numerical binding mismatch")
        require(gate.get("parameter_updates") == provenance["parameter_updates"] and gate.get("diagnostics") == source["diagnostics"], "Gate parameters/diagnostics changed")
        require(gate.get("implementation_passed") is True and gate.get("efficacy_assessed") is False
                and data["feedback"].get("implementation_passed") is True, "Candidate did not pass implementation gate")
        require(all(source.get("execution_checks", {}).get(key) is True for key in EXECUTION_CHECKS), "Implementation execution contract incomplete")
        control = source["no_op_control"]
        null = self.rounds[null_round]
        require(control.get("exact_baseline_arithmetic") is True and finite(control.get("max_gradient_relative_change")) and control["max_gradient_relative_change"] == 0
                and null["feedback"].get("null_control_parity") is True
                and null["feedback"].get("null_control_terminal_parity") is True, "Actual null control no longer holds")
        diag = source["diagnostics"]
        require(all(finite(value) for value in diag.values()), "Nonfinite implementation diagnostic")
        def positive(key):
            require(finite(diag.get(key)) and diag[key] > 1e-8, "Mechanism did not respond: " + key)
        require(diag.get("contrast_gradient_relative_change") == data["formula"]["maximum_gradient_relative_change"], "Formula audit/gate gradient diagnostic mismatch")
        if "branch_gradient_relative_change" in diag:
            require(diag["branch_gradient_relative_change"] == data["formula"]["maximum_gradient_relative_change"], "Branch formula audit/gate gradient diagnostic mismatch")
        if workflow in {"flowcompat-supplemental-agent-2.0", "flowcompat-regional-agent-1.0"} and p["reward_view"] == "endpoint_branch_mixture":
            params = p.get("branch_mixture", {})
            mass, sign, mix = params.get("virtual_mass", 0), params.get("direction_sign", 1), params.get("region_weight_mix", 0)
            require(finite(mass) and 0 <= mass <= .5 and finite(sign) and -1 <= sign <= 1 and finite(mix) and 0 <= mix <= 1, "Branch parameters outside registered bounds")
            if workflow == "flowcompat-regional-agent-1.0":
                require(mix == 0, "Fresh regional profile does not register cost changes")
            keys = ("branch_gradient_relative_change", "paired_window_coordinate_rms_A")
            virtual = ("branch_virtual_mass_mean", "branch_virtual_teacher_shift_rms_A")
            active = mass > 0 and sign != 0
            if active or mix > 0:
                for key in keys:
                    positive(key)
                if active:
                    for key in virtual:
                        positive(key)
                    require(diag["branch_virtual_mass_mean"] <= mass + max(1e-8, mass * 1e-6), "Virtual teacher budget exceeded")
                else:
                    require(all(diag.get(key) == 0 for key in virtual), "Inactive virtual field is nonzero")
                if mix > 0:
                    positive("branch_region_weight_rms_change")
            else:
                require(all(diag.get(key) == 0 for key in keys + virtual), "Empty branch mechanism is nonzero")
        else:
            changes = provenance["parameter_updates"]
            require(len(changes) <= 1, "Registered single-axis gate received multiple changes")
            if changes:
                key, value = next(iter(changes.items()))
                if value != (1. if key == "flow_control.parallel_component_scale" else 0.):
                    positive("paired_window_coordinate_rms_A")
                    needed = ["contrast_gradient_relative_change"] if key.startswith("innovation.") else ["flowcompat_gradient_adjustment_relative_rms"]
                    if key == "innovation.field_strength_A":
                        needed.append("contrast_teacher_shift_rms_A")
                    if key == "innovation.reliability_power":
                        require(p.get("innovation", {}).get("field_strength_A", 0) != 0, "Reliability change has no active field")
                    if key in {"flow_control.time_envelope_power", "flow_control.jacobian_gain_saturation", "flow_control.gradient_norm_saturation"}:
                        needed = ["flowcompat_schedule_absolute_change"]
                    if key == "native_rms_ratio":
                        needed = ["guidance_dose_relative_change"]
                    for name in needed:
                        positive(name)
        for group in ("workflow_files", "context_files", "frozen_originals"):
            for name, sha in provenance.get(group, {}).items():
                file = Path(name) if Path(name).is_absolute() else self.repo / name
                require(valid_sha(sha) and digest(file) == sha, "Bound workflow/context source changed: " + name)
                self.inputs[str(file.resolve())] = sha
        if workflow == "flowcompat-regional-agent-1.0":
            recipe = provenance["regional_recipe"]
            for stem in ("receipt", "reference", "rules"):
                file = Path(recipe[stem + "_path"])
                file = file if file.is_absolute() else self.repo / file
                require(digest(file) == recipe[stem + "_sha256"], "Regional recipe artifact changed: " + stem)
                self.inputs[str(file.resolve())] = digest(file)
        return {"workflow_version": workflow, "null_round": null_round,
                "interpretation": "Independently rechecked binding and registered mechanism criteria; efficacy not assessed"}

    def freeze_snapshot(self, number):
        """The selected source and freeze declaration must predate all confirmation labels."""
        commit = self.rounds[number]["execution"]["code_commit"]
        frozen_path = self.docs / "frozen_validation.json"
        source_path = self.cfg / f"round{self.selected:02d}.json"
        tracked = lambda file, default: file.relative_to(self.repo).as_posix() if file.is_relative_to(self.repo) else default
        freeze_name = tracked(frozen_path, "docs/experiments/flowcompat30_20261009/frozen_validation.json")
        source_name = tracked(source_path, "configs/experiments/flowcompat30_v1/" + source_path.name)
        require(hashlib.sha256(self.git.blob(commit, freeze_name)).hexdigest() == digest(frozen_path),
                "Freeze declaration not present unchanged in actual confirmation inference commit")
        require(hashlib.sha256(self.git.blob(commit, source_name)).hexdigest() == digest(source_path),
                "Frozen screening source not present unchanged in actual confirmation inference commit")
        return {"confirmation_commit": commit, "freeze_record_present_before_inference": True}

    def panel_pairing(self):
        all_keys = {arm: [] for arm in ("candidate", "R26", "native")}
        sigs = []
        for control, candidate in zip(CONTROL_ROUNDS, CANDIDATE_ROUNDS):
            a, b = self.rounds[control], self.rounds[candidate]
            require(loading_upper_limit(a["config"]) == loading_upper_limit(b["config"]),
                    "Paired candidate/control loader enumeration limits differ")
            self.native_settings(b, a)
            for batch in a["config"]["experiment"]["batch_indices"]:
                keys = [a["execution"]["initial_state_signatures"][f"{arm}/{batch}"] for arm in ("gradient", "unguided")]
                keys.append(b["execution"]["initial_state_signatures"][f"gradient/{batch}"])
                require(len(set(keys)) == 1, "Candidate/R26/native initial-state mismatch in batch " + str(batch))
                sigs.append(keys[0])
            for row in a["metrics"]:
                all_keys["R26" if row["arm"] == "gradient" else "native"].append((int(row["batch"]), int(row["slot"])))
            all_keys["candidate"].extend((int(row["batch"]), int(row["slot"])) for row in b["metrics"])
        require(len(sigs) == len(set(sigs)) == 6, "Repeated initial-state batch in held-out panel")
        screening = set(self.rounds[1]["execution"]["initial_state_signatures"].values())
        require(not set(sigs) & screening, "Confirmation initial states repeat screening batches")
        expected = {(batch, slot) for batch in BATCHES for slot in SLOTS}
        require(all(len(keys) == len(set(keys)) == 300 and set(keys) == expected for keys in all_keys.values()),
                "Confirmation must have exactly six complete distinct 50-slot batches in all arms")
        for n in CONTROL_ROUNDS + CANDIDATE_ROUNDS:
            self.native_settings(self.rounds[n], self.rounds[1])
        upstream = [self.rounds[n]["upstream"] for n in CONTROL_ROUNDS + CANDIDATE_ROUNDS]
        require(all(item["files"] == upstream[0]["files"] and item["input_dataset"] == upstream[0]["input_dataset"] for item in upstream),
                "Declared upstream sources changed across confirmation rounds")
        return {"batches": list(BATCHES), "slots_per_batch": 50, "attempts_per_arm": 300,
                "upstream_scope": sorted(UPSTREAM_FILES)}

    def run(self):
        frozen = self.check("frozen_candidate", lambda: self.load(self.docs / "frozen_validation.json"))
        source = base = None
        if frozen is not None:
            def validate_frozen():
                self.selected = frozen["selected_round"]
                require(type(self.selected) is int and 3 <= self.selected <= 24 and self.selected not in (8, 10), "Invalid screening source round")
                require(frozen.get("before_confirmation_labels") is True and frozen.get("confirmation_batches") == list(BATCHES), "Candidate was not frozen for the declared panel")
                file = self.cfg / f"round{self.selected:02d}.json"
                require(digest(file) == frozen["program_sha256"], "Frozen screening program bytes changed")
                program = self.load(file)
                require(program["reference_sha256"] == frozen["reference_sha256"], "Frozen teacher reference mismatch")
                require(isinstance(frozen.get("screening_admissible"), bool), "Missing pre-confirmation screening admissibility")
                return program
            source = self.check("frozen_program", validate_frozen)
        base = self.check("R26_baseline_program", lambda: self.load(self.repo / "configs/experiments/skill_ablation_v1/incumbent.json"))
        if source is not None and base is not None:
            self.check("frozen_vs_R26_learned_window", lambda: require(source["window"] == base["window"], "Frozen candidate and R26 learned windows differ"))
            null_round = 8 if source.get("generation_interface") == "flowcompat_v2" else 2
            for number in dict.fromkeys((1, 2, null_round, self.selected)):
                expected = source if number == self.selected else dict(base)
                if number == 8:
                    expected["generation_interface"] = "flowcompat_v2"
                arms = ("gradient",) if number in (8, self.selected) else ("unguided", "gradient")
                self.check(f"round{number:02d}_retention_and_data", lambda n=number, p=expected, a=arms: self.round_data(n, (47, 48), a, p))
                if number in self.rounds:
                    self.check(f"round{number:02d}_executed_source_closure", lambda n=number: self.code_closure(n))
            if 1 in self.rounds and null_round in self.rounds:
                def null_pair():
                    a, b = self.rounds[1], self.rounds[null_round]
                    self.native_settings(b, a)
                    cols = ("arm", "batch", "slot", "pic50_on_rescore", "smiles")
                    arms = {row["arm"] for row in b["metrics"]}
                    order = lambda rows: sorted(tuple(row[k] for k in cols) for row in rows if row["arm"] in arms)
                    require(order(a["metrics"]) == order(b["metrics"]), "Actual null terminal results differ from R26")
                    require(b["execution"]["initial_state_signatures"] == {key: value for key, value in a["execution"]["initial_state_signatures"].items() if key.split("/")[0] in arms}, "Null initial state differs")
                    window_hashes = lambda data: {f'{row["arm"]}/{row["batch"]}': row["window_state_sha256"] for row in data["feedback"]["batch_diagnostics"]}
                    left, right = window_hashes(a), window_hashes(b)
                    require(set(right) == set(b["execution"]["initial_state_signatures"]) and all(valid_sha(value) and left.get(key) == value for key, value in right.items()),
                            "Actual null full-window state file SHA differs from original R26")
                    require(b["feedback"].get("null_control_parity") is True and b["feedback"].get("null_control_terminal_parity") is True, "Recorded null arithmetic/terminal parity failed")
                    return {"round": null_round, "exact_terminal_control": True}
                self.check("actual_null_control", null_pair)
            if self.selected in self.rounds and "code_files" in self.rounds[self.selected] and null_round in self.rounds:
                self.check("screening_implementation_gate", lambda: self.gate(self.selected, null_round))
            for number in range(25, 31):
                is_control = number in CONTROL_ROUNDS
                batches = (49 + 2 * ((number - 25) // 2), 50 + 2 * ((number - 25) // 2))
                expected = base if is_control else source
                self.check(f"round{number:02d}_retention_and_data", lambda n=number, b=batches, p=expected, c=is_control: self.round_data(n, b, ("unguided", "gradient") if c else ("gradient",), p))
                if number in self.rounds:
                    self.check(f"round{number:02d}_pre_inference_freeze_snapshot", lambda n=number: self.freeze_snapshot(n))
                    origin = self.rounds.get(2 if is_control else self.selected, {})
                    self.check(f"round{number:02d}_frozen_source_closure", lambda n=number, o=origin: self.code_closure(n, o["code_files"]))
                    self.check(f"round{number:02d}_upstream_attestation", lambda n=number: self.upstream(n))
                    if not is_control and "code_files" in self.rounds[number] and null_round in self.rounds:
                        self.check(f"round{number:02d}_implementation_gate", lambda n=number: self.gate(n, null_round))
            self.check("six_batch_pairing_native_runtime_and_upstream", self.panel_pairing)
        passed = bool(self.checks) and all(row["passed"] for row in self.checks)
        return {"schema_version": VERSION, "integrity_passed": passed,
                "eligible_for_efficacy_decision": passed and frozen.get("screening_admissible") is True if frozen else False,
                "adoption_decision": "not assessed; production default remains historical R26",
                "repo": str(self.repo), "input_dataset": str(self.docs), "configuration_dataset": str(self.cfg),
                "validator_sha256": digest(__file__), "checks": self.checks,
                "input_files": self.inputs,
                "boundaries": ["Retained reports and Git/source attestations are evidence records; retired raw arrays are not re-created or re-audited.",
                               "Upstream source identity covers only four declared FLOWR files, not all installed dependencies or hardware.",
                               "Checkpoint identity uses recorded SHA; this audit does not reread the remote checkpoint.",
                               "Pre-binding configuration field identity uses a separately attested canonical SHA; recorded raw byte SHA is not reconstructed from the parsed object.",
                               "Static local imports are reconstructed from executed Git objects; dynamic third-party imports are outside this closure.",
                               "Six fixed-seed generation batches support a paired batch analysis, not six biological replicates or an affinity validation.",
                               "A passing audit checks integrity and implementation; the separate preregistered efficacy criteria must still pass."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="Local Git repository containing executed commits and immutable sources")
    parser.add_argument("--input-dataset", help="Explicit retained campaign reports directory (defaults to this repo's campaign)")
    parser.add_argument("--configuration-dataset", help="Explicit frozen round configuration directory")
    parser.add_argument("--output", required=True, help="Separate JSON audit report; never a retained or frozen input")
    args = parser.parse_args(argv)
    audit = ConfirmationAudit(args.repo, args.input_dataset, args.configuration_dataset)
    result = audit.run()
    output = Path(args.output).resolve()
    require(str(output) not in result["input_files"], "Audit output cannot overwrite an input")
    require(not output.is_relative_to(audit.cfg) and output != audit.docs / "frozen_validation.json"
            and not (output.is_relative_to(audit.docs) and re.fullmatch(r"round\d+", output.relative_to(audit.docs).parts[0])),
            "Audit output cannot overwrite frozen configuration or retained round evidence")
    if output.exists():
        require(read_json(output).get("schema_version") == VERSION, "Audit output cannot overwrite an existing non-audit artifact")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"integrity_passed": result["integrity_passed"], "eligible_for_efficacy_decision": result["eligible_for_efficacy_decision"],
                      "failed_checks": [row["check"] for row in result["checks"] if not row["passed"]], "output": str(output)}, ensure_ascii=False))
    return 0 if result["integrity_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
