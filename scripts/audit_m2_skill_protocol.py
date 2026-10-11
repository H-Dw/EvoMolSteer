"""Audit frozen plans against recorded calls and execution; never run a study.

Only --output is written. No API, SSH, compilation, inference or cleanup occurs.
Checksums establish file provenance, not correctness of scientific reasoning.
"""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator

from evomolsteer.continuous.m2_skill_api import effective_signature, instruction


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def messages(path):
    record = read(path)
    result = []
    for message in record["body"]["messages"]:
        if "content" in message:
            result.append(message)
            continue
        content = (path.parent / message["content_relative_path"]).resolve()
        if sha(content) != message["content_sha256"]:
            raise ValueError(f"Recorded input checksum mismatch: {content}")
        result.append({"role": message["role"], "content": content.read_text(encoding="utf-8")})
    return record, result


def audit(repo, plan, study):
    protocol = read(plan / "protocol.json")
    activation = read(study / "activation.json")
    state = read(study / "run_status.json")
    originals = {r["condition_id"]: r for r in csv.DictReader(
        (plan / "conditions.csv").open(encoding="utf-8-sig"))}
    active = {r["condition_id"]: r for r in activation["conditions"]}
    integrity = []
    for row in read(plan / "artifact_manifest.json")["files"]:
        path = plan / row["path"]
        integrity.append({"path": row["path"], "expected_sha256": row["sha256"],
                          "actual_sha256": sha(path), "matches": sha(path) == row["sha256"]})
    row_changes = [{"condition": name, "field": key, "planned": value,
                    "activated": active[name].get(key)}
                   for name, original in originals.items() if name in active
                   for key, value in original.items()
                   if key not in {"requested_agent_model", "status"}
                   and active[name].get(key) != value]
    baseline = protocol["execution_baseline"]
    baseline_checks = {key: sha(repo / baseline[key]) == baseline[key + "_sha256"]
                       for key in ("program", "reference")}
    calls = []
    candidates = []
    for folder in sorted((study / "calls").glob("*/chain_*")):
        condition = folder.parent.name
        for role in ("Analyst", "Designer"):
            response_path = folder / f"{role}.json"
            request_path = folder / f"{role}.request.json"
            receipt_path = folder / f"{role}.receipt.json"
            if not all(p.exists() for p in (response_path, request_path, receipt_path)):
                continue
            request, delivered = messages(request_path)
            response, receipt = read(response_path), read(receipt_path)
            recorded_instruction = (folder / f"{role}.instructions.md").read_text(encoding="utf-8")
            expected_instruction = instruction(repo, study, active[condition], role)
            calls.append({"condition": condition, "chain": folder.name, "role": role,
                          "schema_valid": Draft202012Validator(request["response_schema"]).is_valid(response),
                          "request_receipt_matches": sha(request_path) == receipt["request_sha256"],
                          "response_receipt_matches": sha(response_path) == receipt["response_sha256"],
                          "instruction_matches_frozen_modules": recorded_instruction == expected_instruction,
                          "system_instruction_delivered": delivered[0]["content"] == recorded_instruction,
                          "frozen_evidence_delivered": json.loads(delivered[1]["content"]) == read(study / "evidence.json"),
                          "requested_model": receipt["requested_model"],
                          "returned_model": receipt["returned_model"],
                          "provider_request_id": receipt["provider_request_id"]})
        compiler_path = folder / "compiler.json"
        if not compiler_path.exists():
            continue
        compiler = read(compiler_path)
        program_path = repo / compiler["program"]
        program = read(program_path)
        incumbent = read(repo / baseline["program"])
        # This field is an operative default absent from the legacy M2 JSON.
        incumbent.setdefault("pointcloud_delta_A", 1.)
        program.setdefault("pointcloud_delta_A", 1.)
        excluded = {"program_id", "round", "robust_delta", "geometry_block_weights", "target_definition"}
        changes = {key: {"baseline": incumbent.get(key), "candidate": program.get(key)}
                   for key in sorted(set(incumbent) | set(program))
                   if key not in excluded and incumbent.get(key) != program.get(key)}
        designer = read(folder / "Designer.json")
        edit = designer["edit"]
        allowed = protocol["lanes"]["G"]["prospective_allowed_changes"]
        legal = (edit is None or (edit["parameter"] in allowed
                                 and isinstance(allowed[edit["parameter"]], list)
                                 and edit["value"] in allowed[edit["parameter"]]))
        # Region recipes require separate evidence binding; do not label them
        # invalid because their prospective protocol describes a recipe in prose.
        if edit and edit["parameter"] == "region_index":
            legal = None
        if active[condition]["lane"] == "U" and edit:
            legal = edit["parameter"] in protocol["lanes"]["U"] and edit["value"] in protocol["lanes"]["U"][edit["parameter"]]
        candidates.append({"condition": condition, "chain": folder.name,
                           "decision": designer["decision"], "edit": edit,
                           "program": compiler["program"], "operative_changes": changes,
                           "edit_within_original_plan_grid": legal,
                           "compiler_response_matches": compiler["response_sha256"] == sha(folder / "Designer.json"),
                           "compiler_program_matches": compiler["program_sha256"] == sha(program_path),
                           "effective_signature_matches": compiler["effective_sha256"] == effective_signature(read(program_path)),
                           "reference_matches_frozen_M2": compiler["reference_sha256"] == baseline["reference_sha256"],
                           "executions": [r for r in state.get("chains", [])
                                          if r["identity"] == condition + ":" + str(int(folder.name.split("_")[-1]))]})
    snapshot = study / "protocol_audit_20261011/remote_snapshot.json"
    return {"scope": "Read-only plan/call/program provenance audit; not a functional or performance ablation",
            "plan_file_integrity": integrity, "all_manifest_files_unchanged": all(r["matches"] for r in integrity),
            "frozen_M2_integrity": baseline_checks,
            "module_registry_matches_plan": read(study / "module_registry.json")["modules"] == read(plan / "module_registry.json")["modules"],
            "planned_phase_counts": dict(collections.Counter(r["phase"] for r in originals.values())),
            "activated_phase_counts": dict(collections.Counter(r["phase"] for r in active.values())),
            "activated_matrix_changes_except_model_and_queue_status": row_changes,
            "not_activated": [name for name in originals if name not in active],
            "execution_status": state["status"], "active_task_at_stop": state.get("task"),
            "completed_role_chains": len(state.get("chains", [])), "expected_chains_per_condition": activation["chain_replicates"],
            "completed_inference_tasks": state.get("completed_tasks", []),
            "completed_module_deletion_or_alternative_chains": [r for r in state.get("chains", []) if active[r["condition"]]["changed_module"] or active[r["condition"]]["removed_modules"]],
            "failed_interface": state.get("error", "").splitlines()[:1],
            "verified_calls": calls, "compiled_candidates": candidates,
            "recorded_activation_amendments": activation["protocol_amendments"],
            "planned_Designer_input_policy": protocol["agent_replicates"]["designer_phase_Analyst_inputs"],
            "actual_configured_Designer_input_policy": activation["analyst_freeze"],
            "planned_U_reward_shape": protocol["lanes"]["U"]["reward_shape"],
            "actual_configured_U_reward_shape": activation["dose_shape"],
            "later_phase_deviations_have_executed": any(r["condition"].startswith("D_") for r in state.get("chains", [])),
            "remote_observation": read(snapshot) if snapshot.exists() else None,
            "limits": ["Recorded amendments are not evidence of user approval.",
                       "Byte integrity does not establish scientific correctness or absence of all external tampering.",
                       "P_struct rewards are baseline policy outputs, not D2 deletion/alternative effects.",
                       "Independent schema-repair probes outside calls/ are excluded from scientific role chains.",
                       "This tool does not repair or resume the protocol."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.repo.resolve(), args.plan.resolve(), args.study.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "plan_unchanged": result["all_manifest_files_unchanged"],
                      "verified_calls": len(result["verified_calls"]), "completed_chains": result["completed_role_chains"],
                      "status": result["execution_status"]}, ensure_ascii=False))
