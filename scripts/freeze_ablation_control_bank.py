"""Freeze an already generated/evaluated 400+400 cohort; never run inference.

Requires a checksummed downloaded transport and downloaded final SDF files.
Data live in ignored data/control_banks; small manifests remain in Git.
"""
import argparse
import gzip
import hashlib
import shutil
from pathlib import Path

from run_m2_skill_api_study import Study
from evaluate_dependency_archive import run as verify_evaluation
from analyze_dependency_study import metrics
from evomolsteer.ablation.shared_controls import batch_key, verify_attempts
from evomolsteer.continuous.m2_skill_api import BASE, REF, effective_signature
from evomolsteer.io import digest, read_json, write_json
from evomolsteer.storage.generation_archive import restoration_destination


def freeze(repo, config, name):
    owner = Study(repo, config)
    location = owner.study / "shared_control_bank"
    manifest = location / "manifest.json"
    if manifest.exists():
        raise ValueError("Control bank is already frozen; do not overwrite")
    remote = read_json(location / "remote_completed.json")
    if remote["name"] != name or remote["status"] != "complete":
        raise ValueError("Completed inference receipt required")
    spec = read_json(owner.config_root / "jobs" / (name + ".json"))
    if spec["n"] != 400 or spec["arms"] != "gradient,unguided" or spec["prior_seed_mode"] != "indexed_batch":
        raise ValueError("Completed paired 400-attempt indexed bank required")
    archive = owner.transport / (name + ".tar.gz")
    metadata = owner.transport / (name + ".tar.gz.json")
    restored = restoration_destination(owner.transport, name)
    reports = owner.study / "transport_records"
    output = repo / "results" / owner.study.name / name
    reference = repo / "configs/experiments/ck2_terminal_seed42_v1/local_reference.json.gz"
    verify_evaluation(name, archive, metadata, restored, reports, output, reference, 2)
    root = restored / "results" / name
    execution = read_json(root / "config.json")
    if remote["initial_state_signatures"] != {
        f"{arm}/batch_{b:03d}": remote["initial_state_signatures"][f"gradient/batch_{b:03d}"]
        for arm in ["gradient", "unguided"] for b in spec["batches"]}:
        raise ValueError("Initial-state mismatch between bank arms")
    data = repo / "data/control_banks" / owner.study.name / name
    data.mkdir(parents=True, exist_ok=True)
    preserved = []
    for source in [archive, metadata, output / "candidate_metrics.csv", output / "terminal_report.json",
                   output / "dependency_evaluation_sources.json"]:
        destination = data / source.name
        shutil.copyfile(source, destination)
        preserved.append(destination)
    for arm in ["gradient", "unguided"]:
        source = location / (arm + ".sdf.gz")
        if hashlib.sha256(gzip.decompress(source.read_bytes())).hexdigest() != remote["seed_structures"][arm]["sha256"]:
            raise ValueError("Downloaded terminal SDF differs from completed cohort")
        destination = data / source.name
        shutil.move(source, destination)
        preserved.append(destination)
    delivery = owner.delivery(root)
    write_json(reports / (name + ".delivery.json"), delivery)
    owner.forecast_response(root, reports / (name + ".forecast_features.parquet"))
    outcomes = {}
    for arm in ["gradient", "unguided"]:
        frame, summary = metrics(output / "candidate_metrics.csv", arm)
        verify_attempts(frame, spec["batches"], spec["batch"])
        destination = reports / (name + "." + arm + ".metrics.csv")
        columns = ["batch", "slot", "pic50_on_rescore", "valid_connected", "pb_fast_pass", "energy_status",
                   "mmff_relief_per_heavy", "smiles", "failure_reason", "decode_status", "build_success"]
        frame[columns].to_csv(destination, index=False)
        outcomes[arm] = {"summary": summary, "metrics": destination.relative_to(repo).as_posix()}
        preserved.append(destination)
    signature = effective_signature(read_json(repo / BASE))
    result = {"name": name, "effective_signature": signature, "code_commit": remote["code_commit"],
        "prior_seed_mode": spec["prior_seed_mode"], "outcomes": outcomes, "delivery": delivery,
        "archive_sha256": digest(archive), "initial_state_signatures": remote["initial_state_signatures"]}
    generator_paths = [repo / "scripts/generate_dependency_flowr.py"]
    for package in ["generation", "continuous"]:
        generator_paths.extend((repo / "src/evomolsteer" / package).glob("*.py"))
    m = {"schema_version": "paired-control-bank-1.0", "batches": spec["batches"], "batch_size": spec["batch"],
        "master_seed": spec["seed"], "steps": spec["steps"], "prior_seed_mode": spec["prior_seed_mode"],
        "program": BASE, "effective_signature": signature, "result": result,
        "checkpoint_sha256": execution["extension"]["checkpoint_sha256"],
        "native_integrator_parameters": execution["extension"]["native_integrator_parameters"],
        "structures": {arm: (data / (arm + ".sdf.gz")).relative_to(repo).as_posix() for arm in outcomes},
        "all_attempt_tensor_and_decode_archive": (data / archive.name).relative_to(repo).as_posix(),
        "files": {p.relative_to(repo).as_posix(): digest(p) for p in preserved + [repo / BASE, repo / REF]},
        "generator_sources": {p.relative_to(repo).as_posix(): digest(p) for p in generator_paths},
        "selection_rule": "Latest fully generated cohort with all terminal structures still retained; no performance selection",
        "failures_retained": True, "historical_completed_comparisons_unchanged": True}
    write_json(manifest, m)
    key = batch_key(spec["batches"])
    owner.state.setdefault("matched_controls", {})[key] = result
    owner.state.setdefault("measured", {})[signature + ":" + key] = result
    infer_key = signature + ":gradient,unguided:v3:indexed_batch:" + key + ":" + digest(repo / REF)
    owner.state["program_results"][infer_key] = result
    if name not in owner.state["completed_tasks"]:
        owner.state["completed_tasks"].append(name)
    owner.save(status="ready_for_shared_controls", task=None, error=None, error_type=None)
    print({"manifest": str(manifest), "attempts_per_arm": 400, "bytes_preserved": sum(p.stat().st_size for p in preserved)})
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cohort", required=True)
    args = parser.parse_args()
    freeze(Path(__file__).resolve().parents[1], read_json(args.config), args.cohort)
