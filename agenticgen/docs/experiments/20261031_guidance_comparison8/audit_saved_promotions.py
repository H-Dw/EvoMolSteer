"""CPU-only reconstruction of the two failed population views; no new sampling."""

import argparse
from pathlib import Path

import h5py
import torch

from agenticgen.uniguide.io import read, write, sha
from agenticgen.uniguide.native_tools import load_pt, tensor_digest
from agenticgen.uniguide.geometry_sampling import candidate_identity, validate_receipt
from agenticgen.uniguide.geometry_shards import combine_products
from agenticgen.uniguide.calibration_search import assess, strength


def audit(source, evidence, destination):
    source, destination = Path(source), Path(destination)
    preserved = evidence["source_files_sha256"]
    if any(sha(path) != digest for path, digest in preserved.items()):
        raise ValueError("Source evidence changed before reconstruction")
    destination.mkdir(parents=True, exist_ok=False)
    failure = evidence["failed"][0]
    old = source / Path(failure["path"]).parent
    actual = candidate_identity(failure["identity"])
    shards = []
    for receipt in sorted((source / "sample_shards").glob("*/COMPLETE.json")):
        row = validate_receipt(receipt.parent)
        if candidate_identity(row) == actual:
            shards.append(row)
    if len(shards) != 5:
        raise ValueError(
            "The failed fifty-population view requires five complete shards"
        )
    shards.sort(
        key=lambda row: read(source / row["id"] / "manifest.json")["identity"][
            "logical_slots"
        ][0]
    )
    rng = torch.get_rng_state().clone()
    combine_products(source, shards, destination)
    checks = {
        "terminal_tensors_bitwise_equal": tensor_digest(load_pt(old / "terminal.pt.gz"))
        == tensor_digest(load_pt(destination / "terminal.pt.gz")),
        "records_equal": read(old / "records.json")
        == read(destination / "records.json"),
        "steps_equal": read(old / "steps.json") == read(destination / "steps.json"),
        "random_ledger_equal": read(old / "rng_ledger.json")
        == read(destination / "rng_ledger.json"),
        "no_cpu_rng_consumption": torch.equal(rng, torch.get_rng_state()),
    }
    with (
        h5py.File(old / "trace.h5") as before,
        h5py.File(destination / "trace.h5") as after,
    ):
        checks["all_trace_arrays_bitwise_equal"] = set(before) == set(after) and all(
            before[key].shape == after[key].shape
            and before[key].dtype == after[key].dtype
            and all(
                before[key][i].tobytes() == after[key][i].tobytes()
                for i in range(len(before[key]))
            )
            for key in before
        )
    checks["source_files_unchanged"] = all(
        sha(path) == digest for path, digest in preserved.items()
    )
    if not all(checks.values()):
        raise ValueError("Saved promotion reconstruction differs: " + repr(checks))
    result = dict(
        checks,
        actual_scales=actual,
        completed_source_shards=5,
        attempts=50,
        new_sampling_attempts=0,
        llm_calls=0,
        raw_sources_retained=True,
        original_failure_retained=(old / "FAILED.json").exists(),
        merged_output_policy=read(destination / "OUTPUT_POLICY.json"),
    )
    write(destination / "AUDIT_COMPLETE.json", result)
    return result


def audit_five_window_screen(source, evidence, destination):
    source, destination = Path(source), Path(destination)
    preserved = evidence["source_files_sha256"]
    if any(sha(path) != digest for path, digest in preserved.items()):
        raise ValueError("Source screening evidence changed")
    rows = [
        validate_receipt(p.parent) for p in source.glob("calibration/**/COMPLETE.json")
    ]
    rows = [r for r in rows if r["attempts"] == 10]
    if len(rows) != len(evidence["budget"]) or any(
        len(r["effective_guidance_scale_schedule"]) != 5 for r in rows
    ):
        raise ValueError("Saved five-window screening populations differ")
    usable, curve, boundary = assess(rows)
    result = dict(
        measured_attempts=sum(r["attempts"] for r in rows),
        completed_profiles=len(rows),
        usable_actual_scales=[strength(r) for r in usable],
        curve=curve,
        boundary=boundary,
        five_window_profiles_and_displacements_accepted=True,
        new_sampling_attempts=0,
        llm_calls=0,
        source_files_unchanged=True,
    )
    destination.mkdir(parents=True, exist_ok=False)
    write(destination / "AUDIT_COMPLETE.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-plan", required=True)
    parser.add_argument("--preservation", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plan, evidence = read(args.source_plan), read(args.preservation)
    selected = {job["id"]: job for job in plan["jobs"]}
    results = {}
    for name, rows in evidence["jobs"].items():
        check = (
            audit
            if selected[name]["experiment"] == "prefix_joint"
            else audit_five_window_screen
        )
        results[name] = check(selected[name]["root"], rows, Path(args.output) / name)
    write(Path(args.output) / "COMPLETE.json", results)


if __name__ == "__main__":
    main()
