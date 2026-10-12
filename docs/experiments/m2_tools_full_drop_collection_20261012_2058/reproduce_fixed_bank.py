"""Reproduce paired summaries from an immutable collection snapshot.

Usage: python reproduce_fixed_bank.py --repo REPO --snapshot SNAPSHOT_DIRECTORY
No API call or inference; outputs are written beside the captured state.
"""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evomolsteer.ablation.collection import merge_observations, quality, historical_difference
from evomolsteer.ablation.shared_controls import verify_attempts, verify_initial_states
from evomolsteer.ablation.statistics import compare, compare_strain
from evomolsteer.io import digest, read_json, write_json


def main(repo, snapshot):
    source = read_json(snapshot / "source_manifest.json")
    blob = gzip.decompress((snapshot / "state_snapshot.json.gz").read_bytes())
    import hashlib
    if hashlib.sha256(blob).hexdigest() != source["state_sha256"]:
        raise ValueError("Captured state changed")
    state = json.loads(blob)
    study = repo / source["study"]
    bank_path = study / "shared_control_bank/manifest.json"
    bank = read_json(bank_path)
    if digest(bank_path) != state["shared_control_bank"]["manifest_sha256"]:
        raise ValueError("Fixed bank differs from captured execution")
    threshold = source["elite_valid_threshold"]
    batches, size = bank["batches"], bank["batch_size"]
    groups, unique_cohorts = {}, set()
    bindings = {bank_path.relative_to(repo).as_posix(): digest(bank_path)}
    deliveries = []
    for result in state["program_results"].values():
        if result["name"] in unique_cohorts:
            continue
        unique_cohorts.add(result["name"])
        for arm, record in result["outcomes"].items():
            path = repo / record["metrics"]
            if digest(path) != source["source_files"][record["metrics"]]:
                raise ValueError("Metric file differs from captured snapshot")
            f = pd.read_csv(path)
            f = f.loc[f.batch.isin(batches)]
            if f.empty:
                continue
            verify_attempts(f, batches, size)
            verify_initial_states(result, bank["result"], batches)
            sha = "unguided" if arm == "unguided" else result["effective_signature"]
            groups.setdefault(sha, []).append(f)
            bindings[record["metrics"]] = digest(path)
        if any(int(row["batch"]) in batches for row in result["delivery"]["batches"]):
            delivery = result["delivery"]["batches"]
            response = study / "transport_records" / (result["name"] + ".response_vs_M2.json")
            observed = read_json(response) if response.exists() else {}
            deliveries.append({"cohort": result["name"], "signature": result["effective_signature"],
                "mean_injection_rms_A": float(np.mean([r["mean_injection_rms_A"] for r in delivery])),
                "post_window_nonzero": sum(r["post_window_nonzero"] for r in delivery),
                "mean_forecast_difference_z": observed.get("mean_forecast_difference_z"),
                "max_forecast_difference_z": observed.get("max_forecast_difference_z")})
    frames = {sha: merge_observations(fs) for sha, fs in groups.items()}
    base_sha = bank["effective_signature"]
    labels, rows = {}, []
    for sha, frame in frames.items():
        if sha == "unguided":
            label = "无引导"
        elif sha == base_sha:
            label = "M2"
        else:
            p = study / "programs" / (sha + ".json")
            if digest(p) != source["source_files"][p.relative_to(repo).as_posix()]:
                raise ValueError("Numerical program differs from captured snapshot")
            program = read_json(p)
            bindings[p.relative_to(repo).as_posix()] = digest(p)
            label = ("时间教师混合 " + str(program["temporal_teacher_blend"]) if "temporal_teacher_blend" in program
                     else "时间强度幂 " + str(program.get("time_ramp_power")))
        labels[sha] = label
        rows.append({"program": sha, "label": label, **quality(frame, threshold)})
    history_path = study / "evaluation_controls/historical_steer.metrics.parquet"
    if digest(history_path) != source["source_files"][history_path.relative_to(repo).as_posix()]:
        raise ValueError("Historical control differs from captured snapshot")
    history = merge_observations([pd.read_parquet(history_path)])
    bindings[history_path.relative_to(repo).as_posix()] = digest(history_path)
    rows.append({"program": "historical_Steer", "label": "历史 Steer（不成对）", **quality(history, threshold)})
    gradient = [s for s in frames if s != "unguided"]
    family = len(gradient) + len(gradient) - 1
    comparisons, strains = {}, {}
    for sha in gradient:
        for control in [base_sha, "unguided"]:
            if sha == control:
                continue
            comparisons[sha + "_minus_" + control] = compare([frames[sha]], [frames[control]], family=family, replicates=100000)
        comparisons[sha + "_minus_historical_Steer"] = historical_difference(frames[sha], history, family=len(gradient))
        if sha != base_sha:
            strains[sha] = compare_strain([frames[base_sha]], [frames[sha]], family=family, replicates=100000)
    pd.DataFrame(rows).to_csv(snapshot / "fixed_bank_quality.csv", index=False)
    write_json(snapshot / "fixed_bank_comparisons.json", comparisons)
    write_json(snapshot / "fixed_bank_strain_comparisons.json", strains)
    write_json(snapshot / "fixed_bank_delivery.json", deliveries)
    write_json(snapshot / "fixed_bank_analysis_manifest.json", {"batches": batches, "attempts_per_program": size * len(batches),
        "state_sha256": source["state_sha256"], "source_files": bindings, "labels": labels,
        "analysis_source_sha256": digest(Path(__file__)), "elite_valid_threshold": threshold,
        "paired_comparison_family": family, "independent_final_confirmation": False,
        "historical_comparison_unpaired_and_budget_unequal": True, "repeated_programs_not_new_observations": True})
    print(pd.DataFrame(rows)[["label", "mean_pic50", "strain_median", "strain_p90", "valid_count", "PB_count", "elite_valid_count", "max_pic50"]].to_string(index=False))
    for name, value in comparisons.items():
        print(name[:12], name.split("_minus_")[-1][:12], round(value["delta_pic50"], 6), value["simultaneous_CI"])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True, type=Path)
    p.add_argument("--snapshot", required=True, type=Path)
    a = p.parse_args()
    main(a.repo.resolve(), a.snapshot.resolve())
