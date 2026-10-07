"""Read-only, compact audit supporting the elite-path literature proposal.

This is a report snapshot, not a new mining or generation workflow. It reads
only genealogy, clocks, probabilities and existing metadata; no model calls.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(repo):
    library_path = repo / "configs/experiments/skill_ablation_v1/endpoint_reference.json.gz"
    program_path = repo / "configs/experiments/skill_ablation_v1/incumbent.json"
    credit_path = repo / "docs/experiments/ck2_affinity_geometry30_20261007/terminal_lineage_v4/batch_node_credit.parquet"
    library = json.loads(gzip.decompress(library_path.read_bytes()))
    program = json.loads(program_path.read_text(encoding="utf-8"))
    times = np.asarray(library["times"], dtype=float)
    donors = set(library["discovery_batches"])
    credit = pd.read_parquet(credit_path)
    rows, sources, protocols = [], [], []
    dataset = repo / "data/raw/ck2_clk3_lineage_20261003/results/main1000_w050"
    for batch in range(20):
        path = dataset / f"single/batch_{batch:03d}/trajectory.npz"
        with np.load(path, allow_pickle=False) as archive:
            root = archive["root_slot"].astype(np.int64)
            selected = archive["selected_indices"].astype(np.int64)
            clock = np.round(archive["score_time"][:, 0].astype(float), 6)
            state_time = archive["state_time"].astype(float)
            probability = archive["selection_probability"].astype(float)
            resampled = archive["resampled"].astype(bool)
        if not np.array_equal(root[1:], np.take_along_axis(root[:-1], selected[:-1], axis=1)):
            raise ValueError(f"Root/parent mapping inconsistent in batch {batch}")
        indices = np.flatnonzero(np.isin(clock, times))
        if not np.array_equal(clock[indices], times) or not resampled[indices].all():
            raise ValueError(f"Observed library clocks inconsistent in batch {batch}")
        for node in indices:
            counts = np.unique(root[node], return_counts=True)[1].astype(float)
            fraction = counts / counts.sum()
            weights = probability[node]
            if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
                raise ValueError("Invalid recorded selection probabilities")
            weights = weights / weights.sum()
            rows.append({"batch": batch, "time": float(clock[node]),
                         "root_count": int(len(counts)),
                         "root_ess": float(1 / np.square(fraction).sum()),
                         "weight_ess": float(1 / np.square(weights).sum())})
        chosen = np.flatnonzero(resampled)
        protocols.append({"batch": batch, "resampling_count": int(len(chosen)),
                          "last_score_time": float(clock[chosen[-1]]),
                          "last_proposal_state_time": float(state_time[chosen[-1]])})
        sources.append({"path": str(path.relative_to(repo)).replace("\\", "/"),
                        "sha256": sha256(path), "root_parent_mapping_verified": True})
    table = pd.DataFrame(rows)
    summary = {}
    for name, keep in (("donor14", table.batch.isin(donors)),
                       ("non_donor6", ~table.batch.isin(donors)),
                       ("all20", np.ones(len(table), dtype=bool))):
        group = table.loc[keep]
        summary[name] = {str(time): {
            field + "_mean": float(group.loc[np.isclose(group.time, time), field].mean())
            for field in ("root_count", "root_ess", "weight_ess")}
            for time in (0, .1, .25, .49)}
    counterfactuals = []
    for step in (10, 30, 50):
        path = dataset / f"counterfactual/step_{step:03d}/final_records.json"
        records = json.loads(path.read_text(encoding="utf-8"))
        records = records if isinstance(records, list) else records["records"]
        counterfactuals.append({"path": str(path.relative_to(repo)).replace("\\", "/"),
                               "sha256": sha256(path), "records": len(records),
                               "source_arms": sorted({r["original_node"].split("_b")[0] for r in records}),
                               "restart_phases": sorted({r.get("restart_phase", "not_in_record") for r in records}),
                               "selected_statuses": sorted({r["original_selection"] for r in records})})
    return {
        "schema_version": "elite-path-information-audit-1.0", "date": "2026-10-07",
        "source_code_sha256": sha256(Path(__file__)),
        "execution": "Read-only local audit; no network/model/inference call",
        "library": {"path": str(library_path.relative_to(repo)).replace("\\", "/"),
                    "sha256": sha256(library_path), "schema": library["schema_version"],
                    "window": library["window"], "times": [float(times.min()), float(times.max())],
                    "frames": len(times), "donor_batches": sorted(donors),
                    "frame_keys": list(library["frames"][0]),
                    "teacher_counts": sorted({len(f["teacher_scores"]) for f in library["frames"]}),
                    "has_teacher_slots": all("teacher_slots" in f for f in library["frames"]),
                    "has_teacher_parent_ids": all("teacher_parent_ids" in f for f in library["frames"]),
                    "recorded_teacher_score_ranges": {
                        str(f["time"]): [min(f["teacher_scores"]), max(f["teacher_scores"])]
                        for f in (library["frames"][0], library["frames"][-1])}},
        "program": {"path": str(program_path.relative_to(repo)).replace("\\", "/"),
                    "sha256": sha256(program_path),
                    "settings": {k: program[k] for k in (
                        "program_id", "window", "reward_view", "teacher_neighbors", "native_rms_ratio",
                        "affinity_head_gradient", "additional_per_step_affinity_calls", "derivative_path")}},
        "existing_terminal_credit": {
            "path": str(credit_path.relative_to(repo)).replace("\\", "/"), "sha256": sha256(credit_path),
            "label_semantics": "Existing v4 uses step99 preselection joint-latent head; not decoded terminal rescore",
            "rows": len(credit), "independent_batches": int(credit.batch.nunique()),
            "nodes_with_live_ancestors_below3": int((credit.live_ancestors < 3).sum()),
            "nodes_with_live_ancestors_below6": int((credit.live_ancestors < 6).sum()),
            "live_ancestors_mean_by_time": {str(time): float(credit.loc[
                np.isclose(credit.score_time, time), "live_ancestors"].mean()) for time in (0, .1, .25, .49)}},
        "forward_root_information": {"definition": "Current-state roots at score clock; copy multiplicity is not independent evidence",
                                     "rows_audited": len(table), "summary": summary},
        "original_selection_protocol": protocols, "counterfactuals": counterfactuals,
        "sources": sources,
        "interpretation_limits": [
            "Forward root diversity and backward terminal live-ancestor counts are different quantities",
            "Uniform weights after resampling do not recover lost ancestor diversity",
            "Missing terminal outcomes of pruned paths are unknown, not low affinity",
            "Archived descendants include future Steer selection; not calibrated native continuation probabilities",
            "Counterfactual records here originate in joint, not the current single objective"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("audit_snapshot.json"))
    args = parser.parse_args()
    result = audit(args.repo.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2,
                                      allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"output": str(args.output), "root_information": result["forward_root_information"]["summary"],
                      "counterfactuals": result["counterfactuals"]}, ensure_ascii=False))
