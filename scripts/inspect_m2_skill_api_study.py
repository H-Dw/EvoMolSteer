"""Read-only inspection of partial real-API studies; never launches inference.

Example:
  python scripts/inspect_m2_skill_api_study.py --study docs/... --output docs/.../inspection --elite-threshold VALUE
Independent generation policies are summarized once, even if many LLM chains
retain the same program. Unfinished conditions are not labeled as ablations.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evomolsteer.generation.terminal_statistics import converged_energy
from evomolsteer.io import digest, read_json, write_json


def quality(path, threshold):
    frame = pd.read_csv(path)
    required = {"batch", "slot", "pic50_on_rescore", "valid_connected", "pb_fast_pass", "energy_status"}
    if not required.issubset(frame) or frame.duplicated(["batch", "slot"]).any():
        raise ValueError("Incomplete or duplicate identified candidates")
    if not np.isfinite(frame.pic50_on_rescore).all():
        raise ValueError("All-attempt affinity coverage required")
    frame["energy_status"] = frame.energy_status.fillna("not_applicable")
    valid = frame.loc[frame.valid_connected]
    energy = converged_energy(frame)
    row = {
        "n": len(frame), "batches": sorted(frame.batch.unique().tolist()),
        "mean_pic50_all": float(frame.pic50_on_rescore.mean()),
        "mean_pic50_valid": float(valid.pic50_on_rescore.mean()),
        "p95_pic50_all": float(frame.pic50_on_rescore.quantile(.95)),
        "max_pic50_all": float(frame.pic50_on_rescore.max()),
        "valid": int(frame.valid_connected.sum()), "pb_fast_pass": int(frame.pb_fast_pass.sum()),
        "converged_energy_n": len(energy),
        "strain_per_heavy_median": float(energy.median()) if len(energy) else None,
        "strain_per_heavy_p90": float(energy.quantile(.9)) if len(energy) else None,
        "unique_valid_graphs": int(valid.smiles.nunique()),
        "mean_pic50_unique_first_occurrence": float(valid.drop_duplicates("smiles").pic50_on_rescore.mean()),
        "elite_valid": int((valid.pic50_on_rescore >= threshold).sum()),
        "source": str(path), "source_sha256": digest(path),
    }
    return frame, row


def paired(left, right):
    merged = left[["batch", "slot", "pic50_on_rescore"]].merge(
        right[["batch", "slot", "pic50_on_rescore"]], on=["batch", "slot"],
        validate="one_to_one", suffixes=("_left", "_right"))
    if len(merged) != len(left) or len(merged) != len(right):
        raise ValueError("Incomplete matched generation cohorts")
    merged["delta"] = merged.pic50_on_rescore_left - merged.pic50_on_rescore_right
    grouped = merged.groupby("batch").delta
    counts = grouped.size()
    if counts.nunique() != 1:
        raise ValueError("Inspection expects equally sized generation batches")
    batch = grouped.mean()
    if len(batch) < 2:
        raise ValueError("At least two generation batches required")
    values = batch.to_numpy()
    rng = np.random.default_rng(42)
    bootstrap = values[rng.integers(0, len(values), (10000, len(values)))].mean(1)
    result = {
        "delta_mean_pic50": float(merged.delta.mean()), "generation_batches": len(batch),
        "batch_deltas": {str(k): float(v) for k, v in batch.items()},
        "positive_batch_fraction": float((values > 0).mean()),
        "descriptive_batch_bootstrap_CI95": np.quantile(bootstrap, [.025, .975]).tolist(),
        "bootstrap_replicates": 10000, "bootstrap_seed": 42,
        "limitation": "Few generation batches; bootstrap is descriptive. Conditional on fixed target, model, master seed and teacher bank.",
    }
    if len(values) <= 16:
        signs = np.array(list(itertools.product([-1, 1], repeat=len(values))))
        result["two_sided_exact_batch_signflip_p"] = float(
            (np.abs((signs * values).mean(1)) >= abs(values.mean()) - 1e-12).mean())
        result["signflip_assumption"] = "Symmetric batch differences/exchangeable paired labels under the null; not independent molecule-level tests."
    return result


def feature_response(base, trial):
    left, right = pd.read_parquet(base), pd.read_parquet(trial)
    keys = ["batch", "score_time", "state_time"]
    left, right = left.sort_values(keys).reset_index(drop=True), right.sort_values(keys).reset_index(drop=True)
    if not left[keys].equals(right[keys]):
        raise ValueError("Forecast clocks and batches must match exactly")
    fields = [c for c in left if c not in keys]
    if fields != [c for c in right if c not in keys]:
        raise ValueError("Feature definitions differ")
    rows = []
    for batch, group in left.groupby("batch"):
        index = group.index
        time = group.score_time.to_numpy()
        if len(time) < 2 or np.any(np.diff(time) <= 0):
            raise ValueError("Strictly increasing observed score clock required")
        for feature in fields:
            delta = (right.loc[index, feature] - left.loc[index, feature]).to_numpy()
            if not np.isfinite(delta).all():
                raise ValueError("Finite geometry required")
            rows.append({"batch": int(batch), "feature": feature,
                "observed_score_start": float(time[0]), "observed_score_end": float(time[-1]),
                "window_mean_delta": float(np.trapezoid(delta, time) / (time[-1] - time[0])),
                "last_observed_delta": float(delta[-1]),
                "first_observed_delta": float(delta[0]),
                "change_rate_delta": float((delta[-1] - delta[0]) / (time[-1] - time[0]))})
    frame = pd.DataFrame(rows)
    overview = frame.groupby("feature").agg(
        window_mean_delta=("window_mean_delta", "mean"),
        positive_batch_fraction=("window_mean_delta", lambda x: float((x > 0).mean())),
        last_observed_delta=("last_observed_delta", "mean"),
        change_rate_delta=("change_rate_delta", "mean")).reset_index()
    return frame, overview, {
        "base_sha256": digest(base), "trial_sha256": digest(trial), "rows_per_policy": len(left),
        "features": len(fields), "nonidentical_values": int((left[fields] != right[fields]).sum().sum()),
        "representation": "Pre-injection predicted endpoint features; these are not actual intermediate x_t coordinates.",
        "integration": "Trapezoid on the observed score clock, normalized by its duration; no stage bins or extrapolation.",
        "limitation": "Batch means cannot reconstruct individual trajectories; this is trial-minus-incumbent response, not a new Steer enrichment analysis.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--elite-threshold", required=True, type=float)
    parser.add_argument("--remote-snapshot")
    args = parser.parse_args()
    study, out = Path(args.study), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    state, activation = read_json(study / "run_status.json"), read_json(study / "activation.json")
    records = study / "transport_records"
    completed = state.get("completed_tasks", [])
    controls = [name for name in completed if (records / (name + ".unguided.metrics.csv")).exists()]
    if len(controls) != 1:
        raise ValueError("Exactly one completed matched M2/native control job expected")
    control = controls[0]
    frames, outcomes, sources, deliveries = {}, {}, {}, {}
    policies = [("native", control, "unguided"), ("M2", control, "gradient")]
    policies += [(name, name, "gradient") for name in completed if name != control]
    for label, name, arm in policies:
        frames[label], outcomes[label] = quality(records / f"{name}.{arm}.metrics.csv", args.elite_threshold)
        sources[name] = {key: read_json(records / (name + suffix)) for key, suffix in [
            ("remote", ".remote.json"), ("execution", ".execution_audit.json"),
            ("archive_verification", ".archive_verification.json"), ("retirement", ".retirement.json")]}
        if arm == "gradient":
            deliveries[label] = read_json(records / (name + ".delivery.json"))
    signatures = sources[control]["remote"]["initial_state_signatures"]
    for label, name, arm in policies:
        for batch in outcomes[label]["batches"]:
            sig = sources[name]["remote"]["initial_state_signatures"][f"{arm}/batch_{batch}"]
            if sig != signatures[f"unguided/batch_{batch}"]:
                raise ValueError("Unmatched initial state signature")
    comparisons = {"M2_minus_native": paired(frames["M2"], frames["native"])}
    for label, name, arm in policies[2:]:
        comparisons[label + "_minus_M2"] = paired(frames[label], frames["M2"])
        comparisons[label + "_minus_native"] = paired(frames[label], frames["native"])
        batch, feature, provenance = feature_response(
            records / (control + ".forecast_features.parquet"), records / (name + ".forecast_features.parquet"))
        batch.to_csv(out / (name + ".feature_response_by_batch.csv"), index=False)
        feature.to_csv(out / (name + ".feature_response.csv"), index=False)
        write_json(out / (name + ".feature_response.semantics.json"), provenance)
    chains = []
    for folder in sorted((study / "calls").glob("*/*")):
        row = {"condition": folder.parent.name, "chain": folder.name,
            "analyst_complete": (folder / "Analyst.json").exists(),
            "designer_complete": (folder / "Designer.json").exists(),
            "invalid_attempts": len(list(folder.glob("*.invalid.receipt.json")))}
        if row["designer_complete"]:
            design = read_json(folder / "Designer.json")
            row.update(decision=design["decision"], edit=design["edit"])
            row["compiler"] = read_json(folder / "compiler.json")
        chains.append(row)
    receipts = [read_json(f) for f in (study / "calls").rglob("*.receipt.json")]
    audit = {
        "status": state["status"], "updated_unix": state["updated_unix"],
        "error_type": state.get("error_type"), "error_first_line": str(state.get("error", "")).splitlines()[0] if state.get("error") else None,
        "planned_rows_including_controls": len(activation["conditions"]),
        "planned_instruction_conditions": sum(int(r["planned_role_chains"]) > 0 for r in activation["conditions"]),
        "chains": chains,
        "completed_conditions": [key for key in {r["condition"] for r in chains}
            if sum(r["analyst_complete"] and r["designer_complete"] for r in chains if r["condition"] == key) == activation["chain_replicates"]],
        "review_files": [str(f) for f in (study / "reviews").glob("*.json")],
        "valid_role_responses": sum(r.get("valid") is not False for r in receipts),
        "failed_interface_attempts": sum(r.get("valid") is False for r in receipts),
        "returned_models": sorted({r["returned_model"] for r in receipts}),
        "scientific_role_chains_are_not_generation_replicates": True,
    }
    result = {"schema_version": "m2-api-partial-inspection-1.0", "status_audit": audit,
        "outcomes": outcomes, "paired_comparisons": comparisons, "delivery": deliveries,
        "initial_state_signatures_matched": True, "elite_threshold": args.elite_threshold,
        "initial_input_hashes_still_match": {f: digest(f) == sha for f, sha in activation["bound_inputs"].items()},
        "provenance": sources, "script_sha256": digest(__file__),
        "source_status_sha256": digest(study / "run_status.json"),
        "remote_snapshot": read_json(args.remote_snapshot) if args.remote_snapshot else None,
        "primary": "FLOWR predicted pIC50, not measured binding affinity",
        "secondary": "Same-graph local MMFF94s relaxation relief per heavy atom, not physical binding free energy"}
    write_json(out / "inspection.json", result)
    pd.DataFrame([{"policy": label, **row} for label, row in outcomes.items()]).to_csv(out / "quality_table.csv", index=False)
    print(json.dumps({"status": audit["status"], "unique_generated_policies": len(policies),
        "completed_conditions": audit["completed_conditions"], "output": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
