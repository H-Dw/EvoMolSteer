"""Read-only audit of a completed NC campaign; no API or generation calls.

Writes compact parameter/agent/dose tables and algebraic normalization probes.
The probes check the implemented control mathematics, not FLOWR performance.
"""
import argparse
import ast
import csv
import hashlib
import json
import textwrap
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalization_probe():
    # d[-delta^2(sqrt(1+q/delta^2)-1)]/dx is a positive scalar
    # times -dq/dx. Per-sample unit-RMS removes that scalar.
    rng = np.random.default_rng(42)
    residual = rng.normal(size=(4, 20, 3))
    q = np.square(residual).sum(-1).mean(-1)
    grad_q = 2 * residual / residual.shape[1]
    rms = lambda a: np.sqrt(np.square(a).sum(-1).mean(-1))[:, None, None]
    directions = []
    for delta in (0.1, 1.0, 10.0):
        grad = -grad_q / (2 * np.sqrt(1 + q / delta**2))[:, None, None]
        directions.append(grad / np.maximum(rms(grad), 1e-10))
    h, n = directions[1], rng.normal(size=residual.shape)
    n = n / rms(n)
    mixture = 0.85 * h + 0.15 * n
    shifts = [s * mixture / rms(s * mixture) * 0.015 for s in (0.1, 1, 10)]
    return {
        "kind": "algebraic offline probe; no actual-model or performance claim",
        "deltas_A": [0.1, 1.0, 10.0],
        "max_normalized_geometry_gradient_difference": float(max(np.abs(d - directions[1]).max() for d in directions)),
        "common_weight_multipliers": [0.1, 1, 10],
        "max_fixed_dose_displacement_difference": float(max(np.abs(s - shifts[1]).max() for s in shifts)),
        "two_term_max_turn_degrees_before_clipping": {
            "type_0.15_geometry_0.85": float(np.degrees(np.arcsin(0.15 / 0.85))),
            "type_0.4_geometry_0.6": float(np.degrees(np.arcsin(0.4 / 0.6))),
        },
        "assumptions": "fixed assignment; finite nonzero derivatives above normalization floors; same dose",
    }


def native_source_audit(repo, master):
    cache = repo / "data/analysis_cache" / Path(master["campaign_study"]).name / "workflow_audit"
    fm, integrator = cache / "fm_pocket.py", cache / "integrator.py"
    if not fm.exists() or not integrator.exists():
        return {"status": "source cache absent; remote source was not queried"}
    source = fm.read_text(encoding="utf-8")
    node = next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef) and n.name == "_generate")
    block = textwrap.dedent("\n".join(source.splitlines()[node.lineno - 1:node.end_lineno])) + "\n"
    observed = read(repo / master["generated_root"] / "R18/source_audit.json")
    block_sha = hashlib.sha256(block.encode()).hexdigest()
    # Exact native probability rule, illustration only: perfect prediction of
    # the CURRENT category still leaves 14 other categories with 0.01 each.
    k, dt, eta = 15, 0.01, 1.0
    noise_floor = (k - 1) * dt * eta
    return {
        "fm_source_sha256": sha(fm), "integrator_source_sha256": sha(integrator),
        "generate_sha256": block_sha,
        "generate_matches_executed_R18": block_sha == observed["upstream_native_sha256"],
        "integrator_limit": "Integrator file retrieved read-only on audit day; no historical file hash was saved during R18. Runtime hparams are independently recorded.",
        "illustrative_categorical_probability": {"K": k, "dt": dt, "noise": eta,
            "leave_current_when_prediction_is_perfect": noise_floor,
            "never_switch_in_49_noisy_steps_under_perfect_prediction": (1 - noise_floor)**49,
            "not_a_final_retention_probability": True},
        "illustrative_coordinate_random_vector_RMS_A": {
            str(t): float(dt * np.sqrt(3 * 2 * (1 / (t + 0.01)) * 0.2)) for t in (0.0, 0.1, 0.49, 0.5)
        },
        "coordinate_noise_limit": "Expected random term only, coord_scale=1; excludes SDE score drift and realized trajectory correlations",
    }


def audit(repo, config, output):
    repo, config, output = Path(repo).resolve(), Path(config), Path(output)
    master = read(config)
    study = repo / master["campaign_study"]
    state = read(study / "state.json")
    rows, settings, hashes = [], {}, {str(config): sha(config)}
    for entry in state["rounds"]:
        name = entry["round"]
        folder = study / name
        cfg = read(repo / master["config_dir"] / (name + ".json"))
        row = {"round": name, "status": entry.get("status", "generated")}
        for role in ("Analyst", "Designer"):
            request_path = folder / (role + ".request.json")
            request, receipt = read(request_path), read(folder / (role + ".receipt.json"))
            assert receipt["request_sha256"] == sha(request_path)
            assert receipt["response_sha256"] == sha(folder / (role + ".json"))
            contents = []
            for item in request["body"]["messages"]:
                path = folder / item["content_relative_path"]
                assert sha(path) == item["content_sha256"]
                contents.append(path.read_text(encoding="utf-8"))
            row[role.lower() + "_prompt_tokens"] = receipt["usage"]["prompt_tokens"]
            row[role.lower() + "_repairs"] = len(receipt.get("interface_repairs", []))
            assert receipt["requested_model"] == receipt["returned_model"] == "deepseek-flash"
            assert request["body"]["temperature"] == 0
            hashes[str(request_path.relative_to(repo))] = sha(request_path)
            if role == "Analyst":
                user = json.loads(contents[1])
                row["allowed_patch_count"] = len(request["response_schema"]["properties"]["patch_id"]["enum"])
                row["history_rounds"] = len(user["feedback"]["history"])
                row["candidate_summaries"] = len(user["candidate_summaries"])
            else:
                assert request["response_schema"]["properties"]["execution_contract"]["const"] == cfg["execution_contract"]
                row["designer_contract_fixed"] = True
        for key, value in cfg["execution_contract"].items():
            settings.setdefault(key, set()).add(json.dumps(value, sort_keys=True))
        if entry.get("metrics"):
            base = repo / master["generated_root"] / name / name
            path = base / "guidance_audit.json"
            actual = read(path)
            hashes[str(path.relative_to(repo))] = sha(path)
            active = [r for r in actual["steps"] if r["active"]]
            row.update(active_steps=len(active), geometry_points=len(actual["program"]["atom_slots"]),
                       typed_points=len(actual["program"]["typed_slots"]),
                       cumulative_RMS_A=actual["mean_cumulative_control_rms_A"],
                       injected_RMS_A_mean=float(np.mean([r["injected_rms_A_mean"] for r in active])),
                       assigned_probability_first=active[0]["typed_correct_probability"],
                       assigned_probability_last=active[-1]["typed_correct_probability"],
                       mean_pic50=entry["metrics"]["mean_all_pic50"],
                       max_qualified_pic50=entry["metrics"]["best_qualified_pic50"])
        rows.append(row)
    output.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (output / "agent_and_control_audit.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "schema": "nc-workflow-audit-1", "rounds": len(rows),
        "singleton_analyst_rounds": [r["round"] for r in rows if r["allowed_patch_count"] == 1],
        "designer_contract_fixed_rounds": sum(r["designer_contract_fixed"] for r in rows),
        "parameter_values": {k: [json.loads(v) for v in sorted(values)] for k, values in settings.items()},
        "normalization_probe": normalization_probe(),
        "native_source_audit": native_source_audit(repo, master),
        "source_sha256": hashes,
        "limits": ["No new LLM call or FLOWR inference", "Valid delivery is not scientific claim validation",
                   "Existing mean telemetry cannot recover per-sample clipping or gradient conflict",
                   "Probe checks algebra, not target-specific responsiveness"],
    }
    (output / "workflow_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "rounds": len(rows),
                      "singleton_analyst_rounds": report["singleton_analyst_rounds"],
                      "normalization_probe": report["normalization_probe"]}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    audit(args.repo, args.config, args.output)
