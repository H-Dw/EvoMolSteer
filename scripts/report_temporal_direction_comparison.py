"""Consolidate frozen temporal trials and the latest completed role conditions.

Read-only inputs; no API, SSH, FLOWR generation, or expanded structure caches.
Screen and extension overlap is checked, not counted as new observations.
Role-runner arguments are ordered: the last completed revision of a condition
wins. Failed retries never replace an earlier completed condition.
"""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from check_temporal_role_condition import audit
from report_temporal_guidance_results import quality
from evomolsteer.io import digest, read_json, write_json
from evomolsteer.temporal_guidance.policy import program_signature
from evomolsteer.temporal_guidance.evidence import FINISH_POLICY
from evomolsteer.temporal_guidance.workflow import compile_recipe


def resolve(repo, name):
    # Older Windows manifests contain backslashes; permit their use on Linux.
    return (repo / str(name).replace("\\", "/")).resolve()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_quality(actual, recorded, context):
    for key, value in actual.items():
        require(np.isclose(value, recorded[key], rtol=0, atol=1e-12),
                f"Changed {context} statistic: {key}")


def rates(row):
    row = dict(row)
    for count, name in [("valid", "valid_rate"), ("PB_pass", "PB_rate"),
                        ("converged_energy", "energy_coverage"),
                        ("elite_valid", "elite_valid_rate")]:
        row[name] = row[count] / row["attempts"]
    return row


def collect(repo, numerical_config, frozen_report, runners, output, plot=True):
    inputs = {}

    def pin(path, expected=None):
        path = resolve(repo, path)
        sha = digest(path)
        require(expected is None or sha == expected, f"Frozen input changed: {path}")
        key = path.relative_to(repo).as_posix() if path.is_relative_to(repo) else str(path)
        inputs[key] = sha
        return path

    config = read_json(pin(numerical_config))
    teacher_audit = {}
    for name in ("static_reference", "temporal_reference"):
        reference = json.loads(gzip.decompress(pin(
            config[name], config["source_sha256"][name]).read_bytes()))
        teacher_audit[name] = {
            "path": config[name], "window": reference["window"],
            "label_semantics": reference.get("label_semantics"),
            "frames": len(reference["frames"]),
            "teacher_counts": sorted({len(f["teacher_endpoint_A"]) for f in reference["frames"]}),
            "identity_fields_in_all_frames": {
                key: all(key in f for f in reference["frames"])
                for key in ("teacher_node_ids", "teacher_path_ids")},
        }
    frozen = read_json(pin(frozen_report))
    for path, sha in frozen["source_sha256"].items():
        pin(path, sha)
    pin(__file__)
    root = resolve(repo, config["output"])
    state = read_json(pin(root / "run_status.json"))
    bank = read_json(pin(config["control_manifest"]))
    threshold = config["evaluation"]["elite_threshold"]
    frames, rows, definitions = {}, [], []
    for recipe, definition in config["recipes"].items():
        program, _ = compile_recipe(repo, config, recipe)
        signature = program_signature(program)
        definitions.append({"recipe": recipe, "signature": signature, **definition})
        for stage in ("screen", "confirmation"):
            task = state["jobs"].get(f"control_{recipe}_{stage}")
            if not task or task["status"] != "complete":
                continue
            result = task["result"]
            require(result["result"]["effective_signature"] == signature,
                    f"Reward signature mismatch: {recipe}")
            frame = pd.read_csv(pin(result["result"]["outcomes"]["gradient"]["metrics"]))
            batches = result["comparison"]["batches"]
            require(not frame.duplicated(["batch", "slot"]).any(), "Duplicate attempt")
            require(set(frame.batch) == set(batches), "Generation batch mismatch")
            stats = quality(frame, threshold)
            require(stats["attempts"] == stats["scored"], "Incomplete affinity scoring")
            row = next(r for r in frozen["numerical_results"]
                       if (r["recipe"], r["stage"]) == (recipe, stage))
            check_quality(stats, row, f"{recipe}/{stage}")
            rows.append(rates({**row, "signature": signature,
                               "independent_batches": len(batches)}))
            frames[recipe, stage] = frame

    # The extension contains the screen, not a second independent confirmation.
    for (recipe, stage), frame in frames.items():
        if stage == "confirmation":
            screen = frames[recipe, "screen"]
            overlap = frame.loc[frame.batch.isin(screen.batch)]
            pd.testing.assert_frame_equal(
                screen.sort_values(["batch", "slot"]).reset_index(drop=True),
                overlap.sort_values(["batch", "slot"]).reset_index(drop=True))

    controls, controls_rows = {}, []
    for label, arm in [("M2", "gradient"), ("unguided", "unguided")]:
        controls[label] = pd.read_csv(pin(bank["result"]["outcomes"][arm]["metrics"]))
    controls["historical_Steer"] = pd.read_parquet(pin(config["historical_metrics"]))
    for label, frame in controls.items():
        check_quality(quality(frame, threshold), frozen["controls"][label], label)
        require(not frame.duplicated(["batch", "slot"]).any(), "Duplicate control attempt")
        controls_rows.append(rates({"cohort": label, "scope": "all_available",
                                    "independent_batches": frame.batch.nunique(),
                                    **quality(frame, threshold)}))
        if label != "historical_Steer":
            for scope, batches in [("screen", config["screen_batches"]),
                                   ("confirmation", config["confirmation_batches"])]:
                part = frame.loc[frame.batch.isin(batches)]
                require(set(part.batch) == set(batches), "Missing matched control batch")
                controls_rows.append(rates({"cohort": label, "scope": scope,
                                            "independent_batches": len(batches),
                                            **quality(part, threshold)}))

    new_attempts, cohort_keys = 0, set()
    for key, task in state["jobs"].items():
        if not key.startswith("generation_") or task["status"] != "complete":
            continue
        result = task["result"]
        frame = pd.read_csv(pin(result["outcomes"]["gradient"]["metrics"]))
        require(len(frame) == result["outcomes"]["gradient"]["summary"]["n"],
                "Raw cohort count mismatch")
        for batch, slot in zip(frame.batch, frame.slot):
            index = (result["effective_signature"], int(batch), int(slot))
            require(index not in cohort_keys, "Repeated generation counted twice")
            cohort_keys.add(index)
        new_attempts += len(frame)
    require(new_attempts == frozen["new_attempts"], "Frozen generation count mismatch")

    selected_versions = {}
    for runner_path in runners:
        runner = read_json(pin(runner_path))
        plan = read_json(pin(runner["plan_config"]))
        folder = resolve(repo, plan["output"])
        status = read_json(pin(folder / "run_status.json"))
        for condition in plan["conditions"]:
            name = condition["id"]
            if status["jobs"].get("roles_" + name, {}).get("status") == "complete":
                selected_versions[name] = (runner_path, plan, folder, condition)
    expected_conditions = {c["id"] for c in config["conditions"]}
    require(set(selected_versions) == expected_conditions, "Missing full role conditions")
    role_rows, chain_rows, api_temperatures = [], [], set()
    numeric_by_recipe = {r["recipe"]: r for r in rows if r["stage"] == "screen"}
    for name in [c["id"] for c in config["conditions"]]:
        runner_path, plan, folder, condition = selected_versions[name]
        checked = audit(repo, runner_path, name)
        for path, sha in checked["source_sha256"].items():
            pin(path, sha)
        selection = read_json(pin(folder / "selections" / (name + ".json")))
        counts = Counter(c["recipe"] for c in selection["rows"])
        require(len(selection["rows"]) == plan["chains"], "Incomplete role chain count")
        winner = selection["selected"]["recipe"]
        winner_stats = numeric_by_recipe.get(winner)
        role_rows.append({
            "condition": name, "revision": folder.name,
            "evidence_protocol": plan.get("evidence_protocol", FINISH_POLICY),
            "Analyst_extensions": json.dumps(condition["extensions"]["Analyst"]),
            "Designer_extensions": json.dumps(condition["extensions"]["Designer"]),
            "chains": plan["chains"], "backend": "real_api", "model": "deepseek-flash",
            "selected": winner, "selected_votes": counts[winner],
            "selection_tied": list(counts.values()).count(max(counts.values())) > 1,
            "recipe_counts": json.dumps(dict(sorted(counts.items())), sort_keys=True),
            "API_action_calls": checked["API_action_calls"],
            "rejected_then_repaired_finishes": checked["rejected_then_repaired_finishes"],
            "selected_delta_vs_M2_screen": winner_stats["delta_vs_M2"] if winner_stats else 0.,
            "new_FLOWR_runs_for_same_signature": checked["new_FLOWR_runs_for_same_signature"],
        })
        for chain in checked["chains"]:
            chain_folder = folder / "calls" / name / f"chain_{chain['chain']:02d}"
            for role in ("Analyst", "Designer"):
                pin(chain_folder / f"{role}.json")
                pin(chain_folder / role / "wire_result.json")
                completed = read_json(chain_folder / role / "completed.json")
                for step in range(completed["steps"]):
                    request = read_json(pin(chain_folder / role / f"step_{step:02d}.request.json"))
                    require("temperature" in request["body"], "Unrecorded API temperature")
                    api_temperatures.add(request["body"]["temperature"])
            chain_rows.append({"condition": name, "revision": folder.name,
                               "hypotheses": len(chain["hypothesis_temporal_sources"]),
                               "curves_read": len(chain["actual_curve_sources_read"]),
                               "curves_cited": len(chain["designer_curves_cited"]),
                               **{k: v for k, v in chain.items() if k in
                                  {"chain", "recipe", "API_action_calls",
                                   "compiled_signature", "rejected_then_repaired_finishes"}}})

    batch_rows, extension_rows = [], []
    m2 = controls["M2"]
    for (recipe, stage), frame in frames.items():
        joined = frame.merge(m2, on=["batch", "slot"], suffixes=["_candidate", "_M2"],
                             validate="one_to_one")
        require(len(joined) == len(frame), "Missing paired random input")
        for batch, part in joined.groupby("batch"):
            batch_rows.append({"recipe": recipe, "stage": stage, "batch": int(batch),
                               "attempts": len(part),
                               "delta_pic50": float((part.pic50_on_rescore_candidate -
                                                     part.pic50_on_rescore_M2).mean())})
        if stage == "confirmation":
            extra = frame.loc[~frame.batch.isin(config["screen_batches"])]
            extra_control = m2.loc[m2.batch.isin(extra.batch)]
            q, cq = quality(extra, threshold), quality(extra_control, threshold)
            extension_rows.append({"recipe": recipe, "added_batches": extra.batch.nunique(),
                                   **rates(q), "delta_vs_M2": q["mean_pic50"] - cq["mean_pic50"],
                                   "M2_strain_p90": cq["strain_p90"],
                                   "strain_p90_change_fraction": q["strain_p90"] / cq["strain_p90"] - 1,
                                   "independent_confirmation": False})

    numerical = pd.DataFrame(rows).sort_values(["stage", "recipe"])
    roles = pd.DataFrame(role_rows)
    chains = pd.DataFrame(chain_rows)
    eligible = Counter()
    for condition in config["conditions"]:
        for recipe in condition["recipes"]:
            eligible[recipe] += int(roles.loc[roles.condition.eq(condition["id"]), "chains"].iloc[0])
    require(len(cohort_keys) == new_attempts, "Generated sample deduplication failed")
    report = {
        "schema_version": "temporal-direction-comparison-1.0",
        "scope": "W/S/C temporal exploration; latest complete conditions, not all historical campaigns",
        "distinct_numerical_programs": len(config["recipes"]),
        "numerical_summary_rows_with_overlap": len(rows),
        "completed_generation_cohorts": frozen["completed_remote_cohorts"],
        "new_attempts_unique_by_signature_batch_slot": new_attempts,
        "matched_reused_controls": {k: len(v) for k, v in controls.items()},
        "complete_Agent_conditions": len(role_rows),
        "complete_role_pairs": len(chain_rows), "complete_role_outputs": 2 * len(chain_rows),
        "majority_or_tiebreak_selected_programs": sorted(set(roles.selected)),
        "all_chain_recipe_counts": dict(sorted(Counter(chains.recipe).items())),
        "eligible_chain_count_by_recipe": dict(sorted(eligible.items())),
        "teacher_library_audit": teacher_audit,
        "model": "deepseek-flash", "API_temperatures": sorted(api_temperatures),
        "master_seed": config["master_seed"], "steps": config["steps"],
        "learning_window": frozen["learning_window"],
        "controlled_steps": frozen["controlled_steps"], "native_suffix_steps": frozen["native_suffix_steps"],
        "elite_threshold": threshold, "S3_paired_response": frozen["S3_paired_response"],
        "S3_vs_M2_quantiles": frozen["S3_vs_M2_quantiles"],
        "exploratory_timing_contrasts": frozen["exploratory_timing_contrasts"],
        "actual_injection": frozen["delivery"], "recipe_definitions": definitions,
        "checks": {"frozen_input_sha256": "passed", "metric_recalculation": "passed",
                   "extension_screen_overlap": "passed", "attempt_deduplication": "passed",
                   "role_scientific_delivery_and_API_binding": "passed",
                   "compiled_reward_signatures": "passed"},
        "limitations": [
            "Full 400 cohorts include their 200 screen; no independent confirmation.",
            "Only four/eight RNG batches per numerical program; reused role votes are not molecular replicates.",
            "Latest complete prompt conditions mix delivery revisions v2/v3; not a clean same-interface module ablation.",
            "The recipes were host preregistered; an Agent selection is not a newly designed numerical reward.",
            "Frozen historical Steer has a different online budget and dependent clones; do not pair its seeds.",
            "Observed clock forecasts are endpoint predictions, not noisy states or decoded terminal structures.",
            "Energy summaries require converged MMFF; report energy coverage separately from all-attempt affinity.",
        ],
        "source_sha256": inputs,
    }
    output.mkdir(parents=True, exist_ok=True)
    numerical.to_csv(output / "numerical_comparison.csv", index=False)
    pd.DataFrame(controls_rows).to_csv(output / "controls_by_scope.csv", index=False)
    roles.to_csv(output / "latest_Agent_conditions.csv", index=False)
    chains.to_csv(output / "latest_role_chains.csv", index=False)
    pd.DataFrame(batch_rows).to_csv(output / "paired_batch_deltas.csv", index=False)
    pd.DataFrame(extension_rows).to_csv(output / "added_batch_quality.csv", index=False)
    pd.DataFrame(frozen["S3_vs_M2_quantiles"]).to_csv(output / "S3_quantile_changes.csv", index=False)
    write_json(output / "comparison.json", report)
    if plot:
        import matplotlib
        matplotlib.use("Agg")
        matplotlib.rcParams["svg.hashsalt"] = "temporal-direction-comparison"
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8),
                                 gridspec_kw={"width_ratios": [1.7, 1]})
        for ax, stage, title in zip(axes, ["screen", "confirmation"],
                                  ["Screen: 200 attempts / 4 RNG batches",
                                   "Extension: 400 attempts / 8 RNG batches"]):
            part = numerical.loc[numerical.stage.eq(stage)].sort_values("recipe")
            for j, row in enumerate(part.to_dict("records")):
                color = "#00816A" if row["recipe"].startswith("S3") else "#365A82"
                ax.plot([row["M2_simultaneous_CI_low"], row["M2_simultaneous_CI_high"]],
                        [j, j], color=color, lw=2)
                ax.plot(row["delta_vs_M2"], j, "o", color=color, ms=6)
            ax.set_yticks(range(len(part)), [x.split("_")[0] for x in part.recipe])
            ax.invert_yaxis()
            ax.axvline(0, color="black", lw=.8)
            ax.axvline(-config["evaluation"]["affinity_noninferiority_margin"],
                       color="#777777", lw=.8, ls="--")
            ax.set(xlabel="Mean final predicted pIC50 change vs matched M2", title=title)
            ax.grid(axis="x", alpha=.18)
        fig.text(.5, .02, "Simultaneous batch-bootstrap intervals (family = 9). Extension includes screen; no independent confirmation.",
                 ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .06, 1, 1))
        fig.savefig(output / "affinity_effects.png", dpi=180)
        svg = output / "affinity_effects.svg"
        fig.savefig(svg, metadata={"Date": None})
        svg.write_text("\n".join(line.rstrip() for line in svg.read_text(
            encoding="utf-8").splitlines()) + "\n", encoding="utf-8", newline="\n")
        plt.close(fig)
    return {k: report[k] for k in ["distinct_numerical_programs", "new_attempts_unique_by_signature_batch_slot",
                                  "complete_Agent_conditions", "complete_role_pairs", "checks"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--numerical-config", required=True)
    parser.add_argument("--frozen-report", required=True)
    parser.add_argument("--role-runner", action="append", required=True,
                        help="Ordered revisions; last complete condition wins")
    parser.add_argument("--output", required=True)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    print(json.dumps(collect(repo, args.numerical_config, args.frozen_report,
                             args.role_runner, resolve(repo, args.output), not args.no_plot),
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
