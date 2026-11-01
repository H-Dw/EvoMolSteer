"""Read saved candidate metrics; compare completed five-round tasks without inference."""
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
OLD = EXPERIMENTS / "20261101_all_results_summary"
METRICS = ("rmsd", "pic50", "sa")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def mean(values):
    return math.fsum(values) / len(values) if values else None


def candidate_views(path):
    result = defaultdict(list)
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            result[(row["teacher"], row["group"])].append(row)
    return result


def validate_rows(rows, stats, population):
    assert len(rows) == population
    assert len({(r["seed"], r["slot"]) for r in rows}) == population
    for metric in METRICS:
        values = [r[metric] for r in rows if finite(r.get(metric))]
        assert len(values) == stats[metric]["n"]
        assert math.isclose(mean(values), stats[metric]["mean"], abs_tol=1e-12)


def paired(old_rows, new_rows):
    old_rows = {(r["seed"], r["slot"]): r for r in old_rows}
    new_rows = {(r["seed"], r["slot"]): r for r in new_rows}
    assert old_rows.keys() == new_rows.keys()
    result = {}
    for metric in METRICS:
        differences = []
        by_seed = defaultdict(list)
        for key, old_row in old_rows.items():
            a, b = old_row.get(metric), new_rows[key].get(metric)
            if finite(a) and finite(b):
                difference = b - a
                differences.append(difference)
                by_seed[key[0]].append(difference)
        signed = [(-d if metric == "pic50" else d) for d in differences]
        result[metric] = dict(
            paired_finite=len(differences), missing_pairs=len(old_rows) - len(differences),
            mean_new_minus_old=mean(differences),
            improved=sum(d < 0 for d in signed), worse=sum(d > 0 for d in signed),
            equal=sum(d == 0 for d in signed),
            per_seed_mean_new_minus_old={str(k): mean(v) for k, v in sorted(by_seed.items())},
        )
    return result


def main():
    inputs = [
        OLD / "完整任务最优轮对比.json",
        OLD / "五轮原测与严格工具重测对比.json",
        OLD / "RMSD对齐与基线迭代读数.json",
        OLD / "raw/all_results_deduplicated.json",
        OLD / "raw/candidate_rows.jsonl.gz",
        HERE / "final_results.json",
        HERE / "monitoring/20261102_0134/final_results_raw.json",
        HERE / "monitoring/20261102_0134/candidate_metrics.jsonl.gz",
    ]
    old_selected, old_tasks, aligned, old_raw = map(read, inputs[:4])
    current, current_raw = read(inputs[5]), read(inputs[6])
    old_views, new_views = candidate_views(inputs[4]), candidate_views(inputs[7])
    for name in ("rdkit", "sascorer_sha256", "sa_fragments_sha256"):
        assert old_raw[name] == current_raw[name], name
    for label, expected in old_selected["source_sha256"].items():
        path = EXPERIMENTS / Path(label.replace("\\", "/"))
        assert path.is_file() and sha(path) == expected, str(path)
    assert sha(inputs[6]) == current["raw_review_sha256"]
    old_groups = {g["folder"]: g for g in old_raw["groups"]}
    new_groups = {(g["teacher"], g["id"]): g for g in current["groups"]}
    current_teachers = {t["teacher"]: t for t in current["teachers"]}
    raw_teachers = {t["teacher"]: t for t in current_raw["teachers"]}
    task_lookup = {(t["suite"], t["id"]): t for t in old_tasks["tasks"]}
    rows, comparisons, trajectories = [], [], []
    for selected in old_selected["selected"]:
        assert selected["rounds_completed"] == 5
        g = selected["best_configuration"]
        source_rows = old_views[(g["teacher"], g["folder"])]
        validate_rows(source_rows, g["stats"], 50)
        t = current_teachers[g["teacher"]]
        best = new_groups[(g["teacher"], t["best"])]
        new_rows = new_views[(g["teacher"], t["best"])]
        validate_rows(new_rows, best["metrics"], 50)
        assert g["condition_sha256"] == t["domain"]["condition_sha256"]
        assert g["stats"]["rmsd"]["excluded"] == best["metrics"]["rmsd"]["excluded"] == 0
        task = task_lookup[(selected["suite"], selected["id"])]
        rows.append(dict(
            teacher=g["teacher"], strategy=selected["strategy"], suite=selected["suite"],
            skills=selected["skills"], selected_round=selected["selected_round"],
            schedule=g["schedule"], population=50, built=g["built"],
            stats={k: g["stats"][k] for k in METRICS}, source=g["folder"],
            record_sha256=g["record_sha256"], terminal_sha256=g["terminal_sha256"],
            attempts=task["attempts"], action_accounting=selected["full_run_action_accounting"],
        ))
        comparisons.append(dict(
            teacher=g["teacher"], prior_strategy=selected["strategy"], prior_suite=selected["suite"],
            mean_difference_new_minus_old={k: best["metrics"][k]["mean"] - g["stats"][k]["mean"] for k in METRICS},
            rmsd_percent_change=100 * (best["metrics"]["rmsd"]["mean"] / g["stats"]["rmsd"]["mean"] - 1),
            paired_same_seed_slot=paired(source_rows, new_rows),
        ))
        trajectories.append(dict(
            teacher=g["teacher"], strategy=selected["strategy"], skills=selected["skills"],
            rounds=selected["round_performance"],
        ))
    current_calibration, previous_calibration = [], []
    for teacher, t in current_teachers.items():
        best, root = [new_groups[(teacher, t[k])] for k in ("best", "baseline")]
        best_rows, root_rows = [new_views[(teacher, t[k])] for k in ("best", "baseline")]
        best_round = next(r["iteration"] for r in t["rounds"] if t["best"] in r["groups"])
        rows.append(dict(
            teacher=teacher, strategy="六窗口同步引导", suite="六窗口五轮", skills="2.10.1",
            selected_round=best_round, schedule=best["schedule"], population=50, built=best["built"],
            stats={k: best["metrics"][k] for k in METRICS}, source=t["root"] + "/" + t["best"],
            record_sha256=best["record_sha256"], terminal_sha256=best["terminal_sha256"],
            attempts=t["budget"], action_accounting=dict(formal_groups=15, formal_attempts=750),
        ))
        current_calibration.append(dict(
            teacher=teacher, schedule=root["schedule"], stats=root["metrics"],
            rmsd_percent_change_to_best=100 * (best["metrics"]["rmsd"]["mean"] / root["metrics"]["rmsd"]["mean"] - 1),
            paired_same_seed_slot=paired(root_rows, best_rows),
        ))
        rounds = []
        for r in t["rounds"]:
            complete = [new_groups[(teacher, p)] for p in r["groups"]]
            assert len(complete) == 3 and all(g["population"] == 50 for g in complete)
            rounds.append(dict(
                iteration=r["iteration"],
                best_new_complete50_ranking_rmsd=min(g["metrics"]["rmsd"]["filtered_mean"] for g in complete),
                rebase=r["rebase"],
            ))
        trajectories.append(dict(teacher=teacher, strategy="六窗口同步引导", skills="2.10.1", rounds=rounds))
    for node in aligned["baseline_calibration"]:
        best = node["best"]
        teacher = best["teacher"]
        new_t = current_teachers[teacher]
        new_root = new_groups[(teacher, new_t["baseline"])]
        new_best = new_groups[(teacher, new_t["best"])]
        prior_best = min(g["stats"]["rmsd"]["mean"] for g in old_groups.values()
                         if g["teacher"] == teacher and f"/jobs/{node['job']}/" in g["folder"] and g["population"] == 50)
        previous_calibration.append(dict(
            teacher=teacher, prior_initial_schedule=best["schedule"],
            prior_initial_rmsd=best["absolute_rmsd_A"]["mean"],
            prior_best_including_calibration=prior_best,
            current_initial_rmsd=new_root["metrics"]["rmsd"]["mean"],
            current_initial_percent_change=100 * (new_root["metrics"]["rmsd"]["mean"] / best["absolute_rmsd_A"]["mean"] - 1),
            current_final_percent_change_vs_prior_best_including_calibration=100 * (new_best["metrics"]["rmsd"]["mean"] / prior_best - 1),
        ))
    # Identical constant strengths make window partitioning irrelevant. This comparison
    # still combines the earlier gradient phase and earlier chemical state injection.
    constant_profiles = []
    for teacher, t in current_teachers.items():
        job = "01_baseline_6pyy" if teacher.startswith("6pyy") else "02_baseline_3w54"
        for g in old_groups.values():
            if f"/jobs/{job}/" not in g["folder"] or "/calibration/" not in g["folder"]:
                continue
            if len(set(g["schedule"])) != 1 or g["population"] not in (10, 50):
                continue
            matches = [n for n in current["groups"] if n["teacher"] == teacher and n["stage"] == "calibration"
                       and n["population"] == g["population"] and set(n["schedule"]) == set(g["schedule"])]
            for n in matches:
                old_rows = old_views[(teacher, g["folder"])]
                new_rows = new_views[(teacher, n["id"])]
                validate_rows(old_rows, g["stats"], g["population"])
                validate_rows(new_rows, n["metrics"], n["population"])
                constant_profiles.append(dict(
                    teacher=teacher, constant_scale=g["schedule"][0], population=g["population"],
                    prior_rmsd=g["stats"]["rmsd"]["mean"], current_rmsd=n["metrics"]["rmsd"]["mean"],
                    prior_source=g["folder"], current_source=t["root"] + "/" + n["id"],
                    paired_same_seed_slot=paired(old_rows, new_rows),
                ))
    prompt_readouts = []
    for teacher, t in raw_teachers.items():
        for r in t["rounds"]:
            prompt_readouts.append(dict(
                teacher=teacher, iteration=r["iteration"],
                proposal_reason=r["proposal"]["reason"], synthesis_reason=r["synthesis"]["reason"],
                selected=r["synthesis"]["selected"],
                exploration_tiers=[p["exploration_tier"] for p in r["proposal"]["proposals"]
                                   if p["id"] in r["synthesis"]["selected"]],
            ))
    result = dict(
        timestamp=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        policy="Completed five-round jobs only; one best complete50 configuration per job, same configuration for all metrics",
        rmsd_definition=current_raw["rmsd_definition"],
        historical_rmsd_calculation="Recomputed from saved coordinates in double precision; current saved scalar differences at sub-microangstrom precision do not explain changes",
        missing_policy="SA measured only on built molecules; paired SA uses only both-finite pairs; RMSD/pIC50 retain all 50",
        comparability=dict(same_condition_sha256=True, same_sa_implementation=True,
                           same_seed_slot_sets=True, ligand_fit=False,
                           independent_validation=False, equal_search_cost=False,
                           changed_scientific_phase=True, isolated_six_window_effect=False),
        source_sha256={p.relative_to(EXPERIMENTS).as_posix(): sha(p) for p in inputs},
        selected=rows, comparisons=comparisons, trajectories=trajectories,
        current_calibration=current_calibration, previous_baseline_calibration=previous_calibration,
        matched_constant_profiles=constant_profiles, actual_prompt_readouts=prompt_readouts,
        incomplete_prior_tasks_excluded=old_selected["excluded"],
        new_sampling=0, new_model_forwards=0, new_llm_calls=0,
    )
    target = HERE / "previous_comparison.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    with (HERE / "previous_comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, lineterminator="\n", fieldnames=["teacher", "strategy", "suite", "skills", "selected_round", "schedule", "population", "built", "sa_n"]
                                + [f"{m}_{s}" for m in METRICS for s in ("mean", "best", "filtered_mean", "top10_mean")])
        writer.writeheader()
        for r in rows:
            writer.writerow({**{k: r[k] for k in ("teacher", "strategy", "suite", "skills", "selected_round", "schedule", "population", "built")},
                             "sa_n": r["stats"]["sa"]["n"],
                             **{f"{m}_{s}": r["stats"][m][s] for m in METRICS for s in ("mean", "best", "filtered_mean", "top10_mean")}})
    print(json.dumps(dict(selected=len(rows), candidate_views_validated=sum(r["population"] for r in rows),
                          comparisons=comparisons, previous_baseline_calibration=previous_calibration,
                          constant_profiles=len(constant_profiles), output_sha256=sha(target)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
