"""Compare saved populations and actual search receipts; never run generation.

Run locally after read_saved_populations.py has read the immutable remote inputs.
The bootstrap enumerates all 5**5 resamples of the five paired seed clusters.
It describes this adaptively selected fixed panel, not independent validation.
"""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import collections
import csv
import gzip
import hashlib
import itertools
import json
import math
import statistics


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def percentile(values, fraction):
    values = sorted(values)
    location = (len(values)-1)*fraction
    lower = math.floor(location)
    upper = math.ceil(location)
    return values[lower]*(upper-location)+values[upper]*(location-lower) if lower != upper else values[lower]


def paired(current, previous, metric):
    current = {(r["seed"],r["slot"]):r for r in current}
    previous = {(r["seed"],r["slot"]):r for r in previous}
    assert current.keys() == previous.keys() and len(current) == 50
    pairs = [(key, current[key][metric], previous[key][metric]) for key in sorted(current)]
    pairs = [(key,c,p) for key,c,p in pairs if c is not None and p is not None and math.isfinite(c) and math.isfinite(p)]
    seeds = collections.defaultdict(list)
    for (seed,slot), c, p in pairs:
        seeds[seed].append(c-p)
    assert len(seeds) == 5
    clusters = list(seeds.values())
    bootstrap = []
    for choices in itertools.product(range(5), repeat=5):
        bootstrap.append(sum(sum(clusters[i]) for i in choices)/sum(len(clusters[i]) for i in choices))
    direction = -1 if metric == "pic50" else 1
    return dict(n=len(pairs), current_mean=statistics.mean(c for _,c,p in pairs),
        previous_mean=statistics.mean(p for _,c,p in pairs),
        mean_delta=statistics.mean(c-p for _,c,p in pairs),
        improved=sum(direction*(c-p) < -1e-12 for _,c,p in pairs),
        worsened=sum(direction*(c-p) > 1e-12 for _,c,p in pairs),
        tied=sum(abs(c-p) <= 1e-12 for _,c,p in pairs),
        seed_mean_delta={str(s):statistics.mean(v) for s,v in seeds.items()},
        descriptive_cluster_bootstrap_95=[percentile(bootstrap,.025),percentile(bootstrap,.975)],
        bootstrap_resamples=len(bootstrap), independent_validation=False)


def csv_table(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    saved = read(HERE/"saved_population_readings.json")
    assert sha(HERE/"saved_population_readings.json") == "949e5ea0a97ee0cf6d118364a757c84cd56d0fe37004f139850040381f3992cc"
    assert sha(HERE/"comparison_candidates.jsonl.gz") == "fd5702b3a711718fc22fab7d05bf1ad42a1b3fe2abe57bc0d8f4b9781e227e98"
    assert saved["cases_sha256"] == sha(HERE/"source_cases.json")
    assert saved["analysis_code_sha256"] == sha(HERE/"read_saved_populations.py")
    groups = {g["id"]:g for g in saved["groups"]}
    with gzip.open(HERE/"comparison_candidates.jsonl.gz", "rt", encoding="utf-8") as handle:
        candidates = [json.loads(line) for line in handle]
    rows = collections.defaultdict(list)
    for row in candidates:
        rows[row["case_id"]].append(row)
    assert len(candidates) == 1100 and len(groups) == 22
    for group_id, values in rows.items():
        assert len(values) == 50 and len({(r["seed"],r["slot"]) for r in values}) == 50
        assert sum(r["built"] for r in values) == groups[group_id]["built"]
        for metric in ("rmsd","pic50","sa","rigid_fit_rmsd"):
            finite = [r[metric] for r in values if r[metric] is not None and math.isfinite(r[metric])]
            assert len(finite) == groups[group_id]["metrics"][metric]["n"]
            assert abs(statistics.mean(finite)-groups[group_id]["metrics"][metric]["mean"]) < 1e-12
    for teacher in ("6pyy","3w54"):
        related = [g for key,g in groups.items() if key.startswith(teacher)]
        assert len({g["condition_sha256"] for g in related}) == 1
        assert all(g["remove_aromaticity"] for g in related)
        for g in related:
            frame = g["protein_alignment"]
            assert frame["policy"] == "protein_anchored_proper_kabsch" and not frame["ligand_participates_in_fit"]
            assert frame["fitted_protein_rmsd_angstrom"] < 1e-8
        for a,b in zip(rows[teacher+"_six_window"], rows[teacher+"_box_calibration"]):
            assert (a["seed"],a["slot"]) == (b["seed"],b["slot"])
            for metric in ("rmsd","pic50","sa"):
                assert a[metric] == b[metric]

    old_best_path = EXPERIMENTS/"20261101_all_results_summary/完整任务最优轮对比.json"
    old_best = read(old_best_path)
    reconciled = []
    for old in old_best["selected"]:
        matches = [g for g in groups.values() if g["folder"] == old["best_configuration"]["folder"]]
        assert len(matches) == 1
        current = matches[0]
        for metric in ("rmsd","pic50","sa"):
            for stat in ("mean","best","filtered_mean","top10_mean"):
                delta = current["metrics"][metric][stat]-old["best_configuration"]["stats"][metric][stat]
                assert abs(delta) < 1e-10
        reconciled.append(current["id"])

    comparison_rows, pair_rows = [], []
    for teacher in ("6pyy","3w54"):
        current_id = teacher+"_box_search"
        for previous_name in ("four_window","joint_prefix","five_window","six_window"):
            previous_id = teacher+"_"+previous_name
            metrics = {m:paired(rows[current_id],rows[previous_id],m) for m in ("rmsd","pic50","sa","rigid_fit_rmsd")}
            current, previous = groups[current_id], groups[previous_id]
            rmsd_relative = 100*(current["metrics"]["rmsd"]["mean"]/previous["metrics"]["rmsd"]["mean"]-1)
            comparison_rows.append(dict(current=current_id,previous=previous_id,rmsd_relative_percent=rmsd_relative,paired=metrics))
            for metric, result in metrics.items():
                pair_rows.append(dict(teacher=teacher,current=current["label"],previous=previous["label"],metric=metric,
                    n=result["n"],mean_delta=result["mean_delta"],improved=result["improved"],worsened=result["worsened"],
                    seed_bootstrap_low=result["descriptive_cluster_bootstrap_95"][0],seed_bootstrap_high=result["descriptive_cluster_bootstrap_95"][1]))

    metrics_table, quality_table = [], []
    for g in groups.values():
        for metric in ("rmsd","pic50","sa","rigid_fit_rmsd"):
            m = g["metrics"][metric]
            metrics_table.append(dict(teacher=g["teacher"].split("_")[0],experiment=g["label"],role=g["role"],
                selected_round=g.get("selected_round"),scales=json.dumps(g["schedule"]),metric=metric,
                n=m["n"],missing=m["missing"],mean=m["mean"],best=m["best"],filtered_mean=m["filtered_mean"],
                top10_mean=m["top10_mean"],top10_n=m["top10_n"],excluded=len(m["excluded"]),
                built=g["built"],population=50,source=g["folder"]))
        q = g["quality"]
        chemistry = g["chemistry"]
        quality_table.append(dict(teacher=g["teacher"].split("_")[0],experiment=g["label"],role=g["role"],
            built=g["built"],population=50,unique_smiles=g["unique_smiles"],
            atomics_fraction=chemistry["atomics_fraction"]["mean"],charges_fraction=chemistry["charges_fraction"]["mean"],
            positive_bond_fraction=chemistry["positive_bond_fraction"]["mean"],
            all_measured_channels_count=round(50*chemistry["all_measured_channels_and_induced_bonds"]["mean"]),
            checked=q["checked_attempts"] if q else None,posebusters_pass=q["posebusters_pass"] if q else None,
            posebusters_evaluable=q["posebusters_evaluable"] if q else None,
            severe_clash=q["severe_clash"] if q else None,plip_evaluable=q["plip_evaluable"] if q else None,
            plip_recovery_mean=g["plip_mean"]))

    trajectories, costs = [], []
    current_readout_path = EXPERIMENTS/"20261102_box_exclusion_formal/reasoning_readout.json"
    six_readout_path = EXPERIMENTS/"20261101_six_window_sync_formal/final_results.json"
    current_readout, six_readout = read(current_readout_path), read(six_readout_path)
    current_results_path = EXPERIMENTS/"20261102_box_exclusion_formal/final_results.json"
    current_results = read(current_results_path)
    for previous in (current_results, six_readout):
        for key in ("rdkit","sascorer_sha256","sa_fragments_sha256"):
            assert previous[key] == saved[key]
    for t in current_readout["teachers"]:
        teacher = t["teacher"].split("_")[0]
        latest = groups[teacher+"_box_calibration"]["saved_ranking"]
        trajectories.append(dict(teacher=teacher,phase="current",iteration=0,round_best=latest,global_best=latest,schedule=groups[teacher+"_box_calibration"]["schedule"]))
        for r in t["rounds"]:
            index = next(i for i in range(3) if r["best_source"].endswith(f"action_{i+1:03d}"))
            latest = min(latest,r["round_best"])
            trajectories.append(dict(teacher=teacher,phase="current",iteration=r["iteration"],round_best=r["round_best"],
                global_best=latest,schedule=r["schedules"][index],tiers=r["tiers"],input_best=r["input_best"]))
        costs.append(dict(teacher=teacher,experiment="当前负面排除＋多样组合搜索",preexperiment=t["actual_budget"]["preexperiment"],
            formal=t["actual_budget"]["formal"],formal_built=t["formal_built"],formal_quality=t["formal_quality"],
            selected_tiers=t["selected_tiers"],native_requests=t["native_requests"],accepted_receipts=t["accepted_receipts"],excluded_boxes=t["eligible_boxes"]))
    for t in six_readout["teachers"]:
        teacher = t["teacher"].split("_")[0]
        latest = groups[teacher+"_six_calibration"]["saved_ranking"]
        trajectories.append(dict(teacher=teacher,phase="previous_six",iteration=0,round_best=latest,global_best=latest,schedule=groups[teacher+"_six_calibration"]["schedule"]))
        for r in t["rounds"]:
            p = r["checkpoint"]["performance"]
            latest = min(latest,p)
            trajectories.append(dict(teacher=teacher,phase="previous_six",iteration=r["iteration"],round_best=p,global_best=latest))
        costs.append(dict(teacher=teacher,experiment="先前六窗口同步",preexperiment=t["budget"]["calibration"],formal=t["budget"]["formal"],
            formal_built=t["formal_built"],formal_quality=t["quality_totals"]))

    calibration_gains = []
    for teacher in ("6pyy","3w54"):
        for suffix in ("four","six","box"):
            best_id = teacher+"_"+dict(four="four_window",six="six_window",box="box_search")[suffix]
            root_id = teacher+"_"+suffix+"_calibration"
            b,root = groups[best_id],groups[root_id]
            bm,rm = b["metrics"]["rmsd"]["mean"],root["metrics"]["rmsd"]["mean"]
            calibration_gains.append(dict(teacher=teacher,experiment=b["label"],root_rmsd=rm,best_agent_rmsd=bm,
                relative_change_percent=100*(bm/rm-1),budget=b["job_budget"],
                retained_global_best=min(bm,rm)))

    with gzip.open(EXPERIMENTS/"20261102_box_exclusion_formal/candidate_metrics.jsonl.gz","rt",encoding="utf-8") as handle:
        all_current = [json.loads(line) for line in handle]
    round_pairs = []
    for teacher in ("6pyy","3w54"):
        rs = [r for r in current_readout["teachers"] if r["teacher"].startswith(teacher)][0]["rounds"]
        for a,b in zip(rs,rs[1:]):
            prev = [dict(r,rmsd=r["recomputed_absolute_rmsd"]) for r in all_current if r["folder"] == a["best_source"]]
            now = [dict(r,rmsd=r["recomputed_absolute_rmsd"]) for r in all_current if r["folder"] == b["best_source"]]
            round_pairs.append(dict(teacher=teacher,previous_round=a["iteration"],current_round=b["iteration"],
                rmsd=paired(now,prev,"rmsd")))

    validation = dict(source_hashes_verified=True,groups=22,candidate_views=1100,primary_populations=10,
        primary_candidate_views=500,all_sources_complete_50=True,all_primary_jobs_completed_five_rounds=True,
        original_six_completed_cases_reconciled=reconciled,condition_semantics_identical_within_teacher=True,
        protein_fit_rmsd_below_1e_minus_8_angstrom=True,SA_same_standard_code=True,
        duplicate_fixed_panel_current_root_equals_previous_six_winner=True,
        independent_validation=False,claims_scope="Retrospective comparison of adaptively selected fixed five-seed populations; no causal skill or box-gating effect identified.")
    report = dict(timestamp=datetime.now(timezone(timedelta(hours=8))).isoformat(),validation=validation,
        analysis_code_sha256=sha(Path(__file__)),
        sources={str(p.relative_to(EXPERIMENTS)):sha(p) for p in [HERE/"saved_population_readings.json",HERE/"comparison_candidates.jsonl.gz",old_best_path,current_readout_path,six_readout_path,current_results_path]},
        comparisons=comparison_rows,calibration_gains=calibration_gains,trajectories=trajectories,
        consecutive_round_pairs=round_pairs,costs=costs,excluded_old_incomplete=old_best["excluded"],
        metric_definitions={k:saved[k] for k in ("rmsd_definition","affinity_definition","SA_definition","outlier_policy","top10_policy")},
        primary_distribution=[dict(id=g["id"],rmsd_below_half_angstrom=sum(r["rmsd"]<.5 for r in rows[g["id"]]),
            rmsd_median=statistics.median(r["rmsd"] for r in rows[g["id"]]),
            built_only_pic50_mean=statistics.mean(r["pic50"] for r in rows[g["id"]] if r["built"]))
            for g in groups.values() if g["role"] == "primary"],
        bootstrap_definition="All 3125 ordered five-seed cluster resamples, preserving each seed's paired rows; percentile 95% descriptive interval, adaptive fixed panel, not independent inference.",
        new_sampling=0,new_model_forwards=0,new_llm_calls=0)
    write(HERE/"comparison_readout.json",report)
    csv_table(HERE/"population_metrics.csv",metrics_table)
    csv_table(HERE/"paired_changes.csv",pair_rows)
    csv_table(HERE/"chemistry_and_quality.csv",quality_table)
    csv_table(HERE/"round_trajectories.csv",[{k:r.get(k) for k in ("teacher","phase","iteration","round_best","global_best","schedule")} for r in trajectories])
    print(json.dumps(dict(validation=validation,output_sha256={p.name:sha(p) for p in HERE.iterdir() if p.suffix in (".json",".csv")}),ensure_ascii=True))


if __name__ == "__main__":
    main()
