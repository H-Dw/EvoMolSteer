"""Post-cohort tail, cost, interaction and teacher-difficulty report; no sampling."""
import argparse
import csv
import html
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from evomolsteer.adaptive_schedule.io import read, write, eligible, DEFAULT_ROOT
from evomolsteer.adaptive_schedule.analysis import analyze, write_csv
from evomolsteer.adaptive_schedule.agent import node
from evomolsteer.adaptive_schedule.quality import metrics
from evomolsteer.adaptive_schedule.retention import artifact_folder


def correlation(rows, x, y):
    points = [(r[x], r[y]) for r in rows if r.get(x) is not None and r.get(y) is not None]
    if len(points) < 3 or len({p[0] for p in points}) < 2 or len({p[1] for p in points}) < 2:
        return {"n": len(points), "spearman_r": None}
    a, b = zip(*points)
    return {"n": len(points), "spearman_r": float(spearmanr(a, b).statistic),
            "interpretation": "exploratory teacher-level association, not a causal or generalization claim"}


def diverse_smiles(records):
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    smiles = sorted({r["smiles"] for r in records if eligible(r)})
    fp = [generator.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles]
    similarities = [v for i, f in enumerate(fp) for v in DataStructs.BulkTanimotoSimilarity(f, fp[:i])]
    return {"unique_smiles": len(smiles),
            "mean_pairwise_tanimoto_distance": 1 - float(np.mean(similarities)) if similarities else None}


def draw_molecule(sdf, node_id, path):
    from rdkit import Chem
    from rdkit.Chem import Draw
    for mol in Chem.SDMolSupplier(str(sdf), removeHs=True):
        if mol is not None and (node_id is None or (mol.HasProp("node_id") and mol.GetProp("node_id") == node_id)):
            copy = Chem.Mol(mol)
            from rdkit.Chem import rdDepictor
            rdDepictor.Compute2DCoords(copy)
            Draw.MolToFile(copy, str(path), size=(450, 330))
            return True
    return False


def draw_smiles(smiles, path):
    from rdkit import Chem
    from rdkit.Chem import Draw
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is not None:
        Draw.MolToFile(molecule, str(path), size=(450, 330))


def report(root, version):
    analyze(root, version)
    folder = root / "analysis" / version
    summaries = read(folder / "summary.json")
    teachers = read(root / "teachers.json")
    index = {(r["teacher_id"], r["arm"], r["deduplicated"]): r for r in summaries}
    cohort, curves, cases, branch_ids, call_paths = [], [], [], set(), set()
    for teacher in teachers:
        tid = teacher["teacher_id"]
        histories = []
        for p in sorted((root / "versions" / version / tid).glob("batch_*/histories.json")):
            histories.extend(read(p))
        if not histories:
            continue
        base = index[(tid, "r0", False)]
        best = index[(tid, version + "_best_per_root", False)]
        config = read(root / "config.json")
        fixed = index.get((tid, version + f"_fixed_r{config['rounds']:02d}", False))
        row = {"teacher_id": tid, "target": teacher["target"], "teacher_pic50": teacher["source_record"]["pic50_on_rescore"],
               "teacher_interactions": len(teacher["interactions"]), "core_coverage": teacher["core_coverage"],
               "r0_mean": base["mean"], "initial_teacher_gap": teacher["source_record"]["pic50_on_rescore"] - base["mean"]
               if base["mean"] is not None else None,
               "recovered_fraction": best["teacher_recovered_fraction_all_attempts"]}
        for metric in ["mean", "best", "top10_mean"]:
            row[metric] = best[metric]
            row[f"fixed_r{config['rounds']:02d}_{metric}"] = fixed[metric] if fixed else None
            for baseline in ["r0", "unguided", "steer"]:
                previous = index[(tid, baseline, False)][metric]
                row[f"{metric}_delta_{baseline}"] = best[metric] - previous if best[metric] is not None and previous is not None else None
                row[f"fixed_r{config['rounds']:02d}_{metric}_delta_{baseline}"] = (
                    fixed[metric] - previous if fixed and fixed[metric] is not None and previous is not None else None)
            row[f"{metric}_delta_teacher"] = best[metric] - row["teacher_pic50"] if best[metric] is not None else None
        selected = [next(h for h in c["history"] if h["id"] == c["best_id"]) for c in histories]
        row.update(diverse_smiles([h["record"] for h in selected]))
        cohort.append(row)
        for round_no in range(config["rounds"] + 1):
            values, hits = [], 0
            round_chains = [c for c in histories if len(c["history"]) > round_no]
            if not round_chains:
                continue
            for c in round_chains:
                attempts = [h for h in c["history"][:round_no + 1] if eligible(h["record"])]
                if attempts:
                    value = max(h["record"]["pic50_on_rescore"] for h in attempts)
                    values.append(value)
                    hits += value >= row["teacher_pic50"]
            curves.append({"teacher_id": tid, "round": round_no, "chains": len(round_chains),
                           "complete_round": len(round_chains) == config["batches"] * config["batch_size"],
                           "mean_best_so_far": float(np.mean(values)) if values else None,
                           "best_so_far": max(values) if values else None,
                           "top10_best_so_far": float(np.mean(sorted(values)[-10:])) if len(values) >= 10 else None,
                           "teacher_recovery_fraction": hits / len(round_chains)})
        for c in histories:
            for h in c["history"]:
                branch_ids.add(h["branch"])
                if h.get("call"):
                    call_paths.add(root / h["call"])
        eligible_selected = [h for h in selected if eligible(h["record"])]
        if eligible_selected:
            winner = max(eligible_selected, key=lambda h: h["record"]["pic50_on_rescore"])
            case_dir = folder / "cases" / tid
            case_dir.mkdir(parents=True, exist_ok=True)
            draw_molecule(teacher["sdf"], None, case_dir / "teacher.svg")
            slot = winner["record"]["slot"]
            retained = artifact_folder(root, version, tid, winner["record"]["batch"], winner["branch"], slot)
            draw_molecule(retained / "molecules_all_built.sdf", winner["record"]["node_id"], case_dir / "best.svg")
            q = metrics(root, winner["branch"], slot, teacher)
            case = {"teacher_id": tid, "teacher_score": row["teacher_pic50"], "best_record": winner["record"],
                    "ticks": winner["ticks"], "quality": q,
                    "teacher_complex": str(Path(teacher["profile_path"]).parent / "complex.pdb"),
                    "best_complex": str(retained / "profiles" / f"slot_{slot:03d}" / "complex.pdb"),
                    "terminal_local_features": node(root, winner["branch"], 100, slot)}
            comparisons = []
            original_chain = next(c for c in histories
                                  if any(h["branch"] == winner["branch"] and h["record"]["node_id"] == winner["record"]["node_id"]
                                         for h in c["history"]))
            initial = original_chain["history"][0]
            draw_molecule(root / initial["branch"] / "molecules_all_built.sdf",
                          initial["record"]["node_id"], case_dir / "r0.svg")
            comparisons.append({"arm": "same_root_r0", "record": initial["record"], "image": "r0.svg",
                                "quality": metrics(root, initial["branch"], slot, teacher),
                                "complex": str(root / initial["branch"] / "profiles" / f"slot_{slot:03d}" / "complex.pdb")})
            from evomolsteer.adaptive_schedule.quality import metrics_folder
            for arm in ["unguided", "steer"]:
                records = [r for r in read(root / "baselines" / f"{teacher['target']}__{arm}.json") if eligible(r)]
                if records:
                    chosen = max(records, key=lambda r: r["pic50_on_rescore"])
                    draw_smiles(chosen["smiles"], case_dir / f"{arm}.svg")
                    profile = Path(read(root / "config.json")["history"]) / "baseline_profiles_graph_v2" / teacher["target"] / arm / chosen["node_id"]
                    comparisons.append({"arm": f"historical_{arm}_best", "record": chosen, "image": f"{arm}.svg",
                                        "quality": metrics_folder(profile, teacher), "complex": str(profile / "complex.pdb")})
            case["comparisons"] = comparisons
            write(case_dir / "case.json", case)
            cases.append(case)
    write_csv(folder / "teacher_effects.csv", cohort)
    write_csv(folder / "optimization_curves.csv", curves)
    associations = {}
    for group in ["all"] + sorted({r["target"] for r in cohort}):
        rows = cohort if group == "all" else [r for r in cohort if r["target"] == group]
        associations[group] = {f"{x}__{y}": correlation(rows, x, y)
            for x in ["teacher_interactions", "teacher_pic50", "core_coverage", "initial_teacher_gap"]
            for y in ["mean_delta_r0", "best_delta_r0", "top10_mean_delta_r0", "recovered_fraction"]}
    write(folder / "teacher_associations.json", associations)
    receipts = []
    for call in call_paths:
        receipts.extend(read(p) for p in call.parent.glob(call.stem + "*.receipt.json"))
    costs = {"scientific_branches": len(branch_ids), "decision_calls": len(call_paths),
             "receipt_count_including_failures": len(receipts),
             "model_identifiers": sorted({str(r.get("returned_model")) for r in receipts}),
             "system_fingerprints": sorted({str(r.get("system_fingerprint")) for r in receipts}),
             "recorded_prompt_tokens": sum((r.get("usage") or {}).get("prompt_tokens", 0) for r in receipts),
             "recorded_completion_tokens": sum((r.get("usage") or {}).get("completion_tokens", 0) for r in receipts),
             "sampling_seconds": sum(read(root / b / "SAMPLING_COMPLETE.json")["seconds"] for b in branch_ids),
             "native_batch_steps_executed": sum(read(root / b / "SAMPLING_COMPLETE.json")["steps_executed"] for b in branch_ids),
             "unknown_usage_note": "Timeouts without provider receipts may have unobserved usage; no invented financial estimate"}
    write(folder / "costs.json", costs)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for teacher in teachers:
        data = [x for x in curves if x["teacher_id"] == teacher["teacher_id"]]
        for ax, key in zip(axes, ["mean_best_so_far", "top10_best_so_far", "teacher_recovery_fraction"]):
            ax.plot([x["round"] for x in data], [x[key] for x in data], marker=".", label=teacher["teacher_id"])
            ax.set_xlabel("Schedule update")
            ax.set_ylabel(key.replace("_", " "))
    axes[-1].legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    fig.savefig(folder / "optimization_curves.svg", bbox_inches="tight")
    fig.savefig(folder / "optimization_curves.pdf", bbox_inches="tight")
    plt.close(fig)
    cards = []
    for case in cases:
        tid = html.escape(case["teacher_id"])
        examples = "".join(f"<figure><img src='cases/{tid}/{x['image']}'>"
                           f"<figcaption>{x['arm']}: {html.escape(str(x['record'].get('pic50_on_rescore')))}</figcaption></figure>"
                           for x in case["comparisons"])
        cards.append(f"<section><h2>{tid}</h2><p>Teacher {case['teacher_score']:.4f}; "
                     f"best {case['best_record']['pic50_on_rescore']:.4f}</p>"
                     f"<img src='cases/{tid}/teacher.svg'><img src='cases/{tid}/best.svg'>"
                     f"<p>Schedule ticks: {html.escape(str(case['ticks']))}</p>{examples}</section>")
    page = "<!doctype html><meta charset='utf-8'><title>PLIP schedule report</title><style>body{font-family:sans-serif;max-width:1200px;margin:30px auto}img{max-width:100%}section img{width:45%}section{border-top:1px solid #ddd}</style>"
    page += f"<h1>PLIP schedule optimization — {version}</h1><p>Objective: Mean first, Best second. Primary system output: per-chain best-of-search Mean. Complete fixed-round R10 distributions are reported separately to distinguish proposal quality from selection gains. Three selected teachers from one target; shared seeds and teacher-origin lineages. Four amplitude intervals; all five batches advance together. Predictive affinity, not measured binding. Teacher associations are descriptive only.</p>"
    page += "<img src='optimization_curves.svg'>" + "".join(cards)
    (folder / "report.html").write_text(page, encoding="utf-8")
    return {"report": str(folder / "report.html"), "teachers": len(cohort), "costs": costs}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--skill-version", choices=["v1", "v2"], default="v1")
    options = parser.parse_args()
    print(json.dumps(report(options.root, options.skill_version), ensure_ascii=False))
