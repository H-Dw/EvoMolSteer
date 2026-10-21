"""Fixed-round and best-of-search distributions with exhaustive prompt-review indices."""
import csv
import math
from pathlib import Path

import numpy as np

from .io import read, write, eligible, sha


def distribution(records, teacher_score, deduplicate=False):
    valid = [r for r in records if eligible(r)]
    if deduplicate:
        retained = {}
        for r in valid:
            smiles = r["smiles"]
            if smiles not in retained or r["pic50_on_rescore"] > retained[smiles]["pic50_on_rescore"]:
                retained[smiles] = r
        valid = list(retained.values())
    x = np.array([r["pic50_on_rescore"] for r in valid], dtype=float)
    x.sort()
    return {"attempts": len(records), "eligible": len(valid), "unique_smiles": len({r["smiles"] for r in valid}),
            "mean": float(x.mean()) if len(x) else None, "median": float(np.median(x)) if len(x) else None,
            "best": float(x[-1]) if len(x) else None, "top10_mean": float(x[-10:].mean()) if len(x) >= 10 else None,
            "q10_q25_q75_q90_q95_q99": np.quantile(x, [.1, .25, .75, .9, .95, .99]).tolist() if len(x) else None,
            "teacher_recovered_count": int((x >= teacher_score).sum()),
            "teacher_exceeded_count": int((x > teacher_score).sum()),
            "teacher_recovered_fraction_all_attempts": float((x >= teacher_score).sum() / len(records)) if records else None}


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def analyze(root, version):
    root = Path(root)
    summaries, trials, paired, chains_flat, round_summaries = [], [], [], [], []
    for teacher in read(root / "teachers.json"):
        histories = []
        for p in sorted((root / "versions" / version / teacher["teacher_id"]).glob("batch_*/histories.json")):
            histories.extend(read(p))
        if not histories:
            continue
        score = teacher["source_record"]["pic50_on_rescore"]
        from .quality import metrics, metrics_folder
        from .agent import node
        def enriched(item, slot):
            record = dict(item["record"], **metrics(root, item["branch"], slot, teacher))
            n = node(root, item["branch"], 100, slot)
            record["local_loss"] = n["local_loss"]
            labels, charges = n["local_atom_labels"], n["local_charge_labels"]
            record["local_type_accuracy"] = float(np.mean([a == teacher["atom_labels"][i] for a, i in zip(labels, teacher["core_atoms"])]))
            record["local_charge_accuracy"] = float(np.mean([a == teacher["charge_labels"][i] for a, i in zip(charges, teacher["core_atoms"])]))
            return record
        selected = [enriched(next(h for h in c["history"] if h["id"] == c["best_id"]), c["slot"]) for c in histories]
        latest = [enriched(c["history"][-1], c["slot"]) for c in histories]
        initial = [enriched(c["history"][0], c["slot"]) for c in histories]
        arms = {"r0": initial, version + "_best_per_root": selected, version + "_latest": latest}
        # Keep unselected proposals separate from the monotone best-of-search curve.
        # Partial batches never masquerade as a complete fixed-budget result.
        config = read(root / "config.json")
        expected = config["batches"] * config["batch_size"]
        for round_no in range(config["rounds"] + 1):
            rows = [c["history"][round_no]["record"] for c in histories if len(c["history"]) > round_no]
            if rows:
                for dedup in [False, True]:
                    round_summaries.append({"teacher_id": teacher["teacher_id"], "target": teacher["target"],
                        "round": round_no, "deduplicated": dedup, "expected_attempts": expected,
                        "complete_round": len(rows) == expected, **distribution(rows, score, dedup)})
        final_rows = [enriched(c["history"][config["rounds"]], c["slot"]) for c in histories
                      if len(c["history"]) > config["rounds"]]
        if len(final_rows) == expected:
            arms[version + f"_fixed_r{config['rounds']:02d}"] = final_rows
        for arm in ["steer", "unguided"]:
            arms[arm] = read(root / "baselines" / f"{teacher['target']}__{arm}.json")
            for record in arms[arm]:
                folder = Path(read(root / "config.json")["history"]) / "baseline_profiles_graph_v2" / teacher["target"] / arm / record["node_id"]
                record.update(metrics_folder(folder, teacher))
        for arm, records in arms.items():
            for dedup in [False, True]:
                summaries.append({"teacher_id": teacher["teacher_id"], "target": teacher["target"],
                                  "arm": arm, "deduplicated": dedup, **distribution(records, score, dedup)})
            evaluated = [r for r in records if r.get("quality_evaluated")]
            quality_summary = {"teacher_id": teacher["teacher_id"], "target": teacher["target"],
                               "arm": arm, "quality_attempts": len(records), "quality_evaluated": len(evaluated)}
            for key in ["posebusters_pass", "severe_clash", "plip_recovery", "plip_new_fingerprints",
                        "local_loss", "local_type_accuracy", "local_charge_accuracy"]:
                values = [r[key] for r in records if r.get(key) is not None]
                quality_summary[key + "_mean"] = float(np.mean(values)) if values else None
            summaries[-2]["quality"] = quality_summary
            pb = [r for r in records if r.get("posebusters_pass")]
            summaries.append({"teacher_id": teacher["teacher_id"], "target": teacher["target"],
                              "arm": arm + "_PB_pass", "deduplicated": False,
                              **distribution(pb, score)})
        for chain in histories:
            best = next(h for h in chain["history"] if h["id"] == chain["best_id"])
            base = chain["history"][0]
            hit = next((i for i, h in enumerate(chain["history"])
                        if eligible(h["record"]) and h["record"]["pic50_on_rescore"] >= score), None)
            row = {"teacher_id": teacher["teacher_id"], "target": teacher["target"],
                   "batch": base["record"]["batch"], "slot": chain["slot"],
                   "rounds": len(chain["history"]) - 1, "best_id": chain["best_id"], "first_teacher_recovery_round": hit,
                   "best": best["record"].get("pic50_on_rescore"), "r0": base["record"].get("pic50_on_rescore"),
                   "best_eligible": eligible(best["record"]), "r0_eligible": eligible(base["record"]),
                   "best_branch": best["branch"], "teacher_pic50": score,
                   "teacher_core_atoms": len(teacher["core_atoms"]),
                   "teacher_interactions": len(teacher["interactions"]),
                   "teacher_core_coverage": len(teacher["core_atoms"]) / len(teacher["coords_A"])}
            row["delta_r0"] = row["best"] - row["r0"] if row["best_eligible"] and row["r0_eligible"] else None
            chains_flat.append(row)
            for item in chain["history"]:
                trials.append({"teacher_id": teacher["teacher_id"], "slot": chain["slot"],
                               "batch": row["batch"], "round": item["id"], "branch": item["branch"],
                               "parent_id": item["parent_id"], "ticks": str(item["ticks"]),
                               "eligible": eligible(item["record"]),
                               "pic50_on_rescore": item["record"].get("pic50_on_rescore"),
                               "improved_parent": item.get("improved_parent"),
                               "improved_incumbent": item.get("improved_incumbent")})
        deltas = [r["delta_r0"] for r in chains_flat if r["teacher_id"] == teacher["teacher_id"] and r["delta_r0"] is not None]
        paired.append({"teacher_id": teacher["teacher_id"], "pairs": len(deltas),
                       "mean_delta": float(np.mean(deltas)) if deltas else None,
                       "median_delta": float(np.median(deltas)) if deltas else None,
                       "win_fraction": float(np.mean(np.array(deltas) > 0)) if deltas else None,
                       "quantiles": np.quantile(deltas, [0, .1, .25, .5, .75, .9, 1]).tolist() if deltas else []})
    folder = root / "analysis" / version
    write(folder / "summary.json", summaries)
    write_csv(folder / "fixed_round_distributions.csv", round_summaries)
    write(folder / "paired_r0.json", paired)
    write_csv(folder / "chains.csv", chains_flat)
    write_csv(folder / "all_trials.csv", trials)
    # Exploratory uncertainty uses five paired native seed batches, never 11*100 pseudoreplicates.
    rng = np.random.default_rng(42)
    bootstrap = []
    for teacher in read(root / "teachers.json"):
        rows = [r for r in chains_flat if r["teacher_id"] == teacher["teacher_id"]]
        batch_means = [np.mean([r["delta_r0"] for r in rows if r["batch"] == b and r["delta_r0"] is not None])
                       for b in range(5) if any(r["batch"] == b and r["delta_r0"] is not None for r in rows)]
        ci = None
        if len(batch_means) == 5:
            draws = rng.choice(batch_means, (5000, 5), replace=True).mean(1)
            ci = np.quantile(draws, [.025, .975]).tolist()
        bootstrap.append({"teacher": teacher["teacher_id"], "paired_seed_batches": len(batch_means),
                          "exploratory_mean_delta_ci": ci})
    write(folder / "batch_bootstrap.json", bootstrap)
    comparison = []
    other = root / "analysis/v1/chains.csv"
    if version == "v2" and other.exists():
        with other.open(encoding="utf-8-sig", newline="") as f:
            old = {(r["teacher_id"], int(r["batch"]), int(r["slot"])): r for r in csv.DictReader(f)}
        for row in chains_flat:
            key = row["teacher_id"], row["batch"], row["slot"]
            if key in old:
                previous = old[key]
                d = row["best"] - float(previous["best"]) if row["best_eligible"] and previous["best_eligible"] == "True" else None
                comparison.append({"teacher_id": key[0], "batch": key[1], "slot": key[2], "delta_v2_v1": d})
        write_csv(folder / "paired_v2_v1.csv", comparison)
    lines = [f"# PLIP幅度调度：{version}", "",
             "主要系统输出指标为每条链TOP1构成的Mean，同时报告Best和Top10；完整固定轮次R10的Mean用于区分实际生成分布与跨轮择优收益。",
             "fixed_round_distributions.csv保留每轮未做跨轮择优的结果。latest在未完成时可能混合轮次，不作为完整R10。历史基线及跨教师共享初态不重复计作独立样本。",
             "缺失与结构失败保留；Top10有效分子不足10个时为空。多轮搜索不代表等计算预算优势。", "",
             "|教师|条件|有效数|Mean|Best|Top10均值|", "|---|---|---:|---:|---:|---:|"]
    fmt = lambda x: "NA" if x is None else f"{x:.4f}"
    for r in summaries:
        if not r["deduplicated"]:
            lines.append(f"|{r['teacher_id']}|{r['arm']}|{r['eligible']}|{fmt(r['mean'])}|{fmt(r['best'])}|{fmt(r['top10_mean'])}|")
    (folder / "report.zh-CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"chains": len(chains_flat), "trials_including_r0": len(trials), "report": str(folder / "report.zh-CN.md")}


def review_pack(root, version):
    root = Path(root)
    records = []
    for path in sorted((root / "versions" / version).glob("*/batch_*/histories.json")):
        for c in read(path):
            for item in c["history"][1:]:
                call = root / item["call"]
                records.append({"teacher": path.parents[1].name, "batch": item["record"]["batch"],
                                "slot": c["slot"], "id": item["id"], "proposal": item["proposal"],
                                "parent_id": item["parent_id"], "outcome": item["record"],
                                "improved_parent": item["improved_parent"],
                                "improved_incumbent": item["improved_incumbent"],
                                "prompt_evidence": str(call.with_suffix(".evidence.json").relative_to(root)),
                                "prompt_evidence_sha256": sha(call.with_suffix(".evidence.json")),
                                "call": item["call"]})
    folder = root / "review" / version
    shards = []
    for start in range(0, len(records), 50):
        p = folder / f"shard_{start//50:04d}.json"
        write(p, records[start:start+50])
        shards.append({"path": str(p.relative_to(root)), "sha256": sha(p), "decisions": len(records[start:start+50])})
    manifest = {"version": version, "decisions": len(records), "shards": shards,
                "requires_main_agent_review": True,
                "coverage": "Review each unique decision/input delta; inspect raw prompt for anomalous cases",
                "shared_skill": f"versions/{version}/SKILL.md"}
    write(folder / "manifest.json", manifest)
    return manifest
