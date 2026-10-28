"""Read-only analysis of archived measurements; never loads a model or samples.

Run from this directory (or with an absolute script path). Inputs are the
downloaded, hash-bound RESULTS and original per-slot evidence. Derived files
are kept separate from the frozen machine report and experiment receipts.
"""
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def slot_key(row):
    return row.get("logical_seed", row.get("seed")), row.get("logical_slot", row.get("slot"))


def paired(before, after, field):
    a, b = ({slot_key(r): r for r in rows} for rows in (before, after))
    assert set(a) == set(b) and len(a) == 50
    seeds = sorted({s for s, _ in a})
    values = [statistics.mean(float(b[(s, i)][field]) - float(a[(s, i)][field])
                              for i in range(10)) for s in seeds]
    assert len(values) == 5 and all(math.isfinite(v) for v in values)
    mean = statistics.mean(values)
    half = 2.7764451051977987 * statistics.stdev(values) / math.sqrt(5)
    return {"mean_delta": mean, "seed_mean_deltas": values,
            "exploratory_95pct_seed_cluster_interval": [mean-half, mean+half]}


def main():
    result, evidence, audit = (read(n) for n in
        ("results.json", "review_slot_evidence.json", "review_audit.json"))
    assert hashlib.sha256((ROOT / "results.json").read_bytes()).hexdigest() == audit["results_sha256"]
    assert audit["results_matches_finalized"] and not audit["changed_protected"]
    assert not audit["processes"] and set(audit["locks"].values()) == {"free"}
    derived = {"measurement_source_sha256": audit["results_sha256"],
               "audit_time_shanghai": audit["time_shanghai"], "groups": {},
               "paired_statistics_recomputed": [], "all_primary_denominators": 50,
               "uncertainty_unit": "five seed clusters, not 50 independent validation replicates",
               "new_attempts": 900, "reused_control_attempts": 300,
               "new_terminal_records": 600, "failed_group_charged_attempts": 300,
               "total_complete_terminal_records_including_reused": 900}
    for key, group in evidence["groups"].items():
        assert all(audit["groups"][key]["record_and_quality_validation"].values())
        rows = group["records"]
        rmsd = statistics.mean(r["condition_rmsd_A"] for r in rows)
        score = statistics.mean(r["pic50_on_rescore"] for r in rows)
        assert math.isclose(rmsd, result["groups"][key]["condition_atom_rmsd_angstrom"], abs_tol=1e-12)
        assert math.isclose(score, result["groups"][key]["mean_pic50"], abs_tol=1e-12)
        quality = result["groups"][key]["quality"]
        plip = [r["plip_recovery"] for r in group["quality"]
                if r.get("quality", {}).get("plip_evaluable") and r.get("plip_recovery") is not None]
        assert len(plip) == quality["plip_evaluable"]
        derived["groups"][key] = {
            "mean_condition_atom_rmsd_angstrom": rmsd,
            "mean_model_predicted_pic50": score,
            "sampled_plip_evaluable": len(plip),
            "sampled_plip_mean_recovery_evaluable_only": statistics.mean(plip) if plip else None,
            "sampled_unbuilt": quality["unbuilt_selected"],
            "sampled_posebusters_pass": quality["posebusters_pass"],
            "sampled_posebusters_measured_not_pass": quality["posebusters_evaluable"] - quality["posebusters_pass"],
            "sampled_posebusters_unevaluable_built": quality["posebusters_unevaluable_built"],
            "sampled_plip_unevaluable_built": quality["plip_unevaluable_built"],
        }
    for c in result["comparisons"]:
        if c["paired_metrics"] is None:
            continue
        target = c["target"]
        before = evidence["groups"][target + "/" + c["baseline"]]["records"]
        after = evidence["groups"][target + "/" + c["treatment"]]["records"]
        rec = {k: c[k] for k in ("target", "baseline", "treatment", "contrast")}
        rec["metrics"] = {}
        for field, reference in c["paired_metrics"].items():
            calc = paired(before, after, field)
            for name in ("mean_delta", "exploratory_95pct_seed_cluster_interval"):
                old, new = reference[name], calc[name]
                old, new = (old, new) if isinstance(old, list) else ([old], [new])
                assert all(math.isclose(x, y, abs_tol=1e-10) for x, y in zip(old, new))
            rec["metrics"][field] = calc
        derived["paired_statistics_recomputed"].append(rec)
    outlier_rows = evidence["groups"]["6pyy_H7S_A_402/chemical__raw999__window"]["records"]
    worst = max(outlier_rows, key=lambda r: r["condition_rmsd_A"])
    derived["outlier_audit"] = {"target": "6pyy_H7S_A_402", "arm": "chemical__raw999__window",
        "seed": worst["logical_seed"], "logical_slot": worst["logical_slot"],
        "condition_atom_rmsd_angstrom": worst["condition_rmsd_A"],
        "connected": worst["connected"], "build_success": worst["build_success"],
        "mean_without_one_worst_sensitivity_only_not_primary": statistics.mean(
            r["condition_rmsd_A"] for r in outlier_rows if slot_key(r) != slot_key(worst)),
        "not_in_fixed_10_slot_quality_sample": worst["logical_slot"] not in (0, 1)}
    (ROOT / "analysis.json").write_text(json.dumps(derived, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    plot(evidence)
    print(json.dumps({"complete_groups_recomputed": len(derived["groups"]),
                      "available_contrasts_recomputed": len(derived["paired_statistics_recomputed"]),
                      "all_checks_passed": True}))


def plot(evidence):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    colors = {"window": "#1874a5", "full_interior": "#d96c30"}
    endpoints = ["native1", "raw1", "raw999"]
    for row, target in enumerate(("6pyy_H7S_A_402", "3w54_RNB_A_504")):
        for col, chemistry in enumerate(("gradient", "chemical")):
            ax = axes[row, col]
            for i, endpoint in enumerate(endpoints):
                for coverage, offset in (("window", -.16), ("full_interior", .16)):
                    key = f"{target}/{chemistry}__{endpoint}__{coverage}"
                    if key not in evidence["groups"]:
                        ax.text(i+offset, .94, "FAILED", color=colors[coverage],
                                ha="center", va="top", fontsize=8, rotation=90, transform=ax.get_xaxis_transform())
                        continue
                    vals = [r["condition_rmsd_A"] for r in evidence["groups"][key]["records"]]
                    assert all(v > 0 for v in vals)
                    bp = ax.boxplot([vals], positions=[i+offset], widths=.25,
                                    patch_artist=True, showfliers=False,
                                    medianprops={"color": "black"})
                    bp["boxes"][0].set(facecolor=colors[coverage], alpha=.24)
                    # Deterministic visual jitter only; no experimental random state.
                    xx = [i+offset+((j*17)%49-24)*.0034 for j in range(len(vals))]
                    ax.scatter(xx, vals, s=9, color=colors[coverage], alpha=.55, edgecolors="none")
                    ax.scatter([i+offset], [statistics.mean(vals)], marker="D", s=35,
                               color=colors[coverage], edgecolors="black", linewidth=.5, zorder=4)
            ax.set(title=target.split('_')[0]+" | "+("coordinate gradient" if chemistry=="gradient" else "+ chemical inpainting"),
                   xticks=range(3), xticklabels=endpoints, yscale="log", ylim=(.01,200))
            ax.grid(axis="y", alpha=.18, which="both")
            if col == 0: ax.set_ylabel("Condition-atom coordinate RMSD (angstrom)")
            ax.spines[["right", "top"]].set_visible(False)
    fig.legend(handles=[Line2D([0],[0],color=colors["window"],lw=5,label="Historical window (steps 10-90)"),
                        Line2D([0],[0],color=colors["full_interior"],lw=5,label="Full interior (steps 1-99)"),
                        Line2D([0],[0],marker="D",color="black",linestyle="none",label="Mean; box line = median")],
               loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(.5,.985))
    fig.text(.5,.02,"50 slots per completed group; paired seeds. Historical controls reused. Failed groups have no terminal distribution.",
             ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.93))
    fig.savefig(ROOT/"condition_rmsd_distributions.png",dpi=200)
    fig.savefig(ROOT/"condition_rmsd_distributions.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
