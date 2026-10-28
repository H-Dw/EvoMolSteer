"""Recompute paired statistics from archived measurements; no model or sampling."""
import hashlib
import json
import math
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def slot(row):
    return row["logical_seed"], row["logical_slot"]


def paired(before, after, field):
    a, b = ({slot(r): r for r in rows} for rows in (before, after))
    assert set(a) == set(b) and len(a) == 50
    seeds = sorted({s for s, _ in a})
    vals = [st.mean(float(b[s, i][field])-float(a[s, i][field]) for i in range(10)) for s in seeds]
    assert len(vals) == 5 and all(math.isfinite(v) for v in vals)
    mean = st.mean(vals)
    half = 2.7764451051977987 * st.stdev(vals) / math.sqrt(5)
    return {"mean_delta": mean, "seed_mean_deltas": vals,
            "exploratory_95pct_seed_cluster_interval": [mean-half, mean+half]}


def main():
    result, evidence, audit, process = (read(n) for n in
        ("results.json", "review_slot_evidence.json", "review_audit.json", "review_process_audit.json"))
    assert hashlib.sha256((ROOT/"results.json").read_bytes()).hexdigest() == audit["results_sha256"]
    assert audit["results_matches_finalized"] and not audit["changed_protected"]
    assert audit["all_group_checks_pass"] and audit["quality_modules_match"]
    assert not process["processes"] and all(v in ("free", "absent") for v in process["locks"].values())
    assert not audit["quality_worker"]["pid_still_present"]
    derived = {"measurement_source_sha256": audit["results_sha256"],
        "audit_time_shanghai": audit["time_shanghai"], "groups": {}, "paired_statistics_recomputed": [],
        "new_attempts": 400, "new_completed_terminal_records": 400,
        "reused_complete_attempts": 600, "referenced_prior_failed_attempts": 200,
        "uncertainty_unit": "five paired seed clusters; exploratory, not held-out validation",
        "new_primary_denominators": 50, "quality_selected_per_group": 10,
        "protected_files_unchanged": audit["protected_file_count"]}
    for key, group in evidence["groups"].items():
        assert all(audit["groups"][key]["record_and_quality_validation"].values())
        rows = group["records"]
        assert len(rows) == 50 and len({slot(r) for r in rows}) == 50
        rmsds = [r["condition_rmsd_A"] for r in rows]
        g = result["groups"][key]
        rmsd, score = st.mean(rmsds), st.mean(r["pic50_on_rescore"] for r in rows)
        assert math.isclose(rmsd, g["condition_atom_rmsd_angstrom"], abs_tol=1e-12)
        assert math.isclose(score, g["mean_pic50"], abs_tol=1e-12)
        built, connected = sum(r["build_success"] for r in rows), sum(r["build_success"] and r["connected"] for r in rows)
        assert (built, connected) == (g["built"], g["built_and_connected"])
        q = g["quality"]
        plip = [r["plip_recovery"] for r in group["quality"] if
                r.get("quality", {}).get("plip_evaluable") and r.get("plip_recovery") is not None]
        assert len(plip) == q["plip_evaluable"]
        derived["groups"][key] = {
            "mean_condition_atom_coordinate_rmsd_angstrom": rmsd,
            "median_condition_atom_coordinate_rmsd_angstrom": st.median(rmsds),
            "maximum_condition_atom_coordinate_rmsd_angstrom": max(rmsds),
            "mean_model_predicted_pic50": score, "built": built, "built_and_connected": connected,
            "unique_smiles": g["unique_smiles"], "sampled_posebusters_pass": q["posebusters_pass"],
            "sampled_posebusters_evaluable": q["posebusters_evaluable"],
            "sampled_plip_evaluable": len(plip), "sampled_plip_mean_recovery_evaluable_only": st.mean(plip) if plip else None,
            "sampled_unbuilt": q["unbuilt_selected"],
            "sampled_posebusters_unevaluable_built": q["posebusters_unevaluable_built"],
            "sampled_plip_unevaluable_built": q["plip_unevaluable_built"]}
    for c in result["comparisons"]:
        if c["paired_metrics"] is None:
            continue
        before = evidence["groups"][c["target"]+"/"+c["baseline"]]["records"]
        after = evidence["groups"][c["target"]+"/"+c["treatment"]]["records"]
        rec = {k: c[k] for k in ("target", "baseline", "treatment", "contrast")}
        rec["metrics"] = {}
        for field, reference in c["paired_metrics"].items():
            calc = paired(before, after, field)
            for name in ("mean_delta", "seed_mean_deltas", "exploratory_95pct_seed_cluster_interval"):
                old, new = reference[name], calc[name]
                old, new = (old, new) if isinstance(old, list) else ([old], [new])
                assert len(old) == len(new) and all(math.isclose(x, y, abs_tol=1e-10) for x, y in zip(old, new))
            rec["metrics"][field] = calc
        derived["paired_statistics_recomputed"].append(rec)
    key = "6pyy_H7S_A_402/chemical__raw999__"
    before, after = (evidence["groups"][key+v]["records"] for v in ("window", "late_only"))
    a, b = ({slot(r): r for r in rows} for rows in (before, after))
    worst = max(b, key=lambda k: b[k]["condition_rmsd_A"])
    total_delta = st.mean(b[k]["condition_rmsd_A"]-a[k]["condition_rmsd_A"] for k in a)
    contribution = (b[worst]["condition_rmsd_A"]-a[worst]["condition_rmsd_A"])/50
    derived["outlier_audit"] = {"seed_slot": worst, "before_angstrom": a[worst]["condition_rmsd_A"],
        "after_angstrom": b[worst]["condition_rmsd_A"], "mean_delta_all_50": total_delta,
        "one_slot_contribution_to_50_slot_mean_delta": contribution,
        "fraction_of_mean_delta_from_this_slot": contribution/total_delta,
        "remaining_49_paired_mean_delta_sensitivity_only": st.mean(
            b[k]["condition_rmsd_A"]-a[k]["condition_rmsd_A"] for k in a if k != worst),
        "after_build_success": b[worst]["build_success"], "after_connected": b[worst]["connected"],
        "outside_fixed_quality_sample": worst[1] not in (0, 1)}
    new = [g for k, g in derived["groups"].items() if "late_only" in k]
    derived["new_totals"] = {f: sum(g[f] for g in new) for f in
        ("built", "built_and_connected", "sampled_posebusters_pass", "sampled_posebusters_evaluable", "sampled_plip_evaluable")}
    derived["new_totals"]["not_selected_for_slow_quality"] = 320
    (ROOT/"analysis.json").write_text(json.dumps(derived, ensure_ascii=False, indent=2)+"\n", encoding="utf-8", newline="\n")
    plot(evidence)
    print(json.dumps({"complete_groups_recomputed": len(derived["groups"]),
        "available_contrasts_recomputed": len(derived["paired_statistics_recomputed"]),
        "new_totals": derived["new_totals"], "outlier": derived["outlier_audit"], "all_checks_pass": True}))


def plot(evidence):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True)
    colors = {"window": "#19769b", "late_only": "#da7b25"}
    for row, target in enumerate(("6pyy_H7S_A_402", "3w54_RNB_A_504")):
        for col, chem in enumerate(("gradient", "chemical")):
            ax = axes[row, col]
            for i, endpoint in enumerate(("raw1", "raw999")):
                for coverage, offset in (("window", -.15), ("late_only", .15)):
                    vals = [r["condition_rmsd_A"] for r in evidence["groups"][f"{target}/{chem}__{endpoint}__{coverage}"]["records"]]
                    bp = ax.boxplot([vals], positions=[i+offset], widths=.24, patch_artist=True,
                                    showfliers=False, medianprops={"color": "black"})
                    bp["boxes"][0].set(facecolor=colors[coverage], alpha=.24)
                    xx = [i+offset+((j*17)%49-24)*.003 for j in range(50)]
                    ax.scatter(xx, vals, s=10, alpha=.55, color=colors[coverage], edgecolors="none")
                    ax.scatter([i+offset], [st.mean(vals)], marker="D", s=30, color=colors[coverage],
                               edgecolors="black", linewidth=.5, zorder=4)
            ax.set(title=target.split('_')[0]+(" | coordinate gradient" if chem=="gradient" else " | gradient + chemical inpainting"),
                   xticks=[0, 1], xticklabels=["raw1", "raw999"], yscale="log", ylim=(.01,200), xlim=(-.5,1.5))
            ax.grid(axis="y", alpha=.2, which="both")
            ax.spines[["right", "top"]].set_visible(False)
            if col == 0: ax.set_ylabel("Condition-atom coordinate RMSD\n(angstrom)", fontsize=9)
    fig.legend(handles=[Line2D([0],[0], color=colors[c], lw=5, label=l) for c,l in
               (("window","Steps 10-90 (reused)"),("late_only","Steps 10-99 (new)"))]+
               [Line2D([0],[0],color="black",marker="D",linestyle="none",label="Mean; box line = median")],
               loc="upper center", ncol=3, frameon=False)
    fig.text(.5,.02,"All 50 paired slots shown per group; outliers retained. Log scale. Same five seeds, not held-out validation.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.94))
    fig.savefig(ROOT/"late_window_distributions.png", dpi=200)
    fig.savefig(ROOT/"late_window_distributions.svg")
    plt.close(fig)
    svg = ROOT/"late_window_distributions.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines())+"\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
