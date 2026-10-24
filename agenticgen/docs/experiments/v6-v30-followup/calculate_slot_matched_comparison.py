"""Aggregate preserved outcomes for the campaign's original 5 x 5 slot subset.

This is a report calculation. It performs no sampling, scoring, optimization,
network requests, or software tests. Run with Python's standard library only.
"""
import csv
import hashlib
import json
import statistics
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    historical_path = HERE / "historical_template_per_molecule.csv"
    formal_path = HERE / "performance_comparison.json"
    with historical_path.open(encoding="utf-8-sig", newline="") as stream:
        historical = list(csv.DictReader(stream))
    formal = json.loads(formal_path.read_text(encoding="utf-8"))["rows"]
    expected = {f"unguided_b{batch:03d}_final_{slot:03d}"
                for batch in range(5) for slot in range(5)}
    rows = []
    identities = {}
    labels = ("shared_R0_guided25", "V17_primary25", "V30_primary25")
    for teacher in ("top1", "top2"):
        selected = [r for r in historical if r["teacher"] == teacher
                    and r["population"] == "unguided" and r["node_id"] in expected]
        if len(selected) != 25 or {r["node_id"] for r in selected} != expected:
            raise ValueError("Historical slot subset is incomplete or duplicated")
        identities[teacher] = sorted(r["node_id"] for r in selected)
        errors = [float(r["template_RMSD_A"]) for r in selected]
        scores = [float(r["pIC50"]) for r in selected]
        baseline = {
            "teacher": teacher, "population": "unguided_historical_slotmatched25", "n": 25,
            "template_RMSD_mean_A": statistics.mean(errors),
            "template_RMSD_min_A": min(errors),
            "pIC50_mean": statistics.mean(scores), "pIC50_best": max(scores),
        }
        comparisons = [baseline] + [
            {k: value for k, value in next(r for r in formal if r["teacher"] == teacher
                                           and r["population"] == label).items()
             if k in baseline}
            for label in labels
        ]
        for row in comparisons:
            delta = row["template_RMSD_mean_A"] - baseline["template_RMSD_mean_A"]
            row["delta_template_RMSD_mean_vs_slotmatched25_A"] = delta
            row["relative_template_RMSD_reduction_pct"] = -100 * delta / baseline["template_RMSD_mean_A"]
            row["delta_pIC50_mean_vs_slotmatched25"] = row["pIC50_mean"] - baseline["pIC50_mean"]
            row["delta_pIC50_best_vs_slotmatched25"] = row["pIC50_best"] - baseline["pIC50_best"]
            rows.append(row)
    code_paths = ["wave_native.py", "common_random.py", "flowr.py", "contracts.py", "vendor/reward.py"]
    sources = [historical_path, formal_path, HERE / "baseline_geometry_inputs.json", Path(__file__).resolve()]
    sources += [REPO / "agenticgen/src/agenticgen/adaptive_schedule" / name for name in code_paths]
    result = {
        "method": "Prespecified original batch 0..4 x slot 0..4 subset, selected by identity only; aggregate existing outcomes",
        "comparison_status": "initial-slot-matched historical reference; not a newly executed strict paired zero-guidance experiment",
        "remaining_limits": [
            "Historical PDB coordinates have 0.001 A precision; formal guided geometry uses native full precision.",
            "The campaign preserves original initial slots and native random draw shapes; no zero-guidance terminal replay was executed through the current 25-row implementation.",
            "Historical/native execution and batch numerical equivalence are not newly established by this aggregation.",
            "Guided representatives were selected after search; these are descriptive selected outcomes, not an unbiased estimate of generic guidance effectiveness.",
        ],
        "selected_historical_node_ids": identities,
        "source_sha256": {p.relative_to(REPO).as_posix(): sha(p) for p in sources},
        "rows": rows, "new_sampling": False, "new_scoring": False,
        "remote_execution": False, "automated_tests_run": False,
    }
    (HERE / "slot_matched_comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    with (HERE / "slot_matched_comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
