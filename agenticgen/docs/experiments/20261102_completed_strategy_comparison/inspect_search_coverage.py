"""Read archived steps and executed profiles to diagnose search coverage only."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import collections
import csv
import hashlib
import json
import math
import statistics
import tarfile


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
SOURCE = HERE.parent / "20261102_box_exclusion_formal"


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def summarize(values):
    assert len(values) == 50 and all(math.isfinite(v) for v in values)
    return dict(mean=statistics.mean(values), median=statistics.median(values), maximum=max(values))


def main():
    plan = json.loads((SOURCE/"machine_plan.json").read_text(encoding="utf-8"))
    runtime = ["box_gates.py","linear_scale.py","profile_diversity.py","linear_council.py","condition.py","sampler.py"]
    runtime_sha = {}
    for filename in runtime:
        name = "agenticgen/src/agenticgen/uniguide/" + filename
        runtime_sha[name] = sha(REPO/name)
        assert runtime_sha[name] == plan["runtime_source_sha256"][name]
    with (SOURCE/"formal_actions.csv").open(encoding="utf-8-sig",newline="") as handle:
        actions = list(csv.DictReader(handle))
    with (SOURCE/"preexperiment_readings.csv").open(encoding="utf-8-sig",newline="") as handle:
        preexperiment = list(csv.DictReader(handle))
    reports = []
    winners = {"6pyy":"01_box_exclusion_6pyy/attempt_001/versions/GD01/6pyy_H7S_A_402/iteration_004/action_001",
               "3w54":"02_box_exclusion_3w54/attempt_001/versions/GD01/3w54_RNB_A_504/iteration_005/action_001"}
    archive_dir = SOURCE/"downloads/20261102_0850"
    manifest = {r["path"]:r for r in json.loads((archive_dir/"MANIFEST.json").read_text(encoding="utf-8"))}
    with tarfile.open(archive_dir/"completed_increment.tar.gz", "r:gz") as tar:
        members = {m.name:m for m in tar.getmembers()}
        for teacher,subpath in winners.items():
            name = "workspace/agenticgen_box_exclusion_20261102/jobs/"+subpath+"/steps.json"
            member = members[name]
            data = tar.extractfile(member).read()
            assert len(data) == manifest[name]["bytes"]
            assert hashlib.sha256(data).hexdigest() == manifest[name]["sha256"]
            steps = json.loads(data)
            assert [r["step"] for r in steps] == list(range(100))
            measured = [r for r in actions if r["teacher"].startswith(teacher)]
            profiles = [json.loads(r["actual_scale"]) for r in measured]
            assert len(profiles) == 15
            coverage = [dict(window=i, values=dict(sorted(collections.Counter(p[i] for p in profiles).items()))) for i in range(6)]
            step_rows = []
            for index in [0,1,5,10,11,50,98,99]:
                row = steps[index]
                step_rows.append(dict(step=index,time=row["time"],scale=row["scale"],gamma=row["gamma"],
                    **{key:summarize(row[key]) for key in ["coefficient","gradient_norm","max_atom_increment_A","delta_native_ratio","assignment_switch_fraction"]}))
            reports.append(dict(teacher=teacher,archive_member=name,source_bytes=len(data),source_sha256=manifest[name]["sha256"],
                archive_member_verified_against_manifest=True,executed_profiles=15,window_coverage=coverage,
                executed_early_changes=[{k:r[k] for k in ["round","action","actual_scale","rank_rmsd"]}
                    for r in measured if json.loads(r["actual_scale"])[0] != (50 if teacher=="6pyy" else 30)],
                coarse_early_profiles=[r for r in preexperiment if r["teacher"].startswith(teacher)
                    and (r["profile"].endswith("LowerRescue_0") or r["profile"].endswith("EarlyUpperRescue"))],
                selected_step_diagnostics=step_rows))
    report = dict(timestamp=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        science_commit=plan["commit"],runtime_sha256=runtime_sha,analysis_code_sha256=sha(Path(__file__)),
        sources={str(p.relative_to(SOURCE)):sha(p) for p in [SOURCE/"formal_actions.csv",SOURCE/"preexperiment_readings.csv",archive_dir/"MANIFEST.json"]},
        diagnostics=reports,
        delta_definition="-dt * scale * ((1-t+1e-6)/t + gamma(t)) * full_input_VJP; no clipping or gradient normalization.",
        ratio_definition="Per-candidate L2 norm of the entire added coordinate update divided by L2 norm of the native step, including its stochastic part.",
        displacement_definition="Per-candidate maximum over atoms of the added update magnitude in Angstrom; table statistics are over 50 candidates.",
        interpretation="Observational diagnostics only. Large early updates motivate low-dose search but do not prove that reducing scale improves terminal performance.",
        new_sampling=0,new_model_forwards=0,new_llm_calls=0)
    (HERE/"search_coverage_diagnostics.json").write_text(json.dumps(report,ensure_ascii=True,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(dict(teachers=len(reports),runtime_files_verified=len(runtime_sha),step_members_verified=2,
        output_sha256=sha(HERE/"search_coverage_diagnostics.json"),new_sampling=0,new_model_forwards=0,new_llm_calls=0)))


if __name__ == "__main__":
    main()
