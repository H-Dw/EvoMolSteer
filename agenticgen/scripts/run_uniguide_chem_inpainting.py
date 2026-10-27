"""Paired chemical-only inpainting pilot. Two saved controls, 100 new attempts.

Run only on the specified remote host. The original UG campaign stays paused.
"""
import argparse
import ast
import copy
import datetime
import functools
import json
import math
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from agenticgen.uniguide.io import read, write, sha, value_sha, frozen
from agenticgen.uniguide.protocol import SEEDS, WINDOWS
from agenticgen.uniguide.chem_inpainting import POLICY, PilotBudget


CRITICAL = ["uniguide/"+name+".py" for name in
    ("sampler", "condition", "prepare", "inputs", "native_tools", "batching", "chunking", "random_streams")]
CRITICAL += ["runtime/scoring.py", "runtime/rng.py"]


def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def select_controls(source):
    teachers = read(source/"teachers.json")
    selected = {}
    for teacher in teachers:
        rows = []
        for path in sorted((source/"versions").glob("UG*/"+teacher["target"]+"/R*/**/aggregate.json")):
            row = read(path)
            if row.get("complete") and row["attempts"] == 50 and all(math.isfinite(row[k]) for k in ("mean_pic50", "C", "gap")):
                rows.append(row)
        if not rows:
            raise ValueError("No completed full-population Agent configuration")
        # Stable earliest ID breaks exact ties; repetitions are not extra evidence.
        winner = min(rows, key=lambda r: (-r["mean_pic50"], r["id"]))
        selected[teacher["target"]] = dict(group=winner["id"], aggregate=winner,
            complete_groups=len(rows), unique_schedules=len({tuple(r["schedule"]) for r in rows}),
            rule="maximum all-50 mean endpoint-rescored pIC50; earliest original ID breaks ties")
    return teachers, selected


def prepare(root, source):
    repo = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"]).strip():
        raise ValueError("Pilot code checkout must be a clean fixed commit")
    teachers, controls = select_controls(source)
    protected = {"attempts.json": sha(source/"attempts.json"), "config.json": sha(source/"config.json"),
                 "teachers.json": sha(source/"teachers.json"), "calibration/domain.json": sha(source/"calibration/domain.json")}
    package = repo/"agenticgen/src/agenticgen"
    for target, control in controls.items():
        group = source/control["group"]
        manifest = read(group/"manifest.json")
        teacher = next(t for t in teachers if t["target"] == target)
        if manifest["teacher_sha256"] != value_sha(teacher) or manifest["schedule"] != control["aggregate"]["schedule"]:
            raise ValueError("Saved control teacher/schedule mismatch")
        version_folder=source/"versions"/Path(control["group"]).parts[1]
        vm = read(version_folder/"manifest.json")
        protected[(version_folder/"manifest.json").relative_to(source).as_posix()]=sha(version_folder/"manifest.json")
        for skill in sorted((version_folder/"skills").glob("*.md")):
            if vm["skills"][skill.name]!=sha(skill):
                raise ValueError("Original Agent skills changed")
            protected[skill.relative_to(source).as_posix()]=sha(skill)
        for name in CRITICAL:
            if vm["source_binding"][name] != sha(package/name):
                raise ValueError("Scientific control implementation changed: "+name)
        # Only the action-validator error text changed in protocol.py. Check the
        # actual schedule expansion, seeds and windows used by the sampler.
        old = subprocess.check_output(["git","-C",str(repo),"show",
            vm["code_commit"]+":agenticgen/src/agenticgen/uniguide/protocol.py"],text=True)
        def sampling_protocol(text):
            nodes=ast.parse(text).body
            return [ast.dump(n,include_attributes=False) for n in nodes
                if (isinstance(n,ast.FunctionDef) and n.name=="expand") or
                (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ("SEEDS","WINDOWS") for t in n.targets))]
        if sampling_protocol(old)!=sampling_protocol((package/"uniguide/protocol.py").read_text()):
            raise ValueError("Sampling seeds, windows or scale expansion changed")
        for name in ("aggregate.json", "records.json", "COMPLETE.json", "manifest.json", "proposal.json", "trace.h5", "steps.json", "molecules_all_built.sdf", "QUALITY_COMPLETE.json", "quality.json"):
            path = group/name
            protected[path.relative_to(source).as_posix()] = sha(path)
        complete = read(group/"COMPLETE.json")
        if complete["records_sha256"] != sha(group/"records.json") or complete["manifest_sha256"] != sha(group/"manifest.json"):
            raise ValueError("Saved control receipt mismatch")
        rows = read(group/"records.json")
        if len(rows) != 50 or {(r["logical_seed"], r["logical_slot"]) for r in rows} != {(s,i) for s in SEEDS for i in range(10)}:
            raise ValueError("Saved control lacks the full original population")
        qc = read(group/"QUALITY_COMPLETE.json")
        if qc.get("checked_attempts") != 10:
            raise ValueError("Pilot diagnostics require matching 10/50 control coverage")
        checkpoint = source/"R0"/target/"checkpoint_010.pt.gz"
        protected[checkpoint.relative_to(source).as_posix()] = sha(checkpoint)
        protected[checkpoint.with_suffix(".sha.json").relative_to(source).as_posix()] = sha(checkpoint.with_suffix(".sha.json"))
    config = copy.deepcopy(read(source/"config.json"))
    config.update(root=str(root), chemical_only_pilot=dict(policy=POLICY, source_root=str(source),
        new_attempt_cap=100, population_per_teacher=50, start_step=10, projection_steps=[10,99],
        terminal_projection=True, self_conditioning_projection=False))
    spec = dict(policy=POLICY, created_from="current user-authorized paired experiment", code_commit=commit,
        config_sha256=value_sha(config), source_root=str(source), controls=controls,
        teacher_hashes={t["target"]:value_sha(t) for t in teachers}, source_hashes=protected,
        coordinate_guidance="original native FLOWR VJP with saved teacher-specific best Agent schedule",
        chemical_features="core atom element/charge/hybridization (when modeled), induced off-diagonal bonds including no-bonds",
        coordinate_replacement=False, chemical_projection_steps=[10,99], terminal_projection=True,
        core_scope="original PLIP direct interaction heavy atoms only; no expansion",
        source_quality_policy=dict(attempts=50, checked_attempts=10, not_selected=40, coverage=.2),
        seeds=SEEDS, slots=list(range(10)), windows=WINDOWS,
        new_attempt_cap=100, reused_control_attempts=100, original_UG_budget_unchanged=True,
        independent_validation=False, interpretation="exploratory paired intervention on seeds used for Agent selection")
    root.mkdir(parents=True, exist_ok=True)
    frozen(root/"STUDY.json", spec)
    frozen(root/"config.json", config)
    frozen(root/"teachers.json", teachers)
    for name in ("initials", "random_stream_reference"):
        link = root/name
        if link.exists() or link.is_symlink():
            if not link.is_symlink() or link.resolve() != (source/name).resolve():
                raise ValueError("Pilot source link differs")
        else:
            link.symlink_to(source/name, target_is_directory=True)
    for target, control in controls.items():
        link = root/"controls"/target
        link.parent.mkdir(exist_ok=True)
        if not link.exists():
            link.symlink_to(source/control["group"], target_is_directory=True)
        if link.resolve() != (source/control["group"]).resolve():
            raise ValueError("Pilot control link differs")
    return config, teachers, spec


def verify_sources(source, spec):
    bad = [name for name, digest in spec["source_hashes"].items() if sha(source/name) != digest]
    if bad:
        raise ValueError("Frozen control changed: "+str(bad))


def run(root, source):
    import fcntl
    import torch
    from agenticgen.uniguide import sampler
    from agenticgen.uniguide.chem_inpainting import ChemicalControl, teacher_features, terminal_projection
    from agenticgen.uniguide.prepare import load_model
    from agenticgen.uniguide.quality_queue import enqueue
    root.mkdir(parents=True, exist_ok=True)
    with (root/".executor.lock").open("a+") as own, (source/".executor.lock").open("r+") as shared:
        fcntl.flock(own, fcntl.LOCK_EX|fcntl.LOCK_NB)
        fcntl.flock(shared, fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/"FAILED.json").exists():
            raise ValueError("Pilot failed previously; preserve evidence and audit before any recovery")
        config, teachers, spec = prepare(root, source)
        verify_sources(source, spec)
        write(root/"STARTED.json", dict(time_shanghai=now(), pid=os.getpid(), code_commit=spec["code_commit"]))
        for teacher in teachers:
            target = teacher["target"]
            folder = root/"groups"/target
            if (folder/"COMPLETE.json").exists():
                enqueue(config, folder, teacher)
                continue
            model = load_model(config, teacher)
            features, meta = teacher_features(model, teacher)
            frozen(root/"features"/(target+".json"), meta)
            control = spec["controls"][target]
            original_control, original_budget = sampler.Control, sampler.Budget
            original_generate = model._generate_selective
            def generate(*args, **kwargs):
                return terminal_projection(original_generate(*args, **kwargs), teacher, features, folder)
            model._generate_selective = generate
            sampler.Control = functools.partial(ChemicalControl, chemical_features=features,
                                                baseline_trace=source/control["group"]/"trace.h5")
            sampler.Budget = lambda ignored: PilotBudget(root)
            try:
                sampler.execute(model, config, teacher, control["aggregate"]["schedule"], folder, "formal", 50,
                    resume_path=source/"R0"/target/"checkpoint_010.pt.gz")
            finally:
                sampler.Control, sampler.Budget = original_control, original_budget
                model._generate_selective = original_generate
            enqueue(config, folder, teacher)
            verify_sources(source, spec)
            write(root/"progress.json", dict(time_shanghai=now(), completed=[p.parent.name for p in (root/"groups").glob("*/COMPLETE.json")]))
            del model
            torch.cuda.empty_cache()
        verify_sources(source, spec)
        frozen(root/"SAMPLING_COMPLETE.json", dict(time_shanghai=now(), groups=2, new_attempts=100,
                                                  source_files_unchanged=True, quality_check_pending=True))


def summarize(root):
    import numpy as np
    from scipy.stats import t as student_t
    spec = read(root/"STUDY.json")
    rows = []
    paired_rows = []
    for target, control in spec["controls"].items():
        base = read(root/"controls"/target/"records.json")
        treat = read(root/"groups"/target/"records.json")
        by_key = lambda records: {(r["logical_seed"],r["logical_slot"]):r for r in records}
        a,b = by_key(base),by_key(treat)
        if set(a) != set(b) or len(a) != 50:
            raise ValueError("Incomplete paired population; never drop failures")
        aggregate = read(root/"groups"/target/"aggregate.json")
        changes = {}
        for metric in ("pic50_on_rescore", "condition_rmsd_A"):
            seed_deltas = [float(np.mean([b[s,i][metric]-a[s,i][metric] for i in range(10)])) for s in SEEDS]
            mean = float(np.mean(seed_deltas));sem = float(np.std(seed_deltas,ddof=1)/np.sqrt(5))
            width = float(student_t.ppf(.975,4)*sem)
            changes[metric] = dict(mean_delta=mean, seed_deltas=seed_deltas, exploratory_95pct_t_interval=[mean-width,mean+width], clusters=5)
        quality = {}
        for arm, folder in (("gradient_only",root/"controls"/target),("gradient_plus_chemical",root/"groups"/target)):
            if (folder/"QUALITY_COMPLETE.json").exists():quality[arm]=dict(status="complete",receipt=read(folder/"QUALITY_COMPLETE.json"))
            elif (folder/"QUALITY_FAILED.json").exists():quality[arm]=dict(status="engineering_failure",error=read(folder/"QUALITY_FAILED.json"))
            else:quality[arm]=dict(status="pending")
        rows.append(dict(target=target, control=control, treatment=aggregate, changes=changes, quality=quality,
            final_chemical_projection=read(root/"groups"/target/"FINAL_CHEMICAL_PROJECTION.json")))
        for s,i in sorted(a):
            paired_rows.append(dict(target=target,seed=s,logical_slot=i,
                base_pic50=a[s,i]["pic50_on_rescore"],chemical_pic50=b[s,i]["pic50_on_rescore"],
                delta_pic50=b[s,i]["pic50_on_rescore"]-a[s,i]["pic50_on_rescore"],
                base_C=a[s,i]["condition_rmsd_A"],chemical_C=b[s,i]["condition_rmsd_A"],
                base_built=a[s,i]["build_success"],chemical_built=b[s,i]["build_success"]))
    result = dict(time_shanghai=now(), policy=POLICY, results=rows, reused_controls=100, new_attempts=100,
        selection_bias="same five seeds used to choose Agent configuration; exploratory, not held-out validation",
        fidelity_interpretation="post-projection raw feature fidelity is imposed, not an independent measure of improvement",
        original_sources_unchanged=True)
    verify_sources(Path(spec["source_root"]),spec)
    write(root/"RESULTS.json",result)
    write(root/"paired_records.json",paired_rows)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=["prepare","run","report"])
    parser.add_argument("--root",required=True)
    parser.add_argument("--source-root",required=True)
    args=parser.parse_args()
    root,source=Path(args.root).resolve(),Path(args.source_root).resolve()
    if root==source or root.is_relative_to(source) or source.is_relative_to(root):
        raise ValueError("Pilot must use a separate sibling result root")
    try:
        if args.mode=="prepare":prepare(root,source)
        elif args.mode=="run":run(root,source)
        else:summarize(root)
    except BaseException as error:
        if args.mode=="run" and root.exists():
            write(root/"FAILED.json",dict(time_shanghai=now(),error_type=type(error).__name__,error=str(error),no_automatic_retry=True))
        raise


if __name__=="__main__":
    main()
