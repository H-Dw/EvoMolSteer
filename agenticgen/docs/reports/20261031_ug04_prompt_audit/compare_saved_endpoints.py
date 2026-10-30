"""Post-hoc coordinate measurements of saved outputs; no generation or API calls."""
import argparse
import datetime
import gzip
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def measure(root, group, teacher, core, *, current=False):
    folder = root / group
    with gzip.open(folder / "terminal.pt.gz", "rb") as f:
        terminal = torch.load(f, map_location="cpu", weights_only=False)
    xyz = terminal["coords"].numpy().astype(np.float64)
    valid = terminal["mask"].numpy().astype(bool)
    rows = read(folder / "records.json")
    with h5py.File(folder / "trace.h5", "r") as f:
        first = f["assignment"][10].astype(int)
        last = f["assignment"][99].astype(int)
    dynamic = []
    for coordinates, mask in zip(xyz, valid):
        ids = np.flatnonzero(mask)
        cost = np.sum((core[:, None] - coordinates[ids][None])**2, axis=-1)
        left, right = linear_sum_assignment(cost)
        dynamic.append(float(np.sqrt(cost[left, right].mean())))
    errors = xyz[np.arange(len(xyz))[:, None], first] - core[None]
    frozen = np.sqrt(np.sum(errors*errors, axis=(1, 2))/len(core))
    field = "condition_coordinate_root_mean_square_deviation_angstrom" if current else "condition_rmsd_A"
    recorded = np.asarray([r[field] for r in rows])
    check = frozen if current else np.asarray(dynamic)
    assert np.max(np.abs(recorded-check)) < 1e-5
    return dict(
        group=group, teacher=teacher, population=len(rows),
        recorded_mean=float(recorded.mean()),
        terminal_geometry_hungarian_mean=float(np.mean(dynamic)),
        first_guidance_assignment_retained_mean=float(frozen.mean()),
        recorded_recalculation_maximum_difference=float(np.max(np.abs(recorded-check))),
        first_last_assignment_different_fraction=float(np.mean(first != last)),
        built=sum(bool(r["build_success"]) for r in rows),
        terminal_sha256=sha(folder / "terminal.pt.gz"),
        trace_sha256=sha(folder / "trace.h5"),
        rows=[dict(logical_seed=r["logical_seed"],logical_slot=r["logical_slot"],
                   node_id=r["node_id"],recorded=float(saved),
                   terminal_geometry_hungarian=float(d),
                   first_guidance_assignment_retained=float(f))
              for r,saved,d,f in zip(rows,recorded,dynamic,frozen)],
        diagnostic_only=True,original_ranking_unchanged=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("old_root", type=Path)
    parser.add_argument("new_root", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    teachers = {r["target"]:r for r in read(args.old_root / "teachers.json")}
    old = read(args.old_root / "versions/UG04/FINALIZED.json")
    new = read(args.new_root / "versions/GD01/AWAITING_GPT_REVIEW.json")["current"]
    rows, cores = [], {}
    for name, teacher in teachers.items():
        condition = read(args.new_root / "conditions" / (name + ".json"))
        lookup = {a["id"]:i for i,a in enumerate(condition["atoms"])}
        core = np.asarray([condition["reference_coordinates_angstrom"][lookup[i]]
                           for i in condition["controlled_atom_ids"]])
        previous = np.asarray(teacher["core_A"])
        cost = np.sum((core[:, None] - previous[None])**2, axis=-1)
        left, right = linear_sum_assignment(cost)
        assert cost[left,right].max() < 1e-12
        frame = condition["protein_alignment"]
        assert np.allclose(frame["rotation"],np.eye(3),atol=1e-12)
        assert np.allclose(frame["translation_angstrom"],0,atol=1e-12)
        cores[name] = dict(new=core,old=previous)
    for label, root, version in [("UG04",args.old_root,old),("GD01",args.new_root,new)]:
        for group in version["groups"]:
            name = group.get("teacher_id") or group["id"].split("/")[2]
            row = measure(root,group["id"],name,cores[name]["new" if label=="GD01" else "old"],current=label=="GD01")
            rows.append(dict(version=label,**row))
    baselines = []
    domains = read(args.new_root / "calibration/domains.json")
    for name in cores:
        for label, root, group, current in [("R0",args.old_root,"R0/"+name,False),
                                           ("calibrated_GD01",args.new_root,domains[name]["initial_source"],True)]:
            baselines.append(dict(version=label,**measure(root,group,name,cores[name]["new" if current else "old"],current=current)))
    result = dict(timestamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
                  operation="read_only_saved_terminal_posthoc_measurement_no_model_inference",
                  core_coordinates_equal_as_sets=True,protein_alignment_identity=True,
                  groups=rows,baselines=baselines,
                  note="Terminal Hungarian rescoring is diagnostic only; it does not establish retention of the chemically conditioned atoms or replace the frozen-assignment ranking.")
    args.destination.parent.mkdir(parents=True,exist_ok=True)
    args.destination.write_text(json.dumps(result,ensure_ascii=True,indent=2)+"\n",encoding="utf-8")
    compact=[]
    for version in ["UG04","GD01"]:
        for name in cores:
            selected=[r for r in rows if r["version"]==version and r["teacher"]==name]
            compact.append(dict(version=version,teacher=name,
                mean_recorded=float(np.mean([r["recorded_mean"] for r in selected])),
                mean_terminal_hungarian=float(np.mean([r["terminal_geometry_hungarian_mean"] for r in selected])),
                mean_first_assignment_retained=float(np.mean([r["first_guidance_assignment_retained_mean"] for r in selected])),
                best_terminal_hungarian=min(r["terminal_geometry_hungarian_mean"] for r in selected),
                best_first_assignment_retained=min(r["first_guidance_assignment_retained_mean"] for r in selected)))
    print(json.dumps(dict(destination=str(args.destination),sha256=sha(args.destination),groups=len(rows),summary=compact)))


if __name__ == "__main__":
    main()
