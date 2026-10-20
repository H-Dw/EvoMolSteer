"""Independent CPU profiling queue; never imports the FLOWR sampler or its RNG."""
import concurrent.futures
import time
from pathlib import Path

from .io import read, write


def tasks(root, version):
    from rdkit import Chem
    teachers = {t["teacher_id"]: t for t in read(root / "teachers.json")}
    seen, wanted = set(), []
    for p in sorted((root / "versions" / version).glob("*/batch_*/histories.json")):
        template = teachers[p.parents[1].name]
        for chain in read(p):
            for h in chain["history"]:
                slot = chain["slot"]
                key = (h["branch"], slot)
                if key in seen:
                    continue
                seen.add(key)
                out = root / h["branch"] / "profiles" / f"slot_{slot:03d}"
                if (out / "complete.json").exists() or (out / "failure.json").exists():
                    continue
                r = h["record"]
                if not r.get("build_success"):
                    write(out / "failure.json", {"reason": "builder_failed", "node_id": r["node_id"]})
                    continue
                wanted.append((h["branch"], slot, template, out, r["node_id"]))
    # Read each generated SDF once per scan.
    by_branch = {}
    for relative, slot, template, out, node_id in wanted:
        if relative not in by_branch:
            mols = {}
            for mol in Chem.SDMolSupplier(str(root / relative / "molecules_all_built.sdf"), removeHs=True):
                if mol is not None and mol.HasProp("slot"):
                    mols[int(mol.GetProp("slot"))] = Chem.MolToMolBlock(mol)
            by_branch[relative] = mols
        block = by_branch[relative].get(slot)
        if block is None:
            write(out / "failure.json", {"reason": "missing_built_molecule", "node_id": node_id})
            continue
        yield {"molblock": block, "protein": template["protein"], "node_id": node_id, "out": str(out)}


def profile_one(task):
    from .vendor.profiling import profile_task
    try:
        return profile_task(task)
    except Exception as exc:
        result = {"error_type": type(exc).__name__, "error": str(exc), "node_id": task["node_id"]}
        write(Path(task["out"]) / "failure.json", result)
        return result


def profile(root, version, watch=False):
    root = Path(root)
    count = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        while True:
            batch = list(tasks(root, version))
            for _ in pool.map(profile_one, batch, chunksize=1):
                count += 1
            write(root / "quality_status.json", {"processed_this_session": count, "version": version, "time": time.time()})
            finished = (root / "versions" / version / "COMPLETE.json").exists()
            state = read(root / "status.json") if (root / "status.json").exists() else {}
            if not watch or finished or state.get("phase") == "attention":
                break
            time.sleep(30)
    return {"profiled_this_session": count}


def metrics(root, branch, slot, template):
    folder = root / branch / "profiles" / f"slot_{slot:03d}"
    return metrics_folder(folder, template)


def metrics_folder(folder, template):
    complete = folder / "complete.json"
    if not complete.exists():
        return {"quality_evaluated": False, "quality_failed": (folder / "failure.json").exists()}
    result = read(complete)
    p = read(folder / "interactions.json")
    teacher = {tuple(x) for x in template["fingerprints"]}
    current = {tuple(x) for x in p["fingerprints"]}
    return {**result["quality"], "quality_evaluated": True, "quality_failed": False,
            "plip_recovery": len(teacher & current) / len(teacher),
            "plip_new_fingerprints": len(current - teacher),
            "plip_interaction_count": len(p["interactions"])}
