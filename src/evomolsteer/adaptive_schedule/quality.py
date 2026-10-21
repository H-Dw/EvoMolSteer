"""Independent CPU profiling queue; never imports the FLOWR sampler or its RNG."""
import concurrent.futures
import multiprocessing
import time
from pathlib import Path

from .io import read, write


def selected_tasks(root, selections):
    from rdkit import Chem
    seen, wanted = set(), []
    for template, slot, h in selections:
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


def tasks(root, version):
    teachers = {t["teacher_id"]: t for t in read(root / "teachers.json")}
    selections = ((teachers[p.parents[1].name], c["slot"], h)
                  for p in sorted((root / "versions" / version).glob("*/batch_*/histories.json"))
                  for c in read(p) for h in c["history"])
    yield from selected_tasks(root, selections)


def profile_chains(root, template, chains):
    """Synchronous quality barrier: the worker cannot prune a pending profile input."""
    import importlib.util
    selections = [(template, c["slot"], h) for c in chains for h in c["history"]]
    pending = list(selected_tasks(root, selections))
    if not pending:
        return
    for module in ["plip", "posebusters", "openbabel"]:
        if importlib.util.find_spec(module) is None:
            raise RuntimeError("Quality dependency missing; retaining unprofiled structures: " + module)
    # The caller owns a loaded GPU model: do not fork its CUDA/HIP/thread state.
    with concurrent.futures.ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as pool:
        list(pool.map(profile_one, pending, chunksize=1))


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
    from .retention import enabled
    import contextlib
    from .runner import lock
    retention = enabled(read(root / "config.json"))
    if watch and retention:
        raise ValueError("TOP1 mode profiles synchronously before pruning; do not start a quality watcher")
    # Standalone profiling cannot race the retention-owning GPU worker.
    guard = lock(root) if retention else contextlib.nullcontext()
    with guard:
        return _profile(root, version, watch)


def _profile(root, version, watch):
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
