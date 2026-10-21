"""Materialize the twenty incumbents without changing native batch/RNG semantics."""
from contextlib import ExitStack
from pathlib import Path
import shutil

from .io import read, write, sha, value_sha
from .retention import checked_path
from .grid import CHECKPOINT_STEPS, HEAD_STEPS


def payload_branch(root, relative):
    manifest = read(Path(root) / relative / "manifest.json")
    return manifest.get("payload_branch", relative)


def payload_source(root, relative, step, filename, slot):
    relative = payload_branch(root, relative)
    folder = Path(root) / relative
    if (folder / filename).exists():
        return folder / filename
    manifest = read(folder / "manifest.json")
    parent = manifest["parents"][slot]
    if parent is None or step >= manifest["start_step"]:
        raise ValueError("Required incumbent payload is missing: " + str(folder / filename))
    return payload_source(root, parent, step, filename, slot)


def trace_segments(root, relative, slot):
    relative = payload_branch(root, relative)
    folder = Path(root) / relative
    manifest = read(folder / "manifest.json")
    start = manifest["start_step"]
    prefix = []
    if start:
        for p, a, b in trace_segments(root, manifest["parents"][slot], slot):
            if a < start:
                prefix.append((p, a, min(start, b)))
    return prefix + [(folder / "trace.h5", start, 100)]


def materialize(root, relative, state_sources, payload_sources, best, template, batch):
    import h5py
    from rdkit import Chem
    from .flowr import load_pt, save_pt, compose, tensor_digest, tree
    from .runner import resolve_checkpoint
    from .agent import node, step_rows

    root = Path(root)
    folder = checked_path(root, relative, "retained")
    folder.mkdir(parents=True, exist_ok=True)
    manifest = {"teacher_id": template["teacher_id"], "batch": batch,
                "start_step": 0, "parents": [None] * 20,
                "schedules": [h["ticks"] for h in best], "retained_incumbents": True}
    write(folder / "manifest.json", manifest)
    # Inspect every diagnostic boundary and exact interval entry. If any context differs, preserve full
    # native batch contexts per slot instead of silently forcing a mixed batch.
    mixed = True
    for step in CHECKPOINT_STEPS:
        cache = {}
        states = [resolve_checkpoint(root, s, step, cache) for s in state_sources]
        try:
            packed = compose(states)
        except ValueError:
            mixed = False
            break
        save_pt(folder / f"checkpoint_{step:03d}.pt.gz", packed)
        if tensor_digest(load_pt(folder / f"checkpoint_{step:03d}.pt.gz")) != tensor_digest(packed):
            raise ValueError("Compacted checkpoint failed its round-trip check")
    if mixed:
        restart_refs = [relative] * 20
    else:
        # Remove only our incomplete mixed attempt; no published archive is edited.
        for p in folder.glob("checkpoint_*.pt.gz"):
            p.unlink()
        restart_refs = [relative + f"/context_{i:03d}" for i in range(20)]
        for slot, ref in enumerate(restart_refs):
            write(root / ref / "manifest.json", dict(read(root / state_sources[slot] / "manifest.json"),
                  payload_branch=relative, parents=[None] * 20, start_step=0))
        for step in CHECKPOINT_STEPS:
            cache = {}
            for slot, ref in enumerate(restart_refs):
                state = resolve_checkpoint(root, state_sources[slot], step, cache)
                dest = root / ref / f"checkpoint_{step:03d}.pt.gz"
                save_pt(dest, state)
                if tensor_digest(load_pt(dest)) != tensor_digest(state):
                    raise ValueError("Full-context checkpoint failed its round-trip check")

    def select(value, slot):
        if isinstance(value, dict) and "_retained_slots" in value:
            return value["_retained_slots"][slot]
        return tree(value, lambda t: t[slot:slot+1].clone() if t.ndim and t.shape[0] == 20 else t.clone())

    for step in HEAD_STEPS:
        cache, values = {}, []
        for slot, source in enumerate(payload_sources):
            p = payload_source(root, source, step, f"head_{step:03d}.pt.gz", slot)
            if p not in cache:
                cache[p] = load_pt(p)
            values.append(select(cache[p], slot))
        save_pt(folder / f"head_{step:03d}.pt.gz", {"_retained_slots": values})
    for filename in ["terminal.pt.gz", "rng_terminal.pt.gz"]:
        cache, values = {}, []
        for slot, source in enumerate(payload_sources):
            p = root / source / filename
            if p not in cache:
                cache[p] = load_pt(p)
            values.append(select(cache[p], slot))
        save_pt(folder / filename, {"_retained_slots": values})
    # Actual noise/velocity/probabilities are copied from each winner's own prefix
    # and suffix, not reconstructed from coordinate differences.
    with ExitStack() as stack, h5py.File(folder / "trace.h5", "w") as dest:
        opened = {}
        for slot, source in enumerate(payload_sources):
            for p, start, end in trace_segments(root, source, slot):
                if p not in opened:
                    opened[p] = stack.enter_context(h5py.File(p, "r"))
                for key, data in opened[p].items():
                    if data.shape[:2] != (100, 20):
                        raise ValueError("Unrecognized native trace batch layout")
                    if key not in dest:
                        dest.create_dataset(key, shape=data.shape, dtype=data.dtype,
                                            chunks=(1, 1) + data.shape[2:], compression="gzip", compression_opts=1)
                    dest[key][start:end, slot] = data[start:end, slot]
    nodes = {}
    for step in CHECKPOINT_STEPS:
        rows = [node(root, h["branch"], step, i) for i, h in enumerate(best)]
        keys = set().union(*(r.keys() for r in rows)) - {"step"}
        nodes[str(step)] = {"step": step, **{k: [r.get(k) for r in rows] for k in keys}}
    write(folder / "nodes.json", nodes)
    trajectories = [step_rows(root, h["branch"], i) for i, h in enumerate(best)]
    if any([r["step"] for r in rows] != list(range(100)) for rows in trajectories):
        raise ValueError("Incomplete per-step documentary history")
    steps = []
    for step in range(100):
        rows = [r[step] for r in trajectories]
        keys = set().union(*(r.keys() for r in rows)) - {"step"}
        steps.append({"step": step, **{k: [r.get(k) for r in rows] for k in keys}})
    write(folder / "steps.json", steps)
    write(folder / "final_records.json", [h["record"] for h in best])
    molecules = {}
    for source in set(payload_sources):
        sdf = root / source / "molecules_all_built.sdf"
        molecules[source] = {int(m.GetProp("slot")): m for m in
                             Chem.SDMolSupplier(str(sdf), removeHs=False)
                             if m is not None and m.HasProp("slot")} if sdf.exists() and sdf.stat().st_size else {}
    writer = Chem.SDWriter(str(folder / "molecules_all_built.sdf"))
    try:
        for slot, source in enumerate(payload_sources):
            mol = molecules[source].get(slot)
            if best[slot]["record"].get("build_success") and mol is None:
                raise ValueError("Built incumbent molecule is missing")
            if mol is not None:
                writer.write(mol)
            profile = root / source / "profiles" / f"slot_{slot:03d}"
            if not profile.exists():
                raise ValueError("Incumbent quality documents are missing")
            shutil.copytree(profile, folder / "profiles" / f"slot_{slot:03d}", dirs_exist_ok=True)
    finally:
        writer.close()
    identity = [{"id": h["id"], "branch": h["branch"], "slot": i} for i, h in enumerate(best)]
    payloads = {str(p.relative_to(folder)): sha(p) for p in sorted(folder.rglob("*"))
                if p.is_file() and p.name != "COMPLETE.json"}
    write(folder / "COMPLETE.json", {"identity_sha256": value_sha(identity), "restart_refs": restart_refs,
          "native_context_mode": "mixed" if mixed else "full_batch_per_slot", "payload_sha256": payloads,
          "retained_attempts": 20})
