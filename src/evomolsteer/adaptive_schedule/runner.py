"""Durable per-slot histories, shared-prefix branches and bounded background execution."""
import concurrent.futures
import contextlib
import os
import shutil
import time
import traceback
from pathlib import Path

from .io import (read, write, sha, value_sha, defaults, source_fingerprint, repository,
                 commit, better, eligible, TARGETS, SEEDS)
from .contracts import earliest


def prepare(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    config = defaults()
    history = Path(config["history"])
    templates = read(history / "teachers.json")
    if len(templates) != 10 or any(not t["core_atoms"] for t in templates):
        raise ValueError("Exactly ten original, nonempty teacher templates are required")
    from .vendor.common import TEACHER_IDS, BASELINE_DATASETS, campaign, protein
    refs = {}
    for target in TARGETS:
        observed = [t["source_record"]["node_id"] for t in sorted(templates, key=lambda t: t["rank"]) if t["target"] == target]
        if observed != TEACHER_IDS[target]:
            raise ValueError("Frozen teacher ranking differs")
        for b in range(5):
            p = history / "initials" / target / f"batch_{b:03d}.pt.gz"
            meta = read(p.with_suffix(".json"))
            if sha(p) != meta["state_file_sha256"] or meta["seed"] != SEEDS[b]:
                raise ValueError("Historical prior provenance failed")
            refs[str(p)] = sha(p)
        refs[str(protein(target))] = sha(protein(target))
        for arm in BASELINE_DATASETS:
            if not Path(BASELINE_DATASETS[arm]).exists():
                raise ValueError("Historical baseline dataset missing")
            source = campaign(arm, target) / "final_records.json"
            # Original campaign versions may store final records in the arm subdirectory.
            if not source.exists():
                source = campaign(arm, target) / ("single" if arm == "steer" else "unguided") / "final_records.json"
            if not source.exists():
                rows = []
                for b in range(5):
                    from .vendor.common import batch_folder
                    p = batch_folder(arm, target, b) / "final_records.json"
                    rows.extend(read(p))
                    refs[str(p)] = sha(p)
            else:
                rows = read(source)
                refs[str(source)] = sha(source)
            if len(rows) != 100:
                raise ValueError("Historical target must retain exactly 100 attempts")
            write(root / "baselines" / f"{target}__{arm}.json", rows)
    for t in templates:
        for key in ["sdf", "protein"]:
            refs[t[key]] = sha(t[key])
    refs[str(history / "teachers.json")] = sha(history / "teachers.json")
    if (root / "config.json").exists() and read(root / "config.json") != config:
        raise ValueError("Do not change a frozen experiment config")
    write(root / "config.json", config)
    write(root / "teachers.json", templates)
    write(root / "historical_references.json", refs)
    write(root / "PREPARED.json", {"commit": commit(), "source_fingerprint": source_fingerprint(),
                                 "teacher_count": 10, "initial_batches": 10,
                                 "api_key_present": bool(os.environ.get("DEEPSEEK_API_KEY"))})
    return {"root": str(root), "teachers": 10, "chains_per_version": 1000}


def assert_prepared(root):
    p = read(root / "PREPARED.json")
    if p["source_fingerprint"] != source_fingerprint():
        raise ValueError("Code changed since preparation; prepare and validate exact deployed commit")
    return read(root / "config.json")


def bind_version(root, version, skill):
    folder = root / "versions" / version
    folder.mkdir(parents=True, exist_ok=True)
    skill = Path(skill)
    binding = {"source_fingerprint": source_fingerprint(), "skill_sha256": sha(skill),
               "config_sha256": sha(root / "config.json"), "teachers_sha256": sha(root / "teachers.json")}
    path = folder / "binding.json"
    if path.exists() and read(path) != binding:
        raise ValueError("Runtime or skill changed within a frozen cohort")
    write(path, binding)
    saved = folder / "SKILL.md"
    if saved.exists() and sha(saved) != sha(skill):
        raise ValueError("Frozen skill changed")
    if not saved.exists():
        shutil.copyfile(skill, saved)
    if not (folder / "deployment.json").exists():
        write(folder / "deployment.json", {"commit": commit(), "started": time.time()})
    return saved


def resolve_checkpoint(root, branch, step, cache=None):
    from .flowr import load_pt, compose
    cache = {} if cache is None else cache
    key = (branch, step)
    if key in cache:
        return cache[key]
    folder = root / branch
    p = folder / f"checkpoint_{step:03d}.pt.gz"
    if p.exists():
        state = load_pt(p)
    else:
        parents = read(folder / "manifest.json")["parents"]
        if any(x is None for x in parents):
            raise ValueError("Checkpoint absent and no complete parent references")
        state = compose([resolve_checkpoint(root, x, step, cache) for x in parents])
    cache[key] = state
    return state


def finish_nodes(root, folder, template, batch):
    from .flowr import load_pt, load_initial
    import torch
    from .vendor.reward import LocalReward
    output = load_pt(folder / "terminal.pt.gz")
    state = load_pt(folder / "checkpoint_100.pt.gz")
    initial, _ = load_initial(read(root / "config.json"), template["target"], batch)
    xyz = output["coords"]
    current = state["current"]["coords"] * initial["coord_scale"] + initial["target_com"][:, None]
    reward = LocalReward(template, "C", "endpoint")
    value, _, assignments, _ = reward.loss(current, xyz, {}, output["mask"].bool())
    ids = assignments[:, reward.core_columns]
    bi = torch.arange(20)[:, None]
    records = read(folder / "final_records.json")
    nodes = read(folder / "nodes.json")
    nodes["100"] = {
        "step": 100, "affinity_head_at_t": output["affinity"]["pic50"].reshape(20).tolist(),
        "pic50_on_rescore": [r["pic50_on_rescore"] for r in records], "local_loss": value.tolist(),
        "assignment": assignments.tolist(), "local_endpoint_A": xyz[bi, ids].tolist(),
        "local_current_A": current[bi, ids].tolist(),
        "local_atom_labels": output["atomics"].argmax(-1)[bi, ids].tolist(),
        "local_charge_labels": output["charges"].argmax(-1)[bi, ids].tolist(),
        "final_head_time": .9999}
    write(folder / "nodes.json", nodes)


def branch(model, config, root, template, batch, relative, schedules, parents, start, resume=None):
    from .flowr import execute
    folder = root / relative
    folder.mkdir(parents=True, exist_ok=True)
    manifest = {"teacher_id": template["teacher_id"], "batch": batch, "schedules": schedules,
                "parents": parents, "start_step": start, "source_fingerprint": source_fingerprint(),
                "template_sha256": value_sha(template)}
    p = folder / "manifest.json"
    if p.exists() and read(p) != manifest:
        raise ValueError("Branch ID reused with different scientific inputs")
    write(p, manifest)
    if (folder / "COMPLETE.json").exists():
        return read(folder / "final_records.json")
    attempts_path = folder / "recovery.json"
    recovery = read(attempts_path) if attempts_path.exists() else {"attempts": 0}
    if recovery["attempts"] >= config["max_recoveries"]:
        raise RuntimeError("Branch exhausted its recovery budget: " + relative)
    recovery["attempts"] += 1
    write(attempts_path, recovery)
    try:
        rows = execute(model, config, template, batch, schedules, folder,
                       root / "contexts" / template["target"] / f"batch_{batch:03d}.pt.gz", resume)
        finish_nodes(root, folder, template, batch)
        write(folder / "COMPLETE.json", {"attempts": 20, "start_step": start, "end_step": 100})
        return rows
    except Exception:
        write(folder / f"failure_{recovery['attempts']:02d}.json", {"traceback": traceback.format_exc(), "time": time.time()})
        raise


def r0(model, config, root, template, batch):
    relative = f"branches/{template['teacher_id']}/batch_{batch:03d}/r0"
    rows = branch(model, config, root, template, batch, relative, [[5] * 8 for _ in range(20)],
                  [None] * 20, 0)
    return relative, rows


def initial_history(relative, rows):
    return [{"slot": i, "best_id": "r00", "history": [{
        "id": "r00", "branch": relative, "ticks": [5] * 8, "record": row,
        "parent_id": None, "improved_parent": False, "improved_incumbent": False
    }]} for i, row in enumerate(rows)]


def run_batch(model, root, config, template, batch, version, skill, fast):
    from .flowr import compose
    from .agent import propose
    relative, rows = r0(model, config, root, template, batch)
    parent_folder = root / "versions" / version / template["teacher_id"] / f"batch_{batch:03d}"
    parent_folder.mkdir(parents=True, exist_ok=True)
    path = parent_folder / "histories.json"
    chains = read(path) if path.exists() else initial_history(relative, rows)
    write(path, chains)
    from .retention import enabled, compact_batch, restart_refs
    if enabled(config):
        compact_batch(root, config, template, batch, version, chains)
    for round_no in range(1, config["rounds"] + 1):
        if all(len(c["history"]) > round_no for c in chains):
            continue
        if any(len(c["history"]) != round_no for c in chains):
            raise ValueError("Incomplete round commit; histories must advance atomically")
        call_folder = parent_folder / f"round_{round_no:02d}"
        call_folder.mkdir(exist_ok=True)
        actions = [None] * 20
        with concurrent.futures.ThreadPoolExecutor(max_workers=config["llm"]["concurrency"]) as executor:
            tasks = {executor.submit(propose, root, config, template, c["slot"], c["history"], c["best_id"],
                                     skill, call_folder / f"slot_{c['slot']:03d}.json"): c["slot"] for c in chains}
            failures = []
            for future in concurrent.futures.as_completed(tasks):
                slot = tasks[future]
                try:
                    actions[slot] = future.result()
                except Exception:
                    failures.append({"slot": slot, "traceback": traceback.format_exc()})
            if failures:
                write(call_folder / "errors.json", failures)
                raise RuntimeError(f"LLM proposal failure in {len(failures)} slots; successful calls retained")
        parents = [next(h for h in c["history"] if h["id"] == a["parent_id"]) for c, a in zip(chains, actions)]
        starts = [earliest(p["ticks"], a["ticks"]) for p, a in zip(parents, actions)]
        schedules = [a["ticks"] for a in actions]
        parent_refs = [p["branch"] for p in parents]
        if enabled(config):
            if any(a["parent_id"] != c["best_id"] for a, c in zip(actions, chains)):
                raise ValueError("A pruned historical branch cannot be selected as restart parent")
            parent_refs = restart_refs(root, version, template["teacher_id"], batch, chains)
        cache = {}
        use_fast = fast
        if use_fast:
            start = min(starts)
            try:
                resume = compose([resolve_checkpoint(root, p, start, cache) for p in parent_refs])
            except ValueError as exc:
                use_fast = False
                write(call_folder / "strict_fallback.json", {"reason": str(exc)})
        if use_fast:
            new_ref = f"branches/{template['teacher_id']}/batch_{batch:03d}/{version}_r{round_no:02d}"
            outputs = branch(model, config, root, template, batch, new_ref, schedules, parent_refs, start, resume)
            refs = [new_ref] * 20
        else:
            outputs, refs = [], []
            for slot in range(20):
                p = parent_refs[slot]
                full = read(root / p / "manifest.json")["schedules"]
                full[slot] = schedules[slot]
                resume = resolve_checkpoint(root, p, starts[slot], cache)
                ref = f"branches/{template['teacher_id']}/batch_{batch:03d}/{version}_r{round_no:02d}_s{slot:03d}"
                result = branch(model, config, root, template, batch, ref, full, [p] * 20, starts[slot], resume)
                refs.append(ref)
                outputs.append(result[slot])
        for slot, (chain, action, parent, record) in enumerate(zip(chains, actions, parents, outputs)):
            old_best = next(h for h in chain["history"] if h["id"] == chain["best_id"])
            item = {"id": f"r{round_no:02d}", "branch": refs[slot], "ticks": action["ticks"],
                    "record": record, "parent_id": action["parent_id"], "proposal": action,
                    "improved_parent": better(record, parent["record"]),
                    "improved_incumbent": better(record, old_best["record"]),
                    "call": str((call_folder / f"slot_{slot:03d}.json").relative_to(root))}
            chain["history"].append(item)
            if item["improved_incumbent"]:
                chain["best_id"] = item["id"]
        write(path, chains)
        write(call_folder / "COMPLETE.json", {"chains": 20, "round": round_no, "fast_batch": use_fast})
        if enabled(config):
            compact_batch(root, config, template, batch, version, chains)
        write(root / "status.json", {"phase": "running", "version": version,
              "teacher": template["teacher_id"], "batch": batch, "round": round_no,
              "time": time.time(), "histories": str(path.relative_to(root))})
    write(parent_folder / "COMPLETE.json", {"chains": 20, "rounds": 10})


@contextlib.contextmanager
def lock(root):
    import fcntl
    root.mkdir(parents=True, exist_ok=True)
    with (root / "worker.lock").open("a") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def run(root, version, skill=None, teacher_id=None, batch_index=None):
    root = Path(root)
    config = assert_prepared(root)
    if not os.environ.get(config["llm"]["key_env"]):
        raise RuntimeError("DEEPSEEK_API_KEY is missing from the background worker environment")
    validation = read(root / "VALIDATED.json")
    if validation["source_fingerprint"] != source_fingerprint() or not validation["restart_pass"]:
        raise ValueError("Exact deployed source has not passed live restart validation")
    if version == "v2" and not (root / "versions/v1/COMPLETE.json").exists():
        raise ValueError("V2 requires a complete V1 cohort and main-agent review")
    if version == "v2" and not (root / "versions/v2/review_approval.json").exists():
        raise ValueError("V2 requires the committed main-agent review coverage and skill revision")
    skill = skill or repository() / "skills/plip-schedule-optimizer/SKILL.md"
    frozen_skill = bind_version(root, version, skill)
    templates = read(root / "teachers.json")
    if teacher_id:
        templates = [t for t in templates if t["teacher_id"] == teacher_id]
        if not templates:
            raise ValueError("Unknown teacher")
    from .flowr import load_model
    import torch
    failures = []
    with lock(root):
        model, loaded_target = None, None
        for template in templates:
            for batch in ([batch_index] if batch_index is not None else range(5)):
                folder = root / "versions" / version / template["teacher_id"] / f"batch_{batch:03d}"
                if (folder / "COMPLETE.json").exists():
                    continue
                if loaded_target != template["target"]:
                    if model is not None:
                        del model
                        torch.cuda.empty_cache()
                    model = load_model(config, template["target"], root)
                    loaded_target = template["target"]
                for attempt in range(config["max_recoveries"]):
                    try:
                        run_batch(model, root, config, template, batch, version, frozen_skill,
                                  validation["fast_batch_pass"])
                        break
                    except Exception:
                        error = {"teacher": template["teacher_id"], "batch": batch,
                                 "attempt": attempt + 1, "traceback": traceback.format_exc(), "time": time.time()}
                        write(folder / f"error_{attempt+1:02d}.json", error)
                        if attempt == config["max_recoveries"] - 1:
                            failures.append(error)
                        else:
                            time.sleep([30, 120][attempt])
        completed = list((root / "versions" / version).glob("*/batch_*/COMPLETE.json"))
        phase = "review_ready" if version == "v1" else "complete"
        if len(completed) == 50 and not failures:
            write(root / "versions" / version / "COMPLETE.json", {"chains": 1000, "rounds": 10})
            write(root / "status.json", {"phase": phase, "version": version, "time": time.time()})
        else:
            write(root / "status.json", {"phase": "attention" if failures else "partial", "version": version,
                  "completed_batches": len(completed), "errors": failures, "time": time.time()})
        if failures:
            raise RuntimeError("Cohort has failed batches; see status.json; successful branches preserved")
