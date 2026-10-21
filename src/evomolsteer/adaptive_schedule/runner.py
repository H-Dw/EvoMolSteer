"""Durable per-slot histories, shared-prefix branches and bounded background execution."""
import contextlib
import os
import shutil
import time
import traceback
from pathlib import Path

from .io import (read, write, sha, value_sha, defaults, source_fingerprint, repository,
                 commit, better, eligible)
from .contracts import earliest
from .cohort import select_templates, counts, progress
from .grid import INITIAL_TICKS
from .population import round_context


def prepare(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    config = defaults()
    # Reject an existing cohort before touching its baseline copies or teacher list.
    if (root / "config.json").exists() and read(root / "config.json") != config:
        raise ValueError("Do not change a frozen experiment config; use a new cohort directory")
    if (root / "PREPARED.json").exists():
        assert_prepared(root)
        return {"root": str(root), **counts(config, read(root / "teachers.json")), "already_prepared": True}
    history = Path(config["history"])
    archive = read(history / "teachers.json")
    templates = select_templates(archive, config["teacher_selection"])
    if sorted({t["target"] for t in templates}) != sorted(config["targets"]):
        raise ValueError("Selected targets disagree with the frozen config")
    from .vendor.common import BASELINE_DATASETS, campaign, protein
    refs, baseline_rows = {}, {}
    for target in config["targets"]:
        for b in range(config["batches"]):
            p = history / "initials" / target / f"batch_{b:03d}.pt.gz"
            meta = read(p.with_suffix(".json"))
            if sha(p) != meta["state_file_sha256"] or meta["seed"] != config["seeds"][b]:
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
                for b in range(config["batches"]):
                    from .vendor.common import batch_folder
                    p = batch_folder(arm, target, b) / "final_records.json"
                    rows.extend(read(p))
                    refs[str(p)] = sha(p)
            else:
                rows = read(source)
                refs[str(source)] = sha(source)
            if len(rows) != config["batches"] * config["batch_size"]:
                raise ValueError("Historical target must retain exactly 100 attempts")
            baseline_rows[f"{target}__{arm}.json"] = rows
    for t in templates:
        for key in ["sdf", "protein"]:
            refs[t[key]] = sha(t[key])
    refs[str(history / "teachers.json")] = sha(history / "teachers.json")
    for name, rows in baseline_rows.items():
        write(root / "baselines" / name, rows)
    write(root / "config.json", config)
    write(root / "teachers.json", templates)
    write(root / "historical_references.json", refs)
    write(root / "teacher_selection.json", {
        "selected": config["teacher_selection"], "source_catalog": str(history / "teachers.json"),
        "source_catalog_sha256": refs[str(history / "teachers.json")],
        "excluded": [{"target": t["target"], "rank": t["rank"], "teacher_id": t["teacher_id"],
                      "node_id": t["source_record"]["node_id"]} for t in archive if t not in templates],
        "scope": config["reporting"]["scope"], "counts": counts(config, templates)})
    write(root / "PREPARED.json", {"commit": commit(), "source_fingerprint": source_fingerprint(),
                                 **counts(config, templates),
                                 "config_sha256": sha(root / "config.json"),
                                 "teachers_sha256": sha(root / "teachers.json"),
                                 "api_key_present": bool(os.environ.get("DEEPSEEK_API_KEY"))})
    return {"root": str(root), **counts(config, templates)}


def assert_prepared(root):
    root = Path(root)
    p = read(root / "PREPARED.json")
    if p["source_fingerprint"] != source_fingerprint():
        raise ValueError("Code changed since preparation; prepare and validate exact deployed commit")
    config = read(root / "config.json")
    for key, name in [("config_sha256", "config.json"), ("teachers_sha256", "teachers.json")]:
        if p.get(key) and p[key] != sha(root / name):
            raise ValueError("Prepared cohort file changed: " + name)
    templates = read(root / "teachers.json")
    if select_templates(templates, config["teacher_selection"]) != templates:
        raise ValueError("Prepared cohort contains unexpected or reordered teachers")
    return config


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
    rows = branch(model, config, root, template, batch, relative, [list(INITIAL_TICKS) for _ in range(20)],
                  [None] * 20, 0)
    return relative, rows


def initial_history(relative, rows):
    return [{"slot": i, "best_id": "r00", "history": [{
        "id": "r00", "branch": relative, "ticks": list(INITIAL_TICKS), "record": row,
        "parent_id": None, "improved_parent": False, "improved_incumbent": False
    }]} for i, row in enumerate(rows)]


def run_batch(model, root, config, template, batch, version, skill, fast, through_round=None, population=None, proposals=None):
    from .flowr import compose
    relative, rows = r0(model, config, root, template, batch)
    parent_folder = root / "versions" / version / template["teacher_id"] / f"batch_{batch:03d}"
    parent_folder.mkdir(parents=True, exist_ok=True)
    path = parent_folder / "histories.json"
    chains = read(path) if path.exists() else initial_history(relative, rows)
    write(path, chains)
    from .retention import enabled, compact_batch, restart_refs
    if enabled(config):
        compact_batch(root, config, template, batch, version, chains)
    limit = config["rounds"] if through_round is None else through_round
    for round_no in range(1, limit + 1):
        if all(len(c["history"]) > round_no for c in chains):
            continue
        if any(len(c["history"]) != round_no for c in chains):
            raise ValueError("Incomplete round commit; histories must advance atomically")
        if population is None or population["decision_round"] != round_no or population["teacher_id"] != template["teacher_id"]:
            raise ValueError("Agent proposals require frozen evidence for all 100 chains in this round")
        call_folder = parent_folder / f"round_{round_no:02d}"
        call_folder.mkdir(exist_ok=True)
        from .prefetch import ProposalPool, collect
        if proposals is not None:
            actions = collect(proposals, chains, population, call_folder)
        else:
            pool = ProposalPool(config["llm"]["concurrency"])
            try:
                bundle = pool.submit(root, config, template, chains, skill, call_folder, population)
                actions = collect(bundle, chains, population, call_folder)
            finally:
                pool.close()
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
    if all(len(c["history"]) == config["rounds"] + 1 for c in chains):
        write(parent_folder / "COMPLETE.json", {"chains": config["batch_size"], "rounds": config["rounds"]})


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
    if batch_index is not None:
        raise ValueError("Mean-first optimization advances all five batches together; select a teacher, not one batch")
    if not os.environ.get(config["llm"]["key_env"]):
        raise RuntimeError("DEEPSEEK_API_KEY is missing from the background worker environment")
    validation = read(root / "VALIDATED.json")
    if validation["source_fingerprint"] != source_fingerprint() or not validation["restart_pass"]:
        raise ValueError("Exact deployed source has not passed live restart validation")
    gradient = read(root / "validation/gradient_checks.json")
    if not gradient.get("passed") or gradient.get("source_fingerprint") != source_fingerprint():
        raise ValueError("Exact deployed source has not passed saved-state gradient checks")
    skill = skill or repository() / "skills/plip-schedule-optimizer/SKILL.md"
    if version == "v2":
        from .review_gate import validate_review
        validate_review(root, config, skill)
    frozen_skill = bind_version(root, version, skill)
    all_templates = read(root / "teachers.json")
    templates = all_templates
    if teacher_id:
        templates = [t for t in templates if t["teacher_id"] == teacher_id]
        if not templates:
            raise ValueError("Unknown teacher")
    from .flowr import load_model
    import torch
    failures = []
    with lock(root):
        state = progress(root, version, config, all_templates)
        if state["invalid"] or state["unexpected"]:
            raise ValueError("Cohort completion markers are inconsistent: " + str(state))
        model, loaded_target = None, None
        if config.get("execution", {}).get("scheduler") == "interleaved_prefetch":
            from .prefetch import interleaved_run
            failures = interleaved_run(root, config, templates, version, frozen_skill, validation["fast_batch_pass"])
        else:
            for template in templates:
                folders = [root / "versions" / version / template["teacher_id"] / f"batch_{b:03d}"
                           for b in range(config["batches"])]
                if all((f / "COMPLETE.json").exists() for f in folders):
                    continue
                if loaded_target != template["target"]:
                    if model is not None:
                        del model
                        torch.cuda.empty_cache()
                    model = load_model(config, template["target"], root)
                    loaded_target = template["target"]
                # R0 for all 100 chains, then R1 for all 100, etc. Every action in
                # a round observes the same frozen population, including on resume.
                for round_no in range(config["rounds"] + 1):
                    population = round_context(root, config, template, version, round_no) if round_no else None
                    round_failed = False
                    for batch, folder in enumerate(folders):
                        path = folder / "histories.json"
                        if round_no > 0 and path.exists() and all(len(c["history"]) > round_no for c in read(path)):
                            continue
                        for attempt in range(config["max_recoveries"]):
                            try:
                                write(root / "status.json", {"phase": "running", "version": version,
                                    "operation": "preparing_r0_population" if round_no == 0 else "optimizing_round",
                                    "teacher": template["teacher_id"], "batch": batch, "round": round_no,
                                    "time": time.time()})
                                run_batch(model, root, config, template, batch, version, frozen_skill,
                                          validation["fast_batch_pass"], round_no, population)
                                break
                            except Exception:
                                error = {"teacher": template["teacher_id"], "batch": batch, "round": round_no,
                                         "attempt": attempt + 1, "traceback": traceback.format_exc(), "time": time.time()}
                                write(folder / f"round_{round_no:02d}" / f"error_{attempt+1:02d}.json", error)
                                if attempt == config["max_recoveries"] - 1:
                                    failures.append(error)
                                    round_failed = True
                                else:
                                    time.sleep([30, 120][attempt])
                    if round_failed:
                        break
        state = progress(root, version, config, all_templates)
        phase = "review_ready" if version == "v1" else "complete"
        if state["complete"] and not failures:
            write(root / "versions" / version / "COMPLETE.json", {
                "chains": state["chains_per_version"], "rounds": config["rounds"],
                "teachers": state["teacher_count"], "batches": state["teacher_batches"],
                "decisions": state["decisions_per_version"], "teachers_sha256": sha(root / "teachers.json")})
            write(root / "status.json", {"phase": phase, "version": version, "time": time.time()})
        else:
            write(root / "status.json", {"phase": "attention" if failures else "partial", "version": version,
                  "cohort_progress": state, "errors": failures, "time": time.time()})
        if failures:
            raise RuntimeError("Cohort has failed batches; see status.json; successful branches preserved")
