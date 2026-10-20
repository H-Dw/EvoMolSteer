"""Compact, evidence-addressed DeepSeek decisions; each candidate owns its memory."""
from pathlib import Path

from evomolsteer.deepseek_api import call_json
from .contracts import schema, validate_action, step_allowances
from .io import read, write, canonical, value_sha


def node(root, branch, step, slot, cache=None):
    cache = {} if cache is None else cache
    key = (branch, step, slot)
    if key in cache:
        return cache[key]
    folder = Path(root) / branch
    nodes = read(folder / "nodes.json")
    if str(step) in nodes:
        row = nodes[str(step)]
        result = {k: (v[slot] if isinstance(v, list) and len(v) == 20 else v) for k, v in row.items()}
    else:
        parent = read(folder / "manifest.json")["parents"][slot]
        if parent is None:
            raise ValueError("Missing checkpoint diagnostic without a parent")
        result = node(root, parent, step, slot, cache)
    cache[key] = result
    return result


def step_rows(root, branch, slot):
    folder = Path(root) / branch
    manifest = read(folder / "manifest.json")
    prefix = []
    if manifest["start_step"]:
        prefix = [r for r in step_rows(root, manifest["parents"][slot], slot) if r["step"] < manifest["start_step"]]
    suffix = [{k: v[slot] if isinstance(v, list) and len(v) == 20 else v for k, v in r.items()
               if k not in {"rng_before", "rng_after"}} for r in read(folder / "steps.json")]
    return prefix + suffix


def packet(root, template, slot, history, best_id):
    ids, evidence, trajectories = [], {}, {}
    best = next(x for x in history if x["id"] == best_id)
    detailed_ids = {history[-1]["id"], best_id}
    cache = {}
    for item in history:
        identifier = item["id"] + ":result"
        ids.append(identifier)
        evidence[identifier] = {
            "record": item["record"], "ticks": item["ticks"], "parent_id": item.get("parent_id"),
            "improved_parent": item.get("improved_parent"), "improved_incumbent": item.get("improved_incumbent"),
            "hypothesis": item.get("proposal", {}).get("hypothesis"),
            "expected_change": item.get("proposal", {}).get("expected_change"),
            "memory_update": item.get("proposal", {}).get("memory_update")}
        curves = []
        for step in range(0, 101, 10):
            n = node(root, item["branch"], step, slot, cache)
            key = item["id"] + f":t{step:03d}"
            ids.append(key)
            keep = {"step": step, "affinity_head_at_t": n.get("affinity_head_at_t"),
                    "pic50_on_rescore": n.get("pic50_on_rescore"), "local_loss": n.get("local_loss")}
            if item["id"] in detailed_ids and step >= 50:
                keep.update({k: v for k, v in n.items() if k not in keep})
            evidence[key] = keep
            curves.append(key)
        trajectories[item["id"]] = curves
    # Actual displacement/clipping is essential: nominal lambda alone is misleading.
    diagnostics = {}
    for item in [best, history[-1]]:
        diagnostics[item["id"]] = step_rows(root, item["branch"], slot)
    data = {
        "teacher": {"id": template["teacher_id"], "pic50_on_rescore": template["source_record"]["pic50_on_rescore"],
                    "core_atoms": template["core_atoms"], "context_atoms": template["context_atoms"],
                    "core_coords_A": [template["coords_A"][i] for i in template["core_atoms"]],
                    "core_atom_labels": [template["atom_labels"][i] for i in template["core_atoms"]],
                    "core_charge_labels": [template["charge_labels"][i] for i in template["core_atoms"]]},
        "candidate_slot": slot, "round": len(history), "incumbent_id": best_id,
        "latest_id": history[-1]["id"], "evidence": evidence, "trajectories": trajectories,
        "step_diagnostics": diagnostics,
        "allowed_double_tick_directions_by_parent": {
            x["id"]: step_allowances(history, x["id"]) for x in history},
        "time_convention": "FLOWR t=0 prior/noise -> t=1 molecule; 100 steps; guidance only steps10..89",
        "affinity_note": "Intermediate head_at_t is diagnostic, not calibrated final pic50_on_rescore",
    }
    return data, ids


def estimated_tokens(text):
    # Conservative byte upper bound. API usage is the authoritative count.
    # JSON numerical data tokenize densely; avoid assuming 4 bytes per token.
    return len(text.encode("utf-8"))


def compact(data):
    """Aggregate repeated per-step diagnostics by interval, keeping extrema and clip counts."""
    import copy
    result = copy.deepcopy(data)
    for key, rows in result["step_diagnostics"].items():
        intervals = []
        for start in range(0, 100, 10):
            group = [r for r in rows if start <= r["step"] < start + 10]
            if not group:
                continue
            item = {"start_step": start, "count": len(group)}
            for metric in ["loss", "clip_factor", "assignment_switch_fraction", "gradient_drift_cosine",
                           "max_atom_guidance_A", "inside_rms_A", "outside_rms_A", "guidance_native_ratio"]:
                values = [r[metric] for r in group if r.get(metric) is not None]
                if values:
                    item[metric] = {"mean": sum(values) / len(values), "min": min(values), "max": max(values)}
            item["clipped_steps"] = sum(r["clip_factor"] < 1 for r in group)
            intervals.append(item)
        result["step_diagnostics"][key] = intervals
    result["diagnostic_compression"] = "interval mean/min/max; all raw steps retained in immutable external artifacts"
    return result


def propose(root, config, template, slot, history, best_id, skill, output, caller=call_json):
    data, ids = packet(root, template, slot, history, best_id)
    data = compact(data)
    instructions = Path(skill).read_text(encoding="utf-8")
    messages = [{"role": "system", "content": instructions},
                {"role": "user", "content": canonical(data)}]
    estimated = estimated_tokens(canonical(messages))
    limit = config["llm"]["input_hard_tokens"]
    if estimated > limit:
        # Never silently drop numerical evidence or failed trials.
        raise ValueError(f"Deduplicated prompt exceeds conservative input budget: {estimated}>{limit}")
    output = Path(output)
    write(output.with_suffix(".evidence.json"), data)
    write(output.with_suffix(".budget.json"),
          {"conservative_byte_token_upper_bound": estimated, "soft_tokens": config["llm"]["input_soft_tokens"],
           "hard_tokens": limit, "expanded_above_soft": estimated > config["llm"]["input_soft_tokens"],
           "evidence_sha256": value_sha(data)})
    action = caller(
        messages, schema(history, ids), output, model=config["llm"]["model"],
        timeout=config["llm"]["timeout"], attempts=config["llm"]["attempts"],
        input_store=Path(root) / "llm_inputs", max_output_tokens=config["llm"]["max_output_tokens"],
        response_validator=lambda a: validate_action(a, history, ids),
        thinking=config["llm"]["thinking"], reasoning_effort=config["llm"]["reasoning_effort"])
    validate_action(action, history, ids)
    return action
