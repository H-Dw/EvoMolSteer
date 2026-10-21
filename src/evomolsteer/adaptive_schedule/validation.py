"""Real native replay checks using scientific R0 states, with no zero-guidance cohort."""
from pathlib import Path

from .io import read, write, source_fingerprint
from .runner import assert_prepared, r0, branch, resolve_checkpoint


def equal_tree(a, b):
    import torch
    if torch.is_tensor(a):
        return torch.is_tensor(b) and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal_tree(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal_tree(x, y) for x, y in zip(a, b))
    return a == b


def select(value, slots):
    import torch
    if torch.is_tensor(value):
        return value[slots] if value.ndim and value.shape[0] == 20 else value
    if isinstance(value, dict):
        return {k: select(v, slots) for k, v in value.items()}
    return value


def validate(root):
    import torch
    from .flowr import load_model, load_pt, compose, tensor_digest
    root = Path(root)
    config = assert_prepared(root)
    templates = [t for t in read(root / "teachers.json") if t["rank"] == 1]
    results = []
    for template in templates:
        model = load_model(config, template["target"], root)
        ref, rows = r0(model, config, root, template, 0)
        terminal = load_pt(root / ref / "terminal.pt.gz")
        terminal_rng = load_pt(root / ref / "rng_terminal.pt.gz")
        checks = []
        for step in range(0, 101, 10):
            out = f"validation/{template['teacher_id']}/resume_{step:03d}"
            records = branch(model, config, root, template, 0, out, [[5] * 8 for _ in range(20)],
                             [ref] * 20, step, resolve_checkpoint(root, ref, step))
            exact = equal_tree(terminal, load_pt(root / out / "terminal.pt.gz"))
            rng_exact = tensor_digest(terminal_rng) == tensor_digest(load_pt(root / out / "rng_terminal.pt.gz"))
            score_exact = all(a["pic50_on_rescore"] == b["pic50_on_rescore"] for a, b in zip(rows, records))
            check = {"step": step, "terminal_bitwise": exact, "rng_bitwise": rng_exact,
                     "rescore_identical": score_exact}
            checks.append(check)
            write(root / "validation/progress.json", {"teacher": template["teacher_id"], "check": check})
            if not (exact and rng_exact and score_exact):
                write(root / "validation/FAILED.json", {"teacher": template["teacher_id"], "checks": checks})
                raise ValueError("Native checkpoint restart is not exact; cohort cannot start")
        # Two distinct branch histories, then mixed-row restart vs one full-prefix reference.
        changed_refs = []
        for slot in range(2):
            schedules = [[5] * 8 for _ in range(20)]
            schedules[slot][2] = 4 if slot == 0 else 6
            out = f"validation/{template['teacher_id']}/change_slot_{slot}"
            branch(model, config, root, template, 0, out, schedules, [ref] * 20, 30,
                   resolve_checkpoint(root, ref, 30))
            changed_refs.append(out)
        altered = load_pt(root / changed_refs[0] / "terminal.pt.gz")
        isolated = equal_tree(select(terminal, list(range(1, 20))), select(altered, list(range(1, 20))))
        schedules = [[5] * 8 for _ in range(20)]
        schedules[0][2], schedules[1][2] = 4, 6
        full = f"validation/{template['teacher_id']}/combined_full"
        branch(model, config, root, template, 0, full, schedules, [ref] * 20, 30,
               resolve_checkpoint(root, ref, 30))
        parent_refs = [changed_refs[0], changed_refs[1]] + [ref] * 18
        mixed = f"validation/{template['teacher_id']}/combined_resume"
        try:
            state = compose([resolve_checkpoint(root, x, 50) for x in parent_refs])
            branch(model, config, root, template, 0, mixed, schedules, parent_refs, 50, state)
            combined_exact = equal_tree(load_pt(root / full / "terminal.pt.gz"), load_pt(root / mixed / "terminal.pt.gz"))
        except ValueError:
            combined_exact = False
        results.append({"teacher": template["teacher_id"], "restart": checks,
                        "slot_isolated": isolated, "mixed_parent_bitwise": combined_exact})
        del model
        torch.cuda.empty_cache()
    report = {"source_fingerprint": source_fingerprint(), "restart_pass": True,
              "fast_batch_pass": all(r["slot_isolated"] and r["mixed_parent_bitwise"] for r in results),
              "targets": results, "zero_guidance_groups": 0,
              "historical_slot_pairing": False}
    write(root / "VALIDATED.json", report)
    from .retention import enabled, prune_payload
    if enabled(config):
        for p in sorted((root / "validation").glob("*/*/COMPLETE.json")):
            prune_payload(root, p.parent.relative_to(root), "validation")
        write(root / "validation/RETENTION_COMPLETE.json", {
            "raw_validation_payloads_removed_after_pass": True, "shared_r0_preserved": True})
    return report
