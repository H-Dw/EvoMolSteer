"""Four-stage teacher-local calibration, at most 5,000 reserved attempts."""
import math
from pathlib import Path
from .io import read, write, frozen, value_sha
from .protocol import coarse_scales, choose
from .scale_domain import (POLICY, PHASE_COUNTS, PHASE_LIMITS, teacher_rows,
                           pareto, derive_domains, resolve_domain)


def fine_profiles(rows, target):
    coarse = teacher_rows(rows, target, "coarse")
    successful = [r for r in coarse if r.get("complete")]
    if not successful:
        raise ValueError("No finite calibration evidence for " + target)
    front = sorted(pareto(successful), key=lambda r: (r["C"], r["gap"], r["id"]))
    centers = [round(2*math.log2(r["schedule"][0])) for r in front]
    order = []
    for offset in [0, -1, 1, -2, 2]:
        for j in centers:
            if j+offset not in order:
                order.append(j+offset)
    low, high = min(centers)-4, max(centers)+4
    for k in range(33):
        j = round(low+(high-low)*k/32)
        if j not in order:
            order.append(j)
    old = {r["schedule"][0] for r in coarse}
    values = []
    for j in order:
        v = 2.**(j/2)
        if any(math.isclose(v, o, rel_tol=1e-12) for o in old):
            continue
        values.append([v]*4)
        if len(values) == PHASE_LIMITS["fine"]:
            break
    return values


def select_anchors(rows, target):
    candidates = [r for r in teacher_rows(rows, target, "fine") if r.get("complete")]
    if len(candidates) < 3:
        raise ValueError("At least three finite fine constants required for " + target)
    chosen = [choose(candidates)]
    # Cover both sides of this teacher's response center; never pool teachers.
    rest = sorted(candidates, key=lambda r: (
        abs(abs(math.log2(r["schedule"][0]/chosen[0]["schedule"][0]))-1), r["C"], r["gap"], r["id"]))
    for row in rest:
        if row not in chosen:
            chosen.append(row)
        if len(chosen) == 3:
            break
    return [r["schedule"][0] for r in chosen]


def phase_profiles(rows, target, phase):
    if phase == "coarse":
        return [[v]*4 for v in coarse_scales()]
    if phase == "fine":
        return fine_profiles(rows, target)
    profiles = []
    if phase == "windows":
        for anchor in select_anchors(rows, target):
            for window in range(4):
                for sign in [-1, 1]:
                    schedule = [anchor]*4
                    schedule[window] = 2.**(round(2*math.log2(anchor))/2+sign*.5)
                    profiles.append(schedule)
    elif phase == "full":
        for family in ["fine", "windows"]:
            pending = [r for r in teacher_rows(rows, target, family) if r.get("complete")]
            if family == "fine" and len(pending) < 3:
                raise ValueError("Insufficient teacher-local fine constants")
            for _ in range(min(10, len(pending))):
                winner = choose(pending)
                profiles.append(winner["schedule"])
                pending.remove(winner)
    else:
        raise ValueError("Unknown calibration phase")
    if len({tuple(s) for s in profiles}) != len(profiles) or len(profiles) > PHASE_LIMITS[phase]:
        raise ValueError("Duplicate profile or calibration phase budget exceeded")
    return profiles


def bind_calibration(folder, teachers):
    folder = Path(folder)
    policy = folder/"POLICY.json"
    # An existing shared calibration must remain readable with its old checkout.
    # Never reinterpret its profile IDs, failure ledger, domain or R0 in place.
    if not policy.exists() and folder.exists() and any(folder.iterdir()):
        raise ValueError("Existing calibration is frozen; use an explicitly authorized new protocol/root")
    return frozen(policy, dict(policy=POLICY, teachers={t["target"]: value_sha(t) for t in teachers},
        phase_counts=PHASE_COUNTS, phase_limits=PHASE_LIMITS,
        maximum_attempts=sum(PHASE_COUNTS[p]*PHASE_LIMITS[p] for p in PHASE_COUNTS)*len(teachers)))


def run(config):
    import torch
    from .prepare import load_model
    from .sampler import execute
    root = Path(config["root"])
    folder = root/"calibration"
    teachers = read(root/"teachers.json")
    if len(teachers) != 2 or len({t["target"] for t in teachers}) != 2:
        raise ValueError("Exactly two frozen teachers required")
    bind_calibration(folder, teachers)
    if (folder/"domain.json").exists():
        domain = read(folder/"domain.json")
        for teacher in teachers:
            resolve_domain(domain, teacher)
        return domain
    all_rows = []
    for phase, count in PHASE_COUNTS.items():
        by_teacher = {t["target"]: phase_profiles(all_rows, t["target"], phase) for t in teachers}
        frozen(folder/f"{phase}_profiles.json", dict(policy=POLICY, by_teacher=by_teacher))
        phase_rows = []
        for teacher in teachers:
            model = load_model(config, teacher)
            for index, schedule in enumerate(by_teacher[teacher["target"]]):
                out = folder/phase/teacher["target"]/f"p{index:03d}"
                failure = out/"CALIBRATION_FAILURE.json"
                if failure.exists():
                    row = read(failure)
                    if row["schedule"] != schedule or row["attempts"] != count:
                        raise ValueError("Failed calibration identity changed")
                else:
                    try:
                        row = execute(model, config, teacher, schedule, out, "calibration", count)
                    except FloatingPointError as error:
                        row = dict(id=str(out.relative_to(root)), schedule=schedule, complete=False,
                                   error=str(error), attempts=count)
                        write(failure, row)
                row = dict(row, target=teacher["target"], phase=phase)
                if phase == "full" and row["complete"]:
                    from .quality_queue import enqueue
                    row["quality"] = enqueue(config, out, teacher)
                phase_rows.append(row)
                print(dict(phase=phase, target=teacher["target"], profile=index, complete=row["complete"]), flush=True)
            del model
            torch.cuda.empty_cache()
        write(folder/f"{phase}_results.json", phase_rows)
        all_rows.extend(phase_rows)
    domain = derive_domains(all_rows, teachers)
    frozen(folder/"domain.json", domain)
    write(folder/"COMPLETE.json", dict(profiles=len(all_rows), domain=domain))
    return domain
