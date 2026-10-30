"""Teacher-local adaptive calibration with immutable evidence and hard budgets."""
from dataclasses import asdict
from pathlib import Path
import math

from .io import read, write, frozen, value_sha, sha
from .calibration_search import (POLICY, EFFECTIVE, SearchLimits, SearchInsufficient,
                                 assess, strength, next_probe, panel_candidates,
                                 select_panel, derive_domain)


class TeacherSearch:
    def __init__(self, config, subject, binding, folder, executor, model, imported=()):
        self.config, self.subject, self.binding = config, subject, binding
        self.root, self.folder = Path(config["root"]), Path(folder)
        self.target = subject["teacher"]["target"]
        self.executor, self.model, self.limits = executor, model, SearchLimits()
        self.rows = {self.key(strength(r), r["attempts"]): dict(r, phase="coarse") for r in imported}
        self.new_scales, self.decisions = 0, []

    @staticmethod
    def key(scale, count):
        return (round(math.log2(scale), 12), count)

    def population(self, count):
        return [r for (_, n), r in self.rows.items() if n == count]

    def persist(self, stage):
        write(self.folder / "SEARCH_STATE.json", dict(
            policy=POLICY, stage=stage, teacher_id=self.target,
            new_screening_scales=self.new_scales, decisions=self.decisions,
            measurements=list(self.rows.values()), consumed_attempts=self.used()))

    def used(self):
        path = self.root / "attempts.json"
        return sum(r["count"] for r in (read(path) if path.exists() else {}).values()
                   if r["stage"] == "calibration" and r.get("teacher_id") == self.target)

    def measure(self, scales, count, phase):
        from .geometry_sampling import execute
        jobs, missing = [], []
        for scale in scales:
            key = self.key(scale, count)
            if key in self.rows:
                continue
            prior = [r["attempts"] for (s, _), r in self.rows.items() if s == key[0] and r["complete"]]
            missing.append((scale, count - max(prior, default=0)))
            jobs.append(dict(schedule=[scale] * 4, folder=self.folder / f"population_{count}" / value_sha(scale),
                             stage="calibration", count=count))
        if self.used() + sum(delta for _, delta in missing) > self.limits.maximum_attempts_per_teacher:
            raise SearchInsufficient("Teacher budget exhausted before allocating new candidates")
        if jobs:
            if self.executor is execute:
                from .geometry_batch import execute_many
                measured = execute_many(self.model, self.config, self.subject["teacher"], self.subject["condition"], jobs)
            else:
                measured = []
                for job in jobs:
                    try:
                        measured.append(self.executor(self.model, self.config, self.subject["teacher"],
                            self.subject["condition"], job["schedule"], job["folder"], "calibration", count))
                    except FloatingPointError as error:
                        measured.append(error)
            for job, result in zip(jobs, measured):
                scale = job["schedule"][0]
                if isinstance(result, FloatingPointError):
                    result = dict(id=job["folder"].relative_to(self.root).as_posix(),
                        teacher_id=self.target, condition_sha256=self.binding["condition_sha256"],
                        compatibility_sha256=value_sha(self.binding), complete=False, attempts=count,
                        population_is_requested_not_completed=True, error=str(result),
                        **{EFFECTIVE: [scale] * 4})
                    frozen(job["folder"] / "CALIBRATION_FAILURE.json", result)
                self.rows[self.key(scale, count)] = dict(result, phase=phase)
        self.persist(phase)
        return [self.rows[self.key(s, count)] for s in scales]

    def run(self, declared):
        self.measure(declared["coarse_actual_scales"], 10, "coarse")
        for _ in range(self.limits.maximum_bracket_probes):
            proposed = next_probe(self.population(10), self.limits)
            if proposed is None:
                break
            self.decisions.append(dict(operation="logarithmic_midpoint", actual_scale=proposed,
                                       same_configuration_retry=False))
            self.measure([proposed], 10, "bracket")
            self.new_scales += 1
        center = declared["search_center_actual_scale"]
        panels = []
        for panel in panel_candidates(self.population(10), center):
            missing = [s for s in panel if self.key(s, 10) not in self.rows]
            if self.new_scales + len(missing) > self.limits.maximum_new_screening_scales:
                continue
            self.measure(missing, 10, "integer_lattice_support")
            self.new_scales += len(missing)
            panels.append(panel)
        if not panels:
            raise SearchInsufficient("No bounded three-level integer support panel")
        candidates = sorted({s for p in panels for s in p})
        usable, _, _ = assess(self.population(10))
        allowed = {self.key(strength(r), 10) for r in usable}
        candidates = [s for s in candidates if self.key(s, 10) in allowed]
        if len(candidates) > self.limits.maximum_promoted_configurations:
            raise ValueError("Search planner exceeded promotion allocation")
        self.measure(candidates, 20, "refinement")
        selected = select_panel(self.population(20))
        self.decisions.append(dict(operation="confirm_integer_support_panel",
                                   actual_scales=[strength(r) for r in selected]))
        self.measure([strength(r) for r in selected], 50, "full")
        domain = derive_domain(list(self.rows.values()), self.target,
                               self.binding["condition_sha256"], value_sha(self.binding))
        self.persist("ready")
        return domain


def run(config, subjects, model_loader, executor):
    from .geometry_calibration import plan, verify_frozen
    from .geometry_sampling import compatibility
    from .calibration_import import import_measurements
    root = Path(config["root"])
    imported = import_measurements(config, subjects)
    frozen(root / "ATTEMPT_LIMITS.json", dict(
        policy=POLICY, stage_caps=dict(calibration=700, formal=1500),
        per_teacher_stage_caps=dict(calibration=350)))
    status_path=root / "calibration/TEACHER_STATUS.json"
    domains, statuses = {}, read(status_path) if status_path.exists() else {}
    for subject in subjects:
        target = subject["teacher"]["target"]
        binding = compatibility(config, subject["teacher"], subject["condition"])
        key = value_sha([target, subject["condition"]["condition_sha256"], value_sha(binding)])
        folder = root / "calibration/frozen" / key
        if (folder / "COMPLETE.json").exists():
            domains[target] = verify_frozen(root, read(folder / "domain.json"), binding)
            statuses.setdefault(target,dict(status="ready", domain_sha256=domains[target]["domain_sha256"]))
            continue
        if (folder / "INSUFFICIENT_EVIDENCE.json").exists():
            statuses.setdefault(target,dict(status="needs_evidence", evidence=str(folder / "INSUFFICIENT_EVIDENCE.json")))
            continue
        if (folder / "SEARCH_CLAIM.json").exists():
            raise RuntimeError("Interrupted adaptive search claim; explicit audit required")
        declared = plan(config, target)
        frozen(folder / "PLAN.json", dict(plan=declared, limits=asdict(SearchLimits()), sampling_protocol=binding))
        frozen(folder / "SEARCH_CLAIM.json", dict(search_policy=POLICY, teacher_id=target,
                                                   plan_sha256=sha(folder / "PLAN.json")))
        write(root / "BUDGET_SCOPE.json", dict(stage="calibration", teacher_id=target))
        model = model_loader(config, subject["teacher"])
        search = TeacherSearch(config, subject, binding, folder, executor, model, imported.get(target, ()))
        try:
            domain = search.run(declared)
            receipts = {r["id"]: sha(root / r["id"] / "COMPLETE.json")
                        for r in search.rows.values() if r["complete"]}
            frozen(folder / "domain.json", domain)
            frozen(folder / "all_results.json", list(search.rows.values()))
            frozen(folder / "COMPLETE.json", dict(domain_sha256=domain["domain_sha256"],
                measurement_receipts=receipts, fitted_once=True, search_is_local=True,
                boundary_best_allowed=True, promoted_slots_are_reused_not_independent=True))
            domains[target] = domain
            statuses[target] = dict(status="ready", domain_status=domain["domain_status"],
                                    consumed_attempts=search.used(), domain_sha256=domain["domain_sha256"])
        except SearchInsufficient as error:
            search.persist("needs_evidence")
            frozen(folder / "INSUFFICIENT_EVIDENCE.json", dict(error=str(error),
                completed_and_failed_evidence=list(search.rows.values()), no_automatic_retry=True,
                continue_other_teachers=True, consumed_attempts=search.used()))
            statuses[target] = dict(status="needs_evidence", reason=str(error), consumed_attempts=search.used())
        finally:
            search.model = model = None
            import torch
            torch.cuda.empty_cache()
        write(root / "calibration/TEACHER_STATUS.json", statuses)
    frozen(root / "calibration/domains.json", domains)
    frozen(root / "calibration/TEACHER_STATUS.json", statuses)
    marker = "COMPLETE.json" if len(domains) == len(subjects) else "PARTIAL.json"
    frozen(root / "calibration" / marker, dict(domain_sha256=value_sha(domains),
        calibration_keys={t: d["calibration_key"] for t, d in domains.items()},
        ready_teachers=list(domains), statuses=statuses, all_teachers_ready=len(domains) == len(subjects)))
    return domains
