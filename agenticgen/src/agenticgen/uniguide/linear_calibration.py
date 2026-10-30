"""Budgeted linear screening with upper-neighbour coverage and frozen references."""

from pathlib import Path

from . import linear_scale as scale
from .calibration_search import assess, best, SearchInsufficient
from .io import read, write, frozen, value_sha, sha


def plan(config, target):
    settings = scale.teacher_settings(
        scale.configure(config)["guidance_control"], target
    )
    return dict(
        policy=scale.POLICY,
        control_settings=settings,
        coarse_actual_scales=settings["initial_scales"],
        populations=[10, 20, 50],
        reference_budget=settings["maximum_attempts_per_teacher"]
        - settings["range_evidence_attempts"],
    )


def candidate_order(rows, settings):
    usable, _, _ = assess(rows)
    if not usable:
        return []
    center = best(usable)[scale.EFFECTIVE][0]
    available = scale.grid(settings)
    return sorted(available, key=lambda x: (abs(x - center), x < center, x))


def next_screen(rows, settings):
    tested = {r[scale.EFFECTIVE][0] for r in rows}
    usable, _, boundary = assess(rows)
    if not usable:
        return None
    center = best(usable)[scale.EFFECTIVE][0]
    unit = settings[scale.UNIT]
    # Obtain the next two upper neighbours before spending on far lower support.
    choices = [
        center + unit,
        center + 2 * unit,
        center - unit,
        *candidate_order(rows, settings),
    ]
    for value in choices:
        if value not in scale.grid(settings) or value in tested:
            continue
        if boundary is not None and value >= boundary["effective_scale"]:
            continue
        return value
    return None


def promotion_levels(rows, settings):
    usable, _, _ = assess(rows)
    by_scale = {r[scale.EFFECTIVE][0]: r for r in usable}
    if not by_scale:
        return []
    center = best(usable)[scale.EFFECTIVE][0]
    unit = settings[scale.UNIT]
    order = [
        center,
        center + unit,
        center + 2 * unit,
        center - unit,
        *candidate_order(rows, settings),
    ]
    selected = []
    for value in order:
        if value in by_scale and value not in selected:
            selected.append(value)
        if len(selected) == settings["maximum_confirmation_levels"]:
            break
    return selected


def derive_domain(
    rows, target, condition_hash, protocol_hash, settings, exploration=None
):
    full = [r for r in rows if r["attempts"] == 50]
    if any(
        r["teacher_id"] != target
        or r["condition_sha256"] != condition_hash
        or r["compatibility_sha256"] != protocol_hash
        for r in full
    ):
        raise ValueError("Incompatible full-population linear calibration")
    usable, curve, boundary = assess(full)
    if not usable:
        raise SearchInsufficient("No complete eligible constant reference")
    winner = best(usable)
    reference = winner[scale.EFFECTIVE][0]
    levels = sorted(r[scale.EFFECTIVE][0] for r in usable)
    body = dict(
        scale_policy=scale.POLICY,
        teacher_id=target,
        condition_sha256=condition_hash,
        sampling_protocol_sha256=protocol_hash,
        calibration_key=value_sha([target, condition_hash, protocol_hash]),
        **{scale.REFERENCE: reference, scale.UNIT: settings[scale.UNIT]},
        control_settings=settings,
        initial_source=winner["id"],
        initial_guidance_scale_tick_schedule=[0] * 4,
        bootstrap_sources=[r["id"] for r in usable],
        confirmed_effective_levels=levels,
        confirmed_profiles=[],
        constant_comparison=curve,
        risk_boundary=boundary,
        evidence_revision=0,
        source_rows_sha256=value_sha(rows),
        boundary_status=(
            "declared_limit"
            if reference == settings["search_maximum"]
            else (
                "observed_boundary"
                if boundary
                and reference + settings[scale.UNIT] >= boundary["effective_scale"]
                else (
                    "interior_supported"
                    if reference + settings[scale.UNIT] in levels
                    else "coverage_limited"
                )
            )
        ),
        stability_guarantee=False,
        global_optimum_claimed=False,
    )
    if exploration is not None:
        body["exploration_policy"] = exploration
    return scale.seal(body)


def used(root, target):
    path = Path(root) / "attempts.json"
    return sum(
        r["count"]
        for r in (read(path) if path.exists() else {}).values()
        if r["stage"] == "calibration" and r.get("teacher_id") == target
    )


def ensure_limits(config, subject_count):
    return frozen(
        Path(config["root"]) / "ATTEMPT_LIMITS.json",
        dict(
            policy=scale.POLICY,
            stage_caps=dict(
                calibration=350 * subject_count,
                formal=min(1500, 50 * 3 * config["rounds"] * subject_count),
            ),
            per_teacher_stage_caps=dict(calibration=350),
        ),
    )


def evaluate(model, config, subject, job, executor):
    from .geometry_sampling import execute

    if executor is execute:
        from .geometry_batch import execute_many

        return execute_many(
            model, config, subject["teacher"], subject["condition"], [job]
        )[0]
    try:
        return executor(
            model,
            config,
            subject["teacher"],
            subject["condition"],
            job["schedule"],
            job["folder"],
            job["stage"],
            job.get("count", 50),
            **({"domain": job["domain"]} if job.get("domain") else {}),
        )
    except FloatingPointError as error:
        return error


class LinearSearch:
    def __init__(self, config, subject, binding, folder, model, executor):
        self.config, self.subject, self.binding = config, subject, binding
        self.root, self.folder = Path(config["root"]), Path(folder)
        self.target = subject["teacher"]["target"]
        self.model, self.executor = model, executor
        self.declared = plan(config, self.target)
        self.settings = self.declared["control_settings"]
        self.rows, self.decisions = {}, []

    def persist(self, phase):
        write(
            self.folder / "SEARCH_STATE.json",
            dict(
                policy=scale.POLICY,
                teacher_id=self.target,
                stage=phase,
                measurements=list(self.rows.values()),
                decisions=self.decisions,
                consumed_attempts=used(self.root, self.target),
            ),
        )

    def measure(self, value, count):
        key = (value, count)
        if key in self.rows:
            return self.rows[key]
        prior = [r for (s, _), r in self.rows.items() if s == value]
        if any(not r["complete"] for r in prior):
            raise ValueError(
                "Failed constant configuration cannot be promoted or retried"
            )
        delta = count - max((r["attempts"] for r in prior), default=0)
        if used(self.root, self.target) + delta > self.declared["reference_budget"]:
            raise SearchInsufficient(
                "Reference budget exhausted; range evidence reserve retained"
            )
        folder = self.folder / f"population_{count}" / value_sha(value)
        job = dict(
            schedule=[value] * 4, folder=folder, stage="calibration", count=count
        )
        result = evaluate(self.model, self.config, self.subject, job, self.executor)
        if isinstance(result, FloatingPointError):
            result = dict(
                id=folder.relative_to(self.root).as_posix(),
                teacher_id=self.target,
                condition_sha256=self.binding["condition_sha256"],
                compatibility_sha256=value_sha(self.binding),
                complete=False,
                attempts=count,
                error=str(result),
                **{scale.EFFECTIVE: [value] * 4},
            )
            frozen(folder / "CALIBRATION_FAILURE.json", result)
        self.rows[key] = dict(result, phase="full" if count == 50 else "screening")
        self.persist(f"population_{count}")
        return self.rows[key]

    def population(self, count):
        return [r for (_, n), r in self.rows.items() if n == count]

    def run(self):
        for value in self.settings["initial_scales"]:
            self.measure(value, 10)
        while len(self.population(10)) < self.settings["maximum_screening_levels"]:
            value = next_screen(self.population(10), self.settings)
            if value is None:
                break
            self.decisions.append(
                dict(operation="linear_neighbour", actual_scale=value)
            )
            self.measure(value, 10)
        for value in promotion_levels(self.population(10), self.settings):
            self.measure(value, 20)
        for value in promotion_levels(self.population(20), self.settings):
            self.measure(value, 50)
        return derive_domain(
            list(self.rows.values()),
            self.target,
            self.binding["condition_sha256"],
            value_sha(self.binding),
            self.settings,
            self.config.get("guidance_exploration"),
        )


def run(config, subjects, model_loader, executor):
    import torch
    from .geometry_sampling import compatibility
    from .geometry_calibration import verify_frozen

    root = Path(config["root"])
    ensure_limits(config, len(subjects))
    status_path = root / "calibration/TEACHER_STATUS.json"
    domains, statuses = {}, read(status_path) if status_path.exists() else {}
    for subject in subjects:
        target = subject["teacher"]["target"]
        binding = compatibility(config, subject["teacher"], subject["condition"])
        key = value_sha(
            [target, subject["condition"]["condition_sha256"], value_sha(binding)]
        )
        folder = root / "calibration/frozen" / key
        if (folder / "COMPLETE.json").exists():
            domains[target] = verify_frozen(root, read(folder / "domain.json"), binding)
            statuses.setdefault(
                target,
                dict(status="ready", domain_sha256=domains[target]["domain_sha256"]),
            )
            continue
        if (folder / "INSUFFICIENT_EVIDENCE.json").exists():
            statuses[target] = dict(status="needs_evidence")
            continue
        frozen(
            folder / "PLAN.json",
            dict(plan=plan(config, target), sampling_protocol=binding),
        )
        claim = folder / "SEARCH_CLAIM.json"
        if claim.exists():
            raise RuntimeError(
                "Interrupted linear calibration; preserve claim and audit before continuation"
            )
        frozen(claim, dict(plan_sha256=sha(folder / "PLAN.json")))
        write(root / "BUDGET_SCOPE.json", dict(stage="calibration", teacher_id=target))
        model = model_loader(config, subject["teacher"])
        search = LinearSearch(config, subject, binding, folder, model, executor)
        try:
            domain = search.run()
            receipts = {
                r["id"]: sha(root / r["id"] / "COMPLETE.json")
                for r in search.rows.values()
                if r["complete"]
            }
            frozen(folder / "domain.json", domain)
            frozen(folder / "all_results.json", list(search.rows.values()))
            frozen(
                folder / "COMPLETE.json",
                dict(
                    domain_sha256=domain["domain_sha256"],
                    measurement_receipts=receipts,
                    fitted_once=True,
                    promoted_slots_are_reused_not_independent=True,
                ),
            )
            domains[target] = domain
            statuses[target] = dict(
                status="ready",
                boundary_status=domain["boundary_status"],
                consumed_attempts=used(root, target),
            )
        except SearchInsufficient as error:
            frozen(
                folder / "INSUFFICIENT_EVIDENCE.json",
                dict(error=str(error), measurements=list(search.rows.values())),
            )
            statuses[target] = dict(status="needs_evidence", reason=str(error))
        finally:
            del search, model
            torch.cuda.empty_cache()
        write(root / "calibration/TEACHER_STATUS.json", statuses)
    frozen(root / "calibration/domains.json", domains)
    frozen(root / "calibration/TEACHER_STATUS.json", statuses)
    marker = "COMPLETE.json" if len(domains) == len(subjects) else "PARTIAL.json"
    completion = root / "calibration" / marker
    if completion.exists():
        if read(completion)["domain_sha256"] != value_sha(domains):
            raise ValueError(
                "Calibration completion belongs to another frozen reference set"
            )
    else:
        frozen(
            completion,
            dict(
                domain_sha256=value_sha(domains),
                ready_teachers=list(domains),
                statuses=statuses,
            ),
        )
    return domains
