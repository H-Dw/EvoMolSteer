"""Variable allocation rounds, exact frontier evidence, and full-history retention."""

import shutil
from pathlib import Path

from . import linear_scale as scale
from .coordinate_statistics import (
    ranking_eligible,
    summarize,
    policy as statistics_policy,
)
from .geometry_protocol import choose, source_binding, primary_score
from .geometry_sampling import compatibility, validate_receipt
from .linear_calibration import evaluate, used
from .io import read, write, frozen, sha, value_sha


def history(config, teacher, condition, domain):
    root = Path(config["root"])
    expected = value_sha(compatibility(config, teacher, condition))
    scale.validate_domain(
        domain, teacher["target"], condition["condition_sha256"], expected
    )
    records, failures = {}, []
    for path in sorted((root / "measurement_registry").glob("*.json")):
        claim = read(path)
        identity = claim["identity"]
        if identity["compatibility"]["teacher_id"] != teacher["target"]:
            continue
        if value_sha(identity["compatibility"]) != expected:
            raise ValueError(
                "Incompatible historical measurement; do not mix scale/sampling protocols"
            )
        folder = root / claim["source_group"]
        if not (folder / "COMPLETE.json").exists():
            error = (
                read(folder / "FAILED.json")
                if (folder / "FAILED.json").exists()
                else dict(error="interrupted")
            )
            failures.append(
                dict(
                    id=claim["source_group"],
                    failure=error,
                    **{scale.EFFECTIVE: identity[scale.EFFECTIVE]},
                )
            )
            continue
        row = validate_receipt(folder)
        reason = None
        if not ranking_eligible(row):
            reason = "statistics_ineligible"
        elif row.get("stability_status") == "large_finite_displacement_growth":
            reason = "measured_displacement_growth"
        elif row["attempts"] != 50:
            reason = "smaller_population_not_rankable"
        if reason:
            from .absolute_coordinate import METRICS
            from .coordinate_statistics import RANKING, SUMMARY

            failures.append(
                dict(
                    id=row["id"],
                    evidence_status=reason,
                    attempts=row["attempts"],
                    **{scale.EFFECTIVE: row[scale.EFFECTIVE]},
                    **{
                        k: row[k]
                        for k in (
                            *METRICS,
                            RANKING,
                            SUMMARY,
                            "window_diagnostics",
                            "stability_status",
                        )
                        if k in row
                    },
                )
            )
            continue
        try:
            ticks = scale.encode(row[scale.EFFECTIVE], domain)
        except ValueError:
            failures.append(
                dict(
                    id=row["id"],
                    evidence_status="outside_declared_linear_grid",
                    **{scale.EFFECTIVE: row[scale.EFFECTIVE]},
                )
            )
            continue
        row = dict(row, **scale.binding(ticks, domain))
        if (folder / "proposal.json").exists():
            row.update(
                proposal=read(folder / "proposal.json"),
                proposal_sha256=sha(folder / "proposal.json"),
            )
        records[row["fingerprint"]] = row
    if domain["initial_source"] not in {row["id"] for row in records.values()}:
        raise ValueError("The measured calibration reference is missing")
    return sorted(records.values(), key=lambda r: r["id"]), failures


def range_allowance(config, target):
    settings = scale.teacher_settings(config["guidance_control"], target)
    root = Path(config["root"])
    claims = list((root / "range_evidence" / target).glob("*/CLAIM.json"))
    charged = 0
    for claim in claims:
        receipt = claim.with_name("COMPLETE.json")
        if not receipt.exists():
            raise RuntimeError(
                "Interrupted range evidence claim; audit instead of retrying"
            )
        charged += read(receipt)["charged_attempts"]
    return dict(
        remaining_attempts=max(
            0,
            min(
                settings["range_evidence_attempts"] - charged,
                settings["maximum_attempts_per_teacher"] - used(root, target),
            ),
        ),
        remaining_configurations=max(
            0, settings["maximum_range_evidence_configurations"] - len(claims)
        ),
    )


def formal_allowance(config):
    root = Path(config["root"])
    limits = read(root / "ATTEMPT_LIMITS.json")
    rows = read(root / "attempts.json") if (root / "attempts.json").exists() else {}
    return max(
        0,
        limits["stage_caps"]["formal"]
        - sum(r["count"] for r in rows.values() if r["stage"] == "formal"),
    )


def promotion_evidence(config, reference_folder, candidate_folder):
    """Compare identical raw identities at the smaller population, with fixed mask intersection."""
    from .absolute_coordinate import RMSD

    reference = {
        (r["logical_seed"], r["logical_slot"]): r
        for r in read(Path(reference_folder) / "records.json")
    }
    candidates = read(Path(candidate_folder) / "records.json")
    keys = [(r["logical_seed"], r["logical_slot"]) for r in candidates]
    if len(set(keys)) != len(keys) or any(key not in reference for key in keys):
        raise ValueError(
            "Range evidence is not the incumbent's nested seed/slot population"
        )
    left = [reference[key][RMSD] for key in keys]
    right = [r[RMSD] for r in candidates]
    settings = statistics_policy(config)
    a, b = summarize(left, settings), summarize(right, settings)
    excluded = set(a["excluded_high_indices"]) | set(b["excluded_high_indices"])
    paired = [
        candidate - reference
        for index, (reference, candidate) in enumerate(zip(left, right))
        if index not in excluded
    ]
    delta = sum(paired) / len(paired) if paired else None
    tolerance = config["guidance_control"]["improvement_tolerance_angstrom"]
    return dict(
        population=len(keys),
        retained_intersection=len(paired),
        paired_mean_change_angstrom=delta,
        reference_statistics=a,
        candidate_statistics=b,
        promote=bool(
            a["ranking_eligible"]
            and b["ranking_eligible"]
            and delta is not None
            and delta <= tolerance
        ),
        reference_identity_sha256=value_sha(keys),
    )


def remaining_profile_cost(root, target, actual):
    """Promotions reuse complete raw measurements; failed claims never count as reusable slots."""
    populations = [0]
    for path in (Path(root) / "measurement_registry").glob("*.json"):
        claim = read(path)
        identity = claim["identity"]
        if (
            identity["compatibility"]["teacher_id"] != target
            or identity[scale.EFFECTIVE] != actual
        ):
            continue
        folder = Path(root) / claim["source_group"]
        if not (folder / "COMPLETE.json").exists():
            raise RuntimeError("Existing failed/interrupted profile cannot be promoted")
        populations.append(validate_receipt(folder)["attempts"])
    return max(0, 50 - max(populations))


def range_evidence(
    config, subject, domain, resolved, incumbent, model, executor, folder
):
    root, folder = Path(config["root"]), Path(folder)
    target = subject["teacher"]["target"]
    receipt = folder / "COMPLETE.json"
    if receipt.exists():
        saved = read(receipt)
        if (
            saved["action_sha256"] != value_sha(resolved)
            or saved["input_domain_sha256"] != domain["domain_sha256"]
        ):
            raise ValueError("Saved range evidence belongs to another action/domain")
        for relative, digest in saved["source_receipts"].items():
            if sha(root / relative / "COMPLETE.json") != digest:
                raise ValueError("Range evidence source receipt changed")
            validate_receipt(root / relative)
        for relative, digest in saved["failure_files"].items():
            if sha(root / relative) != digest:
                raise ValueError("Range failure evidence changed")
        return saved
    if (folder / "CLAIM.json").exists():
        raise RuntimeError("Existing interrupted range claim; no automatic retry")
    allowance = range_allowance(config, target)
    required = remaining_profile_cost(root, target, resolved[scale.EFFECTIVE])
    if (
        allowance["remaining_attempts"] < required
        or allowance["remaining_configurations"] < 1
    ):
        return dict(
            status="budget_limited", row=None, charged_attempts=0, source_receipts={}
        )
    write(root / "BUDGET_SCOPE.json", dict(stage="calibration", teacher_id=target))
    before = used(root, target)
    frozen(
        folder / "CLAIM.json",
        dict(
            action_sha256=value_sha(resolved),
            domain_sha256=domain["domain_sha256"],
            maximum_new_attempts=required,
            incumbent_id=incumbent["id"],
        ),
    )
    sources, failure_files, measured, promotion = {}, {}, None, None
    status = "measured_not_promoted"
    for count in (20, 50):
        out = folder / f"population_{count}"
        frozen(out / "proposal.json", resolved)
        job = dict(
            schedule=resolved[scale.TICKS],
            folder=out,
            stage="calibration",
            count=count,
            domain=domain,
        )
        result = evaluate(model, config, subject, job, executor)
        if isinstance(result, FloatingPointError):
            status = "nonfinite_measurement"
            frozen(
                out / "CALIBRATION_FAILURE.json",
                dict(error=str(result), **{scale.EFFECTIVE: resolved[scale.EFFECTIVE]}),
            )
            for name in ("CALIBRATION_FAILURE.json", "FAILED.json"):
                path = out / name
                if path.exists():
                    failure_files[path.relative_to(root).as_posix()] = sha(path)
            break
        sources[result["id"]] = sha(root / result["id"] / "COMPLETE.json")
        if (
            not ranking_eligible(result)
            or result.get("stability_status") == "large_finite_displacement_growth"
        ):
            status = "measured_ineligible"
            break
        if count == 20:
            promotion = promotion_evidence(
                config, root / incumbent["id"], root / result["id"]
            )
            frozen(folder / "PROMOTION.json", promotion)
            if not promotion["promote"]:
                break
        else:
            measured, status = result, "full_population_complete"
    charged = used(root, target) - before
    if charged > required:
        raise ValueError("Range evidence exceeded its reserved incremental cost")
    result = dict(
        status=status,
        row=measured,
        charged_attempts=charged,
        source_receipts=sources,
        failure_files=failure_files,
        action_sha256=value_sha(resolved),
        input_domain_sha256=domain["domain_sha256"],
        promotion=promotion,
    )
    frozen(receipt, result)
    return result


def run_version(config, version, model_loader, executor, advisor, retainer):
    import torch
    from . import absolute_runtime as runtime
    from .geometry_calibration import verify_frozen

    root = Path(config["root"])
    if not (
        version.startswith("GD")
        and version[2:].isdigit()
        and 1 <= int(version[2:]) <= 20
        and version == f"GD{int(version[2:]):02d}"
    ):
        raise ValueError("Versions must be GD01--GD20")
    folder = root / "versions" / version
    if (folder / "AWAITING_GPT_REVIEW.json").exists():
        return read(folder / "AWAITING_GPT_REVIEW.json")
    if int(version[2:]) > 1:
        approved = read(
            root / "versions" / f"GD{int(version[2:])-1:02d}" / "REVIEW_ACCEPTED.json"
        )
        if approved["next_version"] != version or approved[
            "skill_hashes"
        ] != runtime.skill_hashes(config):
            raise ValueError("Previous main GPT review has not approved this version")
    from .tool_runtime import bind_experiment

    bind_experiment(config)
    skills = folder / "skills"
    skills.mkdir(parents=True, exist_ok=True)
    for name, source in runtime.skill_assets(config).items():
        dest = skills / name
        if dest.exists() and sha(dest) != sha(source):
            raise ValueError("Runtime skills changed inside an experiment")
        if not dest.exists():
            shutil.copyfile(source, dest)
    manifest = dict(
        scale_policy=scale.POLICY,
        config_sha256=value_sha(config),
        source_binding=source_binding(),
        skill_hashes={p.name: sha(p) for p in skills.glob("*.md")},
        rounds=config["rounds"],
    )
    frozen(folder / "manifest.json", manifest)
    domains = read(root / "calibration/domains.json")
    subjects = runtime.calibrated(config)
    from .linear_calibration import ensure_limits

    ensure_limits(config, len(subjects))
    missing = [
        s["teacher"]["target"]
        for s in subjects
        if s["teacher"]["target"] not in domains
    ]
    subjects = [s for s in subjects if s["teacher"]["target"] in domains]
    if not subjects:
        raise ValueError("No teacher has a calibrated reference")
    summaries, all_groups, contexts, rounds = {}, [], [], []
    for subject in subjects:
        teacher, condition = subject["teacher"], subject["condition"]
        target = teacher["target"]
        domain = verify_frozen(
            root, domains[target], compatibility(config, teacher, condition)
        )
        if domain["control_settings"] != scale.teacher_settings(
            config["guidance_control"], target
        ):
            raise ValueError(
                "Linear search rules differ from the frozen domain; an explicit new binding is required"
            )
        if domain.get("exploration_policy") != config.get("guidance_exploration"):
            raise ValueError(
                "Jump exploration hypotheses differ from the frozen experiment contract"
            )
        if int(version[2:]) > 1:
            previous = read(
                root / "versions" / f"GD{int(version[2:])-1:02d}" / "FINALIZED.json"
            )
            previous_summary = previous["teachers"].get(target, {})
            if previous_summary.get("domain_path"):
                inherited = read(root / previous_summary["domain_path"])
                scale.validate_domain(
                    inherited,
                    target,
                    condition["condition_sha256"],
                    domain["sampling_protocol_sha256"],
                )
                if inherited["domain_sha256"] != previous_summary[
                    "domain_sha256"
                ] or any(
                    inherited[k] != domain[k]
                    for k in (scale.REFERENCE, "calibration_key", "control_settings")
                ):
                    raise ValueError(
                        "Inherited evidence changed the frozen reference or search contract"
                    )
                domain = inherited
        out = folder / target
        frozen(out / "domain_revisions/000.json", domain)
        groups = []
        model = model_loader(config, teacher)
        try:
            if (root / "INITIAL_GUIDANCE.json").exists() and version == "GD01":
                from .geometry_reuse import verify_initial

                selection = verify_initial(root, domains)[target]
                result = evaluate(
                    model,
                    config,
                    subject,
                    dict(
                        schedule=selection[scale.TICKS],
                        folder=root / "initial_guidance" / target,
                        stage="formal",
                        domain=domain,
                    ),
                    executor,
                )
                if isinstance(result, Exception):
                    raise result
                frozen(
                    out / "INITIAL_MEASUREMENT.json",
                    dict(
                        selected=selection,
                        source_group=result["id"],
                        fingerprint=result["fingerprint"],
                    ),
                )
            for iteration in range(1, config["rounds"] + 1):
                call_folder = out / f"advice_{iteration:03d}"
                completed = call_folder / "ROUND_COMPLETE.json"
                if completed.exists():
                    saved = read(completed)
                    if saved["input_domain_sha256"] != domain["domain_sha256"]:
                        raise ValueError("Round replay domain chain changed")
                    for relative, digest in saved["source_receipts"].items():
                        if sha(root / relative / "COMPLETE.json") != digest:
                            raise ValueError("Round source completion changed")
                        validate_receipt(root / relative)
                    for relative, digest in saved["advice_files"].items():
                        if sha(root / relative) != digest:
                            raise ValueError("Completed round advice changed")
                    for event in saved["range_events"]:
                        for relative, digest in event.get("failure_files", {}).items():
                            if sha(root / relative) != digest:
                                raise ValueError(
                                    "Completed round failure evidence changed"
                                )
                    domain = read(root / saved["output_domain_path"])
                    scale.validate_domain(domain, target, condition["condition_sha256"])
                    if domain["domain_sha256"] != saved["output_domain_sha256"]:
                        raise ValueError("Round output domain changed")
                    groups.extend(saved["groups"])
                    rounds.append(saved)
                    continue
                current, failed = history(config, teacher, condition, domain)
                from .checkpoint_learning import (
                    context as checkpoint_context,
                    checkpoint,
                )

                learning = checkpoint_context(
                    config, current, [r for r in rounds if r["teacher_id"] == target]
                )
                snapshot = dict(
                    history=current,
                    failures=failed,
                    domain=domain,
                    checkpoint_learning=learning,
                )
                frozen(call_folder / "history_snapshot.json", snapshot)
                advisor_config = (
                    dict(config, guidance_checkpoint_context=learning)
                    if learning
                    else config
                )
                decision = advisor(
                    advisor_config, current, domain, call_folder, skills, failed
                )
                if decision["decision"] not in (
                    "hold",
                    "propose",
                    "request_domain_evidence",
                ):
                    raise ValueError("Unknown allocation decision")
                actions = decision["actions"]
                if len(actions) > domain["control_settings"]["maximum_selected"] or (
                    decision["decision"] == "hold"
                ) != (not actions):
                    raise ValueError("Invalid variable allocation count")
                input_domain = domain["domain_sha256"]
                action_domain = domain
                seen, measured, receipts, events = list(current), [], {}, []
                incumbent = choose(current)
                for index, action in enumerate(actions, 1):
                    resolved = scale.resolve_action(
                        action,
                        {r["id"]: r for r in current},
                        action_domain,
                        seen,
                        failed,
                    )
                    seen.append(resolved)
                    action_folder = (
                        out / f"iteration_{iteration:03d}" / f"action_{index:03d}"
                    )
                    frozen(action_folder / "agent_action.json", action)
                    frozen(action_folder / "proposal.json", resolved)
                    if resolved["task_type"] == "range_evidence":
                        evidence_folder = (
                            root
                            / "range_evidence"
                            / target
                            / value_sha([version, iteration, action])
                        )
                        event = range_evidence(
                            config,
                            subject,
                            action_domain,
                            resolved,
                            incumbent,
                            model,
                            executor,
                            evidence_folder,
                        )
                        events.append(dict(action_id=action["id"], **event))
                        receipts.update(event["source_receipts"])
                        row = event["row"]
                        if row is not None:
                            domain = scale.revise(domain, row)
                    else:
                        required = remaining_profile_cost(
                            root, target, resolved[scale.EFFECTIVE]
                        )
                        if formal_allowance(config) < required:
                            events.append(
                                dict(
                                    action_id=action["id"],
                                    status="budget_limited",
                                    stage="formal",
                                    charged_attempts=0,
                                )
                            )
                            continue
                        write(
                            root / "BUDGET_SCOPE.json",
                            dict(stage="formal", teacher_id=target),
                        )
                        row = evaluate(
                            model,
                            config,
                            subject,
                            dict(
                                schedule=resolved[scale.TICKS],
                                folder=action_folder,
                                stage="formal",
                                domain=domain,
                            ),
                            executor,
                        )
                        if isinstance(row, Exception):
                            raise row
                        receipts[row["id"]] = sha(root / row["id"] / "COMPLETE.json")
                    if row is not None:
                        measured.append(
                            dict(
                                row,
                                selected_proposal=resolved,
                                iteration=iteration,
                                task_type=resolved["task_type"],
                            )
                        )
                revision = (
                    out / "domain_revisions" / f"{domain['evidence_revision']:03d}.json"
                )
                frozen(revision, domain)
                saved = dict(
                    teacher_id=target,
                    iteration=iteration,
                    decision=decision["decision"],
                    requested_actions=len(actions),
                    groups=measured,
                    range_events=events,
                    source_receipts=receipts,
                    input_domain_sha256=input_domain,
                    output_domain_path=revision.relative_to(root).as_posix(),
                    output_domain_sha256=domain["domain_sha256"],
                    advice_files={
                        p.relative_to(root).as_posix(): sha(p)
                        for p in call_folder.glob("*.json")
                    },
                )
                if learning:
                    saved["checkpoint"] = checkpoint(measured, incumbent)
                    saved["checkpoint_learning"] = learning
                frozen(completed, saved)
                groups.extend(measured)
                rounds.append(saved)
            current, failed = history(config, teacher, condition, domain)
            advisor(
                config,
                current,
                domain,
                out / "analysis_final",
                skills,
                failed,
                final=True,
            )
            winner = choose(current)
            retainer(root, current, winner, out / "retention")
            formal = [
                r
                for r in groups
                if r["task_type"] == "optimization"
                and ranking_eligible(r)
                and r.get("stability_status") != "large_finite_displacement_growth"
            ]
            formal_best = choose(formal)["id"] if formal else None
            from .absolute_coordinate import METRICS
            from .coordinate_statistics import RANKING

            summaries[target] = dict(
                representative=winner["id"],
                best_formal_group=formal_best,
                **{k: winner[k] for k in (*METRICS, RANKING) if k in winner},
                complete=True,
                domain_sha256=domain["domain_sha256"],
                calibration_reference=domain[scale.REFERENCE],
                domain_path=(
                    out / "domain_revisions" / f"{domain['evidence_revision']:03d}.json"
                )
                .relative_to(root)
                .as_posix(),
            )
            contexts.extend(
                [
                    str(out / "analysis_final/proposal.json"),
                    str(out / "analysis_final/synthesis.json"),
                ]
            )
            all_groups.extend(groups)
        finally:
            del model
            torch.cuda.empty_cache()
    result = dict(
        version=version,
        teachers=summaries,
        groups=all_groups,
        rounds=rounds,
        strategy_contexts=contexts,
        manifest_sha256=sha(folder / "manifest.json"),
        complete=not missing,
        missing_teachers=missing,
        all_available_teachers_processed=True,
        configured_rounds=config["rounds"],
        completed_rounds=len(rounds),
        requested_groups=sum(r["requested_actions"] for r in rounds),
        measured_or_reused_groups=len(all_groups),
        held_rounds=sum(r["decision"] == "hold" for r in rounds),
        quality_source_groups=sorted(
            {r["id"] for r in all_groups}
            | {r["representative"] for r in summaries.values()}
        ),
    )
    frozen(folder / "FINALIZED.json", result)
    previous = [
        read(p)
        for p in sorted((root / "versions").glob("GD*/FINALIZED.json"))
        if p.parent.name < version
    ]
    best_versions = {
        target: min(
            [p for p in previous + [result] if target in p["teachers"]],
            key=lambda v: primary_score(v["teachers"][target]),
        )["version"]
        for target in summaries
    }
    request = dict(
        version=version,
        current=result,
        previous=previous[-1] if previous else None,
        best_version_by_teacher=best_versions,
        measured_initials={t: d["initial_source"] for t, d in domains.items()},
        next_version=None if version == "GD20" else f"GD{int(version[2:])+1:02d}",
        strategy_contexts=contexts,
        requires_actual_main_agent_review=True,
    )
    frozen(folder / "AWAITING_GPT_REVIEW.json", request)
    return request
