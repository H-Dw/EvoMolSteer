"""Two complete backgrounds plus eight independent early-window neighbours."""
from pathlib import Path
from .io import read, write, frozen, sha, value_sha
from . import linear_scale as scale
from . import path_search
from .box_search import grids, early_shape, validate_profile
from .condition_evaluation import CONFIG, POLICY as EVALUATION
from .geometry_protocol import choose, primary_score

POLICY = 'two-background-eight-early-neighbours-v1'
BACKGROUNDS = {
    "6pyy_H7S_A_402": [[50,70,70,50,50,70], [50,100,40,40,30,50]],
    "3w54_RNB_A_504": [[30,40,40,40,60,80], [30,40,110,110,200,200]],
}


def plan(config, target):
    if target not in BACKGROUNDS:
        raise ValueError("No declared path-calibration backgrounds for teacher " + target)
    return dict(policy=POLICY, backgrounds=BACKGROUNDS[target],
                populations=[50], reference_budget=500, maximum_profiles=10, selection_agent=False)


def neighbours(actual, declared_grids):
    profiles = []
    for i in range(4):
        grid = declared_grids[i]; index = grid.index(actual[i])
        positions = sorted((j for j in range(len(grid)) if j != index), key=lambda j: (abs(j-index), j))[:2]
        for j in positions:
            p = list(actual); p[i] = grid[j]
            profiles.append((f"B{i+1}_{grid[j]}", p))
    return profiles


def derive_domain(config, subject, binding, rows):
    from .calibration_search import displacement
    target = subject["teacher"]["target"]
    winner = choose(rows)
    protocol = value_sha(binding); condition = subject["condition"]["condition_sha256"]
    body = dict(scale_policy=scale.OPEN_POLICY, path_search_policy=path_search.POLICY,
        condition_evaluation=dict(policy=EVALUATION), teacher_id=target, condition_sha256=condition,
        sampling_protocol_sha256=protocol, calibration_key=value_sha([target, condition, protocol]),
        control_settings=config["guidance_control"], calibrated_guidance_reference_scale=0,
        guidance_scale_grid_unit=1, initial_source=winner["id"],
        initial_guidance_scale_tick_schedule=winner[scale.EFFECTIVE], guidance_windows=config["windows"],
        bootstrap_sources=[r["id"] for r in rows],
        confirmed_effective_levels=sorted({v for r in rows for v in r[scale.EFFECTIVE]}),
        confirmed_profiles=[dict(source_id=r["id"], actual_schedule=r[scale.EFFECTIVE]) for r in rows],
        constant_comparison=[dict(source_id=r["id"], eligible=True, absolute_displacement=displacement(r)) for r in rows],
        risk_boundary=None, evidence_revision=0, boundary_status="profile_scoped_evidence",
        stability_guarantee=False, global_optimum_claimed=False, source_rows_sha256=value_sha(rows),
        exploration_policy=config["guidance_exploration"], initial_calibration_performance=primary_score(winner),
        guidance_box_mask=[True]*6, admission_policy="declared_grid_and_actual_budget; all_boxes_available",
        search_grids=grids(config,target), early_guidance_shape=early_shape(config),
        execute_all_proposed=True, distance_unit="per_window_grid_index")
    domain = scale.seal(body); scale.validate_domain(domain)
    return domain


def run(config, subjects, model_loader, executor):
    import torch
    from .geometry_sampling import compatibility
    from .geometry_calibration import verify_frozen
    from .linear_calibration import ensure_limits, evaluate, used
    from .coordinate_statistics import ranking_eligible
    root = Path(config["root"]); ensure_limits(config,len(subjects))
    domains, statuses = {}, {}
    for subject in subjects:
        target = subject["teacher"]["target"]
        declared = plan(config,target); binding = compatibility(config,subject["teacher"],subject["condition"])
        key = value_sha([target,subject["condition"]["condition_sha256"],value_sha(binding)])
        dest = root/"calibration/frozen"/key
        folder = root/"calibration/path"/target
        if (dest/"COMPLETE.json").exists():
            domains[target] = verify_frozen(root,read(dest/"domain.json"),binding)
            statuses[target] = dict(status="ready",domain_sha256=domains[target]["domain_sha256"])
            continue
        if (folder/"SEARCH_CLAIM.json").exists():
            raise RuntimeError("Path calibration already claimed; no automatic failed/interrupted restart")
        frozen(folder/"PLAN.json",declared)
        frozen(folder/"SEARCH_CLAIM.json",dict(config_sha256=value_sha(config),binding=binding))
        rows, failed, seen = [], [], set()
        model = model_loader(config,subject["teacher"])
        try:
            def measure(name, actual):
                validate_profile(actual,{"search_grids":grids(config,target)})
                if tuple(actual) in seen:
                    return
                seen.add(tuple(actual))
                if used(root,target)+50 > 500:
                    raise ValueError("Path calibration budget exhausted")
                write(root/"BUDGET_SCOPE.json",dict(stage="calibration",teacher_id=target))
                row = evaluate(model,config,subject,dict(schedule=actual,folder=folder/name,
                                                       stage="calibration",count=50),executor)
                if isinstance(row, Exception):
                    failure = dict(id=(folder/name).relative_to(root).as_posix(),error=str(row),
                                   attempts=50,**{scale.EFFECTIVE:actual})
                    failed.append(failure); frozen(folder/name/"CALIBRATION_FAILURE.json",failure)
                elif (row.get(CONFIG) == dict(policy=EVALUATION) and row.get("complete")
                      and ranking_eligible(row)):
                    primary_score(row)
                    rows.append(row)
                else:
                    failure = dict(id=row['id'], evidence_status='statistics_ineligible',
                                   attempts=50, **{scale.EFFECTIVE:actual},
                                   reason='Missing or invalid complete condition-aligned measurement')
                    failed.append(failure); frozen(folder/name/'CALIBRATION_FAILURE.json',failure)
                write(folder/"SEARCH_STATE.json",dict(measurements=rows, failures=failed, consumed_attempts=used(root,target)))
            for i, actual in enumerate(declared["backgrounds"]): measure(f"background_{i+1}",actual)
            if rows:
                fixed = choose(rows)
                frozen(folder/"NEIGHBOUR_REFERENCE.json",dict(source_id=fixed["id"],actual_schedule=fixed[scale.EFFECTIVE]))
                for name, actual in neighbours(fixed[scale.EFFECTIVE],grids(config,target)):
                    measure(name,actual)
        finally:
            del model; torch.cuda.empty_cache()
        if not rows:
            statuses[target] = dict(status="no_valid_root",failures=failed)
            frozen(folder/"INSUFFICIENT_EVIDENCE.json",statuses[target]); continue
        domain = derive_domain(config,subject,binding,rows)
        frozen(dest/"domain.json",domain)
        frozen(dest/"COMPLETE.json",dict(domain_sha256=domain["domain_sha256"],
            measurement_receipts={r["id"]:sha(root/r["id"]/"COMPLETE.json") for r in rows}))
        domains[target]=domain; statuses[target]=dict(status="ready",domain_sha256=domain["domain_sha256"])
    frozen(root/"calibration/domains.json",domains)
    frozen(root/"calibration/TEACHER_STATUS.json",statuses)
    frozen(root/"calibration/COMPLETE.json",dict(domain_sha256=value_sha(domains),selection_agent=False))
    return domains
