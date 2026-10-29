"""One immutable multiplier per teacher, condition and native sampling protocol."""
import math
from pathlib import Path
from .io import read, frozen, write, value_sha, sha
from .geometry_condition import DISTANCE
from .geometry_protocol import EFFECTIVE, choose, displacement, derive_domain, validate_domain, GROWTH_LIMIT
from .geometry_sampling import compatibility, validate_receipt


DEFAULT_PLAN=dict(policy="successive-population-screening-v1",coarse_log2_exponents=list(range(-12,13,2)),
    coarse_population=10,refinement_population=20,confirmation_population=50,
    refinement_offsets=[-2,-1,0,1,2],maximum_attempts_per_teacher=350)


def plan(config,target):
    # This is a declared search grid, never a reused old optimum or a fitted multiplier.
    settings=config.get("calibration",{})
    centers=settings.get("search_center_by_teacher",{})
    center=centers.get(target,1.)
    if type(center) not in (int,float) or not math.isfinite(center) or center<=0:
        raise ValueError("Positive finite teacher-local search center required")
    if set(settings)-{"search_center_by_teacher"}:raise ValueError("Unknown calibration settings")
    scales=[center*2.**i for i in DEFAULT_PLAN["coarse_log2_exponents"]]
    if not all(math.isfinite(s) and s>0 for s in scales):raise ValueError("Invalid calibration grid")
    return dict(DEFAULT_PLAN,search_center_actual_scale=center,coarse_actual_scales=scales,
        center_is_fitted_multiplier=False,local_search_not_global_optimum=True)


def refinement(coarse):
    from .coordinate_statistics import ranking_eligible
    ordered=sorted(coarse,key=lambda r:r[EFFECTIVE][0]);eligible=[];previous=None;blocked=False
    for row in ordered:
        if not row["complete"] or not ranking_eligible(row):blocked=True;continue
        absolute=displacement(row)
        if previous is not None:
            ratio=row[EFFECTIVE][0]/previous[EFFECTIVE][0]
            prior=displacement(previous)
            if max(absolute[k]/max(prior[k]*ratio,1e-12) for k in absolute)>GROWTH_LIMIT:blocked=True
        if not blocked:eligible.append(row)
        previous=row
    if len(eligible)<3:raise ValueError("Insufficient finite stable screening evidence")
    best=choose(eligible)
    if best in (eligible[0],eligible[-1]):raise ValueError("Screening optimum at boundary; no automatic expansion")
    return [best[EFFECTIVE][0]*2.**offset for offset in DEFAULT_PLAN["refinement_offsets"]]


def verify_frozen(root,domain,binding):
    validate_domain(domain,binding["teacher_id"],binding["condition_sha256"],value_sha(binding))
    receipt=read(root/"calibration/frozen"/domain["calibration_key"]/"COMPLETE.json")
    if receipt["domain_sha256"]!=domain["domain_sha256"]:raise ValueError("Frozen multiplier receipt differs")
    for path,digest in receipt["measurement_receipts"].items():
        if sha(root/path/"COMPLETE.json")!=digest:raise ValueError("Frozen calibration evidence changed")
        validate_receipt(root/path)
    return domain


def run(config,subjects,model_loader,executor):
    root=Path(config["root"]);folder=root/"calibration";domains={}
    for subject in subjects:
        teacher,condition=subject["teacher"],subject["condition"];target=teacher["target"]
        binding=compatibility(config,teacher,condition);protocol_hash=value_sha(binding)
        key=value_sha([target,condition["condition_sha256"],protocol_hash]);out=folder/"frozen"/key
        declared=plan(config,target)
        if (out/"COMPLETE.json").exists():
            domains[target]=verify_frozen(root,read(out/"domain.json"),binding)
            continue  # No model load, API call or sampling when this key is already calibrated.
        frozen(out/"PLAN.json",dict(plan=declared,sampling_protocol=binding,calibration_key=key))
        if (out/"INSUFFICIENT_EVIDENCE.json").exists():
            raise RuntimeError("Frozen calibration stopped for insufficient evidence; explicit redesign required")
        model=model_loader(config,teacher);rows=[]
        def measure(scale,count,phase):
            destination=out/phase/value_sha(scale)
            failure=destination/"CALIBRATION_FAILURE.json"
            if failure.exists():row=read(failure)
            else:
                try:row=executor(model,config,teacher,condition,[scale]*4,destination,"calibration",count)
                except FloatingPointError as error:
                    row=dict(id=destination.relative_to(root).as_posix(),teacher_id=target,
                        condition_sha256=condition["condition_sha256"],compatibility_sha256=protocol_hash,
                        complete=False,attempts=count,population_is_requested_not_completed=True,error=str(error),**{EFFECTIVE:[scale]*4})
                    frozen(failure,row)
            rows.append(dict(row,phase=phase))
            return rows[-1]
        def phase_measure(scales,count,phase):
            from .geometry_sampling import execute
            if executor is not execute:return [measure(scale,count,phase) for scale in scales]
            from .geometry_batch import execute_many
            jobs=[dict(schedule=[scale]*4,folder=out/phase/value_sha(scale),stage="calibration",count=count) for scale in scales]
            measured=execute_many(model,config,teacher,condition,jobs)
            result=[]
            for scale,job,row in zip(scales,jobs,measured):
                if isinstance(row,FloatingPointError):
                    row=dict(id=job["folder"].relative_to(root).as_posix(),teacher_id=target,
                        condition_sha256=condition["condition_sha256"],compatibility_sha256=protocol_hash,
                        complete=False,attempts=count,population_is_requested_not_completed=True,error=str(row),**{EFFECTIVE:[scale]*4})
                    frozen(job["folder"]/"CALIBRATION_FAILURE.json",row)
                rows.append(dict(row,phase=phase));result.append(rows[-1])
            return result
        try:
            coarse=phase_measure(declared["coarse_actual_scales"],10,"coarse")
            frozen(out/"coarse_results.json",coarse)
            scales=refinement(coarse);frozen(out/"refinement_scales.json",scales)
            fine=phase_measure(scales,20,"refinement")
            frozen(out/"refinement_results.json",fine)
            phase_measure([r[EFFECTIVE][0] for r in fine if r["complete"]],50,"full")
            for row in fine:
                if not row["complete"]:rows.append(dict(row,phase="full",attempts=50,not_promoted_due_to_failure=True))
            frozen(out/"all_results.json",rows)
            domain=derive_domain(rows,target,condition["condition_sha256"],protocol_hash)
            frozen(out/"domain.json",domain)
            receipts={r["id"]:sha(root/r["id"]/"COMPLETE.json") for r in rows if r["complete"]}
            frozen(out/"COMPLETE.json",dict(domain_sha256=domain["domain_sha256"],measurement_receipts=receipts,
                fitted_once=True,search_is_local=True,promoted_slots_are_reused_not_independent=True))
            domains[target]=domain
        except ValueError as error:
            write(out/"INSUFFICIENT_EVIDENCE.json",dict(error=str(error),completed_and_failed_evidence=rows,
                no_automatic_expansion=True,no_weaker_retry=True))
            raise
        finally:
            model = None
            import torch
            torch.cuda.empty_cache()
    frozen(folder/"domains.json",domains)
    frozen(folder/"COMPLETE.json",dict(domain_sha256=value_sha(domains),calibration_keys={t:d["calibration_key"] for t,d in domains.items()}))
    return domains
