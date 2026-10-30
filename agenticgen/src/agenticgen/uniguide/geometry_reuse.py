"""Explicit, hash-verified calibration reuse and user-selected initial windows."""
import shutil
from pathlib import Path
from .io import read, frozen, sha, value_sha
from .geometry_sampling import compatibility, validate_receipt
from .geometry_calibration import verify_frozen
from .geometry_protocol import NORMALIZED, EFFECTIVE, MULTIPLIER, ticks, effective


def safe_source(root,relative):
    path=(root/relative).resolve()
    if path==root.resolve() or not path.is_relative_to(root.resolve()):raise ValueError("Evidence path escapes source root")
    return path


def tree_hashes(path):
    return {p.relative_to(path).as_posix():sha(p) for p in sorted(path.rglob("*")) if p.is_file()}


def select_initial(config,schedules,scale_units="ticks"):
    root=Path(config["root"]);domains=read(root/"calibration/domains.json")
    if set(schedules)!=set(domains):raise ValueError("Provide four normalized window values for every teacher")
    selected={}
    for target,domain in domains.items():
        selected[target]=initial_binding(schedules[target],domain,scale_units)
    from .linear_scale import enabled as linear_enabled
    payload=dict(teachers=selected,scale_units="linear_reference_ticks" if all(linear_enabled(d) for d in domains.values()) else "normalized_integer_log2",
        calibration_sha256=sha(root/"calibration/domains.json"))
    if (root/"versions/GD01/manifest.json").exists() and not (root/"INITIAL_GUIDANCE.json").exists():
        raise ValueError("Cannot change initialization after Agent inference has started")
    frozen(root/"INITIAL_GUIDANCE.json",payload)
    return payload


def initial_binding(values,domain,scale_units="ticks"):
    from . import linear_scale
    if scale_units not in ("ticks","actual"):
        raise ValueError("Initial scale units must be ticks or actual")
    if linear_scale.enabled(domain):
        linear_scale.validate_domain(domain)
        if scale_units=="actual":values=linear_scale.encode(values,domain)
        result=linear_scale.binding(values,domain)
        if any(x not in domain["confirmed_effective_levels"] for x in result[EFFECTIVE]):
            raise ValueError("Initial strengths require complete constant confirmation")
    else:
        if scale_units!="ticks":raise ValueError("Explicit actual initialization requires the linear protocol")
        values=list(ticks(values))
        if any(not domain["normalized_minimum"]<=v<=domain["normalized_maximum"] for v in values):
            raise ValueError("Initial scale outside measured domain")
        result={NORMALIZED:values,MULTIPLIER:domain[MULTIPLIER],EFFECTIVE:effective(values,domain[MULTIPLIER]),"guidance_enabled":True}
    return dict(result,domain_sha256=domain["domain_sha256"],condition_sha256=domain["condition_sha256"])


def verify_initial(root,domains):
    payload=read(root/"INITIAL_GUIDANCE.json")
    if payload["calibration_sha256"]!=sha(root/"calibration/domains.json") or set(payload["teachers"])!=set(domains):
        raise ValueError("Initial window selection belongs to another calibration")
    for target,selected in payload["teachers"].items():
        from .linear_scale import enabled as linear_enabled, TICKS
        if linear_enabled(domains[target]):
            if initial_binding(selected[TICKS],domains[target])!=selected:
                raise ValueError("Linear initialization binding differs")
            continue
        domain=domains[target];values=ticks(selected[NORMALIZED])
        if selected["domain_sha256"]!=domain["domain_sha256"] or selected[MULTIPLIER]!=domain[MULTIPLIER]:raise ValueError("Initial multiplier binding differs")
        if any(not domain["normalized_minimum"]<=v<=domain["normalized_maximum"] for v in values):raise ValueError("Initial scale outside domain")
        if selected[EFFECTIVE]!=effective(values,domain[MULTIPLIER]) or not selected["guidance_enabled"]:raise ValueError("Initial effective scale differs")
    return payload["teachers"]


def reuse(config,sources,schedules,scale_units="ticks"):
    from .absolute_coordinate import POLICY as absolute_policy
    from . import absolute_runtime, geometry_runtime
    runtime=absolute_runtime if config.get("protocol")==absolute_policy else geometry_runtime
    calibrated,require_acceptance=runtime.calibrated,runtime.require_acceptance
    require_acceptance(config)
    subjects=calibrated(config);root=Path(config["root"]).resolve();domains={};copies={};audit={}
    if set(sources)!={s["teacher"]["target"] for s in subjects}:raise ValueError("Provide one calibration source root per teacher")
    if (root/"versions").exists():raise ValueError("Import calibration before starting any Agent version")
    for subject in subjects:
        teacher,condition=subject["teacher"],subject["condition"];target=teacher["target"]
        source=Path(sources[target]).resolve()
        if source==root or source.is_relative_to(root) or root.is_relative_to(source):raise ValueError("Calibration source and destination must be independent")
        path=source/"calibration/domains.json"
        if not path.exists():raise ValueError("Calibration is incomplete; coarse screening does not establish a multiplier: "+target)
        domain=read(path)[target];binding=compatibility(config,teacher,condition)
        verify_frozen(source,domain,binding)  # Exact condition, mapping, model, protocol, RNG and implementation.
        from . import linear_scale
        if linear_scale.enabled(domain) and domain["control_settings"]!=linear_scale.teacher_settings(config["guidance_control"],target):
            raise ValueError("Reused linear search rules differ; an explicit domain revision is required")
        domains[target]=domain
        frozen_dir=Path("calibration/frozen")/domain["calibration_key"]
        for part in (source/frozen_dir).rglob("*"):
            if part.is_file():copies[part.relative_to(source).as_posix()]=part
        # Import complete and failed evidence, including all shard source products.
        for registry in ("measurement_registry","slot_registry"):
            for entry in (source/registry).glob("*.json"):
                claim=read(entry);identity=claim["identity"]
                if identity["compatibility"]["teacher_id"]!=target:continue
                if identity["compatibility"]!=binding:raise ValueError("Incompatible historical sampling protocol")
                group=safe_source(source,claim["source_group"])
                if (group/"COMPLETE.json").exists():validate_receipt(group)
                copies[entry.relative_to(source).as_posix()]=entry
                for part in group.rglob("*"):
                    if part.is_file():copies[part.relative_to(source).as_posix()]=part
        audit[target]=dict(source_root=str(source),domain_sha256=domain["domain_sha256"],
            source_attempts_sha256=sha(source/"attempts.json") if (source/"attempts.json").exists() else None)
    # Validate requested initialization before copying any evidence.
    if set(schedules)!=set(domains):raise ValueError("Initial selection must include each teacher")
    for target,values in schedules.items():
        initial_binding(values,domains[target],scale_units)
    hashes={relative:sha(source) for relative,source in copies.items()}
    for relative,digest in hashes.items():
        dest=safe_source(root,relative)
        if dest.exists() and sha(dest)!=digest:raise ValueError("Import would overwrite existing evidence: "+relative)
    frozen(root/"CALIBRATION_IMPORT.json",dict(sources=audit,files=hashes,new_sampling_attempts=0,
        imported_measurements_are_not_independent=True,original_attempt_ledgers_preserved=True))
    for relative,source in copies.items():
        dest=safe_source(root,relative);dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():shutil.copyfile(source,dest)
        if sha(dest)!=hashes[relative] or sha(source)!=hashes[relative]:raise ValueError("Evidence changed while importing")
    frozen(root/"calibration/domains.json",domains)
    frozen(root/"calibration/COMPLETE.json",dict(domain_sha256=value_sha(domains),
        calibration_keys={t:d["calibration_key"] for t,d in domains.items()}))
    for subject in subjects:
        target=subject["teacher"]["target"]
        verify_frozen(root,domains[target],compatibility(config,subject["teacher"],subject["condition"]))
    return select_initial(config,schedules,scale_units)
