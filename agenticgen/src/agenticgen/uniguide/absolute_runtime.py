"""Isolated geometry protocol. Never migrate or resume the old UG campaign."""
import copy
import math
import shutil
import subprocess
from pathlib import Path
import torch
from .io import read,write,frozen,sha,value_sha
from .absolute_coordinate import prepare as prepare_condition, validate as validate_condition, RMSD, MAX_ERROR, POLICY, ASSIGNMENT
DISTANCE=CONTROL_DISTANCE=RMSD
NEIGHBOR_DISTANCE=MAX_ERROR
from .geometry_protocol import NORMALIZED,EFFECTIVE,MULTIPLIER,validate_domain,choose,ticks,resolve_action
from .geometry_sampling import execute,compatibility,validate_receipt
from .geometry_council import advise
from .prepare import load_model
from .geometry_protocol import source_binding
from .quality import built_molecules
from .protocol import SEEDS
from .geometry_dynamics import WINDOWS,policy as dynamics_policy
from .geometry_reward import WEIGHTS,INTERACTION_DISTANCE
from .geometry_protocol import seen_configuration,primary_score


def skill_assets(config=None):
    if config and config.get("agent_tools"):
        from .tool_runtime import assets
        return assets(config)
    resources=Path(__file__).resolve().parents[1]/"resources/skills"
    return {name+".md":resources/("distance-guidance-"+name)/"SKILL.md" for name in ("analysis","proposal","synthesis")}


def skill_hashes(config=None):return {name:sha(path) for name,path in skill_assets(config).items()}


def common_valid_slots(source,target):
    """One frozen mapping must be valid for every logical sample, not just row zero."""
    from .native_tools import load_pt
    common=None
    for index in range(5):
        mask=load_pt(Path(source)/"initials"/target/f"batch_{index:03d}.pt.gz")["prior"]["mask"]
        if mask.ndim!=2 or len(mask)<10:raise ValueError("Incomplete initial population masks")
        present=mask[:10].bool().all(dim=0)
        if common is not None and present.shape!=common.shape:raise ValueError("Initial slot shapes differ")
        common=present if common is None else common & present
    return common.nonzero().flatten().tolist()


def validate_config(config):
    if "guidance_reward" in config:
        raise ValueError("guidance_reward normalization settings were removed; use a new experiment and recalibrate")
    from .tool_runtime import configure
    config=configure(config)
    if config.get("protocol")!=POLICY:raise ValueError("Explicit new geometry protocol required")
    root=Path(config["root"]).resolve();source=Path(config["generation_context_root"]).resolve()
    if root==source or root.is_relative_to(source) or source.is_relative_to(root):
        raise ValueError("New experiment must be separate from the frozen source experiment")
    if (config.get("steps",100)!=100 or config.get("noise_level",.2)!=.2 or
            type(config["rounds"]) is not int or config["rounds"]<1):raise ValueError("Invalid integration/iteration configuration")
    if config.get("population",50)!=50:raise ValueError("Formal population remains fifty")
    from .task_scheduling import configure as schedule_tasks
    config=schedule_tasks(config)
    dynamics_policy(config)
    from .geometry_chemistry import policy as chemical_policy
    config=dict(config,chemical_inpainting={"enabled":chemical_policy(config)["enabled"]})
    from .coordinate_statistics import policy as statistics_policy, CONFIG
    config = dict(config, **{CONFIG: statistics_policy(config)})
    from .linear_scale import enabled as linear_enabled
    declared_release=config.get("agent_tools",{}).get("skill_release")
    required_release=((declared_release if declared_release in ("2.8.0","2.8.1","2.8.2") else "2.7.0") if config.get("guidance_exploration") else "2.6.0") if linear_enabled(config) else "2.5.0"
    if config.get("agent_tools",{}).get("skill_release")!=required_release:
        raise ValueError("Coordinate control and runtime skill release differ")
    from .absolute_coordinate import ASSIGNMENT, DYNAMIC_ASSIGNMENT
    expected_assignment=DYNAMIC_ASSIGNMENT if required_release in ("2.7.0","2.8.0","2.8.1","2.8.2") else ASSIGNMENT
    if config.get("atom_assignment_policy",expected_assignment)!=expected_assignment:
        raise ValueError("Coordinate assignment and skill contracts differ; use a new compatible release")
    from .geometry_calibration import plan
    for item in config.get("conditions",[]):plan(config,item["teacher_id"])
    from .guidance_windows import windows
    windows(config)
    extended = (config.get("prefix_inpainting",{}).get("enabled")
        or config.get("checkpoint_learning",{}).get("enabled")
        or windows(config) != [list(w) for w in WINDOWS])
    if extended and declared_release not in ("2.8.0","2.8.1","2.8.2"):
        raise ValueError("Optional prefix, variable windows and checkpoint learning require skills 2.8.x")
    if config.get("seeds",SEEDS)!=SEEDS:
        raise ValueError("Native sampling windows and random seeds are fixed")
    return dict(config,initials_root=str(source))


def prepare(config):
    config=validate_config(config);root=Path(config["root"])
    if root.exists() and any(root.iterdir()) and not (root/"GEOMETRY_CONFIG.json").exists():
        allowed={".executor.lock"}
        if any(p.name not in allowed for p in root.iterdir()):raise ValueError("Existing experiment cannot be migrated")
    frozen(root/"GEOMETRY_CONFIG.json",config)
    if config.get("agent_tools"):
        from .tool_runtime import bind_experiment
        bind_experiment(config)
    source=Path(config["generation_context_root"])
    catalog={t["target"]:t for t in read(source/"teachers.json")};subjects=[]
    for item in config["conditions"]:
        teacher=copy.deepcopy(catalog[item["teacher_id"]])
        # Explicit interaction target atoms must be in the same frame as this generation context.
        condition=prepare_condition(read(item["features_path"]),teacher["protein"],
            item.get("reference_protein",teacher["protein"]),common_valid_slots(source,teacher["target"]),
            assignment_policy=config.get("atom_assignment_policy",ASSIGNMENT))
        frozen(root/"conditions"/(teacher["target"]+".json"),condition)
        subjects.append(dict(teacher=teacher,condition=condition))
    if not subjects or len({s["teacher"]["target"] for s in subjects})!=len(subjects):raise ValueError("Unique generation contexts required")
    frozen(root/"SUBJECTS.json",subjects)
    return subjects


def calibrated(config):
    config=validate_config(config);root=Path(config["root"])
    if read(root/"GEOMETRY_CONFIG.json")!=config:raise ValueError("Frozen geometry configuration differs")
    subjects=read(root/"SUBJECTS.json")
    for subject in subjects:
        validate_condition(subject["condition"])
        if subject["condition"]["assignment_policy"]!=config.get("atom_assignment_policy",ASSIGNMENT):
            raise ValueError("Prepared condition uses another atom assignment policy; prepare a new experiment")
    return subjects


def require_acceptance(config):
    receipt=read(Path(config["root"])/"acceptance/COMPLETE.json")
    if receipt["source_binding"]!=source_binding() or receipt["config_sha256"]!=value_sha(config):
        raise ValueError("Geometry acceptance is stale or belongs to another configuration")


def calibrate(config,model_loader=load_model,executor=execute):
    from .geometry_calibration import run
    config=validate_config(config);require_acceptance(config)
    return run(config,calibrated(config),model_loader,executor)


def history(config,teacher,condition,domain):
    from .linear_scale import enabled as linear_enabled
    if linear_enabled(domain):
        from .linear_runtime import history as linear_history
        return linear_history(config,teacher,condition,domain)
    root=Path(config["root"]);expected=value_sha(compatibility(config,teacher,condition))
    validate_domain(domain,teacher["target"],condition["condition_sha256"],expected)
    records={};failures=[]
    for entry in sorted((root/"measurement_registry").glob("*.json")):
        claim=read(entry);identity=claim["identity"]
        if identity["compatibility"]["teacher_id"]!=teacher["target"]:continue
        if value_sha(identity["compatibility"])!=expected:
            raise ValueError("Incompatible measurement in this experiment; do not silently omit history")
        group=root/claim["source_group"]
        if not (group/"COMPLETE.json").exists():
            error=read(group/"FAILED.json") if (group/"FAILED.json").exists() else dict(error="interrupted without completion")
            failures.append(dict(id=claim["source_group"],effective_guidance_scale_schedule=identity[EFFECTIVE],failure=error,
                **({WEIGHTS:identity[WEIGHTS]} if WEIGHTS in identity else {})));continue
        row=validate_receipt(group)
        # Calibration populations stay separate; only full populations are action anchors.
        if row["attempts"]!=50:
            failures.append(dict(id=row["id"],evidence_status="smaller_population_not_directly_rankable",
                attempts=row["attempts"],**{EFFECTIVE:row[EFFECTIVE],DISTANCE:row[DISTANCE]}));continue
        if not all(x>0 for x in row[EFFECTIVE]):
            failures.append(dict(id=row["id"],evidence_status="disabled_guidance_not_an_action_anchor",**{EFFECTIVE:row[EFFECTIVE],DISTANCE:row[DISTANCE]}));continue
        normalized=[math.log2(x/domain[MULTIPLIER]) for x in row[EFFECTIVE]]
        normalized=[round(x) if abs(x-round(x))<1e-8 else x for x in normalized]
        try:ticks(normalized)
        except ValueError:
            failures.append(dict(id=row["id"],evidence_status="not_on_current_normalized_lattice",**{EFFECTIVE:row[EFFECTIVE],DISTANCE:row[DISTANCE]}));continue
        # Outside empirical-domain results are visible as evidence, never legal action anchors.
        if any(not domain["normalized_minimum"]<=x<=domain["normalized_maximum"] for x in normalized):
            failures.append(dict(id=row["id"],evidence_status="measured_outside_current_empirical_domain",
                **{EFFECTIVE:row[EFFECTIVE],DISTANCE:row[DISTANCE]},window_diagnostics=row["window_diagnostics"]));continue
        row=dict(row,**{NORMALIZED:normalized,MULTIPLIER:domain[MULTIPLIER]})
        if (group/"proposal.json").exists():row.update(proposal=read(group/"proposal.json"),proposal_sha256=sha(group/"proposal.json"))
        records[row["fingerprint"]]=row
    if domain["initial_source"] not in {r["id"] for r in records.values()}:raise ValueError("Measured multiplier baseline is absent")
    return sorted(records.values(),key=lambda r:r["id"]),failures


def retain(root,groups,winner,folder):
    from rdkit import Chem
    folder.mkdir(parents=True,exist_ok=True);pool={}
    for group in {g["id"]:g for g in groups}.values():
        for row in read(root/group["id"]/"records.json"):
            pool.setdefault((row["logical_seed"],row["logical_slot"]),[]).append(dict(row,
                source_group=group["id"],id=group["id"]+"/"+row["node_id"],complete=True,**{EFFECTIVE:group[EFFECTIVE]},
                **({WEIGHTS:group[WEIGHTS]} if WEIGHTS in group else {})))
    winners=[choose(v) for _,v in sorted(pool.items())]
    if len(winners)!=50:raise ValueError("Incomplete per-slot geometry retention")
    frozen(folder/"per_slot_top1.json",winners)
    with Chem.SDWriter(str(folder/"per_slot_top1_built.sdf")) as writer:
        for row in winners:
            mol=built_molecules(root/row["source_group"]/"molecules_all_built.sdf").get(row["node_id"])
            if mol is not None:mol.SetProp("source_group",row["source_group"]);writer.write(mol)
    frozen(folder/"RETENTION_COMPLETE.json",dict(entries=50,representative=winner["id"],all_sources_preserved=True,
        representative_terminal_sha256=sha(root/winner["id"]/"terminal.pt.gz"),representative_trace_sha256=sha(root/winner["id"]/"trace.h5")))


def run_version(config,version,model_loader=load_model,executor=execute,advisor=advise,retainer=retain):
    config=validate_config(config);require_acceptance(config);root=Path(config["root"]);subjects=calibrated(config)
    from .linear_scale import enabled as linear_enabled
    if linear_enabled(config):
        from .linear_runtime import run_version as run_linear
        from .linear_council import advise as linear_advise
        return run_linear(config,version,model_loader,executor,linear_advise if advisor is advise else advisor,retainer)
    if (not version.startswith("GD") or not version[2:].isdigit() or not 1<=int(version[2:])<=20
            or version!=f"GD{int(version[2:]):02d}"):raise ValueError("Geometry versions use GD01--GD20")
    folder=root/"versions"/version
    if (folder/"AWAITING_GPT_REVIEW.json").exists():return read(folder/"AWAITING_GPT_REVIEW.json")
    if config.get("agent_tools"):
        from .tool_runtime import bind_experiment
        bind_experiment(config)
    if int(version[2:])>1:
        approved=read(root/"versions"/f"GD{int(version[2:])-1:02d}"/"REVIEW_ACCEPTED.json")
        if approved["next_version"]!=version or approved["skill_hashes"]!=skill_hashes(config):
            raise ValueError("Previous actual GPT review has not accepted these skills")
    skills=folder/"skills";skills.mkdir(parents=True,exist_ok=True)
    for name,source in skill_assets(config).items():
        dest=skills/name
        if dest.exists() and sha(dest)!=sha(source):raise ValueError("Skills changed during geometry version")
        if not dest.exists():shutil.copyfile(source,dest)
    manifest=dict(protocol=POLICY,config_sha256=value_sha(config),source_binding=source_binding(),
        skill_hashes={p.name:sha(p) for p in skills.glob("*.md")},rounds=config["rounds"])
    frozen(folder/"manifest.json",manifest)
    domains=read(root/"calibration/domains.json");all_groups=[];summaries={};contexts=[]
    from .calibration_search import POLICY as adaptive_policy
    adaptive=config.get("calibration",{}).get("search_policy")==adaptive_policy
    missing_teachers=[s["teacher"]["target"] for s in subjects if s["teacher"]["target"] not in domains]
    skipped_branches=[]
    if adaptive:
        subjects=[s for s in subjects if s["teacher"]["target"] in domains]
        if not subjects:raise ValueError("No teacher has a confirmed calibration domain")
        frozen(folder/"CALIBRATION_SELECTION.json",dict(ready_teachers=list(domains),
            missing_teachers=missing_teachers,all_teachers_ready=not missing_teachers))
    from .geometry_calibration import verify_frozen
    for subject in subjects:
        verify_frozen(root,domains[subject["teacher"]["target"]],compatibility(config,subject["teacher"],subject["condition"]))
    for subject in subjects:
        teacher,condition=subject["teacher"],subject["condition"];target=teacher["target"];domain=domains[target]
        groups=[];troot=folder/target
        model=model_loader(config,teacher)
        try:
            initial_path=root/"INITIAL_GUIDANCE.json"
            if initial_path.exists() and version=="GD01":
                from .geometry_reuse import verify_initial
                selected=verify_initial(root,domains)[target]
                out=root/"initial_guidance"/target
                if executor is execute:
                    from .geometry_batch import execute_many
                    row=execute_many(model,config,teacher,condition,[dict(schedule=selected[NORMALIZED],folder=out,stage="formal",domain=domain)])[0]
                    if isinstance(row,Exception):raise row
                else:row=executor(model,config,teacher,condition,selected[NORMALIZED],out,"formal",domain=domain)
                frozen(troot/"INITIAL_MEASUREMENT.json",dict(selected=selected,source_group=row["id"],fingerprint=row["fingerprint"]))
            for iteration in range(1,config["rounds"]+1):
                if source_binding()!=manifest["source_binding"]:raise ValueError("Executable sources changed mid-version")
                if {p.name:sha(p) for p in skills.glob("*.md")}!=manifest["skill_hashes"]:
                    raise ValueError("Frozen skills changed mid-version")
                call_folder=troot/f"advice_{iteration:03d}"
                snapshot=call_folder/"history_snapshot.json"
                current,failed=history(config,teacher,condition,domain)
                if not snapshot.exists():frozen(snapshot,dict(history=current,failures=failed,domain_sha256=domain["domain_sha256"]))
                saved=read(snapshot)
                if saved["domain_sha256"]!=domain["domain_sha256"]:raise ValueError("History snapshot domain changed")
                current_by_id={r["id"]:r for r in current}
                if any(value_sha(current_by_id.get(r["id"]))!=value_sha(r) for r in saved["history"]):raise ValueError("Historical source changed on replay")
                proposals=advisor(config,saved["history"],domain,call_folder,skills,saved["failures"])
                branches=[p["branch"] for p in proposals]
                expected={"increase","decrease","redistribute"}
                if len(branches)!=len(set(branches)) or set(branches)-expected or not adaptive and set(branches)!=expected:
                    raise ValueError("Incomplete or duplicated inclination selection")
                for branch in sorted(expected-set(branches)):
                    boundary=read(call_folder/(branch+".BOUNDARY_REACHED.json"))
                    skipped_branches.append(dict(boundary,teacher_id=target,iteration=iteration))
                resolved_seen=[seen_configuration(r[NORMALIZED],r.get(WEIGHTS)) for r in saved["history"]]
                jobs=[];resolved_actions=[]
                for proposal in proposals:
                    if condition.get("reward_policy"):
                        from .analysis_tools import EvidenceTools
                        registry=EvidenceTools(root,saved["history"],domain,call_folder/"task_memory/calculations",condition=condition)
                        proposal=registry.resolve_loss_control(proposal,allow_saved=True)
                    resolved=resolve_action(proposal,{r["id"]:r for r in saved["history"]},domain,resolved_seen)
                    resolved_seen.append(seen_configuration(resolved[NORMALIZED],resolved.get(WEIGHTS)))
                    out=troot/f"iteration_{iteration:03d}"/proposal["branch"]
                    frozen(out/"agent_action.json",proposal);frozen(out/"proposal.json",resolved)
                    jobs.append(dict(schedule=resolved[NORMALIZED],folder=out,stage="formal",domain=domain,
                        **({"loss_weights":resolved[WEIGHTS]} if WEIGHTS in resolved else {})));resolved_actions.append(resolved)
                if executor is execute:
                    from .geometry_batch import execute_many
                    measured=execute_many(model,config,teacher,condition,jobs)
                else:measured=[executor(model,config,teacher,condition,j["schedule"],j["folder"],"formal",domain=domain,
                    **({"loss_weights":j["loss_weights"]} if "loss_weights" in j else {})) for j in jobs]
                for row,resolved in zip(measured,resolved_actions):
                    if isinstance(row,Exception):raise row
                    groups.append(dict(row,selected_proposal=resolved,iteration=iteration))
            current,failed=history(config,teacher,condition,domain)
            advisor(config,current,domain,troot/"analysis_final",skills,failed,final=True)
            winner=choose(groups or current);retainer(root,groups or current,winner,troot/"retention")
            contexts.extend([str(troot/"analysis_final"/"increase.json"),str(troot/"analysis_final"/"synthesis.json")])
            from .coordinate_statistics import RANKING, SUMMARY
            summaries[target]=dict(representative=winner["id"],**{k:winner[k] for k in (DISTANCE,CONTROL_DISTANCE,NEIGHBOR_DISTANCE,RANKING,SUMMARY)},
                **{k:winner[k] for k in (WEIGHTS,INTERACTION_DISTANCE) if k in winner},
                domain_sha256=domain["domain_sha256"],complete=True)
            all_groups.extend(groups)
        finally:
            del model
            torch.cuda.empty_cache()
    if len(all_groups)+len(skipped_branches)!=len(subjects)*3*config["rounds"]:raise ValueError("Incomplete configured iteration count")
    result=dict(version=version,teachers=summaries,groups=all_groups,strategy_contexts=contexts,
        manifest_sha256=sha(folder/"manifest.json"),complete=not missing_teachers and not skipped_branches,
        missing_teachers=missing_teachers,skipped_branches=skipped_branches,
        requested_groups=(len(subjects)+len(missing_teachers))*3*config["rounds"],
        measured_or_reused_groups=len(all_groups),all_available_teachers_processed=True)
    frozen(folder/"FINALIZED.json",result)
    previous=[read(p) for p in sorted((root/"versions").glob("GD*/FINALIZED.json")) if p.parent.name<version]
    best={target:min(previous+[result],key=lambda v:primary_score(v["teachers"][target]))["version"] for target in summaries}
    request=dict(version=version,current=result,previous=previous[-1] if previous else None,best_version_by_teacher=best,
        measured_initials={target:domain["initial_source"] for target,domain in domains.items()},
        next_version=None if version=="GD20" else f"GD{int(version[2:])+1:02d}",
        strategy_contexts=contexts,requires_actual_main_agent_review=True)
    write(folder/"AWAITING_GPT_REVIEW.json",request);return request


def accept_review(config,version,path):
    config=validate_config(config)
    root=Path(config["root"]);folder=root/"versions"/version;request=read(folder/"AWAITING_GPT_REVIEW.json");review=read(path)
    if review.get("version")!=version or review.get("review_request_sha256")!=sha(folder/"AWAITING_GPT_REVIEW.json"):
        raise ValueError("Review must bind actual completed geometry evidence")
    if set(review.get("read_strategy_contexts",[]))!=set(request["strategy_contexts"]):raise ValueError("Read all required contexts")
    before={p.name:sha(p) for p in (folder/"skills").glob("*.md")};after=skill_hashes(config)
    changed={k for k in before if before[k]!=after.get(k)};patches=review.get("patches",[])
    if config.get("agent_tools"):
        from .tool_runtime import bind_experiment
        binding=bind_experiment(config)
        if before!=after or after!=binding["skill_hashes"] or patches:
            raise ValueError("Experiment review cannot change frozen skills; external RSI uses a new experiment/release")
        frozen(folder/"REVIEW_ACCEPTED.json",dict(review_sha256=sha(path),skill_hashes=after,
            next_version=request["next_version"],review_scope="results only; skills remain experiment-frozen"))
        return
    if set(before)!=set(after) or not 1<=len(changed)<=3 or {p["module"] for p in patches}!=changed or len(patches)!=len(changed):
        raise ValueError("Review must account for one to three actual skill changes")
    for patch in patches:
        if not all(patch.get(k) for k in ["observation","evidence","mechanism_applicability","skill_edit","expected_behavior","next_version_falsifier"]):
            raise ValueError("Incomplete evidence-to-skill review chain")
    frozen(folder/"REVIEW_ACCEPTED.json",dict(review_sha256=sha(path),skill_hashes=after,next_version=request["next_version"]))
