"""Remote engineering checks only: no completed molecules, LLM calls or fitted multipliers."""
import functools
from pathlib import Path
import torch
import h5py
import numpy as np
from .io import read,write,frozen,sha,value_sha
from .native_tools import load_pt,device,tensor_digest
from .batching import take
from .prepare import load_model
from .sampler import sample,population
from .geometry_sampling import GeometryControl,scale_binding
from .absolute_coordinate import loss, match
from .geometry_protocol import source_binding
from .random_streams import capture,equal
from .geometry_shards import initial_shard
from .geometry_dynamics import Raw999Sampling
from .sampler import Control


class UnguidedRaw999(Raw999Sampling,Control):
    """Independent native-transition reference with the same grid/output policy."""
    def expand_schedule(self,schedule):return [0.]*100


def real_vjp(model,config,teacher,condition):
    prefix=Path(config["root"])/"acceptance"/teacher["target"]/"raw999_vjp_prefix"
    initial=initial_shard(config,teacher["target"],[0,1])
    disabled=scale_binding([0.]*4,1.,False,condition)
    factory=functools.partial(GeometryControl,condition=condition,scale_binding=disabled)
    sample(model,config,teacher,disabled["effective_guidance_scale_schedule"],prefix,10,
        stop_step=10,control_factory=factory,check_affinity=False,initial_override=initial)
    source=prefix/"checkpoint_010.pt.gz";state=load_pt(source)
    curr,cond,times=(device(state[k]) for k in ("current","cond","times"))
    pocket=device(initial["pocket_target"]);com=initial["target_com"].to("cuda")
    with torch.no_grad():
        equis,invs=model.gen.get_pocket_encoding(pocket["coords"],pocket["atom_names"],
            pocket_atom_charges=pocket["charges"].argmax(-1),pocket_bond_types=pocket["bonds"].argmax(-1),
            pocket_res_types=pocket["res_names"],pocket_atom_mask=pocket["mask"])
    hooks=[];fixed_assignment=None
    def denied(gradient):raise AssertionError("Affinity head entered geometry VJP")
    def objective(x):
        nonlocal fixed_assignment
        out=model(dict(curr,coords=x),pocket,times,cond_batch=cond if model.self_condition else None,
            pocket_equis=equis,pocket_invs=invs,training=False)
        for value in out.get("affinity",{}).values():
            if torch.is_tensor(value) and value.requires_grad:hooks.append(value.register_hook(denied))
        pred,_=model._get_predictions(out)
        endpoint=pred["coords"]*model.coord_scale+com[:,None]
        if fixed_assignment is None:fixed_assignment=match(endpoint,condition,curr["mask"].bool())
        return loss(endpoint,condition,curr["mask"].bool(),fixed_assignment)[0].sum()
    before=capture();x=curr["coords"].detach().clone().requires_grad_(True)
    gradient=torch.autograd.grad(objective(x),x)[0].detach();norm=gradient.norm()
    if not torch.isfinite(gradient).all() or norm<=1e-8:raise ValueError("Invalid real model input gradient")
    direction=gradient/norm;errors=[]
    with torch.no_grad():
        for epsilon in (.01,.005):
            numerical=(objective(x.detach()+epsilon*direction)-objective(x.detach()-epsilon*direction))/(2*epsilon)
            errors.append(float(abs(numerical-norm)/norm))
    if max(errors)>.05:raise ValueError("Real coordinate VJP finite-difference discrepancy: "+str(errors))
    if not equal(before,capture()):raise ValueError("VJP consumed sampling RNG")
    if any(p.requires_grad or p.grad is not None for p in model.parameters()):raise ValueError("Model parameters not frozen")
    for hook in hooks:hook.remove()
    return dict(relative_errors=errors,tolerance=.05,input_gradient_norm=float(norm),
        affinity_gradient_entered=False,sampling_rng_unchanged=True,model_parameters_frozen=True,
        source_checkpoint_sha256=sha(source),condition_sha256=condition["condition_sha256"])


def run(config):
    from .absolute_runtime import calibrated
    root=Path(config["root"]);subjects=calibrated(config);folder=root/"acceptance";checks={}
    original=Path(config["generation_context_root"])/"attempts.json";before=sha(original)
    for subject in subjects:
        teacher,condition=subject["teacher"],subject["condition"];out=folder/teacher["target"]
        model=load_model(config,teacher)
        numerical=real_vjp(model,config,teacher,condition)
        off=scale_binding([0.]*4,1.,False,condition)
        factory=functools.partial(GeometryControl,condition=condition,scale_binding=off)
        sample(model,config,teacher,off["effective_guidance_scale_schedule"],out/"disabled",10,stop_step=3,
            control_factory=factory,check_affinity=False)
        sample(model,config,teacher,[1.]*4,out/"native_prefix",10,stop_step=3,control_factory=UnguidedRaw999)
        a,b=(load_pt(out/name/"checkpoint_003.pt.gz") for name in ("disabled","native_prefix"))
        disabled={k:tensor_digest(a[k])==tensor_digest(b[k]) for k in ("current","cond","times","stream_states")}
        if not all(disabled.values()):raise ValueError("Disabled distance guidance differs from native prefix")
        active=scale_binding([0.]*4,1.,True,condition)  # Test fixture only, NOT a fitted lambda.
        factory=functools.partial(GeometryControl,condition=condition,scale_binding=active)
        sample(model,config,teacher,active["effective_guidance_scale_schedule"],out/"continuous",10,stop_step=12,
            control_factory=factory,check_affinity=False)
        sample(model,config,teacher,active["effective_guidance_scale_schedule"],out/"resumed",10,stop_step=12,
            resume_path=out/"continuous/checkpoint_010.pt.gz",control_factory=factory,check_affinity=False)
        a,b=(load_pt(out/name/"checkpoint_012.pt.gz") for name in ("continuous","resumed"))
        resumed={k:tensor_digest(a[k])==tensor_digest(b[k]) for k in ("current","cond","times","stream_states","previous_assignment")}
        if not all(resumed.values()):raise ValueError("Distance guidance resume changed numerical state")
        steps=read(out/"continuous/steps.json")
        if any(any(v!=0 for v in s["assignment_switch_fraction"]) for s in steps if s["step"]>10):raise ValueError("Mapping changed during trajectory")
        if any(not any(v>0 for v in s["gradient_norm"]) for s in steps if s["step"]>=10):raise ValueError("Normalized zero incorrectly disabled guidance")
        chemical_check={"enabled":config.get("chemical_inpainting",{"enabled":True})["enabled"]}
        if chemical_check["enabled"]:
            # Chemistry starts after the first gradient update; all native/coordinate
            # operations through that update must equal a paired non-inpainting run.
            plain=dict(config,chemical_inpainting={"enabled":False})
            sample(model,plain,teacher,active["effective_guidance_scale_schedule"],out/"without_chemistry_prefix",10,stop_step=12,
                control_factory=factory,check_affinity=False)
            with h5py.File(out/"continuous/trace.h5") as enabled_trace,h5py.File(out/"without_chemistry_prefix/trace.h5") as disabled_trace:
                for key in disabled_trace:
                    if not np.array_equal(enabled_trace[key][:11],disabled_trace[key][:11]):
                        raise ValueError("Chemistry changed the pre-injection prefix: "+key)
            enabled_state=load_pt(out/"continuous/checkpoint_012.pt.gz")
            disabled_state=load_pt(out/"without_chemistry_prefix/checkpoint_012.pt.gz")
            if tensor_digest(enabled_state["stream_states"])!=tensor_digest(disabled_state["stream_states"]):
                raise ValueError("Chemical inpainting consumed extra RNG")
            chemical_check.update(first_pre_inpainting_update_exact=True,extra_rng_consumed=False,
                coordinates_projected=False,self_conditioning_projected=False)
        sample(model,config,teacher,active["effective_guidance_scale_schedule"],out/"prefix_twenty",20,stop_step=12,
            control_factory=factory,check_affinity=False)
        selected=initial_shard(config,teacher["target"],[2,3])
        sample(model,config,teacher,active["effective_guidance_scale_schedule"],out/"promoted_slots",10,stop_step=12,
            control_factory=factory,check_affinity=False,initial_override=selected)
        full=load_pt(out/"prefix_twenty/checkpoint_012.pt.gz");shard=load_pt(out/"promoted_slots/checkpoint_012.pt.gz")
        indices=[seed*4+slot for seed in range(5) for slot in (2,3)]
        promotion={k:tensor_digest(take(full[k],indices,20))==tensor_digest(shard[k]) for k in ("current","cond","times","previous_assignment")}
        promotion["native_rng_states"]=tensor_digest(full["stream_states"])==tensor_digest(shard["stream_states"])
        if not all(promotion.values()):raise ValueError("Promoted native slots differ from original prefix: "+str(promotion))
        sample(model,config,teacher,active["effective_guidance_scale_schedule"],out/"promoted_resumed",10,stop_step=12,
            control_factory=factory,check_affinity=False,initial_override=selected,resume_path=out/"promoted_slots/checkpoint_010.pt.gz")
        resumed_shard=load_pt(out/"promoted_resumed/checkpoint_012.pt.gz")
        if any(tensor_digest(shard[k])!=tensor_digest(resumed_shard[k]) for k in ("current","cond","times","stream_states","previous_assignment")):
            raise ValueError("Promoted slots changed after resume")
        checks[teacher["target"]]=dict(real_model_vjp=numerical,disabled_equivalence=disabled,exact_resume=resumed,
            chemical_state_inpainting=chemical_check,
            exact_native_slot_promotion=promotion,promoted_slot_resume=True,
            fixed_correspondence=True,normalized_zero_is_active=True,fixture_multiplier=1.,fixture_is_not_calibration=True)
        from .geometry_batch_acceptance import run_teacher
        checks[teacher["target"]]["candidate_batch_equivalence"]=run_teacher(model,config,teacher,condition,out/"candidate_batch")
        del model;torch.cuda.empty_cache()
    if sha(original)!=before:raise ValueError("Original attempt ledger changed")
    receipt=dict(source_binding=source_binding(),config_sha256=value_sha(config),checks=checks,
        torch_version=torch.__version__,new_completed_candidates=0,llm_calls=0,fitted_multipliers=False,
        engineering_trajectory_evaluations=60*len(subjects),engineering_outputs_not_experiment_measurements=True,
        original_attempts_sha256=before,engineering_only=True)
    frozen(folder/"COMPLETE.json",receipt);return receipt
