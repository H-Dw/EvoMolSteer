"""Real-model comparison of mixed candidate execution against independent groups."""
import copy
import functools
from pathlib import Path
import h5py
import numpy as np
import torch
from . import sampler
from .geometry_batch import CandidateControl, batch_initial, lower_capacity, device_oom
from .geometry_sampling import GeometryControl, scale_binding
from .geometry_shards import initial_shard
from .geometry_protocol import EFFECTIVE
from .geometry_reward import weights
from .native_tools import load_pt,tensor_digest
from .batching import take
from .random_streams import capture,restore,equal
from .io import frozen,write,read


def compare_trees(left,right):
    differences=[]
    def visit(a,b,path):
        if torch.is_tensor(a):
            if a.shape!=b.shape or a.dtype!=b.dtype:raise ValueError("Shape/dtype differ at "+path)
            if not torch.equal(a,b):
                error=float((a.double()-b.double()).abs().max()) if a.numel() else 0.
                differences.append(dict(path=path,maximum_absolute_error=error))
        elif isinstance(a,dict):
            if set(a)!=set(b):raise ValueError("Keys differ at "+path)
            for k in a:visit(a[k],b[k],path+"/"+str(k))
        elif isinstance(a,(list,tuple)):
            if len(a)!=len(b):raise ValueError("Length differs at "+path)
            for i,(x,y) in enumerate(zip(a,b)):visit(x,y,path+"/"+str(i))
        elif a!=b:differences.append(dict(path=path,non_tensor_difference=True))
    visit(left,right,"")
    return differences


def run_teacher(model,config,teacher,condition,folder):
    folder=Path(folder);units=[];separate=[]
    # Heterogeneous schedules and non-prefix RNG slots, deliberately not fitted lambdas.
    for i,(slots,schedule) in enumerate([([0,1],[.001,.002,.001,.004]),([6,7],[.004,.001,.002,.001])]):
        chosen=None
        if condition.get("reward_policy"):
            chosen=weights(condition)
            if i and any(p["category"]=="controlled_neighborhood" for p in condition["pairs"]):
                chosen["condition_outer_weight"]=1
        scale=scale_binding(schedule,condition=condition,loss_weights=chosen);unit=dict(id="fixture"+str(i),slots=slots,scale=scale)
        units.append(unit);out=folder/f"separate_{i}"
        factory=functools.partial(GeometryControl,condition=condition,scale_binding=scale)
        sampler.sample(model,config,teacher,schedule,out,10,control_factory=factory,check_affinity=False,
            initial_override=initial_shard(config,teacher["target"],slots))
        separate.append(out)
    out=folder/"mixed_twenty";before=capture();prior=getattr(model,"_ug_physical_rows",10)
    prior_workers=getattr(model,"_ug_parallel_workers",1)
    try:
        model._ug_physical_rows=10;model._ug_parallel_workers=2
        factory=functools.partial(CandidateControl,condition=condition,scale_binding=units[0]["scale"],units=units,physical_rows=20)
        _,_,_,control=sampler.sample(model,config,teacher,units[0]["scale"][EFFECTIVE],out,20,
            control_factory=factory,check_affinity=False,initial_override=batch_initial(config,teacher["target"],units))
        if control.failed_units:raise ValueError("Acceptance fixture produced numerical failures")
    finally:model._ug_physical_rows=prior;model._ug_parallel_workers=prior_workers
    if not equal(before,capture()):raise ValueError("Mixed batch consumed external RNG")
    differences=[];joint=load_pt(out/"terminal.pt.gz");state=load_pt(out/"checkpoint_100.pt.gz")
    for i,source in enumerate(separate):
        indices=list(range(i*10,(i+1)*10));original=load_pt(source/"checkpoint_100.pt.gz")
        differences.extend(compare_trees(load_pt(source/"terminal.pt.gz"),take(joint,indices,20)))
        for field in ("current","cond","times","previous_assignment"):
            differences.extend(compare_trees(original[field],take(state[field],indices,20)))
        if tensor_digest(original["stream_states"])!=tensor_digest(state["stream_states"][i]):raise ValueError("Mixed RNG diverged")
        with h5py.File(source/"trace.h5","r") as a,h5py.File(out/"trace.h5","r") as b:
            for key in a:
                left=a[key][:];right=b[key][:,indices]
                if not np.array_equal(left,right):differences.append(dict(path="trace/"+key,maximum_absolute_error=float(np.max(np.abs(left.astype(float)-right.astype(float))))))
            if not np.array_equal(load_pt(source/"terminal.pt.gz")["coords"].numpy(),a["endpoint_A"][99]):
                raise ValueError("Returned coordinates are not the uncovered last-loop prediction")
            if config.get("chemical_inpainting",{"enabled":True})["enabled"]:
                chemical=read(source/"CHEMICAL_CONDITION.json");ids=chemical["generated_slots"]
                for key,target in chemical["features"].items():
                    actual=a["inpainted_chemical_"+key][10:100]
                    expected=np.array(target).argmax(-1)
                    if key=="bonds":
                        for left in range(len(ids)):
                            for right in range(len(ids)):
                                if left!=right and not (actual[:,:,ids[left],ids[right]]==expected[left,right]).all():
                                    raise ValueError("Induced chemical bond was not inpainted")
                    elif not (actual[:,:,ids]==expected).all():raise ValueError("Controlled chemical channels were not inpainted")
    receipt=dict(exact_equivalence=not differences,differences=differences,
        independent_groups=2,mixed_batch_size=20,full_steps=100,external_rng_unchanged=True,
        fixture_is_not_calibration=True,engineering_trajectory_evaluations=40,
        raw999_coordinates_exact=True,chemical_state_inpainting_checked=bool(config.get("chemical_inpainting",{"enabled":True})["enabled"]))
    write(folder/"COMPARISON.json",receipt)
    if differences:raise ValueError("Mixed batches changed independent trajectories; see COMPARISON.json")
    try:
        model._ug_physical_rows=10;model._ug_parallel_workers=2
        sampler.sample(model,config,teacher,units[0]["scale"][EFFECTIVE],folder/"mixed_resumed",20,
            control_factory=factory,check_affinity=False,initial_override=batch_initial(config,teacher["target"],units),
            resume_path=out/"checkpoint_010.pt.gz")
    finally:model._ug_physical_rows=prior;model._ug_parallel_workers=prior_workers
    if compare_trees(joint,load_pt(folder/"mixed_resumed/terminal.pt.gz")):raise ValueError("Mixed batch resume changed output")
    receipt.update(exact_mixed_resume=True,engineering_trajectory_evaluations=60,network_microbatch_size=10,
        concurrent_units=2,physical_batch_shape_is_fixed=True)
    # Try all ninety candidates; reduce only upon a real device OOM.
    capacity=90;ooms=[]
    while True:
        selected=[copy.deepcopy(units[i%2]) for i in range(capacity//10)]
        for i,u in enumerate(selected):u["id"]=f"capacity_fixture_{i}"
        before=capture();error=None
        try:
            model._ug_physical_rows=10;model._ug_parallel_workers=capacity//10
            factory=functools.partial(CandidateControl,condition=condition,scale_binding=selected[0]["scale"],units=selected,physical_rows=capacity)
            sampler.sample(model,config,teacher,selected[0]["scale"][EFFECTIVE],folder/f"capacity_{capacity}",capacity,
                control_factory=factory,check_affinity=False,initial_override=batch_initial(config,teacher["target"],selected),stop_step=12)
        except Exception as exc:
            if not device_oom(exc):raise
            error=str(exc)
        finally:
            restore(before);model._ug_physical_rows=prior;model._ug_parallel_workers=prior_workers
            model._adaptive=None;model.integrator._adaptive_observer=None
        if error is None:break
        ooms.append(dict(capacity=capacity,error=error));capacity=lower_capacity(capacity)
        import gc
        gc.collect();torch.cuda.empty_cache()
        if capacity is None:raise RuntimeError("No accepted GPU capacity")
    receipt.update(capacity_probe_initial=90,capacity_probe_success=capacity,capacity_probe_ooms=ooms,
        capacity_probe_steps=12,capacity_probe_not_calibration=True)
    with h5py.File(folder/f"capacity_{capacity}"/"trace.h5","r") as combined:
        for i in range(capacity//10):
            with h5py.File(separate[i%2]/"trace.h5","r") as single:
                for key in single:
                    if not np.array_equal(single[key][:12],combined[key][:12,i*10:(i+1)*10]):
                        raise ValueError("Capacity probe changed independent prefix: "+key)
    receipt["capacity_probe_exact_prefix_equivalence"]=True
    # Force only the second candidate unit across the radius; the first must
    # remain identical to its separately executed low-strength trajectory.
    stress=[copy.deepcopy(units[0]),dict(id="limited_fixture",slots=[6,7],scale=scale_binding([1e6]*4,condition=condition))]
    limited=folder/"candidate_local_limit"
    try:
        model._ug_parallel_workers=2
        factory=functools.partial(CandidateControl,condition=condition,scale_binding=stress[0]["scale"],units=stress,physical_rows=20)
        sampler.sample(model,config,teacher,stress[0]["scale"][EFFECTIVE],limited,20,control_factory=factory,
            check_affinity=False,initial_override=batch_initial(config,teacher["target"],stress),stop_step=12)
    finally:model._ug_parallel_workers=prior_workers
    steps=read(limited/"steps.json")
    assert any(any(s["guidance_increment_limited"][10:]) for s in steps)
    assert not any(any(s["guidance_increment_limited"][:10]) for s in steps)
    limit=config["maximum_single_atom_guidance_increment_angstrom"]
    with h5py.File(limited/"trace.h5") as mixed,h5py.File(separate[0]/"trace.h5") as single:
        assert np.linalg.norm(mixed["guidance_delta_A"][:12].astype(np.float64),axis=-1).max()<=limit
        for key in single:
            assert np.array_equal(single[key][:12],mixed[key][:12,:10]),key
    receipt.update(candidate_local_limit_triggered=True,unaffected_candidates_exact=True,
        guidance_increment_limit_angstrom=limit,limiter_stress_steps=12,
        stress_is_engineering_only=True,stress_multiplier_is_not_calibrated=True)
    frozen(folder/"COMPLETE.json",receipt)
    return receipt
