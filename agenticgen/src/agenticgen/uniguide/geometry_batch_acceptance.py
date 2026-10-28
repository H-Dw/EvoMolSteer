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
from .native_tools import load_pt,tensor_digest
from .batching import take
from .random_streams import capture,restore,equal
from .io import frozen,write


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
        scale=scale_binding(schedule);unit=dict(id="fixture"+str(i),slots=slots,scale=scale)
        units.append(unit);out=folder/f"separate_{i}"
        factory=functools.partial(GeometryControl,condition=condition,scale_binding=scale)
        sampler.sample(model,config,teacher,schedule,out,10,control_factory=factory,check_affinity=False,
            initial_override=initial_shard(config,teacher["target"],slots))
        separate.append(out)
    out=folder/"mixed_twenty";before=capture();prior=getattr(model,"_ug_physical_rows",10)
    try:
        model._ug_physical_rows=20
        factory=functools.partial(CandidateControl,condition=condition,scale_binding=units[0]["scale"],units=units,physical_rows=20)
        _,_,_,control=sampler.sample(model,config,teacher,units[0]["scale"][EFFECTIVE],out,20,
            control_factory=factory,check_affinity=False,initial_override=batch_initial(config,teacher["target"],units))
        if control.failed_units:raise ValueError("Acceptance fixture produced numerical failures")
    finally:model._ug_physical_rows=prior
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
    receipt=dict(exact_equivalence=not differences,differences=differences,
        independent_groups=2,mixed_batch_size=20,full_steps=100,external_rng_unchanged=True,
        fixture_is_not_calibration=True,engineering_trajectory_evaluations=40)
    write(folder/"COMPARISON.json",receipt)
    if differences:raise ValueError("Mixed batches changed independent trajectories; see COMPARISON.json")
    # Try all ninety candidates; reduce only upon a real device OOM.
    capacity=90;ooms=[]
    while True:
        selected=[copy.deepcopy(units[i%2]) for i in range(capacity//10)]
        for i,u in enumerate(selected):u["id"]=f"capacity_fixture_{i}"
        before=capture();error=None
        try:
            model._ug_physical_rows=capacity
            factory=functools.partial(CandidateControl,condition=condition,scale_binding=selected[0]["scale"],units=selected,physical_rows=capacity)
            sampler.sample(model,config,teacher,selected[0]["scale"][EFFECTIVE],folder/f"capacity_{capacity}",capacity,
                control_factory=factory,check_affinity=False,initial_override=batch_initial(config,teacher["target"],selected),stop_step=12)
        except Exception as exc:
            if not device_oom(exc):raise
            error=str(exc)
        finally:
            restore(before);model._ug_physical_rows=prior;model._adaptive=None;model.integrator._adaptive_observer=None
        if error is None:break
        ooms.append(dict(capacity=capacity,error=error));capacity=lower_capacity(capacity)
        import gc
        gc.collect();torch.cuda.empty_cache()
        if capacity is None:raise RuntimeError("No accepted GPU capacity")
    receipt.update(capacity_probe_initial=90,capacity_probe_success=capacity,capacity_probe_ooms=ooms,
        capacity_probe_steps=12,capacity_probe_not_calibration=True)
    frozen(folder/"COMPLETE.json",receipt)
    return receipt
