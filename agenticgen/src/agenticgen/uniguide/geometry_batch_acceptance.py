"""Real-model comparison of mixed candidate execution against independent groups."""
import copy
import functools
from pathlib import Path
import h5py
import numpy as np
import torch
from . import sampler
from .geometry_batch import CandidateControl, batch_initial, DEFAULT_BATCH_SIZE
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


class NonfiniteGradientFixture(CandidateControl):
    """Engineering-only fault injection into the second immutable RNG shard."""

    def check_rows(self, name, value, ids=None):
        should_inject = (
            name == "input gradient" and ids is not None and ids[0] == 10
            and self.active_step == 10
        )
        if should_inject:
            value = value.clone()
            value[0, 0, 0] = float("nan")
        return super().check_rows(name, value, ids)


def verify_failure_isolation(model, config, teacher, condition, folder, units, reference):
    """An invalid gradient must not alter a different healthy microbatch."""
    destination = folder / "nonfinite_isolation"
    factory = functools.partial(
        NonfiniteGradientFixture, condition=condition, scale_binding=units[0]["scale"],
        units=units, physical_rows=20,
    )
    _, _, _, control = sampler.sample(
        model, config, teacher, units[0]["scale"][EFFECTIVE], destination, 20,
        control_factory=factory, check_affinity=False,
        initial_override=batch_initial(config, teacher["target"], units), stop_step=12,
    )
    if set(control.failed_units) != {1}:
        raise ValueError("Injected failure was not isolated to its RNG shard")
    with h5py.File(destination / "trace.h5") as mixed, h5py.File(reference / "trace.h5") as single:
        for key in single:
            if not np.array_equal(single[key][:12], mixed[key][:12, :10]):
                raise ValueError("Failed shard changed a healthy trajectory: " + key)
    return dict(healthy_prefix_exact=True, failed_shards=[1], shard_rows=10,
                injected_step=10, checked_steps=12, evidence_preserved=True)


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
    out=folder/"mixed_twenty";before=capture()
    factory=functools.partial(CandidateControl,condition=condition,scale_binding=units[0]["scale"],units=units,physical_rows=20)
    _,_,_,control=sampler.sample(model,config,teacher,units[0]["scale"][EFFECTIVE],out,20,
        control_factory=factory,check_affinity=False,initial_override=batch_initial(config,teacher["target"],units))
    if control.failed_units:raise ValueError("Acceptance fixture produced numerical failures")
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
                dynamic_initial = not ids
                if dynamic_initial:
                    ids=a["assignment"][10:100]
                    if not (ids==ids[0:1]).all():raise ValueError("Initial assignment was not frozen")
                for key,target in chemical["features"].items():
                    actual=a["inpainted_chemical_"+key][10:100]
                    expected=np.array(target).argmax(-1)
                    if dynamic_initial:
                        if key=="bonds":
                            ti=np.arange(len(ids))[:,None];bi=np.arange(ids.shape[1])[None,:]
                            for left in range(ids.shape[-1]):
                                for right in range(ids.shape[-1]):
                                    if left!=right and not (actual[ti,bi,ids[:,:,left],ids[:,:,right]]==expected[left,right]).all():
                                        raise ValueError("Frozen assigned chemical bond was not inpainted")
                        elif not (np.take_along_axis(actual,ids,axis=2)==expected[None,None]).all():
                            raise ValueError("Frozen assigned chemical channel was not inpainted")
                        continue
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
    sampler.sample(model,config,teacher,units[0]["scale"][EFFECTIVE],folder/"mixed_resumed",20,
        control_factory=factory,check_affinity=False,initial_override=batch_initial(config,teacher["target"],units),
        resume_path=out/"checkpoint_010.pt.gz")
    if compare_trees(joint,load_pt(folder/"mixed_resumed/terminal.pt.gz")):raise ValueError("Mixed batch resume changed output")
    receipt.update(exact_mixed_resume=True,engineering_trajectory_evaluations=60,network_microbatch_size=10,
        concurrent_units=1,physical_batch_shape_is_fixed=True)
    # Validate the production default over the full integration, not a short prefix.
    capacity=DEFAULT_BATCH_SIZE
    selected=[copy.deepcopy(units[i%2]) for i in range(capacity//10)]
    for i,u in enumerate(selected):u["id"]=f"capacity_fixture_{i}"
    before=capture()
    try:
        factory=functools.partial(CandidateControl,condition=condition,scale_binding=selected[0]["scale"],units=selected,physical_rows=capacity)
        sampler.sample(model,config,teacher,selected[0]["scale"][EFFECTIVE],folder/f"capacity_{capacity}",capacity,
            control_factory=factory,check_affinity=False,initial_override=batch_initial(config,teacher["target"],selected),stop_step=100)
    except Exception as exc:
        write(folder/"CAPACITY_FAILED.json",dict(capacity=capacity,error=str(exc),
            no_automatic_retry=True,no_capacity_fallback=True))
        raise
    finally:
        restore(before)
        model._adaptive=None;model.integrator._adaptive_observer=None
    receipt.update(capacity_probe_initial=DEFAULT_BATCH_SIZE,capacity_probe_success=capacity,capacity_probe_ooms=[],
        capacity_probe_steps=100,capacity_probe_not_calibration=True)
    with h5py.File(folder/f"capacity_{capacity}"/"trace.h5","r") as combined:
        for i in range(capacity//10):
            with h5py.File(separate[i%2]/"trace.h5","r") as single:
                for key in single:
                    if not np.array_equal(single[key][:100],combined[key][:100,i*10:(i+1)*10]):
                        raise ValueError("Capacity probe changed independent trajectory: "+key)
    receipt["capacity_probe_exact_prefix_equivalence"]=True
    from .geometry_dynamics import policy as dynamics_policy
    if dynamics_policy(config)["limiter"] == "none":
        receipt.update(verify_uncapped_guidance(model,config,teacher,condition,folder,units,separate[0]))
        receipt["nonfinite_isolation"] = verify_failure_isolation(model, config, teacher, condition, folder, units, separate[0])
        receipt.update(engineering_inventory(folder))
        frozen(folder/"COMPLETE.json",receipt)
        return receipt
    # Legacy distance protocols retain their own historical radius acceptance.
    # Force only the second candidate unit across the radius; the first must
    # remain identical to its separately executed low-strength trajectory.
    stress=[copy.deepcopy(units[0]),dict(id="limited_fixture",slots=[6,7],scale=scale_binding([1e6]*4,condition=condition))]
    limited=folder/"candidate_local_limit"
    factory=functools.partial(CandidateControl,condition=condition,scale_binding=stress[0]["scale"],units=stress,physical_rows=20)
    sampler.sample(model,config,teacher,stress[0]["scale"][EFFECTIVE],limited,20,control_factory=factory,
        check_affinity=False,initial_override=batch_initial(config,teacher["target"],stress),stop_step=12)
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
    receipt["nonfinite_isolation"] = verify_failure_isolation(
        model, config, teacher, condition, folder, units, separate[0],
    )
    receipt.update(engineering_inventory(folder))
    frozen(folder/"COMPLETE.json",receipt)
    return receipt


def verify_uncapped_guidance(model,config,teacher,condition,folder,units,separate):
    """One large finite update is observed, never followed by a weaker retry."""
    stress=[copy.deepcopy(units[0]),dict(id="uncapped_fixture",slots=[6,7],
        scale=scale_binding([1e6]*4,condition=condition))]
    destination=folder/"uncapped_large_increment"
    factory=functools.partial(CandidateControl,condition=condition,scale_binding=stress[0]["scale"],
        units=stress,physical_rows=20)
    sampler.sample(model,config,teacher,stress[0]["scale"][EFFECTIVE],destination,20,
        control_factory=factory,check_affinity=False,
        initial_override=batch_initial(config,teacher["target"],stress),stop_step=11)
    steps=read(destination/"steps.json")
    assert all("guidance_increment_limited" not in row for row in steps)
    assert max(steps[10]["max_atom_increment_A"][10:])>3.
    with h5py.File(destination/"trace.h5") as mixed,h5py.File(separate/"trace.h5") as reference:
        assert np.array_equal(mixed["guidance_delta_A"][10].astype(np.float64),
                              mixed["unconstrained_guidance_delta_A"][10])
        for key in reference:
            assert np.array_equal(reference[key][:11],mixed[key][:11,:10]),key
    return dict(uncapped_large_increment_preserved=True,unaffected_candidates_exact=True,
        guidance_increment_limiter="none",stress_is_engineering_only=True,
        stress_multiplier_is_not_calibrated=True,stress_steps=11)


def engineering_inventory(folder):
    """Count saved engineering passes without starting work or allocating GPU state."""
    evaluations = full = calls = 0
    for path in Path(folder).rglob("trace.h5"):
        with h5py.File(path) as trace:
            count = trace["endpoint_A"].shape[1]
        calls += 1
        evaluations += count
        if (path.parent / "checkpoint_100.pt.gz").exists():
            full += count
    return dict(engineering_trajectory_calls=calls, engineering_trajectory_evaluations=evaluations,
        engineering_full_100_interval_passes=full, engineering_repeated_passes_are_not_independent=True)
