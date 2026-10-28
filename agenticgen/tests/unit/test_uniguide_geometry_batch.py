import copy
from pathlib import Path
from types import SimpleNamespace
import h5py
import numpy as np
import pytest
import torch
from agenticgen.uniguide import geometry_batch as batch
from agenticgen.uniguide import geometry_sampling as sampling
from agenticgen.uniguide import geometry_reuse as reuse
from agenticgen.uniguide import geometry_protocol as protocol
from agenticgen.uniguide.geometry_condition import DISTANCE
from agenticgen.uniguide.io import read,write,value_sha,sha
from agenticgen.uniguide.native_tools import save_pt,load_pt,tensor_digest
from test_uniguide_geometry import features,condition,domain,diagnostics


def test_capacity_ladder_and_specific_oom():
    assert [batch.lower_capacity(n) for n in (100,90,80,70,60,50,40,30,20,10)]==[80,80,60,60,40,40,20,20,10,None]
    assert batch.device_oom(torch.cuda.OutOfMemoryError("fixture"))
    assert batch.device_oom(RuntimeError("HIP out of memory"))
    assert not batch.device_oom(FloatingPointError("nonfinite loss"))
    assert not batch.device_oom(RuntimeError("host out of memory"))


def test_mixed_streams_preserve_native_slots_and_external_rng():
    states=[]
    for seed in (1,2,3,4,5):
        generator=torch.Generator().manual_seed(seed)
        states.append(dict(torch_cpu=generator.get_state(),torch_cuda=[]))
    slots=[[0,1],[4,5],[8,9]]
    independent=[batch.SlotStreams(states,s) for s in slots]
    mixed=batch.CandidateStreams([states]*3,slots)
    external=torch.get_rng_state().clone()
    probabilities=torch.arange(1,1+30*4*6,dtype=torch.float32).reshape(30,4,6)
    probabilities/=probabilities.sum(-1,keepdim=True)
    for _ in range(3):
        expected=torch.cat([s.categorical(probabilities[i*10:(i+1)*10]) for i,s in enumerate(independent)])
        assert torch.equal(expected,mixed.categorical(probabilities))
        expected=torch.cat([s.normal("randn",(10,4,3)) for s in independent])
        assert torch.equal(expected,mixed.normal("randn",(30,4,3)))
    assert tensor_digest(mixed.states)==tensor_digest([s.states for s in independent])
    assert torch.equal(external,torch.get_rng_state())


def setup_scheduler(tmp_path,monkeypatch):
    spec=condition.prepare(features());teacher=dict(target="teacher")
    binding=dict(teacher_id="teacher",condition_sha256=spec["condition_sha256"])
    for module in (batch,sampling):monkeypatch.setattr(module,"compatibility",lambda *a:binding)
    monkeypatch.setattr(torch.cuda,"get_device_name",lambda :"test-device")
    monkeypatch.setattr(batch,"capture",lambda :torch.get_rng_state().clone())
    monkeypatch.setattr(batch,"restore",lambda s:torch.set_rng_state(s))
    monkeypatch.setattr(batch,"batch_initial",lambda c,t,u:dict(population=10*len(u)))
    class Model:
        integrator=SimpleNamespace(_guard_warned=False)
        def _generate_mols(self,output,sanitise=True):return [None]*len(output["coords"])
    model=Model();config=dict(root=str(tmp_path));calls=[]
    def sample(model,config,teacher,schedule,folder,count,control_factory,**kwargs):
        units=control_factory.keywords["units"];calls.append(count)
        xyz=torch.tensor(spec["reference_coordinates_angstrom"],dtype=torch.float32)[None].repeat(count,1,1)
        for i,u in enumerate(units):
            for seed in range(5):
                for j,slot in enumerate(u["slots"]):xyz[i*10+seed*2+j]+=slot+seed*10
        output=dict(coords=xyz,mask=torch.ones(count,4,dtype=torch.bool))
        save_pt(folder/"terminal.pt.gz",output)
        with h5py.File(folder/"trace.h5","w") as trace:trace["guidance_delta_A"]=np.zeros((100,count,4,3),dtype=np.float32)
        steps=[dict(step=i,loss=[0.]*count,assignment_switch_fraction=[0.]*count,gradient_norm=[1.]*count,
            coefficient=[1.]*count,max_atom_increment_A=[0.]*count,delta_native_ratio=[0.]*count,
            sampling_rng_after_sha256="fixture",candidate_rng_after_sha256=["fixture"]*len(units)) for i in range(100)]
        parts=[SimpleNamespace(ledger=[]) for u in units]
        control=SimpleNamespace(units=units,rows=steps,random_streams=SimpleNamespace(parts=parts),failed_units={})
        return output,None,None,control
    return config,teacher,spec,model,calls,sample


def test_cross_configuration_batch_oom_accounting_promotion_and_reuse(tmp_path,monkeypatch):
    config,teacher,spec,model,calls,sample=setup_scheduler(tmp_path,monkeypatch)
    attempted=[];external=torch.get_rng_state().clone()
    def oom(*args,**kwargs):
        size=args[5];attempted.append(size)
        if size>60:
            torch.rand(10)
            raise torch.cuda.OutOfMemoryError("injected")
        return sample(*args,**kwargs)
    jobs=[dict(schedule=[i+1.]*4,folder=tmp_path/f"grid{i}",stage="calibration",count=10) for i in range(9)]
    rows=batch.execute_many(model,config,teacher,spec,jobs,sample_fn=oom)
    assert attempted==[90,80,60,30] and calls==[60,30]
    assert torch.equal(external,torch.get_rng_state())
    assert all(r["attempts"]==10 for r in rows)
    assert sum(r["count"] for r in read(tmp_path/"attempts.json").values())==90
    assert len(list((tmp_path/"candidate_batches/teacher").glob("*/OOM.json")))==2
    # Reuse old slots, sample only the remaining forty for this configuration.
    jobs=[dict(schedule=[1.]*4,folder=tmp_path/"full",stage="calibration",count=50)]
    row=batch.execute_many(model,config,teacher,spec,jobs,sample_fn=sample)[0]
    assert calls[-1]==40 and row["attempts"]==50
    assert sum(r["count"] for r in read(tmp_path/"attempts.json").values())==130
    before=list(calls)
    assert batch.execute_many(model,config,teacher,spec,jobs,sample_fn=sample)[0]["id"]==row["id"]
    assert calls==before
    records=read(tmp_path/"full/records.json")
    assert [r["logical_slot"] for r in records]==list(range(10))*5
    assert len({(r["logical_seed"],r["logical_slot"]) for r in records})==50


def test_formal_directions_share_batches_teacher_capacity_is_independent(tmp_path,monkeypatch):
    config,teacher,spec,model,calls,sample=setup_scheduler(tmp_path,monkeypatch)
    binding=batch.compatibility(config,teacher,spec)
    d=domain(digest=spec["condition_sha256"],sampling_hash=value_sha(binding))
    body={k:v for k,v in d.items() if k!="domain_sha256"}
    body["constant_comparison"]=[dict(eligible=True,absolute_displacement=dict(max_atom_increment_A=1.))]
    d=dict(body,domain_sha256=value_sha(body))
    jobs=[dict(schedule=[v,0,0,0],folder=tmp_path/f"direction{v}",stage="formal",domain=d) for v in (-1,0,1)]
    rows=batch.execute_many(model,config,teacher,spec,jobs,sample_fn=sample)
    assert calls==[100,50] and all(r["attempts"]==50 for r in rows)
    assert sum(r["count"] for r in read(tmp_path/"attempts.json").values())==150
    assert not (tmp_path/"candidate_batches/other_teacher/capacity.json").exists()


def test_numerical_failure_is_not_retried_as_oom(tmp_path,monkeypatch):
    config,teacher,spec,model,calls,sample=setup_scheduler(tmp_path,monkeypatch)
    def fail(*args,**kwargs):
        calls.append(args[5]);raise FloatingPointError("invalid field")
    job=dict(schedule=[1.]*4,folder=tmp_path/"grid",stage="calibration",count=10)
    with pytest.raises(FloatingPointError):batch.execute_many(model,config,teacher,spec,[job],sample_fn=fail)
    with pytest.raises(RuntimeError,match="explicit audit"):batch.execute_many(model,config,teacher,spec,[job],sample_fn=fail)
    assert calls==[10]
    assert sum(r["count"] for r in read(tmp_path/"attempts.json").values())==10


def test_quarantined_nonfinite_unit_does_not_change_healthy_rows(tmp_path):
    import threading
    control=object.__new__(batch.CandidateControl)
    control.failure_lock=threading.RLock()
    control.initial=dict(population=20);control.failed_units={};control.rows=[];control.folder=tmp_path
    value=torch.ones(20,3,requires_grad=True)
    broken=value.clone();broken[13,0]=float("nan")
    safe=control.check_rows("fixture",broken)
    assert torch.equal(safe[:10],value[:10]) and torch.equal(safe[10:],torch.zeros(10,3))
    assert list(control.failed_units)==[1] and (tmp_path/"nonfinite_unit_001.pt.gz").exists()
    gradient=torch.autograd.grad(safe.sum(),value)[0]
    assert torch.equal(gradient[:10],torch.ones(10,3)) and torch.equal(gradient[10:],torch.zeros(10,3))


def test_initial_selection_is_explicit_bounded_and_frozen(tmp_path):
    d=domain();write(tmp_path/"calibration/domains.json",dict(teacher=d))
    selected=reuse.select_initial(dict(root=str(tmp_path)),dict(teacher=[-1,0,1,0]))
    assert selected["teachers"]["teacher"][protocol.EFFECTIVE]==[1.,2.,4.,2.]
    assert reuse.verify_initial(tmp_path,dict(teacher=d))==selected["teachers"]
    with pytest.raises(ValueError):reuse.select_initial(dict(root=str(tmp_path)),dict(teacher=[.5,0,0,0]))
    with pytest.raises(ValueError):reuse.select_initial(dict(root=str(tmp_path)),dict(teacher=[0,0,0,0]))
    with pytest.raises(ValueError):reuse.select_initial(dict(root=str(tmp_path)),dict(teacher=[10,0,0,0]))


def test_failed_terminal_preserves_structural_masks_and_healthy_units(tmp_path):
    import threading
    control=object.__new__(batch.CandidateControl)
    control.initial=dict(population=20);control.failed_units={};control.rows=[];control.folder=tmp_path
    control.failure_lock=threading.RLock();control.geometry_condition=condition.prepare(features())
    xyz=torch.tensor(control.geometry_condition["reference_coordinates_angstrom"])[None].repeat(20,1,1)
    original=xyz.clone();xyz[13,0,0]=float("nan");mask=torch.ones(20,4,dtype=torch.bool)
    output=control.check_terminal(dict(coords=xyz,mask=mask))
    assert torch.equal(output["mask"],mask) and torch.equal(output["coords"][:10],original[:10])
    assert list(control.failed_units)==[1] and torch.isfinite(output["coords"]).all()


def test_reuse_rejects_incomplete_calibration_before_copy(tmp_path,monkeypatch):
    from agenticgen.uniguide import geometry_runtime as runtime
    root=tmp_path/"dest";source=tmp_path/"source"
    monkeypatch.setattr(runtime,"require_acceptance",lambda c:None)
    monkeypatch.setattr(runtime,"calibrated",lambda c:[dict(teacher=dict(target="teacher"),condition={})])
    with pytest.raises(ValueError,match="incomplete"):
        reuse.reuse(dict(root=str(root)),dict(teacher=str(source)),dict(teacher=[0]*4))
    assert not root.exists()


def test_reuse_copies_verified_complete_and_failed_sources_without_new_attempts(tmp_path,monkeypatch):
    from agenticgen.uniguide import geometry_runtime as runtime
    source=tmp_path/"source";root=tmp_path/"destination"
    binding=dict(teacher_id="teacher",condition_sha256="condition")
    d=domain(sampling_hash=value_sha(binding))
    subject=dict(teacher=dict(target="teacher"),condition=dict(condition_sha256="condition"))
    monkeypatch.setattr(runtime,"require_acceptance",lambda c:None)
    monkeypatch.setattr(runtime,"calibrated",lambda c:[subject])
    monkeypatch.setattr(reuse,"compatibility",lambda *a:binding)
    write(source/"calibration/domains.json",dict(teacher=d))
    write(source/"baseline/aggregate.json",dict(id="baseline",attempts=50))
    write(source/"baseline/COMPLETE.json",dict(files={"aggregate.json":sha(source/"baseline/aggregate.json")}))
    write(source/"bad/FAILED.json",dict(error="nonfinite",no_weaker_retry=True))
    for name in ("baseline","bad"):
        write(source/"measurement_registry"/(name+".json"),dict(source_group=name,identity=dict(compatibility=binding)))
    write(source/"calibration/frozen"/d["calibration_key"]/"domain.json",d)
    write(source/"calibration/frozen"/d["calibration_key"]/"COMPLETE.json",dict(domain_sha256=d["domain_sha256"],
        measurement_receipts={"baseline":sha(source/"baseline/COMPLETE.json")}))
    write(source/"attempts.json",dict(original=60));before=reuse.tree_hashes(source)
    result=reuse.reuse(dict(root=str(root)),dict(teacher=str(source)),dict(teacher=[0,1,0,-1]))
    assert result["teachers"]["teacher"][protocol.MULTIPLIER]==2.
    assert read(root/"bad/FAILED.json")["error"]=="nonfinite"
    assert not (root/"attempts.json").exists() and before==reuse.tree_hashes(source)
    assert reuse.reuse(dict(root=str(root)),dict(teacher=str(source)),dict(teacher=[0,1,0,-1]))==result
    write(root/"baseline/aggregate.json",dict(tampered=True))
    with pytest.raises(ValueError,match="overwrite"):
        reuse.reuse(dict(root=str(root)),dict(teacher=str(source)),dict(teacher=[0,1,0,-1]))
