"""Scientific and orchestration contracts for the isolated geometry protocol."""
import copy
import json
import math
from pathlib import Path
import pytest
import torch
import jsonschema
from agenticgen.uniguide import geometry_condition as condition
from agenticgen.uniguide import geometry_protocol as protocol
from agenticgen.uniguide import geometry_runtime as runtime
from agenticgen.uniguide import geometry_sampling as sampling
from agenticgen.uniguide import geometry_council as council
from agenticgen.uniguide.io import write,read,value_sha,sha


def features():
    return dict(atoms=[dict(id=k,element="C",formal_charge=0) for k in "abcd"],
        bonds=[dict(atom_ids=[a,b],order="single") for a,b in zip("abc","bcd")],
        controlled_atom_ids=["a","b"],coordinates_angstrom=[[0,0,0],[1.5,0,0],[2,1.4,0],[3,1.4,1]])


def domain(multiplier=2.,teacher="teacher",digest="condition",sampling_hash="sampling"):
    body=dict(policy=protocol.POLICY,teacher_id=teacher,condition_sha256=digest,
        sampling_protocol_sha256=sampling_hash,calibration_key=value_sha([teacher,digest,sampling_hash]),
        calibrated_guidance_multiplier=multiplier,normalized_minimum=-3.,normalized_maximum=3.,initial_source="baseline")
    return dict(body,domain_sha256=value_sha(body))


def row(key="baseline",schedule=None,error=1.):
    return dict(id=key,complete=True,attempts=50,**{protocol.NORMALIZED:schedule or [0.]*4,
        protocol.EFFECTIVE:[2.]*4,condition.DISTANCE:error,condition.CONTROL_DISTANCE:error,
        condition.NEIGHBOR_DISTANCE:error,condition.MAX_ERROR:error})


def diagnostics(value=1.):
    return [dict(steps=list(w),increment_vector_rms_A=value,max_atom_increment_A=value,
        gradient_norm_mean=1.,native_increment_ratio_mean=.1) for w in protocol.WINDOWS]


def test_internal_pairs_default_neighborhood_and_fixed_mapping():
    spec=condition.prepare(features())
    assert [p["atom_ids"] for p in spec["pairs"]]==[["a","b"],["a","c"],["b","c"],["b","d"]]
    assert spec["atom_to_generated_slot"]==dict(a=0,b=1,c=2,d=3)
    assert spec["chemistry_locked"] is False and not spec["coordinates_replaced"]
    f=features();f["neighborhood_atom_ids"]={"a":["d"],"b":[]}
    assert [p["atom_ids"] for p in condition.prepare(f)["pairs"]]==[["a","b"],["a","d"]]


def test_default_mapping_uses_valid_slots_with_holes_across_all_initial_rows(tmp_path):
    from agenticgen.uniguide.native_tools import save_pt
    for i in range(5):
        mask=torch.ones(20,7,dtype=torch.bool);mask[:,1]=False;mask[:,5]=False
        if i==4:mask[9,4]=False
        save_pt(tmp_path/"initials/teacher"/f"batch_{i:03d}.pt.gz",dict(prior=dict(mask=mask)))
    slots=runtime.common_valid_slots(tmp_path,"teacher")
    assert slots==[0,2,3,6]
    spec=condition.prepare(features(),slots)
    assert spec["atom_to_generated_slot"]==dict(a=0,b=2,c=3,d=6)
    x=torch.zeros(1,7,3,dtype=torch.float64)
    x[:,slots]=torch.tensor(spec["reference_coordinates_angstrom"],dtype=x.dtype)
    valid=torch.zeros(1,7,dtype=torch.bool);valid[:,slots]=True
    assert condition.loss(x,spec,valid)[0].item()<1e-25
    with pytest.raises(ValueError,match="Insufficient"):condition.prepare(features(),[0,2,3])
    f=features();f["atom_to_generated_slot"]=dict(a=0,b=1,c=2,d=3)
    with pytest.raises(ValueError,match="unavailable"):condition.prepare(f,slots)


def test_rdkit_embedding_reproducible_without_original_coordinates():
    f=features();del f["coordinates_angstrom"]
    first,second=condition.prepare(f),condition.prepare(f)
    assert first==second and first["conformer_preparation"]["source"]=="ETKDGv3"
    assert first["conformer_preparation"]["random_seed"]==42


def test_rigid_motion_and_mirror_invariance_and_zero_reference_loss():
    spec=condition.prepare(features());x=torch.tensor(spec["reference_coordinates_angstrom"],dtype=torch.float64)[None]
    valid=torch.ones(x.shape[:2],dtype=torch.bool)
    assert condition.loss(x,spec,valid)[0].item()<1e-25
    rotation=torch.tensor([[0.,1,0],[-1,0,0],[0,0,1]],dtype=x.dtype)
    for changed in [x@rotation+12,x*torch.tensor([-1.,1,1])]:
        assert torch.allclose(condition.loss(changed,spec,valid)[0],torch.zeros(1,dtype=x.dtype),atol=1e-25)
    shifted=features();shifted["coordinates_angstrom"]=(x[0]@rotation+40).tolist()
    second=condition.prepare(shifted)
    assert torch.allclose(condition.loss(x,second,valid)[0],torch.zeros(1,dtype=x.dtype),atol=1e-25)


def test_distance_gradient_both_atoms_and_finite_difference():
    f=dict(atoms=[dict(id="a",element="C"),dict(id="b",element="O")],
        bonds=[dict(atom_ids=["a","b"],order="single")],controlled_atom_ids=["a","b"],
        coordinates_angstrom=[[0,0,0],[1,0,0]])
    spec=condition.prepare(f);x=torch.tensor([[[0.,0,0],[2.,0,0],[8,8,8]]],dtype=torch.float64,requires_grad=True)
    valid=torch.tensor([[True,True,False]])
    assert torch.autograd.gradcheck(lambda v:condition.loss(v,spec,valid)[0],(x,))
    gradient=torch.autograd.grad(condition.loss(x,spec,valid)[0].sum(),x)[0]
    assert torch.equal(gradient[0,0],-gradient[0,1]) and gradient[0,0,0]==-1
    assert torch.equal(gradient[0,2],torch.zeros(3,dtype=x.dtype))
    with torch.no_grad():x[0,1]=x[0,0]
    assert torch.isfinite(torch.autograd.grad(condition.loss(x,spec,valid)[0].sum(),x)[0]).all()


@pytest.mark.parametrize("mutation",["duplicate_atom","duplicate_bond","self_bond","bad_mapping","bad_neighbor","empty_pairs","nan"])
def test_invalid_conditions_rejected(mutation):
    f=features()
    if mutation=="duplicate_atom":f["atoms"][1]["id"]="a"
    elif mutation=="duplicate_bond":f["bonds"].append(f["bonds"][0])
    elif mutation=="self_bond":f["bonds"][0]["atom_ids"]=["a","a"]
    elif mutation=="bad_mapping":f["atom_to_generated_slot"]=dict(a=0,b=0,c=2,d=3)
    elif mutation=="bad_neighbor":f["neighborhood_atom_ids"]={"a":["missing"],"b":[]}
    elif mutation=="empty_pairs":f["controlled_atom_ids"]=["a"];f["neighborhood_atom_ids"]={"a":[]}
    else:f["coordinates_angstrom"][0][0]=float("nan")
    with pytest.raises((ValueError,KeyError)):condition.prepare(f)


def test_condition_corruption_and_masked_mapping_rejected():
    spec=condition.prepare(features());x=torch.randn(1,4,3);valid=torch.ones(1,4,dtype=torch.bool)
    valid[0,1]=False
    with pytest.raises(ValueError,match="masked"):condition.loss(x,spec,valid)
    spec["pairs"][0]["reference_distance_angstrom"]+=1
    with pytest.raises(ValueError,match="binding"):condition.validate(spec)


def test_invalid_frozen_pairs_even_if_hash_is_recomputed():
    for mutation in ("duplicate","self","category","mapping"):
        spec=condition.prepare(features())
        if mutation=="duplicate":spec["pairs"].append(spec["pairs"][0])
        elif mutation=="self":spec["pairs"][0]["atom_ids"]=["a","a"]
        elif mutation=="category":spec["pairs"][0]["category"]="controlled_neighborhood"
        else:spec["pairs"][0]["generated_slots"]=[3,2]
        spec["condition_sha256"]=value_sha({k:v for k,v in spec.items() if k!="condition_sha256"})
        with pytest.raises(ValueError):condition.validate(spec)


def test_embedding_failure_never_falls_back_to_two_dimensions(monkeypatch):
    f=features();del f["coordinates_angstrom"]
    monkeypatch.setattr(condition.AllChem,"EmbedMolecule",lambda *a:-1)
    with pytest.raises(ValueError,match="embedding failed"):condition.prepare(f)


def test_frame_and_reflection_diagnostics_do_not_change_distance_loss():
    spec=condition.prepare(features());xyz=torch.tensor(spec["reference_coordinates_angstrom"],dtype=torch.float64)[None]
    changed=xyz*torch.tensor([-1.,1.,1.])+12
    diagnostic=sampling.pose_diagnostics(changed,spec)[0]
    assert diagnostic["unconstrained_alignment_prefers_reflection"]
    assert diagnostic["mapped_centroid_translation_angstrom"]>1
    assert condition.loss(changed,spec,torch.ones(1,4,dtype=torch.bool))[0].item()<1e-25
    assert diagnostic["stereochemistry_assessment"].startswith("not_evaluated")


@pytest.mark.parametrize("schedule",[[0,0,0],[-11,0,0,0],[10.5,0,0,0],[.5,0,0,0],[float("nan"),0,0,0],[True,0,0,0]])
def test_invalid_normalized_schedule(schedule):
    with pytest.raises(ValueError):protocol.effective(schedule,2.)


def test_normalized_scale_zero_negative_boundaries_and_explicit_disabled():
    assert protocol.effective([-10,-1,0,10],2.)==[2./1024,1.,2.,2048.]
    assert protocol.effective([0]*4,2.,False)==[0.]*4
    assert protocol.expand_effective([2.]*4)[9]==0 and protocol.expand_effective([2.]*4)[10]==2
    with pytest.raises(ValueError):protocol.validate_domain(domain(),teacher_id="other")
    with pytest.raises(ValueError):protocol.validate_domain(domain(),condition_hash="other")
    with pytest.raises(ValueError):protocol.effective([10.]*4,1e308)


def test_executor_rejects_old_strength_interface_and_wrong_teacher_before_claim(tmp_path):
    spec=condition.prepare(features());config=dict(root=str(tmp_path))
    with pytest.raises(ValueError,match="pre-experiments"):
        sampling.execute(None,config,dict(target="teacher"),spec,[2.]*4,tmp_path/"bad","formal")
    with pytest.raises(ValueError,match="Another teacher"):
        sampling.execute(None,config,dict(target="teacher"),spec,[0.]*4,tmp_path/"wrong","formal",domain=domain(teacher="other"))
    assert not (tmp_path/"attempts.json").exists()


def test_resume_rejects_changed_mapping_before_native_model_access():
    control=object.__new__(sampling.GeometryControl)
    control.geometry_condition=condition.prepare(features());control.scale_binding=sampling.scale_binding([0.]*4,1.)
    control.sampling_policy={}
    control.resume=dict(geometry_binding=dict(condition_sha256="different"))
    with pytest.raises(ValueError,match="identical distance condition"):
        control.initialize()


def test_actions_use_integer_changes_and_executor_owned_schedules():
    anchors={"baseline":row()};d=domain();seen=[[0.]*4]
    action=dict(anchor_id="baseline",branch="decrease",updates=[dict(window_id=0,step=-1)])
    assert protocol.validate_action(action,anchors,d,seen)==[-1,0,0,0]
    resolved=protocol.resolve_action(action,anchors,d,seen)
    assert resolved[protocol.NORMALIZED]==[-1,0,0,0] and resolved[protocol.EFFECTIVE]==[1,2,2,2]
    assert resolved[protocol.MULTIPLIER]==2 and resolved["calibration_key"]==d["calibration_key"]
    with pytest.raises(ValueError):protocol.validate_action(dict(action,branch="increase"),anchors,d,seen)
    with pytest.raises(ValueError):protocol.validate_action(action,anchors,d,seen+[resolved[protocol.NORMALIZED]])
    with pytest.raises(ValueError):protocol.validate_action(dict(action,**{protocol.NORMALIZED:[-1,0,0,0]}),anchors,d,seen)
    for step in (-.5,1.,True,0,3):
        with pytest.raises(ValueError):protocol.validate_action(dict(action,updates=[dict(window_id=0,step=step)]),anchors,d,seen)
    with pytest.raises(ValueError):protocol.validate_action(dict(action,updates=action["updates"]*2),anchors,d,seen)
    with pytest.raises(ValueError):protocol.validate_action(dict(anchor_id="baseline",branch="increase",mode="fine",schedule=[2.]*4),anchors,d,seen)


def test_teacher_specific_calibration_uses_geometry_not_affinity_and_excludes_growth():
    rows=[]
    for teacher,errors in [("left",[8.,4.,1.,2.,3.,.01]),("right",[8.,4.,2.,1.,3.,.01])]:
        for index,(scale,error) in enumerate(zip([.5,1.,2.,4.,8.,16.],errors)):
            rows.append(dict(id=teacher+str(index),teacher_id=teacher,phase="full",condition_sha256="cond",
                compatibility_sha256="sampling",complete=True,attempts=50,window_diagnostics=diagnostics(10000 if scale==16 else scale),
                **{protocol.EFFECTIVE:[scale]*4,condition.DISTANCE:error},mean_pic50=100-error))
    left=protocol.derive_domain(rows,"left","cond","sampling");right=protocol.derive_domain(rows,"right","cond","sampling")
    assert left[protocol.MULTIPLIER]==2 and right[protocol.MULTIPLIER]==4
    assert left["risk_boundary"] and not left["stability_guarantee"]
    wrong=copy.deepcopy(rows);wrong[0]["condition_sha256"]="wrong"
    with pytest.raises(ValueError):protocol.derive_domain(wrong,"left","cond","sampling")
    with pytest.raises(ValueError):protocol.derive_domain(rows,"left","cond","other-sampler")


def test_registry_reuses_complete_fingerprint_without_more_budget_and_blocks_failure(tmp_path):
    identity={"effective_guidance_scale_schedule":[2.]*4,"population":50}
    fingerprint,cached=sampling.reserve_group(tmp_path,"groups/a","formal",50,identity)
    assert cached is None
    with pytest.raises(RuntimeError):sampling.reserve_group(tmp_path,"groups/b","formal",50,identity)
    group=tmp_path/"groups/a";write(group/"aggregate.json",row("groups/a"))
    write(group/"COMPLETE.json",dict(files={"aggregate.json":sha(group/"aggregate.json")}))
    again,cached=sampling.reserve_group(tmp_path,"groups/b","formal",50,identity)
    assert again==fingerprint and cached["id"]=="groups/a"
    assert sum(x["count"] for x in read(tmp_path/"attempts.json").values())==50
    write(group/"aggregate.json",{})
    with pytest.raises(ValueError):sampling.reserve_group(tmp_path,"groups/c","formal",50,identity)


def test_all_history_is_preserved_and_incompatible_conditions_are_rejected(tmp_path,monkeypatch):
    spec=condition.prepare(features());teacher=dict(target="teacher")
    binding=dict(teacher_id="teacher",condition_sha256=spec["condition_sha256"])
    d=domain(digest=spec["condition_sha256"],sampling_hash=value_sha(binding))
    monkeypatch.setattr(runtime,"compatibility",lambda *a:binding)
    def stored(key,scale=2.,count=50,failed=False):
        identity=dict(compatibility=copy.deepcopy(binding),**{protocol.EFFECTIVE:[scale]*4},population=count)
        fp=value_sha(identity);group=tmp_path/key
        write(tmp_path/"measurement_registry"/(fp+".json"),dict(source_group=key,identity=identity))
        if failed:write(group/"FAILED.json",dict(type="FloatingPointError",error="nonfinite state",attempts=count))
        else:
            r=dict(row(key),attempts=count,fingerprint=fp,window_diagnostics=diagnostics(),**{protocol.EFFECTIVE:[scale]*4})
            write(group/"aggregate.json",r);write(group/"COMPLETE.json",dict(files={"aggregate.json":sha(group/"aggregate.json")}))
    stored("baseline");stored("smaller",count=10);stored("nonlattice",scale=3.);stored("failed",scale=8.,failed=True)
    observed,evidence=runtime.history(dict(root=str(tmp_path)),teacher,spec,d)
    assert [r["id"] for r in observed]==["baseline"] and len(evidence)==3
    assert {r.get("evidence_status") for r in evidence}=={None,"smaller_population_not_directly_rankable","not_on_current_normalized_lattice"}
    binding["condition_sha256"]="different condition"
    with pytest.raises(ValueError,match="sampling protocol"):
        runtime.history(dict(root=str(tmp_path)),teacher,spec,d)


def test_nonfinite_and_finite_large_displacement_remain_distinct():
    spec=condition.prepare(features());xyz=torch.full((1,4,3),float("nan"))
    with pytest.raises(FloatingPointError):condition.loss(xyz,spec,torch.ones(1,4,dtype=torch.bool))
    assert protocol.displacement(dict(window_diagnostics=diagnostics(1e12)))["max_atom_increment_A"]==1e12
    with pytest.raises(ValueError):protocol.displacement(dict(window_diagnostics=diagnostics(float("inf"))))


def test_build_failures_keep_geometry_and_unknown_chemistry_without_refill(tmp_path):
    spec=condition.prepare(features());xyz=torch.tensor(spec["reference_coordinates_angstrom"])[None].repeat(10,1,1)
    class BrokenBuilder:
        def _generate_mols(self,output,sanitise=True):return [None]*len(output["coords"])
    rows=sampling.terminal_records(BrokenBuilder(),dict(coords=xyz,mask=torch.ones(10,4,dtype=torch.bool)),spec,tmp_path,10)
    assert len(rows)==10 and all(not r["build_success"] for r in rows)
    assert all(r[condition.DISTANCE]<1e-6 for r in rows)
    assert all(r["failure_reason"]=="native_builder_returned_none" for r in rows)
    assert all(r["geometry_diagnostics_not_used_for_ranking"]["chemical_identity_assessment"].startswith("not_enforced") for r in rows)


def test_fixed_json_roles_normalized_fields_full_names_and_claim(tmp_path):
    skills=tmp_path/"skills";skills.mkdir()
    for name,source in runtime.skill_assets().items():(skills/name).write_bytes(source.read_bytes())
    history=[row()];d=domain();calls=[]
    def provider(messages,schema,out,response_validator,**kwargs):
        context=json.loads(messages[1]["content"]);branch=context.get("branch");calls.append(context)
        if branch:
            options=context["feasible_examples"]
            answer=dict(analysis="measured internal distances",mechanism_uncertainty="unknown neighborhood effect",proposals=[
                dict(p,id=str(i),anchor_reason="measured anchor",evidence="baseline",prediction="lower distance deviation",falsifier="higher distance deviation")
                for i,p in enumerate(options)])
        else:
            answer=dict(analysis="history",choice_evidence="existing candidates")
            if not context["no_future_execution"]:answer["selected"]={b:b+":0" for b in protocol.BRANCHES}
        jsonschema.validate(answer,schema);response_validator(answer);write(out,answer);return answer
    proposals=council.advise({"llm":{}},history,d,tmp_path/"call",skills,provider=provider)
    assert len(proposals)==3 and all("updates" in p and protocol.NORMALIZED not in p and "schedule" not in p for p in proposals)
    assert "pic50" not in json.dumps(calls) and "coordinate root mean square deviation" not in (skills/"analysis.md").read_text()
    assert council.advise({"llm":{}},history,d,tmp_path/"final",skills,final=True,provider=provider)==[]
    def failure(*args,**kwargs):raise RuntimeError("exhausted output")
    with pytest.raises(RuntimeError):council.advise({"llm":{}},history,d,tmp_path/"fail",skills,provider=failure)
    with pytest.raises(FileExistsError):council.advise({"llm":{}},history,d,tmp_path/"fail",skills,provider=failure)


@pytest.mark.parametrize("bad",['{"a":1,"a":2}','{"a":NaN}','```json\n{}\n```','{}{}'])
def test_strict_output(bad):
    with pytest.raises(json.JSONDecodeError):council.parse_output(bad)


@pytest.mark.parametrize("rounds",[1,2,7])
def test_nondefault_rounds_and_final_analysis_are_external(tmp_path,monkeypatch,rounds):
    from agenticgen.uniguide import geometry_calibration as calibration
    config=dict(protocol=protocol.POLICY,root=str(tmp_path/"new"),generation_context_root=str(tmp_path/"source"),rounds=rounds,
        output_policy="raw999",maximum_single_atom_guidance_increment_angstrom=3.)
    root=Path(config["root"]);d=domain();subject=dict(teacher=dict(target="teacher"),condition=dict(condition_sha256="condition"))
    write(root/"calibration/domains.json",dict(teacher=d))
    monkeypatch.setattr(runtime,"require_acceptance",lambda c:None)
    monkeypatch.setattr(runtime,"calibrated",lambda c:[subject])
    monkeypatch.setattr(runtime,"history",lambda *a:([row()],[]))
    monkeypatch.setattr(runtime,"compatibility",lambda *a:{})
    monkeypatch.setattr(calibration,"verify_frozen",lambda *a:None)
    calls=[]
    def advisor(config,history,domain,folder,skills,failures,final=False):
        calls.append((str(folder),final))
        return [] if final else [protocol.legal_options(history,domain,b)[0] for b in protocol.BRANCHES]
    def executor(model,config,teacher,condition,schedule,folder,stage,domain):
        return row(folder.relative_to(root).as_posix())
    result=runtime.run_version(config,"GD01",model_loader=lambda *a:object(),executor=executor,advisor=advisor,retainer=lambda *a:None)
    assert len(result["current"]["groups"])==rounds*3
    assert len(calls)==rounds+1 and calls[-1][1] and "analysis_final" in calls[-1][0]
    assert (root/"versions/GD01/AWAITING_GPT_REVIEW.json").exists()


def test_staged_calibration_is_teacher_local_frozen_and_never_reloads_for_skill_changes(tmp_path,monkeypatch):
    from agenticgen.uniguide import geometry_calibration as calibration
    from agenticgen.uniguide.io import Budget
    subjects=[dict(teacher=dict(target=t),condition=dict(condition_sha256=t+"condition")) for t in ("left","right")]
    monkeypatch.setattr(calibration,"compatibility",lambda c,t,g:dict(teacher_id=t["target"],condition_sha256=g["condition_sha256"],sampler=c.get("sampler","native")))
    sampled=set();loads=[]
    def executor(model,config,teacher,condition,schedule,folder,stage,count):
        strength=schedule[0];target=teacher["target"];group=folder.relative_to(tmp_path).as_posix()
        new=[(target,strength,seed,slot) for seed in range(5) for slot in range(count//5)
            if (target,strength,seed,slot) not in sampled]
        sampled.update(new)
        if new:Budget(tmp_path).reserve(group,stage,len(new),value_sha(new))
        optimum=1. if target=="left" else 4.
        result=dict(id=group,teacher_id=target,condition_sha256=condition["condition_sha256"],
            compatibility_sha256=value_sha(calibration.compatibility(config,teacher,condition)),complete=True,
            attempts=count,window_diagnostics=diagnostics(strength),**{protocol.EFFECTIVE:schedule,
            condition_module_distance:math.log2(strength/optimum)**2+.1})
        write(folder/"aggregate.json",result);write(folder/"COMPLETE.json",dict(files={"aggregate.json":sha(folder/"aggregate.json")}))
        return result
    condition_module_distance=condition.DISTANCE
    config=dict(root=str(tmp_path))
    domains=calibration.run(config,subjects,lambda c,t:loads.append(t["target"]),executor)
    assert len(sampled)==700 and loads==["left","right"]
    assert domains["left"][protocol.MULTIPLIER]==1 and domains["right"][protocol.MULTIPLIER]==4
    assert sum(x["count"] for x in read(tmp_path/"attempts.json").values())==700
    def forbidden(*a):raise AssertionError("Frozen calibration must not load a model or sample")
    assert calibration.run(dict(config,rounds=99,llm=dict(irrelevant="changed")),subjects,forbidden,forbidden)==domains
    with pytest.raises(ValueError,match="sampling protocol"):
        calibration.verify_frozen(tmp_path,domains["left"],dict(teacher_id="left",condition_sha256="leftcondition",sampler="raw999"))
    with pytest.raises(ValueError,match="Frozen calibration evidence"):
        receipt=tmp_path/"calibration/frozen"/domains["left"]["calibration_key"]/"COMPLETE.json"
        source=next(iter(read(receipt)["measurement_receipts"]))
        write(tmp_path/source/"COMPLETE.json",{})
        calibration.run(config,subjects,forbidden,forbidden)


def test_calibration_boundary_stops_without_expansion(tmp_path,monkeypatch):
    from agenticgen.uniguide import geometry_calibration as calibration
    rows=[dict(row(str(i),error=1./scale),window_diagnostics=diagnostics(scale),**{protocol.EFFECTIVE:[scale]*4})
        for i,scale in enumerate([.25,1.,4.,16.])]
    with pytest.raises(ValueError,match="boundary"):calibration.refinement(rows)
    for r in rows:r.update(teacher_id="teacher",condition_sha256="condition",compatibility_sha256="sampling",phase="full")
    with pytest.raises(ValueError):protocol.derive_domain(rows,"teacher","condition","sampling")


def test_real_population_promotion_uses_only_unmeasured_slots_and_preserves_build_failure(tmp_path,monkeypatch):
    import numpy as np
    import h5py
    from agenticgen.uniguide import geometry_shards as shards
    from agenticgen.uniguide.native_tools import save_pt,load_pt
    spec=condition.prepare(features());teacher=dict(target="teacher");binding=dict(teacher_id="teacher",condition_sha256=spec["condition_sha256"])
    monkeypatch.setattr(sampling,"compatibility",lambda *args:binding)
    monkeypatch.setattr(shards,"initial_shard",lambda c,t,slots:dict(logical_slots=slots,population=10))
    calls=[]
    class Builder:
        def _generate_mols(self,output,sanitise=True):return [None]*len(output["coords"])
    def sample(model,config,teacher,schedule,folder,count,initial_override,**kwargs):
        slots=initial_override["logical_slots"];calls.append(slots)
        xyz=torch.tensor(spec["reference_coordinates_angstrom"])[None].repeat(10,1,1)
        for seed in range(5):
            for offset,slot in enumerate(slots):xyz[seed*2+offset]+=slot+seed*10
        output=dict(coords=xyz,mask=torch.ones(10,4,dtype=torch.bool))
        save_pt(folder/"terminal.pt.gz",output)
        with h5py.File(folder/"trace.h5","w") as trace:trace["guidance_delta_A"]=np.zeros((100,10,4,3),dtype=np.float32)
        steps=[dict(step=i,loss=[0.]*10,assignment_switch_fraction=[0.]*10,gradient_norm=[1.]*10,
            coefficient=[1.]*10,max_atom_increment_A=[0.]*10,delta_native_ratio=[0.]*10,sampling_rng_after_sha256="fixture") for i in range(100)]
        write(folder/"steps.json",steps);write(folder/"rng_ledger.json",dict(slots=slots))
        return output,None,None,None
    monkeypatch.setattr(sampling.sampler,"sample",sample)
    config=dict(root=str(tmp_path));model=Builder()
    ten=sampling.execute(model,config,teacher,spec,[2.]*4,tmp_path/"ten","calibration",10)
    twenty=sampling.execute(model,config,teacher,spec,[2.]*4,tmp_path/"twenty","calibration",20)
    fifty=sampling.execute(model,config,teacher,spec,[2.]*4,tmp_path/"fifty","calibration",50)
    assert calls==[[0,1],[2,3],[4,5],[6,7],[8,9]]
    assert fifty["built"]==0 and fifty["attempts"]==50 and fifty["population_view"]
    records=read(tmp_path/"fifty/records.json")
    assert len({(r["logical_seed"],r["logical_slot"]) for r in records})==50
    assert [r["logical_slot"] for r in records]==list(range(10))*5
    assert torch.equal(load_pt(tmp_path/"ten/terminal.pt.gz")["coords"],load_pt(tmp_path/"fifty/terminal.pt.gz")["coords"][[s*10+i for s in range(5) for i in (0,1)]])
    assert sum(x["count"] for x in read(tmp_path/"attempts.json").values())==50
    d=domain(digest=spec["condition_sha256"],sampling_hash=value_sha(binding))
    assert sampling.execute(model,config,teacher,spec,[0]*4,tmp_path/"formal","formal",domain=d)["id"]==fifty["id"]
    assert len(calls)==5 and read(tmp_path/"formal/REUSED.json")["new_attempts"]==0


def test_sampling_binding_ignores_agent_and_rounds_but_tracks_sampling(tmp_path):
    from agenticgen.uniguide.native_tools import save_pt
    for i in range(5):save_pt(tmp_path/"initials/teacher"/f"batch_{i:03d}.pt.gz",dict(test=i))
    spec=condition.prepare(features());teacher=dict(target="teacher")
    config=dict(root=str(tmp_path),checkpoint_sha256="model",native_hashes={},output_policy="raw999",
        maximum_single_atom_guidance_increment_angstrom=3.)
    original=sampling.compatibility(config,teacher,spec)
    assert original==sampling.compatibility(dict(config,rounds=99,llm=dict(model="other")),teacher,spec)
    for field,value in [("maximum_single_atom_guidance_increment_angstrom",2.),("checkpoint_sha256","other"),("noise_level",.1)]:
        assert original!=sampling.compatibility(dict(config,**{field:value}),teacher,spec)
    for field,value in [("steps",200),("output_policy","native_final_corrector")]:
        with pytest.raises(ValueError):sampling.compatibility(dict(config,**{field:value}),teacher,spec)


def test_fractional_integer_looking_steps_are_rejected_by_application():
    # JSON Schema permits 1.0 as an integer; the execution contract deliberately does not.
    d=domain();a=dict(anchor_id="baseline",branch="increase",updates=[dict(window_id=0,step=1.0)])
    with pytest.raises(ValueError,match="integers"):protocol.resolve_action(a,{"baseline":row()},d,[])
