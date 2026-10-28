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


def domain(multiplier=2.,teacher="teacher",digest="condition"):
    body=dict(policy=protocol.POLICY,teacher_id=teacher,condition_sha256=digest,
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


@pytest.mark.parametrize("schedule",[[0,0,0],[-11,0,0,0],[10.5,0,0,0],[.1,0,0,0],[float("nan"),0,0,0],[True,0,0,0]])
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
    control.resume=dict(geometry_binding=dict(condition_sha256="different"))
    with pytest.raises(ValueError,match="identical distance condition"):
        control.initialize()


def test_actions_use_additive_half_steps_not_old_positive_lattice():
    anchors={"baseline":row()};d=domain();seen=[[0.]*4]
    action=dict(anchor_id="baseline",branch="decrease",mode="fine",**{protocol.NORMALIZED:[-.5,0,0,0]})
    assert protocol.validate_action(action,anchors,d,seen)==[-.5,0,0,0]
    with pytest.raises(ValueError):protocol.validate_action(dict(action,branch="increase"),anchors,d,seen)
    with pytest.raises(ValueError):protocol.validate_action(action,anchors,d,seen+[action[protocol.NORMALIZED]])
    with pytest.raises(KeyError):protocol.validate_action(dict(anchor_id="baseline",branch="increase",mode="fine",schedule=[2.]*4),anchors,d,seen)


def test_teacher_specific_calibration_uses_geometry_not_affinity_and_excludes_growth():
    rows=[]
    for teacher,errors in [("left",[4.,1.,2.,.01]),("right",[4.,2.,1.,.01])]:
        for index,(scale,error) in enumerate(zip([1.,2.,4.,8.],errors)):
            rows.append(dict(id=teacher+str(index),teacher_id=teacher,phase="full",condition_sha256="cond",
                complete=True,attempts=50,window_diagnostics=diagnostics(10000 if scale==8 else scale),
                **{protocol.EFFECTIVE:[scale]*4,condition.DISTANCE:error},mean_pic50=100-error))
    left=protocol.derive_domain(rows,"left","cond");right=protocol.derive_domain(rows,"right","cond")
    assert left[protocol.MULTIPLIER]==2 and right[protocol.MULTIPLIER]==4
    assert left["risk_boundary"] and not left["stability_guarantee"]
    wrong=copy.deepcopy(rows);wrong[0]["condition_sha256"]="wrong"
    with pytest.raises(ValueError):protocol.derive_domain(wrong,"left","cond")


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
    spec=condition.prepare(features());teacher=dict(target="teacher");d=domain(digest=spec["condition_sha256"])
    binding=dict(teacher_id="teacher",condition_sha256=spec["condition_sha256"])
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
    with pytest.raises(ValueError,match="Incompatible measurement"):
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
    assert len(proposals)==3 and all(protocol.NORMALIZED in p and "schedule" not in p for p in proposals)
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
    config=dict(protocol=protocol.POLICY,root=str(tmp_path/"new"),generation_context_root=str(tmp_path/"source"),rounds=rounds)
    root=Path(config["root"]);d=domain();subject=dict(teacher=dict(target="teacher"),condition=dict(condition_sha256="condition"))
    write(root/"calibration/domains.json",dict(teacher=d))
    monkeypatch.setattr(runtime,"require_acceptance",lambda c:None)
    monkeypatch.setattr(runtime,"calibrated",lambda c:[subject])
    monkeypatch.setattr(runtime,"history",lambda *a:([row()],[]))
    calls=[]
    def advisor(config,history,domain,folder,skills,failures,final=False):
        calls.append((str(folder),final))
        return [] if final else [dict(branch=b,**{protocol.NORMALIZED:[0.]*4}) for b in protocol.BRANCHES]
    def executor(model,config,teacher,condition,schedule,folder,stage,domain):
        return row(folder.relative_to(root).as_posix())
    result=runtime.run_version(config,"GD01",model_loader=lambda *a:object(),executor=executor,advisor=advisor,retainer=lambda *a:None)
    assert len(result["current"]["groups"])==rounds*3
    assert len(calls)==rounds+1 and calls[-1][1] and "analysis_final" in calls[-1][0]
    assert (root/"versions/GD01/AWAITING_GPT_REVIEW.json").exists()
