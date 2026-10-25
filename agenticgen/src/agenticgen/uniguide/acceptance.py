"""Remote-only acceptance: numerical VJP, native zero delta and resume equivalence."""
import ast
import copy
import subprocess
import sys
import types
import time
from pathlib import Path
from types import SimpleNamespace
import torch
from .io import read,write
from .native_tools import device,cpu,load_pt,tensor_digest
from .prepare import load_model
from .sampler import sample,population,Control
from .pipeline import source_binding
from .condition import condition_map,coordinate_loss
from .batching import take
from .chunking import chunk_call


class NativeStop(Exception): pass


def real_model_vjp(model,config,teacher,state_path):
    """Directional finite difference through the real frozen endpoint predictor."""
    state=load_pt(state_path);initial=population(config,teacher["target"],50)
    ids=list(range(10))
    curr,cond,times=(device(take(state[k],ids,50)) for k in ["current","cond","times"])
    pocket=device(take(initial["pocket_target"],ids,50));com=initial["target_com"][ids].to("cuda")
    with torch.no_grad():
        equis,invs=model.gen.get_pocket_encoding(pocket["coords"],pocket["atom_names"],
            pocket_atom_charges=pocket["charges"].argmax(-1),pocket_bond_types=pocket["bonds"].argmax(-1),
            pocket_res_types=pocket["res_names"],pocket_atom_mask=pocket["mask"])
    def endpoint(x):
        out=model(dict(curr,coords=x),pocket,times,cond_batch=cond if model.self_condition else None,
                  pocket_equis=equis,pocket_invs=invs,training=False)
        prediction,_=model._get_predictions(out)
        return prediction["coords"]*model.coord_scale+com[:,None]
    with torch.no_grad():
        args=(curr,pocket,times)
        kwargs=dict(cond_batch=cond if model.self_condition else None,pocket_equis=equis,pocket_invs=invs,training=False)
        full=model._ug_original_forward(*args,**kwargs)
        chunks=chunk_call(model._ug_original_forward,args,kwargs,10,rows=5,pad_to=10)
        difference=full["coords"]-chunks["coords"]
        write(Path(state_path).parent/"chunk_comparison.json",{
            "max_abs_native":float(difference.abs().max()),"rms_native":float(difference.square().mean().sqrt()),
            "per_row_max_abs":difference.flatten(1).abs().amax(1).tolist(),
            "full_abs_max":float(full["coords"].abs().max()),"torch_matmul_precision":torch.get_float32_matmul_precision(),
            "logical_partition":5,"physical_rows":10})
        if not torch.allclose(full["coords"],chunks["coords"],atol=2e-5,rtol=2e-5):
            raise ValueError("Physical chunking changed per-row model coordinates beyond numerical tolerance")
    x=curr["coords"].detach().clone().requires_grad_(True)
    predicted=endpoint(x)
    target,selected,_=condition_map(predicted,teacher["core_A"],curr["mask"].bool())
    gradient=torch.autograd.grad(coordinate_loss(predicted,target,selected).sum(),x)[0].detach()
    norm=gradient.norm()
    if not torch.isfinite(norm) or norm<=1e-8: raise ValueError("Real model input Jacobian is zero/nonfinite")
    direction=gradient/norm
    errors=[]
    with torch.no_grad():
        for eps in [.01,.005]:
            plus=coordinate_loss(endpoint(x.detach()+eps*direction),target,selected).sum()
            minus=coordinate_loss(endpoint(x.detach()-eps*direction),target,selected).sum()
            numerical=(plus-minus)/(2*eps)
            errors.append(float(abs(numerical-norm)/norm))
    if max(errors)>.05: raise ValueError("Real FLOWR finite-difference VJP mismatch: "+str(errors))
    return {"relative_errors":errors,"epsilons":[.01,.005],"analytic_directional_derivative":float(norm),
            "fixed_assignment":True,"condition_target_detached":True,"coord_scale":float(model.coord_scale),
            "physical_forward_partition_checked":True}


def zero_delta_check(model,config,teacher,folder):
    """Compare first three steps to the unmodified native loop, same RNG algorithms."""
    _,_,_,_=sample(model,config,teacher,[1.]*4,folder/"wrapped",50,stop_step=3)
    wrapped=load_pt(folder/"wrapped/checkpoint_003.pt.gz")
    initial=population(config,teacher["target"],50)
    observer=Control(model,teacher,initial,[1.]*4,folder/"native")
    prior,pt,po=(device(initial[k]) for k in ["prior","pocket_target","pocket_off"])
    for pocket in [pt,po]: pocket["complex"]=[SimpleNamespace(com=c.numpy()) for c in initial["target_com"]]
    source=(Path(config["root"])/"provenance/native/native_selective.py").read_text()
    tree=ast.parse(source);fn=tree.body[0]
    fn.body=[n for n in fn.body if not (isinstance(n,ast.Assert) and ast.unparse(n.test)=="apply_guidance")]
    loops=[n for n in ast.walk(fn) if isinstance(n,ast.For) and ast.unparse(n.iter)=="enumerate(step_sizes)"]
    if len(loops)!=1: raise ValueError("Ambiguous acceptance native loop")
    loops[0].body[:0]=ast.parse("if i == 3:\n    capture_native(curr, cond_batch, times)\n").body
    saved={}
    def capture_native(curr,cond,times):
        saved.update(current=cpu(curr),cond=cpu(cond),times=cpu(times),stream_states=cpu(observer.random_streams.states))
        raise NativeStop()
    scope=dict(model._generate_selective.__func__.__globals__);scope["capture_native"]=capture_native
    exec(compile(ast.fix_missing_locations(tree),"acceptance_native_short.py","exec"),scope)
    native=types.MethodType(scope["_generate_selective"],model)
    try:
        native(prior=prior,pocket_data_target=pt,pocket_data_untarget=po,steps=100,
               times=[torch.zeros(50,device="cuda") for _ in range(3)],apply_guidance=False,coord_noise_level=.2)
    except NativeStop: pass
    finally: observer.close()
    checks={k:tensor_digest(saved[k])==tensor_digest(wrapped[k]) for k in ["current","cond","times","stream_states"]}
    if not all(checks.values()): raise ValueError("Native zero-delta equivalence failed: "+str(checks))
    return checks


def run(config):
    root=Path(config["root"]);folder=root/"acceptance/runs"/str(time.time_ns())
    folder.mkdir(parents=True,exist_ok=True)
    repo=Path(__file__).resolve().parents[4]
    result=subprocess.run([sys.executable,"-m","pytest",str(repo/"agenticgen/tests/test_uniguide.py"),"-q"],capture_output=True,text=True)
    (folder/"pytest.log").write_text(result.stdout+result.stderr)
    if result.returncode: raise RuntimeError("Remote unit/contract tests failed; inspect "+str(folder/"pytest.log"))
    checks={}
    for teacher in read(root/"teachers.json"):
        if teacher["core_sdf_atoms"]!=teacher["plip_profile"]["core_atoms"]:
            raise ValueError("Teacher condition expanded beyond direct PLIP atoms")
        target=teacher["target"];out=folder/target
        model=load_model(config,teacher)
        checks[target]={"zero_delta":zero_delta_check(model,config,teacher,out/"zero")}
        sample(model,config,teacher,[1.]*4,out/"continuous",50,stop_step=12)
        checks[target]["real_model_vjp"]=real_model_vjp(model,config,teacher,out/"continuous/checkpoint_010.pt.gz")
        sample(model,config,teacher,[1.]*4,out/"resumed",50,resume_path=out/"continuous/checkpoint_010.pt.gz",stop_step=12)
        a=load_pt(out/"continuous/checkpoint_012.pt.gz");b=load_pt(out/"resumed/checkpoint_012.pt.gz")
        same={k:tensor_digest(a[k])==tensor_digest(b[k]) for k in ["current","cond","times","stream_states"]}
        if not all(same.values()): raise ValueError("Exact pre-step-10 resume failed")
        if any(p.grad is not None or p.requires_grad for p in model.parameters()): raise ValueError("Model parameters not frozen")
        checks[target]["resume"]=same
        checks[target]["frozen_parameters"]=True
        del model;torch.cuda.empty_cache()
    receipt={"source_binding":source_binding(),"unit_tests":True,"gpu_checks":checks,"evidence_directory":str(folder),
        "torch_threads":torch.get_num_threads(),"torch_version":torch.__version__,"hip":torch.version.hip,
        "scientific_zero_guidance_groups":0,"physical_layout":"50 logical rows; network forward/VJP/pocket encoding physical rows fixed at 10 (padded outputs discarded); twenty-row native random draws per seed",
        "engineering_layout":"five seeds, ten logical rows each; short 3/12-step trajectories"}
    write(folder/"COMPLETE.json",receipt)
    write(root/"acceptance/COMPLETE.json",receipt)
