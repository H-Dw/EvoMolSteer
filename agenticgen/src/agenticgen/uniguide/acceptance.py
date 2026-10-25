"""Remote-only acceptance: numerical VJP, native zero delta and resume equivalence."""
import ast
import copy
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace
import torch
from .io import read,write
from .native_tools import device,cpu,load_pt,tensor_digest
from .prepare import load_model
from .sampler import sample,population,Control
from .pipeline import source_binding


class NativeStop(Exception): pass


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
    root=Path(config["root"]);folder=root/"acceptance"
    folder.mkdir(parents=True,exist_ok=True)
    repo=Path(__file__).resolve().parents[4]
    result=subprocess.run([sys.executable,"-m","pytest",str(repo/"agenticgen/tests/test_uniguide.py"),"-q"],capture_output=True,text=True)
    (folder/"pytest.log").write_text(result.stdout+result.stderr)
    if result.returncode: raise RuntimeError("Remote unit/contract tests failed; inspect acceptance/pytest.log")
    checks={}
    for teacher in read(root/"teachers.json"):
        target=teacher["target"];out=folder/target
        model=load_model(config,teacher)
        checks[target]={"zero_delta":zero_delta_check(model,config,teacher,out/"zero")}
        sample(model,config,teacher,[1.]*4,out/"continuous",50,stop_step=12)
        sample(model,config,teacher,[1.]*4,out/"resumed",50,resume_path=out/"continuous/checkpoint_010.pt.gz",stop_step=12)
        a=load_pt(out/"continuous/checkpoint_012.pt.gz");b=load_pt(out/"resumed/checkpoint_012.pt.gz")
        same={k:tensor_digest(a[k])==tensor_digest(b[k]) for k in ["current","cond","times","stream_states"]}
        if not all(same.values()): raise ValueError("Exact pre-step-10 resume failed")
        if any(p.grad is not None or p.requires_grad for p in model.parameters()): raise ValueError("Model parameters not frozen")
        checks[target]["resume"]=same
        checks[target]["frozen_parameters"]=True
        del model;torch.cuda.empty_cache()
    write(folder/"COMPLETE.json",{"source_binding":source_binding(),"unit_tests":True,"gpu_checks":checks,
        "scientific_zero_guidance_groups":0,"physical_layout":"one profile per call; five seeds, ten logical rows per seed for formal population; twenty-row native random draws",
        "engineering_layout":"five seeds, ten logical rows each; short 3/12-step trajectories"})
