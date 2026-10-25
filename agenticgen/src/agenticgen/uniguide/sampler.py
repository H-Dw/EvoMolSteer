"""Native FLOWR inference plus a single input VJP at each active step."""
import copy
import time
from pathlib import Path
from types import SimpleNamespace
import h5py
import numpy as np
import torch
from .io import read, write, sha, frozen, value_sha, Budget
from .native_tools import cpu, device, detach, tensor_digest, save_pt, load_pt
from .batching import join, take
from .random_streams import NativeStreams, capture
from .condition import loss, guidance_delta, finite
from .protocol import expand, SEEDS, WINDOWS
from agenticgen.runtime.rng import rng_state, set_rng


def population(config, target, count):
    if count not in [10,20,25,50]: raise ValueError("Unapproved population")
    n=count//5
    states=[load_pt(Path(config["root"])/"initials"/target/f"batch_{b:03d}.pt.gz") for b in range(5)]
    result={k:join([s[k] for s in states for _ in range(n)],[20]*count,list(range(n))*5)
            for k in ["prior","pocket_target","pocket_off","target_com"]}
    result["stream_states"]=[s["rng"] for s in states]
    result["population"]=count
    return result


class Control:
    def __init__(self, model, teacher, initial, schedule, folder, resume=None, stop_step=None):
        self.model,self.teacher,self.initial=model,teacher,initial
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
        self.schedule=expand(schedule)
        self.com=initial["target_com"].to("cuda")
        self.resume=resume
        self.start_step=resume["step"] if resume else 0
        self.stop_step=stop_step
        self.random_streams=NativeStreams(resume["stream_states"] if resume else initial["stream_states"],1,initial["population"]//5)
        self.noise_samples,self.categories,self.rows=[],[],[]
        self.trace=h5py.File(self.folder/"trace.h5","w")
        self.last_assignment=None
        model.integrator._adaptive_observer=self
        model._adaptive=self

    def initialize(self,curr,cond,times,prior,pt,po,et,it,eo,io,grid):
        context={"prior":prior,"pt":{k:v for k,v in pt.items() if k!="complex"},
                 "po":{k:v for k,v in po.items() if k!="complex"},"com":self.com,
                 "et":et,"it":it,"eo":eo,"io":io,"grid":grid,
                 "coord_scale":float(self.model.coord_scale),"integrator":self.model.integrator.hparams}
        self.context_sha=tensor_digest(context)
        if self.resume:
            if self.resume["context_sha256"]!=self.context_sha: raise ValueError("Resume context mismatch")
            curr,cond,times=(device(self.resume[k]) for k in ["current","cond","times"])
            self.model.integrator._guard_warned=self.resume["guard_warned"]
            self.last_assignment=device(self.resume["previous_assignment"])
        return curr,cond,times,et,it,eo,io

    def checkpoint(self,step,curr,cond,times):
        if step in [10,100] or step==self.stop_step:
            save_pt(self.folder/f"checkpoint_{step:03d}.pt.gz",dict(step=step,current=curr,cond=cond,times=times,
                context_sha256=self.context_sha,stream_states=self.random_streams.states,
                previous_assignment=self.last_assignment,guard_warned=self.model.integrator._guard_warned))
        if step==self.stop_step: raise StopIteration("Engineering short-step endpoint")

    def store(self,key,step,value):
        a=value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
        if key not in self.trace:
            self.trace.create_dataset(key,shape=(100,)+a.shape,dtype=a.dtype,chunks=(1,)+a.shape,compression="gzip",compression_opts=1)
        self.trace[key][step]=a

    def forward(self,curr,pocket,times,cond,equis,invs,step,dt):
        self.noise_samples,self.categories=[],[]
        self.before=curr["coords"].detach()
        self.times,self.dt=times,dt
        entry=tensor_digest(capture())
        scale=self.schedule[step]
        with torch.set_grad_enabled(scale!=0):
            x=self.before.clone().requires_grad_(scale!=0)
            out=self.model(dict(curr,coords=x),pocket,times,cond_batch=detach(cond),pocket_equis=equis,pocket_invs=invs,training=False)
            for value in out.get("affinity",{}).values():
                if torch.is_tensor(value) and value.requires_grad:
                    def denied(g): raise RuntimeError("Affinity head entered coordinate VJP")
                    value.register_hook(denied)
            pred,_=self.model._get_predictions(out)
            endpoint=pred["coords"]*self.model.coord_scale+self.com[:,None]
            value,target,selected,assignment=loss(endpoint,self.teacher["core_A"],curr["mask"].bool())
            gradient=torch.autograd.grad(value.sum(),x)[0].detach() if scale else torch.zeros_like(x)
        finite("loss",value);finite("input gradient",gradient)
        if tensor_digest(capture())!=entry: raise ValueError("Forward/VJP consumed sampling RNG")
        gamma=self.model.integrator.noise_scheduler(times[0][0])
        self.delta,coefficient=guidance_delta(gradient,times[0],dt,[scale]*len(x),gamma,curr["mask"])
        self.velocity=(pred["coords"].detach()-self.before)/(1-times[0][:,None,None])
        q=(times[0][:,None,None]*self.velocity-self.before)/(1-times[0]+1e-6)[:,None,None]
        self.drift=self.velocity+gamma*q
        switches=(assignment!=self.last_assignment).float().mean(1).tolist() if self.last_assignment is not None else [0.]*len(x)
        self.last_assignment=assignment
        self.rows.append(dict(step=step,time=float(times[0][0]),dt=float(dt),scale=scale,
            gamma=float(gamma),coefficient=coefficient.tolist(),loss=value.detach().tolist(),
            assignment_switch_fraction=switches,gradient_norm=gradient.flatten(1).norm(dim=1).tolist(),affinity_vjp=False))
        for key,v in {"endpoint_A":endpoint,"current_A":x*self.model.coord_scale+self.com[:,None],
                      "target_A":target,"input_gradient":gradient,"guidance_delta_A":self.delta*self.model.coord_scale,
                      "assignment":assignment,"selected":selected,"mask":curr["mask"],"native_drift":self.drift}.items():
            self.store(key,step,v)
        return detach(out)

    def after(self,curr,step):
        if len(self.noise_samples)!=1 or len(self.categories)<3: raise ValueError("Native draw structure changed")
        gamma=self.model.integrator.noise_scheduler(self.times[0][0])
        noise=self.noise_samples[0]*torch.sqrt(2*gamma*self.model.integrator.coord_noise_level)*self.dt
        expected=(self.before+self.drift*self.dt+noise)*curr["mask"][...,None]
        native=curr["coords"]
        if not torch.allclose(native,expected,atol=2e-5,rtol=2e-6): raise ValueError("Native drift/noise mismatch")
        result=dict(curr,coords=native+self.delta)
        finite("guided coordinates",result["coords"])
        self.store("native_noise",step,noise)
        self.store("native_proposal",step,native)
        self.store("guided_proposal",step,result["coords"])
        self.rows[-1].update(max_atom_increment_A=(self.delta*self.model.coord_scale).norm(dim=-1).amax(1).tolist(),
            delta_native_ratio=(self.delta.flatten(1).norm(dim=1)/(native-self.before).flatten(1).norm(dim=1).clamp_min(1e-12)).tolist())
        if step%10==9:
            self.trace.flush();write(self.folder/"steps.json",self.rows)
            write(self.folder/"progress.json",{"step_completed":step+1,"unix":time.time()})
        return result

    def close(self):
        self.trace.flush();self.trace.close()
        write(self.folder/"steps.json",self.rows)
        write(self.folder/"rng_ledger.json",self.random_streams.ledger)


def sample(model,config,teacher,schedule,folder,count=50,resume_path=None,stop_step=None):
    initial=population(config,teacher["target"],count)
    pt,po,prior=(device(initial[k]) for k in ["pocket_target","pocket_off","prior"])
    for pocket in [pt,po]: pocket["complex"]=[SimpleNamespace(com=c.numpy()) for c in initial["target_com"]]
    resume=load_pt(resume_path) if resume_path else None
    control=Control(model,teacher,initial,schedule,folder,resume,stop_step)
    try:
        output=model._generate_selective(prior=prior,pocket_data_target=pt,pocket_data_untarget=po,
            steps=100,times=[torch.zeros(count,device="cuda") for _ in range(3)],apply_guidance=False,coord_noise_level=.2)
        save_pt(Path(folder)/"terminal.pt.gz",output)
        return output,pt,po,control
    except StopIteration:
        if stop_step is None: raise
        return None,pt,po,control
    except Exception as error:
        save_pt(Path(folder)/"failure_state.pt.gz",{"before":getattr(control,"before",None),"delta":getattr(control,"delta",None)})
        write(Path(folder)/"FAILED.json",{"error":str(error),"type":type(error).__name__,"no_weaker_retry":True})
        raise
    finally:
        control.close()


def summarize(rows,teacher,run_id,schedule):
    cs=np.array([r["condition_rmsd_A"] for r in rows])
    scores=np.array([r["pic50_on_rescore"] for r in rows])
    return dict(id=run_id,schedule=schedule,attempts=len(rows),complete=bool(np.isfinite(cs).all() and np.isfinite(scores).all()),
        C=float(cs.mean()),C_se=float(cs.std(ddof=1)/np.sqrt(len(cs))),
        mean_pic50=float(scores.mean()),gap=float(abs(scores.mean()-teacher["pic50"])),best=float(scores.max()),
        built=sum(r["build_success"] for r in rows),unique_smiles=len({r["smiles"] for r in rows if r["smiles"]}))


def diagnostics(folder):
    """Small measured feedback for the agents; retain full vectors in HDF5."""
    folder=Path(folder);steps=read(folder/"steps.json");windows=[]
    with h5py.File(folder/"trace.h5","r") as trace:
        for lo,hi in WINDOWS:
            rows=[r for r in steps if lo<=r["step"]<=hi]
            delta=trace["guidance_delta_A"][lo:hi+1]
            windows.append(dict(steps=[lo,hi],increment_vector_rms_A=float(np.sqrt(np.mean(delta*delta))),
                gradient_norm_mean=float(np.mean([r["gradient_norm"] for r in rows])),
                native_increment_ratio_mean=float(np.mean([r["delta_native_ratio"] for r in rows])),
                max_atom_increment_A=max(max(r["max_atom_increment_A"]) for r in rows),
                assignment_switch_mean=float(np.mean([r["assignment_switch_fraction"] for r in rows])),
                first_endpoint_loss_mean=float(np.mean(rows[0]["loss"])),last_endpoint_loss_mean=float(np.mean(rows[-1]["loss"])),
                coefficient_first=rows[0]["coefficient"][0],coefficient_last=rows[-1]["coefficient"][0]))
    return windows


def execute(model,config,teacher,schedule,folder,stage,count=50,resume_path=None):
    from agenticgen.runtime.scoring import final_metrics
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    run_id=str(folder.relative_to(Path(config["root"])))
    manifest=dict(id=run_id,teacher_sha256=value_sha(teacher),schedule=schedule,count=count,
                  resume_sha256=sha(resume_path) if resume_path else None,config_sha256=value_sha(config))
    frozen(folder/"manifest.json",manifest)
    if (folder/"COMPLETE.json").exists():
        receipt=read(folder/"COMPLETE.json")
        if receipt["records_sha256"]!=sha(folder/"records.json"): raise ValueError("Corrupt results")
        return read(folder/"aggregate.json")
    if (folder/"FAILED.json").exists() or (folder/"STARTED.json").exists():
        raise RuntimeError("Previous failed/interrupted attempt requires explicit engineering audit; no automatic refill")
    Budget(config["root"]).reserve(run_id,stage,count,value_sha(manifest))
    write(folder/"STARTED.json",{"unix":time.time()})
    output,pt,po,control=sample(model,config,teacher,schedule,folder,count,resume_path)
    scoring=SimpleNamespace(pt=pt,po=po,arm="ug",batch=0,seed=None,roots=np.arange(count),parents=np.arange(count),events=[None]*100)
    rows=final_metrics(model,output,scoring,folder)
    values,_,_,assignment=loss(output["coords"],teacher["core_A"],output["mask"].bool())
    for slot,row in enumerate(rows):
        row.update(condition_rmsd_A=float((2*values[slot]).sqrt()),assignment=assignment[slot].tolist(),
            logical_seed=SEEDS[slot//(count//5)],logical_slot=slot%(count//5),
            coordinate_source="native final corrector prediction returned by sampler; separately preserve state at step 100")
    write(folder/"records.json",rows)
    aggregate=summarize(rows,teacher,run_id,schedule)
    aggregate["window_diagnostics"]=diagnostics(folder)
    write(folder/"aggregate.json",aggregate)
    write(folder/"COMPLETE.json",{"records_sha256":sha(folder/"records.json"),"manifest_sha256":sha(folder/"manifest.json")})
    return aggregate
