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
from .chunking import ROWS,concatenate
from .random_streams import NativeStreams, capture
from .condition import loss, guidance_delta, finite
from .protocol import expand, SEEDS, WINDOWS
from agenticgen.runtime.rng import rng_state, set_rng


def population(config, target, count):
    if count not in [10,20,25,50]: raise ValueError("Unapproved population")
    n=count//5
    states=[load_pt(Path(config.get("initials_root",config["root"]))/"initials"/target/f"batch_{b:03d}.pt.gz") for b in range(5)]
    result={k:join([s[k] for s in states for _ in range(n)],[20]*count,list(range(n))*5)
            for k in ["prior","pocket_target","pocket_off","target_com"]}
    result["stream_states"]=[s["rng"] for s in states]
    result["population"]=count
    return result


class Control:
    def __init__(self, model, teacher, initial, schedule, folder, resume=None, stop_step=None):
        self.model,self.teacher,self.initial=model,teacher,initial
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
        self.schedule=self.expand_schedule(schedule)
        self.com=initial["target_com"].to("cuda")
        self.resume=resume
        self.start_step=resume["step"] if resume else 0
        self.stop_step=stop_step
        self.random_streams=self.make_streams(resume["stream_states"] if resume else initial["stream_states"])
        self.noise_samples,self.categories,self.rows=[],[],[]
        self.trace=h5py.File(self.folder/"trace.h5","w")
        self.last_assignment=None
        model.integrator._adaptive_observer=self
        model._adaptive=self

    def expand_schedule(self,schedule):
        return expand(schedule)

    def make_streams(self,states):
        return NativeStreams(states,1,self.initial["population"]//5)

    def step_scales(self,step,count):
        return [self.schedule[step]]*count

    def physical_rows(self):
        return ROWS

    def execute_blocks(self,function,starts):
        return [function(start) for start in starts]

    def condition_loss_for_rows(self,endpoint,valid,ids):
        return self.condition_loss(endpoint,valid)

    def check_rows(self,name,value,ids=None):
        finite(name,value)
        return value

    def prepare_output(self,out,ids):
        return out

    def check_terminal(self,output):
        finite("native final corrector coordinates",output["coords"])
        return output

    def condition_loss(self,endpoint,valid):
        return loss(endpoint,self.teacher["core_A"],valid)

    def condition_trace(self,target):
        return {"target_A":target}

    def checkpoint_metadata(self):
        return {}

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
                previous_assignment=self.last_assignment,guard_warned=self.model.integrator._guard_warned,
                **self.checkpoint_metadata()))
        if step==self.stop_step: raise StopIteration("Engineering short-step endpoint")

    def store(self,key,step,value):
        a=value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
        if key not in self.trace:
            self.trace.create_dataset(key,shape=(100,)+a.shape,dtype=a.dtype,chunks=(1,)+a.shape,compression="gzip",compression_opts=1)
        self.trace[key][step]=a

    def forward(self,curr,pocket,times,cond,equis,invs,step,dt):
        self.active_step=step
        self.noise_samples,self.categories=[],[]
        self.before=curr["coords"].detach()
        self.times,self.dt=times,dt
        entry=tensor_digest(capture())
        scale=self.schedule[step]
        count=len(self.before)
        scales=self.step_scales(step,count);physical=self.physical_rows()
        def block(start):
            ids=list(range(start,min(start+physical,count)))
            actual=len(ids)
            # Keep the input VJP at the same physical shape as the forward.
            # Padding only inside model.forward leaves a five-row input VJP
            # for a 25-row population and changes backward reduction layout.
            physical_ids=(ids*((physical+actual-1)//actual))[:physical]
            current=take(curr,physical_ids,count)
            active=any(scales[i]!=0 for i in physical_ids)
            with torch.set_grad_enabled(active):
                x=current["coords"].detach().clone().requires_grad_(active)
                out=self.model(dict(current,coords=x),take(pocket,physical_ids,count),take(times,physical_ids,count),
                    cond_batch=detach(take(cond,physical_ids,count)),pocket_equis=take(equis,physical_ids,count),pocket_invs=take(invs,physical_ids,count),training=False)
                out=self.prepare_output(out,physical_ids)
                for head in out.get("affinity",{}).values():
                    if torch.is_tensor(head) and head.requires_grad:
                        def denied(g): raise RuntimeError("Affinity head entered coordinate VJP")
                        head.register_hook(denied)
                head=None
                pred,_=self.model._get_predictions(out)
                for key in ["coords","atomics","charges","bonds","hybridization"]:
                    if key in pred:self.check_rows("native prediction "+key,pred[key],physical_ids)
                endpoint=pred["coords"]*self.model.coord_scale+self.com[physical_ids,None]
                endpoint=self.check_rows("predicted endpoint",endpoint,physical_ids)
                value,target,selected,assignment=self.condition_loss_for_rows(endpoint,current["mask"].bool(),physical_ids)
                value=self.check_rows("loss",value,physical_ids)
                gradient=torch.autograd.grad(value.sum(),x)[0] if active else torch.zeros_like(x)
                gradient=self.check_rows("input gradient",gradient,physical_ids)
            return take(detach(dict(out=out,pred=pred,endpoint=endpoint,value=value,target=target,
                                      selected=selected,assignment=assignment,gradient=gradient)),list(range(actual)),physical),actual
        parts=self.execute_blocks(block,range(0,count,physical));blocks,sizes=zip(*parts)
        combined=concatenate(blocks,sizes)
        out,pred,endpoint,value,target,selected,assignment,gradient=(combined[k] for k in
            ["out","pred","endpoint","value","target","selected","assignment","gradient"])
        value=self.check_rows("loss",value);gradient=self.check_rows("input gradient",gradient)
        if tensor_digest(capture())!=entry: raise ValueError("Forward/VJP consumed sampling RNG")
        gamma=self.model.integrator.noise_scheduler(times[0][0])
        self.delta,coefficient=guidance_delta(gradient,times[0],dt,scales,gamma,curr["mask"])
        self.delta=self.check_rows("guidance increment",self.delta)
        self.velocity=(pred["coords"].detach()-self.before)/(1-times[0][:,None,None])
        q=(times[0][:,None,None]*self.velocity-self.before)/(1-times[0]+1e-6)[:,None,None]
        self.drift=self.velocity+gamma*q
        switches=(assignment!=self.last_assignment).float().mean(1).tolist() if self.last_assignment is not None else [0.]*count
        self.last_assignment=assignment
        self.rows.append(dict(step=step,time=float(times[0][0]),dt=float(dt),scale=scale,
            gamma=float(gamma),coefficient=coefficient.tolist(),loss=value.detach().tolist(),
            assignment_switch_fraction=switches,gradient_norm=gradient.flatten(1).norm(dim=1).tolist(),affinity_vjp=False))
        for key,v in {"endpoint_A":endpoint,"current_A":self.before*self.model.coord_scale+self.com[:,None],
                      "input_gradient":gradient,"guidance_delta_A":self.delta*self.model.coord_scale,
                      "assignment":assignment,"selected":selected,"mask":curr["mask"],"native_drift":self.drift,
                      **self.condition_trace(target)}.items():
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
        result["coords"]=self.check_rows("guided coordinates",result["coords"])
        self.store("native_noise",step,noise)
        self.store("native_proposal",step,native)
        self.store("guided_proposal",step,result["coords"])
        self.rows[-1].update(max_atom_increment_A=(self.delta*self.model.coord_scale).norm(dim=-1).amax(1).tolist(),
            sampling_rng_after_sha256=tensor_digest(self.random_streams.states),
            delta_native_ratio=(self.delta.flatten(1).norm(dim=1)/(native-self.before).flatten(1).norm(dim=1).clamp_min(1e-12)).tolist())
        if step%10==9:
            self.trace.flush();write(self.folder/"steps.json",self.rows)
            write(self.folder/"progress.json",{"step_completed":step+1,"unix":time.time()})
        return result

    def close(self):
        self.trace.flush();self.trace.close()
        write(self.folder/"steps.json",self.rows)
        write(self.folder/"rng_ledger.json",self.random_streams.ledger)


def sample(model,config,teacher,schedule,folder,count=50,resume_path=None,stop_step=None,control_factory=None,check_affinity=True,initial_override=None):
    initial=population(config,teacher["target"],count) if initial_override is None else initial_override
    if initial["population"]!=count:raise ValueError("Initial population size differs")
    pt,po,prior=(device(initial[k]) for k in ["pocket_target","pocket_off","prior"])
    for pocket in [pt,po]: pocket["complex"]=[SimpleNamespace(com=c.numpy()) for c in initial["target_com"]]
    resume=load_pt(resume_path) if resume_path else None
    control=(control_factory or Control)(model,teacher,initial,schedule,folder,resume,stop_step)
    try:
        output=model._generate_selective(prior=prior,pocket_data_target=pt,pocket_data_untarget=po,
            steps=100,times=[torch.zeros(count,device="cuda") for _ in range(3)],apply_guidance=False,coord_noise_level=.2)
        output=control.check_terminal(output)
        for key,value in output.get("affinity",{}).items():
            if check_affinity and torch.is_tensor(value): finite("terminal affinity "+key,value)
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
    if not all(np.isfinite(r["pic50_on_rescore"]) for r in rows):
        write(folder/"FAILED.json",{"type":"nonfinite_terminal_rescore","no_weaker_retry":True})
        raise FloatingPointError("Nonfinite terminal rescore")
    values,_,_,assignment=loss(output["coords"],teacher["core_A"],output["mask"].bool())
    renamed={}
    for slot,row in enumerate(rows):
        old_id=row["node_id"]
        batch,index=divmod(slot,count//5)
        row.update(condition_rmsd_A=float((2*values[slot]).sqrt()),assignment=assignment[slot].tolist(),
            node_id=f"ug_b{batch:03d}_final_{index:03d}",batch=batch,slot=index,root_slot=index,parent_slot=index,
            seed=SEEDS[batch],logical_seed=SEEDS[batch],logical_slot=index,tensor_row=slot,
            coordinate_source="native final corrector prediction returned by sampler; separately preserve state at step 100")
        renamed[old_id]=row
    # The generic scorer emits flat row IDs; publish the actual seed/slot identity
    # consistently in both structures and records without changing coordinates.
    from rdkit import Chem
    for name in ["molecules_all_built.sdf","molecules_raw_decodable.sdf"]:
        path=folder/name
        molecules=([m for m in Chem.SDMolSupplier(str(path),sanitize=False,removeHs=False) if m is not None]
                   if path.stat().st_size else [])
        with Chem.SDWriter(str(path)) as writer:
            for mol in molecules:
                row=renamed[mol.GetProp("node_id")]
                for key in ["node_id","batch","slot","root_slot","parent_slot","seed","logical_seed","logical_slot","tensor_row"]:
                    mol.SetProp(key,str(row[key]))
                writer.write(mol)
    write(folder/"final_records.json",rows)
    write(folder/"records.json",rows)
    aggregate=summarize(rows,teacher,run_id,schedule)
    aggregate["window_diagnostics"]=diagnostics(folder)
    ledger={str(r["step"]):r["sampling_rng_after_sha256"] for r in read(folder/"steps.json")}
    ref=Path(config["root"])/"random_stream_reference"/(teacher["target"]+".json")
    if ref.exists():
        reference=read(ref)
        if any(reference.get(step)!=digest for step,digest in ledger.items()):
            write(folder/"FAILED.json",{"type":"probability_dependent_native_rng_consumption"})
            raise RuntimeError("Native RNG consumption changed across hypotheses or population layout")
    elif len(ledger)==100:
        frozen(ref,ledger)
    else:
        raise ValueError("A complete native RNG reference must precede resumed formal runs")
    write(folder/"aggregate.json",aggregate)
    write(folder/"COMPLETE.json",{"records_sha256":sha(folder/"records.json"),"manifest_sha256":sha(folder/"manifest.json")})
    return aggregate
