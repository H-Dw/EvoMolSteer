"""FLOWR.ROOT CK2/CLK3 controller, promoted from the validated run_lineage.py.

The upstream integration loop is copied at runtime and instrumented, not rewritten.
Main changes are explicit: common-frame off-pocket translation; controlled SMC arm;
historical categorical sampler and raw argmax decode. Both source versions are saved.
EvoMolSteer adds synchronous step commits; no trajectory NPZ is accumulated.
"""
import argparse, copy, csv, gzip, hashlib, inspect, json, os, random, sys, textwrap, time, types
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import QED, Descriptors, rdMolDescriptors
from flowr.gen.generate_from_pdb_selective import get_args, get_dataset
from flowr.gen import utils as util
from flowr.scriptutil import load_model
import flowr.models.fm_pocket as fm
from evomolsteer.storage.streaming import StepTrajectoryWriter
from evomolsteer.storage.transactions import atomic_json
from .instrumentation import instrument_source

RDLogger.DisableLog('rdApp.warning')

def cpu_tree(x):
    if torch.is_tensor(x): return x.detach().cpu()
    if hasattr(x,'items'): return {k:cpu_tree(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)): return [cpu_tree(v) for v in x]
    return x

def nparr(x): return x.detach().cpu().numpy()

def rng_state():
    return {'torch_cpu':torch.get_rng_state(),'torch_cuda':torch.cuda.get_rng_state_all(),
            'numpy':np.random.get_state(),'python':random.getstate()}

def set_rng(x):
    torch.set_rng_state(x['torch_cpu']); torch.cuda.set_rng_state_all(x['torch_cuda'])
    np.random.set_state(x['numpy']); random.setstate(x['python'])

class Trace:
    def __init__(self,path,arm,seed,batch,scale,save=True,steps=100):
        self.path=Path(path); self.path.mkdir(parents=True,exist_ok=True)
        self.arm=arm; self.seed=seed; self.batch=batch; self.scale=scale; self.save=save
        self.arr=defaultdict(list); self.events=[]; self.roots=None; self.parents=None
        self.steps=steps
        self.anchor_steps={min(steps-1,round(t*steps)) for t in [0,.1,.2,.3,.4,.5,.75,.99]}
        self.writer=StepTrajectoryWriter(self.path,expected_steps=steps,
            metadata={'arm':arm,'seed':int(seed),'batch':int(batch),'coord_scale':float(scale)}) if save else None
        self.final_state=None; self.final_cond=None

    def begin(self,prior,pt,po):
        b=prior['coords'].shape[0]; self.roots=np.arange(b); self.parents=np.arange(b)
        self.pt,self.po=pt,po
        self.initial_rng=rng_state()
        if self.save:
            with gzip.open(self.path/'initial_state.pt.gz','wb',compresslevel=3) as f:
                torch.save({'prior':cpu_tree(prior),'rng':self.initial_rng,
                            'pocket_target':cpu_tree({k:v for k,v in pt.items() if k!='complex'}),
                            'pocket_off':cpu_tree({k:v for k,v in po.items() if k!='complex'}),
                            'target_com':np.array([np.asarray(s.com) for s in pt['complex']]),
                            'coord_scale':self.scale},f)

    def before(self,i,curr,cond,times,dt,prior):
        self.i=i; self.t=float(times[0][0]); self.dt=float(dt)
        self.b=curr['coords'].shape[0]; self.selected=np.arange(self.b)
        self.did_resample=False
        self.arr['current_coords'].append(nparr(curr['coords']))
        self.arr['mask'].append(nparr(curr['mask']).astype(np.uint8))
        for k in ['atomics','bonds','charges','hybridization']:
            if k in curr and torch.is_tensor(curr[k]):
                v=curr[k]; hard=v.argmax(-1)
                # Current categorical channels are one-hot; assert this before compacting.
                assert bool(((v==0)|(v==1)).all()), f'non-onehot current {k}'
                self.arr['current_'+k].append(nparr(hard).astype(np.uint8))
        self.arr['score_time'].append(np.array([float(v[0]) for v in times],np.float32))
        self.arr['step_size'].append(np.float32(self.dt))
        self.arr['root_slot'].append(self.roots.copy())
        self.arr['parent_slot'].append(self.parents.copy())
        if i in self.anchor_steps and self.save:
            with gzip.open(self.path/f'restart_step_{i:03d}.pt.gz','wb',compresslevel=3) as f:
                torch.save({'step':i,'times':cpu_tree(times),'current':cpu_tree(curr),
                            'cond':cpu_tree(cond),'prior':cpu_tree(prior), 'rng':rng_state(),
                            'roots':self.roots.copy()},f)

    def score(self,pred,onoff):
        self.a=pred['affinity']['pic50'].squeeze(-1)
        self.bscore=onoff['affinity']['pic50'].squeeze(-1)
        assert torch.isfinite(self.a).all() and torch.isfinite(self.bscore).all()
        self.wa=self.a.softmax(0); self.wb=(-self.bscore).softmax(0)
        self.w = self.wa if self.arm=='single' else (self.wa+self.wb)/2
        if self.arm=='unguided': self.w=torch.ones_like(self.wa)/len(self.wa)
        self.arr['pic50_on'].append(nparr(self.a)); self.arr['pic50_off'].append(nparr(self.bscore))
        self.arr['weight_on'].append(nparr(self.wa)); self.arr['weight_off'].append(nparr(self.wb))
        self.arr['selection_probability'].append(nparr(self.w))
        self.arr['predicted_coords'].append(nparr(pred['coords']))
        for k in ['atomics','bonds','charges','hybridization']:
            if k in pred and torch.is_tensor(pred[k]):
                self.arr['predicted_'+k].append(nparr(pred[k].argmax(-1)).astype(np.uint8))
                if k!='bonds': self.arr['predicted_'+k+'_probs_f16'].append(nparr(pred[k]).astype(np.float16))
        if self.i in self.anchor_steps and self.save:
            with gzip.open(self.path/f'endpoint_step_{self.i:03d}.pt.gz','wb',compresslevel=3) as f:
                torch.save(cpu_tree(pred),f)

    def propose(self,curr):
        self.proposal=curr
        self.arr['proposal_coords'].append(nparr(curr['coords']))
        for k in ['atomics','bonds','charges','hybridization']:
            if k in curr and torch.is_tensor(curr[k]):
                self.arr['proposal_'+k].append(nparr(curr[k].argmax(-1)).astype(np.uint8))

    def choose(self,original,**kw):
        if self.arm=='unguided':
            return tuple(kw[k] for k in ['predicted','prior','current','pocket_data','pocket_equis','pocket_invs','cond_batch'])
        original_multinomial=torch.multinomial
        def selection(weights,num_samples,replacement,**extra):
            actual=weights if self.arm=='joint' else self.wa
            ids=original_multinomial(actual,num_samples,replacement,**extra)
            self.selected=nparr(ids); self.did_resample=True
            return ids
        torch.multinomial=selection
        try: result=original(**kw)
        finally: torch.multinomial=original_multinomial
        assert torch.equal(result[2]['coords'],kw['current']['coords'][self.selected])
        return result

    def end(self,curr,cond,times):
        ids=self.selected; counts=np.bincount(ids,minlength=self.b)
        self.arr['selected_indices'].append(ids.copy()); self.arr['offspring_count'].append(counts)
        self.arr['resampled'].append(np.uint8(self.did_resample))
        self.arr['state_time'].append(np.float32(float(times[0][0])))
        self.events.append({'step':self.i,'score_time':self.t,'state_time':float(times[0][0]),
                            'resampled':self.did_resample,'ess':float(1/self.w.square().sum()),
                            'eliminated':int((counts==0).sum()),'unique_roots_before':int(len(set(self.roots)))})
        self.roots=self.roots[ids]; self.parents=ids.copy()
        self.final_state=curr; self.final_cond=cond
        # This callback is after the resampling decision, so rejected candidates
        # and the complete pre-selection population are included in every commit.
        if self.save:
            if any(len(v)!=1 for v in self.arr.values()):
                raise ValueError('Trace callbacks did not produce exactly one complete step')
            self.writer.append(int(self.i),{k:v[0] for k,v in self.arr.items()})
        self.arr.clear()
        self.proposal=None
        if self.i%10==0 or self.i==self.steps-1:
            print(json.dumps({'event':'trajectory_step_committed','arm':self.arm,'batch':self.batch,
                              'step':int(self.i),'score_time':self.t,'resampled':self.did_resample}),flush=True)

    def finish(self,output):
        self.output=output
        if self.save:
            self.writer.finalize()
            atomic_json(self.path/'events.json',self.events)
            with gzip.open(self.path/'final_prediction.pt.gz','wb',compresslevel=3) as f: torch.save(cpu_tree(output),f)

def instrument(model,path):
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    original_method=model._generate_selective
    source=textwrap.dedent(inspect.getsource(original_method))
    changed=instrument_source(source)
    # No graph/fragment inpainting is used, so capture occurs immediately after time update.
    ns=dict(original_method.__func__.__globals__)
    original_guidance=ns['apply_selective_smc_guidance']
    ns['apply_selective_smc_guidance']=lambda **kw:model._lineage.choose(original_guidance,**kw)
    exec(compile(changed,str(path/'instrumented_generate_selective.py'),'exec'),ns)
    model._generate_selective=types.MethodType(ns['_generate_selective'],model)
    (path/'upstream_generate_selective.py').write_text(source)
    (path/'instrumented_generate_selective.py').write_text(changed)
    return original_method

def make_inputs(model,batch_t,batch_o,device):
    prior,pt,_,_=batch_t; _,po,_,_=batch_o
    lig=model.builder.extract_ligand_from_complex(prior)
    for k in ['interactions','fragment_mask','fragment_mode']: lig[k]=prior[k]
    def pocket(p):
        d=model.builder.extract_pocket_from_complex(p)
        d['interactions']=p['interactions']; d['complex']=p['complex']
        return {k:v.to(device) if torch.is_tensor(v) else v for k,v in d.items()}
    lig={k:v.to(device) if torch.is_tensor(v) else v for k,v in lig.items()}
    target,off=pocket(pt),pocket(po)
    ct=torch.stack([torch.as_tensor(s.com) for s in target['complex']]).reshape(-1,3).to(device)
    co=torch.stack([torch.as_tensor(s.com) for s in off['complex']]).reshape(-1,3).to(device)
    shift=(co-ct)/model.coord_scale
    off['coords']=off['coords']+shift[:,None,:]*off['mask'][...,None]
    return lig,target,off,{'target_com':nparr(ct).tolist(),'off_com':nparr(co).tolist(),'off_internal_shift':nparr(shift).tolist()}

def final_metrics(model,output,trace,path):
    # Preserve upstream scores and supply separately named endpoint-rescore scores.
    # Both rescores see identical decoded categorical state and endpoint coordinates.
    device=trace.pt['coords'].device
    b=output['coords'].shape[0]
    final={k:v.to(device) for k,v in output.items() if torch.is_tensor(v) and k!='affinity'}
    com=torch.stack([torch.as_tensor(s.com) for s in trace.pt['complex']]).reshape(-1,3).to(device)
    final['coords']=(final['coords']-com[:,None,:])/model.coord_scale
    for k in ['atomics','bonds','charges','hybridization']:
        if k in final: final[k]=torch.nn.functional.one_hot(final[k].argmax(-1),final[k].shape[-1]).float()
    times=[torch.full((b,),.9999,device=device) for _ in range(3)]
    with torch.no_grad():
        scores=[]
        for pocket in [trace.pt,trace.po]:
            o=model._predict_affinity(final,final,pocket,times)
            scores.append(nparr(o['affinity']['pic50'].squeeze(-1)))
    mols=model._generate_mols(output)
    assert len(mols)==b
    rows=[]; writer=Chem.SDWriter(str(path/'molecules_all_built.sdf'))
    rawwriter=Chem.SDWriter(str(path/'molecules_raw_decodable.sdf'))
    raw_mols=model._generate_mols(output,sanitise=False)
    for slot,m in enumerate(mols):
        node=f'{trace.arm}_b{trace.batch:03d}_final_{slot:03d}'
        r={'arm':trace.arm,'batch':trace.batch,'seed':trace.seed,'slot':slot,'node_id':node,
           'root_slot':int(trace.roots[slot]),'parent_step':len(trace.events)-1,'parent_slot':int(trace.parents[slot]),
           'pic50_on_upstream':float(output['affinity']['pic50'][slot]),
           'pic50_off_upstream':float(output['affinity']['pic50_untarget'][slot]),
           'pic50_on_rescore':float(scores[0][slot]),'pic50_off_rescore':float(scores[1][slot]),
           'build_success':m is not None,'failure_reason':'','smiles':''}
        r['gap_upstream']=r['pic50_on_upstream']-r['pic50_off_upstream']
        r['gap_rescore']=r['pic50_on_rescore']-r['pic50_off_rescore']
        raw=raw_mols[slot]
        if raw is not None:
            raw.SetProp('node_id',node); rawwriter.write(raw)
        if m is None:
            r['failure_reason']='builder_returned_none'
            if raw is not None:
                try: Chem.SanitizeMol(Chem.Mol(raw))
                except Exception as e: r['failure_reason']=type(e).__name__+': '+str(e)
        else:
            try:
                Chem.SanitizeMol(m); r['smiles']=Chem.MolToSmiles(m)
                r['connected']=len(Chem.GetMolFrags(m))==1
                r['qed']=float(QED.qed(m)); r['mw']=float(Descriptors.MolWt(m)); r['heavy_atoms']=m.GetNumHeavyAtoms()
                r['rings']=int(rdMolDescriptors.CalcNumRings(m))
                xyz=m.GetConformer().GetPositions()
                pc=nparr(trace.pt['coords'][slot][trace.pt['mask'][slot].bool()])*model.coord_scale+nparr(com[slot])
                d=np.linalg.norm(xyz[:,None,:]-pc[None,:,:],axis=-1)
                r['min_protein_distance_A']=float(d.min()); r['pairs_below_1_2A']=int((d<1.2).sum())
            except Exception as e:
                r['failure_reason']=type(e).__name__+': '+str(e)
            for k,v in r.items(): m.SetProp(k,str(v))
            writer.write(m)
        rows.append(r)
    writer.close(); rawwriter.close()
    (path/'final_records.json').write_text(json.dumps(rows,indent=2))
    return rows

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',required=True); p.add_argument('--checkpoint',required=True)
    p.add_argument('--campaign',default='pilot'); p.add_argument('--n',type=int,default=8); p.add_argument('--batch',type=int,default=8)
    p.add_argument('--arms',default='unguided,single,joint'); p.add_argument('--window',type=float,default=.5)
    p.add_argument('--window-start',type=float,default=0.);p.add_argument('--steps',type=int,default=100)
    p.add_argument('--seed',type=int,default=42); p.add_argument('--verify-passive',action='store_true')
    opt=p.parse_args(); root=Path(opt.root); out=root/'results'/opt.campaign; out.mkdir(parents=True,exist_ok=True)
    assert not (out/'COMPLETE.json').exists(),'Do not overwrite completed campaign'
    inp=root/'inputs'; argv=sys.argv
    sys.argv=['lineage','--arch','pocket','--pocket_type','holo','--gpus','1','--num_workers','0',
              '--batch_cost',str(opt.batch),'--sample_n_molecules_per_target',str(opt.n),
              '--ckpt_path',opt.checkpoint,'--save_dir',str(out),'--integration_steps',str(opt.steps),
              '--cut_pocket','--pocket_cutoff','7','--max_sample_iter','0',
              '--no_cat_noise_euler_guard','--no_ligand_valence_repair',
              '--pdb_file_target',str(inp/'3PE1_protein_aligned.pdb'),
              '--ligand_file_target',str(inp/'3PE1_ligand_aligned.sdf'),
              '--pdb_file_untarget',str(inp/'6KHF_protein_aligned.pdb'),
              '--ligand_file_untarget',str(inp/'6KHF_ligand_aligned.sdf')]
    args=get_args(); sys.argv=argv
    args.seed=opt.seed; torch.set_float32_matmul_precision('high')
    loaded=load_model(args); model,hparams,vocab,vc,vh,va,vpa,vpr=loaded
    model=model.to('cuda').eval(); transform,interpolant=util.load_util(args,hparams,vocab,vc,vh,va)
    st,so=util.load_data_from_pdb_selective(args,remove_hs=hparams['remove_hs'],remove_aromaticity=hparams['remove_aromaticity'])
    dst=get_dataset(st,transform,vocab,interpolant,args,hparams); dso=get_dataset(so,transform,vocab,interpolant,args,hparams)
    dlt=util.get_dataloader(args,dst,interpolant); dlo=util.get_dataloader(args,dso,interpolant)
    original=instrument(model,out/'provenance')
    model.integrator.use_sde_simulation=True; model.integrator.coord_noise_level=.2
    config={'experiment':vars(opt),'flowr_args':vars(args),'torch':torch.__version__,'hip':torch.version.hip,
            'device':torch.cuda.get_device_name(0),'coord_scale':model.coord_scale,
            'atom_vocabulary':getattr(vocab,'token_idx_map',str(vocab)),
            'charge_vocabulary':getattr(vc,'token_idx_map',str(vc)),
            'hybridization_vocabulary':getattr(vh,'token_idx_map',str(vh)),
            'full_probability_steps':sorted({min(opt.steps-1,round(t*opt.steps)) for t in [0,.1,.2,.3,.4,.5,.75,.99]}),
            'coordinate_convention':'model coordinates; x_world = x_model * coord_scale + target pocket COM',
            'recording':f'all {opt.steps} pre-integration and proposed states, endpoints, scores; all candidates including no-offspring',
            'trajectory_storage':'per-step verified HDF5; consolidated trajectory.h5; verified stage files retired; no raw trajectory NPZ',
            'selection_mode':'selective SMC resampling on each upstream integration step inside the configured window; no coordinate reward gradient',
            'deviations':'common-frame correction; matched SDE all arms; fixed reference ligand atom count; exact CK2 paper window and batch size unreported; no diversity filter during collection'}
    (out/'config.json').write_text(json.dumps(config,indent=2,default=str))
    allrows=[]; started=time.time()
    for batch,(bt,bo) in enumerate(zip(dlt,dlo)):
        lig,pt,po,frame=make_inputs(model,bt,bo,'cuda')
        shared=rng_state(); seed=opt.seed+batch*100003
        (out/f'frame_batch_{batch:03d}.json').write_text(json.dumps(frame,indent=2))
        for arm in opt.arms.split(','):
            path=out/arm/f'batch_{batch:03d}'
            if (path/'COMPLETE.json').exists():
                allrows.extend(json.loads((path/'final_records.json').read_text())); continue
            torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
            trace=Trace(path,arm,seed,batch,model.coord_scale,steps=opt.steps); model._lineage=trace
            b=lig['coords'].shape[0]
            kwargs=dict(prior=copy.deepcopy(lig),pocket_data_target=copy.deepcopy(pt),pocket_data_untarget=copy.deepcopy(po),
                        steps=opt.steps,times=[torch.zeros(b,device='cuda') for _ in range(3)],
                        apply_guidance=True,guidance_window_start=opt.window_start,guidance_window_end=opt.window,coord_noise_level=.2)
            before=rng_state(); t0=time.time()
            output=model._generate_selective(**kwargs); trace.finish(output)
            if opt.verify_passive and batch==0 and arm=='joint':
                set_rng(before)
                baseline=original(prior=copy.deepcopy(lig),pocket_data_target=copy.deepcopy(pt),pocket_data_untarget=copy.deepcopy(po),
                                  steps=opt.steps,times=[torch.zeros(b,device='cuda') for _ in range(3)],apply_guidance=True,
                                  guidance_window_start=opt.window_start,guidance_window_end=opt.window,coord_noise_level=.2)
                checks={k:bool(torch.equal(output[k],baseline[k])) for k in ['coords','atomics','bonds','charges','mask']}
                (path/'passive_capture_equivalence.json').write_text(json.dumps(checks,indent=2)); assert all(checks.values()),checks
            rows=final_metrics(model,output,trace,path); allrows.extend(rows)
            summary={'arm':arm,'batch':batch,'n':b,'built':sum(r['build_success'] for r in rows),
                     'seconds':time.time()-t0,'resampling_steps':[e['step'] for e in trace.events if e['resampled']],
                     'bytes':sum(f.stat().st_size for f in path.rglob('*') if f.is_file())}
            (path/'COMPLETE.json').write_text(json.dumps(summary,indent=2))
            print(json.dumps(summary),flush=True)
            del output,trace; torch.cuda.empty_cache()
        # Restore pre-sampler RNG so next prior is independent of the number of experiment arms.
        set_rng(shared)
        (out/'progress.json').write_text(json.dumps({'batch_done':batch,'rows':len(allrows),'elapsed':time.time()-started}))
    (out/'final_records.json').write_text(json.dumps(allrows,indent=2))
    result={'status':'complete','records':len(allrows),'seconds':time.time()-started}
    for arm in opt.arms.split(','):
        rows=[r for r in allrows if r['arm']==arm]
        result[arm]={'n':len(rows),'built':sum(r['build_success'] for r in rows),
                     'unique_smiles':len(set(r['smiles'] for r in rows if r['smiles'])),
                     **{k:float(np.mean([r[k] for r in rows])) for k in ['pic50_on_upstream','pic50_off_upstream','gap_upstream','pic50_on_rescore','pic50_off_rescore','gap_rescore']}}
    (out/'COMPLETE.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__': main()
