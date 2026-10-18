"""One GPU model per condition, fixed B50 and durable whole-batch results."""
import argparse
import copy
import gzip
import hashlib
import importlib
import inspect
import os
from pathlib import Path
import sys
import textwrap
import time
import traceback
import types

import numpy as np
import torch

from evomolsteer.io import digest,read_json
from evomolsteer.storage.transactions import atomic_json
from .common import validate,condition_program,groups_for,batch_path,binding,read_batch,fail_batch
from .native import InitialPool,HistoricalObserver,final_equivalence,instrument_replay
from .tape import ReplayRandom


def load_model(config,work):
    sys.path.insert(0,config['flowr_root'])
    from flowr.gen.generate_from_pdb_selective import get_args
    from flowr.scriptutil import load_model as load
    argv=sys.argv
    sys.argv=['path-replay','--arch','pocket','--pocket_type','holo','--gpus','1','--num_workers','0',
              '--batch_cost','50','--sample_n_molecules_per_target','50','--ckpt_path',config['checkpoint'],
              '--save_dir',str(work),'--integration_steps','100','--no_cat_noise_euler_guard','--no_ligand_valence_repair']
    try: args=get_args()
    finally: sys.argv=argv
    args.seed=config['master_seed'];torch.set_float32_matmul_precision('high')
    model=load(args)[0].to('cuda').eval();model.requires_grad_(False)
    if getattr(model,'sc_recycle_steps',1)!=1 or model.inpainting_mode or model.graph_inpainting or model.integrator.use_cosine_scheduler:
        raise ValueError('Historical native model configuration is unsupported')
    for k,v in config['native_options'].items():
        if k!='final_corr_pred': setattr(model.integrator,k,v)
    return model


def attach(model):
    source=textwrap.dedent(inspect.getsource(model._generate));namespace=dict(model._generate.__func__.__globals__)
    def forbidden(*a,**kw): raise RuntimeError('SMC is forbidden in path-replayed inference')
    namespace.update(apply_smc_guidance=forbidden,apply_selective_smc_guidance=forbidden)
    changed=instrument_replay(source);exec(compile(changed,'historical_path_native.py','exec'),namespace)
    model._generate=types.MethodType(namespace['_generate'],model)
    from evomolsteer.final_guidance.native import instrument_coord_telemetry
    source_coord=textwrap.dedent(inspect.getsource(model.integrator.coord_step))
    ns=dict(model.integrator.coord_step.__func__.__globals__)
    exec(compile(instrument_coord_telemetry(source_coord),'historical_path_coord.py','exec'),ns)
    model.integrator.coord_step=types.MethodType(ns['coord_step'],model.integrator)
    return {'native_function_sha256':hashlib.sha256(source.encode()).hexdigest(),
            'instrumented_function_sha256':hashlib.sha256(changed.encode()).hexdigest()}


def make_control(model,condition,program,teacher,com):
    from evomolsteer.final_guidance.native import Control
    cls=Control
    if condition['kind']=='legacy':
        from evomolsteer.elite_alignment.control import LegacyControl
        cls=LegacyControl
    elif program is not None:
        choices={'local_nc_layout':('evomolsteer.nc_guidance.reward','NCControl'),
                 'local_motif_v2':('evomolsteer.local_motif.reward','MotifControl'),
                 'ancestral_modes_v3':('evomolsteer.ancestral_modes.reward','AncestorControl')}
        module,symbol=choices[program['family']];cls=getattr(importlib.import_module(module),symbol)
    return cls(model,copy.deepcopy(program),copy.deepcopy(teacher),com,.5)


def score(model,output,pocket,com):
    final={k:v.to('cuda') for k,v in output.items() if torch.is_tensor(v) and k!='affinity'}
    final['coords']=(final['coords']-com[:,None])/model.coord_scale
    for k in ('atomics','bonds','charges'):
        final[k]=torch.nn.functional.one_hot(final[k].argmax(-1),final[k].shape[-1]).float()
    with torch.no_grad():
        scores=model._predict_affinity(final,final,pocket,[torch.full((50,),.9999,device='cuda')]*3)['affinity']['pic50'].flatten().cpu().numpy()
    if scores.shape!=(50,) or not np.isfinite(scores).all(): raise ValueError('Invalid final head scores')
    return scores


def run(repo,config_path,work,condition_id,only_group=None):
    repo=Path(repo).resolve();work=Path(work).resolve();cfg=read_json(config_path);validate(cfg)
    c=next(c for c in cfg['conditions'] if c['id']==condition_id)
    if c.get('alias_of'): raise ValueError('Aliased condition must not launch another GPU worker')
    def heartbeat(**kw): atomic_json(work/'worker_state.json',{'condition':c['id'],'pid':os.getpid(),'unix':time.time(),**kw})
    heartbeat(phase='loading');program,teacher=condition_program(repo,c)
    model=load_model(cfg,work);source=attach(model);pool=InitialPool(cfg)
    atomic_json(work/'methods'/c['id']/'source_audit.json',{
        **source,'condition':c,'protocol_sha256':cfg['protocol_sha256'],'flowr_commit':cfg['flowr_commit'],
        'batch_size':50,'torch':torch.__version__,'no_smc':True,'affinity_head_gradient':False,
        'initial_inputs':'exact saved prior / pocket / COM; no new priors',
        'random_inputs':'per-step original pre-selection ancestor-slot primitive draws',
        'code_commit':os.environ.get('EVOMOLSTEER_RUNTIME_COMMIT')})
    from evomolsteer.elite_alignment.worker import make_posebusters
    from evomolsteer.agentic_steer.native import molecule_quality,slice_tree
    from evomolsteer.local_motif.quality import binary_checks
    from rdkit import Chem,RDLogger
    RDLogger.DisableLog('rdApp.warning')
    pb,pb_error=make_posebusters();by_id={a['id']:a for a in cfg['attempts']}
    for group in groups_for(cfg,c):
        if only_group is not None and group['index']!=only_group: continue
        if read_batch(work,cfg,c,group) is not None: continue
        folder=batch_path(work,c,group).parent;folder.mkdir(parents=True,exist_ok=True)
        rows=[by_id[k] for k in group['ids']];started=time.time()
        heartbeat(phase='inference',group=group['index'],step=-1)
        try:
            prior,pocket,com,scale=pool.load(rows)
            if abs(scale-model.coord_scale)>1e-12: raise ValueError('Saved initial/model coordinate scale mismatch')
            replay=ReplayRandom(cfg,work,rows);model._path_random=replay
            audit_cfg=dict(cfg,_coord_scale=scale)
            observer=HistoricalObserver(audit_cfg,work,rows) if program is None else None
            model._path_observer=observer if observer is not None else lambda *args:None
            control=make_control(model,c,program,teacher,com)
            forward=control.forward
            def observed(curr,pocket_data,times,cond,equis,invs,i,dt):
                if i%10==0: heartbeat(phase='inference',group=group['index'],step=i)
                return forward(curr,pocket_data,times,cond,equis,invs,i,dt)
            control.forward=observed;model._half_guidance=control
            with replay.activate():
                output=model._generate(prior,pocket,steps=100,times=[torch.zeros(50,device='cuda') for _ in range(3)],
                    apply_guidance=False,strategy='linear',solver='euler',save_traj=False,
                    coord_noise_level=cfg['native_options']['coord_noise_level'],final_corr_pred=True)
            expected=[] if program is None else [i for i in range(100) if i/100>=program['window'][0]-1e-6 and (i+1)/100<=program['window'][1]+1e-6]
            active=[r['step'] for r in control.rows if r['active']]
            if active!=expected or len(control.rows)!=100 or len(replay.records)!=100:
                raise ValueError('Guidance window or historical random coverage differs')
            if program and not any(r['nonzero_molecules'] for r in control.rows): raise ValueError('Guidance injected no displacement')
            heartbeat(phase='scoring',group=group['index'],step=100);scores=score(model,output,pocket,com)
            arrays={'terminal_coords_A':output['coords'].numpy().astype(np.float32),
                    'mask':output['mask'].numpy().astype(bool),'half_coords_A':np.asarray(control.teacher_snapshot['coords_A'],dtype=np.float32)}
            arrays.update({k:output[k].argmax(-1).numpy().astype(np.uint8) for k in ('atomics','bonds','charges')})
            with (folder/'states.npz').open('wb') as f: np.savez_compressed(f,**arrays)
            equivalence=None
            if observer is not None:
                equivalence={**observer.summary(),**final_equivalence(cfg,rows,output,scores)}
                equivalence['passed']=equivalence['passed'] and equivalence['max_coordinate_error_A']<=cfg['coordinate_tolerance_A'] and not any(equivalence['categorical_differences'].values())
                atomic_json(folder/'equivalence.json',equivalence)
                if not equivalence['passed']: raise ValueError('Zero-guidance historical path replay failed; see equivalence.json')
            guidance={'steps':control.rows,'active_steps':active,'cumulative_rms_A':control.total.detach().cpu().tolist(),
                      'no_smc':True,'affinity_head_gradient':False,'random_steps':replay.records,
                      'path_bindings':[{k:a[k] for k in ('id','batch','final_slot','root_slot','path_sha256')} for a in rows]}
            with gzip.open(folder/'guidance.json.gz','wt',encoding='utf-8') as f:
                import json
                json.dump(guidance,f,allow_nan=False)
            heartbeat(phase='quality',group=group['index'],step=100)
            mols=model._generate_mols(output);records=[]
            with gzip.open(folder/'terminal.sdf.gz','wt',encoding='utf-8') as f:
                for slot,(a,mol) in enumerate(zip(rows,mols)):
                    pocket_one=slice_tree(pocket,slot,50,'cuda')
                    quality=molecule_quality(mol,pocket_one,com[slot].cpu().numpy(),scale)
                    quality['pb_status']='not_decodable' if mol is None else 'unavailable'
                    if pb_error: quality['pb_error']=pb_error
                    if mol is not None and pb is not None:
                        try:
                            checks=binary_checks(pb.bust(mol,mol_cond=Path(cfg['protein']),full_report=False).iloc[0].to_dict())
                            quality.update(pb_status='evaluated',pb_pass=bool(checks and all(v is True for v in checks.values())),pb_tests=checks)
                        except Exception as e: quality.update(pb_status='error',pb_error=type(e).__name__+': '+str(e))
                    record={**a,'status':'complete','condition':c['id'],'group':group['index'],'row':slot,
                            'pic50':float(scores[slot]),'cumulative_guidance_rms_A':float(control.total[slot]),**quality}
                    records.append(record)
                    if mol is not None:
                        mol.SetProp('_Name',a['id']);f.write(Chem.MolToMolBlock(mol)+'\n> <path_id>\n'+a['id']+'\n\n> <pic50>\n'+str(float(scores[slot]))+'\n\n$$$$\n')
            if len(records)!=50: raise ValueError('Decoder returned fewer than B50 molecules')
            files=['states.npz','guidance.json.gz','terminal.sdf.gz']+(['equivalence.json'] if equivalence is not None else [])
            receipt={'status':'complete','binding':binding(cfg,c,group),'condition':c['id'],'group':group['index'],
                     'records':records,'active_steps':active,'seconds':time.time()-started,'finished_unix':time.time(),
                     'artifacts':{k:digest(folder/k) for k in files}}
            atomic_json(folder/'COMPLETE.json',receipt)
            print({'condition':c['id'],'group':group['index'],'n':50,'mean_pic50':float(scores.mean()),
                   'best_pic50':float(scores.max()),'valid':sum(r['valid_connected'] for r in records),'seconds':receipt['seconds']},flush=True)
        except Exception as error:
            fail_batch(work,cfg,c,group,traceback.format_exc())
            print({'condition':c['id'],'group':group['index'],'error':str(error)},flush=True)
            if program is None or isinstance(error,torch.cuda.OutOfMemoryError) or any(s in str(error).lower() for s in ('device-side assert','illegal memory','hip error')):
                raise
        finally:
            model._path_observer=None;model._path_random=None;model._half_guidance=None
            torch.cuda.empty_cache()
    heartbeat(phase='complete')


def main():
    p=argparse.ArgumentParser()
    for k in ('repo','config','work','condition'): p.add_argument('--'+k,required=True)
    p.add_argument('--group',type=int);a=p.parse_args()
    try: run(a.repo,a.config,a.work,a.condition,a.group)
    except Exception:
        atomic_json(Path(a.work)/'worker_error.json',{'condition':a.condition,'error':traceback.format_exc(),'unix':time.time()});raise


if __name__=='__main__': main()
