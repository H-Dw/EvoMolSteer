"""Locate replay divergence without altering source data or weakening any gate."""
import argparse
import copy
from pathlib import Path
import time

import numpy as np
import torch

from evomolsteer.io import read_json
from evomolsteer.storage.transactions import atomic_json
from .common import load_state
from .native import InitialPool
from .tape import ReplayRandom
from .worker import load_model


def move(value):
    if torch.is_tensor(value): return value.clone().to('cuda')
    if isinstance(value,dict): return {k:move(v) for k,v in value.items()}
    if isinstance(value,list): return [move(v) for v in value]
    return copy.deepcopy(value)


def compare(a,b):
    return {k:{'equal':bool(torch.equal(a[k].cpu(),b[k].cpu())),
               'max_abs':float((a[k].cpu()-b[k].cpu()).abs().max())}
            for k in ('coords','atomics','charges','bonds')}


def encode(model,pocket):
    return model.gen.get_pocket_encoding(pocket['coords'],pocket['atom_names'],
        pocket_atom_charges=pocket['charges'].argmax(-1),pocket_bond_types=pocket['bonds'].argmax(-1),
        pocket_res_types=pocket['res_names'],pocket_atom_mask=pocket['mask'])


def predict(model,curr,pocket,times,cond,enc):
    return model._get_predictions(model(curr,pocket,times,cond_batch=cond,
        pocket_equis=enc[0],pocket_invs=enc[1],training=False))[0]


def run(repo,cfg,work,batch=0):
    root=Path(cfg['remote_dataset'])/cfg['sources'][str(batch)]['relative']
    initial=load_state(root/'initial_state.pt.gz');restart=load_state(root/'restart_step_000.pt.gz')
    endpoint=load_state(root/'endpoint_step_000.pt.gz');model=load_model(cfg,work)
    prior=move(restart['prior']);curr=move(restart['current']);times=move(restart['times']);cond=move(restart['cond'])
    pocket=move(initial['pocket_target']);dt=torch.linspace(0,1,101)[1]
    with np.load(root/'trajectory.npz') as z:
        expected={'coords':torch.from_numpy(z['proposal_coords'][0]),
                  **{k:torch.from_numpy(z['proposal_'+k][0]).long() for k in ('atomics','charges','bonds')}}
    flags={k:p.requires_grad for k,p in model.named_parameters()}
    result={'batch':batch,'torch':torch.__version__,'unix':time.time(),
            'native_grad_flags':{'true':sum(flags.values()),'false_names':[k for k,v in flags.items() if not v]}}
    with torch.no_grad():
        torch.set_rng_state(restart['rng']['torch_cpu']);torch.cuda.set_rng_state_all(restart['rng']['torch_cuda'])
        direct=model.integrator.step(move(curr),move(endpoint),prior,times,dt)
        result['native_integrator_original_endpoint']={
            'coordinate_error':float((direct['coords'].cpu()-expected['coords']).abs().max()),
            'categorical_differences':{k:int((direct[k].argmax(-1).cpu()!=expected[k]).sum()) for k in ('atomics','charges','bonds')}}
        rows=[{'batch':batch,'final_slot':i} for i in range(50)]
        replay=ReplayRandom(cfg,work,rows)
        replay.tables[batch]['before'][0]=np.arange(50)
        replay.begin_step(0)
        with replay.activate(): injected=model.integrator.step(move(curr),move(endpoint),prior,times,dt)
        replay.end_step()
        result['private_generator_vs_original_default_generator']=compare(direct,injected)
        enc=encode(model,pocket);p1=predict(model,curr,pocket,times,cond,enc)
        p2=predict(model,curr,pocket,times,cond,enc)
        result['same_input_repeat']=compare(p1,p2)
        result['same_original_B50_vs_saved_endpoint']=compare(p1,move(endpoint))
        idx=torch.arange(49,-1,-1,device='cuda')
        def perm(v):
            if torch.is_tensor(v) and v.ndim and v.shape[0]==50: return v[idx]
            if isinstance(v,dict): return {k:perm(x) for k,x in v.items()}
            if isinstance(v,list): return [perm(x) for x in v]
            return v
        swapped=predict(model,perm(curr),perm(pocket),perm(times),perm(cond),perm(list(enc)))
        result['reverse_batch_then_restore_order']=compare(p1,perm(swapped))
        by_id={a['id']:a for a in cfg['attempts']};group=[by_id[k] for k in cfg['groups'][0]['ids']]
        pool=InitialPool(cfg);packed,pp,_,_=pool.load(group)
        cond_p={'coords':packed['coords'],'atomics':torch.zeros_like(packed['atomics']),'bonds':torch.zeros_like(packed['bonds'])}
        pred_p=predict(model,packed,pp,[torch.zeros(50,device='cuda')]*3,cond_p,encode(model,pp))
        original_rows=[(i,a['root_slot']) for i,a in enumerate(group) if a['batch']==batch]
        result['packed_shared_group_vs_original_rows']={k:float(torch.stack([
            (pred_p[k][i]-p1[k][j]).abs().max() for i,j in original_rows]).max()) for k in ('coords','atomics','charges','bonds')}
        result['packed_input_vs_original_rows']={}
        for kind,got,expected in [('current',packed,curr),('pocket',pp,pocket),('cond',cond_p,cond)]:
            for key in expected:
                if torch.is_tensor(expected[key]) and expected[key].ndim and expected[key].shape[0]==50:
                    result['packed_input_vs_original_rows'][kind+'.'+key]=max(float((got[key][i].float()-expected[key][j].float()).abs().max()) for i,j in original_rows)
        cached,after_selection=pool.encoding(model,group)
        cached_value=predict(model,packed,pp,[torch.zeros(50,device='cuda')]*3,cond_p,cached)
        result['source_cached_context_vs_original_rows']={k:float(torch.stack([
            (cached_value[k][i]-p1[k][j]).abs().max() for i,j in original_rows]).max()) for k in ('coords','atomics','charges','bonds')}
        result['packed_encoder_vs_source_cached']={str(k):float(torch.stack([
            (encode(model,pp)[k][i]-enc[k][j]).abs().max() for i,j in original_rows]).max()) for k in (0,1)}
        result['precision_and_grad_modes']={}
        for precision,grad in [('high',True),('highest',False),('highest',True),('medium',False)]:
            torch.set_float32_matmul_precision(precision);model.requires_grad_(grad)
            value=predict(model,curr,pocket,times,cond,encode(model,pocket))
            result['precision_and_grad_modes'][precision+'_parameters_grad_'+str(grad)]=compare(value,move(endpoint))
            if precision=='high' and grad:
                packed_value=predict(model,packed,pp,[torch.zeros(50,device='cuda')]*3,cond_p,encode(model,pp))
                result['packed_all_true_vs_original_rows']={k:float(torch.stack([
                    (packed_value[k][i]-value[k][j]).abs().max() for i,j in original_rows]).max()) for k in ('coords','atomics','charges','bonds')}
        result['backend']={'device':torch.cuda.get_device_name(0),'threads':torch.get_num_threads(),
                           'hip':torch.version.hip,'allow_tf32':torch.backends.cuda.matmul.allow_tf32}
    path=Path(work)/'diagnostics'/f'batch_{batch:03d}_primitive_and_model.json';atomic_json(path,result)
    print(result,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--config',required=True)
    p.add_argument('--work',required=True);p.add_argument('--batch',type=int,default=0);a=p.parse_args()
    run(a.repo,read_json(a.config),a.work,a.batch)


if __name__=='__main__': main()
