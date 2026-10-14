"""Restore complete historical states into the upstream native sampling loop.

Only initialization and the loop's skipped prefix are instrumented. Native
integration, categorical sampling and terminal decoding remain upstream code.
Each continuation is executed alone: its RNG identity never depends on the
other trials that happen to be present in a transport batch.
"""
import argparse
import ast
import copy
import gzip
import hashlib
import inspect
import json
from pathlib import Path
import random
import sys
import time
import types
from types import SimpleNamespace

import numpy as np

from evomolsteer.io import digest, read_json, write_json


def instrument_native(source):
    tree=ast.parse(source)
    fn=tree.body[0]
    assignments=[i for i,node in enumerate(fn.body) if isinstance(node,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='step_sizes' for t in node.targets)]
    if len(assignments)!=1:raise ValueError('Unsupported native step-grid initialization')
    fn.body[assignments[0]:assignments[0]]=ast.parse(
        'prior, curr, cond_batch, times = self._agentic_suffix.restore()').body
    loops=[n for n in ast.walk(fn) if isinstance(n,ast.For) and isinstance(n.iter,ast.Call)
        and isinstance(n.iter.func,ast.Name) and n.iter.func.id=='enumerate'
        and n.iter.args and isinstance(n.iter.args[0],ast.Name) and n.iter.args[0].id=='step_sizes']
    if len(loops)!=1:raise ValueError('Unsupported native integration loop')
    loop=loops[0]
    loop.body[0:0]=ast.parse('if i < self._agentic_suffix.start_step:\n    continue\n'
                            'self._agentic_suffix.before(i, curr, times)').body
    loop.body.extend(ast.parse('self._agentic_suffix.after(i, curr, times)').body)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)+'\n'


def slice_tree(value,slot,batch_size,device):
    import torch
    if torch.is_tensor(value):
        if value.ndim and value.shape[0]==batch_size:return value[slot:slot+1].clone().to(device)
        return value.clone().to(device)
    if isinstance(value,dict):return {k:slice_tree(v,slot,batch_size,device) for k,v in value.items()}
    if isinstance(value,list):
        if len(value)==batch_size:return [copy.deepcopy(value[slot])]
        return [slice_tree(v,slot,batch_size,device) for v in value]
    if isinstance(value,tuple):return tuple(slice_tree(v,slot,batch_size,device) for v in value)
    return copy.deepcopy(value)


class RestoredSuffix:
    def __init__(self,state,slot,steps,device):
        self.state=state;self.slot=slot;self.steps=steps;self.device=device
        self.start_step=int(state['step']);self.batch_size=len(state['current']['coords'])
        self.executed=[];self.expected_time=float(state['times'][0][slot])
    def restore(self):
        return tuple(slice_tree(self.state[k],self.slot,self.batch_size,self.device)
                     for k in ['prior','current','cond','times'])
    def before(self,i,curr,times):
        if i!=self.start_step+len(self.executed):raise ValueError('Non-contiguous native suffix')
        if abs(float(times[0][0])-i/self.steps)>3e-6:raise ValueError('Restored generation clock drift')
        self.executed.append(int(i))
    def after(self,i,curr,times):
        if abs(float(times[0][0])-(i+1)/self.steps)>3e-6:raise ValueError('Native suffix time update mismatch')


def molecule_quality(mol,pocket,com,scale,severe_distance=.8):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    result={'valid_connected':False,'qualified_structure':False,'smiles':None,
            'energy_status':'not_decodable','mmff_relief_per_heavy':None,'severe_pairs':None}
    if mol is None:return result
    try:
        Chem.SanitizeMol(mol)
        result['valid_connected']=len(Chem.GetMolFrags(mol))==1
        result['smiles']=Chem.MolToSmiles(mol)
        xyz=mol.GetConformer().GetPositions()
        pc=pocket['coords'][0][pocket['mask'][0].bool()].detach().cpu().numpy()*scale+com
        distances=np.linalg.norm(xyz[:,None]-pc[None],axis=-1)
        result['severe_pairs']=int((distances<severe_distance).sum())
        result['min_receptor_distance_A']=float(distances.min())
        result['qualified_structure']=result['valid_connected'] and result['severe_pairs']==0
        expanded=Chem.AddHs(Chem.Mol(mol),addCoords=True)
        prop=AllChem.MMFFGetMoleculeProperties(expanded,mmffVariant='MMFF94s')
        if prop is None:result['energy_status']='unsupported';return result
        ff=AllChem.MMFFGetMoleculeForceField(expanded,prop)
        before=float(ff.CalcEnergy());code=ff.Minimize(maxIts=500);after=float(ff.CalcEnergy())
        result['energy_status']='converged' if code==0 else 'not_converged'
        result['mmff_relief_per_heavy']=(before-after)/max(1,mol.GetNumHeavyAtoms())
    except Exception as error:
        result['failure_reason']=type(error).__name__+': '+str(error)
    return result


def run_job(path):
    import torch
    job=read_json(path);out=Path(job['output']);out.mkdir(parents=True,exist_ok=True)
    checkpoint=Path(job['checkpoint']);root=Path(job['flowr_root']);dataset=Path(job['dataset'])
    sys.path.insert(0,str(root))
    from flowr.gen.generate_from_pdb_selective import get_args
    from flowr.scriptutil import load_model
    argv=sys.argv
    sys.argv=['agentic-native','--arch','pocket','--pocket_type','holo','--gpus','1',
        '--num_workers','0','--batch_cost','1','--sample_n_molecules_per_target','1',
        '--ckpt_path',str(checkpoint),'--save_dir',str(out),'--integration_steps',str(job['steps']),
        '--no_cat_noise_euler_guard','--no_ligand_valence_repair',
        '--pdb_file_target',str(dataset/'inputs'/job['inputs']['protein']),
        '--ligand_file_target',str(dataset/'inputs'/job['inputs']['ligand']),
        '--pdb_file_untarget',str(dataset/'inputs'/job['inputs']['off_protein']),
        '--ligand_file_untarget',str(dataset/'inputs'/job['inputs']['off_ligand'])]
    try:args=get_args()
    finally:sys.argv=argv
    args.seed=job['master_seed'];torch.set_float32_matmul_precision('high')
    model=load_model(args)[0].to('cuda').eval();model.requires_grad_(False)
    if model.inpainting_mode or model.graph_inpainting or model.sc_recycle_steps!=1:
        raise ValueError('Unsupported historical continuation semantics')
    model.integrator.use_sde_simulation=True
    model.integrator.coord_noise_level=float(job['coord_noise_level'])
    original=model._generate
    import textwrap
    source=textwrap.dedent(inspect.getsource(original));changed=instrument_native(source)
    namespace=dict(original.__func__.__globals__)
    exec(compile(changed,'agentic_instrumented_native.py','exec'),namespace)
    model._generate=types.MethodType(namespace['_generate'],model)
    import flowr.models.fm_pocket as fm
    def forbidden(*args,**kwargs):raise RuntimeError('SMC called inside a native continuation')
    fm.apply_smc_guidance=forbidden;fm.apply_selective_smc_guidance=forbidden
    namespace['apply_smc_guidance']=forbidden
    write_json(out/'source_attestation.json',{'upstream_native_sha256':hashlib.sha256(source.encode()).hexdigest(),
        'instrumented_sha256':hashlib.sha256(changed.encode()).hexdigest(),'checkpoint_sha256':digest(checkpoint),
        'torch':torch.__version__,'flowr_source':str(root),'particle_resampling':False,
        'gradient_guidance':False,'runtime_device':torch.cuda.get_device_name(0)})
    (out/'upstream_native.py').write_text(source);(out/'instrumented_native.py').write_text(changed)
    cache={};completed=[];new_steps=0;reused=0
    for trial in job['trials']:
        record=out/(trial['trial_id']+'.json')
        if record.exists():
            saved=read_json(record)
            if saved['trial']!=trial or saved['status']!='complete':raise ValueError('Existing trial binding mismatch')
            completed.append(saved);reused+=1;continue
        started=time.time();batch=trial['batch'];slot=trial['slot'];step=trial['step']
        folder=dataset/'results'/job['campaign']/'single'/f'batch_{batch:03d}'
        restart=folder/f'restart_step_{step:03d}.pt.gz';initial=folder/'initial_state.pt.gz'
        if digest(restart)!=trial['restart_sha256'] or digest(initial)!=trial['initial_sha256']:
            raise ValueError('Historical full-state hash mismatch')
        if batch not in cache:
            with gzip.open(initial,'rb') as f:init=torch.load(f,map_location='cpu',weights_only=False)
            with gzip.open(restart,'rb') as f:state=torch.load(f,map_location='cpu',weights_only=False)
            cache[batch]=(init,state)
        init,state=cache[batch];device=torch.device('cuda')
        suffix=RestoredSuffix(state,slot,job['steps'],device);model._agentic_suffix=suffix
        pocket=slice_tree(init['pocket_target'],slot,len(init['target_com']),device)
        com=np.asarray(init['target_com'][slot]);pocket['complex']=[SimpleNamespace(com=com)]
        prior,current,cond,times=suffix.restore()
        with torch.no_grad():
            first=model._get_predictions(model(current,pocket,times,
                cond_batch=cond if model.self_condition else None,training=False))[0]
        actual=first['coords'][0].detach().cpu().numpy()*model.coord_scale+com
        difference=float(np.sqrt(np.square(actual-np.asarray(trial['expected_forecast_A'])).sum(1).mean()))
        if difference>job['expected_forecast_rms_tolerance_A']:
            raise ValueError(f'Restored first forecast mismatch: {difference:.6g} A')
        torch.manual_seed(trial['rng_seed']);np.random.seed(trial['rng_seed']);random.seed(trial['rng_seed'])
        output=model._generate(prior,pocket,steps=job['steps'],times=times,apply_guidance=False,
            strategy='linear',solver='euler',save_traj=False,coord_noise_level=job['coord_noise_level'])
        if suffix.executed!=list(range(step,job['steps'])):raise ValueError('Incomplete native suffix')
        terminal={k:v.clone() for k,v in output.items() if torch.is_tensor(v) and k!='affinity'}
        terminal['coords']=(terminal['coords']-terminal['coords'].new_tensor(com)[None,None])/model.coord_scale
        for k in ['atomics','bonds','charges']:
            terminal[k]=torch.nn.functional.one_hot(terminal[k].argmax(-1),terminal[k].shape[-1]).float()
        with torch.no_grad():
            scores=model._predict_affinity(terminal,terminal,pocket,[torch.full((1,),.9999,device=device)]*3)
        pic50=float(scores['affinity']['pic50'].reshape(-1)[0]);mol=model._generate_mols(output)[0]
        quality=molecule_quality(mol,pocket,com,model.coord_scale)
        if mol is not None:
            from rdkit import Chem
            text=Chem.MolToMolBlock(mol)+'\n$$$$\n'
            with gzip.open(out/(trial['trial_id']+'.sdf.gz'),'wt') as f:f.write(text)
        result={'status':'complete','trial':trial,'pic50':pic50,**quality,
            'elite_hit':bool(quality['qualified_structure'] and pic50>=job['threshold']),
            'suffix_steps':len(suffix.executed),'first_forecast_rms_A':difference,
            'first_head_difference':float(first['affinity']['pic50'].reshape(-1)[0])-trial['expected_joint_head'],
            'particle_resampling':False,'gradient_guidance':False,'elapsed_seconds':time.time()-started,
            'rng_identity':'One trial per forward population; node/replicate-derived child stream',
            'terminal_coords_sha256':hashlib.sha256(output['coords'].detach().cpu().numpy().tobytes()).hexdigest()}
        write_json(record,result);completed.append(result);new_steps+=len(suffix.executed)
        print(json.dumps({'trial':trial['trial_id'],'pic50':pic50,'valid':quality['valid_connected'],
            'elite':result['elite_hit'],'seconds':result['elapsed_seconds']}),flush=True)
    write_json(out/(Path(path).stem+'.complete.json'),{'complete':True,'job_sha256':digest(path),
        'trials':len(completed),'reused_trials':reused,'new_seed_steps':new_steps})


def main():
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);a=p.parse_args();run_job(a.job)


if __name__=='__main__':main()
