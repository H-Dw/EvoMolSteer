"""Native from-zero FLOWR pilot with actual-teacher-clock coordinate VJP."""
import argparse
import ast
import gzip
import hashlib
import inspect
from pathlib import Path
import random
import sys
import textwrap
import time
import types
from types import SimpleNamespace

import numpy as np
import torch

from evomolsteer.io import read_json,write_json,digest
from evomolsteer.agentic_steer.native import molecule_quality
from evomolsteer.generation.coordinate_reward import predictive_flow_increment
from .reward import HalfStateReward,bridge


def tensor_tree(value,n,b,device):
    if torch.is_tensor(value):return (value[:n] if value.ndim and value.shape[0]==b else value).clone().to(device)
    if isinstance(value,dict):return {k:tensor_tree(v,n,b,device) for k,v in value.items()}
    if isinstance(value,list):return value[:n] if len(value)==b else [tensor_tree(v,n,b,device) for v in value]
    return value


def instrument(source):
    tree=ast.parse(source);fn=tree.body[0]
    loops=[n for n in ast.walk(fn) if isinstance(n,ast.For) and isinstance(n.iter,ast.Call) and isinstance(n.iter.func,ast.Name)
        and n.iter.func.id=='enumerate' and n.iter.args and isinstance(n.iter.args[0],ast.Name) and n.iter.args[0].id=='step_sizes']
    if len(loops)!=1:raise ValueError('Ambiguous native flow loop')
    loop=loops[0];sites=[]
    for index,node in enumerate(loop.body):
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='out' and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='self':sites.append(index)
    if len(sites)!=1:raise ValueError('Ambiguous first native forward')
    loop.body[sites[0]]=ast.parse('out = self._half_guidance.forward(curr, pocket_data, times, cond, pocket_equis, pocket_invs, i, step_size)').body[0]
    loop.body.extend(ast.parse('curr = self._half_guidance.after(curr, i)').body)
    # Capture the true integrator endpoint independently of final_corr_pred.
    position=fn.body.index(next(n for n in fn.body if loop in ast.walk(n)))
    fn.body.insert(position+1,ast.parse('if isinstance(curr,dict):\n    self._half_guidance.integration_snapshot = {k: v.detach().clone() for k,v in curr.items() if torch.is_tensor(v)}').body[0])
    ast.fix_missing_locations(tree);return ast.unparse(tree)+'\n'


def instrument_coord_telemetry(source):
    """Observe existing random draws and drift; never draw additional noise."""
    tree=ast.parse(source);fn=tree.body[0]
    last=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Return))
    code="""self._guidance_components = {}
if self.use_sde_simulation and self.coord_strategy != 'velocity-sample':
    self._guidance_components = {'predictive':coord_velocity*step_size, 'score_drift':gt*score*step_size, 'random':eps*std_eps*step_size}
"""
    fn.body[last:last]=ast.parse(code).body;ast.fix_missing_locations(tree);return ast.unparse(tree)+'\n'


class Control:
    def __init__(self,model,program,teacher,com,capture_time=None):
        self.model=model;self.program=program;self.reward=HalfStateReward(program,teacher) if program else None
        self.com=com;self.rows=[];self.cached=None;self.total=torch.zeros(len(com),device=com.device)
        self.capture_time=capture_time;self.teacher_snapshot=None
    def forward(self,curr,pocket,times,cond,equis,invs,i,dt):
        self.cached=None;t=float(times[0][0]);self.dt=float(dt);self.before=curr['coords'].detach().clone()
        enabled=self.reward is not None and self.reward.active(t,float(dt));self.t=t
        kwargs=dict(cond_batch=cond,pocket_equis=equis,pocket_invs=invs,training=False)
        if not enabled:
            return self.model(curr,pocket,times,**kwargs)
        with torch.enable_grad():
            x=self.before.clone().requires_grad_(True);state=dict(curr,coords=x);out=self.model(state,pocket,times,**kwargs)
            if 'affinity' in out:
                def denied(gradient):raise RuntimeError('Affinity head entered gradient reward')
                for v in out['affinity'].values():
                    if torch.is_tensor(v) and v.requires_grad:v.register_hook(denied)
            pred,_=self.model._get_predictions(out)
            y=pred['coords']*self.model.coord_scale+self.com[:,None]
            current=x*self.model.coord_scale+self.com[:,None]
            z=bridge(current,y,t,self.reward.teacher['time']);value,detail=self.reward.value(z,curr['mask'].bool())
            gx,gz=torch.autograd.grad(value.sum(),(x,z))
        if not bool(torch.isfinite(gx).all() and torch.isfinite(value).all()):raise ValueError('Nonfinite real FLOWR pullback')
        self.cached=(gx.detach(),pred['coords'].detach(),float(dt),detail['direct_mask'].detach(),value.detach(),gz.detach())
        return {k:({a:b.detach() if torch.is_tensor(b) else b for a,b in v.items()} if isinstance(v,dict) else v.detach() if torch.is_tensor(v) else v) for k,v in out.items()}
    def after(self,curr,i):
        row={'step':i,'time':self.t,'active':self.cached is not None,'nonzero_molecules':0,'injected_rms_A_mean':0.}
        for name,value in getattr(self.model.integrator,'_guidance_components',{}).items():
            mask=curr['mask'].bool();rr=((value.square()*mask[...,None]).sum((1,2))/mask.sum(1)).sqrt()
            row['native_'+name+'_rms_A_mean']=float((rr*self.model.coord_scale).mean())
        if self.cached is not None:
            gradient,endpoint,dt,direct,value,gz=self.cached;mask=curr['mask'].bool();weight=mask[...,None]
            flow=predictive_flow_increment(self.before,endpoint,self.t,dt,self.model.integrator.use_cosine_scheduler)
            rms=lambda a:((a.square()*weight).sum((1,2))/mask.sum(1)).sqrt()
            gnorm=rms(gradient);requested=self.program['dose']['native_rms_ratio']*rms(flow)
            # New controllers may attenuate the dose. Legacy behavior is exact
            # when this attribute is absent; no additional random/model calls.
            if hasattr(self,'dose_gain'):requested=requested*self.dose_gain
            shift=gradient*(requested/gnorm.clamp_min(1e-20))[:,None,None]*weight
            norms=shift.norm(dim=-1)*self.model.coord_scale;cap=self.program['dose']['max_atom_step_A']
            atom_clip=(norms>cap)&mask
            shift*=torch.minimum(torch.ones_like(norms),cap/norms.clamp_min(1e-20))[...,None]
            displacement=rms(shift)*self.model.coord_scale
            remaining=(self.program['dose']['max_cumulative_rms_A']-self.total).clamp_min(0)
            cumulative_clip=displacement>remaining
            shift*=torch.minimum(torch.ones_like(displacement),remaining/displacement.clamp_min(1e-20))[:,None,None]
            displacement=rms(shift)*self.model.coord_scale;self.total+=displacement
            curr=dict(curr,coords=curr['coords']+shift)
            row.update(nonzero_molecules=int((displacement>1e-9).sum()),injected_rms_A_mean=float(displacement.mean()),
                current_gradient_rms_mean=float(gnorm.mean()),direct_unmatched_gradient_max=float(gz.masked_fill(direct[...,None],0).abs().max()),
                current_unmatched_gradient_max=float(gradient.masked_fill(direct[...,None],0).abs().max()),reward_mean=float(value.mean()),
                matched_fraction_mean=float(direct.float().mean()),production_forward_calls=1)
            row.update(requested_rms_A_mean=float((requested*self.model.coord_scale).mean()),
                atom_clip_fraction=float(atom_clip.sum()/mask.sum()),cumulative_clip_fraction=float(cumulative_clip.float().mean()),
                predictive_flow_rms_A_mean=float((rms(flow)*self.model.coord_scale).mean()),
                native_observed_rms_A_mean=float((rms(curr['coords']-shift-self.before)*self.model.coord_scale).mean()))
            if self.program.get('family')=='ancestral_modes_v3':
                row.update(injected_rms_A_per_seed=displacement.cpu().tolist(),
                    requested_rms_A_per_seed=(requested*self.model.coord_scale).cpu().tolist())
        if self.capture_time is not None and abs(self.t+self.dt-self.capture_time)<2e-6:
            if self.teacher_snapshot is not None:raise ValueError('Teacher-clock snapshot repeated')
            self.teacher_snapshot={'time':self.capture_time,'representation':'current_world_A',
                'coords_A':(curr['coords']*self.model.coord_scale+self.com[:,None]).detach().cpu().tolist(),
                'mask':curr['mask'].detach().cpu().bool().tolist()}
            self.teacher_categories={k:curr[k].argmax(-1).detach().cpu().numpy().astype(np.uint8) for k in ['atomics','bonds','charges'] if k in curr}
        self.rows.append(row);return curr


def run(repo,cfg):
    required=['study','remote_output','checkpoint','remote_dataset','campaign','arm','pilot_initial_batch',
        'pilot_samples','seed','steps','teacher_time','terminal_time']
    missing=[key for key in required if key not in cfg]
    if missing:raise ValueError('Incomplete native inference configuration: '+', '.join(missing))
    if cfg['terminal_time']!=1.:raise ValueError('Native adapter requires a completed terminal clock at 1')
    from flowr.gen.generate_from_pdb_selective import get_args
    from flowr.scriptutil import load_model
    root=Path(repo);study=root/cfg['study'];catalog=list((study/'candidates').glob('*/bindings.json'))
    entries=[]
    for path in catalog:
        binding=read_json(path);teacher_path=root/binding['teacher'];teacher=read_json(teacher_path)
        for item in binding['programs']:
            if digest(root/item['program'])!=item['sha256']:raise ValueError('Compiled program altered')
            p=read_json(root/item['program'])
            if digest(teacher_path)!=p['teacher_sha256']:raise ValueError('Teacher altered')
            if p.get('teacher_time',teacher['time'])!=teacher['time'] or cfg['teacher_time']!=teacher['time']:
                raise ValueError('Runtime capture and learned teacher clocks differ')
            if 'native_options' in p and p['native_options']!=cfg.get('native_options'):
                raise ValueError('Compiled native parameters differ from runtime')
            entries.append((p,teacher,item['operative_text']))
    if 'only_program_ids' in cfg:
        requested=set(cfg['only_program_ids']);entries=[entry for entry in entries if entry[0]['program_id'] in requested]
        if {e[0]['program_id'] for e in entries}!=requested:raise ValueError('Unregistered requested program')
    if not entries and not cfg.get('include_unguided',True):raise ValueError('No approved programs; cannot invent local teacher slots')
    remote=Path(cfg['remote_output']);remote.mkdir(parents=True,exist_ok=True)
    original_argv=sys.argv;sys.argv=['final-guidance','--arch','pocket','--pocket_type','holo','--gpus','1','--num_workers','0',
        '--batch_cost',str(cfg['pilot_samples']),'--sample_n_molecules_per_target',str(cfg['pilot_samples']),
        '--ckpt_path',cfg['checkpoint'],'--save_dir',str(remote),'--integration_steps',str(cfg['steps']),
        '--no_cat_noise_euler_guard','--no_ligand_valence_repair']
    try:args=get_args()
    finally:sys.argv=original_argv
    args.seed=cfg['seed'];torch.set_float32_matmul_precision('high');model=load_model(args)[0].to('cuda').eval();model.requires_grad_(False)
    if model.sc_recycle_steps!=1 or model.inpainting_mode or model.graph_inpainting or model.integrator.use_cosine_scheduler:raise ValueError('Unsupported native linear bridge configuration')
    native=cfg.get('native_options',{})
    allowed={'coord_noise_level','cat_noise_level','cat_noise_euler_guard','final_corr_pred','use_sde_simulation'}
    if set(native)-allowed:raise ValueError('Unknown native options: '+str(set(native)-allowed))
    model.integrator.use_sde_simulation=native.get('use_sde_simulation',True)
    model.integrator.coord_noise_level=float(native.get('coord_noise_level',.2))
    if 'cat_noise_level' in native:model.integrator.cat_noise_level=float(native['cat_noise_level'])
    if 'cat_noise_euler_guard' in native:model.integrator.cat_noise_euler_guard=bool(native['cat_noise_euler_guard'])
    source=textwrap.dedent(inspect.getsource(model._generate))
    if cfg.get('resume'):
        from evomolsteer.agentic_steer.native import instrument_native
        changed=instrument(instrument_native(source))
    else:changed=instrument(source)
    namespace=dict(model._generate.__func__.__globals__)
    coord_source=textwrap.dedent(inspect.getsource(model.integrator.coord_step))
    coord_changed=instrument_coord_telemetry(coord_source);coord_namespace=dict(model.integrator.coord_step.__func__.__globals__)
    exec(compile(coord_changed,'guidance_coord_telemetry.py','exec'),coord_namespace)
    model.integrator.coord_step=types.MethodType(coord_namespace['coord_step'],model.integrator)
    def forbidden(*args,**kwargs):raise RuntimeError('Unexpected SMC inside gradient pilot')
    namespace['apply_smc_guidance']=forbidden
    exec(compile(changed,'final_teacher_native.py','exec'),namespace);model._generate=types.MethodType(namespace['_generate'],model)
    initial=Path(cfg['remote_dataset'])/'results'/cfg['campaign']/cfg['arm']/f"batch_{cfg['pilot_initial_batch']:03d}"/'initial_state.pt.gz'
    with gzip.open(initial,'rb') as f:init=torch.load(f,map_location='cpu',weights_only=False)
    b=len(init['prior']['coords']);n=cfg['pilot_samples'];device=torch.device('cuda')
    if not 1<=n<=b:raise ValueError('Unsupported pilot sample count')
    selected_indices=cfg.get('resume',{}).get('slots',list(range(n)))
    if len(selected_indices)!=n:raise ValueError('Continuation sample count differs from slots')
    com=torch.as_tensor(np.asarray(init['target_com'])[selected_indices].reshape(n,3),device=device,dtype=torch.float32)
    source_audit={'upstream_native_sha256':hashlib.sha256(source.encode()).hexdigest(),'instrumented_sha256':hashlib.sha256(changed.encode()).hexdigest(),
        'control_adapter_sha256':digest(__file__),'reward_adapter_sha256':digest(Path(__file__).with_name('reward.py')),
        'checkpoint_sha256':digest(cfg['checkpoint']),'initial_sha256':digest(initial),'initial_slots':list(range(n)),
        'seed':cfg['seed'],'native_graph_transitions':True,'smc':False,'affinity_head_gradient':False,'additional_per_step_affinity_calls':0,
        'native_integrator_hparams':model.integrator.hparams,
        'native_integrator_sha256':digest(inspect.getsourcefile(type(model.integrator))),
        'coord_telemetry_sha256':hashlib.sha256(coord_changed.encode()).hexdigest(),
        'resolved_native_options':{**native,'final_corr_pred':native.get('final_corr_pred',True)},
        'model_derivative':'Actual FLOWR autograd VJP plus bridge identity term','bridge':'One-forward linear teacher-clock approximation, not exact SDE lookahead'}
    if any(entry[0]['family']=='local_nc_layout' for entry in entries):
        source_audit['nc_extension_sha256']=digest(Path(__file__).parent.parent/'nc_guidance/reward.py')
    if any(entry[0]['family']=='local_motif_v2' for entry in entries):
        source_audit['motif_extension_sha256']=digest(Path(__file__).parent.parent/'local_motif/reward.py')
    restart=None
    if cfg.get('resume'):
        from evomolsteer.local_motif.suffix_probe import GroupedSuffix,select_tree
        restart_path=initial.parent/('restart_step_%03d.pt.gz'%cfg['resume']['step'])
        if digest(restart_path)!=cfg['resume']['restart_sha256']:raise ValueError('Restart state altered')
        with gzip.open(restart_path,'rb') as f:restart=torch.load(f,map_location='cpu',weights_only=False)
        source_audit.update(resume=cfg['resume'],initial_slots=selected_indices,
            continuation_identity='Saved complete state with newly seeded paired suffix stream, not historical RNG replay')
    write_json(remote/'source_audit.json',source_audit);(remote/'instrumented_native.py').write_text(changed)
    conditions=([(None,None,'Unselected random-prior from-zero native control')] if cfg.get('include_unguided',True) else [])+entries
    for p,teacher,text in conditions:
        name='unguided' if p is None else p['program_id'];out=remote/name;out.mkdir(exist_ok=True)
        if (out/'COMPLETE.json').exists():continue
        prior=tensor_tree(init['prior'],n,b,device);pocket=tensor_tree(init['pocket_target'],n,b,device)
        if restart is not None:
            prior=select_tree(init['prior'],selected_indices,b,device)
            pocket=select_tree(init['pocket_target'],selected_indices,b,device)
            model._agentic_suffix=GroupedSuffix(restart,selected_indices,cfg['steps'],device)
        pocket['complex']=[SimpleNamespace(com=c.cpu().numpy()) for c in com]
        torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);random.seed(cfg['seed']);started=time.time()
        control_type=Control
        if p is not None and p['family']=='actual_partial_two_clock':
            from evomolsteer.partial_search.two_clock import TwoClockControl
            control_type=TwoClockControl
        if p is not None and p['family']=='local_nc_layout':
            from evomolsteer.nc_guidance.reward import NCControl
            control_type=NCControl
        if p is not None and p['family']=='local_motif_v2':
            from evomolsteer.local_motif.reward import MotifControl
            control_type=MotifControl
        if p is not None and p['family']=='ancestral_modes_v3':
            from evomolsteer.ancestral_modes.reward import AncestorControl
            control_type=AncestorControl
        control=control_type(model,p,teacher,com,cfg['teacher_time']);model._half_guidance=control
        output=model._generate(prior,pocket,steps=cfg['steps'],times=[torch.zeros(n,device=device) for _ in range(3)],
            apply_guidance=False,strategy='linear',solver='euler',save_traj=False,
            coord_noise_level=model.integrator.coord_noise_level,final_corr_pred=native.get('final_corr_pred',True))
        start_step=cfg.get('resume',{}).get('step',0)
        actual=[r['step'] for r in control.rows if r['active']];expected=[] if p is None else [s for s in p['expected_active_steps'] if s>=start_step]
        if actual!=expected or len(control.rows)!=cfg['steps']-start_step:raise ValueError('Actual active windows differ from compiled decision')
        if p is not None and p['family'] not in ('local_motif_v2','ancestral_modes_v3') and not any(r['nonzero_molecules'] for r in control.rows):raise ValueError('Guidance did not inject any displacement')
        if control.teacher_snapshot is None:raise ValueError('Actual teacher-clock state was not captured')
        final={k:v.to(device) for k,v in output.items() if torch.is_tensor(v) and k!='affinity'}
        final['coords']=(final['coords']-com[:,None])/model.coord_scale
        for k in ['atomics','bonds','charges']:final[k]=torch.nn.functional.one_hot(final[k].argmax(-1),final[k].shape[-1]).float()
        with torch.no_grad():scores=model._predict_affinity(final,final,pocket,[torch.full((n,),.9999,device=device)]*3)['affinity']['pic50'].reshape(-1).cpu().numpy()
        mols=model._generate_mols(output);records=[]
        from rdkit import Chem
        for slot,mol in enumerate(mols):
            single={k:(v[slot:slot+1] if torch.is_tensor(v) and v.ndim and v.shape[0]==n else v) for k,v in pocket.items()}
            q=molecule_quality(mol,single,com[slot].cpu().numpy(),model.coord_scale)
            records.append({'slot':slot,'pic50':float(scores[slot]),**q})
            if mol is not None:
                with gzip.open(out/f'{slot:03d}.sdf.gz','wt') as f:f.write(Chem.MolToMolBlock(mol)+'\n$$$$\n')
        integration=control.integration_snapshot
        integration_summary={'time':1.,'representation':'integration_current_world_A',
            'coords_A':(integration['coords']*model.coord_scale+com[:,None]).cpu().tolist(),
            'atomic_indices':integration['atomics'].argmax(-1).cpu().tolist()}
        write_json(out/'geometry.json',{'teacher_clock':control.teacher_snapshot,'integration_current':integration_summary,'terminal':{
            'time':cfg['terminal_time'],'representation':'terminal_world_A','coords_A':output['coords'].tolist(),'mask':output['mask'].bool().tolist(),
            'atomic_indices':output['atomics'].argmax(-1).cpu().tolist()}})
        write_json(out/'records.json',records);write_json(out/'guidance_audit.json',{'program':p,'operative_text':text,'steps':control.rows,
            'actual_active_steps':actual,'actual_steps':len(control.rows),'mean_cumulative_control_rms_A':float(control.total.mean()),
            'no_smc':True,'no_affinity_head_gradient':True,'real_generator_vjp':p is not None,'input_pairing':source_audit})
        if cfg.get('save_categorical_states',False) or (p is not None and p['family']=='local_motif_v2'):
            graph_arrays={'terminal_'+k:output[k].argmax(-1).cpu().numpy().astype(np.uint8) for k in ['atomics','bonds','charges']}
            graph_arrays.update({'boundary_'+k:v for k,v in getattr(control,'teacher_categories',{}).items()})
            np.savez_compressed(out/'categorical_states.npz',**graph_arrays)
        summary={'condition':name,'n':n,'mean_pic50_all':float(scores.mean()),'best_pic50':float(scores.max()),
            'qualified':sum(r['qualified_structure'] for r in records),'seconds':time.time()-started,'active_updates':len(actual),
            'nonzero_updates':sum(r['nonzero_molecules']>0 for r in control.rows)}
        write_json(out/'COMPLETE.json',summary);print(summary,flush=True)
    write_json(remote/'COMPLETE.json',{'conditions':len(conditions),'samples_per_condition':n,'paired_initial_prior':True,
        'full_inference_steps':cfg['steps'] if not cfg.get('resume') else None,'integration_grid_steps':cfg['steps'],
        'executed_steps_per_sample':cfg['steps']-cfg.get('resume',{}).get('step',0),'resume_start_step':cfg.get('resume',{}).get('step',0)})


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',default='.');p.add_argument('--config',required=True);p.add_argument('--flowr-root',required=True);a=p.parse_args()
    sys.path.insert(0,str(Path(a.flowr_root).resolve()));run(a.repo,read_json(a.config))


if __name__=='__main__':main()
