"""Native-loop hooks, exact saved priors and historical trajectory checks."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from evomolsteer.agentic_steer.native import slice_tree
from evomolsteer.final_guidance.native import instrument
from .common import load_state,identity


def instrument_replay(source):
    tree=ast.parse(instrument(source));fn=tree.body[0]
    loops=[n for n in ast.walk(fn) if isinstance(n,ast.For) and isinstance(n.iter,ast.Call)
           and isinstance(n.iter.func,ast.Name) and n.iter.func.id=='enumerate'
           and n.iter.args and isinstance(n.iter.args[0],ast.Name) and n.iter.args[0].id=='step_sizes']
    if len(loops)!=1: raise ValueError('Ambiguous historical flow loop')
    loop=loops[0]
    loop.body[:0]=ast.parse('self._path_random.begin_step(i)').body
    loop.body += ast.parse('self._path_random.end_step()\nself._path_observer(i, curr, predicted)').body
    ast.fix_missing_locations(tree);return ast.unparse(tree)+'\n'


def join_trees(values):
    first=values[0]
    if torch.is_tensor(first):
        if first.ndim and all(v.shape[0]==1 for v in values): return torch.cat(values,0)
        if not all(torch.equal(first,v) for v in values): raise ValueError('Nonbatch tensor differs')
        return first.clone()
    if isinstance(first,dict): return {k:join_trees([v[k] for v in values]) for k in first}
    if isinstance(first,(list,tuple)):
        if all(len(v)==1 for v in values): return [v[0] for v in values]
        if not all(len(v)==len(first) for v in values): raise ValueError('Nonbatch tree shape differs')
        return [join_trees([v[i] for v in values]) for i in range(len(first))]
    if not all(v==first for v in values): raise ValueError('Nonbatch metadata differs')
    return copy.deepcopy(first)


class InitialPool:
    def __init__(self,config): self.config=config;self.cache={};self.encoding_cache={}
    def load(self,attempts,device='cuda'):
        priors=[];pockets=[];coms=[];scale=None
        for a in attempts:
            b=a['batch'];folder=Path(self.config['remote_dataset'])/self.config['sources'][str(b)]['relative']
            if b not in self.cache: self.cache[b]=load_state(folder/'initial_state.pt.gz')
            saved=self.cache[b];root=a['root_slot']
            priors.append(slice_tree(saved['prior'],root,50,device))
            pockets.append(slice_tree(saved['pocket_target'],root,50,device))
            # The recorder stored COM as [B,1,3]; controllers and scoring need
            # [B,3]. Keep values exact while preventing an extra broadcast axis.
            coms.append(np.asarray(saved['target_com'])[root].reshape(3))
            if scale is not None and scale!=saved['coord_scale']: raise ValueError('Coordinate scale differs by source')
            scale=saved['coord_scale']
        prior=join_trees(priors);pocket=join_trees(pockets)
        com=torch.as_tensor(np.asarray(coms),device=device,dtype=torch.float32)
        pocket['complex']=[SimpleNamespace(com=v.cpu().numpy()) for v in com]
        return prior,pocket,com,scale

    def encoding(self,model,attempts,device='cuda'):
        """Recreate the original constant B50 pocket context before gathering roots.

        SMC copied these tensors; it did not re-encode the regrouped pocket.
        Preserve the initial zero-stride expansion and its contiguous layout
        after the original first selection, without performing any selection.
        """
        def move(value):
            if torch.is_tensor(value): return value.to(device)
            if isinstance(value,dict): return {k:move(v) for k,v in value.items()}
            return value
        for b in {a['batch'] for a in attempts}:
            if b in self.encoding_cache: continue
            pocket=move(self.cache[b]['pocket_target'])
            with torch.no_grad():
                self.encoding_cache[b]=model.gen.get_pocket_encoding(pocket['coords'],pocket['atom_names'],
                    pocket_atom_charges=pocket['charges'].argmax(-1),pocket_bond_types=pocket['bonds'].argmax(-1),
                    pocket_res_types=pocket['res_names'],pocket_atom_mask=pocket['mask'])
        packed=tuple(torch.cat([self.encoding_cache[a['batch']][k][a['root_slot']:a['root_slot']+1] for a in attempts],0) for k in (0,1))
        first=[]
        example=self.encoding_cache[attempts[0]['batch']]
        for value,ref in zip(packed,example):
            if ref.stride(-1)==0 and value.shape[-1]>1: value=value[...,:1].contiguous().expand_as(value)
            first.append(value)
        return tuple(first),tuple(v.contiguous() for v in packed)


class HistoricalObserver:
    def __init__(self,config,work,attempts):
        self.config=config;self.attempts=attempts;self.data={};self.maps={};self.maxima=np.zeros(50)
        self.cat_errors={k:np.zeros(50,dtype=np.int64) for k in ('atomics','bonds','charges')}
        self.steps=[];self.first_difference=None
        for b in {a['batch'] for a in attempts}:
            folder=Path(config['remote_dataset'])/config['sources'][str(b)]['relative']
            with np.load(folder/'trajectory.npz') as z:
                self.data[b]={k:z[k].copy() for k in ('proposal_coords','proposal_atomics','proposal_bonds','proposal_charges')}
            with np.load(Path(work)/'tapes'/f'batch_{b:03d}.npz') as z: self.maps[b]=z['before'].copy()
        for a in attempts:
            if identity(self.maps[a['batch']][:,a['final_slot']].tolist())!=a['path_sha256']:
                raise ValueError('Tape ancestor mapping differs from frozen final path')

    def __call__(self,step,curr,predicted):
        xyz=curr['coords'].detach().cpu().numpy()
        cats={k:curr[k].argmax(-1).cpu().numpy() for k in self.cat_errors}
        max_error=0.;ncat=0
        for i,a in enumerate(self.attempts):
            b=a['batch'];parent=int(self.maps[b][step,a['final_slot']]);data=self.data[b]
            error=float(np.max(np.abs(xyz[i]-data['proposal_coords'][step,parent])))*self.config['_coord_scale']
            self.maxima[i]=max(self.maxima[i],error);max_error=max(max_error,error)
            for k in self.cat_errors:
                count=int(np.count_nonzero(cats[k][i]!=data['proposal_'+k][step,parent]))
                self.cat_errors[k][i]+=count;ncat+=count
            if self.first_difference is None and (error>self.config['coordinate_tolerance_A'] or any(self.cat_errors[k][i] for k in self.cat_errors)):
                self.first_difference={'step':step,'id':a['id'],'max_coordinate_error_A':error,
                                       'categorical_errors':{k:int(v[i]) for k,v in self.cat_errors.items()}}
        self.steps.append({'step':step,'max_coordinate_error_A':max_error,'categorical_differences':ncat})

    def summary(self):
        return {'steps':self.steps,'max_coordinate_error_A':float(self.maxima.max()),
                'per_path_max_error_A':self.maxima.tolist(),
                'categorical_differences':{k:int(v.sum()) for k,v in self.cat_errors.items()},
                'first_difference':self.first_difference}


def final_equivalence(config,attempts,output,scores):
    cache={};errors=[];diffs={k:0 for k in ('atomics','bonds','charges')};score_errors=[]
    for i,a in enumerate(attempts):
        b=a['batch'];slot=a['final_slot']
        if b not in cache:
            cache[b]=load_state(Path(config['remote_dataset'])/config['sources'][str(b)]['relative']/'final_prediction.pt.gz')
        original=cache[b]
        errors.append(float((output['coords'][i]-original['coords'][slot]).abs().max()))
        for k in diffs: diffs[k]+=int((output[k][i].argmax(-1)!=original[k][slot].argmax(-1)).sum())
        score_errors.append(abs(float(scores[i])-a['historical_pic50']))
    return {'terminal_max_coordinate_error_A':max(errors),'terminal_categorical_differences':diffs,
            'terminal_max_pic50_error':max(score_errors),
            'passed':max(errors)<=config['coordinate_tolerance_A'] and max(score_errors)<=config['affinity_tolerance'] and not any(diffs.values())}


def r15_equivalence(repo,config,device='cuda'):
    from .common import condition_program,program_identity
    from evomolsteer.local_motif.reward import MotifReward
    from evomolsteer.final_guidance.native import Control
    specs=[next(c for c in config['conditions'] if c['id']==name) for name in ('motif_R15_B50','motif_R15_v2')]
    a,t=condition_program(repo,specs[0]);b,u=condition_program(repo,specs[1])
    if program_identity(a)!=program_identity(b) or identity(t)!=identity(u): raise ValueError('R15 numeric signature differs')
    # Match real teacher scale; use a dedicated generator without changing the run RNG.
    rewards=[MotifReward(a,t),MotifReward(b,u)]
    target=np.asarray(t['coords_A']);g=torch.Generator(device=device).manual_seed(42)
    z=torch.tensor(target,device=device,dtype=torch.float32)[None].repeat(50,1,1)
    z=(z+torch.randn(z.shape,generator=g,device=device)*.1).requires_grad_(True)
    logits=torch.randn((50,len(target),15),device=device,generator=g)
    mask=torch.ones(z.shape[:2],device=device,dtype=torch.bool)
    outputs=[];grads=[];details=[]
    for reward in rewards:
        value,detail=reward.evaluate(z,logits,mask);grad=torch.autograd.grad(value.sum(),z,retain_graph=True)[0]
        outputs.append(value.detach());grads.append(grad.detach());details.append(detail)
    model=SimpleNamespace(coord_scale=1.,integrator=SimpleNamespace(use_cosine_scheduler=False))
    states=[];totals=[]
    for p,grad,value,detail in zip((a,b),grads,outputs,details):
        control=Control(model,None,None,torch.zeros((50,3),device=device))
        control.program=p;control.before=z.detach()-.01;control.t=.1;control.dt=.01
        control.cached=(grad,z.detach()+.01,.01,detail['direct_mask'],value,grad)
        curr={'coords':z.detach().clone(),'mask':mask}
        states.append(Control.after(control,curr,10)['coords']);totals.append(control.total)
    checks={'numeric_identity':True,'reward_equal':torch.equal(*outputs),'gradient_equal':torch.equal(*grads),
            'injected_state_equal':torch.equal(*states),'cumulative_dose_equal':torch.equal(*totals)}
    if not all(checks.values()): raise ValueError('R15 reward/gradient/controller equivalence failed')
    return {'passed':True,**checks,'numeric_program_sha256':program_identity(a)}
