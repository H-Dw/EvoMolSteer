"""Trusted differentiable compiler for a small declarative reward language.

No generated source is evaluated. Gradients are with respect to endpoint world
coordinates. Integrating them into a generator requires a separately tested chain.
"""
from pathlib import Path
import copy
import json
import numpy as np
import torch
from torch.nn import functional as F
from jsonschema import Draft202012Validator
from .contracts import PROGRAM
from .io import read_json,write_json

class NotApplicable(ValueError):pass

class RewardProgram:
    def __init__(self,program,catalog):
        program=copy.deepcopy(program)
        self.ignored_legacy_constraints=[]
        if 'max_pair_distance_change_A' in program['constraints']:
            program['constraints'].pop('max_pair_distance_change_A')
            self.ignored_legacy_constraints.append('max_pair_distance_change_A')
        json.dumps(program,allow_nan=False)
        Draft202012Validator(PROGRAM).validate(program)
        self.program,self.catalog=program,catalog
        window=program['selection_window']
        if not 0<=window['start']<window['end']<=1:raise ValueError('Invalid selection window')
        for term in program['terms']:
            spec=catalog['features'].get(term['feature'])
            if not spec or not spec['differentiable_supported']:raise ValueError('Unsupported feature')
            if term['lower']>term['upper'] or term['stage_start']>=term['stage_end']:raise ValueError('Invalid reward interval')
            if not window['start']<=term['stage_start']<term['stage_end']<=window['end']:raise ValueError('Reward term outside selection window')

    def observable(self,feature,x,atomics,mask):
        spec=self.catalog['features'][feature];eligible=mask.bool()
        if spec.get('elements'):
            allowed=torch.tensor([self.catalog['atom_vocabulary'][s] for s in spec['elements']],device=x.device)
            eligible=eligible&torch.isin(atomics,allowed)
        if not bool(eligible.any(-1).all()):raise NotApplicable('No eligible atoms for '+feature)
        if spec['kind']=='radius_gyration':
            n=eligible.sum(-1);center=(x*eligible[...,None]).sum(1)/n[:,None]
            return torch.sqrt(((x-center[:,None,:]).square().sum(-1)*eligible).sum(-1)/n+1e-20)
        points=torch.tensor(self.catalog['regions'][spec['region']]['points_A'],device=x.device,dtype=x.dtype)
        dist=torch.sqrt((x[:,:,None,:]-points[None,None,:,:]).square().sum(-1)+1e-20)
        tau=spec['temperature_A']; logits=(-dist/tau).masked_fill(~eligible[...,None],-torch.inf)
        return -tau*(torch.logsumexp(logits.flatten(1),dim=1)-torch.log(eligible.sum(-1).to(x.dtype)*len(points)))

    def __call__(self,x,atomics,mask,time):
        t=torch.as_tensor(time,dtype=x.dtype,device=x.device)
        if not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(t).all()):raise NotApplicable('Nonfinite coordinates/time')
        reward=x.sum((1,2))*0;components={}
        for term in self.program['terms']:
            value=self.observable(term['feature'],x,atomics,mask)
            lo=(term['lower']-value)/term['scale'];hi=(value-term['upper'])/term['scale']
            penalty=F.softplus(lo,beta=10).square()+F.softplus(hi,beta=10).square()
            gate=torch.sigmoid((t-term['stage_start'])/term['gate_width'])*torch.sigmoid((term['stage_end']-t)/term['gate_width'])
            # Exact time support. Differentiability is with respect to coordinates;
            # score time is an external control variable, not an optimized variable.
            boundary_time=torch.round(t*1e6)/1e6
            right=(boundary_time<=term['stage_end']) if term['stage_end']==self.program['selection_window']['end'] else (boundary_time<term['stage_end'])
            gate=gate*((boundary_time>=term['stage_start'])&right).to(x.dtype)
            component=-term['weight']*gate*penalty;reward=reward+component
            components[term['term_id']]={'observable':value,'reward':component,'gate':gate}
        return reward,components

    def projected_gradient(self,x,atomics,mask,time,editable):
        xx=x.detach().clone().requires_grad_(True);reward,_=self(xx,atomics,mask,time)
        grad=torch.autograd.grad(reward.sum(),xx)[0]
        return grad*(mask.bool()&editable.bool())[...,None]

    def guarded_step(self,x,atomics,mask,time,editable,step_size=.01):
        if not np.isfinite(step_size) or step_size<=0:raise ValueError('Positive finite step size required')
        gradient=self.projected_gradient(x,atomics,mask,time,editable);c=self.program['constraints']
        delta=step_size*gradient
        norms=delta.norm(dim=-1,keepdim=True);delta=delta*torch.clamp(c['max_atom_displacement_A']/norms.clamp_min(1e-20),max=1)
        count=(editable.bool()&mask.bool()).sum(-1).clamp_min(1)
        rms=torch.sqrt(delta.square().sum((1,2))/count);delta=delta*torch.clamp(c['max_rms_displacement_A']/rms.clamp_min(1e-20),max=1)[:,None,None]
        proposed=x+delta; accepted=torch.ones(len(x),dtype=torch.bool,device=x.device)
        pm=mask[:,:,None].bool()&mask[:,None,:].bool()
        pair_change=(torch.cdist(proposed,proposed)-torch.cdist(x,x)).abs().masked_fill(~pm,0).amax((1,2))
        points=torch.tensor([p for r in self.catalog['regions'].values() for p in r['points_A']],dtype=x.dtype,device=x.device)
        new_counts=torch.zeros(len(x),dtype=torch.int64,device=x.device)
        if len(points):
            before=torch.cdist(x,points);after=torch.cdist(proposed,points)
            new_counts=((after<c['clash_threshold_A'])&(before>=c['clash_threshold_A'])&mask[...,None].bool()).sum((1,2))
            accepted &= new_counts<=c['max_new_clash_pairs']
        accepted &= torch.isfinite(proposed).all((1,2))
        # A projected/clipped finite step need not increase the objective.
        before_reward=self(x,atomics,mask,time)[0]
        after_reward=self(proposed,atomics,mask,time)[0]
        accepted &= torch.isfinite(after_reward)&(after_reward>=before_reward-1e-12)
        result=torch.where(accepted[:,None,None],proposed,x)
        return result,{'accepted':accepted,'pair_distance_change_A':pair_change,'new_clash_pairs':new_counts,
                       'ignored_legacy_constraints':self.ignored_legacy_constraints,
                       'gradient_norm':gradient.norm(dim=(1,2)),'proposed_reward_delta':after_reward-before_reward,
                       'max_atom_displacement_A':(result-x).norm(dim=-1).amax(1)}

def validate_on_fixture(analysis,program_path):
    analysis=Path(analysis);program=read_json(program_path);catalog=read_json(analysis/'feature_catalog.json')
    runtime=RewardProgram(program,catalog);manifest=read_json(analysis/'ingest_manifest.json')
    root=Path(manifest['source_root']);cfg=read_json(analysis/'config.json');camp=root/'results'/cfg['campaign']
    if not program['terms']:
        write_json(analysis/'agents/reward_validation.json',{'status':'deferred_empty_program','live_generation_tested':False});return
    first=program['terms'][0];time=(first['stage_start']+first['stage_end'])/2
    batch=min(cfg['discovery_batches']);path=camp/'joint'/f'batch_{batch:03d}'/'trajectory.npz'
    z=np.load(path);step=int(np.argmin(np.abs(z['score_time'][:,0]-time)))
    frame=read_json(camp/f'frame_batch_{batch:03d}.json');scale=read_json(camp/'config.json')['coord_scale']
    x=torch.tensor(z['predicted_coords'][step,:1]*scale+np.array(frame['target_com'])[:1,None,:],dtype=torch.float64,requires_grad=True)
    atom=torch.tensor(z['predicted_atomics'][step,:1],dtype=torch.long);mask=torch.tensor(z['mask'][step,:1],dtype=torch.bool)
    value,components=runtime(x,atom,mask,time);gradient=torch.autograd.grad(value.sum(),x)[0]
    flat=gradient.flatten();indices=torch.argsort(flat.abs(),descending=True)[:3];checks=[]
    eps=1e-5
    for ix in indices.tolist():
        plus=x.detach().clone();minus=x.detach().clone();plus.view(-1)[ix]+=eps;minus.view(-1)[ix]-=eps
        numeric=((runtime(plus,atom,mask,time)[0]-runtime(minus,atom,mask,time)[0])/(2*eps)).sum().item()
        analytic=flat[ix].item();error=abs(numeric-analytic)
        checks.append({'flat_index':ix,'analytic':analytic,'finite_difference':numeric,'absolute_error':error})
        if error>1e-5+1e-4*abs(analytic):raise AssertionError('Finite-difference disagreement')
    editable=mask.clone();editable[:,0]=False
    projected=runtime.projected_gradient(x,atom,mask,time,editable)
    assert torch.equal(projected[:,0],torch.zeros_like(projected[:,0]))
    proposed,audit=runtime.guarded_step(x.detach(),atom,mask,time,editable)
    assert torch.equal(proposed[:,0],x.detach()[:,0])
    rotation=torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]],dtype=x.dtype);translation=torch.tensor([2.,-3.,1.],dtype=x.dtype)
    transformed=copy.deepcopy(catalog)
    for r in transformed['regions'].values():r['points_A']=(np.array(r['points_A'])@rotation.numpy()+translation.numpy()).tolist()
    rotated=RewardProgram(program,transformed)(x@rotation+translation,atom,mask,time)[0]
    rigid_error=float((rotated-value).abs().max().detach());assert rigid_error<1e-8
    perm=torch.arange(x.shape[1]-1,-1,-1)
    perm_error=float((runtime(x[:,perm],atom[:,perm],mask[:,perm],time)[0]-value).abs().max().detach());assert perm_error<1e-8
    def serialize(v):return v.detach().cpu().tolist() if torch.is_tensor(v) else v
    result={'status':'passed_offline_numerical_checks','source_trajectory':str(path),'step':step,'gate_time':time,
            'finite_difference':checks,'rigid_invariance_error':rigid_error,'atom_permutation_error':perm_error,
            'fixed_atom_preserved':True,'reward_before':value.detach().tolist(),
            'reward_after_guarded_step':runtime(proposed,atom,mask,time)[0].detach().tolist(),
            'step_audit':{k:serialize(v) for k,v in audit.items()},'live_generation_tested':False,
            'limitation':'A saved-endpoint gradient test is not a generator-guidance or affinity improvement experiment.'}
    write_json(analysis/'agents/reward_validation.json',result)
    return result
