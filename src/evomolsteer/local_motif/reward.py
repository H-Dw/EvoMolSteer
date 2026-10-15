"""Local coordinate potentials; no mandatory N/C labels or affinity derivatives."""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from evomolsteer.final_guidance.native import Control
from evomolsteer.final_guidance.reward import bridge


OPERATORS=('pointwise','point_band','pocket_distance','orientation','pair_distance')


class MotifReward:
    def __init__(self,p,teacher):
        self.program=p;self.teacher=teacher;self.slots=p['atom_slots'];self.local={s:i for i,s in enumerate(self.slots)}
        if p['family']!='local_motif_v2' or p['operator'] not in OPERATORS:raise ValueError('Unknown local operator')
        if not 2<=len(set(self.slots))==len(self.slots)<len(teacher['coords_A']):raise ValueError('Proper unique local support required')
        if p['teacher_time']!=teacher['time'] or not 0<=p['window'][0]<p['window'][1]<=teacher['time']<1:raise ValueError('Incompatible clocks')
        if p['coordinate_updates_only'] is not True or p['smc'] or p['affinity_head_gradient']:raise ValueError('Invalid control scope')
        if p['representation'] not in ('bridge','endpoint'):raise ValueError('Unimplemented target representation')
        self.target=np.asarray(teacher['coords_A'] if p['representation']=='bridge' else teacher['terminal_coords_A'])[self.slots]
        self.labels=[teacher['atom_vocabulary'][teacher['native_elements'][i]] for i in self.slots]
        if p['operator']=='orientation' and not p['orientation_triplets']:raise ValueError('Orientation needs observed local neighbors')
        for group in p['orientation_triplets']:
            if not set(group)<=set(self.slots):raise ValueError('Orientation escaped local support')
        if p['operator']=='pocket_distance' and not p['pocket_sites']:raise ValueError('Pocket field needs fixed sites')
        if p['controller'] not in ('unit_rms','bounded'):raise ValueError('Unknown control response')

    def active(self,t,dt):
        a,b=self.program['window'];return t>=a-2e-6 and t+dt<=b+2e-6 and dt>0

    def evaluate(self,z,logits,mask,assignment=None):
        p=self.program;target=z.new_tensor(self.target)
        if assignment is None:
            assignments=[]
            with torch.no_grad():
                probabilities=logits.log_softmax(-1).detach().cpu().numpy()
                for b,(cloud,m) in enumerate(zip(z.detach().cpu().numpy(),mask.cpu().numpy())):
                    ids=np.flatnonzero(m)
                    if len(ids)<=len(self.slots):raise ValueError('Local support covers all generated atoms')
                    cost=((cloud[ids,None]-self.target[None])**2).sum(-1)
                    cost-=p['assignment_element_cost']*probabilities[b,ids][:,self.labels]
                    rr,cc=linear_sum_assignment(cost);assignments.append(ids[rr[np.argsort(cc)]])
            assignment=torch.as_tensor(np.asarray(assignments),device=z.device,dtype=torch.long)
        points=z.gather(1,assignment[...,None].expand(-1,-1,3));delta=points-target
        band=p['band_multiplier']*p['base_band_A'];operator=p['operator']
        if operator=='pointwise':
            d=p['dose']['pointcloud_delta_A'];q=delta.square().sum(-1)
            value=(-d*d*(torch.sqrt(1+q/(d*d))-1)).mean(1);error=q.mean(1).sqrt()
        elif operator=='point_band':
            distance=delta.norm(dim=-1);excess=(distance-band).clamp_min(0)
            value=-(excess**2).mean(1);error=excess.mean(1)
        elif operator=='pocket_distance':
            sites=z.new_tensor([s['coords_A'] for s in p['pocket_sites']])
            distance=(points[:,:,None]-sites[None,None]).norm(dim=-1)
            desired=(target[:,None]-sites[None]).norm(dim=-1)
            excess=((distance-desired).abs()-band).clamp_min(0)
            value=-(excess**2).mean((1,2));error=excess.mean((1,2))
        elif operator=='pair_distance':
            pair=torch.triu_indices(len(self.slots),len(self.slots),offset=1,device=z.device)
            d=(points[:,pair[0]]-points[:,pair[1]]).norm(dim=-1)
            desired=(target[pair[0]]-target[pair[1]]).norm(dim=-1)
            excess=((d-desired).abs()-band).clamp_min(0)
            value=-(excess**2).mean(1);error=excess.mean(1)
        else:
            errors=[]
            for center,a,b in p['orientation_triplets']:
                i,j,k=[self.local[s] for s in (center,a,b)]
                v=points[:,[j,k]]-points[:,i,None];q=target[[j,k]]-target[i]
                v=v/v.norm(dim=-1,keepdim=True).clamp_min(1e-8);q=q/q.norm(dim=-1,keepdim=True).clamp_min(1e-8)
                # Bond rays are directed; avoids arbitrary plane-normal signs.
                errors.append((1-(v*q).sum(-1)-p['orientation_band']).clamp_min(0).square().mean(1))
            error=torch.stack(errors,1).mean(1);value=-error
        direct=torch.zeros_like(mask,dtype=torch.bool);direct.scatter_(1,assignment,True)
        return value,{'assignment':assignment,'direct_mask':direct,'target_error':error,
            'point_rms_A':delta.square().sum(-1).mean(1).sqrt()}


class MotifControl(Control):
    def __init__(self,model,program,teacher,com,capture_time=None):
        super().__init__(model,None,teacher,com,capture_time);self.program=program;self.reward=MotifReward(program,teacher)
        self.first_gradient_norm=None

    def forward(self,curr,pocket,times,cond,equis,invs,i,dt):
        self.cached=None;self.t=float(times[0][0]);self.dt=float(dt);self.before=curr['coords'].detach().clone()
        kwargs=dict(cond_batch=cond,pocket_equis=equis,pocket_invs=invs,training=False)
        if not self.reward.active(self.t,self.dt):return self.model(curr,pocket,times,**kwargs)
        with torch.enable_grad():
            x=self.before.clone().requires_grad_(True);out=self.model(dict(curr,coords=x),pocket,times,**kwargs)
            if 'affinity' in out:
                def denied(g):raise RuntimeError('Affinity entered local motif derivative')
                for v in out['affinity'].values():
                    if torch.is_tensor(v) and v.requires_grad:v.register_hook(denied)
            pred,_=self.model._get_predictions(out)
            current=x*self.model.coord_scale+self.com[:,None];endpoint=pred['coords']*self.model.coord_scale+self.com[:,None]
            z=bridge(current,endpoint,self.t,self.reward.teacher['time']) if self.program['representation']=='bridge' else endpoint
            value,detail=self.reward.evaluate(z,out['atomics'],curr['mask'].bool())
            gx,gz=torch.autograd.grad(value.sum(),(x,z))
        if not torch.isfinite(gx).all() or not torch.isfinite(value).all():raise ValueError('Nonfinite true FLOWR gradient')
        mask=curr['mask'].bool();norm=((gx.square()*mask[...,None]).sum((1,2))/mask.sum(1)).sqrt()
        if self.first_gradient_norm is None:self.first_gradient_norm=norm.detach().clamp_min(1e-10)
        floor=self.first_gradient_norm*self.program['gradient_floor_multiplier']
        self.dose_gain=(norm/torch.sqrt(norm.square()+floor.square())).detach() if self.program['controller']=='bounded' else torch.ones_like(norm)
        previous=getattr(self,'previous_assignment',None)
        self.extra={'operator':self.program['operator'],'target_error_mean':float(detail['target_error'].detach().mean()),
            'point_rms_A':float(detail['point_rms_A'].detach().mean()),'dose_gain_mean':float(self.dose_gain.mean()),
            'calibrated_gradient_floor_mean':float(floor.mean()),'raw_gradient_rms_mean':float(norm.mean()),
            'assignment_switch_fraction':None if previous is None else float((previous!=detail['assignment']).float().mean()),
            'direct_support_count':len(self.reward.slots),'affinity_vjp':False,'extra_model_forwards':0}
        self.previous_assignment=detail['assignment'].detach().clone()
        if abs(self.t+self.dt-self.program['window'][1])<2e-6:
            self.extra['boundary_generated_slots']=detail['assignment'].detach().cpu().tolist()
        self.cached=(gx.detach(),pred['coords'].detach(),self.dt,detail['direct_mask'].detach(),value.detach(),gz.detach())
        return {k:({a:b.detach() if torch.is_tensor(b) else b for a,b in v.items()} if isinstance(v,dict) else v.detach() if torch.is_tensor(v) else v) for k,v in out.items()}

    def after(self,curr,i):
        enabled=self.cached is not None;curr=super().after(curr,i)
        if enabled:
            self.rows[-1].update(self.extra)
            if 'boundary_generated_slots' in self.extra:
                ids=torch.as_tensor(self.extra['boundary_generated_slots'],device=curr['coords'].device)
                self.rows[-1]['boundary_atomic_indices']=curr['atomics'].argmax(-1).gather(1,ids).detach().cpu().tolist()
        if self.teacher_snapshot is not None and abs(self.t+self.dt-self.capture_time)<2e-6:
            self.teacher_snapshot['atomic_indices']=curr['atomics'].argmax(-1).detach().cpu().tolist()
        return curr
