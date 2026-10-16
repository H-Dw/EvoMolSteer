"""Local ancestral targets with explicit clocks, routing, sensing and real dose."""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from evomolsteer.final_guidance.native import Control
from evomolsteer.final_guidance.reward import bridge
from .bank import lookup


def temporal_gain(p,t,dt):
    a,b=p['window'];u=np.clip((t+dt*.5-a)/(b-a),0,1);name=p['dose_schedule']
    raw=1. if name=='constant' else .5+u if name=='late' else 1.5-u if name=='early' else 6*u*(1-u)
    return float(raw/p.get('schedule_normalization',1.))


def horizon(p,t):
    a,b=p['learning_window'];kind=p['clock_policy']
    if kind=='fixed':return b
    if kind in ('current','endpoint_prediction'):return t
    width=b-a;delta=p['lookahead_fraction']*width
    if kind=='decaying_lookahead':delta*=max(0.,1-(t-a)/width)
    if kind not in ('lookahead','decaying_lookahead'):raise ValueError('Unknown ancestor clock policy')
    return min(b,t+delta)


def attainment_gain(p,error,components):
    residual=error.square()
    for name,key in [('typed','typed_weight'),('charge','charge_weight')]:
        residual=residual+p[key]*(-components[name].detach()).clamp_min(0)
    return (residual/(residual+p['target_band_A']**2)).sqrt()


class AncestorReward:
    def __init__(self,p,teacher):
        self.p=p;self.teacher=teacher;self.modes=teacher['modes'];self.specs=p['mode_specs']
        a,b=p['learning_window'];g0,g1=p['window']
        if not 0<=a<=g0<g1<=b<1:raise ValueError('Invalid learning/control domain')
        if p['smc'] or p['affinity_head_gradient'] or not p['coordinate_updates_only']:raise ValueError('Unsupported intervention')
        if p['routing'] not in ('single','static','compatible','persistent','soft'):raise ValueError('Unknown seed routing')
        if p['routing']=='soft':raise ValueError('Soft multi-mode support unions need a separate validated adapter')
        if p['sensing'] not in ('detached','full') or p['typed_channels'] not in ('all','NC','NO') or p['type_target'] not in ('predicted','terminal'):raise ValueError('Unknown chemical sensing semantics')
        if p['clock_policy'] not in ('fixed','current','endpoint_prediction','lookahead','decaying_lookahead'):raise ValueError('Invalid clock')
        if p['operator'] not in ('pointwise','point_band','pair_distance','pocket_distance','orientation','selection_advantage'):raise ValueError('Invalid operator')
        for s in self.specs:
            m=self.modes[s['template_id']];k=s['atom_slots']
            if not 2<=len(set(k))==len(k)<len(m['coords_A']):raise ValueError('Proper unique local support required')
            if p['operator']=='orientation' and not s['orientation_triplets']:raise ValueError('Missing orientation support')
            for trip in s['orientation_triplets']:
                if not set(trip)<=set(k):raise ValueError('Direction neighbors escaped declared support')
        if not 0<p['dose']['pointcloud_delta_A'] or p['field_sigma_A']<=0:raise ValueError('Invalid spatial scale')
        self.previous=None;self.last_switch=None

    def active(self,t,dt):return dt>0 and t>=self.p['window'][0]-2e-6 and t+dt<=self.p['window'][1]+2e-6

    def evaluate_mode(self,current,endpoint,atom_logits,charge_logits,mask,t,index,assignment=None):
        p=self.p;s=self.specs[index];m=self.modes[s['template_id']];slots=s['atom_slots'];h=horizon(p,t)
        if p['clock_policy']=='endpoint_prediction':
            z=endpoint;target=lookup(m,h,'endpoint_A')[slots]
        else:
            z=current if p['clock_policy']=='current' else bridge(current,endpoint,t,h)
            target=lookup(m,h)[slots]
        q=z.new_tensor(target);logp=atom_logits.log_softmax(-1)
        labels=[m['atom_vocabulary'][m['native_elements'][j]] for j in slots]
        if assignment is None:
            assignments=[]
            for cloud,valid,probs in zip(z.detach().cpu().numpy(),mask.cpu().numpy(),logp.detach().cpu().numpy()):
                ids=np.flatnonzero(valid);cost=np.square(cloud[ids,None]-target).sum(-1)
                cost-=p['assignment_element_cost']*probs[ids][:,labels]
                rr,cc=linear_sum_assignment(cost);assignments.append(ids[rr[np.argsort(cc)]])
            assignment=torch.as_tensor(np.asarray(assignments),device=z.device,dtype=torch.long)
        points=z.gather(1,assignment[...,None].expand(-1,-1,3));d=points-q;distance=d.norm(dim=-1);error=distance.square().mean(1).sqrt()
        delta=p['dose']['pointcloud_delta_A'];band=p['target_band_A'];operator=p['operator']
        if operator=='pointwise':value=(-(delta**2)*(torch.sqrt(1+distance.square()/delta**2)-1)).mean(1)
        elif operator=='point_band':value=-(distance-band).clamp_min(0).square().mean(1)
        elif operator=='pair_distance':
            ij=torch.triu_indices(len(slots),len(slots),offset=1,device=z.device)
            dd=(points[:,ij[0]]-points[:,ij[1]]).norm(dim=-1)-(q[ij[0]]-q[ij[1]]).norm(dim=-1)
            value=-(dd.abs()-band).clamp_min(0).square().mean(1)
        elif operator=='pocket_distance':
            sites=z.new_tensor([site['coords_A'] for site in s['pocket_sites']])
            dd=(points[:,:,None]-sites).norm(dim=-1)-(q[:,None]-sites).norm(dim=-1)
            value=-(dd.abs()-band).clamp_min(0).square().mean((1,2))
        elif operator=='orientation':
            local={j:i for i,j in enumerate(slots)};values=[]
            for center,a,b in s['orientation_triplets']:
                i,j,k=[local[v] for v in (center,a,b)];v=points[:,[j,k]]-points[:,i,None];qq=q[[j,k]]-q[i]
                v=v/v.norm(dim=-1,keepdim=True).clamp_min(1e-8);qq=qq/qq.norm(dim=-1,keepdim=True).clamp_min(1e-8)
                values.append((1-(v*qq).sum(-1)).square().mean(1))
            value=-torch.stack(values,1).mean(1)
        else:
            kernel=torch.exp(-d.square().sum(-1)/(2*p['field_sigma_A']**2))
            phi=torch.cat([kernel.mean(1)[:,None],(kernel[...,None]*d).mean(1)],1)
            profile=s['selection_profile'];times=np.asarray(profile['times'])
            def at(key):return z.new_tensor([np.interp(h,times,np.asarray(profile[key])[:,j]) for j in range(4)])
            mu=at('focal');bg=at('background_mean');sd=at('background_sd').clamp_min(z.new_tensor(p['density_scale_floor']))
            # Low-dimensional explicit density ratio, not a trained predictor or calibrated probability.
            value=(-.5*((phi-mu)/sd).square()+.5*((phi-bg)/sd).square()).mean(1).clamp(-p['advantage_cap'],p['advantage_cap'])
        typed=z.new_zeros(len(z));charge_term=z.new_zeros(len(z));sigma=p['field_sigma_A']
        # Chemical fields refer to ENDPOINT geometry/probabilities, separate from raw/bridge terms.
        if p['typed_weight'] or p['charge_weight']:
            ypoints=endpoint.gather(1,assignment[...,None].expand(-1,-1,3))
            tq=z.new_tensor(lookup(m,h,'endpoint_A')[slots]);kern=torch.exp(-(ypoints[:,:,None]-tq[None,None]).square().sum(-1)/(2*sigma*sigma))
            tkern=torch.exp(-(tq[:,None]-tq[None]).square().sum(-1)/(2*sigma*sigma))
            probs=logp.exp().gather(1,assignment[...,None].expand(-1,-1,logp.shape[-1]))
            if p['sensing']=='detached':probs=probs.detach()
            if p['typed_weight']:
                targets=z.new_tensor(lookup(m,h,'element_probs')[slots]);targets=targets/targets.sum(-1,keepdim=True).clamp_min(1e-8)
                if p['type_target']=='terminal':targets=torch.nn.functional.one_hot(torch.as_tensor(labels,device=z.device),atom_logits.shape[-1]).to(z.dtype)
                if p['typed_channels']!='all':
                    ids=[m['atom_vocabulary'][e] for e in p['typed_channels']];probs=probs[...,ids];targets=targets[...,ids]
                got=torch.einsum('bij,bie->bje',kern,probs)/len(slots);want=tkern.T@targets/len(slots)
                typed=-(got-want).square().sum((1,2))*sigma*sigma
            if p['charge_weight']:
                cp=charge_logits.softmax(-1).gather(1,assignment[...,None].expand(-1,-1,charge_logits.shape[-1]))
                if p['sensing']=='detached':cp=cp.detach()
                targets=z.new_tensor(lookup(m,h,'charge_probs')[slots]);targets=targets/targets.sum(-1,keepdim=True).clamp_min(1e-8)
                vocabulary=m['charge_vocabulary'];groups=[[int(v) for q,v in vocabulary.items() if q!='<PAD>' and int(q)<0],
                    [int(v) for q,v in vocabulary.items() if q!='<PAD>' and int(q)==0],[int(v) for q,v in vocabulary.items() if q!='<PAD>' and int(q)>0]]
                gotp=torch.stack([cp[...,g].sum(-1) for g in groups],-1);wantp=torch.stack([targets[...,g].sum(-1) for g in groups],-1)
                got=torch.einsum('bij,bie->bje',kern,gotp)/len(slots);want=tkern.T@wantp/len(slots)
                charge_term=-(got-want).square().sum((1,2))*sigma*sigma
        base_value=value;value=value+p['typed_weight']*typed+p['charge_weight']*charge_term
        direct=torch.zeros_like(mask);direct.scatter_(1,assignment,True)
        return value,{'assignment':assignment,'direct_mask':direct,'error':error,'clock':h,'typed':typed,'charge':charge_term,'base':base_value,'z':z}

    def evaluate(self,current,endpoint,atom_logits,charge_logits,mask,t,forced_mode=None,assignments=None):
        results=[self.evaluate_mode(current,endpoint,atom_logits,charge_logits,mask,t,k,
            None if assignments is None else assignments[k]) for k in range(len(self.specs))]
        errors=torch.stack([d['error'].detach() for _,d in results],1);n=len(current)
        if forced_mode is not None:choice=forced_mode
        elif self.p['routing']=='single':choice=torch.zeros(n,device=current.device,dtype=torch.long)
        elif self.p['routing']=='static':choice=torch.arange(n,device=current.device)%len(self.specs)
        else:
            best=errors.argmin(1);choice=best
            if self.p['routing']=='persistent' and self.previous is not None:
                prev=self.previous;old=errors.gather(1,prev[:,None]).flatten();new=errors.gather(1,best[:,None]).flatten()
                switch=(old-new>self.p['switch_margin_A'])&(t-self.last_switch>=self.p['mode_memory_t'])
                choice=torch.where(switch,best,prev)
        if self.previous is None:self.last_switch=torch.full((n,),t,device=current.device)
        switches=torch.zeros(n,dtype=torch.bool,device=current.device) if self.previous is None else choice!=self.previous
        self.last_switch=torch.where(switches,torch.full_like(self.last_switch,t),self.last_switch);self.previous=choice.detach()
        values=torch.stack([v for v,_ in results],1).gather(1,choice[:,None]).flatten()
        selected=[]
        for i in range(n):selected.append(results[int(choice[i])][1]['assignment'][i])
        # Supports can differ by mode; boolean union is per seed's SINGLE active mode only.
        direct=torch.stack([results[int(choice[i])][1]['direct_mask'][i] for i in range(n)])
        error=errors.gather(1,choice[:,None]).flatten()
        return values,{'direct_mask':direct,'error':error,'mode':choice,'switches':switches,
            'selected_components':{name:torch.stack([d[name] for _,d in results],1).gather(1,choice[:,None]).flatten() for name in ('base','typed','charge')},
            'all_assignments':[d['assignment'] for _,d in results],'assignments_selected':[v.detach().cpu().tolist() for v in selected],
            'clocks':[d['clock'] for _,d in results],'typed_mean':float(torch.stack([d['typed'].detach() for _,d in results]).mean()),
            'charge_mean':float(torch.stack([d['charge'].detach() for _,d in results]).mean())}


class AncestorControl(Control):
    def __init__(self,model,program,teacher,com,capture_time=None):
        super().__init__(model,None,teacher,com,capture_time);self.program=program;self.reward=AncestorReward(program,teacher)
    def forward(self,curr,pocket,times,cond,equis,invs,i,dt):
        self.cached=None;self.t=float(times[0][0]);self.dt=float(dt);self.before=curr['coords'].detach().clone()
        kwargs=dict(cond_batch=cond,pocket_equis=equis,pocket_invs=invs,training=False)
        if not self.reward.active(self.t,self.dt):return self.model(curr,pocket,times,**kwargs)
        with torch.enable_grad():
            x=self.before.clone().requires_grad_(True);out=self.model(dict(curr,coords=x),pocket,times,**kwargs)
            for v in out.get('affinity',{}).values():
                if torch.is_tensor(v) and v.requires_grad:
                    def denied(g):raise RuntimeError('Affinity derivative forbidden in ancestral guidance')
                    v.register_hook(denied)
            pred,_=self.model._get_predictions(out);current=x*self.model.coord_scale+self.com[:,None]
            endpoint=pred['coords']*self.model.coord_scale+self.com[:,None]
            value,detail=self.reward.evaluate(current,endpoint,out['atomics'],out['charges'],curr['mask'].bool(),self.t)
            probes={};active=self.program['expected_active_steps'];probe_steps={active[0],active[len(active)//2],active[-1]}
            if i in probe_steps:
                for name,key in [('typed','typed_weight'),('charge','charge_weight')]:
                    if self.program[key]>0:
                        component=detail['selected_components'][name]*self.program[key]
                        probes[name]=torch.autograd.grad(component.sum(),x,retain_graph=True)[0].detach()
            gradient=torch.autograd.grad(value.sum(),x)[0]
        if not torch.isfinite(gradient).all():raise ValueError('Nonfinite ancestral coordinate gradient')
        error=detail['error'];gain=torch.ones_like(error)*temporal_gain(self.program,self.t,self.dt)
        if self.program['attainment_response']:gain*=attainment_gain(self.program,error,detail['selected_components'])
        self.dose_gain=gain.detach()
        self.extra={'clock_policy':self.program['clock_policy'],'actual_target_clocks':detail['clocks'],
            'mode_indices':detail['mode'].cpu().tolist(),'mode_switches':int(detail['switches'].sum()),
            'target_error_per_seed_A':error.cpu().tolist(),'dose_gain_per_seed':self.dose_gain.cpu().tolist(),
            'direct_support_per_seed':detail['direct_mask'].sum(1).cpu().tolist(),
            'typed_value_mean':detail['typed_mean'],'charge_value_mean':detail['charge_mean'],
            'actual_local_assignments':detail['assignments_selected'],'affinity_vjp':False,'additional_affinity_calls':0,
            'derivative_scope':'direct_current' if self.program['clock_policy']=='current' and not (self.program['typed_weight'] or self.program['charge_weight']) else 'live_generation_coordinate_VJP'}
        previous=getattr(self,'previous_assignments',None);new=detail['assignments_selected']
        switches=0. if previous is None else float(np.mean([1. if len(a)!=len(b) else np.mean(np.asarray(a)!=b) for a,b in zip(new,previous)]))
        self.previous_assignments=[list(v) for v in new]
        self.extra.update(point_rms_A=float(error.mean()),target_error_mean=float(error.mean()),
            dose_gain_mean=float(self.dose_gain.mean()),assignment_switch_fraction=switches)
        for name,g in probes.items():
            mask=curr['mask'].bool();den=mask.sum(1).clamp_min(1)
            norm=lambda a:((a.square()*mask[...,None]).sum((1,2))/den).sqrt()
            self.extra[name+'_gradient_rms_per_seed']=norm(g).cpu().tolist()
            self.extra[name+'_gradient_fraction_of_total']=(norm(g)/norm(gradient.detach()).clamp_min(1e-12)).cpu().tolist()
            self.extra[name+'_gradient_cosine_with_total']=((g*gradient.detach()).sum((1,2))/(g.square().sum((1,2))*gradient.detach().square().sum((1,2))).sqrt().clamp_min(1e-12)).cpu().tolist()
        self.extra['additional_generation_forwards']=0;self.extra['diagnostic_component_VJPs']=len(probes)
        # gz is used only for direct-support telemetry in the legacy integrator adapter.
        self.cached=(gradient.detach(),pred['coords'].detach(),self.dt,detail['direct_mask'],value.detach(),torch.zeros_like(gradient))
        return {k:({a:b.detach() if torch.is_tensor(b) else b for a,b in v.items()} if isinstance(v,dict) else v.detach() if torch.is_tensor(v) else v) for k,v in out.items()}
    def after(self,curr,i):
        enabled=self.cached is not None;curr=super().after(curr,i)
        if enabled:
            self.rows[-1].update(self.extra)
            self.rows[-1]['direct_unmatched_gradient_max']=None
            self.rows[-1]['direct_output_gradient_diagnostic']='not_computed_for_multiple_output_representations'
        if self.teacher_snapshot is not None and abs(self.t+self.dt-self.capture_time)<2e-6:
            self.teacher_snapshot['atomic_indices']=curr['atomics'].argmax(-1).cpu().tolist()
        return curr
