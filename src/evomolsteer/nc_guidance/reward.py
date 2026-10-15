"""Sparse local geometry and decoder-N/C rewards with actual coordinate pullback."""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from evomolsteer.final_guidance.reward import bridge
from evomolsteer.final_guidance.native import Control


def normalize(v):return v/v.norm(dim=-1,keepdim=True).clamp_min(1e-8)


class NCReward:
    def __init__(self,p,teacher):
        self.program=p;self.teacher=teacher;self.slots=list(p['atom_slots']);self.typed=list(p['typed_slots'])
        if p.get('teacher_time',teacher['time'])!=teacher['time']:raise ValueError('Teacher clock mismatch')
        if not 0<=p['window'][0]<p['window'][1]<=teacher['time']<1:raise ValueError('Guidance outside learned teacher window')
        if p['family']!='local_nc_layout' or p['scope']!='local':raise ValueError('Local NC family required')
        if len(set(self.slots))!=len(self.slots) or not 2<=len(self.slots)<=p['max_local_points']<len(teacher['coords_A']):raise ValueError('Reward must use a bounded proper local subset')
        if not set(self.typed)<=set(self.slots) or len(self.typed)!=len(p['typed_elements']):raise ValueError('Typed sites escaped Analyst region')
        if set(p['typed_elements'])!={'N','C'}:raise ValueError('Explicit local N and C contrasts required')
        self.local={s:i for i,s in enumerate(self.slots)};self.ti=[self.local[s] for s in self.typed]
        self.half=np.asarray(teacher['coords_A'])[self.slots];self.final=np.asarray(teacher['terminal_coords_A'])[self.slots]
        self.labels=[teacher['atom_vocabulary'][e] for e in p['typed_elements']]
        if len(p['anchors']) and not all(set([a['slot']]+a['neighbor_slots'])<=set(self.slots) for a in p['anchors']):raise ValueError('Orientation support outside local patch')
        if p['weights']['orientation']>0 and not p['anchors']:raise ValueError('Unsupported orientation claim')
        if p['weights'].get('role',0)>0 and 'charge_vocabulary' not in teacher:raise ValueError('Chemical-role decoder vocabulary missing')
    def active(self,t,dt):
        a,b=self.program['window'];return t>=a-2e-6 and t+dt<=b+2e-6 and dt>0
    def assignment(self,z,y,logits,mask):
        p=self.program;cost_weight=p.get('assignment_type_cost',0.);result=[]
        with torch.no_grad():
            logp=logits.log_softmax(-1).cpu().numpy()
            for b,(a,endpoint,valid) in enumerate(zip(z.detach().cpu().numpy(),y.detach().cpu().numpy(),mask.cpu().numpy())):
                ids=np.flatnonzero(valid)
                if len(ids)<=len(self.slots):raise ValueError('Selected patch covers every active generated point')
                h=np.square(a[ids,None]-self.half[None]).sum(-1);f=np.square(endpoint[ids,None]-self.final[None]).sum(-1)
                mix=p['assignment_endpoint_fraction'];cost=(1-mix)*h+mix*f
                for j,label in zip(self.ti,self.labels):cost[:,j]-=cost_weight*logp[b,ids,label]
                rows,cols=linear_sum_assignment(cost);result.append(ids[rows[np.argsort(cols)]])
        return torch.as_tensor(np.asarray(result),device=z.device,dtype=torch.long)
    def terms(self,z,y,logits,mask,assignment=None,decoder=None):
        p=self.program;ids=self.assignment(z,y,logits,mask) if assignment is None else assignment
        idx=ids[...,None].expand(-1,-1,3);mh=z.gather(1,idx);mf=y.gather(1,idx)
        half_per=(mh-z.new_tensor(self.half)).square().sum(-1)
        final_per=(mf-y.new_tensor(self.final)).square().sum(-1)
        half_q=half_per.mean(1);final_q=final_per.mean(1)
        delta=p['dose']['pointcloud_delta_A'];robust=lambda q:-delta**2*(torch.sqrt(1+q/delta**2)-1)
        if p.get('geometry_mode','aggregate')=='pointwise':half_value=robust(half_per).mean(1);final_value=robust(final_per).mean(1)
        else:half_value=robust(half_q);final_value=robust(final_q)
        chosen_logits=logits.gather(1,ids[...,None].expand(-1,-1,logits.shape[-1]))[:,self.ti]
        target=torch.as_tensor(self.labels,device=z.device)[None,:,None].expand(len(z),-1,1)
        if p['type_mode']=='cn_conditional':
            cn=chosen_logits[:,:,[self.teacher['atom_vocabulary']['C'],self.teacher['atom_vocabulary']['N']]].log_softmax(-1)
            target_nc=torch.as_tensor([e=='N' for e in p['typed_elements']],device=z.device,dtype=torch.long)[None,:,None].expand(len(z),-1,1)
            per_site=cn.gather(2,target_nc).squeeze(-1)
        else:per_site=chosen_logits.log_softmax(-1).gather(2,target).squeeze(-1)
        weights=z.new_tensor(p.get('type_site_weights',[1/len(self.typed)]*len(self.typed)))
        typing=(per_site*weights).sum(1)
        values=[]
        for anchor in p['anchors']:
            i=self.local[anchor['slot']];neighbors=[self.local[j] for j in anchor['neighbor_slots']]
            ray=normalize(-normalize(mf[:,neighbors]-mf[:,i,None]).sum(1))
            vector=y.new_tensor(anchor['donor']['coords_A'])-mf[:,i];distance=vector.norm(dim=-1);u=normalize(vector)
            facing=(ray*u).sum(-1).clamp_min(0).square()
            facing*=(-u*y.new_tensor(anchor['donor']['outward'])).sum(-1).clamp_min(0).square()
            radial=torch.exp(-.5*((distance-p['anchor_distance_A'])/p['anchor_width_A']).square())
            prob=logits.softmax(-1).gather(1,ids[:,i,None,None].expand(-1,1,logits.shape[-1]))[:,0,self.teacher['atom_vocabulary']['N']]
            values.append(radial*facing*prob)
        orientation=torch.stack(values,1).mean(1) if values else y.sum((1,2))*0
        role=y.sum((1,2))*0
        if p['weights'].get('role',0)>0:
            if decoder is None:raise ValueError('Chemical-role term requires native bond/charge decoder outputs')
            bondlog=decoder['bonds'].log_softmax(-1);chargelog=decoder['charges'].log_softmax(-1);components=[]
            zero=self.teacher['charge_vocabulary']['0'];batch=torch.arange(len(y),device=y.device)
            for slot,element in zip(self.typed,p['typed_elements']):
                if element!='N':continue
                generated_i=ids[:,self.local[slot]]
                components.append(chargelog[batch,generated_i,zero])
                for j in self.teacher.get('local_adjacency',{}).get(str(slot),[]):
                    if j in self.local:components.append(bondlog[batch,generated_i,ids[:,self.local[j]],self.teacher.get('bond_vocabulary',{'aromatic':4})['aromatic']])
            if components:role=torch.stack(components,1).mean(1)
        direct=torch.zeros_like(mask,dtype=torch.bool);direct.scatter_(1,ids,True)
        pair=y.sum((1,2))*0
        if p['weights'].get('pair',0)>0:
            pairs=torch.triu_indices(len(self.slots),len(self.slots),offset=1,device=y.device)
            d=(mf[:,pairs[0]]-mf[:,pairs[1]]).norm(dim=-1)
            ref=y.new_tensor(self.final);target_d=(ref[pairs[0]]-ref[pairs[1]]).norm(dim=-1)
            pair=robust((d-target_d).square()).mean(1)
        return {'half':half_value,'endpoint':final_value,'type':typing,'orientation':orientation,'role':role,'pair':pair}, {
            'assignment':ids,'direct_mask':direct,'half_rms_A':half_q.sqrt(),'final_rms_A':final_q.sqrt(),
            'typed_probabilities':chosen_logits.softmax(-1).gather(2,target).squeeze(-1),
            'typed_correct_probability':chosen_logits.softmax(-1).gather(2,target).squeeze(-1).mean(1)}


class NCControl(Control):
    def __init__(self,model,program,teacher,com,capture_time=None):
        super().__init__(model,None,teacher,com,capture_time);self.program=program;self.reward=NCReward(program,teacher)
    def forward(self,curr,pocket,times,cond,equis,invs,i,dt):
        self.cached=None;self.t=float(times[0][0]);self.dt=float(dt);self.before=curr['coords'].detach().clone()
        kwargs=dict(cond_batch=cond,pocket_equis=equis,pocket_invs=invs,training=False)
        if not self.reward.active(self.t,self.dt):return self.model(curr,pocket,times,**kwargs)
        with torch.enable_grad():
            x=self.before.clone().requires_grad_(True);out=self.model(dict(curr,coords=x),pocket,times,**kwargs)
            if 'affinity' in out:
                def denied(g):raise RuntimeError('Affinity head entered NC gradient')
                for value in out['affinity'].values():
                    if torch.is_tensor(value) and value.requires_grad:value.register_hook(denied)
            pred,_=self.model._get_predictions(out);endpoint=pred['coords']*self.model.coord_scale+self.com[:,None]
            current=x*self.model.coord_scale+self.com[:,None];z=bridge(current,endpoint,self.t,self.reward.teacher['time'])
            terms,detail=self.reward.terms(z,endpoint,out['atomics'],curr['mask'].bool(),decoder=out);parts=[];summary={};derivatives={}
            previous=getattr(self,'previous_assignment',None)
            summary['assignment_switch_fraction']=None if previous is None else float((previous!=detail['assignment']).float().mean())
            self.previous_assignment=detail['assignment'].detach().clone()
            for element in ('N','C'):
                columns=[j for j,e in enumerate(self.program['typed_elements']) if e==element]
                summary['target_'+element+'_probability']=float(detail['typed_probabilities'][:,columns].detach().mean())
            logit_delta=None
            if self.program.get('atomic_logit_strength',0)>0:
                gl=torch.autograd.grad(terms['type'].sum(),out['atomics'],retain_graph=True)[0]
                typed_ids=detail['assignment'][:,self.reward.ti]
                typed_mask=torch.zeros_like(curr['mask'],dtype=torch.bool);typed_mask.scatter_(1,typed_ids,True)
                outside=gl.masked_fill(typed_mask[...,None],0).abs().max()
                if float(outside)>1e-10:raise ValueError('Atomic logit guidance escaped selected N/C sites')
                logit_delta=(self.program['atomic_logit_strength']*gl).clamp(-self.program['logit_max_shift'],self.program['logit_max_shift']).detach()
                after=(out['atomics'].detach()+logit_delta).softmax(-1)
                indices=typed_ids[...,None].expand(-1,-1,after.shape[-1]);labels=torch.as_tensor(self.reward.labels,device=x.device)[None,:,None].expand(len(x),-1,1)
                summary.update(local_logit_guidance=True,logit_shift_max=float(logit_delta.abs().max()),
                    unmatched_logit_change_max=float(outside),typed_probability_after_logit_step=float(after.gather(1,indices).gather(2,labels).mean()))
            active=[(k,w) for k,w in self.program['weights'].items() if w>0]
            for index,(name,weight) in enumerate(active):
                gx=torch.autograd.grad(terms[name].sum(),x,retain_graph=True)[0]
                norm=((gx.square()*curr['mask'][...,None]).sum((1,2))/curr['mask'].sum(1)).sqrt()
                summary[name+'_gradient_rms']=float(norm.detach().mean());summary[name+'_value']=float(terms[name].detach().mean())
                ramp=1.
                if name in ('type','orientation') and self.program.get('type_ramp',False):
                    start,end=self.program['window'];ramp=.1+.9*(self.t-start)/(end-start)
                if name=='type' and self.program.get('type_confidence_weighting','none')=='linear_deficit':
                    deficit=(1-detail['typed_correct_probability'].detach()).clamp(0,1)
                    ramp=ramp*deficit[:,None,None]
                component=weight*ramp*(gx/norm.clamp_min(1e-10)[:,None,None] if self.program['gradient_balance']=='unit_rms' else gx)
                component_rms=((component.square()*curr['mask'][...,None]).sum((1,2))/curr['mask'].sum(1)).sqrt()
                summary[name+'_mixture_input_rms']=float(component_rms.detach().mean())
                parts.append(component)
                derivatives[name]=gx.detach()
            keys=list(derivatives)
            for a in range(len(keys)):
                for b in range(a):
                    ga=derivatives[keys[a]].flatten(1);gb=derivatives[keys[b]].flatten(1)
                    summary['gradient_cosine_'+keys[a]+'_'+keys[b]]=float(((ga*gb).sum(1)/(ga.norm(dim=1)*gb.norm(dim=1)).clamp_min(1e-20)).mean())
            gradient=sum(parts);gz=torch.autograd.grad(terms['half'].sum(),z,allow_unused=True)[0]
        if not torch.isfinite(gradient).all():raise ValueError('Nonfinite coordinate pullback')
        # Coordinate pullback is always used; an explicit optional local logit step
        # changes decoder predictions before the native categorical integrator.
        self.extra={**summary,'selected_local_points':len(self.reward.slots),'typed_local_points':len(self.reward.typed),
            'typed_correct_probability':float(detail['typed_correct_probability'].detach().mean()),
            'endpoint_patch_rms_A':float(detail['final_rms_A'].detach().mean()),'half_patch_rms_A':float(detail['half_rms_A'].detach().mean()),
            'direct_endpoint_support_count':int(detail['direct_mask'].sum(1).max()),'atomic_output_in_coordinate_vjp':self.program['weights']['type']>0,
            'affinity_vjp':False,'extra_model_forwards':0}
        if abs(self.t+self.dt-self.program['window'][1])<2e-6:
            self.extra['boundary_typed_generated_slots']=detail['assignment'][:,self.reward.ti].detach().cpu().tolist()
        value=sum(self.program['weights'].get(k,0)*v for k,v in terms.items())
        self.cached=(gradient.detach(),pred['coords'].detach(),self.dt,detail['direct_mask'].detach(),value.detach(),gz.detach())
        detached={k:({a:b.detach() if torch.is_tensor(b) else b for a,b in v.items()} if isinstance(v,dict) else v.detach() if torch.is_tensor(v) else v) for k,v in out.items()}
        if logit_delta is not None:detached['atomics']=detached['atomics']+logit_delta
        return detached
    def after(self,curr,i):
        enabled=self.cached is not None;curr=super().after(curr,i)
        if enabled:
            self.rows[-1].update(self.extra)
            ids=self.extra.get('boundary_typed_generated_slots')
            if ids is not None:
                index=torch.as_tensor(ids,device=curr['coords'].device)
                self.rows[-1]['boundary_current_atomic_indices']=curr['atomics'].argmax(-1).gather(1,index).detach().cpu().tolist()
        if self.teacher_snapshot is not None and self.capture_time is not None and abs(self.t+self.dt-self.capture_time)<2e-6:
            self.teacher_snapshot['atomic_indices']=curr['atomics'].argmax(-1).detach().cpu().tolist()
        return curr
