"""A registered transfer test on the SAME Analyst-selected real-coordinate points."""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from evomolsteer.final_guidance.reward import HalfStateReward,bridge
from evomolsteer.final_guidance.native import Control


class TwoClockReward(HalfStateReward):
    def __init__(self,program,teacher):
        if program['family']!='actual_partial_two_clock':raise ValueError('Wrong transfer-test reward family')
        super().__init__(dict(program,family='actual_half_state_pointcloud'),teacher)
        self.program=program;self.mix=float(program['endpoint_mix'])
        if not 0<self.mix<1 or teacher.get('terminal_representation')!='terminal_world_A':raise ValueError('Actual final-coordinate transfer evidence required')
        self.final=np.asarray(teacher['terminal_coords_A'],float)[self.slots]
        if self.final.shape!=self.target.shape or not np.isfinite(self.final).all():raise ValueError('Terminal/ancestor slot map differs')
    def joint_assignment(self,z,endpoint,mask):
        result=[]
        for a,b,valid in zip(z.detach().cpu().numpy(),endpoint.detach().cpu().numpy(),mask.detach().cpu().numpy()):
            ids=np.flatnonzero(valid)
            if len(ids)<len(self.slots):raise ValueError('Too few generated points')
            cost=((1-self.mix)*np.square(a[ids,None]-self.target[None]).sum(-1)
                +self.mix*np.square(b[ids,None]-self.final[None]).sum(-1))*self.weights[None]
            rows,cols=linear_sum_assignment(cost);result.append(ids[rows[np.argsort(cols)]])
        return torch.as_tensor(np.asarray(result),device=z.device,dtype=torch.long)
    def value_two(self,z,endpoint,mask,assignment=None):
        ids=self.joint_assignment(z,endpoint,mask) if assignment is None else assignment
        indices=ids[...,None].expand(-1,-1,3);w=z.new_tensor(self.weights)
        q_half=((z.gather(1,indices)-z.new_tensor(self.target)[None]).square().sum(-1)*w).sum(1)
        q_final=((endpoint.gather(1,indices)-z.new_tensor(self.final)[None]).square().sum(-1)*w).sum(1)
        robust=lambda q:-self.delta**2*(torch.sqrt(1+q/self.delta**2)-1)
        value=(1-self.mix)*robust(q_half)+self.mix*robust(q_final)
        direct=torch.zeros_like(mask,dtype=torch.bool);direct.scatter_(1,ids,True)
        return value,{'assignment':ids,'direct_mask':direct,'half_rms_A':q_half.sqrt(),'final_forecast_rms_A':q_final.sqrt()}


class TwoClockControl(Control):
    def __init__(self,model,program,teacher,com,capture_time=None):
        super().__init__(model,None,teacher,com,capture_time)
        self.program=program;self.reward=TwoClockReward(program,teacher)
    def forward(self,curr,pocket,times,cond,equis,invs,i,dt):
        self.cached=None;self.t=float(times[0][0]);self.dt=float(dt);self.before=curr['coords'].detach().clone()
        kwargs=dict(cond_batch=cond,pocket_equis=equis,pocket_invs=invs,training=False)
        if not self.reward.active(self.t,self.dt):return self.model(curr,pocket,times,**kwargs)
        with torch.enable_grad():
            x=self.before.clone().requires_grad_(True);out=self.model(dict(curr,coords=x),pocket,times,**kwargs)
            if 'affinity' in out:
                def denied(gradient):raise RuntimeError('Affinity head entered transfer-test gradient')
                for v in out['affinity'].values():
                    if torch.is_tensor(v) and v.requires_grad:v.register_hook(denied)
            pred,_=self.model._get_predictions(out)
            endpoint=pred['coords']*self.model.coord_scale+self.com[:,None]
            current=x*self.model.coord_scale+self.com[:,None]
            z=bridge(current,endpoint,self.t,self.reward.teacher['time'])
            value,detail=self.reward.value_two(z,endpoint,curr['mask'].bool())
            gx,gz,ge=torch.autograd.grad(value.sum(),(x,z,endpoint))
        if not torch.isfinite(gx).all() or not torch.isfinite(value).all():raise ValueError('Nonfinite two-clock pullback')
        self.extra={'endpoint_mix':self.reward.mix,'half_teacher_rms_A_mean':float(detail['half_rms_A'].mean()),
            'final_forecast_teacher_rms_A_mean':float(detail['final_forecast_rms_A'].mean()),
            'endpoint_direct_unmatched_gradient_max':float(ge.masked_fill(detail['direct_mask'][...,None],0).abs().max())}
        self.cached=(gx.detach(),pred['coords'].detach(),self.dt,detail['direct_mask'].detach(),value.detach(),gz.detach())
        return {k:({a:b.detach() if torch.is_tensor(b) else b for a,b in v.items()} if isinstance(v,dict) else v.detach() if torch.is_tensor(v) else v) for k,v in out.items()}
    def after(self,curr,i):
        active=self.cached is not None;curr=super().after(curr,i)
        if active:self.rows[-1].update(self.extra)
        return curr
