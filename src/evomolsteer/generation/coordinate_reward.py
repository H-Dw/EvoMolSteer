"""Current-state coordinate mixture reward with live, piecewise smooth gradients."""
import math
import numpy as np
import torch


def remove_rigid_pose_gradient(g,x,mask):
    """Orthogonal projection off whole-ligand translation and infinitesimal rotation."""
    weight=mask.to(x.dtype);n=weight.sum(1).clamp_min(1)
    center=(x*weight[...,None]).sum(1)/n[:,None];r=(x-center[:,None])*weight[...,None]
    g=(g-(g*weight[...,None]).sum(1)[:,None]/n[:,None,None])*weight[...,None]
    inertia=torch.eye(3,device=x.device,dtype=x.dtype)[None]*(r.square().sum((1,2)))[:,None,None]-torch.einsum('bni,bnj->bij',r,r)
    torque=torch.cross(r,g,dim=-1).sum(1)
    omega=torch.einsum('bij,bj->bi',torch.linalg.pinv(inertia,hermitian=True,rtol=1e-6),torque)
    return (g-torch.cross(omega[:,None].expand_as(r),r,dim=-1))*weight[...,None]


class CoordinateMixtureReward:
    def __init__(self,program,reference):
        self.program=program;self.reference=reference;self.window=reference['window'];self.times=np.asarray(reference['times'])
        if program['window']!=self.window:raise ValueError('Learning/control window mismatch')
        if reference['schema_version']!='current-coordinate-mixture-1.0':raise ValueError('Wrong reference view')
        self.tau=float(program['mixture_temperature']);self.delta=float(program['robust_delta'])
        if not all(math.isfinite(v) and v>0 for v in (self.tau,self.delta)):raise ValueError('Invalid response law')

    def active(self,t,s):return t>=self.window[0]-1e-6 and s<=self.window[1]+1e-6 and s>t

    def observables(self,x,atoms,mask):
        columns=[];mask=mask.bool();valid=mask.any(1);core=torch.zeros_like(mask)
        if atoms.ndim==3:atoms=atoms.detach().argmax(-1)
        eligible=mask.clone()
        if self.reference['channel']=='NOS':
            eligible&=torch.isin(atoms,atoms.new_tensor([self.reference['atom_vocabulary'][a] for a in ('N','O','S')]))
        valid&=eligible.any(1);safe=torch.where(valid[:,None],eligible,mask)
        for r in self.reference['regions'].values():
            points=x.new_tensor(r['points_A']);d=(x[:,:,None]-points[None,None]).square().sum(-1).clamp_min(1e-20).sqrt().amin(-1)
            uniform=self.reference.get('spatial_weighting')=='uniform_global_control'
            logits=(torch.zeros_like(d) if uniform else -.5*(d/self.reference['spatial_width_A']).square()).masked_fill(~safe,-torch.inf)
            w=logits.softmax(1);center=(w[...,None]*x).sum(1)
            spread=(w*(x-center[:,None]).square().sum(-1)).sum(1).clamp_min(1e-20).sqrt()
            columns.extend([center-points.mean(0),spread[:,None]])
            core|=eligible if uniform else eligible&(d.detach()<=self.program['core_radius_A'])
        return torch.cat(columns,1),valid,core

    def __call__(self,x,atoms,mask,time):
        j=int(np.abs(self.times-time).argmin())
        if abs(self.times[j]-time)>2e-6:raise ValueError('No exact learned coordinate time')
        z,valid,core=self.observables(x,atoms,mask);modes=self.reference['frames'][j]['modes']
        if self.program.get('reward_view')=='spread_upper':
            frame=self.reference['frames'][j]
            upper=z.new_tensor(frame['upper_spread_A']);scale=z.new_tensor(frame['spread_scale_A'])
            violation=((z[:,3::4]-upper)/scale).clamp_min(0)
            q=violation.square().mean(1);reward=-(torch.sqrt(1+q)-1)
            gate=torch.sqrt(q/(1+q))
            return torch.where(valid,reward,z.sum(1)*0),{'observables_A':z,'available':valid,'core_mask':core,
                'spread_violation':violation,'nearest_standardized_rms':q.sqrt(),
                'dose_gate':torch.where(valid,gate,0.)}
        centers=z.new_tensor([v['center_A'] for v in modes])
        precision=z.new_tensor(np.linalg.inv(np.asarray([v['covariance_A2'] for v in modes])))
        residual=z[:,None]-centers[None];q=torch.einsum('bmi,mij,bmj->bm',residual,precision,residual).clamp_min(0)/z.shape[1]
        cost=self.delta**2*(torch.sqrt(1+q/self.delta**2)-1)
        logits=-cost/self.tau-math.log(len(modes));reward=self.tau*torch.logsumexp(logits,1)
        gate=torch.sqrt(q.amin(1)/(1+q.amin(1)))
        return torch.where(valid,reward,z.sum(1)*0),{'observables_A':z,'available':valid,'core_mask':core,
            'nearest_standardized_rms':q.amin(1).sqrt(),'dose_gate':torch.where(valid,gate,0.),
            'mode_responsibilities':logits.softmax(1)}
