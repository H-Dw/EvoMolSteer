"""A data-bound correlated regional acceptable set, not whole-pose imitation."""
import math
import numpy as np
import torch
from .window_reference import load_reference
from pathlib import Path


class LocalIntervalReward:
    def __init__(self,program,reference):
        self.program=program;self.reference=load_reference(reference) if isinstance(reference,(str,Path)) else reference
        self.window=tuple(self.reference['window']);self.times=np.array(self.reference['times'])
        if list(self.window)!=program['window']:raise ValueError('Learning/control support mismatch')
        self.catalog=self.reference['catalog'];self.features=self.reference['features']
        if len(self.features)!=2:raise ValueError('Expected one correlated two-feature patch')
        if not math.isfinite(program['native_rms_ratio']) or program['native_rms_ratio']<0:raise ValueError('Invalid strength')
        self.frames=self.reference['frames']

    def active(self,t,s):return t>=self.window[0]-1e-6 and s<=self.window[1]+1e-6 and s>t

    def reference_at(self,t,x):
        idx=int(abs(self.times-t).argmin())
        if abs(self.times[idx]-t)>2e-6:raise ValueError('Missing exact regional reference')
        f=self.frames[idx]
        return x.new_tensor(f['center_A']),x.new_tensor(np.linalg.inv(f['covariance_A2'])),float(f['radius_squared'])

    def observables(self,x,atoms,mask):
        columns=[];valid=mask.bool().any(1);core=torch.zeros_like(mask,dtype=torch.bool)
        if atoms.ndim==3:atoms=atoms.detach().argmax(-1)
        for name in self.features:
            spec=self.catalog['features'][name];eligible=torch.zeros_like(mask,dtype=torch.bool)
            for element in spec['elements']:eligible|=atoms==self.catalog['atom_vocabulary'][element]
            eligible&=mask.bool();present=eligible.any(1);valid&=present
            safe=torch.where(present[:,None],eligible,mask.bool())
            points=x.new_tensor(self.catalog['regions'][spec['region']]['points_A'])
            d=((x[:,:,None,:]-points[None,None,:,:]).square().sum(-1)+1e-20).sqrt()
            tau=spec['temperature_A'];logits=(-d/tau).masked_fill(~safe[...,None],-torch.inf)
            columns.append(-tau*(torch.logsumexp(logits.flatten(1),1)-(safe.sum(1).to(x.dtype)*len(points)).log()))
            core|=(d.detach().amin(-1)<=self.program.get('core_radius_A',5.0))&mask.bool()
        return torch.stack(columns,-1),valid,core

    def from_features(self,z,valid,t):
        mu,precision,r2=self.reference_at(t,z)
        delta=z-mu;q=torch.einsum('bi,ij,bj->b',delta,precision,delta).clamp_min(1e-20)
        excess=(torch.sqrt(q/r2)-1).clamp_min(0)
        reward=-(torch.sqrt(1+excess.square())-1)
        gate=excess/torch.sqrt(1+excess.square())
        return torch.where(valid,reward,z.sum(1)*0),{'excess':excess,'dose_gate':torch.where(valid,gate,0.),'mahalanobis_radius':q.sqrt(),'inside':valid&(excess==0),'available':valid}

    def __call__(self,x,atoms,mask,t):
        z,valid,core=self.observables(x,atoms,mask);value,detail=self.from_features(z,valid,t)
        return value,{**detail,'observables_A':z,'core_mask':core}


def bounded_local_step(g,native,mask,gate,eta,scale,cap,remaining):
    n=mask.sum(1).clamp_min(1);g=g*mask[...,None]
    grms=(g.square().sum((1,2))/n).sqrt()
    nrms=((native.square()*mask[...,None]).sum((1,2))/n).sqrt()*scale
    requested=eta*nrms*gate
    coeff=torch.where(grms>1e-8,requested/(scale*grms.clamp_min(1e-8)),0.)
    step=coeff[:,None,None]*g
    cap_factor=(cap/(step.norm(dim=-1).amax(1)*scale).clamp_min(1e-30)).clamp(max=1)
    rms=(step.square().sum((1,2))/n).sqrt()*scale
    cap_factor=torch.minimum(cap_factor,(remaining.clamp_min(0)/rms.clamp_min(1e-30)).clamp(max=1))
    return step*cap_factor[:,None,None],{'gradient_rms_native':grms,'native_rms_A':nrms,'requested_rms_A':requested,'cap_factor':cap_factor}
