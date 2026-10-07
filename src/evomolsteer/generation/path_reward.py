"""Non-neural observed-future potential; exact conditional coordinate gradient.

R = tau (log sum b K exp(beta*u) - log sum b K).
The denominator removes geometric density attraction when utilities agree.
Values are retrospective Steer utilities, never calibrated native probabilities.
"""
import math
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from ..continuous.affinity_geometry import numpy_geometry,torch_geometry

class PathValueReward:
    def __init__(self,program,reference):
        self.program,self.reference=program,reference
        self.window=reference['window'];self.times=np.asarray(reference['times'])
        if program['window']!=self.window or program['derivative_path']!='flowr_endpoint_vjp':raise ValueError('Endpoint support/derivative mismatch')
        if reference.get('reference_variant')!='decoded-terminal-path-library-1.0':raise ValueError('Decoded path utilities required')
        self.tau=float(program.get('mixture_temperature',.5));self.beta=float(program.get('teacher_score_beta',2.))
        self.bandwidth=float(program.get('teacher_endpoint_temperature_A2',4.));self.delta=float(program.get('pointcloud_delta_A',1.))
        self.history_strength=float(program.get('path_history_mix',0.))
        self.space=program.get('path_kernel_space','pointcloud')
        self.confidence=program.get('path_contrast_amplitude',False)
        self.previous_endpoint=None;self.previous_time=None
        if not all(math.isfinite(v) for v in [self.tau,self.beta,self.bandwidth,self.delta,self.history_strength]):raise ValueError('Finite kernel parameters required')
        if min(self.tau,self.bandwidth,self.delta)<=0 or self.beta<0 or not 0<=self.history_strength<1:raise ValueError('Invalid potential parameters')
        if self.space not in ['pointcloud','geometry']:raise ValueError('Unknown coordinate space')
        for f in reference['frames']:
            n=len(f['teacher_scores'])
            if not n or len(f['teacher_endpoint_A'])!=n or len(f['teacher_path_ids'])!=n:raise ValueError('Identity-bearing path teachers required')
    def active(self,t,s):return self.window[0]-1e-6<=t and s<=self.window[1]+1e-6 and s>t
    def set_history(self,endpoint,time):
        self.previous_endpoint=endpoint;self.previous_time=time
    def index(self,time):
        j=int(abs(self.times-time).argmin())
        if abs(self.times[j]-time)>2e-6:raise ValueError('Exact learned time required')
        return j
    def assignment(self,cloud,frame):
        costs=[];targets=[]
        for teacher in np.asarray(frame['teacher_endpoint_A']):
            d=((cloud[:,None]-teacher[None])**2).sum(-1);row,col=linear_sum_assignment(d)
            costs.append(d[row,col].mean());targets.append(teacher[col])
        return np.asarray(costs),np.asarray(targets)
    def prior(self,frame,anchor,time):
        b=np.exp(np.asarray(frame['teacher_base_log_weight'],float));b/=b.sum()
        if not self.history_strength or self.previous_endpoint is None:return np.broadcast_to(b,(len(anchor),len(b))).copy()
        old=self.reference['frames'][self.index(self.previous_time)]
        trans=np.array([[len(set(a)&set(c)) for c in frame['teacher_path_ids']] for a in old['teacher_path_ids']],float)
        if np.any(trans.sum(1)==0):raise ValueError('Broken path continuity')
        trans/=trans.sum(1,keepdims=True)
        posterior=[]
        for cloud in self.previous_endpoint.detach().cpu().numpy():
            q,_=self.assignment(cloud,old)
            logit=np.asarray(old['teacher_base_log_weight'])-q/self.bandwidth
            p=np.exp(logit-logit.max());p/=p.sum();posterior.append(p@trans)
        return self.history_strength*np.asarray(posterior)+(1-self.history_strength)*b
    def __call__(self,x,atoms,mask,time,anchor=None):
        if anchor is None or not bool(mask.all()):raise ValueError('Fixed active endpoint clouds required')
        f=self.reference['frames'][self.index(time)];a=anchor.detach().cpu().numpy()
        prior=self.prior(f,a,time);targets=[];weights=[];scores=[]
        for i,cloud in enumerate(a):
            q,target=self.assignment(cloud,f);k=min(int(self.program.get('teacher_neighbors',4)),len(q))
            if k<1:raise ValueError('Positive neighbor count required')
            ids=np.argsort(q,kind='stable')[:k];targets.append(target[ids]);weights.append(prior[i,ids]);scores.append(np.asarray(f['teacher_scores'])[ids])
        x=x.double();target=x.new_tensor(np.asarray(targets));mass=x.new_tensor(np.asarray(weights)).clamp_min(1e-30)
        u=x.new_tensor(np.asarray(scores));u=u-u.mean(1,keepdim=True)
        if self.space=='pointcloud':q=(x[:,None]-target).square().sum(-1).mean(2)
        else:
            z=torch_geometry(x,self.reference['landmarks_A'],self.reference['origin_A'])
            yt=numpy_geometry(np.asarray(targets).reshape(-1,x.shape[1],3),self.reference['landmarks_A'],self.reference['origin_A'])
            y=x.new_tensor(yt).reshape(len(x),len(targets[0]),-1);scale=x.new_tensor(self.reference['feature_scale'])
            q=((z[:,None]-y)/scale).square().mean(2)
        cost=self.delta**2*(torch.sqrt(1+q/self.delta**2)-1)/self.bandwidth
        base=mass.log()-cost/self.tau
        value=self.tau*(torch.logsumexp(base+self.beta*u,1)-torch.logsumexp(base,1))
        phase=np.clip((time-self.window[0])/(self.window[1]-self.window[0]),0,1)
        schedule=(.1+.9*phase)**float(self.program.get('time_ramp_power',0.))
        p=base.softmax(1).detach();mu=(p*u).sum(1);sd=(p*(u-mu[:,None]).square()).sum(1).sqrt()
        amplitude=self.beta*sd/torch.sqrt(1+(self.beta*sd).square()) if self.confidence else torch.ones_like(sd)
        return value,{'observables':x.mean(1),'available':mask.any(1),'core_mask':mask,
            'dose_gate':(amplitude*schedule).detach(),'time_dose_factor':x.new_full((len(x),),schedule),
            'nearest_standardized_rms':q.amin(1).sqrt().detach(),'observed_utility_sd':sd.detach(),
            'contrast_amplitude':amplitude.detach()}
