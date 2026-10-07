"""Coordinate-only affinity-stratified rewards; no head calls or model pullback."""
import math
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from ..continuous.affinity_geometry import torch_geometry,numpy_geometry
from .coordinate_reward import CoordinateMixtureReward

class AffinityGeometryReward(CoordinateMixtureReward):
    def __init__(self,program,reference):
        self.program,self.reference=program,reference;self.window=reference['window'];self.times=np.array(reference['times'])
        if reference['schema_version']!='affinity-coordinate-library-1.0' or program['window']!=self.window:raise ValueError('Coordinate learning contract mismatch')
        self.tau=float(program.get('mixture_temperature',.5));self.scale=np.array(reference['feature_scale'])
        values=[self.tau,float(program.get('teacher_score_beta',2.)),float(program.get('teacher_endpoint_temperature_A2',4.)),float(program.get('pointcloud_delta_A',1.))]
        if not all(math.isfinite(v) for v in values) or self.tau<=0 or values[1]<0 or min(values[2:])<=0:
            raise ValueError('Invalid geometry response')
        if not 1<=int(program.get('teacher_neighbors',4))<=1000 or not np.isfinite(self.scale).all() or (self.scale<=0).any():
            raise ValueError('Invalid teacher support/scales')
        if len(self.scale)!=len(reference['features']):raise ValueError('Geometry feature dimensions')
        blocks=program.get('geometry_block_weights',[1.,1.,1.,1.]);power=float(program.get('time_ramp_power',0.))
        if len(blocks)!=4 or not all(math.isfinite(v) and v>=0 for v in blocks) or not sum(blocks)>0:
            raise ValueError('Four finite nonnegative geometry weights required')
        if not math.isfinite(power) or power<0:raise ValueError('Finite nonnegative time schedule required')
    def __call__(self,x,atoms,mask,time,anchor=None):
        j=int(np.abs(self.times-time).argmin())
        if abs(self.times[j]-time)>2e-6:raise ValueError('No actual learned coordinate node')
        if not bool(mask.all()):raise ValueError('Teacher point cloud requires fixed active slots')
        frame=self.reference['frames'][j];z=torch_geometry(x.double(),self.reference['landmarks_A'],self.reference['origin_A'])
        scaled=z/z.new_tensor(self.scale);family=self.program['reward_view'];weights=self.program.get('geometry_block_weights',[1.,1.,1.,1.])
        w=np.r_[np.full(3,weights[0]/3),np.full(6,weights[1]/6),np.full(5,weights[2]/5),np.full(len(self.scale)-14,weights[3]/(len(self.scale)-14))]
        if family in ('affinity_landmark','affinity_direction'):
            hi=scaled.new_tensor(frame['high_scaled']);lo=scaled.new_tensor(frame['low_scaled']);var=scaled.new_tensor(frame['variance_scaled'])
            if family=='affinity_direction':
                direction=((hi-lo)/var).mean(0).clamp(-3,3)*scaled.new_tensor(w)
                # Smooth linear response rather than falsely using d/dt as a force.
                reward=(scaled*direction).sum(1)
                nearest=((scaled[:,None]-hi).square()/var).mean(2).amin(1).sqrt()
            else:
                qh=((scaled[:,None]-hi).square()/var*scaled.new_tensor(w)).sum(2)
                ql=((scaled[:,None]-lo).square()/var*scaled.new_tensor(w)).sum(2)
                reward=self.tau*(torch.logsumexp(-qh/self.tau,1)-torch.logsumexp(-ql/self.tau,1))
                nearest=qh.amin(1).sqrt()
        elif family=='affinity_pointcloud':
            if anchor is None:raise ValueError('Native endpoint forecast required for conditional correspondence')
            endpoint=np.asarray(frame['teacher_endpoint_A']);proposal=np.asarray(frame['teacher_proposal_A']);scores=np.asarray(frame['teacher_scores'])
            a=anchor.detach().cpu().numpy();matched=[];priors=[]
            # Endpoint correspondence is geometric only, independent of atom type/graph.
            for cloud in a:
                costs=[];assignments=[]
                for teacher in endpoint:
                    d=((cloud[:,None]-teacher[None])**2).sum(-1);row,col=linear_sum_assignment(d)
                    assignments.append(col);costs.append(float(d[row,col].mean()))
                costs=np.asarray(costs);k=min(int(self.program.get('teacher_neighbors',4)),len(costs))
                ids=np.argsort(costs,kind='stable')[:k]
                target=np.stack([proposal[i,assignments[i]] for i in ids])
                logits=-costs[ids]/float(self.program.get('teacher_endpoint_temperature_A2',4.))+float(self.program.get('teacher_score_beta',2.))*(scores[ids]-scores.mean())
                if 'teacher_base_log_weight' in frame:
                    logits=logits+np.asarray(frame['teacher_base_log_weight'])[ids]
                priors.append(logits);matched.append(target)
            target=x.new_tensor(np.array(matched));logits=x.new_tensor(np.array(priors));log_prior=logits.log_softmax(1).detach()
            residual=x[:,None]-target;q=residual.square().sum(-1).mean(2)
            curvature=float(self.program.get('pointcloud_delta_A',1.))
            if curvature<=0:raise ValueError('Positive point-cloud curvature required')
            cost=curvature**2*(torch.sqrt(1+q/curvature**2)-1)
            # Preserve alternative modes; an averaged loss pulls toward a
            # compromise point cloud even when an actual teacher is reached.
            reward=self.tau*torch.logsumexp(log_prior-cost/self.tau,1);nearest=q.amin(1).sqrt()
        else:raise ValueError('Unknown affinity geometry family')
        phase=(time-self.window[0])/(self.window[1]-self.window[0]);schedule=(.1+.9*np.clip(phase,0,1))**float(self.program.get('time_ramp_power',0.))
        valid=mask.any(1);return reward,{'observables':z,'available':valid,'core_mask':mask.bool(),
            'nearest_standardized_rms':nearest,'dose_gate':x.new_full((len(x),),schedule),'time_dose_factor':x.new_full((len(x),),schedule)}
