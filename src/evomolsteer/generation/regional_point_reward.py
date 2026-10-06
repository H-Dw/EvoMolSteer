"""Soft regional emphasis of coherent elite point clouds, with frozen membership.

Assignments, neighborhood priors and spatial attention use the detached current
endpoint anchor. The conditional loss differentiates only endpoint coordinates;
there is no atom-type condition, graph veto, head call or particle selection.
"""
import copy,math
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from .endpoint_reward import EndpointGeometryReward
from ..continuous.affinity_geometry import torch_geometry
from ..io import digest,read_json


class RegionalPointReward(EndpointGeometryReward):
    def __init__(self,program,reference):
        if program['reward_view']!='endpoint_regional_pointcloud':raise ValueError('Registered regional point-cloud family required')
        base=copy.deepcopy(program);base['reward_view']='endpoint_pointcloud';super().__init__(base,reference);self.program=program
        self.region_weights=np.asarray(program['coordinate_region_weights'],float)
        self.radius=float(program.get('coordinate_region_radius_A',5.));self.background=float(program.get('coordinate_background_weight',.25))
        if self.region_weights.shape!=(len(reference['landmarks_A']),) or not np.isfinite(self.region_weights).all() or (self.region_weights<0).any() or not self.region_weights.any():
            raise ValueError('Finite nonzero exact-region weights required')
        if not math.isfinite(self.radius) or self.radius<=0 or not math.isfinite(self.background) or self.background<=0:
            raise ValueError('Positive spatial radius and outside-region weight required')
        self.node_salience=program.get('coordinate_region_node_salience')
        if self.node_salience is not None:
            self.node_salience=np.asarray(self.node_salience,float)
            if self.node_salience.shape!=(len(self.times),len(self.region_weights)) or not np.isfinite(self.node_salience).all() or (self.node_salience<.1).any() or (self.node_salience>2.).any():
                raise ValueError('Exact positive bounded empirical stage salience required')
        if 'region_prior_relative_path' in program:
            root=Path(__file__).resolve().parents[3];path=(root/program['region_prior_relative_path']).resolve()
            if not path.is_relative_to(root) or digest(path)!=program['region_prior_sha256']:raise ValueError('Exact repository regional prior required')
            prior=read_json(path)
            if prior['reference_sha256']!=program['reference_sha256'] or prior['window']!=self.window or prior['times']!=self.times.tolist():raise ValueError('Regional prior support mismatch')
            if self.node_salience is not None:
                for r,w in enumerate(self.region_weights):
                    if w:
                        f=f'landmark_{r:02d}_soft_mass'
                        if f not in prior['supported_node_functions'] or not np.array_equal(self.node_salience[:,r],np.clip(np.abs(prior['supported_node_functions'][f]['values_z']),.1,2.)):
                            raise ValueError('Empirical regional stage amplitude differs from prior')

    def __call__(self,x,atoms,mask,time,anchor=None):
        j=int(np.abs(self.times-time).argmin())
        if abs(self.times[j]-time)>2e-6 or not bool(mask.all()) or anchor is None:raise ValueError('Exact supported node, fixed slots and detached endpoint anchor required')
        frame=self.reference['frames'][j];endpoint=np.asarray(frame['teacher_endpoint_A']);scores=np.asarray(frame['teacher_scores'])
        matched=[];priors=[]
        for cloud in anchor.detach().cpu().numpy():
            costs=[];assignments=[]
            for teacher in endpoint:
                d=((cloud[:,None]-teacher[None])**2).sum(-1);rows,cols=linear_sum_assignment(d)
                assignments.append(cols);costs.append(float(d[rows,cols].mean()))
            costs=np.asarray(costs);k=min(int(self.program.get('teacher_neighbors',4)),len(costs));ids=np.argsort(costs,kind='stable')[:k]
            matched.append(np.stack([endpoint[i,assignments[i]] for i in ids]))
            priors.append(-costs[ids]/float(self.program.get('teacher_endpoint_temperature_A2',4.))+
                          float(self.program.get('teacher_score_beta',2.))*(scores[ids]-scores.mean()))
        target=x.new_tensor(np.asarray(matched));log_prior=x.new_tensor(np.asarray(priors)).log_softmax(1).detach()
        points=x.new_tensor(self.reference['landmarks_A']);distance=(anchor.detach()[:,:,None]-points[None,None]).square().sum(-1)
        weights=x.new_tensor(self.region_weights)
        if self.node_salience is not None:weights=weights*x.new_tensor(self.node_salience[j])
        local=(torch.exp(-distance/(2*self.radius**2))*weights).sum(2)
        attention=(self.background+local).detach()
        q=((x[:,None]-target).square().sum(-1)*attention[:,None]).sum(2)/attention.sum(1)[:,None]
        curvature=float(self.program.get('pointcloud_delta_A',1.));cost=curvature**2*(torch.sqrt(1+q/curvature**2)-1)
        value=self.tau*torch.logsumexp(log_prior-cost/self.tau,1)
        phase=(time-self.window[0])/(self.window[1]-self.window[0]);schedule=(.1+.9*np.clip(phase,0,1))**float(self.program.get('time_ramp_power',0.))
        z=torch_geometry(x.double(),self.reference['landmarks_A'],self.reference['origin_A'])
        core=(distance[:,:,self.region_weights>0].amin(2)<=self.radius**2)&mask
        return value,{'observables':z,'available':mask.any(1),'core_mask':core,
            'nearest_standardized_rms':q.amin(1).sqrt(),'regional_attention_max':attention.amax(1),
            'regional_attention_min':attention.amin(1),'regional_core_fraction':core.float().mean(1),
            'dose_gate':x.new_full((len(x),),schedule),'time_dose_factor':x.new_full((len(x),),schedule)}
