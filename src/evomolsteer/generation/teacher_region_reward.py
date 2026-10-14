"""M2 point-cloud loss restricted to teacher atoms in Agent-selected regions."""
import copy
import math
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from .endpoint_reward import EndpointGeometryReward


class TeacherRegionReward(EndpointGeometryReward):
    def __init__(self, program, reference):
        base=copy.deepcopy(program);base['reward_view']='endpoint_pointcloud'
        super().__init__(base,reference)
        self.program=program
        self.weights=np.asarray(program['coordinate_region_weights'],float)
        self.radius=float(program['coordinate_region_radius_A'])
        if self.weights.shape!=(len(reference['landmarks_A']),) or not np.isfinite(self.weights).all() or (self.weights<0).any() or not self.weights.any() or not math.isfinite(self.radius) or self.radius<=0:
            raise ValueError('Nonempty finite selected coordinate regions required')
        if program.get('coordinate_background_weight') != 0:
            raise ValueError('Teacher-region mode has zero global background loss')

    def __call__(self,x,atoms,mask,time,anchor=None):
        j=int(np.abs(self.times-time).argmin())
        if abs(self.times[j]-time)>2e-6 or anchor is None or not bool(mask.all()):
            raise ValueError('Exact learned clock, fixed active point clouds and anchor required')
        frame=self.reference['frames'][j]
        teachers=np.asarray(frame['teacher_endpoint_A'],float)
        points=np.asarray(self.reference['landmarks_A'],float)
        distance=((teachers[:,:,None]-points[None,None])**2).sum(-1)
        attention=(np.maximum(1-distance/self.radius**2,0)**2*self.weights).sum(2)
        eligible=np.flatnonzero(attention.sum(1)>1e-8)
        if not len(eligible):
            raise ValueError('No teacher has support in selected coordinate regions')
        targets,attentions,priors,teacher_ids=[],[],[],[]
        weights=np.asarray(frame.get('teacher_base_log_weight',np.zeros(len(teachers))),float)
        scores=np.asarray(frame['teacher_scores'],float)
        for cloud in anchor.detach().cpu().numpy():
            costs,permutations=[],[]
            for teacher_id in eligible:
                # Column weights are teacher-region membership. Outside-region
                # atoms carry no cost and no direct endpoint-coordinate force.
                d=((cloud[:,None]-teachers[teacher_id][None])**2).sum(-1)
                rows,cols=linear_sum_assignment(d*attention[teacher_id][None])
                permutations.append(cols)
                costs.append(float((d[rows,cols]*attention[teacher_id,cols]).sum()/attention[teacher_id].sum()))
            keep=np.argsort(costs,kind='stable')[:min(int(self.program['teacher_neighbors']),len(costs))]
            ids=eligible[keep]
            targets.append(np.stack([teachers[i,permutations[k]] for i,k in zip(ids,keep)]))
            attentions.append(np.stack([attention[i,permutations[k]] for i,k in zip(ids,keep)]))
            priors.append(-np.asarray(costs)[keep]/float(self.program['teacher_endpoint_temperature_A2'])+weights[ids]+
                float(self.program['teacher_score_beta'])*(scores[ids]-scores.mean()))
            teacher_ids.append(ids)
        target=x.new_tensor(np.asarray(targets));weight=x.new_tensor(np.asarray(attentions)).detach()
        log_prior=x.new_tensor(np.asarray(priors)).log_softmax(1).detach()
        q=((x[:,None]-target).square().sum(-1)*weight).sum(2)/weight.sum(2)
        delta=float(self.program.get('pointcloud_delta_A',1.))
        cost=delta**2*(torch.sqrt(1+q/delta**2)-1)
        posterior=(log_prior-cost/self.tau).softmax(1).detach()
        value=self.tau*torch.logsumexp(log_prior-cost/self.tau,1)
        core=(weight>0).any(1)&mask
        return value,{'available':mask.any(1),'core_mask':core,'nearest_standardized_rms':q.amin(1).sqrt(),
            'dose_gate':x.new_ones(len(x)),'time_dose_factor':x.new_ones(len(x)),
            'regional_core_fraction':core.float().mean(1),'regional_zero_background':x.new_ones(len(x)),
            'regional_teacher_ids':x.new_tensor(np.asarray(teacher_ids)),
            'regional_teacher_posterior':posterior,'regional_teacher_mass':weight.sum(2)}
