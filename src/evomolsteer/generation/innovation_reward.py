"""Bounded empirical coordinate extrapolation; no learned affinity surrogate.

Neighbour ordering and Hungarian correspondence use the unmodified endpoint
library. Geometry contrast changes the target/cost only, never categorical state.
"""
import copy
import math
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from .endpoint_reward import EndpointGeometryReward


class InnovationReward(EndpointGeometryReward):
    def __init__(self, program, reference):
        self.innovation = dict(program.get('innovation', {}))
        limits = {'field_strength_A': (0, 1), 'reliability_power': (0, 2),
                  'region_weight_mix': (0, 1)}
        if set(self.innovation)-set(limits):
            raise ValueError('Unregistered innovation parameter')
        for key, value in self.innovation.items():
            if not math.isfinite(value) or not limits[key][0] <= value <= limits[key][1]:
                raise ValueError('Unbounded coordinate innovation')
        p = copy.deepcopy(program);p['reward_view'] = 'endpoint_pointcloud'
        super().__init__(p, reference)
        self.strength = self.innovation.get('field_strength_A', 0.)
        self.weight_mix = self.innovation.get('region_weight_mix', 0.)
        self.reliability_power = self.innovation.get('reliability_power', 1.)
        if self.strength or self.weight_mix:
            for f in reference['frames']:
                shape = np.asarray(f['teacher_endpoint_A']).shape
                if np.asarray(f.get('teacher_contrast_direction_unit', [])).shape != shape:
                    raise ValueError('Matched teacher contrast field required')
                c = np.asarray(f['teacher_contrast_confidence'])
                if c.shape != shape[:1] or not np.isfinite(c).all() or np.any((c < 0)|(c > 1)):
                    raise ValueError('Bounded teacher support confidence required')

    def __call__(self, x, atoms, mask, time, anchor=None):
        # The null hypothesis delegates the original arithmetic byte-for-byte.
        if self.strength == 0 and self.weight_mix == 0:
            return super().__call__(x, atoms, mask, time, anchor)
        j = int(np.abs(self.times-time).argmin())
        if abs(self.times[j]-time)>2e-6 or not bool(mask.all()) or anchor is None:
            raise ValueError('Exact teacher node and active endpoint slots required')
        f = self.reference['frames'][j]
        centers = np.asarray(f['teacher_endpoint_A']);scores = np.asarray(f['teacher_scores'])
        direction = np.asarray(f['teacher_contrast_direction_unit'])
        confidence = np.asarray(f['teacher_contrast_confidence'])**self.reliability_power
        weights = np.asarray(f.get('teacher_contrast_atom_weight', np.ones(centers.shape[:2])))
        if weights.shape != centers.shape[:2] or not np.isfinite(weights).all() or np.any(weights < 0):
            raise ValueError('Nonnegative contrast atom weights required')
        matched, priors, atom_weights, shifted_rms = [], [], [], []
        for cloud in anchor.detach().cpu().numpy():
            costs, assignments = [], []
            for teacher in centers:
                d = ((cloud[:,None]-teacher[None])**2).sum(-1)
                row, col = linear_sum_assignment(d)
                assignments.append(col);costs.append(float(d[row,col].mean()))
            costs = np.asarray(costs)
            ids = np.argsort(costs,kind='stable')[:min(int(self.program.get('teacher_neighbors',4)),len(costs))]
            shifts = self.strength*confidence[:,None,None]*direction
            target = np.stack([(centers[i]+shifts[i])[assignments[i]] for i in ids])
            logits = -costs[ids]/float(self.program.get('teacher_endpoint_temperature_A2',4.))+float(self.program.get('teacher_score_beta',2.))*(scores[ids]-scores.mean())
            if 'teacher_base_log_weight' in f:logits += np.asarray(f['teacher_base_log_weight'])[ids]
            matched.append(target);priors.append(logits)
            atom_weights.append(np.stack([(1-self.weight_mix)+self.weight_mix*weights[i,assignments[i]] for i in ids]))
            shifted_rms.append(np.sqrt((shifts[ids]**2).sum(-1).mean()))
        target=x.new_tensor(np.asarray(matched));log_prior=x.new_tensor(np.asarray(priors)).log_softmax(1).detach()
        residual=x[:,None]-target;w=x.new_tensor(np.asarray(atom_weights))
        q=(residual.square().sum(-1)*w).mean(2)
        delta=float(self.program.get('pointcloud_delta_A',1.))
        cost=delta**2*(torch.sqrt(1+q/delta**2)-1)
        reward=self.tau*torch.logsumexp(log_prior-cost/self.tau,1)
        phase=(time-self.window[0])/(self.window[1]-self.window[0])
        schedule=(.1+.9*np.clip(phase,0,1))**float(self.program.get('time_ramp_power',0.))
        return reward,{'available':mask.any(1),'core_mask':mask.bool(),
            'dose_gate':x.new_full((len(x),),schedule),'time_dose_factor':x.new_full((len(x),),schedule),
            'nearest_standardized_rms':q.amin(1).sqrt(),
            'contrast_teacher_shift_rms_A':x.new_tensor(shifted_rms),
            'contrast_reliable_teacher_fraction':x.new_full((len(x),),float((confidence>0).mean()))}
