"""Bounded supported-coordinate attraction and optional lineage increment tracking.

The only learned knowledge is compact original Steer statistics. There is no
affinity network in this loss and no molecular graph acceptance condition.
"""
import copy,math
from pathlib import Path
import numpy as np
import torch
from .sparse_reward import SparseEndpointReward
from ..continuous.affinity_geometry import numpy_geometry,torch_geometry
from ..io import read_json,digest


class PersistentEndpointReward(SparseEndpointReward):
    def __init__(self,program,reference,history_prior=None):
        if program['reward_view']!='endpoint_supported_attractor':raise ValueError('Registered supported-attractor family required')
        base=copy.deepcopy(program);base['reward_view']='endpoint_direction'
        super().__init__(base,reference);self.program=program
        self.history_strength=float(program.get('history_strength',0.));self.history_power=float(program.get('history_time_power',2.))
        self.curvature=float(program.get('prototype_robust_delta',2.))
        if not all(math.isfinite(v) and v>=0 for v in (self.history_strength,self.history_power)) or self.history_strength>10 or not math.isfinite(self.curvature) or self.curvature<=0:
            raise ValueError('Finite bounded history strength/power and positive prototype curvature required')
        self.salience_mode=program.get('geometry_salience_mode','observed_abs_effect')
        if self.salience_mode not in ('constant','observed_abs_effect'):raise ValueError('Unknown salience rule')
        if self.salience_mode=='observed_abs_effect' and self.mode!='linear_node_effect':raise ValueError('Salience requires recorded empirical curves')
        self.previous=None;self.previous_time=None
        if history_prior is None:
            root=Path(__file__).resolve().parents[3];path=(root/program['history_prior_relative_path']).resolve()
            if not path.is_relative_to(root) or digest(path)!=program['history_prior_sha256']:raise ValueError('Exact repository history prior required')
            history_prior=read_json(path)
        if history_prior['reference_sha256']!=program['reference_sha256'] or history_prior['window']!=self.window or history_prior['times']!=self.times.tolist() or history_prior['features']!=reference['features']:
            raise ValueError('Lineage increment support mismatch')
        self.increments=np.asarray(history_prior['node_mean_success_delta_z'],float)
        if self.increments.shape!=(len(self.times),len(self.scale)) or not np.isfinite(self.increments).all():raise ValueError('Invalid exact node increments')
        history_mask=np.asarray(history_prior['history_feature_weights'],float)
        if history_mask.shape!=self.scale.shape or not np.isin(history_mask,[0,1]).all():raise ValueError('Invalid recorded history support')
        self.history_weights=self.field_weights*history_mask
        if self.history_weights.sum()>0:self.history_weights/=self.history_weights.sum()
        elif self.history_strength>0:raise ValueError('No supported active lineage increment field')
        self.prototypes=[];points=np.asarray(reference['landmarks_A'])
        for frame in self.reference['frames']:
            cloud=np.asarray(frame['teacher_endpoint_A'],float)
            self.prototypes.append(numpy_geometry(cloud,points,reference['origin_A'])/self.scale)

    def set_history(self,endpoint,time):
        self.previous=None if endpoint is None else endpoint.detach()
        self.previous_time=time

    def __call__(self,x,atoms,mask,time,anchor=None):
        j=int(np.abs(self.times-time).argmin())
        if abs(self.times[j]-time)>2e-6 or not bool(mask.all()):raise ValueError('Exact selection node and fixed active slots required')
        frame=self.reference['frames'][j];z=torch_geometry(x.double(),self.reference['landmarks_A'],self.reference['origin_A']);scaled=z/z.new_tensor(self.scale)
        variance=scaled.new_tensor(frame['variance_scaled']).mean(0);weights=scaled.new_tensor(self.field_weights)
        if self.salience_mode=='observed_abs_effect':
            effects=scaled.new_tensor([self.functions[f]['values_z'][j] if f in self.functions else 0. for f in self.reference['features']])
            weights=weights*effects.abs().clamp(.1,2.)
        target=scaled.new_tensor(self.prototypes[j]);q=((scaled[:,None]-target[None]).square()/variance*weights).sum(2)
        scores=scaled.new_tensor(frame['teacher_scores']);log_prior=(float(self.program.get('teacher_score_beta',2.))*(scores-scores.mean())).log_softmax(0)
        cost=self.curvature**2*(torch.sqrt(1+q/self.curvature**2)-1)
        attraction=self.tau*torch.logsumexp(log_prior[None]-cost/self.tau,1)
        penalty=torch.zeros_like(attraction)
        phase=float(np.clip((time-self.window[0])/(self.window[1]-self.window[0]),0,1))
        # The first supported node has no observed edge, including when the
        # learned support begins later than t=0. Ignore any pre-window cache.
        history_active=self.history_strength>0 and j>0
        if history_active:
            if self.previous is None or self.previous_time is None or abs(self.previous_time-self.times[j-1])>2e-6 or self.previous.shape!=x.shape:raise ValueError('Detached immediately preceding endpoint required')
            with torch.no_grad():previous= torch_geometry(self.previous.double(),self.reference['landmarks_A'],self.reference['origin_A'])/scaled.new_tensor(self.scale)
            residual=scaled-previous-scaled.new_tensor(self.increments[j])
            h=(residual.square()/variance*scaled.new_tensor(self.history_weights)).sum(1)
            penalty=self.history_strength*phase**self.history_power*self.curvature**2*(torch.sqrt(1+h/self.curvature**2)-1)
        value=attraction-penalty;schedule=(.1+.9*phase)**float(self.program.get('time_ramp_power',0.))
        return value,{'observables':z,'available':mask.any(1),'core_mask':mask.bool(),
            'nearest_standardized_rms':q.amin(1).sqrt(),'prototype_attraction':attraction,
            'lineage_increment_penalty':penalty,'history_active':x.new_full((len(x),),float(history_active)),
            'dose_gate':x.new_full((len(x),),schedule),'time_dose_factor':x.new_full((len(x),),schedule)}
