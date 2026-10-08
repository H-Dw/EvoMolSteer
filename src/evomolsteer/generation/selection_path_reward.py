"""Conservative alternatives to the R26 teacher prior or coordinate metric.

All correspondence, conditional prior and discrete niches are frozen at the
supplied endpoint anchor. Actual endpoint coordinates retain their FLOWR VJP.
"""
import copy
import math
import numpy as np
import torch
from scipy.special import softmax
from scipy.stats import rankdata
from scipy.optimize import linear_sum_assignment

from .endpoint_reward import EndpointGeometryReward
from ..continuous.affinity_geometry import torch_geometry


def ess_floor_prior(logits, fraction):
    p=softmax(np.asarray(logits,float));k=len(p)
    if not 0<=fraction<=1 or not np.isfinite(logits).all():raise ValueError('Finite ESS prior contract')
    if fraction==0 or 1/(p@p)>=fraction*k:return p,0.
    lo,hi=0.,1.
    for _ in range(48):
        mid=(lo+hi)/2;q=(1-mid)*p+mid/k
        if 1/(q@q)<fraction*k:lo=mid
        else:hi=mid
    return (1-hi)*p+hi/k,hi


class SelectionPathReward(EndpointGeometryReward):
    def __init__(self,program,reference):
        p=copy.deepcopy(program);p['reward_view']='endpoint_pointcloud';super().__init__(p,reference)
        self.spec=copy.deepcopy(program.get('selection_path',{}))
        allowed={'quality','ess_fraction','niche_balance','selection','precision_mix','robust_aggregation'}
        if set(self.spec)-allowed:raise ValueError('Unknown selection-path mechanism')
        if self.spec.get('quality','raw') not in ['raw','rank']:raise ValueError('Quality mapping')
        if self.spec.get('selection','nearest') not in ['nearest','niche','batch']:raise ValueError('Teacher coverage')
        if self.spec.get('robust_aggregation','cloud') not in ['cloud','atom']:raise ValueError('Robust aggregation')
        for k in ['ess_fraction','precision_mix']:
            if not math.isfinite(self.spec.get(k,0)) or not 0<=self.spec.get(k,0)<=1:raise ValueError('Bounded selection parameter')
        if not math.isfinite(self.spec.get('niche_balance',0)) or self.spec.get('niche_balance',0)<0:raise ValueError('Niche balance')
        if any(self.spec.get(k,0)>0 for k in ['precision_mix','niche_balance']) or self.spec.get('selection')=='niche':
            if 'selection_path' not in reference:raise ValueError('Offline coordinate niche evidence required')

    def __call__(self,x,atoms,mask,time,anchor=None):
        if not self.spec:return super().__call__(x,atoms,mask,time,anchor)
        if anchor is None or not bool(mask.all()):raise ValueError('Full endpoint correspondence required')
        j=int(np.abs(self.times-time).argmin())
        if abs(self.times[j]-time)>2e-6:raise ValueError('Only actual learned nodes supported')
        frame=self.reference['frames'][j];clouds=np.asarray(frame['teacher_endpoint_A']);scores=np.asarray(frame['teacher_scores'])
        groups=np.asarray(frame.get('teacher_niche',np.zeros(len(clouds),int)))
        batches=np.asarray(frame['teacher_batches']);matched=[];logpriors=[];metrics=[];audit=[]
        for cloud in anchor.detach().cpu().numpy():
            costs=[];assignments=[]
            for teacher in clouds:
                row,col=linear_sum_assignment(((cloud[:,None]-teacher[None])**2).sum(-1))
                costs.append(float(((cloud[row]-teacher[col])**2).sum(-1).mean()));assignments.append(col)
            order=np.argsort(costs,kind='stable');k=min(int(self.program.get('teacher_neighbors',4)),len(order))
            ids=[];used=set();selection=self.spec.get('selection','nearest')
            labels=groups if selection=='niche' else batches
            for i in order:
                if selection!='nearest' and int(labels[i]) in used:continue
                ids.append(int(i));used.add(int(labels[i]))
                if len(ids)==k:break
            # All rows need the same K; if a niche collapses, preserve native support.
            if len(ids)<k:ids += [int(i) for i in order if i not in ids][:k-len(ids)]
            ids=np.asarray(ids);quality=scores-scores.mean()
            if self.spec.get('quality')=='rank':quality=(rankdata(scores,method='average')-.5)/len(scores)-.5
            logits=-np.asarray(costs)[ids]/self.program['teacher_endpoint_temperature_A2']+self.program['teacher_score_beta']*quality[ids]
            strength=self.spec.get('niche_balance',0)
            if strength:logits-=strength*np.log([np.sum(groups[ids]==groups[i]) for i in ids])
            probability,base_chance=ess_floor_prior(logits,self.spec.get('ess_fraction',0))
            logpriors.append(np.log(probability));matched.append(np.stack([clouds[i,assignments[i]] for i in ids]))
            if self.spec.get('precision_mix',0)>0:
                precision=np.asarray(frame['teacher_precision'])
                metrics.append(np.stack([precision[i,assignments[i]] for i in ids]))
            audit.append([1/(probability@probability),base_chance,len(np.unique(groups[ids]))])
        target=x.new_tensor(np.asarray(matched));log_prior=x.new_tensor(np.asarray(logpriors)).detach()
        residual=x[:,None]-target;per_atom=residual.square().sum(-1)
        mix=self.spec.get('precision_mix',0)
        if mix:
            metric=x.new_tensor(np.asarray(metrics));anisotropic=torch.einsum('bkni,bknij,bknj->bkn',residual,metric,residual)
            per_atom=(1-mix)*per_atom+mix*anisotropic
        q=per_atom.mean(2);delta=float(self.program.get('pointcloud_delta_A',1.))
        if self.spec.get('robust_aggregation')=='atom':cost=(delta**2*(torch.sqrt(1+per_atom/delta**2)-1)).mean(2)
        else:cost=delta**2*(torch.sqrt(1+q/delta**2)-1)
        reward=self.tau*torch.logsumexp(log_prior-cost/self.tau,1)
        phase=(time-self.window[0])/(self.window[1]-self.window[0])
        dose=(.1+.9*np.clip(phase,0,1))**float(self.program.get('time_ramp_power',0))
        return reward,{'observables':torch_geometry(x.double(),self.reference['landmarks_A'],self.reference['origin_A']),
          'available':mask.any(1),'core_mask':mask.bool(),'nearest_standardized_rms':q.amin(1).sqrt(),
          'dose_gate':x.new_full((len(x),),dose),'time_dose_factor':x.new_full((len(x),),dose),
          'selection_pressure_audit':x.new_tensor(audit).detach()}
