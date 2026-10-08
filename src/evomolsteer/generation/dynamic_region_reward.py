"""An empirical regional contrast added to the validated endpoint attractor.

The frozen joint head labels only the offline cohorts. This coordinate scalar
uses no atom types, affinity inference, neural surrogate or particle selection.
"""
import copy,math
import numpy as np
import torch
from ..continuous.affinity_geometry import torch_geometry
from .endpoint_reward import EndpointGeometryReward


def regional_response(z,positive,negative,variance,temperature=1.,bound=2.,kind='contrast'):
    """Equal-batch Gaussian moment kernels, averaged over selected fields.

    Both cohorts use a common bandwidth. The ratio is an observational contrast,
    never a native density-ratio or an estimate of causal binding affinity.
    """
    p,n,v=z.new_tensor(positive),z.new_tensor(negative),z.new_tensor(variance)
    cp=.5*((z[:,None]-p[None]).square()/v).mean(-1)
    cn=.5*((z[:,None]-n[None]).square()/v).mean(-1)
    lp=temperature*(torch.logsumexp(-cp/temperature,1)-math.log(len(p)))
    ln=temperature*(torch.logsumexp(-cn/temperature,1)-math.log(len(n)))
    contrast=lp-ln
    if kind=='contrast':reward=bound*torch.tanh(contrast/bound)
    elif kind=='positive':reward=lp
    else:raise ValueError('Unknown regional response')
    return reward,contrast


class DynamicRegionReward(EndpointGeometryReward):
    def __init__(self,program,reference):
        if reference.get('reference_variant')!='dynamic-cohort-endpoint-library-1.0':
            raise ValueError('Dynamic cohort evidence required')
        self.region_reference=reference
        self.region_indices=np.asarray(reference['dynamic_cohort']['selected_feature_indices'],int)
        self.region_weight=float(program.get('regional_weight',.05))
        self.region_temperature=float(program.get('regional_temperature',1.))
        self.region_bound=float(program.get('regional_bound',2.))
        self.region_kind=program.get('regional_response','contrast')
        if not all(math.isfinite(v) for v in [self.region_weight,self.region_temperature,self.region_bound]):
            raise ValueError('Finite regional controls required')
        if self.region_weight<0 or min(self.region_temperature,self.region_bound)<=0:
            raise ValueError('Invalid regional controls')
        if len(self.region_indices)==0 or len(np.unique(self.region_indices))!=len(self.region_indices) or (
                (self.region_indices<14)|(self.region_indices>=len(reference['features']))).any():
            raise ValueError('Nonempty unique spatial feature indices required')
        if self.region_kind not in ('contrast','positive'):raise ValueError('Unknown regional response')
        p=copy.deepcopy(program);p['reward_view']='endpoint_pointcloud'
        super().__init__(p,reference)
        for frame in reference['frames']:
            d=frame['dynamic_region']
            arrays=[np.asarray(d[k],float) for k in ['positive_centers','negative_centers','feature_scale','positive_variance','negative_variance']]
            if not all(np.isfinite(v).all() for v in arrays) or (arrays[2]<=0).any() or min(arrays[3].min(),arrays[4].min())<=0:
                raise ValueError('Finite positive regional scales and variances required')
            if arrays[0].shape!=arrays[1].shape or arrays[0].shape[1]!=len(reference['features']):
                raise ValueError('Aligned positive and negative batch moments required')

    def __call__(self,x,atoms,mask,time,anchor=None):
        base,detail=super().__call__(x,atoms,mask,time,anchor)
        if self.region_weight==0:return base,detail
        j=int(np.abs(self.times-time).argmin());d=self.region_reference['frames'][j]['dynamic_region'];ids=self.region_indices
        z=torch_geometry(x.double(),self.region_reference['landmarks_A'],self.region_reference['origin_A'])[:,ids]
        scale=np.asarray(d['feature_scale'])[ids]
        positive=np.asarray(d['positive_centers'])[:,ids]/scale
        negative=np.asarray(d['negative_centers'])[:,ids]/scale
        variance=.5*(np.asarray(d['positive_variance'])[ids]+np.asarray(d['negative_variance'])[ids])
        aux,contrast=regional_response(z/z.new_tensor(scale),positive,negative,variance,
            self.region_temperature,self.region_bound,self.region_kind)
        detail.update(base_endpoint_reward=base.detach(),regional_reward=aux.detach(),regional_log_contrast=contrast.detach(),
            regional_weight=x.new_full((len(x),),self.region_weight))
        return base+self.region_weight*aux,detail
