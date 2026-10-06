"""Supported endpoint coordinate fields; no head calls or learned predictor."""
import numpy as np
import torch
from .endpoint_reward import EndpointGeometryReward
from ..continuous.affinity_geometry import torch_geometry


class SparseEndpointReward(EndpointGeometryReward):
    def __init__(self,program,reference):
        if program['reward_view']!='endpoint_direction':raise ValueError('Sparse fields currently register endpoint_direction only')
        super().__init__(program,reference)
        weights=np.asarray(program['geometry_feature_weights'],float)
        if weights.shape!=self.scale.shape or not np.isfinite(weights).all() or (weights<0).any() or not (weights>0).any():
            raise ValueError('Invalid exact-feature weights')
        blocks=program['geometry_block_weights'];cuts=[0,3,9,14,len(weights)];normalized=np.zeros_like(weights)
        for k,(a,b) in enumerate(zip(cuts[:-1],cuts[1:])):
            if weights[a:b].sum()>0:normalized[a:b]=blocks[k]*weights[a:b]/weights[a:b].sum()
        if not normalized.any():raise ValueError('No effective nonzero coordinate block')
        self.field_weights=normalized;self.mode=program.get('geometry_direction_mode','node_contrast')
        if self.mode not in ('node_contrast','legendre_effect','linear_node_effect'):raise ValueError('Unregistered coordinate direction mode')
        self.functions=program.get('geometry_direction_functions',{})
        selected={f for f,w in zip(reference['features'],weights) if w>0}
        if self.mode=='legendre_effect':
            audits=program.get('geometry_curve_fidelity_audit',{})
            if any(not audits.get(f,{}).get('legendre_runtime_allowed',False) for f in selected):
                raise ValueError('Polynomial fields require recorded passing node-fidelity certificates')
            if set(self.functions)!=selected:raise ValueError('Each selected field needs an exact recorded function')
            for function in self.functions.values():
                degree=function['degree'];coeff=function['legendre_coefficients']
                if not 0<=degree<=3 or len(coeff)!=degree+1 or not np.isfinite(coeff).all():raise ValueError('Invalid trend coefficients')
                if function['time_start']!=self.times[0] or function['time_end']!=self.times[-1]:raise ValueError('Trend support mismatch')
        elif self.mode=='linear_node_effect':
            if set(self.functions)!=selected or program['geometry_direction_times']!=self.times.tolist():raise ValueError('Exact empirical field/time support required')
            for f in self.functions.values():
                if len(f['values_z'])!=len(self.times) or not np.isfinite(f['values_z']).all():raise ValueError('Invalid empirical curve values')
        elif self.functions:raise ValueError('Node mode has unused functions')

    def __call__(self,x,atoms,mask,time,anchor=None):
        j=int(np.abs(self.times-time).argmin())
        if abs(self.times[j]-time)>2e-6 or not bool(mask.all()):raise ValueError('Recorded node and fixed active slots required')
        frame=self.reference['frames'][j];z=torch_geometry(x.double(),self.reference['landmarks_A'],self.reference['origin_A'])
        scaled=z/z.new_tensor(self.scale);hi=scaled.new_tensor(frame['high_scaled']);lo=scaled.new_tensor(frame['low_scaled'])
        variance=scaled.new_tensor(frame['variance_scaled'])
        if self.mode=='node_contrast':direction=((hi-lo)/variance).mean(0)
        else:
            values=[]
            for f in self.reference['features']:
                function=self.functions.get(f)
                if function is None:values.append(0.);continue
                if self.mode=='linear_node_effect':
                    values.append(float(np.interp(float(time),self.times,function['values_z'])))
                    continue
                u=2*(self.times[j]-function['time_start'])/(function['time_end']-function['time_start'])-1
                values.append(float(np.polynomial.legendre.legval(u,function['legendre_coefficients'])))
            direction=scaled.new_tensor(values)/variance.mean(0)
        direction=direction.clamp(-3,3)*scaled.new_tensor(self.field_weights)
        value=(scaled*direction).sum(1);nearest=((scaled[:,None]-hi).square()/variance).mean(2).amin(1).sqrt()
        phase=(time-self.window[0])/(self.window[1]-self.window[0])
        schedule=(.1+.9*np.clip(phase,0,1))**float(self.program.get('time_ramp_power',0.))
        return value,{'observables':z,'available':mask.any(1),'core_mask':mask.bool(),
            'nearest_standardized_rms':nearest,'dose_gate':x.new_full((len(x),),schedule),
            'time_dose_factor':x.new_full((len(x),),schedule)}
