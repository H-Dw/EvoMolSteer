"""Analytic, permutation-invariant rewards for actual states on a learned grid.

No fitted neural network, atom correspondence, endpoint-to-current relabeling,
or physical-energy interpretation. Coordinate gradients act on native proposals.
"""
import math
import numpy as np
import torch
from .window_reference import load_reference


class WindowReward:
    def __init__(self, program, reference):
        self.program = program
        self.reference = load_reference(reference) if isinstance(reference, (str, bytes)) else reference
        self.window = tuple(self.reference['window'])
        if list(self.window) != program['window']:
            raise ValueError('Reward support must equal learning support')
        self.times = np.asarray(self.reference['times'])
        self.frames = {float(t): [r for r in self.reference['frames'] if abs(r['time']-t)<1e-7] for t in self.times}
        self.cache = {}
        if self.window[0] < 0 or self.window[1] > 1 or self.window[0] >= self.window[1]:
            raise ValueError('Invalid learned window')
        for k in ['native_rms_ratio', 'typed_weight', 'bond_weight']:
            if not math.isfinite(program[k]) or program[k] < 0: raise ValueError(k)
        if program['temperature'] <= 0 or min(program['sigmas_A']) <= 0: raise ValueError('Invalid kernel')

    def active(self, score_time, state_time):
        a,b=self.window
        return score_time >= a-1e-6 and state_time <= b+1e-6 and state_time > score_time

    def bank(self, time, x):
        index = int(np.abs(self.times-time).argmin())
        if abs(self.times[index]-time)>2e-6: raise ValueError('No exact-time reference; coordinate interpolation is forbidden')
        key=(index,str(x.device),x.dtype)
        if key not in self.cache:
            frames=self.frames[float(self.times[index])]
            y=x.new_tensor([r['coords'] for r in frames])
            a=torch.tensor([r['atomics'] for r in frames],device=x.device)
            bonds=torch.tensor([r['bonds'] for r in frames],device=x.device)
            mass=x.new_tensor([r['mass'] for r in frames]);mass=mass/mass.sum()
            self.cache[key]=(y,a,bonds,mass)
        return self.cache[key]

    @staticmethod
    def bond_moments(x, bonds):
        d=torch.cdist(x,x)
        eye=torch.eye(x.shape[1],device=x.device,dtype=torch.bool)[None]
        columns=[]
        for kind in range(1,5):
            w=((bonds==kind)&~eye).to(x.dtype)
            n=w.sum((1,2)); valid=n>0
            # Raw first and second distance moments, scaled to dimensionless.
            for order in (1,2):
                v=((d/3).pow(order)*w).sum((1,2))/n.clamp_min(1)
                columns.append(torch.where(valid,v,0.))
        return torch.stack(columns,1)

    def __call__(self,x,atoms,bonds,time):
        y,ya,yb,mass=self.bank(time,x)
        if x.shape[1]!=y.shape[1]:raise ValueError('Reference atom count mismatch')
        xx=(x[:,:,None,:]-x[:,None,:,:]).square().sum(-1)
        yy=(y[:,:,None,:]-y[:,None,:,:]).square().sum(-1)
        xy=(x[:,None,:,None,:]-y[None,:,None,:,:]).square().sum(-1)
        shape=0.;typed=0.
        for sigma in self.program['sigmas_A']:
            kxx=torch.exp(-xx/(2*sigma*sigma));kyy=torch.exp(-yy/(2*sigma*sigma));kxy=torch.exp(-xy/(2*sigma*sigma))
            shape=shape+kxx.mean((1,2))[:,None]+kyy.mean((1,2))[None]-2*kxy.mean((2,3))
            if self.program['typed_weight']:
                axx=(atoms[:,:,None]==atoms[:,None,:]);ayy=(ya[:,:,None]==ya[:,None,:]);axy=(atoms[:,None,:,None]==ya[None,:,None,:])
                typed=typed+(kxx*axx).mean((1,2))[:,None]+(kyy*ayy).mean((1,2))[None]-2*(kxy*axy).mean((2,3))
        shape=shape/len(self.program['sigmas_A']);typed=typed/len(self.program['sigmas_A'])
        bond=x.new_zeros(shape.shape)
        if self.program['bond_weight']:
            bx=self.bond_moments(x,bonds);by=self.bond_moments(y,yb)
            bond=(bx[:,None,:]-by[None,:,:]).square().mean(-1)
        cost=shape+self.program['typed_weight']*typed+self.program['bond_weight']*bond
        temperature=self.program['temperature']
        logits=mass.log()[None]-cost/temperature
        reward=temperature*torch.logsumexp(logits,1)
        responsibility=logits.softmax(1)
        return reward, {'shape_loss':(responsibility*shape).sum(1),
                        'bond_loss':(responsibility*bond).sum(1),
                        'responsibility_ess':1/responsibility.square().sum(1),
                        'nearest_shape_loss':shape.min(1).values}

    def ratio(self,state_time):
        u=(state_time-self.window[0])/(self.window[1]-self.window[0])
        slope=self.program.get('schedule_slope',0.)
        if abs(slope)>1:raise ValueError('Linear schedule must remain nonnegative')
        return self.program['native_rms_ratio']*(1+slope*(2*u-1))
