"""M2-style robust spatial matching against a real intermediate teacher."""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def bridge(current,endpoint,time,teacher_time):
    if not 0<=time<=teacher_time<1:raise ValueError('Bridge clocks outside linear-flow support')
    a=(1-teacher_time)/(1-time);b=(teacher_time-time)/(1-time)
    return a*current+b*endpoint


def active_updates(steps,start,end):
    grid=np.linspace(0,1,steps+1)
    return [i for i in range(steps) if grid[i]>=start-1e-8 and grid[i+1]<=end+1e-8]


class HalfStateReward:
    def __init__(self,program,teacher):
        if program['family']!='actual_half_state_pointcloud' or teacher['representation']!='current_world_A':raise ValueError('Actual intermediate teacher required')
        if program['teacher_time']!=teacher['time'] or program['candidate_id']!=teacher['candidate_id']:raise ValueError('Teacher binding mismatch')
        self.program=program;self.teacher=teacher;self.slots=list(program['atom_slots'])
        if not self.slots or len(set(self.slots))!=len(self.slots) or not set(self.slots)<=set(teacher['active_slots']):raise ValueError('Unsupported teacher slots')
        if program['scope']=='full' and self.slots!=teacher['active_slots']:raise ValueError('Whole-cloud control omitted active points')
        if program['scope']=='local' and len(self.slots)>=len(teacher['active_slots']):raise ValueError('Local mode must be a proper subset')
        self.target=np.asarray(teacher['coords_A'],float)[self.slots]
        weights=np.asarray(program.get('atom_weights',[1.]*len(self.slots)),float)
        if weights.shape!=(len(self.slots),) or not np.isfinite(weights).all() or (weights<=0).any():raise ValueError('Invalid analyst-bound teacher weights')
        self.weights=weights/weights.sum()
        self.delta=float(program['dose']['pointcloud_delta_A'])
        if not np.isfinite(self.target).all() or not np.isfinite(self.delta) or self.delta<=0:raise ValueError('Invalid teacher geometry or curvature')
    def assignment(self,z,mask):
        choices=[]
        for cloud,valid in zip(z.detach().cpu().numpy(),mask.detach().cpu().numpy()):
            ids=np.flatnonzero(valid)
            if len(ids)<len(self.target):raise ValueError('Too few active generated atoms for distinct spatial matching')
            distance=np.square(cloud[ids,None]-self.target[None]).sum(-1)*self.weights[None]
            rows,cols=linear_sum_assignment(distance)
            choices.append(ids[rows[np.argsort(cols)]])
        return torch.as_tensor(np.asarray(choices),device=z.device,dtype=torch.long)
    def value(self,z,mask,assignment=None):
        indices=self.assignment(z,mask) if assignment is None else assignment
        matched=z.gather(1,indices[...,None].expand(-1,-1,3));target=z.new_tensor(self.target)
        squared=(matched-target[None]).square().sum(-1)
        q=squared.mean(1) if 'atom_weights' not in self.program else (squared*z.new_tensor(self.weights)).sum(1)
        value=-self.delta**2*(torch.sqrt(1+q/self.delta**2)-1)
        direct=torch.zeros_like(mask,dtype=torch.bool);direct.scatter_(1,indices,True)
        return value,{'assignment':indices,'direct_mask':direct,'matched_rms_A':q.sqrt()}
    def active(self,t,dt):
        start,end=self.program['window']
        return t>=start-2e-6 and t+dt<=end+2e-6 and dt>0
    def evaluate(self,current,endpoint,mask,time):
        return self.value(bridge(current,endpoint,time,self.teacher['time']),mask)
