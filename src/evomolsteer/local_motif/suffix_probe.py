"""Restore full state groups into the native loop with new paired RNG streams."""
import copy
import numpy as np


def select_tree(value,slots,b,device):
    import torch
    if torch.is_tensor(value):
        return (value[slots] if value.ndim and value.shape[0]==b else value).clone().to(device)
    if isinstance(value,dict):return {k:select_tree(v,slots,b,device) for k,v in value.items()}
    if isinstance(value,list):return [copy.deepcopy(value[i]) for i in slots] if len(value)==b else [select_tree(v,slots,b,device) for v in value]
    if isinstance(value,tuple):return tuple(select_tree(v,slots,b,device) for v in value)
    return copy.deepcopy(value)


class GroupedSuffix:
    def __init__(self,state,slots,steps,device):
        self.state=state;self.slots=slots;self.steps=steps;self.device=device;self.executed=[]
        self.start_step=int(state['step']);self.b=len(state['current']['coords'])
    def restore(self):return tuple(select_tree(self.state[k],self.slots,self.b,self.device) for k in ['prior','current','cond','times'])
    def before(self,i,curr,times):
        if i!=self.start_step+len(self.executed) or abs(float(times[0][0])-i/self.steps)>3e-6:raise ValueError('Noncontiguous saved-state continuation')
        self.executed.append(i)
    def after(self,i,curr,times):
        if abs(float(times[0][0])-(i+1)/self.steps)>3e-6:raise ValueError('Continuation clock drift')
