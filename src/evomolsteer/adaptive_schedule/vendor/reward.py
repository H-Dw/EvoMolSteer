"""Local pseudo-Huber/decoder-CE losses, with detached geometry-only matching."""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


class LocalReward:
    def __init__(self,template,features,mode):
        self.t=template;self.features=features;self.mode=mode
        self.core=list(template['core_atoms']);self.context=list(template['context_atoms'])
        self.core_columns=[self.context.index(i) for i in self.core]
        self.xyz=np.asarray(template['coords_A'])[self.context]
        self.atom_labels=np.asarray(template['atom_labels'])[self.core]
        self.charge_labels=np.asarray(template['charge_labels'])[self.core]
        bonds=np.asarray(template['bond_labels'])
        self.pairs=[(a,b) for a in range(len(self.context)) for b in range(a+1,len(self.context))
            if self.context[a] in self.core or self.context[b] in self.core]
        self.bond_labels=np.array([bonds[self.context[a],self.context[b]] for a,b in self.pairs],dtype=np.int64)
        if not self.core:raise ValueError('An empty PLIP core cannot guide')
        if set(features)-set('CABQ') or not features:raise ValueError('Unknown feature set')

    def assignment(self,z,mask):
        chosen=[]
        with torch.no_grad():
            for xyz,valid in zip(z.detach().cpu().numpy(),mask.detach().cpu().numpy()):
                active=np.flatnonzero(valid)
                if len(active)<len(self.context):raise ValueError('Too few generated atoms for injective local assignment')
                cost=((self.xyz[:,None,:]-xyz[None,active,:])**2).sum(-1)
                rows,cols=linear_sum_assignment(cost)
                chosen.append(active[cols[np.argsort(rows)]])
        return torch.as_tensor(np.asarray(chosen),device=z.device,dtype=torch.long)

    def terms(self,current_world,endpoint_world,out,mask,assignment=None):
        z=current_world if self.mode=='current' else endpoint_world
        ids=self.assignment(z,mask) if assignment is None else assignment
        core_ids=ids[:,self.core_columns];batch=torch.arange(len(z),device=z.device)[:,None]
        terms={}
        if 'C' in self.features:
            matched=z[batch,core_ids];reference=z.new_tensor(self.t['coords_A'])[self.core]
            square=(matched-reference).square().sum(-1)
            # Algebraically identical pseudo-Huber; avoids sqrt(1+e^2)-1
            # cancellation close to a matched template site.
            terms['C']=(square/(torch.sqrt(1+square)+1)).mean(1)
        for name,key,labels in [('A','atomics',self.atom_labels),('Q','charges',self.charge_labels)]:
            if name not in self.features:continue
            logits=out[key][batch,core_ids]
            if logits.dtype in (torch.float16,torch.bfloat16):logits=logits.float()
            target=torch.as_tensor(labels,device=z.device,dtype=torch.long)[None,:,None].expand(len(z),-1,1)
            terms[name]=-logits.log_softmax(-1).gather(-1,target).squeeze(-1).mean(1)
        if 'B' in self.features:
            if not self.pairs:terms['B']=current_world.sum((1,2))*0
            else:
                p=torch.as_tensor(self.pairs,device=z.device,dtype=torch.long)
                left=ids[:,p[:,0]];right=ids[:,p[:,1]]
                # Bond head is symmetric in the native model; average logits
                # explicitly to give each unordered relationship one weight.
                logits=(out['bonds'][batch,left,right]+out['bonds'][batch,right,left])*.5
                if logits.dtype in (torch.float16,torch.bfloat16):logits=logits.float()
                labels=torch.as_tensor(self.bond_labels,device=z.device,dtype=torch.long)
                target=labels[None,:,None].expand(len(z),-1,1)
                per=-logits.log_softmax(-1).gather(-1,target).squeeze(-1)
                groups=[per[:,selection].mean(1) for selection in [labels>0,labels==0] if bool(selection.any())]
                terms['B']=torch.stack(groups).mean(0)
        direct=torch.zeros_like(mask,dtype=torch.bool);direct.scatter_(1,core_ids,True)
        return terms,ids,direct

    def loss(self,current_world,endpoint_world,out,mask,assignment=None):
        terms,ids,direct=self.terms(current_world,endpoint_world,out,mask,assignment)
        return sum(terms.values()),terms,ids,direct


def strength(schedule,step):
    return step/99 if schedule=='warmup' else float(schedule.split('_')[1])


def displacement(gradient_native,dt,lam,coord_scale,mask):
    # Reward uses world Angstrom / 1 A. Convert its native-coordinate derivative
    # to a world-coordinate gradient and convert the update back to model units.
    delta_native=lam*dt*gradient_native/(coord_scale**2)*mask[...,None]
    lengths=delta_native.norm(dim=-1)*coord_scale
    factor=(.025/lengths.amax(1).clamp_min(1e-30)).clamp(max=1)
    return delta_native*factor[:,None,None],factor
