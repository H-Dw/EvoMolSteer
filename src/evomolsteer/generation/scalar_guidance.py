"""Small, data-only reward basis and live coordinate pullback.

No fitted network, Python evaluation, population selection or affinity-head reward.
The host must supply the live endpoint forward; an offline endpoint is insufficient.
"""
import math
import torch


def detached(value):
    if torch.is_tensor(value): return value.detach()
    if isinstance(value, dict): return {k:detached(v) for k,v in value.items()}
    if isinstance(value, tuple): return tuple(detached(v) for v in value)
    if isinstance(value, list): return [detached(v) for v in value]
    return value


def time_gate(t, start, end, width):
    t = round(float(t), 6)
    if not start <= t < end: return 0.
    def smooth(v):
        v = min(1., max(0., v))
        return v*v*(3-2*v)
    return (1. if start == 0 else smooth((t-start)/width))*smooth((end-t)/width)


def positive_huber(u):
    """C1 one-sided penalty; zero below target, bounded derivative above it."""
    v = u.clamp_min(0)
    return torch.where(v <= 1, .5*v.square(), v-.5)


class RegionalReward:
    def __init__(self, program, catalog):
        if program.get('schema_version') != 'live-regional-1.0': raise ValueError('Unknown live reward schema')
        if program.get('representation') != 'predicted_endpoint_world_A': raise ValueError('Wrong frame')
        self.program, self.catalog = program, catalog
        self.terms = program['terms']
        if not self.terms: raise ValueError('No reward terms')
        ids = [t['id'] for t in self.terms]
        if len(ids) != len(set(ids)): raise ValueError('Duplicate term ids')
        for t in self.terms:
            vals = [t[k] for k in ['target','scale','weight','stage_start','stage_end','gate_width']]
            if not all(type(v) in (int,float) and math.isfinite(v) for v in vals): raise ValueError('Nonfinite term')
            if not 0 <= t['stage_start'] < t['stage_end'] <= .5: raise ValueError('Term outside selection window')
            if min(t['scale'],t['weight'],t['gate_width']) <= 0: raise ValueError('Nonpositive scale/weight')
            if t['operator'] != 'decrease_until_target': raise ValueError('Unsupported operator')
            spec = catalog['features'][t['feature']]
            if spec.get('elements') or spec['kind'] not in ('distance_softmin','radius_gyration'):
                raise ValueError('This live basis supports all-active-atom geometry only')

    def active(self, time, term_ids):
        known = {t['id'] for t in self.terms}
        if not set(term_ids) <= known: raise ValueError('Unknown selected term')
        return [(t,time_gate(time,t['stage_start'],t['stage_end'],t['gate_width'])) for t in self.terms
                if t['id'] in term_ids and time_gate(time,t['stage_start'],t['stage_end'],t['gate_width']) > 0]

    def observable(self, feature, x, mask):
        spec = self.catalog['features'][feature]
        mask = mask.bool()
        n = mask.sum(-1)
        if not bool((n>0).all()): raise ValueError('Empty ligand')
        if spec['kind'] == 'radius_gyration':
            center = (x*mask[...,None]).sum(1)/n[:,None]
            return (((x-center[:,None,:]).square().sum(-1)*mask).sum(-1)/n+1e-20).sqrt()
        points = x.new_tensor(self.catalog['regions'][spec['region']]['points_A'])
        tau = spec['temperature_A']
        dist = ((x[:,:,None,:]-points[None,None,:,:]).square().sum(-1)+1e-20).sqrt()
        logits = (-dist/tau).masked_fill(~mask[...,None],-torch.inf)
        return -tau*(torch.logsumexp(logits.flatten(1),1)-(n.to(x.dtype)*len(points)).log())

    def __call__(self, x, mask, time, term_ids):
        reward = x.sum((1,2))*0
        detail = {}
        for t,gate in self.active(time,term_ids):
            z = self.observable(t['feature'],x,mask)
            value = -t['weight']*gate*positive_huber((z-t['target'])/t['scale'])
            reward = reward+value
            detail[t['id']] = {'observable':z.detach().cpu().tolist(),'reward':value.detach().cpu().tolist(),'gate':gate}
        return reward, detail


def live_pullback(predict, reward, current, mask, scale, com, time, term_ids):
    """Return detached native prediction and J_endpoint^T grad_world R per particle."""
    with torch.enable_grad():
        x = current.detach().clone().requires_grad_(True)
        pred, cond = predict(x)
        value, detail = reward(pred['coords']*scale+com[:,None,:],mask,time,term_ids)
        if value.shape != (len(x),) or not bool(torch.isfinite(value).all()): raise ValueError('Invalid reward')
        # Sum, not mean: another batch member must not change this particle's scale.
        grad, = torch.autograd.grad(value.sum()+x.sum()*0,x)
    if not bool(torch.isfinite(grad).all()): raise ValueError('Nonfinite live gradient')
    return detached(pred), detached(cond), grad.detach()*mask[...,None], value.detach(), detail


def bounded_displacement(gradient, dt, weight, coord_scale, max_atom_A):
    if not all(math.isfinite(v) for v in (dt,weight,coord_scale,max_atom_A)) or dt<=0 or weight<0 or min(coord_scale,max_atom_A)<=0:
        raise ValueError('Invalid injection parameters')
    delta = weight*dt*gradient
    lengths = delta.norm(dim=-1)*coord_scale
    # One scalar per molecule preserves its gradient direction and relative atom moves.
    factor = (max_atom_A/lengths.amax(1).clamp_min(1e-30)).clamp(max=1)
    return delta*factor[:,None,None], factor
