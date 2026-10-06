"""Piecewise target trajectories and a correlated robust mixture reward.

Time selects frozen numerical references. Only world coordinates are optimized;
hard atom categories are detached. This is an empirical imitation potential.
"""
import math
import numpy as np
from scipy.interpolate import PchipInterpolator
import torch


class MultistageReward:
    def __init__(self, program, catalog):
        if program.get('schema_version') != 'live-multistage-1.0':
            raise ValueError('Unknown multistage program')
        if program['representation'] != 'predicted_endpoint_world_A':
            raise ValueError('Wrong coordinate frame')
        self.program, self.catalog = program, catalog
        self.features = program['features']; self.times = np.asarray(program['knot_times'], float)
        self.centers = np.asarray(program['knot_centers_A'], float)
        self.cov_times = np.asarray(program['covariance_times'], float)
        self.covariance = np.asarray(program['covariance_A2'], float)
        self.weights = np.asarray(program['mixture_weights'], float)
        self.temperature = float(program['mixture_temperature'])
        self.delta = float(program['pseudo_huber_delta'])
        if len(self.features) != 2 or len(set(self.features)) != 2:
            raise ValueError('This program requires the correlated two-feature patch')
        if self.centers.shape != (len(self.times), len(self.weights), len(self.features)):
            raise ValueError('Reference dimensions disagree')
        if self.covariance.shape != (len(self.cov_times), len(self.features), len(self.features)):
            raise ValueError('Covariance dimensions disagree')
        for times in (self.times, self.cov_times):
            if times[0] != 0 or times[-1] != .5 or np.any(np.diff(times) <= 0):
                raise ValueError('References must cover complete [0,.5]')
        if not all(np.isfinite(a).all() for a in (self.centers,self.covariance,self.weights)):
            raise ValueError('Nonfinite reference')
        if not np.allclose(self.covariance, self.covariance.swapaxes(-1,-2)) or np.min(np.linalg.eigvalsh(self.covariance)) <= 0:
            raise ValueError('Covariance must be symmetric positive definite')
        if np.any(self.weights <= 0) or not np.isclose(self.weights.sum(), 1):
            raise ValueError('Invalid mixture weights')
        if not math.isfinite(self.temperature) or not math.isfinite(self.delta) or min(self.temperature,self.delta) <= 0:
            raise ValueError('Invalid robust mixture parameters')
        for feature in self.features:
            spec = catalog['features'][feature]
            if spec['kind'] != 'hetero_distance_softmin' or set(spec['elements']) != {'N','O','S'}:
                raise ValueError('Unsupported feature type')
        self.interpolator = PchipInterpolator(self.times, self.centers, axis=0, extrapolate=False)

    def references(self, time, static=False):
        if not math.isfinite(float(time)) or not 0 <= round(float(time),6) <= 1:
            raise ValueError('Time outside control domain')
        t = .5 if static else min(round(float(time), 6), .5)
        # Convex interpolation of SPD matrices; no elementwise cubic interpolation.
        j = min(np.searchsorted(self.cov_times, t, side='right')-1, len(self.cov_times)-2)
        u = (t-self.cov_times[j])/(self.cov_times[j+1]-self.cov_times[j])
        covariance = (1-u)*self.covariance[j] + u*self.covariance[j+1]
        return self.interpolator(t), covariance

    def observables(self, x, atomics, mask):
        mask = mask.bool(); atomics = atomics.detach()
        if atomics.ndim == 3: atomics = atomics.argmax(-1)
        if not bool(mask.any(1).all()) or not bool(torch.isfinite(x).all()):
            raise ValueError('Empty or nonfinite ligand')
        columns = []; valid_all = torch.ones(len(x), dtype=torch.bool, device=x.device)
        for feature in self.features:
            spec = self.catalog['features'][feature]
            eligible = torch.zeros_like(mask)
            for element in spec['elements']:
                eligible |= atomics == self.catalog['atom_vocabulary'][element]
            eligible &= mask; valid = eligible.any(1); valid_all &= valid
            # Safe numerical fallback is masked out of the final reward, never
            # interpreted as a real hetero feature or assigned a favorable value.
            numerical_mask = torch.where(valid[:,None], eligible, mask)
            points = x.new_tensor(self.catalog['regions'][spec['region']]['points_A'])
            d = ((x[:,:,None,:]-points[None,None,:,:]).square().sum(-1)+1e-20).sqrt()
            tau = spec['temperature_A']
            logits = (-d/tau).masked_fill(~numerical_mask[...,None], -torch.inf)
            z = -tau*(torch.logsumexp(logits.flatten(1), 1) -
                      (numerical_mask.sum(1).to(x.dtype)*len(points)).log())
            columns.append(z)
        return torch.stack(columns, -1), valid_all

    def from_features(self, z, valid, time, static=False):
        centers, covariance = self.references(time, static)
        residual = z[:,None,:] - z.new_tensor(centers)[None,:,:]
        precision = z.new_tensor(np.linalg.inv(covariance))
        q = torch.einsum('bmf,fg,bmg->bm', residual, precision, residual).clamp_min(0)
        cost = self.delta**2*(torch.sqrt(1+q/self.delta**2)-1)
        logits = z.new_tensor(np.log(self.weights))[None,:] - cost/self.temperature
        reward = self.temperature*torch.logsumexp(logits, 1)
        reward = torch.where(valid, reward, z.sum(1)*0)
        return reward, {'responsibilities': logits.softmax(1), 'nearest_mahalanobis': q.min(1).values.sqrt()}

    def __call__(self, x, atomics, mask, time, static=False):
        z, valid = self.observables(x, atomics, mask)
        reward, detail = self.from_features(z, valid, time, static)
        return reward, z, valid, detail


def native_relative_step(gradient, native, mask, ratio, scale, max_atom_A, remaining_rms_A):
    """Positive scalar preconditioning of ascent, preserving its direction.

    Per-atom RMS is used, so batch size or ligand atom count do not scale eta.
    No motion is forced for zero gradients or a stationary native step.
    """
    if not all(math.isfinite(v) for v in (ratio,scale,max_atom_A)) or ratio < 0 or min(scale,max_atom_A)<=0:
        raise ValueError('Invalid control strength')
    mask = mask.bool(); n = mask.sum(1).clamp_min(1)
    g = gradient*mask[...,None]
    g_rms = torch.sqrt(g.square().sum((1,2))/n)
    native_rms = torch.sqrt((native.square()*mask[...,None]).sum((1,2))/n)*scale
    desired = ratio*native_rms
    coefficient = torch.where(g_rms > 1e-8, desired/(scale*g_rms.clamp_min(1e-8)), 0.)
    delta = coefficient[:,None,None]*g
    atom_factor = (max_atom_A/(delta.norm(dim=-1).amax(1)*scale).clamp_min(1e-30)).clamp(max=1)
    rms = torch.sqrt(delta.square().sum((1,2))/n)*scale
    path_factor = (remaining_rms_A.clamp_min(0)/rms.clamp_min(1e-30)).clamp(max=1)
    factor = torch.minimum(atom_factor, path_factor)
    delta *= factor[:,None,None]
    return delta, {'native_rms_A':native_rms, 'gradient_rms_native':g_rms,
                   'requested_rms_A':desired, 'preconditioner':coefficient,
                   'cap_factor':factor}


def reject_new_severe_clashes(native_x, delta, mask, pocket_x, pocket_mask, scale, constraints):
    """Reject nonfinite updates or newly introduced severe receptor clashes.

    Ligand pair-distance changes are diagnostics only, never an acceptance
    condition. Atom identities, bonds and topology do not enter this guard.
    Historical pair-distance limits are intentionally ignored; replay original
    experiments with their recorded source commit.
    """
    mask = mask.bool(); pocket_mask = pocket_mask.bool()
    pair_mask = mask[:,:,None] & mask[:,None,:]
    receptor_mask = mask[:,:,None] & pocket_mask[:,None,:]
    before_pairs = torch.cdist(native_x,native_x)*scale
    before_receptor = torch.cdist(native_x,pocket_x)*scale
    accepted = torch.zeros(len(native_x),dtype=torch.bool,device=native_x.device)
    result = torch.zeros_like(delta); factor = torch.zeros(len(native_x),device=native_x.device,dtype=native_x.dtype)
    threshold = constraints['severe_receptor_clash_A']
    for attempt in range(constraints['backtrack_attempts']):
        fraction = .5**attempt
        candidate = native_x + delta*fraction
        receptor = torch.cdist(candidate,pocket_x)*scale
        new_clash = ((receptor<threshold)&(before_receptor>=threshold)&receptor_mask).any((1,2))
        okay = torch.isfinite(candidate).all((1,2)) & ~new_clash
        take = okay & ~accepted
        result = torch.where(take[:,None,None],delta*fraction,result)
        factor = torch.where(take,torch.full_like(factor,fraction),factor)
        accepted |= okay
        if bool(accepted.all()): break
    actual_pair_change = (
        torch.cdist(native_x+result,native_x+result)*scale-before_pairs).abs().masked_fill(~pair_mask,0).amax((1,2))
    return result, {'geometry_accepted':accepted, 'backtrack_factor':factor,
                    'max_actual_pair_distance_change_A':actual_pair_change}
