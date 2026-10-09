"""Teacher-free coordinate hypotheses; geometrical proxies, not binding energies.

Correspondences are conditional on a detached FLOWR endpoint forecast. The
bound ligand is a structural prior, explicitly distinct from generated Steer.
"""
import math
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from .coordinate_reward import CoordinateMixtureReward


class StructureFieldReward(CoordinateMixtureReward):
    def __init__(self, program, reference):
        self.program, self.reference = program, reference
        self.window = reference['window']
        if reference['schema_version'] != 'structure-field-reference-1.0':
            raise ValueError('Explicit structure-only reference required')
        if program['window'] != self.window or reference.get('steer_sources'):
            raise ValueError('Window mismatch or historical source contamination')
        self.weights = [float(program.get(k, 0)) for k in
                        ['ligand_anchor_weight', 'contact_weight', 'repulsion_weight']]
        if not all(math.isfinite(w) and w >= 0 for w in self.weights) or sum(self.weights) <= 0:
            raise ValueError('Finite positive geometric hypothesis required')
        if self.weights[0] and not reference.get('bound_ligand_A'):
            raise ValueError('Bound ligand unavailable')

    def __call__(self, x, atoms, mask, time, anchor=None):
        mask = mask.bool()
        weight = mask.to(x.dtype)
        n = weight.sum(1).clamp_min(1)
        points = x.new_tensor(self.reference['protein_points_A'])
        # Heavy-atom centre distances, deliberately NOT a typed Vina energy.
        d = (x[:, :, None] - points[None, None]).square().sum(-1).clamp_min(1e-12).sqrt()
        shell = torch.exp(-0.5 * ((d - 3.6) / 0.6).square())
        # Saturate local coordination so dense receptor regions do not get
        # unlimited reward solely through protein atom multiplicity.
        contact_per_atom = 1 - torch.exp(-shell.sum(-1) / 4)
        contact = (contact_per_atom * weight).sum(1) / n
        repulsion = (torch.relu(2.0 - d).square().sum(-1) * weight).sum(1) / n
        q = x.sum((1, 2)) * 0
        if self.weights[0]:
            if anchor is None:
                raise ValueError('Detached forecast correspondence required')
            target = np.asarray(self.reference['bound_ligand_A'])
            matched = []
            for cloud, active in zip(anchor.detach().cpu().numpy(), mask.cpu().numpy()):
                ids = np.flatnonzero(active)
                if len(ids) > len(target):
                    raise ValueError('Reference has fewer atoms than generated cloud')
                costs = ((cloud[ids, None] - target[None]) ** 2).sum(-1)
                rows, cols = linear_sum_assignment(costs)
                aligned = np.zeros_like(cloud)
                aligned[ids[rows]] = target[cols]
                matched.append(aligned)
            q = ((x - x.new_tensor(np.asarray(matched))).square().sum(-1) * weight).sum(1) / n
        anchor_reward = -(torch.sqrt(1 + q) - 1)
        reward = self.weights[0] * anchor_reward + self.weights[1] * contact - self.weights[2] * repulsion
        valid = mask.any(1)
        phase = (time - self.window[0]) / (self.window[1] - self.window[0])
        schedule = (.1 + .9 * np.clip(phase, 0, 1)) ** float(self.program.get('time_ramp_power', 0))
        return reward, {'available': valid, 'core_mask': mask,
            'observables': torch.stack([anchor_reward, contact, repulsion], 1),
            'nearest_standardized_rms': q.sqrt(),
            'dose_gate': x.new_full((len(x),), schedule),
            'time_dose_factor': x.new_full((len(x),), schedule)}
