"""Static original-pose / selected-teacher mixture, with optional late contact.

These are proposed geometrical controls, not fitted affinity or energy models.
Correspondences, neighbor selection and distance priors are detached. Mixture
responsibilities remain differentiable within that conditional scalar.
"""
import math
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from .coordinate_reward import CoordinateMixtureReward


class StaticHybridReward(CoordinateMixtureReward):
    def __init__(self, program, reference):
        self.program, self.reference = program, reference
        self.window = reference['window']
        if reference['schema_version'] != 'static-hybrid-coordinate-library-1.0':
            raise ValueError('Explicit static hybrid reference required')
        if program['window'] != self.window or not 0 <= self.window[0] < self.window[1] <= 1:
            raise ValueError('Window mismatch')
        self.rho = float(program['original_pose_prior_mass'])
        self.tau = float(program.get('mixture_temperature', .5))
        self.delta = float(program.get('pointcloud_delta_A', 1))
        self.prior_scale = float(program.get('teacher_endpoint_temperature_A2', 4))
        self.contact = float(program.get('contact_weight', 0))
        self.contact_power = float(program.get('contact_phase_power', 0))
        self.repulsion = float(program.get('repulsion_weight', 0))
        self.ramp = float(program.get('time_ramp_power', 0))
        self.neighbors = int(program.get('teacher_neighbors', 4))
        values = [self.rho, self.tau, self.delta, self.prior_scale, self.contact,
                  self.contact_power, self.repulsion, self.ramp]
        if (not all(math.isfinite(v) for v in values) or not 0 <= self.rho <= 1
                or min(self.tau, self.delta, self.prior_scale) <= 0
                or min(self.contact, self.contact_power, self.repulsion, self.ramp) < 0
                or self.neighbors < 1):
            raise ValueError('Invalid static hybrid parameters')
        self.original = np.asarray(reference['bound_ligand_A'], dtype=float)
        self.teachers = np.asarray(reference['teacher_bank_A'], dtype=float)
        if (self.original.ndim != 2 or self.original.shape[1] != 3
                or self.teachers.ndim != 3 or self.teachers.shape[1:] != self.original.shape
                or not len(self.teachers) or not np.isfinite(self.original).all()
                or not np.isfinite(self.teachers).all()):
            raise ValueError('Finite equally sized source point clouds required')

    def __call__(self, x, atoms, mask, time, anchor=None):
        if anchor is None or anchor.shape != x.shape:
            raise ValueError('Detached forecast correspondence required')
        mask = mask.bool(); weight = mask.to(x.dtype); n = weight.sum(1).clamp_min(1)
        matched, log_priors, original_flags = [], [], []
        for cloud, active in zip(anchor.detach().cpu().numpy(), mask.cpu().numpy()):
            ids = np.flatnonzero(active)
            if len(ids) > len(self.original):
                raise ValueError('Source point clouds have fewer atoms than generation')
            def align(target):
                distances = ((cloud[ids, None] - target[None]) ** 2).sum(-1)
                rows, cols = linear_sum_assignment(distances)
                aligned = np.zeros_like(cloud); aligned[ids[rows]] = target[cols]
                return aligned, float(distances[rows, cols].mean()) if len(ids) else 0.
            candidates, priors, labels = [], [], []
            if self.rho > 0:
                candidates.append(align(self.original)[0]); priors.append(math.log(self.rho)); labels.append(True)
            if self.rho < 1:
                aligned = [align(t) for t in self.teachers]
                costs = np.asarray([a[1] for a in aligned])
                selected = np.argsort(costs, kind='stable')[:min(self.neighbors, len(costs))]
                logits = -costs[selected] / self.prior_scale
                logits = logits - np.logaddexp.reduce(logits) + math.log1p(-self.rho)
                candidates.extend(aligned[i][0] for i in selected)
                priors.extend(logits.tolist()); labels.extend([False] * len(selected))
            matched.append(candidates); log_priors.append(priors); original_flags.append(labels)
        target = x.new_tensor(np.asarray(matched))
        priors = x.new_tensor(np.asarray(log_priors)).detach()
        original_flags = torch.as_tensor(original_flags, device=x.device)
        residual = x[:, None] - target
        q = (residual.square().sum(-1) * weight[:, None]).sum(-1) / n[:, None]
        cost = self.delta ** 2 * (torch.sqrt(1 + q / self.delta ** 2) - 1)
        logits = priors - cost / self.tau
        geometry = self.tau * torch.logsumexp(logits, 1)
        posterior = logits.softmax(1)
        phase = float(np.clip((time - self.window[0]) / (self.window[1] - self.window[0]), 0, 1))
        contact_weight = self.contact * phase ** self.contact_power
        contact = x.sum((1, 2)) * 0; repulsion = contact
        contact_gradient = torch.zeros_like(x)
        if self.contact or self.repulsion:
            protein = x.new_tensor(self.reference['protein_points_A'])
            displacement = x[:, :, None] - protein[None, None]
            d = displacement.square().sum(-1).clamp_min(1e-12).sqrt()
            shell = torch.exp(-.5 * ((d - 3.6) / .6).square())
            contact = ((1 - torch.exp(-shell.sum(-1) / 4)) * weight).sum(1) / n
            repulsion = (torch.relu(2 - d).square().sum(-1) * weight).sum(1) / n
            # Analytic diagnostic only: no extra model forward or VJP.
            shell_derivative = -shell * (d - 3.6) / .6 ** 2
            contact_gradient = (shell_derivative[..., None] * displacement / d[..., None]).sum(2)
            contact_gradient *= (torch.exp(-shell.sum(-1) / 4) / 4 * weight / n[:, None])[..., None]
        reward = geometry + contact_weight * contact - self.repulsion * repulsion
        valid = mask.any(1) & (self.window[0] - 1e-6 <= time <= self.window[1] + 1e-6)
        # Analytic endpoint components quantify cancellation without extra VJPs.
        components = -residual * weight[:, None, :, None] / n[:, None, None, None]
        components = components / torch.sqrt(1 + q / self.delta ** 2)[:, :, None, None]
        mixed = (components * posterior[:, :, None, None]).sum(1)
        denominator = (components.norm(dim=(2, 3)) * posterior).sum(1)
        cancellation = mixed.norm(dim=(1, 2)) / denominator.clamp_min(1e-30)
        geometry_norm = mixed.norm(dim=(1, 2))
        contact_norm = contact_gradient.norm(dim=(1, 2))
        contact_share = contact_weight * contact_norm / (geometry_norm + contact_weight * contact_norm).clamp_min(1e-30)
        contact_cosine = (mixed * contact_gradient).sum((1, 2)) / (geometry_norm * contact_norm).clamp_min(1e-30)
        schedule = (.1 + .9 * phase) ** self.ramp
        gate = valid.to(x.dtype) * schedule
        return torch.where(valid, reward, x.sum((1, 2)) * 0), {
            'available': valid, 'core_mask': mask, 'dose_gate': gate, 'time_dose_factor': gate,
            'observables': torch.stack([geometry, (posterior * original_flags).sum(1),
                                        q.amin(1).sqrt(), contact, repulsion], 1),
            'nearest_standardized_rms': q.amin(1).sqrt(),
            'hybrid_original_posterior_mass': (posterior * original_flags).sum(1).detach(),
            'hybrid_posterior_ess': (1 / posterior.square().sum(1)).detach(),
            'hybrid_endpoint_cancellation_ratio': cancellation.detach(),
            'hybrid_endpoint_contact_gradient_share': contact_share.detach(),
            'hybrid_endpoint_geometry_contact_cosine': contact_cosine.detach(),
            'hybrid_contact_weight': x.new_full((len(x),), contact_weight)}
