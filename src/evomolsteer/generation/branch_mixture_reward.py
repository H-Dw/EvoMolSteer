"""Conditional natural-branch virtual modes, preserving every original teacher.

The confidence qualifies observed directions; it is not an Angstrom multiplier.
Only bounded virtual prior mass and confidence-conditional regional weights change.
Correspondence and all mixture priors remain frozen at the original endpoint.
"""
import copy
import math

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from .endpoint_reward import EndpointGeometryReward


class BranchMixtureReward(EndpointGeometryReward):
    def __init__(self, program, reference):
        if program.get('reward_view') != 'endpoint_branch_mixture':
            raise ValueError('Registered branch-mixture view required')
        self.spec = copy.deepcopy(program.get('branch_mixture', {}))
        limits = {'virtual_mass': (0., .5), 'direction_sign': (-1., 1.),
                  'region_weight_mix': (0., 1.)}
        if set(self.spec) - set(limits):
            raise ValueError('Unregistered branch-mixture parameter')
        for key, value in self.spec.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not limits[key][0] <= value <= limits[key][1]:
                raise ValueError('Finite bounded branch-mixture parameter required')
        self.mass = float(self.spec.get('virtual_mass', 0.))
        self.sign = float(self.spec.get('direction_sign', 1.))
        self.region_mix = float(self.spec.get('region_weight_mix', 0.))
        baseline = copy.deepcopy(program)
        baseline['reward_view'] = 'endpoint_pointcloud'
        super().__init__(baseline, reference)

    def _fields(self, frame, node_index=None):
        centers = np.asarray(frame['teacher_endpoint_A'], float)
        direction = np.asarray(frame.get('teacher_contrast_direction_unit', []), float)
        confidence = np.asarray(frame.get('teacher_contrast_confidence', []), float)
        weights = np.asarray(frame.get('teacher_contrast_atom_weight', []), float)
        provenance = frame.get('teacher_contrast_provenance', [])
        if direction.shape != centers.shape or confidence.shape != centers.shape[:1] or weights.shape != centers.shape[:2] or len(provenance) != len(centers):
            raise ValueError('Bound conditional branch fields required')
        if not all(np.isfinite(v).all() for v in (centers, direction, confidence, weights)) or np.any((confidence < 0) | (confidence > 1)) or np.any((weights < .5) | (weights > 2.)):
            raise ValueError('Invalid branch direction/confidence/regional weights')
        eligibility = np.zeros(len(centers), bool)
        amplitude = np.zeros(len(centers), float)
        # Two predecessor selection events are needed for genuine new branches.
        if node_index is not None and node_index < 2:
            return centers, direction, confidence, weights, eligibility, amplitude
        for k, item in enumerate(provenance):
            raw = item.get('raw_direction_RMS_A', 0.)
            if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(raw) or raw < 0:
                raise ValueError('Finite natural-branch amplitude required')
            lower = item.get('lower_immediate_parent_slots', [])
            count = item.get('distinct_observed_mutations', 0)
            qualified = (item.get('lag2_observed') is True
                         and item.get('common_grandparent_slot') is not None
                         and item.get('immediate_parent_slot') is not None
                         and isinstance(lower, list) and bool(lower)
                         and isinstance(count, int) and not isinstance(count, bool) and count >= 2
                         and confidence[k] > 0 and raw > 0)
            if qualified:
                if item['immediate_parent_slot'] in lower or len(set(lower)) != len(lower):
                    raise ValueError('Distinct same-ancestor branch provenance required')
                norm = float(np.sqrt(np.mean(np.sum(direction[k] ** 2, axis=-1))))
                if not np.isclose(norm, 1., rtol=1e-5, atol=1e-8):
                    raise ValueError('Qualified joint direction must have RMS one')
                eligibility[k] = True
                amplitude[k] = min(float(raw), .2)
        return centers, direction, confidence, weights, eligibility, amplitude

    @staticmethod
    def _diagnostics(x, batch_size, *, shifts=None, eligibility=None, mass=0., weight_difference=None, prior_mass=None):
        zero = x.new_zeros(batch_size)
        shift = zero if shifts is None else x.new_tensor(shifts)
        eligible = zero if eligibility is None else x.new_tensor(eligibility)
        regional = zero if weight_difference is None else x.new_tensor(weight_difference)
        return {'contrast_teacher_shift_rms_A': shift,
                'branch_virtual_teacher_shift_rms_A': shift,
                'branch_virtual_mass_mean': eligible * mass,
                'branch_eligible_teacher_count': eligible,
                'branch_region_weight_rms_difference': regional,
                'branch_virtual_prior_mass': zero if prior_mass is None else x.new_tensor(prior_mass)}

    def _baseline(self, x, atoms, mask, time, anchor):
        value, detail = super().__call__(x, atoms, mask, time, anchor)
        return value, {**detail, **self._diagnostics(x, len(x))}

    def __call__(self, x, atoms, mask, time, anchor=None):
        # Exact no-op uses the unchanged base formula, not algebraic cancellation.
        if (self.mass == 0 or self.sign == 0) and self.region_mix == 0:
            return self._baseline(x, atoms, mask, time, anchor)
        j = int(np.abs(self.times - time).argmin())
        if abs(self.times[j] - time) > 2e-6 or anchor is None or not bool(mask.all()):
            raise ValueError('Exact endpoint node, frozen anchor and full slots required')
        frame = self.reference['frames'][j]
        centers, direction, confidence, weights, eligibility, amplitude = self._fields(frame, j)
        # Missing first-two ancestry and every other zero-support frame are exact.
        if not eligibility.any():
            return self._baseline(x, atoms, mask, time, anchor)
        scores = np.asarray(frame['teacher_scores'], float)
        targets, virtual_targets, priors, selected_weights = [], [], [], []
        selected_eligible, shift_rms = [], []
        for cloud in anchor.detach().cpu().numpy():
            costs, assignments = [], []
            for teacher in centers:
                distance = ((cloud[:, None] - teacher[None]) ** 2).sum(-1)
                row, col = linear_sum_assignment(distance)
                costs.append(float(distance[row, col].mean()))
                assignments.append(col)
            ids = np.argsort(np.asarray(costs), kind='stable')[:min(int(self.program.get('teacher_neighbors', 4)), len(costs))]
            shift = self.sign * amplitude[:, None, None] * direction
            targets.append(np.stack([centers[k, assignments[k]] for k in ids]))
            virtual_targets.append(np.stack([(centers[k] + shift[k])[assignments[k]] for k in ids]))
            logits = (-np.asarray(costs)[ids] / float(self.program.get('teacher_endpoint_temperature_A2', 4.))
                      + float(self.program.get('teacher_score_beta', 2.)) * (scores[ids] - scores.mean()))
            if 'teacher_base_log_weight' in frame:
                logits = logits + np.asarray(frame['teacher_base_log_weight'])[ids]
            priors.append(logits)
            effective_support = confidence * eligibility
            selected_weights.append(np.stack([1. + self.region_mix * effective_support[k] * (weights[k, assignments[k]] - 1.) for k in ids]))
            selected_eligible.append(eligibility[ids])
            shift_rms.append(float(np.sqrt((shift[ids] ** 2).sum(-1).mean())) if self.mass > 0 else 0.)
        target = x.new_tensor(np.asarray(targets))
        virtual = x.new_tensor(np.asarray(virtual_targets))
        log_prior = x.new_tensor(np.asarray(priors)).log_softmax(1).detach()
        e = x.new_tensor(np.asarray(selected_eligible))
        w = x.new_tensor(np.asarray(selected_weights))
        q0 = ((x[:, None] - target).square().sum(-1) * w).mean(2)
        delta = float(self.program.get('pointcloud_delta_A', 1.))
        cost0 = delta ** 2 * (torch.sqrt(1 + q0 / delta ** 2) - 1)
        virtual_mass = self.mass * e
        if self.mass > 0 and self.sign != 0:
            q1 = ((x[:, None] - virtual).square().sum(-1) * w).mean(2)
            cost1 = delta ** 2 * (torch.sqrt(1 + q1 / delta ** 2) - 1)
            # Mixture components keep original teacher mass at least 1-alpha.
            original_logits = log_prior + torch.log1p(-virtual_mass) - cost0 / self.tau
            virtual_logits = log_prior + virtual_mass.log() - cost1 / self.tau
            value = self.tau * torch.logsumexp(torch.cat((original_logits, virtual_logits), 1), 1)
            nearest = torch.minimum(q0.amin(1), torch.where(e.bool(), q1, torch.inf).amin(1)).sqrt()
        else:
            value = self.tau * torch.logsumexp(log_prior - cost0 / self.tau, 1)
            nearest = q0.amin(1).sqrt()
            virtual_mass = torch.zeros_like(e)
        active_rows = e.bool().any(1)
        if not bool(active_rows.all()):
            baseline, _ = super().__call__(x, atoms, mask, time, anchor)
            value = torch.where(active_rows, value, baseline)
        phase = (time - self.window[0]) / (self.window[1] - self.window[0])
        schedule = (.1 + .9 * np.clip(phase, 0, 1)) ** float(self.program.get('time_ramp_power', 0.))
        detail = {'available': mask.any(1), 'core_mask': mask.bool(),
                  'dose_gate': x.new_full((len(x),), schedule),
                  'time_dose_factor': x.new_full((len(x),), schedule),
                  'nearest_standardized_rms': nearest}
        count = e.sum(1).detach().cpu().numpy()
        mean_mass = virtual_mass.mean(1)
        detail.update(self._diagnostics(x, len(x), shifts=shift_rms,
                                       eligibility=count, weight_difference=(w - 1).square().mean((1, 2)).sqrt().detach().cpu().numpy(),
                                       prior_mass=(log_prior.exp() * virtual_mass).sum(1).detach().cpu().numpy()))
        detail['branch_virtual_mass_mean'] = mean_mass.detach()
        return value, detail
