"""Legacy M2 reward/control mathematics on the common native FLOWR loop."""
import torch

from evomolsteer.final_guidance.native import Control
from evomolsteer.generation.coordinate_contrast import make_coordinate_reward
from evomolsteer.generation.flowcompat_controller import measured_pullback
from evomolsteer.generation.coordinate_reward import predictive_flow_increment, calibration_increment, atom_step_cap
from evomolsteer.generation.local_reward import bounded_local_step
from evomolsteer.generation.multistage_reward import reject_new_severe_clashes
from evomolsteer.generation.scalar_guidance import detached


def frozen_reward(program, reference):
    # The old temporal controller selects this wrapper even though reward_view
    # remains endpoint_pointcloud. Ignoring it silently erases S/W/C policies.
    if 'temporal_policy' in program:
        from evomolsteer.generation.scheduled_endpoint_reward import ScheduledEndpointReward
        return ScheduledEndpointReward(program, reference)
    return make_coordinate_reward(program, reference)


def legacy_injection(program, gradient, before, native, endpoint, mask, pocket, time, dt, scale, remaining, detail, first):
    """Exactly the frozen EndpointCoordinateExtension.after_native dose and guard."""
    flow = predictive_flow_increment(before, endpoint, time, dt, False)
    calibration = calibration_increment(native['coords'] - before, flow, mask,
                                        program.get('dose_reference', 'predictive_flow'),
                                        program.get('initial_update_dose', 'native'), first)
    c = program['constraints']
    proposed, control = bounded_local_step(gradient, calibration, mask, detail['dose_gate'].to(gradient),
        program['native_rms_ratio'], scale, atom_step_cap(c, first), remaining)
    actual, guard = reject_new_severe_clashes(native['coords'], proposed, mask, pocket['coords'],
                                             pocket['mask'], scale, c)
    return actual, control, guard


class LegacyControl(Control):
    def __init__(self, model, program, teacher, com, capture_time=.5):
        super().__init__(model, None, None, com, capture_time)
        self.program = program
        self.reward = frozen_reward(program, teacher)
        self.legacy_cache = None
        self.controlled_updates = 0

    def forward(self, curr, pocket, times, cond, equis, invs, i, dt):
        self.cached = None; self.legacy_cache = None
        self.t = float(times[0][0]); self.dt = float(dt)
        self.before = curr['coords'].detach().clone(); self.pocket = pocket
        kw = dict(cond_batch=detached(cond), pocket_equis=equis, pocket_invs=invs, training=False)
        if not self.reward.active(self.t, self.t + self.dt):
            return self.model(curr, pocket, times, **kw)
        captured = []
        def predict(x):
            out = self.model(dict(curr, coords=x), pocket, times, **kw)
            if 'affinity' in out:
                def denied(g):
                    raise RuntimeError('Affinity head entered alignment reward gradient')
                for v in out['affinity'].values():
                    if torch.is_tensor(v) and v.requires_grad:
                        v.register_hook(denied)
            captured.append(out)
            return self.model._get_predictions(out)
        pred, _, gradient, value, detail, _, _ = measured_pullback(predict, self.reward, self.before,
            curr['mask'].bool(), self.model.coord_scale, self.com, self.t)
        self.legacy_cache = (gradient, value, detail, pred['coords'])
        return detached(captured[0])

    def after(self, curr, i):
        active = self.legacy_cache is not None
        telemetry = {}
        if active:
            g, value, detail, endpoint = self.legacy_cache
            mask = curr['mask'].bool(); scale = self.model.coord_scale
            actual, control, guard = legacy_injection(self.program, g, self.before, curr, endpoint, mask,
                self.pocket, self.t, self.dt, scale, self.program['constraints']['max_cumulative_rms_A'] - self.total,
                detail, self.controlled_updates == 0)
            rms = (actual.square().sum((1, 2)) / mask.sum(1)).sqrt() * scale
            self.total += rms; self.controlled_updates += 1
            curr = dict(curr, coords=curr['coords'] + actual)
            telemetry = {'active': True, 'nonzero_molecules': int((rms > 1e-9).sum()),
                'injected_rms_A_mean': float(rms.mean()), 'requested_rms_A_mean': float(control['requested_rms_A'].mean()),
                'cap_factor': float(control['cap_factor'].mean()), 'reward_mean': float(value.mean()),
                'gradient_rms': float((g.square().sum((1, 2))/mask.sum(1)).sqrt().mean()),
                'production_forward_calls': 1, 'controller': 'frozen_legacy_global_atom_cap_and_new_receptor_clash_backtracking'}
        result = super().after(curr, i)
        self.rows[-1].update(telemetry)
        return result
