"""Time-indexed scalar geometry rewards evaluated on ONE native FLOWR endpoint.

Matching/teacher selection are detached. Only the combined scalar enters the
existing endpoint VJP. No differentiation through affinity or trajectory time.
"""
import copy

import numpy as np

from .coordinate_reward import CoordinateMixtureReward
from .endpoint_reward import EndpointGeometryReward
from ..temporal_guidance.policy import validate_policy


class ScheduledEndpointReward(CoordinateMixtureReward):
    def __init__(self, program, reference):
        self.program, self.reference = copy.deepcopy(program), reference
        self.window, self.times = reference["window"], np.asarray(reference["times"], float)
        self.policy = validate_policy(program["temporal_policy"], self.window, self.times)
        p = copy.deepcopy(program)
        if p["reward_view"] != "endpoint_pointcloud" or p.get("time_ramp_power", 0) != 0:
            raise ValueError("Temporal experiments start from the un-ramped M2 point-cloud family")
        self.static = EndpointGeometryReward(p, reference)
        self.dynamic = None
        if any(v > 0 for v in self.policy["teacher_blend"]):
            dynamic = copy.deepcopy(reference)
            for i, frame in enumerate(dynamic["frames"]):
                cloud = np.asarray(frame["trajectory_teacher_endpoint_A"], float)
                batches = np.asarray(frame["trajectory_teacher_batches"])
                if cloud.ndim != 3 or cloud.shape[-1] != 3 or not len(cloud) or not np.isfinite(cloud).all():
                    raise ValueError("Observed-clock historical forecast teachers required")
                if batches.shape != (len(cloud),) or abs(float(frame["time"]) - self.times[i]) > 2e-6:
                    raise ValueError("Teacher provenance/clock mismatch")
                static = np.asarray(frame["teacher_endpoint_A"])
                if cloud.shape[1:] != static.shape[1:]:
                    raise ValueError("Comparable slot count and coordinate frame required")
                frame["teacher_endpoint_A"] = frame["trajectory_teacher_endpoint_A"]
                frame["teacher_scores"] = [0.] * len(cloud)
                frame["teacher_batches"] = frame["trajectory_teacher_batches"]
                frame.pop("teacher_base_log_weight", None)
            self.dynamic = EndpointGeometryReward(p, dynamic)

    def __call__(self, x, atoms, mask, time, anchor=None):
        j = int(np.abs(self.times - time).argmin())
        if abs(self.times[j] - time) > 2e-6:
            raise ValueError("No exact observed teacher/schedule node; extrapolation forbidden")
        blend, dose, delta = [float(self.policy[k][j]) for k in ("teacher_blend", "dose_factors", "delta_A")]
        # Private analytic reward state only; no FLOWR parameter is modified.
        self.static.program["pointcloud_delta_A"] = delta
        if blend == 1:
            self.dynamic.program["pointcloud_delta_A"] = delta
            value, detail = self.dynamic(x, atoms, mask, time, anchor)
        else:
            value, detail = self.static(x, atoms, mask, time, anchor)
            if blend > 0:
                self.dynamic.program["pointcloud_delta_A"] = delta
                other, dynamic_detail = self.dynamic(x, atoms, mask, time, anchor)
                value = (1 - blend) * value + blend * other
                detail = dict(detail, dynamic_teacher_nearest_rms_A=dynamic_detail["nearest_standardized_rms"])
        detail = dict(detail)
        # Scaling R alone is cancelled by the controller's RMS normalization.
        # Preserve exact identity path (no multiplication by 1).
        if dose != 1:
            detail["dose_gate"] = detail["dose_gate"] * dose
        detail.update(temporal_dose_factor=x.new_full((len(x),), dose),
                      temporal_teacher_blend=x.new_full((len(x),), blend),
                      temporal_delta_A=x.new_full((len(x),), delta),
                      time_dose_factor=x.new_full((len(x),), dose))
        return value, detail
