"""Bounded static/observed-node teacher mixture, using one FLOWR endpoint.

Only analytic coordinate rewards are evaluated twice. FLOWR and its affinity
head are not called here. Matching, neighborhood priors and mixture weight are
detached; the registered static endpoint VJP remains the derivative path.
"""
import copy
import math

import numpy as np

from .endpoint_reward import EndpointGeometryReward
from .regional_point_reward import RegionalPointReward
from .coordinate_reward import CoordinateMixtureReward


class TemporalTeacherReward(CoordinateMixtureReward):
    def __init__(self, program, reference):
        if program.get("reward_view") != "endpoint_temporal_pointcloud":
            raise ValueError("Registered temporal point-cloud view required")
        self.blend = float(program["temporal_teacher_blend"])
        if not math.isfinite(self.blend) or not 0 <= self.blend <= 1:
            raise ValueError("Bounded finite temporal blend required")
        self.program,self.reference=program,reference
        self.window=reference["window"]
        self.times=np.asarray(reference["times"],float)
        p = copy.deepcopy(program)
        p["reward_view"] = program.get("temporal_base_view", "endpoint_pointcloud")
        if p["reward_view"] not in {"endpoint_pointcloud", "endpoint_regional_pointcloud"}:
            raise ValueError("Unsupported temporal base view")
        reward_class = RegionalPointReward if p["reward_view"] == "endpoint_regional_pointcloud" else EndpointGeometryReward
        self.static = reward_class(p, reference)
        dynamic = copy.deepcopy(reference)
        for frame in dynamic["frames"]:
            cloud = np.asarray(frame["trajectory_teacher_endpoint_A"], float)
            if cloud.ndim != 3 or cloud.shape[-1] != 3 or not len(cloud) or not np.isfinite(cloud).all():
                raise ValueError("Complete actual-node teacher coordinates required")
            frame["teacher_endpoint_A"] = frame["trajectory_teacher_endpoint_A"]
            frame["teacher_scores"] = [0.] * len(cloud)
            frame["teacher_batches"] = frame["trajectory_teacher_batches"]
        self.dynamic = reward_class(p, dynamic)

    def __call__(self, x, atoms, mask, time, anchor=None):
        value, diagnostics = self.static(x, atoms, mask, time, anchor)
        if self.blend == 0:
            return value, diagnostics
        other, d = self.dynamic(x, atoms, mask, time, anchor)
        diagnostics = dict(diagnostics,
            temporal_teacher_blend=x.new_full((len(x),), self.blend),
            temporal_teacher_nearest_rms_A=d["nearest_standardized_rms"])
        return (1-self.blend)*value + self.blend*other, diagnostics
