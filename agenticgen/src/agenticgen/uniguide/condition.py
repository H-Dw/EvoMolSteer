"""UniGuide condition map; geometry-only matching is a stopped-gradient operation."""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def finite(name, value):
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"Nonfinite {name}; no clipping or weaker retry")


def condition_map(endpoint_A, core_A, valid):
    finite("endpoint", endpoint_A)
    core = torch.as_tensor(core_A, dtype=endpoint_A.dtype, device=endpoint_A.device).detach()
    finite("condition", core)
    if core.ndim != 2 or core.shape[1] != 3 or not len(core):
        raise ValueError("A nonempty PLIP heavy-atom coordinate core is required")
    target = endpoint_A.detach().clone()
    selected = torch.zeros_like(valid, dtype=torch.bool)
    assignments = []
    with torch.no_grad():
        for b in range(len(endpoint_A)):
            ids = torch.where(valid[b].bool())[0]
            if len(ids) < len(core):
                raise ValueError("Too few valid generated slots for injective conditioning")
            cost = (core[:, None] - endpoint_A[b, ids][None]).square().sum(-1)
            rows, cols = linear_sum_assignment(cost.detach().cpu().double().numpy())
            slots = ids[torch.as_tensor(cols[np.argsort(rows)], device=ids.device)]
            target[b, slots] = core
            selected[b, slots] = True
            assignments.append(slots)
    return target.detach(), selected.detach(), torch.stack(assignments).detach()


def coordinate_loss(endpoint_A, target_A, selected):
    """Dimensionless squared error with fixed 1 Angstrom unit, 1/(2 K) reduction."""
    count = selected.sum(-1)
    if bool((count == 0).any()):
        raise ValueError("Empty condition mask")
    return ((endpoint_A - target_A.detach()).square().sum(-1) * selected.detach()).sum(-1) / (2 * count)


def loss(endpoint_A, core_A, valid):
    target, selected, assignment = condition_map(endpoint_A, core_A, valid)
    return coordinate_loss(endpoint_A, target, selected), target, selected, assignment


def guidance_delta(gradient_native, time, dt, scale, gamma, valid):
    """Exact drift difference for q' = q-S*g under the audited FLOWR score proxy.

    q=(t*v-x)/(1-t+eps), b=v+gamma*q. Thus dv=-(1-t+eps)/t*S*g.
    World/native conversion is already in the input Jacobian; no extra a^-2.
    """
    scales = torch.as_tensor(scale, dtype=gradient_native.dtype, device=gradient_native.device).reshape(-1)
    active = scales != 0
    coefficient = torch.zeros_like(scales)
    if bool(active.any()):
        t = time[active]
        if bool(((t <= 0) | (t >= 1)).any()):
            raise ValueError("Guidance is undefined at the flow endpoints")
        coefficient[active] = -dt * scales[active] * ((1 - t + 1e-6) / t + gamma)
    delta = coefficient[:, None, None] * gradient_native * valid[..., None]
    finite("guidance increment", delta)
    return delta, coefficient

