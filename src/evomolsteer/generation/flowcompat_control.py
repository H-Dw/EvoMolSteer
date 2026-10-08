"""Geometry of control relative to a native linear flow, separate from reward.

Preconditioners are experimental controls, not gradients of a new scalar or
exact conditional samplers. A null configuration preserves original bytes.
"""
import math
import torch


def cosine(a,b):
    return (a*b).sum((1,2))/(a.norm(dim=(1,2))*b.norm(dim=(1,2))).clamp_min(1e-30)


def control_geometry(g, flow, mask, time, window, spec, endpoint_gradient_rms=None):
    limits={'time_envelope_power':(0,3),'parallel_component_scale':(0,2),
            'gradient_norm_saturation':(0,10),'jacobian_gain_saturation':(0,10)}
    if set(spec)-set(limits):raise ValueError('Unregistered flow control')
    for key,value in spec.items():
        if not math.isfinite(value) or not limits[key][0]<=value<=limits[key][1]:raise ValueError('Unbounded flow control')
    adjusted=g
    parallel=spec.get('parallel_component_scale',1.)
    if parallel!=1:
        projection=(g*flow).sum((1,2))/flow.square().sum((1,2)).clamp_min(1e-30)
        adjusted=g+(parallel-1)*projection[:,None,None]*flow
    phase=min(1.,max(0.,(time-window[0])/(window[1]-window[0])))
    factor=g.new_full((len(g),),(1-phase)**spec.get('time_envelope_power',0.))
    grms=(g.square().sum((1,2))/mask.sum(1).clamp_min(1)).sqrt()
    saturation=spec.get('gradient_norm_saturation',0.)
    if saturation:factor=factor*grms/(grms+saturation)
    gain=grms/(endpoint_gradient_rms.clamp_min(1e-30)) if endpoint_gradient_rms is not None else torch.zeros_like(grms)
    saturation=spec.get('jacobian_gain_saturation',0.)
    if saturation:
        if endpoint_gradient_rms is None:raise ValueError('Real endpoint gradient required')
        factor=factor*gain/(gain+saturation)
    return adjusted,factor,{'flowcompat_native_cosine_before':cosine(g,flow),
        'flowcompat_native_cosine_after':cosine(adjusted,flow),
        'flowcompat_schedule_factor':factor,
        'flowcompat_gradient_adjustment_relative_rms':(adjusted-g).norm(dim=(1,2))/g.norm(dim=(1,2)).clamp_min(1e-30),
        'flowcompat_jacobian_gain':gain}
