"""Supported, bounded selection/background contrast from empirical joint moments.

The candidate background already contains previous SMC selection. This is an
incremental preference hypothesis, not a native density, density-ratio truth or
causal affinity reward. No extra neural model is trained.
"""
import math
import numpy as np
import torch
from .coordinate_reward import CoordinateMixtureReward


def gaussian_log_components(z,centers,covariances):
    """Normalized equal-weight Gaussian components, including log determinants."""
    center=z.new_tensor(centers);covariance=z.new_tensor(covariances)
    chol=torch.linalg.cholesky(covariance)
    delta=z[:,None]-center[None]
    white=torch.linalg.solve_triangular(chol[None],delta[...,None],upper=False).squeeze(-1)
    squared=white.square().sum(-1)
    logdet=2*torch.diagonal(chol,dim1=-2,dim2=-1).log().sum(-1)
    log_component=-.5*(squared+logdet+z.shape[1]*math.log(2*math.pi))-math.log(len(center))
    return log_component,squared/z.shape[1]


def support_gate(q,lower,upper,posterior):
    """Posterior-weighted empirical support; degenerate quantiles yield zero dose."""
    lo=q.new_tensor(lower);hi=q.new_tensor(upper)
    usable=(hi-lo)>1e-12
    u=((q-lo)/(hi-lo).clamp_min(1e-12)).clamp(0,1)
    # Smoothstep has zero derivative at either edge. It is amplitude only.
    h=torch.where(usable,1-u.square()*(3-2*u),torch.zeros_like(u))
    return (h*posterior).sum(1)


class CoordinateSelectionContrastReward(CoordinateMixtureReward):
    def __init__(self,program,reference):
        super().__init__(program,reference)
        self.bound=float(program['contrast_bound_nats'])
        if not math.isfinite(self.bound) or self.bound<=0:raise ValueError('Positive contrast bound required')
        for frame in reference['frames']:
            for mode in frame['modes']:
                for key in ('background_center_A','background_covariance_A2','paired_gaussian_KL_nats','background_support_q90','background_support_q98'):
                    if key not in mode:raise ValueError('Contrast requires paired empirical background/support')

    def __call__(self,x,atoms,mask,time,anchor=None):
        j=int(np.abs(self.times-time).argmin())
        if abs(self.times[j]-time)>2e-6:raise ValueError('No exact learned coordinate time')
        # Double precision limits cancellation when two densities almost agree;
        # autograd returns a gradient in the incoming coordinate dtype.
        z,valid,core=self.observables(x.double(),atoms,mask,None if anchor is None else anchor.double())
        modes=self.reference['frames'][j]['modes']
        selected,qs=gaussian_log_components(z,[m['center_A'] for m in modes],[m['covariance_A2'] for m in modes])
        background,qb=gaussian_log_components(z,[m['background_center_A'] for m in modes],[m['background_covariance_A2'] for m in modes])
        logratio=torch.logsumexp(selected,1)-torch.logsumexp(background,1)
        bounded=torch.tanh(logratio/self.bound)
        reward=self.bound*bounded
        h=support_gate(qb,[m['background_support_q90'] for m in modes],[m['background_support_q98'] for m in modes],background.softmax(1))
        k=float(np.mean([m['paired_gaussian_KL_nats'] for m in modes]))/z.shape[1]
        amplitude=math.sqrt(2*k/(1+2*k))
        # Preserve support, weak empirical contrast and tanh saturation after
        # gradient unit-normalization. None of these gates changes the direction.
        gate=(h*amplitude*(1-bounded.square())).detach()
        return torch.where(valid,reward,z.sum(1)*0),{
            'observables_A':z,'available':valid,'core_mask':core,
            'nearest_standardized_rms':qs.amin(1).sqrt(),
            'dose_gate':torch.where(valid,gate,0.),'background_support_gate':h.detach(),
            'contrast_amplitude_gate':z.new_full((len(z),),amplitude),
            'bounded_response_gate':(1-bounded.square()).detach(),'log_density_ratio':logratio.detach()}


def make_coordinate_reward(program,reference):
    if program['reward_view']=='endpoint_static_hybrid':
        from .static_hybrid_reward import StaticHybridReward
        return StaticHybridReward(program,reference)
    if program['reward_view']=='endpoint_structure_field':
        from .structure_field_reward import StructureFieldReward
        return StructureFieldReward(program,reference)
    if program['reward_view']=='endpoint_selection_path':
        from .selection_path_reward import SelectionPathReward
        return SelectionPathReward(program,reference)
    if program['reward_view']=='endpoint_dynamic_region':
        from .dynamic_region_reward import DynamicRegionReward
        return DynamicRegionReward(program,reference)
    if program['reward_view']=='endpoint_path_value':
        from .path_reward import PathValueReward
        return PathValueReward(program,reference)
    if program['reward_view']=='endpoint_regional_pointcloud':
        from .regional_point_reward import RegionalPointReward
        return RegionalPointReward(program,reference)
    if program['reward_view']=='endpoint_supported_attractor':
        from .persistent_reward import PersistentEndpointReward
        return PersistentEndpointReward(program,reference)
    if program['reward_view'] in ('endpoint_landmark','endpoint_direction','endpoint_pointcloud'):
        if 'geometry_feature_weights' in program:
            from .sparse_reward import SparseEndpointReward
            return SparseEndpointReward(program,reference)
        from .endpoint_reward import EndpointGeometryReward
        return EndpointGeometryReward(program,reference)
    if program['reward_view'] in ('affinity_landmark','affinity_direction','affinity_pointcloud'):
        from .affinity_geometry_reward import AffinityGeometryReward
        return AffinityGeometryReward(program,reference)
    if program['reward_view'] in ('motif_mixture','motif_contrast'):
        from .motif_reward import MotifReward
        return MotifReward(program,reference)
    if program['reward_view']=='count_conditioned_shape':
        from .coordinate_conditioning import CoordinateCountConditionedShapeReward
        return CoordinateCountConditionedShapeReward(program,reference)
    if program['reward_view']=='shape_mixture':
        from .coordinate_shape import CoordinateShapeReward
        return CoordinateShapeReward(program,reference)
    if program['reward_view']=='selection_contrast':return CoordinateSelectionContrastReward(program,reference)
    if program['reward_view'] not in ('coordinate_mixture','spread_upper'):raise ValueError('Unknown coordinate reward')
    return CoordinateMixtureReward(program,reference)
