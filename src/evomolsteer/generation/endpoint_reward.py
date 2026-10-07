"""Registered pure endpoint geometry objectives, separate from actual-state force."""
import copy
from .affinity_geometry_reward import AffinityGeometryReward

FAMILIES={'endpoint_landmark':'affinity_landmark','endpoint_direction':'affinity_direction','endpoint_pointcloud':'affinity_pointcloud'}

class EndpointGeometryReward(AffinityGeometryReward):
    def __init__(self,program,reference):
        if program.get('derivative_path')!='flowr_endpoint_vjp' or reference['schema_version']!='affinity-endpoint-library-1.0':
            raise ValueError('True FLOWR endpoint derivative contract required')
        if program['reward_view'] not in FAMILIES:raise ValueError('Unregistered endpoint formula')
        allowed=reference.get('allowed_reward_views')
        if allowed is not None and program['reward_view'] not in allowed:
            raise ValueError('Reference labels do not support this endpoint formula')
        if reference.get('reference_variant')=='terminal-descendant-endpoint-library-1.0':
            import numpy as np
            for frame in reference['frames']:
                weights=np.asarray(frame.get('teacher_base_log_weight',[]),float)
                if weights.shape!=(len(frame['teacher_endpoint_A']),) or not np.isfinite(weights).all():
                    raise ValueError('Explicit finite terminal batch/ancestor prior required')
                batches=np.asarray(frame.get('teacher_batches',[]))
                if batches.shape!=weights.shape:raise ValueError('Terminal teacher batch indices required')
                for batch in np.unique(batches):
                    subset=batches==batch
                    if not np.allclose(weights[subset],-np.log(subset.sum()),rtol=0,atol=1e-12):
                        raise ValueError('Terminal base prior must give equal batches then ancestors')
        p=copy.deepcopy(program);r=copy.deepcopy(reference)
        p['reward_view']=FAMILIES[p['reward_view']];r['schema_version']='affinity-coordinate-library-1.0'
        if p['reward_view']=='affinity_pointcloud':
            for frame in r['frames']:
                if not frame.get('teacher_endpoint_A'):raise ValueError('Original elite endpoint teachers required')
                # A single stored point cloud plays both roles for endpoint loss.
                frame['teacher_proposal_A']=frame['teacher_endpoint_A']
        super().__init__(p,r)
