"""Registered pure endpoint geometry objectives, separate from actual-state force."""
import copy
from .affinity_geometry_reward import AffinityGeometryReward

FAMILIES={'endpoint_landmark':'affinity_landmark','endpoint_direction':'affinity_direction','endpoint_pointcloud':'affinity_pointcloud'}

class EndpointGeometryReward(AffinityGeometryReward):
    def __init__(self,program,reference):
        if program.get('derivative_path')!='flowr_endpoint_vjp' or reference['schema_version']!='affinity-endpoint-library-1.0':
            raise ValueError('True FLOWR endpoint derivative contract required')
        if program['reward_view'] not in FAMILIES:raise ValueError('Unregistered endpoint formula')
        p=copy.deepcopy(program);r=copy.deepcopy(reference)
        p['reward_view']=FAMILIES[p['reward_view']];r['schema_version']='affinity-coordinate-library-1.0'
        if p['reward_view']=='affinity_pointcloud':
            for frame in r['frames']:
                if not frame.get('teacher_endpoint_A'):raise ValueError('Original elite endpoint teachers required')
                # A single stored point cloud plays both roles for endpoint loss.
                frame['teacher_proposal_A']=frame['teacher_endpoint_A']
        super().__init__(p,r)
