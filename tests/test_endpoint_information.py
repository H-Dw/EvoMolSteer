import numpy as np
from evomolsteer.continuous.endpoint_information import proper_rigid_residual,full_distances


def test_pose_decomposition_uses_proper_rotation_and_geometric_permutation():
    cloud=np.array([[0.,0,0],[1,0,0],[0,2,0],[0,0,3]])
    rotated=cloud@np.array([[0.,1,0],[-1,0,0],[0,0,1]])+np.array([4.,-2,1])
    assert proper_rigid_residual(cloud,rotated)<1e-12
    reflected=cloud*np.array([-1.,1,1])
    assert proper_rigid_residual(cloud,reflected)>.1
    teachers=np.stack([cloud[::-1],cloud+2])
    distance,orders=full_distances(cloud,teachers)
    assert distance[0]==0 and distance[1]>0
    np.testing.assert_array_equal(teachers[0,orders[0]],cloud)
