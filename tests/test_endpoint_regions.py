import numpy as np
from evomolsteer.continuous.endpoint_regions import regional_moments

def test_local_moments_permutation_invariance_and_receptor_frame():
    rng=np.random.default_rng(42);x=rng.normal(size=(4,9,3));p=rng.normal(size=(3,3))
    a=regional_moments(x,p).reshape(4,3,10)
    np.testing.assert_allclose(a,regional_moments(x[:,::-1],p).reshape(4,3,10),atol=1e-14)
    np.testing.assert_allclose(a,regional_moments(x+2,p+2).reshape(4,3,10),atol=1e-14)
    assert ((a[:,:,0]>0)&(a[:,:,0]<=1)).all()
    assert (a[:,:,4:7]>=0).all()
    assert not np.allclose(a[:,:,1:4],regional_moments(x+2,p).reshape(4,3,10)[:,:,1:4])
