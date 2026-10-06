import numpy as np
from evomolsteer.continuous.coordinate_features import regional_observables


def measure(x,atoms=None):
    anchor=np.zeros_like(x);m=np.ones(x.shape[:2],bool)
    if atoms is None:atoms=np.full(x.shape[:2],4)
    cat={'regions':{'r':{'points_A':[[0,0,0]]}},'atom_vocabulary':{'N':4,'O':5,'S':9}}
    return regional_observables(x,anchor,x,atoms,m,cat,.2,.01,4.,'endpoint',True,'shape')


def test_second_moment_distinguishes_equal_radius_shapes_and_is_permutation_invariant():
    x=np.array([[[-1.,0,0],[1.,0,0]]]);rot=x[..., [1,0,2]]
    a,names,meta=measure(x);b,_,_=measure(rot)
    assert not np.array_equal(a,b)
    trace=lambda z:sum(z[0,names.index('r::all::shape_'+p)] for p in ['xx','yy','zz'])
    assert trace(a)==trace(b)==1.
    c,_,_=measure(x[:,::-1]);np.testing.assert_array_equal(a,c)
    assert meta['r::all::proposal_shape_xy']['unit']=='A^2'


def test_single_nos_slot_has_legal_zero_shape_and_frobenius_scaling_is_exact():
    x=np.array([[[-1.,-2,0],[1.,2,0]]])
    a,names,_=measure(x,np.array([[4,3]]))
    assert a[0,names.index('r::NOS::shape_xx')]==0
    assert a[0,names.index('r::NOS::shape_informative')]==0
    assert a[0,names.index('r::NOS::shape_eligible_slots')]==1
    assert np.isfinite(a[:,[i for i,n in enumerate(names) if '::all::' in n]]).all()
    assert a[0,names.index('r::all::shape_xy')]==2*np.sqrt(2)
    vector=a[0,[names.index('r::all::shape_'+p) for p in ['xx','yy','zz','xy','xz','yz']]];matrix=np.array([[1.,2,0],[2.,4,0],[0,0,0]])
    np.testing.assert_allclose(vector@vector,np.sum(matrix**2))
    absent,_,_=measure(x,np.array([[3,3]]))
    assert np.isnan(absent[0,names.index('r::NOS::shape_xx')])
    assert absent[0,names.index('r::NOS::shape_eligible_slots')]==0
