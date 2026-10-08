import numpy as np
from evomolsteer.continuous.selection_innovation import signed_contrast,match_cloud,parent_innovations,bounded_atom_weights,direction_components

def toy():
    anchor=np.array([[0.,0.,0.],[4.,0.,0.],[0.,5.,0.]])
    clouds=np.stack([anchor-[.2,0,0],anchor-[.4,0,0],anchor-[.3,0,0],anchor-[.5,0,0]])
    return anchor,clouds,np.array([2.,1.,1.5,.5]),np.arange(4)

def test_exact_null_ties_and_identical_clouds():
    a,c,s,r=toy()
    tied=signed_contrast(a,1.,c,np.ones(4),r)
    identical=signed_contrast(a,3.,np.repeat(a[None],4,0),s,r)
    for result in (tied,identical):
        assert np.array_equal(result['direction_unit'],np.zeros_like(a))
        assert result['confidence']==0.

def test_signed_direction_permutation_and_joint_rotation():
    a,c,s,r=toy();out=signed_contrast(a,3.,c,s,r)
    assert np.all(out['direction_unit'][:,0]>0)
    assert np.isclose(np.sqrt((out['direction_unit']**2).sum(-1).mean()),1.)
    assert 0<out['confidence']<.2
    permutation=np.array([2,0,1]); permuted=signed_contrast(a[permutation],c.shape[0]-1,c[:,permutation],s,r)
    assert np.allclose(permuted['direction_unit'],out['direction_unit'][permutation])
    rotation=np.array([[0.,1.,0.],[-1.,0.,0.],[0.,0.,1.]])
    transformed=signed_contrast(a@rotation+7,3.,c@rotation+7,s,r)
    assert np.allclose(transformed['direction_unit'],out['direction_unit']@rotation)
    assert np.isclose(transformed['confidence'],out['confidence'])

def test_single_family_cannot_manufacture_confidence():
    a,c,s,r=toy();out=signed_contrast(a,3.,c,s,np.zeros(4,int))
    assert out['root_weight_ess']==1
    assert out['confidence']==0
    assert np.any(out['direction_unit'])

def test_parent_averaging_and_forecast_innovation_are_not_displacement():
    previous=np.zeros((2,3,3));parents=np.array([0,0,1]);current=previous[parents].copy()
    current[:,:,0]=np.array([1.,3.,2.])[:,None]
    innovation,summary=parent_innovations(current,previous,parents,[2.,4.,3.],[1.,2.],np.ones_like(previous))
    assert np.array_equal(innovation,current)
    assert summary['observed_parent_count']==2
    assert summary['branching_parent_count']==1
    assert np.isclose(summary['observed_child_score_gain'],1.5)
    assert np.isclose(summary['innovation_RMS_A'],2.)
    assert summary['within_parent_innovation_gain_correlation']>0.99

def test_match_preserves_pocket_pose_instead_of_fitting_rigid_transform():
    a,c,_,_=toy();aligned,order=match_cloud(a,c[0,[2,0,1]])
    assert np.allclose(aligned,c[0])
    assert not np.array_equal(aligned,a)

def test_clone_copies_do_not_inflate_support_or_field():
    a,c,s,r=toy();base=signed_contrast(a,3.,c,s,r,10)
    copies=np.array([0,0,0,1,2,3]);duplicate=signed_contrast(a,3.,c[copies],s[copies],r[copies],10)
    assert np.allclose(base['direction_unit'],duplicate['direction_unit'])
    assert np.isclose(base['root_weight_ess'],duplicate['root_weight_ess'])
    assert np.isclose(base['confidence'],duplicate['confidence'])

def test_atom_weights_and_rigid_internal_decomposition():
    weights=bounded_atom_weights([0.,0.,0.,1.,100.])
    assert weights.min()>=.5 and weights.max()<=2. and np.isclose(weights.mean(),1.)
    assert np.array_equal(bounded_atom_weights([0.,0.]),[1.,1.])
    a,_,_,_=toy();v=np.broadcast_to([1.,0.,0.],a.shape)
    components=direction_components(a,v)
    assert np.isclose(components['translation_fraction'],1.)
    assert components['rotation_fraction']<1e-24 and components['internal_fraction']<1e-24
