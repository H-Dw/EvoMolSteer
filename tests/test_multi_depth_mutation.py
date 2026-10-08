import numpy as np
import pytest
from evomolsteer.continuous.multi_depth_mutation import parse_depths,ancestor_indices,signed_regions,conditional_statistics,cross_batch_directions

def test_dynamic_depth_path_composition_and_missing_nodes():
    s=np.array([[2,2,1,0],[1,3,3,0],[0,1,1,3],[3,2,0,0]])
    assert np.array_equal(ancestor_indices(s,3,3),s[0,s[1,s[2]]])
    assert parse_depths('13,2,5')==[2,5,13]
    with pytest.raises(ValueError):ancestor_indices(s,1,2)
    with pytest.raises(ValueError):parse_depths('2,2')

def test_regional_internal_motion_removes_translation_and_respects_atom_permutation():
    a=np.array([[[0.,0.,0.],[2.,0.,0.],[0.,2.,0.]]]);p=np.array([[0.,0.,0.],[10.,0.,0.]])
    translated=a+[1.,2.,3.];r,t,m=signed_regions(translated,a,p)
    assert np.array_equal(r,np.zeros_like(r)) and np.array_equal(t,[[1.,2.,3.]])
    y=a.copy();y[0,0,0]+=.5;region,_,_=signed_regions(y,a,p)
    order=[2,0,1];again,_,_=signed_regions(y[:,order],a[:,order],p)
    assert np.allclose(region,again)

def test_copy_inflation_and_exact_equal_score_null():
    x=np.array([[0.,2.],[1.,3.],[2.,1.],[4.,2.]])
    y=np.array([1.,2.,0.,3.]);p=np.arange(4);g=np.array([0,0,1,1]);z=np.zeros((4,5))
    expected,diag=conditional_statistics(x,y,z,p,g)
    copies=[0,0,0,1,2,3,3];actual,other=conditional_statistics(x[copies],y[copies],z[copies],p[copies],g[copies])
    assert diag==other
    for key in expected:assert np.array_equal(expected[key],actual[key],equal_nan=True)
    tied,_=conditional_statistics(x,np.ones(4),z,p,g)
    assert np.array_equal(tied['raw_covariance'],np.zeros(2)) and np.isnan(tied['raw_correlation']).all()

def test_missing_direction_is_not_a_zero_cosine():
    v=np.array([[[1.,0.,0.],[np.nan,np.nan,np.nan]],[[1.,0.,0.],[0.,0.,0.]],[[np.nan,np.nan,np.nan],[0.,0.,0.]]])
    result=cross_batch_directions(v)
    assert result['observed_pair_count']==1 and result['observed_pair_mean_cosine']==1.
    assert result['direction_observed_fraction']==2/6
    absent=cross_batch_directions(np.zeros((3,2,3)))
    assert absent['observed_pair_count']==0 and np.isnan(absent['observed_pair_mean_cosine'])
