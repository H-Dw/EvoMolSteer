import numpy as np
import pytest
from evomolsteer.continuous.branch_mutation import compose_ancestors,alias_diagnostics,conditional_moments,chemistry_nuisance,conditional_teacher_contrast

def test_composed_parent_path_and_invalid_indices():
    first=np.array([2,2,1,0]);second=np.array([1,3,3,0])
    assert np.array_equal(compose_ancestors(second,first),[2,0,0,2])
    with pytest.raises(ValueError):compose_ancestors([4,0,1,2],first)

def test_exact_immediate_copy_alias_and_genuine_lag2_mutations():
    y=np.zeros((4,3,3));y[2:,:,0]=1.;score=np.array([1.,1.,3.,3.]);parents=np.array([0,0,1,1])
    d=alias_diagnostics(y,score,parents)
    assert d['immediate_coordinate_range_max_A']==0 and d['immediate_score_range_max']==0
    r,s=conditional_moments(np.array([[0.],[0.],[1.],[1.]]),score,np.zeros((4,2)),parents,np.zeros(4,int))
    assert r['raw_correlation'][0]==1 and s['lag2_branching_ancestor_count']==1
    assert s['lag2_distinct_parent_branches']==2

def test_copy_inflation_does_not_change_ancestor_moments():
    x=np.array([[0.,2.],[1.,1.],[2.,5.],[4.,3.]])
    y=np.array([1.,3.,2.,6.]);p=np.arange(4);g=np.array([0,0,1,1]);z=np.zeros((4,2))
    expected,diagnostic=conditional_moments(x,y,z,p,g)
    copies=np.array([0,0,0,0,1,2,3,3]);actual,other=conditional_moments(x[copies],y[copies],z[copies],p[copies],g[copies])
    for key in expected:assert np.array_equal(expected[key],actual[key],equal_nan=True)
    assert diagnostic==other

def test_identical_or_equal_score_nulls_and_nuisance_projection():
    p=np.arange(6);g=np.array([0,0,0,1,1,1]);x=np.array([[-1.],[0.],[1.],[-2.],[0.],[2.]])
    tied,_=conditional_moments(x,np.ones(6),np.zeros((6,2)),p,g)
    assert np.array_equal(tied['raw_covariance'],[0.]) and np.isnan(tied['raw_correlation']).all()
    constant,_=conditional_moments(np.full((6,1),.123456789),x[:,0],np.zeros((6,2)),p,g)
    assert np.array_equal(constant['raw_covariance'],[0.]) and np.isnan(constant['raw_correlation']).all()
    controls=np.column_stack([x[:,0],np.zeros(6)])
    r,_=conditional_moments(x,x[:,0],controls,p,g)
    assert r['raw_correlation'][0]==1
    assert abs(r['adjusted_covariance'][0])<1e-25 and np.isnan(r['adjusted_correlation'][0])

def test_nuisance_counts_unordered_edges_without_graph_veto():
    a=np.zeros((2,3),int);b=np.zeros((2,3,3),int);a[1,0]=1;b[1,0,1]=b[1,1,0]=2
    z=chemistry_nuisance(a,b,np.zeros_like(a),np.zeros_like(b),[0,0])
    assert np.allclose(z,[[0.,0.],[1/3,1/3]])

def test_teacher_field_uses_conditional_mutations_not_initial_roots():
    anchor=np.array([[0.,0.,0.],[4.,0.,0.],[0.,5.,0.]])
    clouds=np.stack([anchor,anchor-[.1,0,0],anchor-[.2,0,0]])
    parents=np.arange(3);ancestors=np.zeros(3,int);scores=np.array([3.,2.,1.])
    base=conditional_teacher_contrast(anchor,3.,clouds,scores,parents,ancestors,0,0)
    assert base['confidence']>0 and np.all(base['direction_unit'][:,0]>0)
    assert base['distinct_observed_mutations']==3
    copies=np.array([0,0,0,1,2,2]);repeat=conditional_teacher_contrast(anchor,3.,clouds[copies],scores[copies],parents[copies],ancestors[copies],0,0)
    assert np.array_equal(base['direction_unit'],repeat['direction_unit']) and base['confidence']==repeat['confidence']
    tied=conditional_teacher_contrast(anchor,1.,clouds,np.ones(3),parents,ancestors,0,0)
    identical=conditional_teacher_contrast(anchor,3.,np.repeat(anchor[None],3,0),scores,parents,ancestors,0,0)
    for result in (tied,identical):assert np.array_equal(result['direction_unit'],np.zeros_like(anchor)) and result['confidence']==0
