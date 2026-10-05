import numpy as np
from evomolsteer.continuous.coordinate_features import regional_observables,regional_moments
from evomolsteer.continuous.coordinate_mining import correlation,partial_correlation,lag_evidence,window_descendants,curve_fit


def catalog():
    return {'regions':{'r':{'points_A':[[0.,0.,0.],[0.,1.,0.]]}},'atom_vocabulary':{'N':4,'O':5,'S':9}}


def test_permutation_and_rigid_frame_equivariance():
    rng=np.random.default_rng(42);x=rng.normal(size=(12,5,3));end=x+.2;proposal=x+.03
    atoms=np.tile([4,3,5,3,3],(12,1));mask=np.ones((12,5),bool)
    a,n,_=regional_observables(x,end,proposal,atoms,mask,catalog(),.2,.01)
    order=[4,1,3,0,2]
    b,nn,_=regional_observables(x[:,order],end[:,order],proposal[:,order],atoms[:,order],mask[:,order],catalog(),.2,.01)
    np.testing.assert_allclose(a,b,atol=1e-12);assert n==nn
    shifted=catalog();shifted['regions']['r']['points_A']=(np.array(shifted['regions']['r']['points_A'])+7).tolist()
    c,_,_=regional_observables(x+7,end+7,proposal+7,atoms,mask,shifted,.2,.01)
    np.testing.assert_allclose(a,c,atol=1e-12)
    assert np.allclose(a[:,n.index('r::all::endpoint_transport_x')],.25)


def test_no_fictitious_missing_nos():
    x=np.ones((10,4,3));a=np.full((10,4),3);m=np.ones((10,4),bool)
    z,n,_=regional_observables(x,x,x,a,m,catalog(),0.,.01)
    assert np.isnan(z[:,[i for i,s in enumerate(n) if '::NOS::' in s]]).all()
    assert np.isfinite(regional_moments(x,a,m,[[0,0,0]],None)).all()


def test_endpoint_anchored_proposal_parity():
    rng=np.random.default_rng(42);x=rng.normal(size=(12,5,3));end=rng.normal(size=x.shape)*5;p=x+.1
    atoms=np.tile([4,3,5,3,3],(12,1));mask=np.ones((12,5),bool)
    z,n,meta=regional_observables(x,end,p,atoms,mask,catalog(),.2,.01,spatial_anchor='endpoint',include_proposal=True)
    expected=regional_moments(p,atoms,mask,catalog()['regions']['r']['points_A'],None,4.,end)
    np.testing.assert_allclose(z[:,[n.index('r::all::proposal_'+k) for k in ('centroid_x','centroid_y','centroid_z','spread')]],expected,atol=1e-12)
    assert meta['r::all::proposal_spread']['spatial_anchor']=='endpoint'


def test_transport_coherence_support_and_rotation_invariance():
    rng=np.random.default_rng(42);x=rng.normal(size=(12,5,3));end=x+.2;proposal=x+.03
    atoms=np.tile([4,3,5,3,3],(12,1));mask=np.ones((12,5),bool)
    z,n,_=regional_observables(x,end,proposal,atoms,mask,catalog(),.2,.01,feature_family='transport')
    assert len(n)==10
    np.testing.assert_allclose(z[:,n.index('r::all::transport_coherence')],1,atol=1e-12)
    neg,_,_=regional_observables(x,end,x-.03,atoms,mask,catalog(),.2,.01,feature_family='transport')
    np.testing.assert_allclose(neg[:,n.index('r::all::transport_coherence')],-1,atol=1e-12)
    zero,_,_=regional_observables(x,end,x,atoms,mask,catalog(),.2,.01,feature_family='transport')
    assert np.isnan(zero[:,n.index('r::all::transport_coherence')]).all()
    assert np.all((z[:,n.index('r::all::effective_slots')]>=1)&(z[:,n.index('r::all::effective_slots')]<=5))
    rot=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
    cat=catalog();cat['regions']['r']['points_A']=(np.asarray(cat['regions']['r']['points_A'])@rot).tolist()
    rotated,_,_=regional_observables(x@rot,end@rot,proposal@rot,atoms,mask,cat,.2,.01,feature_family='transport')
    np.testing.assert_allclose(z,rotated,atol=1e-12)


def test_partial_and_parent_deduplication():
    rng=np.random.default_rng(42);u=rng.normal(size=40);v=rng.normal(size=40)
    x=(u+.2*v)[:,None];score=2*u+.3*rng.normal(size=40)
    assert correlation(x,score)[0]>.9
    assert abs(partial_correlation(x,score,u[:,None])[0])<.4
    chosen=np.repeat(np.arange(20),2);next_score=np.repeat(score[:20]+v[:20],2)
    raw,adj,n=lag_evidence(v[:,None],score,next_score,chosen)
    assert n==20;assert raw[0]>.999999;assert adj[0]>.99
    missing=np.full(40,np.nan);missing[:20]=v[:20]
    mixed=partial_correlation(np.column_stack([x[:,0],missing]),score,u[:,None])
    np.testing.assert_allclose(mixed[0],partial_correlation(x,score,u[:,None])[0],atol=1e-12)


def test_lineage_and_whole_window_fit():
    ids=np.array([[0,0,1,3],[0,2,2,3]])
    np.testing.assert_array_equal(window_descendants(ids),[[1,2,0,1],[1,0,2,1]])
    t=np.linspace(.03,.47,45);y=np.stack([1+2*t+k*.001 for k in range(10)])
    fit=curve_fit(t,y);assert fit['degree']==1
    from numpy.polynomial import Legendre
    f=Legendre(fit['legendre_coefficients'],domain=[fit['time_start'],fit['time_end']])
    np.testing.assert_allclose(f.deriv()(t),2,atol=1e-10)
