import numpy as np
from evomolsteer.continuous.selection_pressure import price_terms,coordinate_niches
from evomolsteer.generation.selection_path_reward import ess_floor_prior

def test_exact_price_identity_and_sampling_noise():
    x=np.array([[0.,1.],[1.,0.],[2.,3.]])
    y=x+np.array([[1.,0.],[2.,1.],[-1.,0.]])
    counts=np.array([0,2,1]);next_x=y[[1,1,2]]
    r=price_terms(x,y,[.1,.5,.4],counts,next_x)
    assert r['identity_error']<1e-12
    assert np.allclose(r['native']+r['realized_selection'],next_x.mean(0)-x.mean(0))
    assert np.allclose(r['resampling_noise'],r['realized_selection']-r['expected_selection'])

def test_ess_base_chance_preserves_mass_and_rank():
    p,epsilon=ess_floor_prior([20.,0.,-10.,-15.],.75)
    assert abs(p.sum()-1)<1e-12 and epsilon>0
    assert 1/(p@p)>=3-1e-12
    assert np.all(np.diff(p)<=0)
    unchanged,zero=ess_floor_prior([0,0,0],.9)
    assert zero==0 and np.allclose(unchanged,1/3)

def test_niche_metric_spd_and_permutation_invariance():
    rng=np.random.default_rng(42);clouds=rng.normal(size=(8,5,3));batches=np.repeat(np.arange(4),2)
    groups,metric,_=coordinate_niches(clouds,batches,3)
    assert np.linalg.eigvalsh(metric).min()>0
    assert np.allclose(np.trace(metric,axis1=-2,axis2=-1),3)
    permutation=rng.permutation(5)
    g2,m2,_=coordinate_niches(clouds[:,permutation],batches,3)
    assert np.array_equal(groups,g2)
    assert np.allclose(metric[:,permutation],m2)
