import numpy as np
import pytest
from evomolsteer.continuous.dynamic_cohorts import dynamic_partition,family_weights,control_weights

def test_event_threshold_adapts_and_has_guard_band():
    s=np.r_[np.linspace(0,1,12),np.linspace(2,3,8)];roots=np.arange(len(s))
    c,w,m=dynamic_partition(s,roots);d,_,m2=dynamic_partition(s+5,roots)
    assert m['identifiable'] and np.array_equal(c,d)
    assert m2['threshold']==pytest.approx(m['threshold']+5)
    assert s[c==1].min()>s[c==-1].max() and w.sum()==pytest.approx(1)

def test_clone_replication_preserves_family_mass():
    assert family_weights([0,0,1]).tolist()==[.25,.25,.5]
    s=np.r_[np.linspace(0,1,6),np.linspace(2,3,6)]
    _,_,a=dynamic_partition(s,np.arange(12),0,1)
    _,_,b=dynamic_partition(np.repeat(s,3),np.repeat(np.arange(12),3),0,1)
    assert a['threshold']==pytest.approx(b['threshold'])

def test_flat_scores_are_ambiguous_not_negative():
    c,_,m=dynamic_partition(np.ones(10),np.arange(10))
    assert not m['identifiable'] and np.all(c==0)

def test_geometric_hardness_is_coordinate_based_and_margin_safe():
    f=np.zeros((8,12));f[:4,0]=[0,1,2,3];f[4:,0]=[.1,10,20,30]
    c=np.r_[np.ones(4),-np.ones(4)];s=np.r_[np.ones(4)*2,np.zeros(4)];w=np.ones(8)/8
    p,n,_=control_weights(f,s,c,w,hard_mix=1)
    assert n[0]>n[-1] and p.sum()==pytest.approx(1) and n.sum()==pytest.approx(1)
    changed=f.copy();changed[:,9:]+=100
    _,other,_=control_weights(changed,s,c,w,hard_mix=1)
    np.testing.assert_array_equal(n,other)

@pytest.mark.parametrize('margin',[-.1,1,float('nan')])
def test_bad_cohort_policy_rejected(margin):
    with pytest.raises(ValueError):dynamic_partition(np.arange(12),np.arange(12),margin)
