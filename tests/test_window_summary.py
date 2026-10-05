import pandas as pd
import pytest
from evomolsteer.generation.window_summary import summarize_curves

def test_global_window_integral_counts_every_observed_node_and_missingness():
    d=pd.DataFrame({'arm':['gradient']*3,'batch':[14]*3,'time':[.2,.3,.4],
                    'n_available':[10,10,10],'mean_deficit':[1.,2.,3.],'inside_fraction_available':[.1,.2,.3]})
    r=summarize_curves(d,[.2,.4],{('gradient',14):20}).iloc[0]
    assert r.nodes==3 and r.mean_deficit_time_mean==pytest.approx(2.)
    assert r.availability_mean_over_time==pytest.approx(.5) and r.inside_all_slot_time_mean==pytest.approx(.1)
    d.loc[1,'mean_deficit']=float('nan');r=summarize_curves(d,[.2,.4],{('gradient',14):20}).iloc[0]
    assert r.mean_deficit_time_mean is None and r.mean_deficit_missing_nodes==1
    with pytest.raises(ValueError):summarize_curves(d.iloc[1:],[.2,.4],{('gradient',14):20})
