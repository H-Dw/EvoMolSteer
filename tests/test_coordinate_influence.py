import numpy as np
import pandas as pd
from evomolsteer.io import write_json
from evomolsteer.continuous.coordinate_influence import analyze


def test_window_integral_identity_and_startup_dominance(tmp_path):
    src=tmp_path/'mining';src.mkdir();t=np.linspace(.03,.47,45)
    write_json(src/'manifest.json',{'times':t.tolist(),'window':[.03,.47],'splits':{'discovery':[0,1,2]}})
    rows=[{'batch':b,'time':float(v),'feature':'linear','selection_shift':2*v+1} for b in range(3) for v in t]
    rows+=[{'batch':b,'time':float(v),'feature':'startup','selection_shift':-100 if i==0 else 0} for b in range(3) for i,v in enumerate(t)]
    pd.DataFrame(rows).to_parquet(src/'batch_coordinate_statistics.parquet',index=False)
    out=tmp_path/'out';analyze(src,out,metrics=('selection_shift',))
    d=pd.read_csv(out/'whole_window_node_influence.csv').set_index('feature')
    assert abs(d.loc['linear','signed_window_effect']-1.5)<1e-12
    assert abs(d.loc['startup','first_fraction_absolute_contributions']-1)<1e-12
    assert abs(d.loc['startup','effective_time_nodes']-1)<1e-12
    assert d.loc['startup','dominant_time']==t[0]
