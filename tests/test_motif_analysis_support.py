import numpy as np
import pandas as pd
from scipy.special import expit
from numpy.polynomial.legendre import legval
from evomolsteer.continuous.coordinate_mining import summarize,METRICS
from evomolsteer.io import read_json


def test_singleton_startup_is_missing_not_a_null_and_bounded_fit(tmp_path):
    rows=[];times=np.array([0.,.1,.2,.3])
    for b in range(4):
        for t in times:
            rows.append({'batch':b,'time':t,'feature':'r::all::proposal_motif_pair_3',
                **{k: .01+t/10+b*.001 for k in METRICS},
                'within_root_affinity_correlation':np.nan if t==0 else .05+b*.01})
    summary=summarize(pd.DataFrame(rows),times,{'discovery':list(range(4))},tmp_path)
    contrast=summary[summary.metric=='within_root_affinity_correlation'].iloc[0]
    assert contrast.n_batches==4 and contrast.mean_time_coverage<1
    assert pd.isna(contrast.start) and contrast.window_mean>0
    fit=read_json(tmp_path/'continuous_functions.json')['r::all::proposal_motif_pair_3']
    curve=expit(legval(np.linspace(-1,1,1001),fit['legendre_coefficients']))
    assert fit['output_transform']=='sigmoid' and ((curve>=0)&(curve<=1)).all()
