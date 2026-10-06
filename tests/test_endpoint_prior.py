from pathlib import Path
import numpy as np
from evomolsteer.continuous.endpoint_prior import build_prior
from evomolsteer.io import read_json


def test_actual_endpoint_prior_retains_uncertainty_and_avoids_floor_fields(tmp_path):
    root=Path(__file__).resolve().parents[1];mining=root/'docs/experiments/ck2_affinity_geometry30_20261007/endpoint_mining'
    r=build_prior(root/'configs/experiments/ck2_affinity_geometry30_v1/endpoint_reference.json.gz',
                  mining/'whole_window_effects.parquet',mining/'effect_functions.json',tmp_path/'prior.json')
    assert r['selected_count']==24 and len(r['evidence'])==74 and r['n_independent_batches']==14
    assert len(r['direction_functions'])==24 and r['window']==[0,.5]
    assert all(v['scale']>1e-5 and v['q']<.05 and v['batch_agreement']>=12/14 for v in r['evidence'] if v['selected'])
    pair=r['direction_functions']['pair_kernel_2.5']
    assert pair['degree']==2 and not np.allclose(np.polynomial.legendre.legval([-1,0,1],pair['legendre_coefficients']),0)
    assert read_json(tmp_path/'prior.json')==r
