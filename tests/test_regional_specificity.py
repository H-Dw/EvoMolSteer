import pandas as pd
import numpy as np
from evomolsteer.io import write_json,read_json
from evomolsteer.continuous.regional_specificity import analyze


def test_common_motion_is_not_unique_regional_advantage(tmp_path):
    source=tmp_path/'mining';source.mkdir();features={f'r{r}::all::spread':{'region':f'r{r}','channel':'all','kind':'spread'} for r in range(4)}
    write_json(source/'manifest.json',{'splits':{'discovery':list(range(5))},'window':[.1,.4]})
    write_json(source/'feature_catalog.json',{'features':features})
    rows=[{'batch':b,'time':t,'feature':f,'selection_shift':-t*(1+b*.1),'retained_shift':-2*t*(1+b*.1)} for b in range(5) for t in [.1,.2,.3,.4] for f in features]
    pd.DataFrame(rows).to_parquet(source/'batch_coordinate_statistics.parquet',index=False)
    output=tmp_path/'out';summary=analyze(source,output)
    assert summary.q.eq(1).all();assert np.allclose(summary.window_mean_residual,0)
    pca=read_json(output/'descriptive_pca.json')['all::spread::selection_shift']
    assert abs(pca['common_projection_energy_fraction']-1)<1e-12
    projection=pd.read_parquet(output/'batch_time_pca_and_rates.parquet');assert 'd_dt_PC1' in projection
