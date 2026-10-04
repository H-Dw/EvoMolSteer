"""Audit continuous artifacts without materializing the large feature cache."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from evomolsteer.io import read_json,write_json,digest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--analysis',required=True);a=p.parse_args()
    root=Path(a.analysis);cfg=read_json(root/'config.json');scope=read_json(root/'selection_scope.json')
    assert cfg['time_analysis']=='continuous_window'
    assert scope['stage_edges']==[scope['window_start'],scope['window_end']]
    records=[];nodes=0;extinct=0
    for path in sorted(root.glob('*/continuous/lineage/window_ancestry.parquet')):
        d=pd.read_parquet(path);assert not d.node_id.duplicated().any()
        assert d.step.isin(scope['steps']).all()
        for _,g in d.groupby(['arm','batch','step']):assert g.window_end_copies.sum()==len(g)
        last=d[d.step==max(scope['steps'])]
        assert np.array_equal(last.window_end_copies,last.offspring_count)
        idx=d.set_index('node_id');later=d[d.parent_node_id!='']
        accumulated=later.groupby('parent_node_id').window_end_copies.sum()
        previous=d[d.step<max(scope['steps'])]
        observed=accumulated.reindex(previous.node_id,fill_value=0).to_numpy()
        assert np.array_equal(observed,previous.window_end_copies)
        nodes+=len(d);extinct+=int((d.window_end_copies==0).sum())
        records.append({'file':str(path.relative_to(root)),'sha256':digest(path),'rows':len(d)})
    table_count=0;fit_count=0;maximum_error=0.
    for path in sorted(root.glob('*/continuous/*/batch_event_curves.parquet')):
        d=pd.read_parquet(path)
        assert 'stage' not in d and not d.duplicated(['representation','arm','batch','feature','step']).any()
        assert d.step.isin(scope['steps']).all();table_count+=1
    for path in sorted(root.glob('discovery/continuous/*/functions.json')):
        fits=read_json(path);curves=pd.read_parquet(path.parent/'fitted_curves.parquet')
        for model_id,g in curves.groupby('model_id',sort=False):
            model=fits[model_id];coef=model['power_coefficients_t'];t=g.time.to_numpy()
            error=float(np.max(np.abs(np.polynomial.polynomial.polyval(t,coef)-g.fitted)))
            assert np.allclose(np.polynomial.polynomial.polyval(t,coef),g.fitted,rtol=1e-9,atol=1e-9)
            assert model['time_start']>=scope['window_start']-1e-7 and model['time_end']<=scope['window_end']+1e-7
            maximum_error=max(maximum_error,error);fit_count+=1
    result={'passed':True,'selection_nodes_per_batch':len(scope['steps']),'window':[scope['window_start'],scope['window_end']],
        'candidate_lineage_rows':nodes,'window_extinct_candidate_rows':extinct,
        'batch_event_tables':table_count,'functions_verified':fit_count,
        'max_power_basis_evaluation_error':maximum_error,'lineage_files':records,
        'feature_cache_status':read_json(root/'feature_cache_manifest.json')['status']}
    write_json(root/'continuous_audit.json',result)
    print(result)


if __name__=='__main__':main()
