"""Expose smoothing bias; a conditional fit band is not a model-adequacy test."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from evomolsteer.io import read_json,write_table,write_json,digest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--analysis',required=True);a=p.parse_args()
    root=Path(a.analysis);all_rows=[];sources={}
    for path in sorted(root.glob('discovery/continuous/*/fitted_curves.parquet')):
        data=pd.read_parquet(path);functions=read_json(path.parent/'functions.json')
        sources[str(path.relative_to(root))]=digest(path)
        rows=[]
        for model_id,g in data.groupby('model_id',sort=True):
            model=functions[model_id];obs=g.observed_mean.to_numpy();fit=g.fitted.to_numpy()
            # Numerically trivial effects are not labelled meaningful sign changes.
            relevant=(np.abs(obs)>1e-12)&(np.abs(fit)>1e-12)
            mismatch=relevant&(np.sign(obs)!=np.sign(fit))
            outside=(obs<g.simultaneous_low.to_numpy())|(obs>g.simultaneous_high.to_numpy())
            error=np.abs(obs-fit);index=int(np.argmax(error))
            rows.append({'method':path.parent.name,'model_id':model_id,'feature':model['feature'],
                'representation':model['representation'],'arm':model['arm'],'contrast':model['contrast'],
                'degree':model['degree'],'mean_curve_r2':model['mean_curve_r2'],
                'max_abs_residual':float(error[index]),'max_residual_time':float(g.time.iloc[index]),
                'sign_mismatch_times':int(mismatch.sum()),'n_times':len(g),
                'observed_mean_outside_conditional_band_times':int(outside.sum()),
                'safe_to_compile_as_pointwise_guidance':False,
                'interpretation':'Descriptive curve; model bias, local uncertainty, and spatial derivatives still require validation'})
        write_table(path.parent/'function_adequacy.csv',rows);all_rows.extend(rows)
    write_json(root/'function_adequacy_audit.json',{'functions':len(all_rows),'sources':sources,
        'warning':'These are in-sample descriptive diagnostics, not extra hypothesis tests. Bootstrap bands condition on model form and omit approximation bias.',
        'fits_with_sign_mismatch':sum(r['sign_mismatch_times']>0 for r in all_rows),
        'no_coefficients_or_model_degrees_changed':True})
    print('Function adequacy diagnostics:',len(all_rows))


if __name__=='__main__':main()
