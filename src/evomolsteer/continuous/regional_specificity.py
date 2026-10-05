"""Does a named region depart from the spatially shared selection pattern?

This contrasts measured selection effects, not individual molecular coordinates
or Pearson coefficients. Commonality/PCA is descriptive, not a causal assay.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from .coordinate_mining import bh
from .functional import quadrature
from ..io import read_json,write_json,write_table,digest


def analyze(mining,output):
    mining=Path(mining);out=Path(output)
    if out.exists():raise FileExistsError(out)
    m=read_json(mining/'manifest.json');cat=read_json(mining/'feature_catalog.json')['features']
    d=pd.read_parquet(mining/'batch_coordinate_statistics.parquet')
    d=d[d.batch.isin(m['splits']['discovery'])].copy()
    for key in ('region','channel','kind'):d[key]=d.feature.map(lambda f:cat[f][key])
    evidence=[];pcas={};batchrows=[];common=[];projection_rows=[]
    for (channel,kind),group in d.groupby(['channel','kind'],sort=True):
        for metric in ('selection_shift','retained_shift'):
            p=group.pivot(index=['batch','time'],columns='region',values=metric).sort_index()
            if p.isna().any().any():continue
            x=p.to_numpy();shared=x.mean(1);residual=x-shared[:,None]
            for batch,g in p.groupby(level='batch',sort=True):
                times=g.index.get_level_values('time').to_numpy();s=g.to_numpy().mean(1);rate=np.gradient(s,times)
                common.extend({'channel':channel,'kind':kind,'metric':metric,'batch':batch,'time':t,'shared_effect':value,'d_dt_shared_effect':dv} for t,value,dv in zip(times,s,rate))
            integrated=[]
            for batch,g in p.groupby(level='batch',sort=True):
                times=g.index.get_level_values('time').to_numpy();r=g.to_numpy()-g.to_numpy().mean(1)[:,None]
                integral=quadrature(times)@r;integrated.append(integral)
                change=r[-1]-r[0]
                batchrows.extend({'batch':batch,'channel':channel,'kind':kind,'metric':metric,'region':region,
                    'residual_window_mean':float(v),'end_minus_start':float(change[j]),
                    'mean_change_rate':float(change[j]/(times[-1]-times[0]))} for j,(region,v) in enumerate(zip(p.columns,integral)))
            a=np.asarray(integrated);mean=a.mean(0);se=a.std(0,ddof=1)/np.sqrt(len(a));ci=stats.t.ppf(.975,len(a)-1)*se
            for j,region in enumerate(p.columns):
                pvalue=stats.ttest_1samp(a[:,j],0).pvalue if a[:,j].std()>1e-14 else (1. if abs(mean[j])<1e-14 else 0.)
                evidence.append({'feature':f'{region}::{channel}::{kind}','metric':metric,'window_mean_residual':mean[j],
                    'ci_low':mean[j]-ci[j],'ci_high':mean[j]+ci[j],'p':pvalue,'n_batches':len(a)})
            centered=x-x.mean(0);_,s,v=np.linalg.svd(centered,full_matrices=False);den=(centered**2).sum()
            v=v[:3];signs=np.sign(v[np.arange(len(v)),np.abs(v).argmax(1)]);v*=signs[:,None]
            projection=pd.DataFrame(centered@v.T,index=p.index)
            for batch,g in projection.groupby(level='batch',sort=True):
                times=g.index.get_level_values('time').to_numpy();derivative=np.gradient(g.to_numpy(),times,axis=0)
                for k,t in enumerate(times):
                    projection_rows.append({'channel':channel,'kind':kind,'metric':metric,'batch':batch,'time':t,
                        **{f'PC{j+1}':float(g.iloc[k,j]) for j in range(len(v))},
                        **{f'd_dt_PC{j+1}':float(derivative[k,j]) for j in range(len(v))}})
            # Uniform common projection energy is more interpretable than a PCA benefit claim.
            ratio=float((len(p.columns)*(centered.mean(1)**2)).sum()/den) if den>1e-20 else None
            pcas[f'{channel}::{kind}::{metric}']={'regions':list(p.columns),'variance_ratios':(s[:3]**2/max(den,1e-20)).tolist(),
                'basis':v.tolist(),'common_projection_energy_fraction':ratio,'centering':x.mean(0).tolist(),
                'interpretation':'PCA of dependent batch/time selection-effect profiles; no significance or benefit assigned to components'}
    table=pd.DataFrame(evidence)
    for _,g in table.groupby('metric'):table.loc[g.index,'q']=bh(g.p.to_numpy())
    write_table(out/'regional_residual_evidence.csv',table)
    pd.DataFrame(batchrows).to_parquet(out/'batch_residual_integrals.parquet',index=False,compression='zstd')
    pd.DataFrame(common).to_parquet(out/'common_time_curves.parquet',index=False,compression='zstd')
    pd.DataFrame(projection_rows).to_parquet(out/'batch_time_pca_and_rates.parquet',index=False,compression='zstd')
    write_json(out/'descriptive_pca.json',pcas)
    write_json(out/'manifest.json',{'schema_version':'regional-specificity-1.0','source_sha256':digest(mining/'batch_coordinate_statistics.parquet'),
        'window':m['window'],'batches':m['splits']['discovery'],'contrasts':'regional selection shift minus arithmetic mean across all 40 aligned-receptor patch shifts',
        'limitations':['Contrasts among overlapping observations do not identify independent residues.',
            'A significant deviation need not have favorable affinity sign or causal meaning.',
            'Individual coordinate residual/affinity correlations cannot be recovered by subtracting correlations.'],
        'files':{p.name:{'bytes':p.stat().st_size,'sha256':digest(p)} for p in out.iterdir() if p.is_file()}})
    return table
