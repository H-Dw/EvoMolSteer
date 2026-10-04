"""Reconstruct reference moments from retained discovery candidate measurements."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from evomolsteer.io import read_json,write_json,write_table,digest
from evomolsteer.generation.prototypes import regularize_covariance

p=argparse.ArgumentParser();p.add_argument('--references',required=True);p.add_argument('--output',required=True)
a=p.parse_args();root=Path(a.references);out=Path(a.output)
packet=read_json(root/'reference_packet.json');data=pd.read_parquet(root/'discovery_patch_candidates.parquet')
features=packet['features'];centers=[];covs=[];spreads=[]
for time in packet['times']:
    means=[];background=[]
    for batch in packet['batches']:
        group=data[data.time.eq(time)&data.batch.eq(batch)]
        selected=group[group.arm.eq('single')]
        x=selected[features].to_numpy();count=selected.window_end_copies.to_numpy()
        finite=np.isfinite(x).all(1);w=np.where(finite,count,0)
        w=w/w.sum();mean=(np.nan_to_num(x)*w[:,None]).sum(0)
        means.append(mean)
        deviation=np.nan_to_num(x)-mean
        within=np.einsum('i,ij,ik->jk',w,deviation,deviation)
        spreads.append({'time':time,'batch':batch,'retained_candidates_n':int((count>0).sum()),
            'copy_ess':float(1/(w@w)),
            **{f'within_retained_cov_{i}_{j}_A2':float(within[i,j]) for i in range(2) for j in range(2)}})
        bg=group[group.arm.eq('unguided')][features].to_numpy()
        bg=bg[np.isfinite(bg).all(1)]
        background.append(np.cov(bg,rowvar=False,ddof=1))
    centers.append(means);covs.append(np.mean(background,axis=0))
center_error=float(np.max(np.abs(np.array(centers)-packet['centers_A'])))
covariance_error=float(np.max(np.abs(regularize_covariance(np.array(covs))-packet['covariance_A2'])))
if max(center_error,covariance_error)>1e-10:raise AssertionError('Reference reconstruction mismatch')
write_table(out/'within_retained_spread.csv',spreads)
write_json(out/'raw_background_covariance.json',{'times':packet['times'],'covariance_A2':covs,
    'definition':'Equal batch mean of within-unguided-batch sample covariance at each exact event'})
write_json(out/'reference_audit.json',{'status':'passed','max_center_error_A':center_error,
    'max_regularized_covariance_error_A2':covariance_error,'candidate_rows':len(data),
    'reference_sha256':digest(root/'reference_packet.json'),
    'candidate_table_sha256':digest(root/'discovery_patch_candidates.parquet'),
    'scope':'Discovery only; covariance and retained spread reconstructible from the preserved targeted candidate table'})
print('Reference reconstruction passed',center_error,covariance_error)
