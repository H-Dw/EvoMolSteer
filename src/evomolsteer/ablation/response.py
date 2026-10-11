"""Compact execution response comparison on matched actual control nodes."""
import numpy as np
import pandas as pd
from pathlib import Path
from evomolsteer.generation.window_reference import load_reference
from evomolsteer.continuous.m2_skill_api import REF


def observed_response(repo, study, candidate, control):
    repo,study=Path(repo),Path(study)
    folder=study/'transport_records'
    a=pd.read_parquet(folder/(candidate['name']+'.forecast_features.parquet')).sort_values(['batch','score_time'])
    b=pd.read_parquet(folder/(control['name']+'.forecast_features.parquet')).sort_values(['batch','score_time'])
    keys=['batch','score_time','state_time']
    if not a[keys].reset_index(drop=True).equals(b[keys].reset_index(drop=True)):
        raise ValueError('Execution response nodes must be exactly paired')
    reference=load_reference(repo/REF)
    columns=reference['features'];scale=np.asarray(reference['feature_scale'],float)
    if scale.shape!=(len(columns),) or not np.isfinite(scale).all() or np.any(scale<=0):
        raise ValueError('Same finite positive observable scales required')
    delta=(a[columns].to_numpy()-b[columns].to_numpy())/scale
    if not np.isfinite(delta).all():raise ValueError('Missing execution response')
    rms=np.sqrt(np.mean(delta**2,axis=0));rank=np.argsort(-rms,kind='stable')[:10]
    node=pd.DataFrame({'time':a.score_time.to_numpy(),'rms_z':np.sqrt(np.mean(delta**2,axis=1))}).groupby('time').rms_z.mean()
    return {'paired_observed_nodes':len(a),'same_numerical_program':candidate['effective_signature']==control['effective_signature'],
        'mean_forecast_difference_z':float(np.sqrt(np.mean(delta**2))),
        'max_forecast_difference_z':float(np.max(np.abs(delta))),
        'top_changed_observables':[{'feature':columns[i],'RMS_difference_z':float(rms[i])} for i in rank],
        'whole_window_observed_node_RMS':{'times':node.index.tolist(),'RMS_z':node.tolist()},
        'comparison':'Actual pre-injection endpoint forecasts under paired initial states; not a direct affinity-gradient effect or individual-path reconstruction',
        'instruction_response_not_assumed_from_program_hash':True}
