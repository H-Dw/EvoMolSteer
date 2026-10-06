"""Compact, paired recorded-head response inside the learned selection support.

Scores are observations, never differentiable reward inputs. Keep one float64
list per batch/time, rather than repeated per-particle coordinate/feature rows.
Outside-window geometry is not consumed. Correlated slots are descriptive units.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from ..io import read_json,write_json,digest
from ..trajectory_source import open_trajectory
from .window_reference import load_reference


def compare(records,native):
    base_records=[r for r in native if r['arm']=='unguided']
    index={(r['batch'],r['score_time']):r for r in base_records}
    if len(index)!=len(base_records):raise ValueError('Duplicate native head node')
    rows=[]
    for r in records:
        if r['arm']!='gradient':continue
        key=(r['batch'],r['score_time'])
        if key not in index:raise ValueError('Missing exact paired native head node')
        base=index[key]
        if r['seed']!=base['seed']:raise ValueError('Unmatched seed')
        a=np.array(r['scores'],float);b=np.array(base['scores'],float)
        if a.shape!=b.shape or a.size==0 or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Incomplete paired score coverage')
        d=a-b;absolute=float(np.abs(d).mean())
        rows.append({'batch':r['batch'],'score_time':r['score_time'],'n':len(d),'mean_delta':float(d.mean()),
                     'median_delta':float(np.median(d)),'mean_absolute_delta':absolute,'positive_fraction':float((d>0).mean()),
                     'net_signal_fraction':abs(float(d.mean()))/absolute if absolute else None,
                     'cancellation_fraction':1-abs(float(d.mean()))/absolute if absolute else None})
    if not rows:raise ValueError('No guided head observations')
    table=pd.DataFrame(rows).sort_values(['batch','score_time']).reset_index(drop=True)
    if table.duplicated(['batch','score_time']).any():raise ValueError('Duplicate guided head node')
    for _,g in table.groupby('batch'):
        if len(g)<2:raise ValueError('At least two supported nodes required')
        table.loc[g.index,'d_dt_mean_delta']=np.gradient(g.mean_delta.to_numpy(),g.score_time.to_numpy())
    return table


def analyze(root,output,window,native_labels=None):
    root=Path(root);out=Path(output);cfg=read_json(root/'config.json');records=[]
    if read_json(root/'COMPLETE.json')['status']!='complete':raise ValueError('Incomplete inference')
    out.mkdir(parents=True,exist_ok=True)
    for arm in cfg['experiment']['arms'].split(','):
        for path in sorted((root/arm).glob('batch_*')):
            batch=int(path.name.split('_')[-1]);final=read_json(path/'final_records.json');seeds={r['seed'] for r in final}
            if len(seeds)!=1:raise ValueError('Recorded final seeds disagree')
            seed=int(next(iter(seeds)))
            with open_trajectory(path/'trajectory.h5') as z:
                if z['resampled'].any():raise ValueError('Paired head response is defined only without resampling')
                times=np.round(z['score_time'][:,0].astype(float),6);states=np.asarray(z['state_time'])
                ids=np.flatnonzero((times>=window[0]-1e-6)&(states<=window[1]+1e-6))
                scores=z['pic50_on'][ids].astype(np.float64)
                for k,i in enumerate(ids):
                    records.append({'arm':arm,'batch':batch,'seed':seed,'score_time':float(times[i]),'scores':scores[k].reshape(-1).tolist()})
    contract={'window':list(window),'checkpoint_sha256':cfg['extension']['checkpoint_sha256'],
              'native_integrator':cfg['extension']['native_integrator_parameters'],
              'required_input_sha256':load_reference(root/'reference.json.gz')['required_input_sha256']}
    schema=pa.schema([('arm',pa.string()),('batch',pa.int32()),('seed',pa.int64()),('score_time',pa.float64()),('scores',pa.list_(pa.float64()))])
    schema=schema.with_metadata({b'paired_native_contract':json.dumps(contract,sort_keys=True).encode()})
    labels=out/'head_scores_window.parquet';pq.write_table(pa.Table.from_pylist(records,schema=schema),labels,compression=None)
    native=[r for r in records if r['arm']=='unguided']
    baseline=None if native else native_labels
    if not native and baseline is not None:
        previous=pq.read_table(baseline)
        if json.loads(previous.schema.metadata[b'paired_native_contract'])!=contract:raise ValueError('Native checkpoint/input/integrator/window contract mismatch')
        native=previous.to_pylist()
    if not native:raise ValueError('No matching retained native labels; retain a native arm before analyzing gradient-only rounds')
    paired=compare(records,native);paired_path=out/'paired_head_window.parquet';paired.to_parquet(paired_path,compression=None,index=False)
    ends=paired.loc[paired.groupby('batch').score_time.idxmax()].to_dict('records')
    report={'schema_version':'paired-head-window-response-1.0','window':list(window),'window_end_nodes':ends,
            'analysis_code_sha256':digest(__file__),'labels_sha256':digest(labels),'labels_bytes':labels.stat().st_size,
            'paired_sha256':digest(paired_path),'inference_commit':cfg['extension']['code_commit'],
            'used_native_source':'current_native_arm' if baseline is None else str(Path(baseline)),
            'native_labels_sha256':digest(baseline) if baseline else digest(labels),
            'interpretation':'Recorded joint-head responses, not per-step fresh rescores. A low mean with large mean absolute delta indicates cancellation, not absent control. Independent inference unit is generation batch; no node/slot p-values.',
            'reward_consumes_head_scores':False,'extra_model_calls':0}
    write_json(out/'head_window_response.json',report);return report


def verify_report(output):
    out=Path(output);report=read_json(out/'head_window_response.json')
    if digest(out/'head_scores_window.parquet')!=report['labels_sha256'] or digest(out/'paired_head_window.parquet')!=report['paired_sha256']:
        raise ValueError('Paired head report checksum mismatch')
    return report
