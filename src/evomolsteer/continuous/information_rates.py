"""Whole-window derivatives of immutable compact coordinate-information results."""
from pathlib import Path
import numpy as np
import pandas as pd
from ..io import digest,read_json,write_json


def analyze(input_folder,output):
    source=Path(input_folder);out=Path(output)
    if out.exists():raise FileExistsError(out)
    summary=read_json(source/'information_summary.json');table=pd.read_parquet(source/'batch_node_information.parquet')
    metrics=[m['metric'] for m in summary['metrics']];times=np.asarray(summary['times']);rates=[];records=[]
    for b,g in table.groupby('batch',sort=True):
        g=g.sort_values('score_time')
        if not np.array_equal(g.score_time.to_numpy(),times):raise ValueError('Exact source time support required')
        derivative=np.gradient(g[metrics].to_numpy(),times,axis=0)
        rate=pd.DataFrame(derivative,columns=metrics);rate.insert(0,'score_time',times);rate.insert(0,'batch',b);rates.append(rate)
        for f in metrics:
            values=g[f].to_numpy();records.append(dict(batch=b,metric=f,
                endpoint_change_rate=(values[-1]-values[0])/(times[-1]-times[0]),
                whole_window_linear_slope=float(np.polyfit(times,values,1)[0])))
    effects=pd.DataFrame(records);rng=np.random.default_rng(42);aggregates=[]
    for f,g in effects.groupby('metric'):
        for measure in ('endpoint_change_rate','whole_window_linear_slope'):
            values=g[measure].to_numpy();boot=values[rng.integers(0,len(values),(2000,len(values)))].mean(1)
            aggregates.append(dict(metric=f,measure=measure,mean=float(values.mean()),CI95=np.quantile(boot,[.025,.975]).tolist()))
    out.mkdir(parents=True);pd.concat(rates).to_parquet(out/'batch_node_rates.parquet',compression=None,index=False)
    effects.to_parquet(out/'whole_window_batch_trends.parquet',compression=None,index=False)
    write_json(out/'report.json',{'window':summary['window'],'times':summary['times'],'trends':aggregates,
        'input_hashes':{p.name:digest(p) for p in (source/'information_summary.json',source/'batch_node_information.parquet')},
        'source_code_sha256':digest(__file__),'derivative':'Centered finite differences on actual grid; one-sided endpoints',
        'limitations':['Linear slope describes the whole window and is not used to override actual node directions',
                       'Distance/matching changes are not causal affinity effects; uncertainty uses batches']})
