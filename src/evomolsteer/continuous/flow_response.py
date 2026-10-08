"""Compact whole-window diagnostics before final affinity/energy assessment."""
import gzip,json
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import torch
from ..io import read_json,write_json,digest
from ..generation.endpoint_reward import EndpointGeometryReward
from ..generation.innovation_reward import InnovationReward
from .coordinate_mining import curve_fit


def conditional_formula_audit(program,reference):
    p=read_json(program);r=json.loads(gzip.decompress(Path(reference).read_bytes()))
    baseline=dict(p);baseline['reward_view']='endpoint_pointcloud';baseline.pop('innovation',None)
    null=dict(p);null['innovation']={}
    old=EndpointGeometryReward(baseline,r);new=InnovationReward(p,r) if p['reward_view']=='endpoint_innovation' else old
    empty=InnovationReward(null,r);records=[]
    ids=sorted(set([0,len(r['times'])//2,len(r['times'])-1]))
    for i in ids:
        f=r['frames'][i];x=torch.tensor(np.asarray(f['teacher_endpoint_A'])[:4]+.1,dtype=torch.float32,requires_grad=True)
        mask=torch.ones(x.shape[:2],dtype=torch.bool);atoms=torch.zeros_like(mask,dtype=torch.long);anchor=x.detach()
        a,_=old(x,atoms,mask,r['times'][i],anchor);b,_=new(x,atoms,mask,r['times'][i],anchor);c,_=empty(x,atoms,mask,r['times'][i],anchor)
        ga,=torch.autograd.grad(a.sum(),x);gb,=torch.autograd.grad(b.sum(),x);gc,=torch.autograd.grad(c.sum(),x)
        records.append({'time':r['times'][i],'null_scalar_exact':torch.equal(a,c),'null_gradient_exact':torch.equal(ga,gc),
            'gradient_relative_change':float((gb-ga).norm()/ga.norm().clamp_min(1e-30))})
    return {'samples':records,'null_exact':all(v['null_scalar_exact'] and v['null_gradient_exact'] for v in records),
        'maximum_gradient_relative_change':max(v['gradient_relative_change'] for v in records),
        'semantics':'Conditional endpoint reward, fixed correspondence; not a model-Jacobian or final-affinity test'}


def mine_execution(dataset,campaign,output):
    root=Path(dataset)/'results'/campaign;out=Path(output);out.mkdir(parents=True,exist_ok=True)
    rows=[];modules=[]
    for folder in sorted(root.glob('*/batch_*')):
        batch=int(folder.name.split('_')[-1]);arm=folder.parent.name
        for r in map(json.loads,(folder/'guidance_trace.jsonl').read_text().splitlines()):
            if not r['reward_evaluated']:continue
            row={'batch':batch,'arm':arm,'time':r['score_time'],'state_time':r['state_time']}
            for key in ['flowcompat_jacobian_gain','flowcompat_native_cosine_before','flowcompat_native_cosine_after',
                'flowcompat_gradient_lag_cosine','flowcompat_schedule_factor','flowcompat_gradient_adjustment_relative_rms',
                'contrast_teacher_shift_rms_A','injection_rms_A','observed_native_rms_A','predictive_flow_rms_A']:
                if key in r and r[key] is not None:row[key]=float(np.mean(r[key]))
            rows.append(row)
            for name,v in r.get('decoder_output_sensitivity',{}).items():
                modules.append({'batch':batch,'arm':arm,'time':r['score_time'],'module':name,**v})
    table=pd.DataFrame(rows)
    if len(table):table.to_parquet(out/'flow_response.parquet',compression=None,index=False)
    if modules:pd.DataFrame(modules).to_parquet(out/'decoder_sensitivity.parquet',compression=None,index=False)
    fits={}
    if len(table) and table.batch.nunique()>1:
        for key in table.columns.difference(['batch','arm','time','state_time']):
            grid=table.pivot(index='batch',columns='time',values=key)
            if grid.shape[1]>3 and np.isfinite(grid.values).all():fits[key]=curve_fit(grid.columns.to_numpy(),grid.values,max_degree=3)
    result={'schema_version':'flow-response-mining-1.0','campaign':campaign,'independent_batches':int(table.batch.nunique()) if len(table) else 0,
        'node_batch_rows':len(table),'module_rows':len(modules),'whole_window_trends':fits,
        'semantics':'Same-forward coordinate response and existing-backward module output sensitivity; no causal module ranking, no affinity head derivative.',
        'storage':'Batch-by-time means and module scalars only; no particle feature cache or intermediate coordinate duplication.'}
    write_json(out/'flow_response_summary.json',result);return result
