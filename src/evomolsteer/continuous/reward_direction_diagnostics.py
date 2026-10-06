"""Streaming offline direction sensitivity; no model calls or particle records."""
import copy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from ..io import read_json,digest,write_table
from ..storage.trajectory import TrajectoryPackage
from ..generation.prototypes import write_json
from ..generation.coordinate_contrast import make_coordinate_reward
from ..generation.window_reference import load_reference


def audit(dataset,campaign,program,reference,output,temperatures=(.05,1.),batches=(0,1)):
    root,program,reference,output=map(Path,(dataset,program,reference,output))
    if output.exists():raise FileExistsError(output)
    p,ref=read_json(program),load_reference(reference)
    if digest(reference)!=p['reference_sha256'] or p['reward_view'] not in ('coordinate_mixture','shape_mixture'):
        raise ValueError('Bound mixture reference with observable mode responsibilities required')
    values=tuple(float(v) for v in temperatures)
    if any(not np.isfinite(v) or v<=0 or v==p['mixture_temperature'] for v in values):
        raise ValueError('Distinct finite positive temperatures required')
    source=root/'results'/campaign;scale=read_json(source/'config.json')['coord_scale']
    rewards={p['mixture_temperature']:make_coordinate_reward(p,ref)}
    for t in values:
        q=copy.deepcopy(p);q['mixture_temperature']=t;rewards[t]=make_coordinate_reward(q,ref)
    torch.set_num_threads(1);rows=[];sources=[]
    for batch in batches:
        path=source/'unguided'/f'batch_{batch:03d}'/'trajectory.h5'
        com=np.asarray(read_json(source/f'frame_batch_{batch:03d}.json')['target_com'])[:,None]
        with TrajectoryPackage(path) as package:
            if package.read('resampled').any():raise ValueError('Native directions require unselected data')
            times=np.round(package.read('score_time')[:,0].astype(float),6)
            for i in np.flatnonzero(np.isin(times,ref['times'])):
                time=float(times[i]);dt=float(package.read('step_size',int(i)).reshape(-1)[0])
                if time+dt>ref['window'][1]+1e-6:continue
                view=ref.get('control_representation','current')
                x=torch.tensor(package.read(view+'_coords',int(i)).astype(float)*scale+com,requires_grad=True)
                atoms=torch.tensor(package.read('predicted_atomics',int(i)));mask=torch.tensor(package.read('mask',int(i)))
                anchor=torch.tensor(package.read('predicted_coords',int(i)).astype(float)*scale+com)
                gradients={};details={}
                for tau,reward in rewards.items():
                    value,detail=reward(x,atoms,mask,time,anchor)
                    gradients[tau]=torch.autograd.grad(value.sum(),x)[0].detach().numpy().reshape(len(x),-1)
                    details[tau]=detail
                baseline=gradients[p['mixture_temperature']];norm=np.linalg.norm(baseline,axis=1)
                for tau in values:
                    g=gradients[tau];other=np.linalg.norm(g,axis=1);valid=(norm>1e-10)&(other>1e-10)
                    cosine=np.divide((baseline*g).sum(1),norm*other,out=np.full(len(x),np.nan),where=valid).clip(-1,1)
                    row={'batch':batch,'time':time,'temperature':tau,'n_attempted':len(x),'n_nonzero_paired':int(valid.sum()),
                         'mean_cosine':float(cosine[valid].mean()) if valid.any() else None,
                         'fraction_cosine_below_095':float(np.mean(cosine[valid]<.95)) if valid.any() else None,
                         'norm_ratio_mean':float((other[valid]/norm[valid]).mean()) if valid.any() else None}
                    if 'mode_responsibilities' in details[tau]:
                        row['mean_mode_ESS']=float((1/details[tau]['mode_responsibilities'].detach().square().sum(1)).mean())
                    rows.append(row)
        sources.append({'path':path.relative_to(root).as_posix(),'sha256':digest(path)})
    output.mkdir(parents=True);table=write_table(output/'batch_time_directions.csv',rows)
    summary=table.groupby('temperature')[['mean_cosine','fraction_cosine_below_095','norm_ratio_mean','mean_mode_ESS']].mean().to_dict('index')
    record={'window':ref['window'],'program_sha256':digest(program),'reference_sha256':digest(reference),
        'parent_temperature':p['mixture_temperature'],'trial_temperatures':values,'sources':sources,'summary':summary,
        'interpretation':'Offline native discovery controls only; direction change is not evidence of better generation. Norm changes are not dose changes under unit-gradient normalization.',
        'storage':'Per-batch/time scalar summaries only; no structures, features or per-particle gradients retained'}
    write_json(output/'audit.json',record);return record
