"""Local gradient-scale calibration for declarative typed/bond geometry weights.

Uses every eligible state time of completed adaptation trajectories; never invokes
the FLOWR model. The result is an engineering starting weight, not an optimum.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from evomolsteer.io import read_json,write_json,digest
from evomolsteer.trajectory_source import open_trajectory
from evomolsteer.generation.window_reward import WindowReward


def calibrate(dataset,campaign,program,reference,output,fraction=.25):
    torch.set_num_threads(1)
    root=Path(dataset)/'results'/campaign;p=read_json(program);reward=WindowReward(p,str(reference))
    cfg=read_json(root/'config.json');scale=cfg['coord_scale'];rows=[]
    for path in sorted((root/'gradient').glob('batch_*')):
        batch=int(path.name.split('_')[-1]);com=np.asarray(read_json(root/f'frame_batch_{batch:03d}.json')['target_com'])
        with open_trajectory(path/'trajectory.h5') as z:
            times=z['score_time'][:,0].astype(float);state_times=z['state_time'].astype(float)
            ids=np.flatnonzero([reward.active(t,s) for t,s in zip(times,state_times)])
            # Four equally spaced slots per batch; all state times, no time bins.
            slots=np.arange(0,len(com),max(1,len(com)//4))[:4]
            for step in ids:
                x=torch.tensor(z['native_proposal_coords'][step][slots].astype(float)*scale+com[slots,None,:],dtype=torch.float64,requires_grad=True)
                atoms=torch.tensor(z['proposal_atomics'][step][slots].astype(np.int64));bonds=torch.tensor(z['proposal_bonds'][step][slots].astype(np.int64))
                costs,mass=reward.component_costs(x,atoms,bonds,float(state_times[step]),compute_all=True)
                gradients={}
                for kind,cost in costs.items():
                    value=p['temperature']*torch.logsumexp(mass.log()[None]-cost/p['temperature'],1)
                    gradients[kind]=torch.autograd.grad(value.sum(),x,retain_graph=True)[0].detach()
                norms={k:g.square().sum((1,2)).div(x.shape[1]).sqrt() for k,g in gradients.items()}
                for k,slot in enumerate(slots):
                    r={'batch':batch,'slot':int(slot),'state_time':float(state_times[step])}
                    for kind in ['typed','bond']:
                        r[f'{kind}_gradient_rms']=float(norms[kind][k]);r['shape_gradient_rms']=float(norms['shape'][k])
                        r[f'{kind}_cosine_shape']=float((gradients[kind][k]*gradients['shape'][k]).sum()/(gradients[kind][k].norm()*gradients['shape'][k].norm()).clamp_min(1e-30))
                    rows.append(r)
    df=pd.DataFrame(rows);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    df.to_parquet(out/'component_gradient_scales.parquet',index=False)
    result={'source_campaign':campaign,'source_program_sha256':digest(program),'reference_sha256':digest(reference),'states_measured':len(rows),
            'rule':'lambda = fraction * median(shape gradient RMS / auxiliary gradient RMS) over nonzero gradients; clipped to [1e-4,10]',
            'fraction':fraction,'limitation':'Pure-component mixtures differ from the joint mixture; these are initial scales, not a guarantee of joint contribution size. Hard labels are detached; no physical energy.'}
    for kind in ['typed','bond']:
        valid=(df[f'{kind}_gradient_rms']>1e-10)&(df['shape_gradient_rms']>1e-10)
        raw=fraction*np.median(df.loc[valid,'shape_gradient_rms']/df.loc[valid,f'{kind}_gradient_rms'])
        result[kind]={'weight':float(np.clip(raw,1e-4,10)),'unclipped':float(raw),'available':int(valid.sum()),
                      'median_cosine_shape':float(df.loc[valid,f'{kind}_cosine_shape'].median())}
    write_json(out/'calibration.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--campaign',required=True)
    p.add_argument('--program',required=True);p.add_argument('--reference',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(calibrate(a.dataset,a.campaign,a.program,a.reference,a.output),indent=2))
