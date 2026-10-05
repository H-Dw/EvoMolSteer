"""Compact actual-state monitoring, separate from independent full-shape metrics."""
import json
from pathlib import Path
import numpy as np
import torch
from ..io import read_json,write_json,write_table,digest
from ..trajectory_source import open_trajectory
from .window_reference import load_reference
from .coordinate_reward import CoordinateMixtureReward


def evaluate(dataset,campaign,output):
    root=Path(dataset)/'results'/campaign;out=Path(output)
    if not (root/'COMPLETE.json').is_file():raise ValueError('Incomplete inference')
    cfg=read_json(root/'config.json');program=read_json(root/'reward_program.json')
    reference=load_reference(root/'reference.json.gz');reward=CoordinateMixtureReward(program,reference)
    rows=[];audit=[];motion=[];torch.set_num_threads(1)
    for arm in cfg['experiment']['arms'].split(','):
        for path in sorted((root/arm).glob('batch_*')):
            batch=int(path.name.split('_')[-1]);com=np.asarray(read_json(root/f'frame_batch_{batch:03d}.json')['target_com'])[:,None,:]
            with open_trajectory(path/'trajectory.h5') as z:
                if z['resampled'].any():raise ValueError('Particle resampling prohibited')
                times=np.round(z['score_time'][:,0].astype(float),6)
                ids=np.flatnonzero((times>=reference['window'][0]-1e-7)&(times<=reference['window'][1]+1e-7))
                coords=z['current_coords'][ids].astype(float)*cfg['coord_scale']+com[None]
                atoms=z['predicted_atomics'][ids];mask=z['mask'][ids]
                native=z['native_proposal_coords'].astype(float);proposal=z['proposal_coords'].astype(float)
                displacement=(proposal-native)*cfg['coord_scale']
                translation=displacement.mean(2);centered=displacement-translation[:,:,None]
                for i,t in enumerate(times):
                    motion.append({'arm':arm,'batch':batch,'time':float(t),
                        'mean_translation_injection_A':float(np.linalg.norm(translation[i],axis=1).mean()),
                        'mean_nontranslation_injection_rms_A':float(np.sqrt((centered[i]**2).sum(-1).mean(1)).mean()),
                        'mean_total_injection_rms_A':float(np.sqrt((displacement[i]**2).sum(-1).mean(1)).mean())})
            for k,i in enumerate(ids):
                t=float(times[i])
                with torch.no_grad():
                    value,detail=reward(torch.tensor(coords[k]),torch.tensor(atoms[k]),torch.tensor(mask[k]),t)
                available=detail['available'].numpy();features=detail['observables_A'].numpy()
                row={'arm':arm,'batch':batch,'time':t,'n_available':int(available.sum()),
                     'mean_reward':float(value[available].mean()) if available.any() else None,
                     'mean_standardized_residual':float(detail['nearest_standardized_rms'][available].mean()) if available.any() else None,
                     **{f:float(np.mean(features[available,j])) if available.any() else None for j,f in enumerate(reference['features'])}}
                rows.append(row)
            trace=[json.loads(v) for v in (path/'guidance_trace.jsonl').read_text().splitlines()]
            active=[v for v in trace if v['active']]
            audit.append({'arm':arm,'batch':batch,'active_steps':len(active),
                'reward_evaluated_steps':sum(v['reward_evaluated'] for v in trace),
                'nonzero_injection_steps':sum(max(v['injection_l2_A'])>0 for v in trace),
                'mean_path_rms_A':float(np.mean(active[-1]['cumulative_rms_A'])) if active else 0.,
                'guard_rejection_fraction':float(np.mean([np.logical_not(v['geometry_accepted']) for v in active])) if active else 0.,
                'cap_fraction':float(np.mean([np.asarray(v['cap_factor'])<.99999 for v in active])) if active else 0.,
                'no_injection_after_window':all(max(v['injection_l2_A'])==0 for v in trace if v['state_time']>reference['window'][1]+1e-6)})
    table=write_table(out/'coordinate_time_metrics.csv',rows)
    for _,g in table.groupby(['arm','batch']):
        for f in ['mean_reward','mean_standardized_residual',*reference['features']]:
            table.loc[g.index,'d_dt_'+f]=np.gradient(g[f],g.time)
    write_table(out/'coordinate_time_rates.csv',table)
    write_table(out/'coordinate_injection_motion.csv',motion)
    preflight=read_json(root/'coordinate_gradient_preflight.json')
    if not preflight['passed']:raise ValueError('Coordinate derivative failed')
    report={'schema_version':'current-coordinate-evaluation-1.0','window':reference['window'],'batch_results':audit,
        'coordinate_preflight':preflight,'inference_commit':cfg['extension']['code_commit'],
        'program_sha256':digest(root/'reward_program.json'),'reference_sha256':digest(root/'reference.json.gz'),
        'endpoint_at_window_end':table[np.isclose(table.time,reference['window'][1])].to_dict('records'),
        'interpretation':'Current reward response is a proxy; use independent whole-shape and terminal measures for conclusions'}
    write_json(out/'coordinate_audit.json',report);return report
