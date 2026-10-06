"""Compact actual-state monitoring, separate from independent full-shape metrics."""
import json
from pathlib import Path
import numpy as np
import torch
from ..io import read_json,write_json,write_table,digest
from ..trajectory_source import open_trajectory
from .window_reference import load_reference
from .coordinate_contrast import make_coordinate_reward


def evaluate(dataset,campaign,output):
    root=Path(dataset)/'results'/campaign;out=Path(output)
    if not (root/'COMPLETE.json').is_file():raise ValueError('Incomplete inference')
    cfg=read_json(root/'config.json');program=read_json(root/'reward_program.json')
    reference=load_reference(root/'reference.json.gz');reward=make_coordinate_reward(program,reference)
    rows=[];audit=[];motion=[];dose_rows=[];torch.set_num_threads(1)
    for arm in cfg['experiment']['arms'].split(','):
        for path in sorted((root/arm).glob('batch_*')):
            batch=int(path.name.split('_')[-1]);com=np.asarray(read_json(root/f'frame_batch_{batch:03d}.json')['target_com'])[:,None,:]
            with open_trajectory(path/'trajectory.h5') as z:
                if z['resampled'].any():raise ValueError('Particle resampling prohibited')
                times=np.round(z['score_time'][:,0].astype(float),6)
                ids=np.flatnonzero((times>=reference['window'][0]-1e-7)&(times<=reference['window'][1]+1e-7))
                view=reference.get('control_representation','current')
                coords=z[f'{view}_coords'][ids].astype(float)*cfg['coord_scale']+com[None]
                anchors=z['predicted_coords'][ids].astype(float)*cfg['coord_scale']+com[None] if reference.get('spatial_anchor')=='endpoint' else None
                state_times=z['state_time'][ids].astype(float) if view=='proposal' else times[ids]
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
                    value,detail=reward(torch.tensor(coords[k]),torch.tensor(atoms[k]),torch.tensor(mask[k]),t,
                        torch.tensor(anchors[k]) if anchors is not None else None)
                available=detail['available'].numpy()
                features=detail['observables' if 'observables' in detail else 'observables_A'].numpy()
                row={'arm':arm,'batch':batch,'time':t,'state_time':float(state_times[k]),'within_control_state_window':bool(state_times[k]<=reference['window'][1]+1e-6),'n_available':int(available.sum()),
                     'mean_reward':float(value[available].mean()) if available.any() else None,
                     'mean_standardized_residual':float(detail['nearest_standardized_rms'][available].mean()) if available.any() else None,
                     **{f:float(np.mean(features[available,j])) if available.any() else None for j,f in enumerate(reference['features'])}}
                rows.append(row)
            trace=[json.loads(v) for v in (path/'guidance_trace.jsonl').read_text().splitlines()]
            active=[v for v in trace if v['active']]
            for v in trace:
                def average(key,fallback=None):
                    values=v.get(key,v.get(fallback) if fallback else None)
                    return float(np.mean(values)) if values is not None else None
                dose_rows.append({'arm':arm,'batch':batch,'score_time':v['score_time'],'state_time':v['state_time'],
                    'reference_time':v.get('reference_time'),'active':v['active'],
                    'dose_reference':v.get('dose_reference',program.get('dose_reference','observed_native')),
                    'first_controlled_update':v.get('first_controlled_update'),
                    'initial_update_dose':v.get('initial_update_dose',program.get('initial_update_dose','native')),
                    'atom_step_cap_A':v.get('atom_step_cap_A'),
                    'raw_gradient_l2_native_mean':average('raw_gradient_l2_native'),
                    'post_projection_l2_native_mean':average('post_projection_l2_native'),
                    'projection_retained_squared_fraction_mean':average('projection_retained_squared_fraction'),
                    'backtrack_factor_mean':average('backtrack_factor'),
                    'max_actual_pair_distance_change_A_mean':average('max_actual_pair_distance_change_A'),
                    'dose_gate_mean':average('dose_gate'),
                    'background_support_gate_mean':average('background_support_gate'),
                    'contrast_amplitude_gate_mean':average('contrast_amplitude_gate'),
                    'bounded_response_gate_mean':average('bounded_response_gate'),
                    'observed_native_rms_A':average('observed_native_rms_A','native_rms_A'),
                    'predictive_flow_rms_A':average('predictive_flow_rms_A'),
                    'calibration_rms_A':average('calibration_rms_A','native_rms_A'),
                    'injection_rms_A':average('injection_rms_A') if v['active'] else 0.,
                    'requested_rms_A':average('requested_rms_A')})
            squared=np.asarray([v['injection_l2_A'] for v in active])**2 if active else np.zeros((0,1))
            audit.append({'arm':arm,'batch':batch,'active_steps':len(active),
                'reward_evaluated_steps':sum(v['reward_evaluated'] for v in trace),
                'nonzero_injection_steps':sum(max(v['injection_l2_A'])>0 for v in trace),
                'mean_path_rms_A':float(np.mean(active[-1]['cumulative_rms_A'])) if active else 0.,
                'guard_rejection_fraction':float(np.mean([np.logical_not(v['geometry_accepted']) for v in active])) if active else 0.,
                'guard_backtrack_fraction':float(np.mean([np.asarray(v['backtrack_factor'])<.99999 for v in active])) if active else 0.,
                'mean_guard_backtrack_factor':float(np.mean([v['backtrack_factor'] for v in active])) if active else 1.,
                'cap_fraction':float(np.mean([np.asarray(v['cap_factor'])<.99999 for v in active])) if active else 0.,
                'first_controlled_update_fraction_squared_injection':float(squared[0].sum()/squared.sum()) if squared.sum()>0 else None,
                'no_injection_after_window':all(max(v['injection_l2_A'])==0 for v in trace if v['state_time']>reference['window'][1]+1e-6)})
    table=write_table(out/'coordinate_time_metrics.csv',rows)
    for _,g in table.groupby(['arm','batch']):
        for f in ['mean_reward','mean_standardized_residual',*reference['features']]:
            table.loc[g.index,'d_dt_'+f]=np.gradient(g[f],g.time)
    write_table(out/'coordinate_time_rates.csv',table)
    write_table(out/'coordinate_injection_motion.csv',motion)
    write_table(out/'coordinate_dose_time.csv',dose_rows)
    preflight=read_json(root/'coordinate_gradient_preflight.json')
    if not preflight['passed']:raise ValueError('Coordinate derivative failed')
    report={'schema_version':'current-coordinate-evaluation-1.0','window':reference['window'],'batch_results':audit,
        'observable_unit':reference.get('feature_unit','A'),
        'coordinate_preflight':preflight,'inference_commit':cfg['extension']['code_commit'],
        'program_sha256':digest(root/'reward_program.json'),'reference_sha256':digest(root/'reference.json.gz'),
        'endpoint_at_window_end':table[np.isclose(table.state_time,reference['window'][1])].to_dict('records'),
        'time_alignment':reference.get('time_alignment','current state'),
        'interpretation':'Controlled response is a proxy; proposal at last score time may be outside the control window. Use independent actual-state whole-shape and terminal measures for conclusions'}
    write_json(out/'coordinate_audit.json',report);return report
