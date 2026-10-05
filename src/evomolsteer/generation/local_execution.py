"""Audit complete native continuation and track regional effects at every time."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from ..io import read_json,write_json,digest
from ..trajectory_source import open_trajectory
from .prototypes import measure_patch
from .window_reference import load_reference


def audit(dataset,campaign,reference_path,output):
    root=Path(dataset)/'results'/campaign;out=Path(output);out.mkdir(parents=True,exist_ok=True)
    cfg=read_json(root/'config.json');ref=load_reference(reference_path)
    a,b=ref['window'];scale=cfg['coord_scale'];steps=cfg['experiment']['steps']
    if cfg['extension'].get('control_domain',ref['window'])!=ref['window']:raise ValueError('Control/diagnostic window mismatch')
    noncore_definition='outside bare 5 A endpoint geometric region; unweighted and injection-weighted summaries are distinct'
    if cfg['extension'].get('schema_version')=='current-coordinate-control-1.0':
        control_ref=load_reference(root/'reference.json.gz')
        noncore_definition='global uniform control: all eligible atom slots are core' if control_ref.get('spatial_weighting')=='uniform_global_control' else (
            'outside bare 5 A region in '+control_ref.get('spatial_anchor','current')+' geometry; fractions weight squared actual injection')
    if cfg['experiment']['seed']!=42 or steps!=100 or not (root/'COMPLETE.json').exists():raise ValueError('Seed/full-inference contract')
    rows=[];batchrows=[];signatures={};sources=[]
    for arm in cfg['experiment']['arms'].split(','):
        for batch in cfg['experiment'].get('batch_indices') or range(cfg['experiment']['n']//cfg['experiment']['batch']):
            folder=root/arm/f'batch_{batch:03d}'
            trace=[json.loads(v) for v in (folder/'guidance_trace.jsonl').read_text().splitlines()]
            if [v['step'] for v in trace]!=list(range(steps)):raise ValueError('Incomplete inference telemetry')
            if any(v['particle_resampled'] for v in trace):raise ValueError('Particle selection occurred')
            outside=[v for v in trace if v['score_time']<a-1e-6 or v['state_time']>b+1e-6]
            if any(max(v['injection_l2_A'])>0 for v in outside):raise ValueError('Control outside learned window')
            com=np.asarray(read_json(root/f'frame_batch_{batch:03d}.json')['target_com'])
            with open_trajectory(folder/'trajectory.h5') as z:
                n=z['current_coords'].shape[1]
                if n!=cfg['experiment']['batch'] or len(z['score_time'])!=100:raise ValueError('Missing candidates/steps')
                if np.any(z['resampled']) or not np.array_equal(z['selected_indices'],np.tile(np.arange(n),(100,1))):raise ValueError('Selection not disabled')
                signature=hashlib.sha256()
                for field in ['current_coords','current_atomics','current_bonds']:
                    signature.update(z[field][0].tobytes())
                signatures[arm+'/'+str(batch)]=signature.hexdigest()
                t=np.round(z['score_time'][:,0].astype(float),6)
                x=z['predicted_coords'].astype(float)*scale+com[None,:,None,:]
                atoms=z['predicted_atomics'];mask=z['mask']
                values=measure_patch(x.reshape(-1,x.shape[2],3),atoms.reshape(-1,atoms.shape[2]),mask.reshape(-1,mask.shape[2]),ref['catalog']).reshape(100,n,-1)
            for i,time in enumerate(t):
                reference_time=min(float(time),b);j=int(np.abs(np.array(ref['times'])-reference_time).argmin());f=ref['frames'][j]
                valid=np.isfinite(values[i]).all(1);v=values[i,valid];delta=v-np.asarray(f['center_A'])
                q=np.einsum('bi,ij,bj->b',delta,np.linalg.inv(f['covariance_A2']),delta)/f['radius_squared']
                rows.append({'arm':arm,'batch':batch,'time':float(time),'reference_time':reference_time,'n_available':int(valid.sum()),
                             'mean_deficit':float(np.maximum(np.sqrt(q)-1,0).mean()) if len(q) else None,
                             'inside_fraction_available':float((q<=1).mean()) if len(q) else None,
                             **{name:float(np.nanmean(values[i,:,k])) for k,name in enumerate(ref['features'])}})
            available=[v for v in trace if v['reward_evaluated']]
            injected=[v for v in trace if max(v['injection_l2_A'])>0]
            if available and 'injection_noncore_fraction' in available[0]:
                fraction=np.array([v['injection_noncore_fraction'] for v in available])
                squared=np.square([v['injection_l2_A'] for v in available])
                weighted_noncore=float((fraction*squared).sum()/squared.sum()) if squared.sum()>0 else None
                conditional_noncore=float(fraction[squared>1e-24].mean()) if (squared>1e-24).any() else None
            else:weighted_noncore=conditional_noncore=None
            batchrows.append({'arm':arm,'batch':batch,'n':n,'inference_steps':100,'reward_evaluated_steps':len(available),
                'nonzero_steps':len(injected),'outside_injection':False,
                'mean_cumulative_injection_rms_A':float(np.mean(available[-1]['cumulative_rms_A'])) if available else 0,
                'mean_noncore_injection_fraction':float(np.mean([v['injection_noncore_fraction'] for v in available])) if available and 'injection_noncore_fraction' in available[0] else None,
                'noncore_fraction_of_total_squared_injection':weighted_noncore,
                'noncore_fraction_mean_nonzero_events':conditional_noncore,
                'noncore_definition':noncore_definition,
                'mean_inside_fraction':float(np.mean([v['inside'] for v in available])) if available and 'inside' in available[0] else None})
            sources.append({'path':str(folder/'trajectory.h5'),'sha256':digest(folder/'trajectory.h5')})
    d=pd.DataFrame(rows);d.to_csv(out/'regional_time_metrics.csv',index=False)
    window=d[(d.time>=a-1e-6)&(d.time<=b+1e-6)].copy()
    for _,group in window.groupby(['arm','batch']):
        for name in ['mean_deficit','inside_fraction_available',*ref['features']]:
            window.loc[group.index,'d_dt_'+name]=np.gradient(group[name],group.time)
    window.to_csv(out/'regional_window_trends.csv',index=False)
    report={'campaign':campaign,'seed':42,'code_commit':cfg['extension']['code_commit'],'reference_sha256':digest(reference_path),
        'program_sha256':digest(root/'reward_program.json'),'window':ref['window'],'integration_steps':100,
        'no_particle_resampling':True,'outside_window_injection':False,
        'initial_state_signatures':signatures,'batch_results':batchrows,'sources':sources,
        'endpoint_diagnostics_at_window_end':d[np.isclose(d.time,b)].to_dict('records'),
        'late_reference_interpretation':'Frozen window-end acceptable set is used only to observe native continuation; no late intervention'}
    if (root/'live_jacobian_preflight.json').exists():
        report['live_jacobian']=read_json(root/'live_jacobian_preflight.json')
        if not report['live_jacobian']['passed']:raise ValueError('Model directional check failed')
    if cfg['extension'].get('schema_version')=='regional-window-control-1.0' and any(v['nonzero_steps'] for v in batchrows):
        if 'live_jacobian' not in report:raise ValueError('Missing actual-model directional check')
    write_json(out/'execution_report.json',report);return report


def original_regional(dataset,campaign,reference_path,batches,output):
    """Original selection-window observations, exposing candidate and copy mass."""
    root=Path(dataset)/'results'/campaign;out=Path(output);out.mkdir(parents=True,exist_ok=True)
    cfg=read_json(root/'config.json');ref=load_reference(reference_path);a,b=ref['window']
    rows=[];sources=[]
    for arm in ['single','unguided']:
        for batch in batches:
            folder=root/arm/f'batch_{batch:03d}';com=np.asarray(read_json(root/f'frame_batch_{batch:03d}.json')['target_com'])
            with open_trajectory(folder/'trajectory.h5') as z:
                times=np.round(z['score_time'][:,0].astype(float),6);ids=np.flatnonzero((times>=a-1e-6)&(times<=b+1e-6))
                x=z['predicted_coords'][ids].astype(float)*cfg['coord_scale']+com[None,:,None,:]
                atoms=z['predicted_atomics'][ids];mask=z['mask'][ids]
                values=measure_patch(x.reshape(-1,x.shape[2],3),atoms.reshape(-1,atoms.shape[2]),mask.reshape(-1,mask.shape[2]),ref['catalog']).reshape(len(ids),len(com),-1)
                weights=z['selection_probability'][ids];copies=z['offspring_count'][ids]
            for i,step in enumerate(ids):
                j=int(np.abs(np.array(ref['times'])-times[step]).argmin());f=ref['frames'][j]
                valid=np.isfinite(values[i]).all(1);delta=values[i,valid]-f['center_A']
                q=np.einsum('bi,ij,bj->b',delta,np.linalg.inv(f['covariance_A2']),delta)/f['radius_squared']
                deficit=np.maximum(np.sqrt(q)-1,0)
                for kind,w in [('candidate',np.ones(valid.sum())),('selection_probability',weights[i,valid]),('realized_copy_mass',copies[i,valid])]:
                    rows.append({'arm':arm,'batch':batch,'time':float(times[step]),'mass':kind,'n_available':int(valid.sum()),
                                 'mean_deficit':float(np.average(deficit,weights=w)) if w.sum()>0 else None,
                                 'inside_fraction_available':float(np.average(q<=1,weights=w)) if w.sum()>0 else None})
            sources.append({'path':str(folder/'trajectory.h5'),'sha256':digest(folder/'trajectory.h5')})
    d=pd.DataFrame(rows);d.to_csv(out/'original_regional_window.csv',index=False)
    write_json(out/'original_regional_report.json',{'window':ref['window'],'sources':sources,'reference_sha256':digest(reference_path),
        'endpoint_diagnostics_at_window_end':d[np.isclose(d.time,b)].to_dict('records'),
        'interpretation':'Selected probability and realized copies condition on available regional observables; original SMC mass is not independent unique molecules.'})
