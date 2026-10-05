"""Local exact-time validation with an independent optimal atom assignment metric."""
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from ..io import read_json, write_json, digest
from ..trajectory_source import open_trajectory


def continuous_diagnostics(root,arm,batch,start,end):
    """All observed window nodes, no arbitrary subinterval binning."""
    cfg=read_json(root/'config.json');scale=cfg['coord_scale']
    com=np.asarray(read_json(root/f'frame_batch_{batch:03d}.json')['target_com'])
    with open_trajectory(root/arm/f'batch_{batch:03d}'/'trajectory.h5') as z:
        times=z['score_time'][:,0].astype(float);ids=np.flatnonzero((times>=start-1e-6)&(times<=end+1e-6))
        x=z['current_coords'][ids].astype(float)*scale+com[None,:,None,:]
        atoms=z['current_atomics'][ids];bonds=z['current_bonds'][ids]
    center=x.mean(2);centered=x-center[:,:,None,:]
    features={f'center_{name}_A':center[...,k] for k,name in enumerate('xyz')}
    features['radius_gyration_A']=np.sqrt((centered**2).sum(-1).mean(-1))
    dist=np.linalg.norm(x[:,:,:,None,:]-x[:,:,None,:,:],axis=-1)
    edge=~np.eye(x.shape[2],dtype=bool)
    for symbol,index in cfg['atom_vocabulary'].items():features[f'atom_fraction_{symbol}']=(atoms==index).mean(-1)
    for k in range(1,5):
        w=(bonds==k)&edge[None,None]
        features[f'bond_fraction_{k}']=(bonds[:,:,edge]==k).mean(-1)
        count=w.sum((2,3));numerator=(dist*w).sum((2,3))
        features[f'bond_length_{k}_A']=np.divide(numerator,count,out=np.full_like(numerator,np.nan),where=count>0)
    rows=[]
    for name,value in features.items():
        available=np.isfinite(value).sum(1)
        mean=np.divide(np.nansum(value,1),available,out=np.full(len(value),np.nan),where=available>0)
        rate=np.gradient(mean,times[ids])
        for j,step in enumerate(ids):
            rows.append({'arm':arm,'batch':batch,'time':float(times[step]),'feature':name,'mean':mean[j],
                         'dmean_dt':rate[j],'available_candidates':int(available[j]),'candidate_count':len(value[j])})
    return rows


def state(root,arm,batch,time):
    cfg=read_json(root/'config.json');scale=cfg['coord_scale']
    com=np.asarray(read_json(root/f'frame_batch_{batch:03d}.json')['target_com'])
    path=root/arm/f'batch_{batch:03d}'/'trajectory.h5'
    with open_trajectory(path) as z:
        times=z['score_time'][:,0].astype(float)
        i=int(np.argmin(abs(times-time)))
        if abs(times[i]-time)>2e-6:raise ValueError('Exact current time missing')
        x=z['current_coords'][i].astype(float)*scale+com[:,None,:]
        a=z['current_atomics'][i];b=z['current_bonds'][i]
        if not z['mask'][i].all():raise ValueError('Unequal active count')
        if arm!='single':
            if z['resampled'].any():raise ValueError('Forbidden particle resampling')
            np.testing.assert_array_equal(z['selected_indices'],np.tile(np.arange(len(x)),(len(times),1)))
            snapshot=path.parent/'window_state.npz'
            with np.load(snapshot,allow_pickle=False) as saved:
                if abs(float(saved['state_time'])-time)>2e-6:raise ValueError('Window snapshot time mismatch')
                np.testing.assert_array_equal(z['current_coords'][i],saved['coords'])
                np.testing.assert_array_equal(a,saved['atomics'])
                np.testing.assert_array_equal(b,saved['bonds'])
        # Main target must be the actual previous proposal after selection.
        if i>0:
            prev=z['proposal_coords'][i-1][z['selected_indices'][i-1]]
            np.testing.assert_array_equal(z['current_coords'][i],prev)
    return x,a,b


def distances(x,a,b,y,ya,yb):
    shape=np.empty((len(x),len(y)));typed=np.empty_like(shape);bond=np.empty_like(shape);bond_union=np.empty_like(shape)
    for i in range(len(x)):
        for j in range(len(y)):
            cost=cdist(x[i],y[j],'sqeuclidean');r,c=linear_sum_assignment(cost)
            shape[i,j]=np.sqrt(cost[r,c].mean())
            # Auxiliary mismatch under geometry-optimal correspondence.
            typed[i,j]=(a[i,r]!=ya[j,c]).mean()
            edge=~np.eye(len(r),dtype=bool)
            left=b[i][np.ix_(r,r)][edge];right=yb[j][np.ix_(c,c)][edge]
            bond[i,j]=(left!=right).mean();union=(left>0)|(right>0)
            bond_union[i,j]=(left[union]!=right[union]).mean() if union.any() else 0.
    return shape,typed,bond,bond_union


def validate_campaign(root):
    cfg=read_json(root/'config.json');opt=cfg['experiment'];completion=read_json(root/'COMPLETE.json')
    arms=opt['arms'].split(',');expected=opt['n']//opt['batch']
    expected_indices=opt.get('batch_indices') or list(range(expected))
    if completion.get('status')!='complete' or completion['records']!=opt['n']*len(arms):raise ValueError('Incomplete campaign')
    for arm in arms:
        paths=sorted((root/arm).glob('batch_*'))
        if [p.name for p in paths]!=[f'batch_{i:03d}' for i in expected_indices]:raise ValueError('Missing/unexpected candidate batch')
        for p in paths:
            if read_json(p/'COMPLETE.json')['n']!=opt['batch']:raise ValueError('Incomplete candidate batch')
            if not (p/'window_state.npz').is_file():raise ValueError('Missing window snapshot')
    return cfg


def evaluate(dataset,campaign,original,original_campaign,output,reference_batches=range(14),save_assignment_details=False):
    root=Path(dataset)/'results'/campaign;original=Path(original)/'results'/original_campaign
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    cfg=validate_campaign(root);p=read_json(root/'reward_program.json');a,end=p['window']
    references=[state(original,'single',i,end) for i in reference_batches]
    y,ya,yb=[np.concatenate([r[k] for r in references]) for k in range(3)]
    results={};batchrows=[];trends=[];initial_signatures={};trajectory_hashes={}
    for arm in cfg['experiment']['arms'].split(','):
        allstates=[];summary=[];dose=[]
        for path in sorted((root/arm).glob('batch_*')):
            batch=int(path.name.split('_')[-1]);x,atoms,bonds=state(root,arm,batch,end)
            with open_trajectory(path/'trajectory.h5') as z:
                h=hashlib.sha256()
                for key in ['current_coords','current_atomics','current_bonds','current_charges','mask']:
                    value=np.ascontiguousarray(z[key][0]);h.update(str((key,value.shape,value.dtype)).encode());h.update(value.tobytes())
                initial_signatures[f'{arm}/{batch}']=h.hexdigest()
            trajectory_hashes[f'{arm}/{batch}']=digest(path/'trajectory.h5')
            trends.extend(continuous_diagnostics(root,arm,batch,a,end))
            if len(x)!=cfg['experiment']['batch']:raise ValueError('Candidate count mismatch')
            shape,typed,bond,bond_union=distances(x,atoms,bonds,y,ya,yb)
            nearest=shape.argmin(1);ids=np.arange(len(x))
            # Bidirectional coverage measured against all discovery molecules, not reward medoids only.
            row={'arm':arm,'batch':batch,'n':len(x),'generated_to_reference_A':float(shape.min(1).mean()),
                 'reference_to_generated_A':float(shape.min(0).mean()),
                 'symmetric_shape_A':float((shape.min(1).mean()+shape.min(0).mean())/2),
                 'atom_mismatch_geometry_assignment':float(typed[ids,nearest].mean()),
                 'bond_mismatch_geometry_assignment':float(bond[ids,nearest].mean()),
                 'bonded_union_mismatch':float(bond_union[ids,nearest].mean()),
                 'atom_fraction_L1':float(np.abs(np.bincount(atoms.ravel(),minlength=15)/atoms.size-np.bincount(ya.ravel(),minlength=15)/ya.size).sum()),
                 'bond_fraction_L1':float(np.abs(np.bincount(bonds[:,~np.eye(bonds.shape[1],dtype=bool)].ravel(),minlength=5)/ (len(bonds)*bonds.shape[1]*(bonds.shape[1]-1))-np.bincount(yb[:,~np.eye(yb.shape[1],dtype=bool)].ravel(),minlength=5)/(len(yb)*yb.shape[1]*(yb.shape[1]-1))).sum())}
            if save_assignment_details:
                np.savez_compressed(out/f'{arm}_{batch:03d}_assignment_metrics.npz',shape_A=shape,typed_mismatch=typed,bond_mismatch=bond,bonded_union_mismatch=bond_union)
            trace=[json.loads(line) for line in (path/'guidance_trace.jsonl').read_text().splitlines()]
            if [r['step'] for r in trace]!=list(range(cfg['experiment']['steps'])):raise ValueError('Incomplete guidance trace')
            eligible=[r for r in trace if r['reward_evaluated']]
            outside=[r for r in trace if r['score_time']<a-1e-6 or r['state_time']>end+1e-6]
            assert all(max(r['injection_l2_A'])==0 for r in outside)
            assert all(not r['particle_resampled'] for r in trace)
            if eligible:
                row.update(active_steps=sum(r['active'] for r in trace),
                  mean_injection_rms_A=float(np.mean([r['injection_rms_A'] for r in eligible])),
                  cap_fraction=float(np.mean(np.array([r['cap_factor'] for r in eligible])<.99999)),
                  nonzero_control_fraction=float(np.mean(np.array([r['injection_l2_A'] for r in eligible])>0)),
                  gradient_below_threshold_fraction=float(np.mean(np.array([r['gradient_rms_native'] for r in eligible])<=1e-8)),
                  backtrack_rejected_fraction=float(np.mean(~np.array([r['geometry_accepted'] for r in eligible],dtype=bool))),
                  path_rms_A=float(np.mean(eligible[-1]['cumulative_rms_A'])))
            else:row.update(active_steps=0,mean_injection_rms_A=0,cap_fraction=0,path_rms_A=0,
                            nonzero_control_fraction=0,gradient_below_threshold_fraction=0,backtrack_rejected_fraction=0)
            batchrows.append(row);summary.append(row);allstates.append((x,atoms,bonds))
        results[arm]={key:float(np.mean([r[key] for r in summary])) for key in summary[0] if key not in ['arm','batch']}
        results[arm]['n']=sum(r['n'] for r in summary)
        results[arm]['batches']=len(summary)
    zero={}
    if {'unguided','gradient_zero'}.issubset(results):
        from ..storage.arrays import byte_equal
        from .zero_equivalence import compare_final
        for path in sorted((root/'unguided').glob('batch_*')):
            with open_trajectory(path/'trajectory.h5') as native, open_trajectory(root/'gradient_zero'/path.name/'trajectory.h5') as z:
                fields=['current_coords','current_atomics','current_bonds','current_charges','mask',
                        'proposal_coords','proposal_atomics','proposal_bonds','proposal_charges',
                        'predicted_coords','predicted_atomics','predicted_bonds','predicted_charges',
                        'pic50_on','pic50_off','score_time','state_time']
                fields += [k for k in ['current_hybridization','proposal_hybridization'] if k in native.files]
                checks={k:byte_equal(native[k],z[k]) for k in fields}
                mismatch_steps={k:np.flatnonzero(np.any(native[k]!=z[k],axis=tuple(range(1,native[k].ndim)))).tolist()
                                for k in fields if not checks[k]}
                zero[path.name]={'byte_equal':checks,'passed':all(checks.values()),
                    'mismatched_steps':mismatch_steps,
                    'current_coordinate_rms_model_units':float(np.sqrt(np.mean(np.square(native['current_coords'].astype(float)-z['current_coords'].astype(float))))),
                    'atom_label_disagreement':float((native['current_atomics']!=z['current_atomics']).mean())}
            final=compare_final(path/'final_prediction.pt.gz',root/'gradient_zero'/path.name/'final_prediction.pt.gz')
            zero[path.name]['final_prediction']=final
            zero[path.name]['passed']=zero[path.name]['passed'] and final['passed']
    report={'schema_version':'window-evaluation-1.0','campaign':campaign,'exact_time':end,'representation':'actual_current_world_A',
            'primary':'symmetric_shape_A, batch mean of bidirectional geometry-optimal assignment RMSD',
            'reference_batches':list(reference_batches),'reference_candidates':len(y),'results':results,
            'batch_results':batchrows,'zero_equivalence':zero,'program_sha256':digest(root/'reward_program.json'),
            'zero_equivalence_passed':all(v['passed'] for v in zero.values()) if zero else None,
            'initial_state_signatures':initial_signatures,'trajectory_sha256':trajectory_hashes,
            'code_commit':cfg['extension']['code_commit'],'seed':cfg['experiment']['seed'],
            'analysis_implementation_sha256':digest(__file__),'numpy':np.__version__,'scipy':scipy.__version__,
            'new_particle_resampling':False,'outside_window_injection':False,'analysis_location':'local',
            'assignment_detail_saved':save_assignment_details}
    pd.DataFrame(trends).to_parquet(out/'all_window_node_descriptors_and_rates.parquet',index=False,compression='zstd')
    write_json(out/'report.json',report);return report
