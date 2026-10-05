"""Local exact-time validation with an independent optimal atom assignment metric."""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from ..io import read_json, write_json, digest
from ..trajectory_source import open_trajectory


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
        # Main target must be the actual previous proposal after selection.
        if i>0:
            prev=z['proposal_coords'][i-1][z['selected_indices'][i-1]]
            np.testing.assert_array_equal(z['current_coords'][i],prev)
    return x,a,b


def distances(x,a,b,y,ya,yb):
    shape=np.empty((len(x),len(y)));typed=np.empty_like(shape);bond=np.empty_like(shape)
    for i in range(len(x)):
        for j in range(len(y)):
            cost=cdist(x[i],y[j],'sqeuclidean');r,c=linear_sum_assignment(cost)
            shape[i,j]=np.sqrt(cost[r,c].mean())
            # Auxiliary mismatch under geometry-optimal correspondence.
            typed[i,j]=(a[i,r]!=ya[j,c]).mean()
            edge=~np.eye(len(r),dtype=bool)
            bond[i,j]=(b[i][np.ix_(r,r)][edge]!=yb[j][np.ix_(c,c)][edge]).mean()
    return shape,typed,bond


def evaluate(dataset,campaign,original,original_campaign,output,reference_batches=range(14)):
    root=Path(dataset)/'results'/campaign;original=Path(original)/'results'/original_campaign
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    cfg=read_json(root/'config.json');p=read_json(root/'reward_program.json');a,end=p['window']
    references=[state(original,'single',i,end) for i in reference_batches]
    y,ya,yb=[np.concatenate([r[k] for r in references]) for k in range(3)]
    results={};batchrows=[]
    for arm in cfg['experiment']['arms'].split(','):
        allstates=[];summary=[];dose=[]
        for path in sorted((root/arm).glob('batch_*')):
            batch=int(path.name.split('_')[-1]);x,atoms,bonds=state(root,arm,batch,end)
            shape,typed,bond=distances(x,atoms,bonds,y,ya,yb)
            nearest=shape.argmin(1);ids=np.arange(len(x))
            # Bidirectional coverage measured against all discovery molecules, not reward medoids only.
            row={'arm':arm,'batch':batch,'n':len(x),'generated_to_reference_A':float(shape.min(1).mean()),
                 'reference_to_generated_A':float(shape.min(0).mean()),
                 'symmetric_shape_A':float((shape.min(1).mean()+shape.min(0).mean())/2),
                 'atom_mismatch_geometry_assignment':float(typed[ids,nearest].mean()),
                 'bond_mismatch_geometry_assignment':float(bond[ids,nearest].mean()),
                 'atom_fraction_L1':float(np.abs(np.bincount(atoms.ravel(),minlength=15)/atoms.size-np.bincount(ya.ravel(),minlength=15)/ya.size).sum()),
                 'bond_fraction_L1':float(np.abs(np.bincount(bonds[:,~np.eye(bonds.shape[1],dtype=bool)].ravel(),minlength=5)/ (len(bonds)*bonds.shape[1]*(bonds.shape[1]-1))-np.bincount(yb[:,~np.eye(yb.shape[1],dtype=bool)].ravel(),minlength=5)/(len(yb)*yb.shape[1]*(yb.shape[1]-1))).sum())}
            np.savez_compressed(out/f'{arm}_{batch:03d}_assignment_metrics.npz',shape_A=shape,typed_mismatch=typed,bond_mismatch=bond)
            trace=[json.loads(line) for line in (path/'guidance_trace.jsonl').read_text().splitlines()]
            eligible=[r for r in trace if r['reward_evaluated']]
            outside=[r for r in trace if r['score_time']<a-1e-6 or r['state_time']>end+1e-6]
            assert all(max(r['injection_l2_A'])==0 for r in outside)
            assert all(not r['particle_resampled'] for r in trace)
            if eligible:
                row.update(active_steps=sum(r['active'] for r in trace),
                  mean_injection_rms_A=float(np.mean([r['injection_rms_A'] for r in eligible])),
                  cap_fraction=float(np.mean(np.array([r['cap_factor'] for r in eligible])<.99999)),
                  path_rms_A=float(np.mean(eligible[-1]['cumulative_rms_A'])))
            else:row.update(active_steps=0,mean_injection_rms_A=0,cap_fraction=0,path_rms_A=0)
            batchrows.append(row);summary.append(row);allstates.append((x,atoms,bonds))
        results[arm]={key:float(np.mean([r[key] for r in summary])) for key in summary[0] if key not in ['arm','batch']}
    zero={}
    if {'unguided','gradient_zero'}.issubset(results):
        for path in sorted((root/'unguided').glob('batch_*')):
            with open_trajectory(path/'trajectory.h5') as native, open_trajectory(root/'gradient_zero'/path.name/'trajectory.h5') as z:
                checks={k:np.array_equal(native[k],z[k]) for k in ['current_coords','current_atomics','current_bonds','proposal_coords']}
                assert all(checks.values());zero[path.name]=checks
    report={'schema_version':'window-evaluation-1.0','campaign':campaign,'exact_time':end,'representation':'actual_current_world_A',
            'primary':'symmetric_shape_A, batch mean of bidirectional geometry-optimal assignment RMSD',
            'reference_batches':list(reference_batches),'reference_candidates':len(y),'results':results,
            'batch_results':batchrows,'zero_equivalence':zero,'program_sha256':digest(root/'reward_program.json'),
            'new_particle_resampling':False,'outside_window_injection':False,'analysis_location':'local'}
    write_json(out/'report.json',report);return report
