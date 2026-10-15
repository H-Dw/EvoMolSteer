"""Continuous, pocket-fixed regional occupancy contrasts over the full window."""
import numpy as np
import pandas as pd
from evomolsteer.io import read_json,write_json,digest
from evomolsteer.storage.trajectory import TrajectoryPackage
from evomolsteer.continuous.elite_path_credit import terminal_ancestors
from evomolsteer.nc_guidance.window_features import unique_state_controls,fit_window


def field(clouds,centers,width=1.5):
    d=clouds[:,:,None]-centers[None,None]
    mass=np.exp(-.5*np.square(d).sum(-1)/width**2)
    den=mass.sum((1,2)).clip(1e-10)
    displacement=(mass[...,None]*d).sum((1,2))/den[:,None]
    return np.column_stack([mass.mean((1,2)),displacement])


def build(repo,master,regions,templates,output_name='trajectory_evidence.json',curves_name='regional_curves.parquet'):
    out=repo/master['campaign_study'];path=out/output_name
    if path.exists():return read_json(path)
    root=repo/master['compact_dataset']/'results'/master['campaign'];scale=read_json(root/'config.json')['coord_scale']
    rows=[];fits={};sources={}
    for candidate,rs in pd.DataFrame([{'candidate':templates[r['template_id']]['candidate_id'],'region':r} for r in regions]).groupby('candidate'):
        group=rs.region.tolist();teacher=templates[group[0]['template_id']];batch=teacher['batch']
        source=root/master['arm']/f'batch_{batch:03d}'/'trajectory.h5'
        com=np.asarray(read_json(root/f'frame_batch_{batch:03d}.json')['target_com']).reshape(-1,1,3)
        sources[str(source.relative_to(repo))]=digest(source)
        with TrajectoryPackage(source) as tr:
            times=tr.read('score_time')[:,0];anc=terminal_ancestors(tr.read('selected_indices').astype(int))
            for step,t in enumerate(times):
                if not master['learning_window'][0]-2e-6<=t<=master['learning_window'][1]+2e-6:continue
                x=tr.read('current_coords',step)*scale+com;a=tr.read('current_atomics',step)
                parent=int(anc[step,teacher['terminal_slot']]);control=unique_state_controls(x,a,parent)
                for r in group:
                    centers=np.asarray(teacher['terminal_coords_A'])[r['atom_slots']]
                    values=field(x,centers)
                    for j,name in enumerate(['spatial_mass','signed_dx_A','signed_dy_A','signed_dz_A']):
                        rows.append({'region_id':r['region_id'],'time':float(t),'observable':name,'ancestor':float(values[parent,j]),
                            'control_mean':float(values[control,j].mean()) if len(control) else None,
                            'contrast':float(values[parent,j]-values[control,j].mean()) if len(control) else None,'n_controls':len(control)})
    frame=pd.DataFrame(rows);frame.to_parquet(out/curves_name,compression=None,index=False)
    for (rid,name),g in frame.groupby(['region_id','observable']):
        if len(g)>=4 and g.contrast.notna().all():
            fits.setdefault(rid,{})[name]={'ancestor_fit':fit_window(g.time,g.ancestor),'contrast_fit':fit_window(g.time,g.contrast)}
    value={'fits':fits,'domain':master['learning_window'],'width_A':1.5,'sources':sources,
           'limits':'Retrospective pocket-fixed fields; no cross-molecule atom-slot correspondence. No time-bin tests, clone pseudoreplication, terminal failure labels, or spatial-gradient interpretation of temporal slopes.'}
    write_json(path,value);return value
