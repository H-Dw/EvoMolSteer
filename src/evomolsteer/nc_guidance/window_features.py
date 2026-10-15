"""Compact, full-window ancestor versus contemporaneous-state trajectories.

Descriptive whole-window contrasts, not independent time-point significance.
Extinct/unlabelled current states are controls, not declared terminal failures.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from evomolsteer.io import read_json,write_json,digest
from evomolsteer.storage.trajectory import TrajectoryPackage
from evomolsteer.continuous.elite_path_credit import terminal_ancestors


def fit_window(t,y):
    t=np.asarray(t,float);y=np.asarray(y,float)
    if len(t)<4 or not np.isfinite(y).all():raise ValueError('Incomplete full-window trend')
    results=[]
    for degree in (1,2):
        coef=np.polynomial.polynomial.polyfit(t,y,degree)
        pred=np.polynomial.polynomial.polyval(t,coef)
        derivative=np.polynomial.polynomial.polyder(coef)
        results.append({'degree':degree,'coefficients_ascending':coef.tolist(),'derivative_coefficients':derivative.tolist(),
            'rmse':float(np.sqrt(np.mean((pred-y)**2))),'domain':[float(t[0]),float(t[-1])],
            'net_rate':float((y[-1]-y[0])/(t[-1]-t[0])),'p_value':None,
            'meaning':'Descriptive complete-window curve; time points and cloned ancestors are not independent repetitions'})
    return results


def unique_state_controls(xyz,atom,parent):
    """Exclude the complete focal state, including its other cloned indices."""
    state=np.concatenate([np.asarray(xyz).reshape(len(xyz),-1),np.asarray(atom).reshape(len(xyz),-1)],axis=1)
    _,representatives,inverse=np.unique(state,axis=0,return_index=True,return_inverse=True)
    return representatives[inverse[representatives]!=inverse[parent]]


def build(repo,master,output_name='window_trends.json',curves_name='analysis/window_curves.parquet'):
    repo=Path(repo);study=repo/master['campaign_study'];out=study/output_name
    if out.exists():return read_json(out)
    templates=read_json(repo/master['prior_study']/'templates.json');seen=set();rows=[];fits=[];sources={}
    root=repo/master['compact_dataset']/'results'/master['campaign']
    for teacher in templates.values():
        key=teacher['candidate_id']
        if key in seen:continue
        seen.add(key);batch=teacher['batch'];slot=teacher['terminal_slot'];source=root/master['arm']/f'batch_{batch:03d}'/'trajectory.h5'
        com=np.asarray(read_json(root/f'frame_batch_{batch:03d}.json')['target_com']).reshape(-1,1,3)
        scale=read_json(root/'config.json')['coord_scale'];target=np.asarray(teacher['terminal_coords_A'])
        nsites=teacher['desired_N_sites'];centers=target[nsites];sources[str(source.relative_to(repo))]=digest(source)
        with TrajectoryPackage(source) as tr:
            times=tr.read('score_time')[:,0];anc=terminal_ancestors(tr.read('selected_indices').astype(int))
            for step,t in enumerate(times):
                if t<master['learning_window'][0]-2e-6 or t>master['learning_window'][1]+2e-6:continue
                xyz=tr.read('current_coords',step)*scale+com;atom=tr.read('current_atomics',step);parent=int(anc[step,slot])
                # Exact repeated states counted once in the descriptive comparison.
                control=unique_state_controls(xyz,atom,parent)
                if not len(control):continue
                distances=np.linalg.norm(xyz[:,:,None]-centers[None,None],axis=-1)
                mass=np.exp(-.5*(distances/.8)**2)*(atom==teacher['atom_vocabulary']['N'])[:,:,None]
                n_mass=mass.sum(1).mean(1)
                rg=np.sqrt(((xyz-xyz.mean(1,keepdims=True))**2).sum(-1).mean(1))
                rms=[]
                for cloud in xyz:
                    cost=((cloud[:,None]-target[None])**2).sum(-1);r,c=linear_sum_assignment(cost);rms.append(np.sqrt(cost[r,c].mean()))
                for name,values in [('native_N_site_mass',n_mass),('gyration_A',rg),('final_cloud_rms_A',np.asarray(rms))]:
                    rows.append({'candidate_id':key,'batch':batch,'time':float(t),'observable':name,
                        'ancestor_value':float(values[parent]),'unique_current_control_mean':float(values[control].mean()),
                        'contrast':float(values[parent]-values[control].mean()),'n_unique_controls':len(control)})
    frame=pd.DataFrame(rows)
    for (key,observable),g in frame.groupby(['candidate_id','observable']):
        fits.append({'candidate_id':key,'observable':observable,'n_clocks':len(g),
                     'contrast_fits':fit_window(g.time,g.contrast)})
    target=study/curves_name;target.parent.mkdir(parents=True,exist_ok=True);frame.to_parquet(target,index=False,compression=None)
    value={'domain':master['learning_window'],'sources':sources,'curves_sha256':digest(target),'fits':fits,
        'analysis_code_sha256':digest(__file__),
        'control_identity':'Unique coordinate-plus-category states, excluding all copies of the focal ancestor',
        'limits':['Candidate-conditioned retrospective comparison; no causal or generalizable high-affinity marker',
            'Contemporary controls do not have terminal failure labels','No sub-window significance bins',
            'Intermediate elements are native categorical states, not certified chemical roles']}
    write_json(out,value);return value
