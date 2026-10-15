"""Small, endpoint-anchored coordinate fits to actual historical ancestors."""
import numpy as np
from evomolsteer.io import read_json, write_json, digest
from evomolsteer.storage.trajectory import TrajectoryPackage
from evomolsteer.continuous.elite_path_credit import terminal_ancestors


def fit_path(times, coordinates, anchor, horizon, domain):
    times=np.asarray(times);x=np.asarray(coordinates);anchor=np.asarray(anchor)
    u=times-horizon;design=np.column_stack([u,u*u])
    coefficients=np.linalg.lstsq(design,(x-anchor).reshape(len(times),-1),rcond=None)[0]
    fitted=anchor+(design@coefficients).reshape(x.shape)
    # Interleaved prediction error is descriptive for one correlated path, not
    # an independent-path generalization estimate or a confidence interval.
    train=np.arange(len(times))%2==0
    cv=np.linalg.lstsq(design[train],(x[train]-anchor).reshape(train.sum(),-1),rcond=None)[0]
    predicted=anchor+(design[~train]@cv).reshape(x[~train].shape)
    return {'domain':domain,'horizon':horizon,'degree':2,'anchor_A':anchor.tolist(),
        'coefficients_A_per_t_power':coefficients.reshape(2,*anchor.shape).tolist(),
        'fit_rms_A':float(np.sqrt(np.square(fitted-x).sum(-1).mean())),
        'interleaved_prediction_rms_A':float(np.sqrt(np.square(predicted-x[~train]).sum(-1).mean())),
        'n_times':len(times),'interpretation':'Retrospective single-path coordinate approximation; no affinity model or independent-path validation.'}


def attach(repo, master, templates):
    folder=repo/master['campaign_study'];path=folder/'historical_path_targets_v1.json'
    if path.exists():profiles=read_json(path)
    else:
        profiles={};root=repo/master['compact_dataset']/'results'/master['campaign']
        scale=read_json(root/'config.json')['coord_scale']
        for tid,t in templates.items():
            b=t['batch'];source=root/master['arm']/f'batch_{b:03d}'/'trajectory.h5'
            com=np.asarray(read_json(root/f'frame_batch_{b:03d}.json')['target_com']).reshape(-1,1,3)
            with TrajectoryPackage(source) as tr:
                clocks=tr.read('score_time')[:,0];anc=terminal_ancestors(tr.read('selected_indices').astype(int))
                ids=np.flatnonzero((clocks>=master['learning_window'][0]-2e-6)&(clocks<=t['time']+2e-6))
                points=np.array([(tr.read('current_coords',int(i))*scale+com)[int(anc[i,t['terminal_slot']])] for i in ids])
                profile=fit_path(clocks[ids],points,np.asarray(t['coords_A']),t['time'],master['learning_window'])
                profile.update(source_sha256=digest(source),candidate_id=t['candidate_id'])
                profiles[tid]=profile
        write_json(path,profiles)
    for tid,t in templates.items():t['historical_path']=profiles[tid]
    return templates


def target_at(profile, clock, slots):
    if not profile['domain'][0]-2e-6<=clock<=profile['horizon']+2e-6:raise ValueError('Path target extrapolation is forbidden')
    u=clock-profile['horizon'];coeff=np.asarray(profile['coefficients_A_per_t_power'])
    return (np.asarray(profile['anchor_A'])+u*coeff[0]+u*u*coeff[1])[slots]
