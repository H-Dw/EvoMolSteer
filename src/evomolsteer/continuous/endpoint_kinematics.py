"""Success-associated endpoint feature changes along exact Steer parent edges.

All coordinates remain in the true selection support. Children of a parent are
averaged before batch statistics; batches, not offspring, are inferential units.
The increment target is observational, not a causal or fitted affinity model.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from ..io import read_json,write_json,digest
from ..storage.trajectory import TrajectoryPackage
from ..generation.window_reference import load_reference
from .affinity_geometry import numpy_geometry
from .coordinate_mining import bh


def parent_mean(values,parents,subset):
    selected=np.flatnonzero(subset)
    if len(selected)<3:raise ValueError('Insufficient score stratum')
    groups=np.unique(parents[selected])
    return np.array([values[selected[parents[selected]==p]].mean(0) for p in groups]).mean(0),len(groups)


def mine(reference,dataset,campaign,output):
    ref=load_reference(reference);root=Path(dataset);source=root/'results'/campaign;out=Path(output)
    if out.exists():raise FileExistsError(out)
    cfg=read_json(source/'config.json');times=np.asarray(ref['times']);scale=np.asarray(ref['feature_scale']);points=np.asarray(ref['landmarks_A'])
    high=[];low=[];counts=[]
    for item in ref['sources']:
        path=root/item['path']
        if digest(path)!=item['sha256']:raise ValueError('Immutable Steer source mismatch')
        batch=int(path.parent.name.split('_')[-1]);com=np.asarray(read_json(source/f'frame_batch_{batch:03d}.json')['target_com'])[:,None]
        h=[];l=[];previous=None
        with TrajectoryPackage(path) as tr:
            grid=np.round(tr.read('score_time')[:,0].astype(float),6);states=tr.read('state_time')
            ids=np.flatnonzero(tr.read('resampled')&(grid>=ref['window'][0]-1e-6)&(states<=ref['window'][1]+1e-6))
            if not np.array_equal(grid[ids],times) or np.any(np.diff(ids)!=1):raise ValueError('Exact contiguous selection grid required')
            for k,i in enumerate(ids):
                y=tr.read('predicted_coords',int(i)).astype(float)*cfg['coord_scale']+com
                phi=numpy_geometry(y,points,ref['origin_A'])/scale
                if k==0:h.append(np.zeros(len(scale)));l.append(np.zeros(len(scale)));previous=phi;continue
                parents=tr.read('selected_indices',int(ids[k-1])).astype(int)
                if parents.shape!=(len(phi),) or np.any((parents<0)|(parents>=len(previous))):raise ValueError('Invalid parent edge')
                delta=phi-previous[parents];score=tr.read('pic50_on',int(i)).reshape(-1)
                q0,q1=np.quantile(score,[.25,.75]);hh,nh=parent_mean(delta,parents,score>=q1);ll,nl=parent_mean(delta,parents,score<=q0)
                h.append(hh);l.append(ll);counts.append({'batch':batch,'score_time':float(times[k]),'unique_high_parents':nh,'unique_low_parents':nl})
                previous=phi
        high.append(h);low.append(l)
    high=np.asarray(high);low=np.asarray(low)
    # These are increments on actual edges, not point samples of a continuous
    # rate. Sum observed edges over elapsed time; never integrate the unobserved
    # first-node placeholder or halve the first/last edge with a trapezoid rule.
    integrated=(high[:,1:]-low[:,1:]).sum(1)/(times[-1]-times[0])
    rng=np.random.default_rng(42);boot=integrated[rng.integers(0,len(high),(2000,len(high)))].mean(1)
    p=np.nan_to_num(ttest_1samp(integrated,0,axis=0).pvalue,nan=1.);q=bh(p);rows=[]
    for j,f in enumerate(ref['features']):
        ci=np.quantile(boot[:,j],[.025,.975]);rows.append({'feature':f,'success_failure_integrated_rate_z':float(integrated[:,j].mean()),
            'CI_low':float(ci[0]),'CI_high':float(ci[1]),'positive_batch_fraction':float((integrated[:,j]>0).mean()),'p':float(p[j]),'q':float(q[j])})
    history_weights=[int(r['q']<.05 and abs(r['success_failure_integrated_rate_z'])>=.1
                         and max(r['positive_batch_fraction'],1-r['positive_batch_fraction'])>=12/14 and scale[j]>1e-5)
                     for j,r in enumerate(rows)]
    result={'schema_version':'endpoint-lineage-kinematics-1.0','reference_sha256':digest(reference),'window':ref['window'],
            'times':ref['times'],'features':ref['features'],'node_mean_success_delta_z':high.mean(0).tolist(),
            'node_mean_failure_delta_z':low.mean(0).tolist(),'sources':ref['sources'],'independent_batches':len(high),
            'history_feature_weights':history_weights,
            'history_support_criteria':{'q_max':.05,'minimum_absolute_integrated_rate_z':.1,'minimum_direction_agreement':12/14,'scale_floor_excluded':True},
            'first_node_observed':False,'statistics_unit':'Equal batches; offspring averaged within each extant parent and score stratum',
            'whole_window_rate_definition':'Sum of observed success-minus-failure edge increments divided by elapsed time; first-node placeholder excluded',
            'target_semantics':'Recorded high-score child endpoint feature minus its selected parent forecast. First node has no edge and an explicit zero placeholder.',
            'limitations':['Joint latent-head labels, not measured affinity','Conditioning on survival and score is observational',
                           'Forecast changes differ from physical atom velocities; feature slots are geometric observables, not atom types']}
    out.mkdir(parents=True);write_json(out/'kinematics_prior.json',result)
    pd.DataFrame(rows).to_parquet(out/'whole_window_velocity_effects.parquet',compression=None,index=False)
    pd.DataFrame(counts).to_parquet(out/'parent_support.parquet',compression=None,index=False)
    write_json(out/'manifest.json',{'reference_sha256':digest(reference),'prior_sha256':digest(out/'kinematics_prior.json'),
        'source_code_sha256':digest(__file__),'q_below_005':int((q<.05).sum()),'feature_count':len(scale),
        'storage':'Mean node increments and whole-window effect summaries; no coordinate or particle feature cache'})
    return result
