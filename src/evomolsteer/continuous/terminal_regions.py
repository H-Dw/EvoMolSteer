"""Terminal-ranked teachers and compact, ancestry-correct regional evidence.

Coordinates are forecasts at real score clocks. Final decoded scores label their
observed ancestors; extinction is censored, never a synthetic negative label.
"""
import copy
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, ttest_1samp

from ..io import clean, digest, read_json, write_json
from ..trajectory_source import open_trajectory
from ..generation.window_reference import load_reference
from .batch_split import source_batches, split_batches
from .coordinate_mining import bh
from .elite_path_credit import terminal_ancestors
from .path_graph import validate_lineage


def select_terminal_elites(metrics, config):
    k = config['top_k']
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError('Configured positive top_k required')
    if metrics.valid_connected.dtype != bool or metrics.duplicated(['batch', 'slot']).any():
        raise ValueError('Boolean terminal validity and unique final locators required')
    eligible = metrics[metrics.valid_connected & np.isfinite(metrics.pic50_on_rescore)].copy()
    eligible = eligible.sort_values(['pic50_on_rescore', 'batch', 'slot'], ascending=[False, True, True], kind='stable')
    if config.get('deduplicate_smiles', False):
        eligible = eligible.drop_duplicates('smiles')
    chosen = eligible.head(k).copy()
    if len(chosen) != k:
        raise ValueError('Fewer eligible terminal molecules than configured top_k')
    chosen['path_id'] = [f'{int(b)}:{int(s)}' for b, s in chosen[['batch', 'slot']].to_numpy()]
    return chosen


def regional_features(x, points, radius):
    """Untyped Gaussian mass, local first moments and six covariance entries."""
    displacement = x[:, :, None, :] - points[None, None, :, :]
    mass = np.exp(-np.square(displacement).sum(-1)/(2*radius**2))
    weight = mass / mass.sum(1, keepdims=True).clip(1e-12)
    mean = (weight[..., None]*displacement).sum(1)
    residual = displacement-mean[:, None]
    cov = np.einsum('bnr,bnri,bnrj->brij', weight, residual, residual)
    return np.concatenate([mass.mean(1)[..., None], mean,
        np.stack([cov[:, :, i, j] for i, j in [(0,0),(1,1),(2,2),(0,1),(0,2),(1,2)]], -1)], -1).reshape(len(x), -1)


def feature_names(count):
    return [f'landmark_{r:02d}_{name}' for r in range(count)
        for name in ['soft_mass','local_dx','local_dy','local_dz','cov_xx','cov_yy','cov_zz','cov_xy','cov_xz','cov_yz']]


def root_weights(roots):
    _, inverse, counts = np.unique(roots, return_inverse=True, return_counts=True)
    weight = 1/counts[inverse]
    return weight/weight.sum()


def partial_rank_correlation(features, future, current, weights):
    """Root-balanced rank association, controlling recorded instantaneous score."""
    x = np.stack([rankdata(v) for v in features.T], 1)
    y = rankdata(future)
    design = np.c_[np.ones(len(y)), rankdata(current)]
    dw = design*np.sqrt(weights)[:, None]
    x -= design@np.linalg.lstsq(dw, x*np.sqrt(weights)[:, None], rcond=None)[0]
    y -= design@np.linalg.lstsq(dw, y*np.sqrt(weights), rcond=None)[0]
    x -= weights@x
    y -= weights@y
    denominator = np.sqrt((weights[:, None]*x*x).sum(0)*(weights*y*y).sum())
    result = np.full(features.shape[1], np.nan)
    np.divide((weights[:, None]*x*y[:, None]).sum(0), denominator,
              out=result, where=denominator>1e-10)
    return result


def _integrate(values, times):
    """Integrate contiguous observed edges only; never bridge missing stages."""
    usable = np.isfinite(values[:-1]) & np.isfinite(values[1:])
    dt = np.diff(times)[:, None]
    mass = (dt*usable).sum(0)
    area = np.nansum((values[:-1]+values[1:])*.5*dt*usable, axis=0)
    return np.divide(area, mass, out=np.full(values.shape[1], np.nan), where=mass>0), mass/(times[-1]-times[0])


def _effect(values, bootstrap, rng, minimum_batches):
    v = values[np.isfinite(values)]
    if len(v) < minimum_batches:
        return {'effect':None,'CI_low':None,'CI_high':None,'p':1.,'n_batches':len(v),'sign_agreement':None}
    samples = v[rng.integers(0, len(v), (bootstrap, len(v)))].mean(1)
    p = float(ttest_1samp(v, 0).pvalue) if np.std(v)>1e-12 else 1.
    ci = np.quantile(samples, [.025,.975])
    return {'effect':float(v.mean()),'CI_low':float(ci[0]),'CI_high':float(ci[1]),'p':p,
        'n_batches':len(v),'sign_agreement':float((np.sign(v)==np.sign(v.mean())).mean())}


def prepare(dataset, campaign, metrics, baseline, config, output):
    root, output = Path(dataset).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Use a fresh terminal-region preparation output')
    config_sha=digest(config) if isinstance(config,(str,Path)) else None
    config = read_json(config) if isinstance(config, (str, Path)) else copy.deepcopy(config)
    if config['teacher_coordinates']!='last_learning_endpoint_forecast_of_true_terminal_ancestor':
        raise ValueError('This M2 experiment implements only the declared last-node ancestor forecast teachers')
    paths = source_batches(root, campaign, config.get('arm', 'single'))
    labels = pd.read_csv(metrics)
    if 'arm' in labels:
        labels = labels[labels.arm.eq(config.get('arm', 'single'))].copy()
    available = sorted(set(paths) & set(map(int, labels.batch.unique())))
    if set(paths)-set(available):
        raise ValueError('An observed batch lacks terminal evaluation labels')
    split = split_batches(available, config['split']['fractions'], config['split']['seed'])
    discovery = split['discovery']
    elites = select_terminal_elites(labels[labels.batch.isin(discovery)], config['elite_selection'])
    elite_ids = set(map(tuple, elites[['batch','slot']].to_numpy()))
    base = load_reference(baseline)
    support = list(config['learning_window'])
    if support != base['window']:
        raise ValueError('Configured source learning window differs from baseline support')
    times = np.asarray(base['times'], float)
    points = np.asarray(base['landmarks_A'], float)
    fields = feature_names(len(points))
    radius = float(config['analysis']['radius_A'])
    cfg = read_json(root/'results'/campaign/'config.json')
    arrays = {name: [] for name in ['elite_contrast','future_partial_correlation']}
    events, sources, teacher_rows, teacher_coords, path_locators = [], [], [], [], []
    for batch in available:
        path = paths[batch]
        before = digest(path)
        m = labels[labels.batch.eq(batch)].sort_values('slot').reset_index(drop=True)
        com = np.asarray(read_json(path.parent.parent.parent/f'frame_batch_{batch:03d}.json')['target_com'], float).reshape(-1,1,3)
        with open_trajectory(path) as a:
            clock, states, selected, roots, offspring, resampled = validate_lineage(a)
            n = selected.shape[1]
            if len(m)!=n or not np.array_equal(m.slot.to_numpy(),np.arange(n)):
                raise ValueError('All terminal slots, including failures, must be present')
            ancestors = terminal_ancestors(selected)
            current, proposal = np.asarray(a['current_coords']), np.asarray(a['proposal_coords'])
            if not np.array_equal(current[1:], proposal[np.arange(len(clock)-1)[:,None], selected[:-1]], equal_nan=True):
                raise ValueError('Source parent indices fail actual coordinate-copy verification')
            steps = []
            for t in times:
                matches = np.flatnonzero(np.isclose(clock,t,rtol=0,atol=2e-6)&resampled&(states<=support[1]+2e-6))
                if len(matches)!=1:
                    raise ValueError('Missing or ambiguous real selection score node')
                steps.append(int(matches[0]))
            forecast = np.asarray(a['predicted_coords'],float)[steps]*cfg['coord_scale']+com
            instant = np.asarray(a['pic50_on'],float)[steps].reshape(len(steps),n)
            if not np.asarray(a['mask'])[steps].all():
                raise ValueError('Fixed active teacher point clouds required')
            valid = m.valid_connected.to_numpy() & np.isfinite(m.pic50_on_rescore.to_numpy(float))
            scores = m.pic50_on_rescore.to_numpy(float)
            terminal_elite = np.array([(batch,s) in elite_ids for s in range(n)])
            for final_slot in np.flatnonzero(terminal_elite):
                for j,step in enumerate(steps):
                    slot=int(ancestors[step,final_slot])
                    path_locators.append({'path_id':f'{batch}:{int(final_slot)}','batch':batch,
                        'terminal_slot':int(final_slot),'step':step,'score_time':float(times[j]),
                        'ancestor_slot':slot,'root_slot':int(roots[step,slot]),
                        'coordinate_field':'predicted_coords','source_sha256':before})
            contrast, correlation = [], []
            for j, step in enumerate(steps):
                features = regional_features(forecast[j],points,radius)
                observed, positive, future = np.zeros(n,bool),np.zeros(n,bool),np.full(n,np.nan)
                for slot in range(n):
                    descendants = np.flatnonzero((ancestors[step]==slot)&valid)
                    if len(descendants):
                        observed[slot]=True
                        unique = m.iloc[descendants].groupby('smiles',sort=True).pic50_on_rescore.max()
                        future[slot]=float(unique.mean())
                        positive[slot]=bool(terminal_elite[descendants].any())
                negative=observed&~positive
                row=np.full(len(fields),np.nan)
                if positive.sum()>=1 and negative.sum()>=2:
                    wp=root_weights(roots[step,positive]); wn=root_weights(roots[step,negative])
                    w=root_weights(roots[step,observed]); mu=w@features[observed]
                    scale=np.sqrt(w@((features[observed]-mu)**2)).clip(1e-5)
                    row=(wp@features[positive]-wn@features[negative])/scale
                contrast.append(row)
                pc=np.full(len(fields),np.nan)
                if observed.sum()>=config['analysis']['minimum_observed_nodes']:
                    pc=partial_rank_correlation(features[observed],future[observed],instant[j,observed],root_weights(roots[step,observed]))
                correlation.append(pc)
                events.append({'batch':batch,'split':next(k for k,v in split.items() if batch in v),
                    'score_time':float(times[j]),'observed_nodes':int(observed.sum()),'elite_ancestor_nodes':int(positive.sum()),
                    'normal_ancestor_nodes':int(negative.sum()),'observed_roots':int(np.unique(roots[step,observed]).size),
                    'future_credit':'Mean maximum score per unique observed descendant graph'})
            for name,value in [('elite_contrast',contrast),('future_partial_correlation',correlation)]:
                arrays[name].append(np.asarray(value))
            for group_slot in sorted(set(int(ancestors[steps[-1],slot]) for slot in np.flatnonzero(terminal_elite))):
                final_slots=np.flatnonzero(terminal_elite&(ancestors[steps[-1]]==group_slot))
                teacher_coords.append(forecast[-1,group_slot])
                teacher_rows.append({'batch':batch,'step':steps[-1],'slot':group_slot,
                    'root_slot':int(roots[steps[-1],group_slot]),'path_ids':[f'{batch}:{int(s)}' for s in final_slots],
                    'terminal_slots':final_slots.tolist(),'terminal_max_pic50':float(scores[final_slots].max()),
                    'score_time':float(times[-1]),'coordinate_semantics':'Forecast endpoint at last learning score node'})
        if digest(path)!=before:
            raise ValueError('Source trajectory changed during analysis')
        sources.append({'batch':batch,'path':path.relative_to(root).as_posix(),'sha256':before})
    if not teacher_rows:
        raise ValueError('No genuine ancestors for configured terminal elites')
    output.mkdir(parents=True)
    elites.to_csv(output/'terminal_elites.csv',index=False)
    pd.DataFrame(path_locators).to_parquet(output/'elite_path_locators.parquet',compression=None,index=False)
    pd.DataFrame(events).to_parquet(output/'node_observability.parquet',compression=None,index=False)
    rng=np.random.default_rng(config['analysis']['seed']); statistics=[]; batch_rows=[]; functions={}
    index={batch:i for i,batch in enumerate(available)}
    for measure,values in arrays.items():
        values=np.asarray(values)
        per_batch=np.stack([_integrate(v,times)[0] for v in values])
        for split_name,batches in split.items():
            selected_values=per_batch[[index[b] for b in batches]]
            for feature,k in zip(fields,range(len(fields))):
                result=_effect(selected_values[:,k],config['analysis']['bootstrap'],rng,config['analysis']['minimum_batches'])
                statistics.append({'evidence_id':f'{measure}:{split_name}:{feature}','split':split_name,
                    'measure':measure,'feature':feature,'region_index':k//10,**result})
            for b in batches:
                batch_rows.append({'batch':b,'split':split_name,'measure':measure,
                    **dict(zip(fields,clean(per_batch[index[b]])))})
        donor_values=values[[index[b] for b in discovery]]
        for k,feature in enumerate(fields):
            node_count=np.isfinite(donor_values[:,:,k]).sum(0)
            average=np.divide(np.nansum(donor_values[:,:,k],axis=0),node_count,out=np.full(len(times),np.nan),where=node_count>0)
            rate=np.full(len(times),np.nan)
            if len(times)>1:
                slopes=np.diff(average)/np.diff(times)
                rate[:-1]=slopes
            functions[measure+'/'+feature]={'measure':measure,'feature':feature,'region_index':k//10,
                'method':'empirical_observed_nodes','values':clean(average),'rate_per_time':clean(rate),
                'observed_batches':node_count.tolist(),'observed_support':[float(times[0]),float(times[-1])],
                'missing_policy':'No interpolation across missing stages; no extrapolation',
                'not_spatial_gradient':True}
    table=pd.DataFrame(statistics)
    for split_name in split:
        loc=table.split.eq(split_name)
        table.loc[loc,'q']=bh(table.loc[loc,'p'].fillna(1).to_numpy())
    table.to_parquet(output/'whole_window_effects.parquet',compression=None,index=False)
    pd.DataFrame(batch_rows).to_parquet(output/'batch_window_statistics.parquet',compression=None,index=False)
    write_json(output/'empirical_trends.json',{'window':support,'times':times.tolist(),'functions':functions})
    # One ancestor point cloud per actual node, regardless of descendant clones.
    teacher_batches=np.array([r['batch'] for r in teacher_rows]); teacher_roots=np.array([r['root_slot'] for r in teacher_rows])
    weight=[]
    for b,r in zip(teacher_batches,teacher_roots):
        roots_in_batch=np.unique(teacher_roots[teacher_batches==b]).size
        nodes_in_root=((teacher_batches==b)&(teacher_roots==r)).sum()
        weight.append(-np.log(roots_in_batch*nodes_in_root))
    ref=copy.deepcopy(base)
    clouds=np.asarray(teacher_coords).tolist()
    for f in ref['frames']:
        for key in list(f):
            if key.startswith('teacher_'):del f[key]
        f.update(teacher_endpoint_A=clouds,teacher_scores=[0.]*len(clouds),teacher_batches=teacher_batches.tolist(),
            teacher_base_log_weight=weight,teacher_node_locators=teacher_rows)
    ref.update(reference_variant='terminal-topk-static-ancestor-1.0',allowed_reward_views=['endpoint_pointcloud'],
        sources=[s for s in sources if s['batch'] in discovery],discovery_batches=discovery,
        label_semantics='Teachers selected by decoded final rescore; runtime score beta zero',
        terminal_selection=config['elite_selection'],source_learning_window=support,
        terminal_paths=clean(elites.to_dict('records')))
    reference_path=output/'terminal_teacher_reference.json.gz'
    reference_path.write_bytes(gzip.compress(json.dumps(ref,separators=(',',':'),allow_nan=False).encode(),mtime=0))
    manifest={'schema_version':'terminal-regional-evidence-1.0','config':config,'split':split,'window':support,
        'times':times.tolist(),'terminal_elites':len(elites),'unique_teacher_ancestors':len(clouds),
        'teacher_terminal_score_range':[float(elites.pic50_on_rescore.min()),float(elites.pic50_on_rescore.max())],
        'terminal_metrics_sha256':digest(metrics),'baseline_reference_sha256':digest(baseline),'sources':sources,
        'reference_sha256':digest(reference_path),'regions':len(points),'feature_names':fields,
        'source_code_sha256':digest(__file__),'config_sha256':config_sha,
        'multiple_testing':'BH within each split across all coordinate features and both measures',
        'statistics_unit':'Root-balanced within a node, equal batches; shared ancestry remains dependent',
        'validation_policy':'Validation labels never select teachers; regions chosen only from discovery evidence',
        'limitations':['Retrospective Steer-conditioned association, not causal or native continuation value',
            'Top-k terminal ranking is predicted quality, not an individual statistical significance test',
            'Shared early ancestors may prevent elite/normal comparisons; missing values remain missing'],
        'storage':'Compact batch effects, curves, observability and one small teacher library; no particle feature cache'}
    write_json(output/'manifest.json',manifest)
    return manifest
