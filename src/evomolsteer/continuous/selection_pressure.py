"""Exact selection/transport accounting and coordinate niches, no fitted head.

Clones are not independent evidence. All inference statistics use batch curves.
Instantaneous selection changes divided by dt are bookkeeping rates, not forces.
"""
import copy
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import ttest_1samp

from ..io import digest, read_json, write_json
from ..trajectory_source import open_trajectory, trajectory_paths, pocket_input_path
from .affinity_geometry import names, numpy_geometry
from .coordinate_mining import bh, curve_fit


def price_terms(current, proposal, probability, counts, next_current=None):
    """Uniform candidate measure; exact expected and realized selection shifts."""
    current, proposal = np.asarray(current, float), np.asarray(proposal, float)
    p, n = np.asarray(probability, float), np.asarray(counts, float)
    if current.shape != proposal.shape or len(p) != len(proposal) or np.any(p < 0) or np.any(n < 0):
        raise ValueError('Aligned finite populations required')
    if not all(np.isfinite(v).all() for v in [current, proposal, p, n]) or p.sum() <= 0 or n.sum() <= 0:
        raise ValueError('Finite positive selection mass required')
    p, w = p / p.sum(), n / n.sum()
    native = proposal.mean(0) - current.mean(0)
    expected = p @ proposal - proposal.mean(0)
    realized = w @ proposal - proposal.mean(0)
    total = native + realized
    error = None if next_current is None else float(np.max(np.abs(np.asarray(next_current).mean(0)-current.mean(0)-total)))
    return {'native': native, 'expected_selection': expected, 'realized_selection': realized,
            'resampling_noise': realized-expected, 'total': total, 'identity_error': error}


def cloud_match(a, b):
    row, col = linear_sum_assignment(((a[:, None]-b[None])**2).sum(-1))
    return float(((a[row]-b[col])**2).sum(-1).mean()), col


def coordinate_niches(clouds, batches, maximum=4):
    """Deterministic world-frame, permutation-invariant farthest-point niches."""
    clouds, batches = np.asarray(clouds, float), np.asarray(batches)
    m, atoms = clouds.shape[:2]
    distance = np.zeros((m, m))
    for i in range(m):
        for j in range(i):
            distance[i, j] = distance[j, i] = cloud_match(clouds[i], clouds[j])[0]
    centers = [int(np.argmin(distance.sum(1)))]
    while len(centers) < min(maximum, m):
        far = distance[:, centers].min(1);far[centers] = -1
        if far.max() <= 1e-12:break
        centers.append(int(np.argmax(far)))
    groups = distance[:, centers].argmin(1)
    precision = np.zeros((m, atoms, 3, 3));records = []
    for g, center in enumerate(centers):
        ids = np.flatnonzero(groups == g)
        count = {v: int(np.sum(batches[ids] == v)) for v in np.unique(batches[ids])}
        w = np.array([1/count[v] for v in batches[ids]], float);w /= w.sum()
        ess = 1/(w @ w)
        aligned, assignments = [], []
        for i in ids:
            _, col = cloud_match(clouds[center], clouds[i])
            aligned.append(clouds[i, col]);assignments.append(col)
        values = np.asarray(aligned);mu = np.einsum('m,mnd->nd',w,values)
        delta = values-mu
        cov = np.einsum('m,mni,mnj->nij',w,delta,delta)
        isotropic = np.trace(cov,axis1=1,axis2=2)/3
        # Shrink small observed modes; a singleton supplies no anisotropy.
        confidence = max(0., (ess-2)/(ess+6))
        cov = confidence*cov + (1-confidence)*isotropic[:,None,None]*np.eye(3)
        cov += np.maximum(isotropic*.1,1e-3)[:,None,None]*np.eye(3)
        ev, vec = np.linalg.eigh(cov)
        inv = 1/ev;inv /= inv.mean(1)[:,None];inv = inv.clip(.5,2.);inv /= inv.mean(1)[:,None]
        metric = np.einsum('nij,nj,nkj->nik',vec,inv,vec)
        for i, col in zip(ids, assignments):
            precision[i, col] = metric
        records.append({'niche':g,'prototype':center,'teachers':len(ids),'independent_batches':len(count),
                        'effective_n':float(ess),'shrinkage_confidence':float(confidence)})
    return groups, precision, records


def augment_reference(source, output):
    source, output = Path(source), Path(output)
    if output.exists():raise FileExistsError(output)
    r = json.loads(gzip.decompress(source.read_bytes()));report=[]
    if r['schema_version']!='affinity-endpoint-library-1.0':raise ValueError('Endpoint library required')
    for frame in r['frames']:
        groups, precision, records = coordinate_niches(frame['teacher_endpoint_A'], frame['teacher_batches'])
        frame['teacher_niche'] = groups.tolist();frame['teacher_precision'] = precision.tolist()
        report.append({'time':frame['time'],'niches':records})
    r['selection_path']={'schema_version':'selection-path-coordinate-metric-1.0',
      'parent_reference_sha256':digest(source),'precision_semantics':'Within geometric niche, family-balanced coordinate covariance, shrunk SPD trace-normalized metric; not energy or native-future fitness.',
      'grouping':'World-frame Hungarian distance, deterministic farthest-point partition; no atom type or graph restriction.'}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_bytes(gzip.compress(json.dumps(r,separators=(',',':'),allow_nan=False).encode(),mtime=0))
    return {'parent_reference_sha256':digest(source),'reference_sha256':digest(output),'bytes':output.stat().st_size,'frames':report}


def mine(dataset, campaign, reference, output, seed=42):
    root, out = Path(dataset), Path(output)
    if out.exists():raise FileExistsError(out)
    ref = json.loads(gzip.decompress(Path(reference).read_bytes()));grid=np.asarray(ref['times'])
    a,b=ref['window'];cfg=read_json(root/'results'/campaign/'config.json')
    points,origin=np.asarray(ref['landmarks_A']),np.asarray(ref['origin_A']);features=names(points)
    curves=[];events=[];sources=[];worst=0.
    expected_folders={Path(item['path']).parent.as_posix() for item in ref['sources']}
    for path in trajectory_paths(root/'results'/campaign):
        if path.relative_to(root).parent.as_posix() not in expected_folders:continue
        batch=int(path.parent.name.split('_')[1])
        if batch not in ref['discovery_batches']:continue
        com=np.asarray(read_json(root/'results'/campaign/f'frame_batch_{batch:03d}.json')['target_com'])[:,None]
        with open_trajectory(path) as tr:
            times=np.round(np.asarray(tr['score_time'])[:,0],6);state=np.asarray(tr['state_time'])
            ids=np.flatnonzero(np.asarray(tr['resampled'],bool)&(times>=a-1e-6)&(state<=b+1e-6))
            if len(ids)!=len(grid) or not np.allclose(times[ids],grid):raise ValueError('Selection support mismatch')
            current=np.asarray(tr['current_coords'],float)*cfg['coord_scale']+com
            proposal=np.asarray(tr['proposal_coords'],float)*cfg['coord_scale']+com
            probability=np.asarray(tr['selection_probability'],float);counts=np.asarray(tr['offspring_count'])
            chosen=np.asarray(tr['selected_indices']);roots=np.asarray(tr['root_slot'])
            records=[]
            for i in ids:
                vc,vp=numpy_geometry(current[i],points,origin),numpy_geometry(proposal[i],points,origin)
                if i+1<len(current):
                    if not np.allclose(current[i+1],proposal[i,chosen[i]],rtol=0,atol=2e-6):raise ValueError('Native/copy edge mismatch')
                    vn=numpy_geometry(current[i+1],points,origin)
                else:vn=None
                terms=price_terms(vc,vp,probability[i],counts[i],vn)
                if terms['identity_error'] is not None:worst=max(worst,terms['identity_error'])
                dt=state[i]-times[i]
                if dt<=0:raise ValueError('Positive actual integrator duration required')
                records.append({k:terms[k]/dt for k in ['native','expected_selection','realized_selection','resampling_noise']})
                p=probability[i]/probability[i].sum();w=counts[i]/counts[i].sum()
                root_mass=np.bincount(roots[i],weights=w,minlength=len(w));root_mass=root_mass[root_mass>0]
                events.append({'batch':batch,'time':times[i],'state_time':state[i],
                  'expected_ess_fraction':float(1/(p@p)/len(p)),
                  'realized_ess_fraction':float(1/(w@w)/len(w)),
                  'root_entropy_nats':float(-(root_mass*np.log(root_mass)).sum()),
                  'surviving_roots':len(root_mass),'selection_KL_from_uniform_nats':float(np.sum(p[p>0]*np.log(p[p>0]*len(p)))),
                  'score_selection_covariance':float(p@np.asarray(tr['pic50_on'])[i].reshape(-1)-np.asarray(tr['pic50_on'])[i].mean())})
            curves.append((batch,records))
        sources.append({'path':path.relative_to(root).as_posix(),'sha256':digest(path)})
    if len(curves)!=len(ref['discovery_batches']):raise ValueError('All discovery batches required')
    out.mkdir(parents=True);pd.DataFrame(events).to_parquet(out/'events.parquet',compression=None,index=False)
    rows=[];functions={};rng=np.random.default_rng(seed)
    from .dynamic_regions import observed_integral
    for term in ['native','expected_selection','realized_selection','resampling_noise']:
        values=np.array([[r[term] for r in c] for _,c in curves]);integrated=observed_integral(values,grid)
        boot=integrated[rng.integers(0,len(curves),(2000,len(curves)))].mean(1)
        pvals=np.nan_to_num(ttest_1samp(integrated,0,axis=0).pvalue,nan=1.)
        qvals=bh(pvals)
        for j,feature in enumerate(features):
            ci=np.quantile(boot[:,j],[.025,.975]);effect=integrated[:,j].mean()
            rows.append({'term':term,'feature':feature,'whole_window_mean_rate':effect,'CI_low':ci[0],'CI_high':ci[1],
              'p':pvals[j],'q':qvals[j],'same_sign_batch_fraction':float(np.mean(np.sign(integrated[:,j])==np.sign(effect)))})
            functions[term+'/'+feature]=curve_fit(grid,values[:,:,j],max_degree=3)
    pd.DataFrame(rows).to_parquet(out/'effects.parquet',compression=None,index=False)
    write_json(out/'functions.json',functions)
    write_json(out/'manifest.json',{'window':ref['window'],'score_grid':grid.tolist(),'sources':sources,
      'reference_sha256':digest(reference),'batches':len(curves),'features':len(features),'events':len(events),
      'maximum_price_identity_error':worst,'statistics_unit':'Independent batch, never clones',
      'rate_semantics':'Population change per integration interval; instantaneous copying has zero physical duration and supplies no spatial gradient.',
      'limitations':['Online head and recorded probabilities share the selector; association is not native-future causal fitness.',
                     'Transport is model integration, not physical molecular dynamics. No terminal failures imputed.'],
      'storage':'Only event and whole-window summaries/functions; no per-seed geometry expansion.'})
    return read_json(out/'manifest.json')
