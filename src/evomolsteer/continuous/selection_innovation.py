"""Joint coordinate contrasts on observed Steer decisions, without a learned head.

The signed field points from nearby lower-score endpoint clouds toward a frozen
teacher. Root-balanced controls prevent copied particles from increasing support.
Only generation batches are independent units for uncertainty and global trends.
Recorded head labels and selection probabilities are observational evidence, not
an endpoint-only affinity oracle or attribution to any neural-network module.
"""
import copy
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.polynomial.legendre import legder
from scipy.optimize import linear_sum_assignment
from scipy.stats import ttest_1samp

from ..io import digest, read_json, write_json
from ..trajectory_source import open_trajectory, trajectory_paths
from .coordinate_mining import bh
from .dynamic_regions import masked_curve_fit, observed_integral


def rms(field):
    return float(np.sqrt(np.mean(np.sum(np.asarray(field, float)**2, axis=-1))))


def cosine(a, b):
    a, b = np.asarray(a, float).reshape(-1), np.asarray(b, float).reshape(-1)
    den = np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/den) if den > 1e-12 else 0.


def bounded_atom_weights(amplitude):
    """Exactly mean-one geometric weights with finite lower/upper bounds."""
    amplitude=np.asarray(amplitude,float)
    if np.any(amplitude<0) or not np.isfinite(amplitude).all():raise ValueError('Finite nonnegative atom amplitudes required')
    if amplitude.max()<=1e-12:return np.ones_like(amplitude)
    scaled=amplitude/amplitude.max()
    # The additive epsilon permits any number of zero-amplitude atoms while
    # keeping the mean-one bounded normalization uniquely solvable.
    scaled=np.maximum(scaled,1e-6);low,high=0.,2e6
    for _ in range(80):
        middle=(low+high)/2
        if np.clip(middle*scaled,.5,2.).mean()<1.:low=middle
        else:high=middle
    return np.clip((low+high)/2*scaled,.5,2.)


def direction_components(anchor, field):
    """Pose translation/infinitesimal rotation versus internal shape variation."""
    a,v=np.asarray(anchor,float),np.asarray(field,float);energy=float((v*v).sum())
    if energy<=1e-24:return {'translation_fraction':0.,'rotation_fraction':0.,'internal_fraction':0.,'atom_participation_fraction':0.}
    center=a-a.mean(0);translation=np.broadcast_to(v.mean(0),v.shape);residual=v-translation
    # Columns are Cartesian infinitesimal rotation axes crossed with each point.
    design=np.stack([np.cross(np.broadcast_to(axis,a.shape),center) for axis in np.eye(3)],-1).reshape(-1,3)
    omega=np.linalg.lstsq(design,residual.reshape(-1),rcond=None)[0]
    rotation=(design@omega).reshape(v.shape);internal=residual-rotation
    mass=(v*v).sum(-1);mass/=mass.sum()
    return {'translation_fraction':float((translation*translation).sum()/energy),
            'rotation_fraction':float((rotation*rotation).sum()/energy),
            'internal_fraction':float((internal*internal).sum()/energy),
            'atom_participation_fraction':float(1/(len(v)*(mass@mass)))}


def match_cloud(anchor, cloud):
    """World-frame matching; rotations/translations relative to pocket are retained."""
    a, b = np.asarray(anchor, float), np.asarray(cloud, float)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3:
        raise ValueError('Equal finite atom clouds required')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Nonfinite coordinate cloud')
    row, col = linear_sum_assignment(((a[:, None]-b[None])**2).sum(-1))
    order = np.empty(len(a), int); order[row] = col
    return b[order], order


def signed_contrast(anchor, teacher_score, clouds, scores, roots, neighbours=6):
    """Bounded, root-balanced higher-minus-lower JOINT coordinate direction.

    This is a descriptive extrapolation hypothesis. Its confidence is a small
    shrinkage of support and coherence, never a causal probability or p-value.
    Exact score ties and geometrically identical clouds contribute exact zero.
    """
    if neighbours < 1: raise ValueError('Positive neighbour count required')
    anchor, clouds = np.asarray(anchor, float), np.asarray(clouds, float)
    scores, roots = np.asarray(scores, float), np.asarray(roots)
    if len(clouds) != len(scores) or len(scores) != len(roots) or not np.isfinite(scores).all():
        raise ValueError('Aligned finite candidates required')
    records = []
    for j in np.flatnonzero(scores < teacher_score-1e-10):
        aligned, _ = match_cloud(anchor, clouds[j]); residual = anchor-aligned
        distance = rms(residual)
        if distance > 1e-10: records.append((distance, int(j), residual))
    # Stable order keeps a reproducible neighbourhood; tied identical candidates
    # affect only root balance, not the direction.
    records.sort(key=lambda v: (v[0], v[1])); records = records[:neighbours]
    zero = np.zeros_like(anchor)
    if not records:
        return {'direction_unit':zero, 'confidence':0., 'atom_weight':np.ones(len(anchor)),
                'raw_rms_A':0., 'coherence':0., 'neighbour_ids':[], 'root_count':0,
                'root_weight_ess':0., 'weighted_score_gap':0., 'noise_rms_A':0.,
                'distance_mean_A':0., 'sign_calibration_cosine':0.}
    distance = np.array([r[0] for r in records]); ids = np.array([r[1] for r in records])
    delta = np.array([r[2] for r in records]); local_roots = roots[ids]
    _, group, multiplicity = np.unique(local_roots, return_inverse=True, return_counts=True)
    scale = max(float(np.median(np.abs(scores-np.median(scores)))*1.4826), .05)
    gap = teacher_score-scores[ids]
    w = np.exp(-distance/max(float(np.median(distance)), .1))/multiplicity[group]
    w /= w.sum()
    # Within each root, particles share a fixed root mass. Their geometric
    # differences remain available, but copying does not create extra evidence.
    w /= np.bincount(group, weights=w)[group]; w /= w.sum()
    bounded = delta/(1+distance[:,None,None]) * np.tanh(gap/scale)[:,None,None]
    field = np.einsum('m,mnd->nd', w, bounded); amplitude = rms(field)
    coherence = amplitude/max(float(np.dot(w, [rms(d) for d in bounded])), 1e-12)
    root_mass = np.bincount(group, weights=w); root_ess = float(1/(root_mass@root_mass))
    noise = np.sqrt(np.einsum('m,mnd,mnd->', w, bounded-field, bounded-field)/len(anchor))
    # At most 0.2 local confidence, further shrunk against small family support.
    confidence = .2*max(root_ess-1, 0)/(root_ess+4)*min(coherence, 1.) if len(root_mass)>1 else 0.
    unit = field/amplitude if amplitude > 1e-12 else zero
    atom = bounded_atom_weights(np.sqrt((field*field).sum(-1)))
    return {'direction_unit':unit, 'confidence':float(confidence), 'atom_weight':atom,
            'raw_rms_A':amplitude, 'coherence':float(coherence), 'neighbour_ids':ids.tolist(),
            'root_count':len(root_mass), 'root_weight_ess':root_ess,
            'weighted_score_gap':float(w@gap), 'noise_rms_A':float(noise),
            'distance_mean_A':float(w@distance), 'sign_calibration_cosine':cosine(field, np.einsum('m,mnd->nd',w,delta))}


def parent_innovations(endpoint, previous_endpoint, parents, score, previous_score, native_displacement):
    """Observed forecast innovation on real edges; clones collapse per parent.

    No offspring future is fabricated for a parent eliminated by resampling.
    Native displacement is the preceding integrator update, not a force.
    """
    endpoint, previous_endpoint = np.asarray(endpoint,float), np.asarray(previous_endpoint,float)
    parents = np.asarray(parents,int)
    if len(parents) != len(endpoint) or np.any((parents<0)|(parents>=len(previous_endpoint))):
        raise ValueError('Invalid parent lineage')
    innovation = endpoint-previous_endpoint[parents]
    score_gain = np.asarray(score,float)-np.asarray(previous_score,float)[parents]
    amplitude = np.sqrt((innovation*innovation).sum(-1).mean(-1))
    native_cosine = np.array([cosine(v, native_displacement[p]) for v,p in zip(innovation,parents)])
    groups = np.unique(parents)
    # Same-parent centered association isolates branch variation. Each extant
    # parent contributes one covariance and variance, independent of copy count.
    covariance, varx, vary = [], [], []
    for p in groups:
        ids=parents==p; x=amplitude[ids]-amplitude[ids].mean();y=score_gain[ids]-score_gain[ids].mean()
        covariance.append(float((x*y).mean()));varx.append(float((x*x).mean()));vary.append(float((y*y).mean()))
    den=np.sqrt(np.mean(varx)*np.mean(vary))
    within_parent_corr=float(np.mean(covariance)/den) if den>1e-12 else np.nan
    avg=lambda x:float(np.mean([np.asarray(x)[parents==p].mean() for p in groups]))
    return innovation, {'innovation_RMS_A':avg(amplitude), 'innovation_native_alignment':avg(native_cosine),
                         'observed_child_score_gain':avg(score_gain),
                         'within_parent_innovation_gain_correlation':within_parent_corr,
                         'observed_parent_count':len(groups), 'branching_parent_count':int(sum((parents==p).sum()>1 for p in groups))}


ZERO_TEST_METRICS = {'innovation_native_alignment','observed_child_score_gain',
                    'within_parent_innovation_gain_correlation','teacher_innovation_contrast_alignment'}


def summarize_events(events, times, output, seed):
    frame = pd.DataFrame(events); batch_ids=sorted(frame.batch.unique()); curves=[]; names=[]
    omit={'batch','time','state_time','teacher_count','teacher_unique_roots','observed_parent_count','branching_parent_count'}
    metrics=[m for m in frame.columns if m not in omit]
    for metric in metrics:
        values=frame.pivot(index='batch',columns='time',values=metric).reindex(index=batch_ids,columns=times).to_numpy()
        if np.isfinite(values).any(): names.append(metric);curves.append(values)
    values=np.stack(curves,-1); integrated=observed_integral(values,np.asarray(times))
    rng=np.random.default_rng(seed);boot=integrated[rng.integers(0,len(batch_ids),(2000,len(batch_ids)))].mean(1)
    pvals=np.array([float(ttest_1samp(integrated[:,i][np.isfinite(integrated[:,i])],0).pvalue)
                   if name in ZERO_TEST_METRICS and np.isfinite(integrated[:,i]).sum()>2 else np.nan
                   for i,name in enumerate(names)])
    qvals=bh(pvals);rows=[]; fits={}
    for i,name in enumerate(names):
        curve=values[:,:,i]; observed_columns=np.isfinite(curve).all(0)
        fit=masked_curve_fit(np.asarray(times)[observed_columns],curve[:,observed_columns],max_degree=3)
        fit['analytic_derivative_legendre_coefficients']=(legder(np.asarray(fit['legendre_coefficients']))*
                     2/(fit['time_end']-fit['time_start'])).tolist()
        fit['interpretation']='Global whole-window ensemble trend. Its time derivative is not a spatial reward gradient.'
        fits[name]=fit
        effect=float(np.nanmean(integrated[:,i]));ci=np.nanquantile(boot[:,i],[.025,.975])
        rows.append({'id':'selection_innovation/'+name,'metric':name,'whole_window_mean':effect,
                     'CI_low':float(ci[0]),'CI_high':float(ci[1]),'p':float(pvals[i]),'q':float(qvals[i]),
                     'positive_batch_fraction':float(np.nanmean(integrated[:,i]>0)),
                     'null_tested':name in ZERO_TEST_METRICS,
                     'interpretation':'Descriptive within-selector association; batch is the independent unit.'})
    frame.to_parquet(output/'event_statistics.parquet',compression=None,index=False)
    pd.DataFrame(rows).to_parquet(output/'whole_window_effects.parquet',compression=None,index=False)
    write_json(output/'fitted_trends.json',fits)
    return rows


def mine(dataset, campaign, reference, output, seed=42, neighbours=6):
    root,out=Path(dataset).resolve(),Path(output).resolve();reference=Path(reference).resolve()
    if out.exists():raise FileExistsError(out)
    original=json.loads(gzip.decompress(reference.read_bytes()));ref=copy.deepcopy(original)
    if ref['schema_version']!='affinity-endpoint-library-1.0':raise ValueError('Frozen endpoint reference required')
    source=root/'results'/campaign;cfg=read_json(source/'config.json');grid=np.asarray(ref['times'],float)
    if len(ref['discovery_batches'])<3:raise ValueError('At least three independent batches required')
    paths={int(p.parent.name.split('_')[1]):p for p in trajectory_paths(source) if p.parent.parent.name=='single'}
    all_stats={};events=[];sources=[];worst=0.;counts={'exact_zero_equal_score':0,'exact_zero_identical_cloud':0}
    for batch in ref['discovery_batches']:
        path=paths[int(batch)];source_sha=digest(path);com=np.asarray(read_json(source/f'frame_batch_{batch:03d}.json')['target_com'])[:,None,:]
        with open_trajectory(path) as tr:
            clock=np.round(np.asarray(tr['score_time'],float)[:,0],6);state=np.asarray(tr['state_time'],float)
            ids=np.flatnonzero(np.asarray(tr['resampled'],bool)&(clock>=ref['window'][0]-2e-6)&(state<=ref['window'][1]+2e-6))
            if not np.array_equal(clock[ids],grid) or np.any(np.diff(ids)!=1):raise ValueError('Exact contiguous learned support required')
            if not np.isclose(state[ids[-1]],ref['window'][1],atol=2e-6):raise ValueError('Incomplete selected integration support')
            if not np.asarray(tr['mask'])[ids].all():raise ValueError('Fixed active slots required')
            y=np.asarray(tr['predicted_coords'],float)[ids]*cfg['coord_scale']+com
            x=np.asarray(tr['current_coords'],float)[ids]*cfg['coord_scale']+com
            proposal=np.asarray(tr['proposal_coords'],float)[ids]*cfg['coord_scale']+com
            score=np.asarray(tr['pic50_on'],float)[ids];roots=np.asarray(tr['root_slot'])[ids]
            selected=np.asarray(tr['selected_indices'],int)[ids];offspring=np.asarray(tr['offspring_count'],int)[ids]
            probability=np.asarray(tr['selection_probability'],float)[ids]
            for k,i in enumerate(ids):
                frame=ref['frames'][k];teacher_ids=np.flatnonzero(np.asarray(frame['teacher_batches'])==batch)
                if not len(teacher_ids):raise ValueError('Discovery batch lacks frozen teachers')
                innovation=None;edge={key:np.nan for key in ('innovation_RMS_A','innovation_native_alignment','observed_child_score_gain',
                      'within_parent_innovation_gain_correlation','observed_parent_count','branching_parent_count')}
                if k:
                    if not np.allclose(x[k],proposal[k-1,selected[k-1]],atol=2e-6,rtol=0):raise ValueError('Observed lineage does not follow selected parents')
                    innovation,edge=parent_innovations(y[k],y[k-1],selected[k-1],score[k],score[k-1],proposal[k-1]-x[k-1])
                diagnostics=[];teacher_roots=[]
                for j in teacher_ids:
                    anchor=np.asarray(frame['teacher_endpoint_A'][j],float);matches=[]
                    for p in range(len(y[k])):
                        aligned,order=match_cloud(anchor,y[k,p]);matches.append((rms(anchor-aligned),p,order))
                    compatible=[m for m in matches if abs(float(frame['teacher_scores'][j])-score[k,m[1]])<=1e-6]
                    if not compatible:raise ValueError(f'Frozen teacher score missing: batch {batch}, node {k}, teacher {j}')
                    error,p,order=min(compatible,key=lambda z:(z[0],z[1]));worst=max(worst,error)
                    if error>1e-5 or abs(float(frame['teacher_scores'][j])-score[k,p])>1e-6:
                        raise ValueError(f'Frozen teacher mismatch: batch {batch}, node {k}, teacher {j}, RMS {error}')
                    result=signed_contrast(anchor,float(frame['teacher_scores'][j]),y[k],score[k],roots[k],neighbours)
                    # Explicit dataset-level null controls; no null outputs cached.
                    equal=signed_contrast(anchor,1.,y[k],np.ones(len(y[k])),roots[k],neighbours)
                    identical=signed_contrast(anchor,float(frame['teacher_scores'][j]),np.repeat(anchor[None],len(y[k]),axis=0),score[k],roots[k],neighbours)
                    if np.any(equal['direction_unit']) or equal['confidence']!=0.:raise AssertionError('Equal-score null is not exact zero')
                    if np.any(identical['direction_unit']) or identical['confidence']!=0.:raise AssertionError('Identical-cloud null is not exact zero')
                    counts['exact_zero_equal_score']+=1;counts['exact_zero_identical_cloud']+=1
                    local_innovation=np.zeros_like(anchor) if innovation is None else innovation[p,order]
                    irms=rms(local_innovation);unit=local_innovation/irms if irms>1e-12 else np.zeros_like(anchor)
                    neighbours_ids=result['neighbour_ids'];fraction=float(np.mean(offspring[k,neighbours_ids]==0)) if neighbours_ids else 0.
                    provenance={'batch':int(batch),'candidate_slot':int(p),'root_slot':int(roots[k,p]),
                       'score_time':float(grid[k]),'state_time':float(state[i]),'source_sha256':source_sha,
                       'teacher_match_RMS_A':error,'lower_neighbour_slots':neighbours_ids,
                       'lower_neighbour_roots':roots[k,neighbours_ids].tolist(),'lower_neighbour_rejected_fraction':fraction,
                       'root_count':result['root_count'],'root_weight_ess':result['root_weight_ess'],
                       'observed_innovation':bool(k),'previous_parent_slot':int(selected[k-1,p]) if k else None,
                       'local_confidence':result['confidence'],'direction_noise_rms_A':result['noise_rms_A']}
                    all_stats[k,int(j)]={'contrast':result,'innovation_unit':unit,'provenance':provenance}
                    teacher_roots.append(int(roots[k,p]));prob=probability[k]/probability[k].sum()
                    components=direction_components(anchor,result['direction_unit'])
                    diagnostics.append({'teacher_contrast_RMS_A':result['raw_rms_A'],'teacher_contrast_coherence':result['coherence'],
                      'teacher_control_root_ESS':result['root_weight_ess'],'teacher_score_gap':result['weighted_score_gap'],
                      'teacher_noise_RMS_A':result['noise_rms_A'],'teacher_lower_rejected_fraction':fraction,
                      'teacher_expected_probability_advantage':float(prob[p]-1/len(prob)),
                      'teacher_sign_calibration_cosine':result['sign_calibration_cosine'],
                      'teacher_innovation_contrast_alignment':cosine(local_innovation,result['direction_unit']) if k else np.nan,
                      **{'teacher_contrast_'+name: value for name,value in components.items()}})
                event={'batch':int(batch),'time':float(grid[k]),'state_time':float(state[i]),
                       'teacher_count':len(teacher_ids),'teacher_unique_roots':len(set(teacher_roots)),**edge}
                # Average teacher diagnostics inside each extant root, then roots.
                for name in diagnostics[0]:
                    event[name]=float(np.mean([np.mean([d[name] for d,r in zip(diagnostics,teacher_roots) if r==root_id]) for root_id in sorted(set(teacher_roots))]))
                events.append(event)
        sources.append({'path':path.relative_to(root).as_posix(),'sha256':source_sha,'reference_source':[s for s in ref['sources'] if Path(s['path']).parent==Path(path.relative_to(root)).parent]})
        print(f'selection innovation batch {batch}: {len(ids)} observed decisions',flush=True)
    # Cross-batch replication supplies extra descriptive support, never treats
    # the two teachers or collapsed roots in a batch as independent replicates.
    for k,frame in enumerate(ref['frames']):
        clouds=np.asarray(frame['teacher_endpoint_A'],float);batches=np.asarray(frame['teacher_batches'])
        direction=[];confidence=[];weights=[];innovations=[];provenance=[]
        for j,anchor in enumerate(clouds):
            record=all_stats[k,j];v=record['contrast']['direction_unit'];batch_alignment=[]
            for donor in sorted(set(batches)-{batches[j]}):
                other=np.flatnonzero(batches==donor);match=[]
                for h in other:
                    aligned,order=match_cloud(anchor,clouds[h]);match.append((rms(anchor-aligned),int(h),order))
                _,h,order=min(match,key=lambda z:(z[0],z[1]));batch_alignment.append(cosine(v,all_stats[k,h]['contrast']['direction_unit'][order]))
            mean_alignment=float(np.mean(batch_alignment))
            agreement=float(np.mean(np.asarray(batch_alignment)>0))
            confidence.append(record['contrast']['confidence']*max(mean_alignment,0.)*agreement)
            direction.append(v.tolist());weights.append(record['contrast']['atom_weight'].tolist())
            innovations.append(record['innovation_unit'].tolist())
            provenance.append({**record['provenance'],'independent_other_batches':len(batch_alignment),
                               'cross_batch_mean_cosine':mean_alignment,'cross_batch_positive_fraction':agreement})
        frame.update(teacher_contrast_direction_unit=direction,teacher_contrast_confidence=confidence,
                     teacher_contrast_atom_weight=weights,teacher_innovation_direction_unit=innovations,
                     teacher_contrast_provenance=provenance)
        # Guarantee original teacher geometry, scores and priors remain intact.
        for key in original['frames'][k]:
            if frame[key]!=original['frames'][k][key]:raise AssertionError('Original frozen reference changed')
    ref['selection_innovation']={'schema_version':'joint-selection-innovation-1.0','parent_reference_sha256':digest(reference),
        'direction_semantics':'Joint Hungarian-matched endpoint difference: higher teacher minus root-balanced nearby strictly lower-score clouds. RMS-normalized, no atom types.',
        'confidence_semantics':'Local root-weight ESS/coherence with maximum 0.2 shrinkage, multiplied by positive cross-independent-batch matched-direction support. Descriptive reliability, not probability of causality.',
        'native_semantics':'Observed model integration displacement. Forecast innovation and copying are not physical forces.',
        'neighbours':neighbours,'original_teachers_unchanged':True,'local_radius_A_floor':.1}
    out.mkdir(parents=True);rows=summarize_events(events,grid,out,seed)
    augmented=out/'augmented_reference.json.gz';augmented.write_bytes(gzip.compress(json.dumps(ref,separators=(',',':'),allow_nan=False).encode(),mtime=0))
    scope={'window':ref['window'],'score_times':ref['times'],'observed_innovation_times':ref['times'][1:],
           'last_selected_state_time':ref['window'][1],'extrapolation_allowed':False}
    limitations=['Recorded latent/head score is also the selector: score-gap association is not independent affinity validation.',
                 'Late family collapse can remove independent local control families; confidence then vanishes exactly.',
                 'Rejected parent futures are unknown. Observed child gains condition on surviving parents.',
                 'Full joint endpoint contrast is a local extrapolation hypothesis, not an estimated affinity gradient.',
                 'Historical coordinates contain no module activations/Jacobians: no model-module attribution is made.',
                 'The original reference points to normalized H5 identities; actual NPZ sources are separately hashed and teacher-equivalence verified.']
    confidence=np.array([c for f in ref['frames'] for c in f['teacher_contrast_confidence']])
    unavailable=[key for key in events[0] if key not in ('batch','time','state_time') and not np.isfinite([e[key] for e in events]).any()]
    evidence={'schema_version':'selection-innovation-evidence-1.0','scope':scope,'evidence_items':rows,
        'reference_sha256':digest(reference),'augmented_reference_sha256':digest(augmented),
        'independent_batches':len(ref['discovery_batches']),'teacher_records':len(confidence),
        'confidence_nonzero_fraction':float(np.mean(confidence>0)), 'confidence_maximum':float(confidence.max()),
        'exact_null_controls':counts,'unavailable_metrics':unavailable,'limitations':limitations}
    write_json(out/'evidence.json',evidence)
    files={p.name:{'sha256':digest(p),'bytes':p.stat().st_size} for p in sorted(out.iterdir()) if p.is_file()}
    manifest={'schema_version':'selection-innovation-mining-1.0','scope':scope,'input_reference_sha256':digest(reference),
      'augmented_reference_sha256':digest(augmented),'source_code_sha256':digest(__file__),'seed':seed,'neighbours':neighbours,
      'independent_batches':len(ref['discovery_batches']),'events':len(events),'teacher_records':len(confidence),
      'sources':sources,'maximum_teacher_match_RMS_A':worst,'exact_null_controls':counts,
      'statistics_unit':'Equal independent generation batches. Teachers and extant roots are not new independent replicates.',
      'storage':'Only event moments, whole-window uncertainty/global trends and bounded teacher-local fields. No node/edge feature cache.',
      'files':files,'limitations':limitations}
    write_json(out/'manifest.json',manifest)
    return manifest
