"""Lag-two natural branch mutations, with exact copy and ancestry accounting.

Immediate siblings can be identical forecasts of an already copied state. This
tool compares distinct immediate-parent branches within an observed common
grandparent instead. Copies are averaged, ancestor moments have equal weight,
and generation batches are the only inferential replicates. Chemistry changes
are diagnostic nuisance variables, never graph gates or affinity predictors.
"""
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.polynomial.legendre import legder
from scipy.stats import ttest_1samp

from ..io import digest, read_json, write_json
from ..trajectory_source import open_trajectory, trajectory_paths
from .affinity_geometry import names, numpy_geometry
from .coordinate_mining import bh
from .dynamic_regions import masked_curve_fit, observed_integral
from .selection_innovation import match_cloud,rms,cosine,signed_contrast


def conditional_teacher_contrast(anchor,teacher_score,endpoint,score,parents,ancestors,teacher_parent,teacher_ancestor,score_tolerance=0.):
    """Same-grandparent fresh mutations; initial root identity is irrelevant.

    Current copies of an immediate-parent proposal do not supply extra support.
    One lower-scored mutation plus the target is two observed branches, with a
    strong small-sample/noise shrinkage rather than an initial-root zero gate.
    """
    branches=[]
    for parent in np.unique(parents):
        ids=np.asarray(parents)==parent
        if parent==teacher_parent or int(np.asarray(ancestors)[ids][0])!=teacher_ancestor:continue
        cloud=np.asarray(endpoint)[ids];values=np.asarray(score)[ids]
        meancloud=cloud[0].copy() if np.array_equal(cloud,np.broadcast_to(cloud[0],cloud.shape)) else cloud.mean(0)
        branches.append((int(parent),meancloud,float(values.mean())))
    lower=[item for item in branches if item[2]<teacher_score-score_tolerance]
    if not lower:
        return {'direction_unit':np.zeros_like(anchor),'confidence':0.,'atom_weight':np.ones(len(anchor)),
          'lower_parent_ids':[],'distinct_observed_mutations':1,'raw_rms_A':0.,'noise_rms_A':0.,'local_coherence':0.,'weighted_score_gap':0.}
    clouds=np.array([v[1] for v in lower]);scores=np.array([v[2] for v in lower]);ids=np.array([v[0] for v in lower])
    result=signed_contrast(anchor,teacher_score,clouds,scores,ids,neighbours=min(6,len(ids)))
    chosen=ids[result['neighbour_ids']];n=len(chosen)+1;amplitude=result['raw_rms_A'];noise=result['noise_rms_A']
    # A nonzero fixed spatial uncertainty prior covers singleton comparisons.
    # This reliability is descriptive, not a causal posterior or independent N.
    confidence=.15*(n-1)/(n+4)*result['coherence']*amplitude/(amplitude+noise+.02) if amplitude>0 else 0.
    return {'direction_unit':result['direction_unit'],'confidence':float(confidence),'atom_weight':result['atom_weight'],
      'lower_parent_ids':chosen.tolist(),'distinct_observed_mutations':n,'raw_rms_A':amplitude,
      'noise_rms_A':noise,'local_coherence':result['coherence'],'weighted_score_gap':result['weighted_score_gap']}


def compose_ancestors(previous_selected, earlier_selected):
    p,e=np.asarray(previous_selected),np.asarray(earlier_selected)
    if p.ndim!=1 or e.shape!=p.shape or not np.issubdtype(p.dtype,np.integer) or not np.issubdtype(e.dtype,np.integer):
        raise ValueError('Equal integer selection arrays required')
    if np.any((p<0)|(p>=len(e))) or np.any((e<0)|(e>=len(e))):raise ValueError('Selection index outside population')
    return e[p]


def alias_diagnostics(endpoint, score, parents):
    """Exact ranges expose identical copies; float variance can create roundoff."""
    y,s,p=np.asarray(endpoint,float),np.asarray(score,float),np.asarray(parents)
    ranges=[];score_ranges=[];coordinate_variance=[];score_variance=[];copies=0
    for family in np.unique(p):
        ids=p==family
        if ids.sum()<2:continue
        copies+=1;z=y[ids];v=s[ids]
        ranges.append(float(np.ptp(z,axis=0).max()));score_ranges.append(float(np.ptp(v)))
        coordinate_variance.append(float(np.var(z,axis=0).max()));score_variance.append(float(np.var(v)))
    return {'immediate_branching_parent_count':copies,
            'immediate_coordinate_range_max_A':max(ranges,default=0.),
            'immediate_score_range_max':max(score_ranges,default=0.),
            'immediate_coordinate_variance_max_A2':max(coordinate_variance,default=0.),
            'immediate_score_variance_max':max(score_variance,default=0.)}


def chemistry_nuisance(atom, bond, ancestor_atom, ancestor_bond, ancestors):
    a,b=np.asarray(atom),np.asarray(bond);g=np.asarray(ancestors,int)
    aa,bb=np.asarray(ancestor_atom)[g],np.asarray(ancestor_bond)[g]
    if a.shape!=aa.shape or b.shape!=bb.shape or b.shape[1:]!=(a.shape[1],a.shape[1]):raise ValueError('Fixed slot chemistry required')
    edge=np.triu(np.ones(b.shape[1:],bool),1)
    return np.column_stack([(a!=aa).mean(1),(b[:,edge]!=bb[:,edge]).mean(1)])


def conditional_moments(features, gain, nuisance, parents, ancestors):
    """Equal common-ancestor conditional covariance and pooled correlation.

    Average copies inside one immediate-parent branch first. Then center each
    branching ancestor and give it total mass 1/G. Weighted least squares removes
    two centered diagnostic chemistry variables; this is not a new predictor.
    """
    x,y,z=np.asarray(features,float),np.asarray(gain,float),np.asarray(nuisance,float)
    p,g=np.asarray(parents,int),np.asarray(ancestors,int)
    if x.ndim!=2 or y.shape!=(len(x),) or z.shape!=(len(x),2) or p.shape!=y.shape or g.shape!=y.shape:
        raise ValueError('Aligned finite feature, score, chemistry and ancestry rows required')
    if not all(np.isfinite(v).all() for v in (x,y,z)):raise ValueError('Finite measured branch data required')
    branches=[]
    for parent in np.unique(p):
        ids=p==parent
        if np.unique(g[ids]).size!=1:raise ValueError('Immediate parent has inconsistent common ancestor')
        # Exact duplicate forecasts take the first row. Averaging repeated float64
        # values can otherwise introduce count-dependent arithmetic drift.
        average=lambda values:values[0].copy() if np.array_equal(values,np.broadcast_to(values[0],values.shape)) else values.mean(0)
        branches.append((int(g[ids][0]),average(x[ids]),float(average(y[ids])),average(z[ids])))
    bx=[];by=[];bz=[];bw=[];ancestor_count=0;counts=[]
    for ancestor in sorted({item[0] for item in branches}):
        children=[item for item in branches if item[0]==ancestor]
        if len(children)<2:continue
        xx=np.array([v[1] for v in children]);yy=np.array([v[2] for v in children]);zz=np.array([v[3] for v in children])
        centered_x=xx-xx.mean(0);centered_x[:,np.ptp(xx,axis=0)==0]=0.
        centered_z=zz-zz.mean(0);centered_z[:,np.ptp(zz,axis=0)==0]=0.
        centered_y=yy-yy.mean() if np.ptp(yy)>0 else np.zeros_like(yy)
        bx.extend(centered_x);by.extend(centered_y);bz.extend(centered_z);bw.extend([1/len(children)]*len(children))
        counts.append(len(children));ancestor_count+=1
    missing=np.full(x.shape[1],np.nan)
    if not ancestor_count:
        return {k:missing.copy() for k in ('raw_covariance','adjusted_covariance','raw_correlation','adjusted_correlation')}, {
            'lag2_branching_ancestor_count':0,'lag2_distinct_parent_branches':0,'chemistry_design_rank':0,
            'lag2_score_variance':np.nan,'mean_atom_change_fraction':np.nan,'mean_bond_change_fraction':np.nan}
    bx,by,bz,bw=np.asarray(bx),np.asarray(by),np.asarray(bz),np.asarray(bw)/ancestor_count
    def moments(xx,yy):
        cov=np.einsum('n,nf,n->f',bw,xx,yy);vx=np.einsum('n,nf,nf->f',bw,xx,xx);vy=float(np.dot(bw,yy*yy))
        den=np.sqrt(vx*vy);corr=np.divide(cov,den,out=missing.copy(),where=den>1e-14)
        return cov,corr,vy
    raw_cov,raw_corr,score_var=moments(bx,by)
    # Same-ancestor centering has already removed intercepts. Rank-deficient or
    # all-zero chemistry controls are handled by the deterministic pseudoinverse.
    weighted=bz*np.sqrt(bw[:,None]);pinv=np.linalg.pinv(weighted,rcond=1e-12)
    rx=bx-bz@(pinv@(bx*np.sqrt(bw[:,None])));ry=by-bz@(pinv@(by*np.sqrt(bw)))
    adjusted_cov,adjusted_corr,_=moments(rx,ry)
    diag={'lag2_branching_ancestor_count':ancestor_count,'lag2_distinct_parent_branches':len(bx),
          'chemistry_design_rank':int(np.linalg.matrix_rank(weighted,tol=1e-12)),
          'lag2_score_variance':score_var,
          'mean_atom_change_fraction':float(np.mean([v[3][0] for v in branches])),
          'mean_bond_change_fraction':float(np.mean([v[3][1] for v in branches]))}
    return {'raw_covariance':raw_cov,'adjusted_covariance':adjusted_cov,
            'raw_correlation':raw_corr,'adjusted_correlation':adjusted_corr},diag


def summarize(curves,grid,features,output,seed):
    rows=[];functions={};batch_rows=[];rng=np.random.default_rng(seed)
    metrics=list(curves)
    for metric in metrics:
        values=np.asarray(curves[metric]);integrated=observed_integral(values,grid)
        for batch in range(len(values)):
            for feature,value in zip(features,integrated[batch]):batch_rows.append({'batch_index':batch,'metric':metric,'feature':feature,'whole_window_value':float(value)})
        boot=integrated[rng.integers(0,len(values),(2000,len(values)))].mean(1)
        for j,feature in enumerate(features):
            finite=np.isfinite(integrated[:,j]);n=int(finite.sum())
            if n>2 and np.ptp(integrated[finite,j])>1e-15:p=float(ttest_1samp(integrated[finite,j],0).pvalue)
            elif n>2 and np.all(integrated[finite,j]==0):p=1.
            else:p=np.nan
            effect=float(np.nanmean(integrated[:,j])) if n else np.nan
            ci=np.nanquantile(boot[:,j],[.025,.975]) if np.isfinite(boot[:,j]).any() else [np.nan,np.nan]
            rows.append({'id':'branch_mutation/'+metric+'/'+feature,'metric':metric,'feature':feature,
              'whole_window_mean':effect,'CI_low':float(ci[0]),'CI_high':float(ci[1]),'p':p,
              'same_sign_batch_fraction':float(np.mean(np.sign(integrated[finite,j])==np.sign(effect))) if n else np.nan,
              'independent_batches_available':n,'observed_event_fraction':float(np.isfinite(values[:,:,j]).mean())})
            complete=np.isfinite(values[:,:,j]).all(0)
            if complete.sum()<5:continue
            fit=masked_curve_fit(grid[complete],values[:,complete,j],max_degree=3)
            fit['analytic_derivative_legendre_coefficients']=(legder(np.array(fit['legendre_coefficients']))*2/(fit['time_end']-fit['time_start'])).tolist()
            functions[metric+'/'+feature]=fit
    # A single correction family covers every tested metric and geometric feature.
    q=bh(np.array([v['p'] for v in rows]))
    for row,value in zip(rows,q):row['q']=float(value)
    pd.DataFrame(rows).to_parquet(output/'whole_window_effects.parquet',compression=None,index=False)
    pd.DataFrame(batch_rows).to_parquet(output/'batch_window_statistics.parquet',compression=None,index=False)
    write_json(output/'fitted_trends.json',functions)
    return rows


def mine(dataset,campaign,reference,output,seed=42,augmented_reference=None):
    root,refpath,out=Path(dataset).resolve(),Path(reference).resolve(),Path(output).resolve()
    if out.exists():raise FileExistsError(out)
    ref=json.loads(gzip.decompress(refpath.read_bytes()));grid=np.asarray(ref['times'],float)
    if ref['schema_version']!='affinity-endpoint-library-1.0' or len(grid)<7:raise ValueError('Frozen full-window endpoint library required')
    batches=list(ref['discovery_batches']);source=root/'results'/campaign;cfg=read_json(source/'config.json')
    paths={int(p.parent.name.split('_')[1]):p for p in trajectory_paths(source) if p.parent.parent.name=='single'}
    points,origin=np.asarray(ref['landmarks_A'],float),np.asarray(ref['origin_A'],float)
    features=names(points)+['joint_innovation_RMS_A','joint_translation_RMS_A','joint_internal_RMS_A']
    if len(batches)<3:raise ValueError('At least three independent batches required')
    curves={key:[] for key in ('raw_covariance','adjusted_covariance','raw_correlation','adjusted_correlation')}
    diagnostics=[];sources=[];maxedge=0.;teacher_fields={};nulls={'equal_score_zero':0,'identical_cloud_zero':0}
    for batch in batches:
        path=paths[int(batch)];com=np.asarray(read_json(source/f'frame_batch_{batch:03d}.json')['target_com'])[:,None,:]
        with open_trajectory(path) as tr:
            t=np.round(np.asarray(tr['score_time'],float)[:,0],6);state=np.asarray(tr['state_time'],float)
            ids=np.flatnonzero(np.asarray(tr['resampled'],bool)&(t>=ref['window'][0]-2e-6)&(state<=ref['window'][1]+2e-6))
            if not np.array_equal(t[ids],grid) or np.any(np.diff(ids)!=1):raise ValueError('Exact learned contiguous support required')
            if not np.isclose(state[ids[-1]],ref['window'][1],atol=2e-6):raise ValueError('Missing final actual selection state')
            if not np.asarray(tr['mask'])[ids].all():raise ValueError('Fixed active physical slots required')
            endpoint=np.asarray(tr['predicted_coords'],float)[ids]*cfg['coord_scale']+com
            current=np.asarray(tr['current_coords'],float)[ids]*cfg['coord_scale']+com
            proposal=np.asarray(tr['proposal_coords'],float)[ids]*cfg['coord_scale']+com
            score=np.asarray(tr['pic50_on'],float)[ids];selected=np.asarray(tr['selected_indices'])[ids]
            atom=np.asarray(tr['predicted_atomics'])[ids];bond=np.asarray(tr['predicted_bonds'])[ids]
            geometry=np.array([numpy_geometry(y,points,origin) for y in endpoint]);history={key:[] for key in curves}
            for k in range(1,len(grid)):
                edge=float(np.abs(current[k]-proposal[k-1,selected[k-1]]).max());maxedge=max(edge,maxedge)
                if edge>2e-6:raise ValueError('Actual parent-to-current transport is inconsistent')
                alias=alias_diagnostics(endpoint[k],score[k],selected[k-1])
                record={'batch':int(batch),'score_time':float(grid[k]),'state_time':float(state[ids[k]]),**alias,'lag2_observed':k>=2}
                if k>=2:
                    parent=selected[k-1];ancestor=compose_ancestors(parent,selected[k-2])
                    delta=endpoint[k]-endpoint[k-2,ancestor];center=delta.mean(1);internal=delta-center[:,None]
                    amplitudes=np.column_stack([np.sqrt((delta*delta).sum(-1).mean(-1)),
                              np.sqrt((center*center).sum(-1)),np.sqrt((internal*internal).sum(-1).mean(-1))])
                    innovation=np.column_stack([geometry[k]-geometry[k-2,ancestor],amplitudes])
                    gain=score[k]-score[k-2,ancestor]
                    nuisance=chemistry_nuisance(atom[k],bond[k],atom[k-2],bond[k-2],ancestor)
                    moments,extra=conditional_moments(innovation,gain,nuisance,parent,ancestor)
                    record.update(extra)
                    for key in history:history[key].append(moments[key])
                    if augmented_reference is not None:
                        frame=ref['frames'][k]
                        for teacher in np.flatnonzero(np.asarray(frame['teacher_batches'])==batch):
                            anchor=np.asarray(frame['teacher_endpoint_A'][teacher],float);target_score=float(frame['teacher_scores'][teacher])
                            options=[]
                            for slot in np.flatnonzero(np.abs(score[k]-target_score)<=1e-6):
                                aligned,_=match_cloud(anchor,endpoint[k,slot]);options.append((rms(anchor-aligned),int(slot)))
                            if not options:raise ValueError('Frozen teacher score absent')
                            error,slot=min(options)
                            if error>1e-5:raise ValueError('Frozen teacher geometry differs from original source')
                            tolerance=2*alias['immediate_score_range_max']+1e-8
                            field=conditional_teacher_contrast(anchor,target_score,endpoint[k],score[k],parent,ancestor,
                                int(parent[slot]),int(ancestor[slot]),tolerance)
                            tied=conditional_teacher_contrast(anchor,1.,endpoint[k],np.ones_like(score[k]),parent,ancestor,int(parent[slot]),int(ancestor[slot]))
                            same=conditional_teacher_contrast(anchor,target_score,np.repeat(anchor[None],len(parent),0),score[k],parent,ancestor,int(parent[slot]),int(ancestor[slot]),tolerance)
                            if np.any(tied['direction_unit']) or tied['confidence']!=0:raise AssertionError('Equal-score teacher null is nonzero')
                            if np.any(same['direction_unit']) or same['confidence']!=0:raise AssertionError('Identical-cloud teacher null is nonzero')
                            nulls['equal_score_zero']+=1;nulls['identical_cloud_zero']+=1
                            teacher_fields[k,int(teacher)]={'field':field,'provenance':{'batch':int(batch),'score_time':float(grid[k]),
                              'teacher_candidate_slot':slot,'common_grandparent_slot':int(ancestor[slot]),'immediate_parent_slot':int(parent[slot]),
                              'teacher_equivalence_RMS_A':error,'head_alias_tolerance':tolerance,
                              'lower_immediate_parent_slots':field['lower_parent_ids'],'distinct_observed_mutations':field['distinct_observed_mutations'],
                              'raw_direction_RMS_A':field['raw_rms_A'],'noise_RMS_A':field['noise_rms_A'],
                              'score_gap':field['weighted_score_gap'],'local_confidence':field['confidence'],
                              'root_identity_used_as_support_gate':False}}
                diagnostics.append(record)
            for key in curves:curves[key].append(history[key])
        sources.append({'batch':int(batch),'path':path.relative_to(root).as_posix(),'sha256':digest(path)})
        print(f'branch mutation batch {batch}: {len(grid)-2} lag-two events',flush=True)
    out.mkdir(parents=True);effects=summarize(curves,grid[2:],features,out,seed)
    pd.DataFrame(diagnostics).to_parquet(out/'ancestry_diagnostics.parquet',compression=None,index=False)
    write_json(out/'feature_catalog.json',{'features':features,'landmarks_A':points.tolist(),'origin_A':origin.tolist(),
       'representation':'Recorded predicted endpoint world Å, physical slot ancestry; geometric features permutation invariant.',
       'units':'Centroids/softmin/joint amplitudes Å; shape second moments Å²; occupancy and pair kernels dimensionless.',
       'nuisance':['Changed atom-label fraction','Changed unordered bond-label fraction'],
       'nuisance_role':'Only diagnostic adjustment, never an atomic-type optimization or chemical-graph veto.'})
    d=pd.DataFrame(diagnostics);obs=d[d.lag2_observed]
    salient=[v for v in effects if v['metric']=='adjusted_correlation' and np.isfinite(v['q'])]
    salient.sort(key=lambda v:(v['q'],-abs(v['whole_window_mean']),v['feature']))
    evidence={'schema_version':'branch-mutation-evidence-1.0','window':ref['window'],'observed_times':grid[2:].tolist(),
       'missing_first_two_lag2_times':grid[:2].tolist(),'independent_batches':len(batches),
       'evidence_items':salient[:16],'all_effects_path':'whole_window_effects.parquet',
       'immediate_alias':{'maximum_coordinate_range_A':float(d.immediate_coordinate_range_max_A.max()),
         'maximum_score_range':float(d.immediate_score_range_max.max()),
         'maximum_coordinate_variance_A2':float(d.immediate_coordinate_variance_max_A2.max()),
         'maximum_score_variance':float(d.immediate_score_variance_max.max())},
       'lag2_support':{'events':len(obs),'mean_branching_ancestors':float(obs.lag2_branching_ancestor_count.mean()),
          'minimum_branching_ancestors':int(obs.lag2_branching_ancestor_count.min()),
          'mean_distinct_branches':float(obs.lag2_distinct_parent_branches.mean()),
          'mean_conditional_score_variance':float(obs.lag2_score_variance.mean())},
       'q_below_005_by_metric':{key:sum(v['q']<.05 for v in effects if v['metric']==key) for key in curves},
       'limitations':['Current immediate-sibling forecasts may be structurally identical because mutation occurs after forecast.',
          'Lag-two branches have two previous selection events in their history: mutation is observed conditional on survival, not randomized treatment.',
          'Extinct-branch future utility remains unknown. Online head labels share the selector; no terminal labels are used.',
          'Chemical nuisance adjustment is observational and can remove mediator signal; raw and adjusted results must both be compared.',
          'No affinity surrogate, neural model, physical energy or module causality is fitted. Time derivatives are not spatial forces.']}
    write_json(out/'evidence.json',evidence)
    if augmented_reference is not None:
        path=Path(augmented_reference).resolve()
        if path.exists():raise FileExistsError(path)
        # Augment in a separate reference artifact. Summary storage is capped
        # independently; the frozen full teacher library is several MiB itself.
        original_frames=json.loads(gzip.decompress(refpath.read_bytes()))['frames']
        confidence_all=[];bytime=[]
        for k,frame in enumerate(ref['frames']):
            clouds=np.asarray(frame['teacher_endpoint_A'],float);batchlabels=np.asarray(frame['teacher_batches'])
            directions=[];confidence=[];weights=[];provenance=[]
            for j,anchor in enumerate(clouds):
                if k<2:
                    directions.append(np.zeros_like(anchor).tolist());confidence.append(0.);weights.append(np.ones(len(anchor)).tolist())
                    provenance.append({'lag2_observed':False,'reason':'No two-step ancestry inside learned window'});continue
                record=teacher_fields[k,j];field=record['field'];direction=field['direction_unit'];alignment=[]
                for donor in sorted(set(batchlabels)-{batchlabels[j]}):
                    options=[]
                    for other in np.flatnonzero(batchlabels==donor):
                        aligned,order=match_cloud(anchor,clouds[other]);options.append((rms(anchor-aligned),int(other),order))
                    _,other,order=min(options,key=lambda v:(v[0],v[1]))
                    alignment.append(cosine(direction,teacher_fields[k,other]['field']['direction_unit'][order]))
                mean_alignment=float(np.mean(alignment));fraction=float(np.mean(np.asarray(alignment)>0))
                c=field['confidence']*max(mean_alignment,0)*fraction
                directions.append(direction.tolist());confidence.append(c);weights.append(field['atom_weight'].tolist())
                provenance.append({**record['provenance'],'lag2_observed':True,'cross_batch_mean_cosine':mean_alignment,
                                  'cross_batch_positive_fraction':fraction,'independent_other_batches':len(alignment)})
            frame.update(teacher_contrast_direction_unit=directions,teacher_contrast_confidence=confidence,
                         teacher_contrast_atom_weight=weights,teacher_contrast_provenance=provenance)
            if any(frame[key]!=original_frames[k][key] for key in original_frames[k]):raise AssertionError('Frozen original teacher field changed')
            confidence_all.extend(confidence);bytime.append({'time':float(grid[k]),'mean_confidence':float(np.mean(confidence)),
                'nonzero_fraction':float(np.mean(np.asarray(confidence)>0))})
        ref['branch_mutation']={'schema_version':'conditional-branch-teacher-field-1.0','parent_reference_sha256':digest(refpath),
          'scope':ref['window'],'first_two_missing':True,'conditioning':'Same observed grandparent, distinct immediate-parent mutation proposals, current duplicate forecasts collapsed.',
          'support':'Observed conditional mutations, never initial root count. Independent statistical replicates remain generation batches.',
          'confidence':'0.15*(mutations−1)/(mutations+4)*coherence*amplitude/(amplitude+noise+0.02A), then positive template-aligned cross-batch mean cosine times positive fraction.',
          'limitations':['Branches are observed after selection: conditional association, not independent randomized mutation treatment.',
                         'Typed chemical changes adjusted only in summary correlations; teacher directions do not claim causal attribution.'],
          'exact_nulls':nulls}
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(gzip.compress(json.dumps(ref,separators=(',',':'),allow_nan=False).encode(),mtime=0))
        write_json(out/'teacher_field_summary.json',{'reference_sha256':digest(path),'reference_bytes':path.stat().st_size,
          'teacher_records':len(confidence_all),'nonzero_fraction':float(np.mean(np.asarray(confidence_all)>0)),
          'maximum_confidence':float(np.max(confidence_all)),'mean_confidence':float(np.mean(confidence_all)),
          'bytime':bytime,'exact_nulls':nulls,'original_teacher_coordinates_scores_batches_unchanged':True})
    manifest={'schema_version':'branch-mutation-mining-1.0','reference_sha256':digest(refpath),'source_code_sha256':digest(__file__),
      'sources':sources,'discovery_batches':batches,'window':ref['window'],'score_times':grid.tolist(),
      'lag2_observed_times':grid[2:].tolist(),'first_two_missing':True,'maximum_parent_transport_error_A':maxedge,
      'feature_count':len(features),'independent_batches':len(batches),'seed':seed,
      'statistics':'Collapse immediate-parent copies, center each common grandparent, give each observed branching ancestor equal total mass. Weighted nuisance projection; batch bootstrap; one BH family across all four metrics.',
      'storage':'Batch-window effects, compact ancestry diagnostics, global functions. No per-particle, per-edge or wide event-feature cache. Optional full augmented frozen teacher reference is budgeted separately.',
      'files':{p.name:{'sha256':digest(p),'bytes':p.stat().st_size} for p in sorted(out.iterdir()) if p.is_file()}}
    write_json(out/'manifest.json',manifest)
    size=sum(p.stat().st_size for p in out.iterdir() if p.is_file() and (augmented_reference is None or p.resolve()!=Path(augmented_reference).resolve()))
    if size>1024*1024:raise ValueError(f'Compact output exceeded declared 1MiB budget: {size}')
    return manifest
