"""Independent trajectory imitation, activity and final-outcome evaluation."""
import gzip
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.special import expit, logsumexp
from scipy.integrate import trapezoid
import torch
from ..io import read_json, write_json, write_table, digest
from ..trajectory_source import trajectory_paths, open_trajectory
from .comparison import summarize_records, paired_effect
from .multistage_reward import MultistageReward
from .prototypes import measure_patch


def control_rows(campaign):
    rows=[]
    for path in sorted(Path(campaign).glob('*/batch_*/guidance_trace.jsonl')):
        arm=path.parent.parent.name;batch=int(path.parent.name.split('_')[1])
        records=[json.loads(s) for s in path.read_text().splitlines()]
        if [r['step'] for r in records] != list(range(100)):
            raise ValueError('Incomplete control trace')
        with open_trajectory(path.parent/'trajectory.h5') as trajectory:
            atom_counts=trajectory['mask'].astype(bool).sum(-1)
        cumulative=np.zeros(atom_counts.shape[1])
        for record in records:
            n=len(record['injection_max_atom_A'])
            if n!=atom_counts.shape[1]:raise ValueError('Telemetry particle count mismatch')
            rms=np.array(record['injection_l2_A'])/np.sqrt(atom_counts[record['step']])
            cumulative+=rms
            for slot in range(n):
                r={'arm':arm,'batch':batch,'slot':slot,'step':record['step'],'time':round(record['score_time'],6),
                   'evaluated':record['reward_evaluated'],'active':record['active']}
                for k in ['injection_max_atom_A','native_rms_A','injection_rms_A','requested_rms_A',
                          'gradient_norm','gradient_rms_native','raw_gradient_norm','cap_factor','backtrack_factor','geometry_accepted',
                          'applied_native_rms_ratio','cumulative_injection_rms_A','gradient_native_cosine',
                          'observable_available','responsibility_ess','responsibility_entropy','reward']:
                    v=record.get(k)
                    r[k]=v[slot] if isinstance(v,list) else None
                if r['native_rms_A'] is None:
                    r['native_rms_A']=record['native_l2_A'][slot]/np.sqrt(atom_counts[record['step'],slot])
                if r['injection_rms_A'] is None:r['injection_rms_A']=rms[slot]
                if r['cumulative_injection_rms_A'] is None:r['cumulative_injection_rms_A']=cumulative[slot]
                if r['applied_native_rms_ratio'] is None:
                    r['applied_native_rms_ratio']=r['injection_rms_A']/max(r['native_rms_A'],1e-30)
                rows.append(r)
    return pd.DataFrame(rows)


def summarize_control(rows):
    summaries=[]
    for arm,g in rows.groupby('arm',sort=True):
        available=g[g.observable_available.eq(True)&g.active.eq(True)&g.requested_rms_A.gt(1e-12)&g.gradient_rms_native.gt(1e-8)]
        efficiency=(available.injection_rms_A/available.requested_rms_A)
        last=g[g.time.gt(.5)]
        summaries.append({'arm':arm,'particle_steps':len(g),'reward_evaluated_fraction':float(g.evaluated.mean()),
            'available_active_particle_steps':len(available),
            'evaluated_steps_per_batch_min':int(g.groupby('batch').apply(lambda b:b.loc[b.evaluated,'step'].nunique(),include_groups=False).min()),
            'nonzero_injection_steps_per_batch_min':int(g.groupby('batch').apply(lambda b:b.loc[b.injection_max_atom_A.gt(0),'step'].nunique(),include_groups=False).min()),
            'late_nonzero_particle_fraction':float(last.injection_max_atom_A.gt(0).mean()),
            'median_applied_over_requested':float(efficiency.median()) if len(efficiency) else None,
            'geometry_rejection_fraction':float(available.geometry_accepted.eq(False).mean()) if len(available) else None,
            'backtrack_fraction':float(available.backtrack_factor.lt(1).mean()) if len(available) else None,
            'cap_fraction':float(available.cap_factor.lt(.999999).mean()) if len(available) else None,
            'maximum_atom_injection_A':float(g.injection_max_atom_A.max()),
            'mean_cumulative_rms_A':float(g[g.step.eq(99)].cumulative_injection_rms_A.mean())
                if g.cumulative_injection_rms_A.notna().any() else None,
            'median_applied_native_ratio':float(available.applied_native_rms_ratio.median()) if len(available) else None,
            'native_conflict_fraction':float(available.gradient_native_cosine.lt(0).mean()) if len(available) else None})
    return pd.DataFrame(summaries)


def calibrate(campaign, output):
    campaign,output=Path(campaign),Path(output)
    if not (campaign/'COMPLETE.json').exists(): raise ValueError('Pilot incomplete')
    records=read_json(campaign/'final_records.json');d=pd.DataFrame(records)
    expected={'unguided','gradient_zero','multistage_r005','multistage_r015','multistage_r030'}
    if set(d.arm)!=expected or d.duplicated(['arm','batch','slot']).any() or not d.groupby('arm').size().eq(12).all():
        raise ValueError('Pilot final candidates do not match the frozen five-arm, 12-per-arm design')
    control=summarize_control(control_rows(campaign))
    baseline=d[d.arm.eq('unguided')]
    if len(baseline)!=12: raise ValueError('Predeclared calibration uses 12 pilot candidates')
    base_valid=int((baseline.build_success.eq(True)&baseline.connected.eq(True)).sum())
    base_clashes=int((baseline.pairs_below_1_2A.fillna(0)>0).sum())
    choices=[]
    for arm,ratio in [('multistage_r005',.05),('multistage_r015',.15),('multistage_r030',.3)]:
        c=control[control.arm.eq(arm)].iloc[0].to_dict();g=d[d.arm.eq(arm)]
        valid=int((g.build_success.eq(True)&g.connected.eq(True)).sum())
        clashes=int((g.pairs_below_1_2A.fillna(0)>0).sum())
        eligible=(c['available_active_particle_steps']>0 and c['median_applied_over_requested']>=.5 and
                  c['geometry_rejection_fraction']<=.05 and valid>=base_valid-1 and clashes<=base_clashes)
        choices.append({'arm':arm,'ratio':ratio,'eligible':bool(eligible),'valid_connected':valid,
                        'final_clash_candidates':clashes,**{k:v for k,v in c.items() if k!='arm'}})
    feasible=[c for c in choices if c['eligible']]
    selected=max(feasible,key=lambda c:c['ratio']) if feasible else None
    result={'status':'selected' if selected else 'no_feasible_ratio','selected_ratio':selected['ratio'] if selected else None,
        'rule':'Largest ratio meeting median applied/requested>=.5, geometry rejection<=5%, valid connected>=baseline-1, final severe clash count<=baseline',
        'uses_affinity_for_selection':False,'pilot_n':12,'baseline_valid_connected':base_valid,
        'baseline_clash_candidates':base_clashes,'choices':choices,
        'clash_count_scope':'Measured built molecules only; missing final geometry is not certified clash-free. All raw final structures are evaluated separately.',
        'source_program_sha256':digest(campaign/'reward_program.json'),
        'limitation':'Engineering feasibility in a small pilot, not optimization or statistical validation'}
    write_json(output/'calibration.json',result);write_table(output/'control_summary.csv',control)
    return result


def independent_observables(x, atoms, mask, catalog):
    """Nonreward observables diagnose shortcuts in the two-distance imitation."""
    mask=mask.astype(bool);n=mask.sum(1)
    center=np.where(mask[...,None],x,0).sum(1)/n[:,None]
    rg=np.sqrt(np.where(mask,((x-center[:,None])**2).sum(-1),0).sum(1)/n)
    result={'radius_gyration_A':rg}
    for region,r in catalog['regions'].items():
        points=np.array(r['points_A']);distance=np.linalg.norm(x[:,:,None]-points[None,None],axis=-1)
        contacts=expit((4.5-distance.min(-1))/.5)*mask
        centroid=(x*contacts[...,None]).sum(1)/np.maximum(contacts.sum(1)[:,None],1e-30)-points.mean(0)
        for k,axis in enumerate('xyz'):result[region+'::centroid_'+axis]=centroid[:,k]
        result[region+'::all_atom_softmin']=-.25*(logsumexp(np.where(mask[...,None],-distance/.25,-np.inf),axis=(1,2))-np.log(n*len(points)))
    hetero=np.isin(atoms,[catalog['atom_vocabulary'][e] for e in ['N','O','S']])
    result['hetero_atom_fraction']=(hetero*mask).sum(1)/n
    return result


def energy_distance_squared(x,y):
    """Biased empirical energy statistic on feature samples; zero for identical multisets."""
    if not len(x) or not len(y): return np.nan
    return float(max(0,2*cdist(x,y).mean()-cdist(x,x).mean()-cdist(y,y).mean()))


def nearest_pose_chamfer(x, mask, reference, reference_mask):
    """Per-candidate nearest SMC shape in the fixed receptor frame, in Angstrom.

    Symmetric mean nearest-heavy-atom distance; no Kabsch rotation or invented
    atom correspondence. This shape metric does not measure chemical identity.
    """
    mask=mask.astype(bool);reference_mask=reference_mask.astype(bool)
    valid=mask.any(1);valid_ref=reference_mask.any(1)
    result=np.full(len(x),np.nan)
    if not valid.any() or not valid_ref.any():return result
    xx=x[valid];rr=reference[valid_ref];mm=mask[valid];rm=reference_mask[valid_ref]
    d=np.linalg.norm(xx[:,None,:,None,:]-rr[None,:,None,:,:],axis=-1)
    forward=np.where(rm[None,:,None,:],d,np.inf).min(-1)
    reverse=np.where(mm[:,None,:,None],d,np.inf).min(-2)
    distances=.5*((forward*mm[:,None,:]).sum(-1)/mm.sum(1)[:,None]+
                   (reverse*rm[None,:,:]).sum(-1)/rm.sum(1)[None,:])
    result[valid]=distances.min(1)
    return result


def heavy_mask(atomics, mask, catalog):
    excluded=[catalog['atom_vocabulary'].get(e,-1) for e in ('H','<PAD>')]
    return mask.astype(bool) & ~np.isin(atomics,excluded)


def final_fingerprint_similarity(records):
    from rdkit import Chem,DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    generator=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
    groups={}
    for row in records:
        if row['build_success'] and row.get('smiles'):
            molecule=Chem.MolFromSmiles(row['smiles'])
            if molecule is not None:
                groups.setdefault((row['arm'],row['batch']),[]).append(generator.GetFingerprint(molecule))
    results=[]
    for (arm,batch),fingerprints in sorted(groups.items()):
        if arm=='single' or ('single',batch) not in groups:continue
        reference=groups['single',batch]
        scores=[max(DataStructs.BulkTanimotoSimilarity(fp,reference)) for fp in fingerprints]
        results.append({'arm':arm,'batch':batch,'valid_queried_n':len(scores),'valid_smc_reference_n':len(reference),
                        'mean_nearest_smc_morgan_tanimoto':float(np.mean(scores))})
    return results


def evaluate(campaign, output):
    campaign,output=Path(campaign),Path(output)
    if not (campaign/'COMPLETE.json').exists(): raise ValueError('Campaign incomplete')
    program=read_json(campaign/'reward_program.json');catalog=read_json(campaign/'reward_catalog.json')
    reward=MultistageReward(program,catalog);cfg=read_json(campaign/'config.json');scale=cfg['coord_scale']
    records=read_json(campaign/'final_records.json')
    batches,summary,paired=summarize_records(records,'unguided')
    for name,table in [('outcome_batches',batches),('outcome_summary',summary),('paired_outcomes',paired)]:
        write_table(output/(name+'.csv'),table)
    controls=control_rows(campaign);write_table(output/'control_summary.csv',summarize_control(controls))
    write_table(output/'control_particle_steps.parquet',controls)
    distributions={};structures={};feature_rows=[];schedule=[];raw_final_rows=[]
    for path in trajectory_paths(campaign):
        arm=path.parent.parent.name;batch=int(path.parent.name.split('_')[1])
        frame=np.array(read_json(campaign/f'frame_batch_{batch:03d}.json')['target_com'])
        with open_trajectory(path) as z:
            times=np.round(z['score_time'][:,0].astype(float),6)
            x=z['predicted_coords'].astype(float)*scale+frame[None,:,None,:]
            atoms=z['predicted_atomics'];mask=z['mask']
            selected=np.flatnonzero(z['resampled']).tolist()
            if (arm=='single' and selected!=list(range(51))) or (arm!='single' and selected):
                raise ValueError('Unexpected resampling schedule')
            schedule.append({'arm':arm,'batch':batch,'resampling_events':len(selected)})
            for step,time in enumerate(times):
                value=measure_patch(x[step],atoms[step],mask[step],catalog)
                distributions[arm,batch,step]=value
                structures[arm,batch,step]=(x[step],heavy_mask(atoms[step],mask[step],catalog))
                valid=np.isfinite(value).all(1)
                with torch.no_grad():
                    r,_=reward.from_features(torch.tensor(np.nan_to_num(value)),torch.tensor(valid),time)
                row={'arm':arm,'batch':batch,'step':step,'time':time,'feature_available_n':int(valid.sum()),
                     'multistage_reward':float(r[valid].mean()) if valid.any() else np.nan}
                row.update({f:float(np.nanmean(value[:,k])) for k,f in enumerate(reward.features)})
                row.update({k:float(np.nanmean(v)) for k,v in independent_observables(x[step],atoms[step],mask[step],catalog).items()})
                feature_rows.append(row)
        with gzip.open(path.parent/'final_prediction.pt.gz','rb') as f:
            final=torch.load(f,map_location='cpu',weights_only=False)
        fx=final['coords'].numpy();fa=final['atomics'].argmax(-1).numpy();fm=final['mask'].numpy()
        value=measure_patch(fx,fa,fm,catalog);distributions[arm,batch,100]=value
        hm=heavy_mask(fa,fm,catalog)
        structures[arm,batch,100]=(fx,hm)
        with gzip.open(path.parent/'initial_state.pt.gz','rb') as f:
            initial=torch.load(f,map_location='cpu',weights_only=False)
        pocket=initial['pocket_target'];pc=pocket['coords'].numpy()*scale+frame[:,None,:]
        distances=np.linalg.norm(fx[:,:,None,:]-pc[:,None,:,:],axis=-1)
        pm=hm[:,:,None]&pocket['mask'].numpy()[:,None,:].astype(bool)
        for slot in range(len(fx)):
            selected=distances[slot][pm[slot]]
            raw_final_rows.append({'arm':arm,'batch':batch,'slot':slot,'raw_heavy_atom_count':int(hm[slot].sum()),
                'geometry_measured':bool(len(selected)),
                'min_pocket_distance_A':float(selected.min()) if len(selected) else None,
                'raw_heavy_pairs_below_1_2A':int((selected<1.2).sum()) if len(selected) else None})
        row={'arm':arm,'batch':batch,'step':100,'time':1.,'feature_available_n':int(np.isfinite(value).all(1).sum())}
        row.update({f:float(np.nanmean(value[:,k])) for k,f in enumerate(reward.features)})
        row.update({k:float(np.nanmean(v)) for k,v in independent_observables(fx,fa,fm,catalog).items()})
        feature_rows.append(row)
    write_table(output/'feature_curve_batches.parquet',feature_rows);write_table(output/'resampling_schedule.csv',schedule)
    write_table(output/'raw_final_geometry.csv',raw_final_rows)
    distance_rows=[]
    for (arm,batch,step),values in sorted(distributions.items()):
        if arm=='single' or ('single',batch,step) not in distributions:continue
        reference=distributions['single',batch,step]
        finite=values[np.isfinite(values).all(1)];reference=reference[np.isfinite(reference).all(1)]
        _,cov=reward.references(step/100)
        whiten=np.linalg.inv(np.linalg.cholesky(cov)).T
        shape=nearest_pose_chamfer(*structures[arm,batch,step],*structures['single',batch,step])
        distance_rows.append({'arm':arm,'batch':batch,'step':step,'time':step/100,
            'energy_squared_to_smc':energy_distance_squared(finite@whiten,reference@whiten),
            'nearest_smc_pose_chamfer_A':float(np.nanmean(shape)) if np.isfinite(shape).any() else np.nan,
            'shape_query_coverage_n':int(np.isfinite(shape).sum()),
            'n':len(finite),'smc_n':len(reference)})
    distance=pd.DataFrame(distance_rows)
    if len(distance):
        write_table(output/'distribution_distance_batches.parquet',distance)
        # Primary global curves, plus explicit observation/control-domain outcomes.
        # These domains test the continuation hypothesis; no .1-bin discovery.
        integrated=[]
        for (arm,batch),g in distance.groupby(['arm','batch'],sort=True):
            for domain,left,right in [('observed_window',0,.5),('continuation',.5,.99)]:
                part=g[g.time.between(left,right)].sort_values('time')
                integrated.append({'arm':arm,'batch':batch,'domain':domain,
                    'mean_energy_squared_to_smc':float(trapezoid(part.energy_squared_to_smc,part.time)/(right-left)),
                    'mean_nearest_smc_pose_chamfer_A':float(trapezoid(part.nearest_smc_pose_chamfer_A,part.time)/(right-left))})
            integrated.append({'arm':arm,'batch':batch,'domain':'final',
                               'mean_energy_squared_to_smc':float(g[g.step.eq(100)].energy_squared_to_smc.iloc[0]),
                               'mean_nearest_smc_pose_chamfer_A':float(g[g.step.eq(100)].nearest_smc_pose_chamfer_A.iloc[0])})
        integrated=pd.DataFrame(integrated);write_table(output/'imitation_batches.csv',integrated)
        contrasts=[]
        for domain,grp in integrated.groupby('domain',sort=True):
            for metric in ['mean_energy_squared_to_smc','mean_nearest_smc_pose_chamfer_A']:
                base=grp[grp.arm.eq('unguided')].set_index('batch')[metric]
                for arm,g in grp.groupby('arm',sort=True):
                    if arm=='unguided':continue
                    effect=g.set_index('batch')[metric]-base
                    finite=effect[np.isfinite(effect)]
                    result=paired_effect(finite.to_numpy()) if len(finite) else {'n_batches':0,'mean_difference':None}
                    contrasts.append({'arm':arm,'domain':domain,'metric':metric,'negative_means_closer':True,
                                      'missing_batch_pairs':len(effect)-len(finite),**result})
        write_table(output/'paired_imitation.csv',contrasts)
    write_table(output/'final_fingerprint_similarity.csv',final_fingerprint_similarity(records))
    write_json(output/'evaluation_manifest.json',{'status':'complete','source':str(campaign),
        'source_program_sha256':digest(campaign/'reward_program.json'), 'n_feature_curve_rows':len(feature_rows),
        'limits':['No physical affinity validation; pic50 is an existing model rescore.',
                  'Similarity in two reward distances does not certify a 3D pose or graph match.',
                  'SMC particle count matches every new arm; different from historical 50-particle study.',
                  'Statistical unit is batch; four new batches give low statistical power.']})
    return summary
