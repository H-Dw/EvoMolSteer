"""Selection-only evidence: no outcome labels, late stages, or terminal IQRs."""
from pathlib import Path
import numpy as np
import pandas as pd
from .scope import SCOPE,load
from .io import read_json,write_json,digest

PRIMARY = {'expected_selection_shift','expected_high_mass_shift'}


def focus_arm(cfg):
    arm = cfg.get('evidence_arm', 'joint')
    if arm not in cfg['selection_arms']:
        raise ValueError('evidence_arm must name an analyzed selection arm')
    return arm


def attach_matched_context(analysis,bundle,window_frame=None):
    """Always attach same-region/stage controls, even when their effects are zero."""
    analysis = Path(analysis)
    arm = bundle.get('evidence_arm', 'joint')
    pairs = {(t['stage'],t['feature']) for t in bundle['targets']}
    existing = {e['evidence_id'] for e in bundle['evidence']}
    for method in ['enrichment','differential','trends']:
        path = analysis/'discovery'/method/'effects.csv'
        table = pd.read_csv(path)
        table['_row'] = np.arange(len(table))
        table = table[(table.arm==arm)&pd.Series([(int(s),f) in pairs for s,f in zip(table.stage,table.feature)],index=table.index)]
        source = {'path':str(path.resolve()),'sha256':digest(path)}
        for _,row in table.iterrows():
            idx = int(row['_row']);eid = f'{method}:effects:row{idx}'
            if eid not in existing:
                bundle['evidence'].append({'evidence_id':eid,'method':method,'statistic':'effects',
                    'selection_reason':'matched_feature_stage_control','source':source,
                    'csv_data_row_zero_based':idx,**row.drop(labels=['_row']).to_dict()})
                existing.add(eid)
    for target in bundle['targets']:
        target['matched_evidence_ids'] = [e['evidence_id'] for e in bundle['evidence']
            if e.get('stage')==target['stage'] and e.get('feature')==target['feature'] and e.get('statistic')=='effects']
    # Compact per-event profiles for rule candidates preserve timing inside bins.
    # These descriptive trajectories have no new p-values and never use later states.
    if not pairs:
        bundle['event_dynamics'] = []
        return bundle
    event_path = analysis/'discovery/trends/events.parquet'
    if event_path.exists():
        data = pd.read_parquet(event_path,
            filters=[('arm','==',arm),('feature','in',sorted({f for _,f in pairs}))])
    else:
        # Compact retention may omit full event diagnostics. Reconstruct only the
        # small candidate-feature profiles from the already audited window frame.
        from .statistics import selection_moments
        if window_frame is None:
            window_frame,cfg,_,_ = load(analysis,'discovery')
        else:
            cfg = read_json(analysis/'config.json')
        records = []
        for stage,feature in sorted(pairs):
            subset = window_frame[(window_frame.arm==arm)&(window_frame.stage==stage)]
            for (rep,batch,step),g in subset.groupby(['representation','batch','step'],sort=True):
                m = selection_moments(g[[feature]].to_numpy(),g.probability,g.offspring_count,cfg['minimum_feature_fraction'])
                records.append({'representation':rep,'batch':batch,'step':step,'stage':stage,'feature':feature,
                    'time':float(g.score_time.iloc[0]),**{k:float(v[0]) for k,v in m.items()}})
        data = pd.DataFrame(records)
    data = data[pd.Series([(int(s),f) in pairs for s,f in zip(data.stage,data.feature)],index=data.index)]
    rows = []
    for (rep,stage,feature),g in data.groupby(['representation','stage','feature'],sort=True):
        profile = g.groupby('step',as_index=False,sort=True).agg(
            score_time=('time','mean'),expected_shift=('expected_selection_shift','mean'),
            realized_shift=('realized_selection_shift','mean'),selection_noise=('selection_noise_shift','mean'),
            population_mean=('population_mean','mean'),n_batches=('expected_selection_shift','count'))
        rows.append({'representation':rep,'stage':int(stage),'feature':feature,
                     'meaning':'descriptive equal-batch event means; no additional inference',
                     'events':profile.to_dict('records')})
    bundle['event_dynamics'] = rows
    write_json(analysis/'agents/selection_feature_profiles.json',rows)
    return bundle


def weighted_quantile(values,weights,quantiles):
    values,weights = np.asarray(values,float),np.asarray(weights,float)
    good = np.isfinite(values)&np.isfinite(weights)&(weights>0)
    order = np.argsort(values[good],kind='stable')
    x,w = values[good][order],weights[good][order]
    if not len(x):
        return np.full(len(quantiles),np.nan)
    # Identical geometry may be represented by multiple clones. Quantiles must
    # depend on total mass, not on how that mass is split across duplicate rows.
    x,index = np.unique(x,return_index=True)
    w = np.add.reduceat(w,index)
    centers = (np.cumsum(w)-w/2)/w.sum()
    return np.interp(quantiles,centers,x)


def target_distribution(g,feature,cfg):
    """Equal batches/events; p-weighted candidates within each eligible event."""
    chunks = []
    for (batch,step),event in g.groupby(['batch','step'],sort=True):
        finite = event[feature].notna()
        valid = event[finite]
        mass = valid.probability.sum()/event.probability.sum()
        if finite.mean()<cfg['minimum_feature_fraction'] or mass<cfg['minimum_feature_fraction']:
            continue
        v = valid[['batch','step','root_id',feature]].copy()
        v['weight'] = valid.probability.to_numpy()/valid.probability.sum()
        v['baseline_weight'] = 1/len(valid)
        chunks.append(v)
    if not chunks:
        return None
    d = pd.concat(chunks,ignore_index=True)
    events = d[['batch','step']].drop_duplicates().groupby('batch').size()
    factor = d.batch.map(events).to_numpy()*len(events)
    w,bw = d.weight.to_numpy()/factor,d.baseline_weight.to_numpy()/factor
    q = weighted_quantile(d[feature],w,[.25,.5,.75])
    bq = weighted_quantile(d[feature],bw,[.25,.5,.75])
    mean = np.sum(bw*d[feature])
    scale = np.sqrt(np.sum(bw*(d[feature]-mean)**2))
    return {'lower':q[0],'upper':q[2],'weighted_median':q[1],
        'uniform_lower':bq[0],'uniform_median':bq[1],'uniform_upper':bq[2],
        'scale':max(float(scale),1e-3),'support_batches':len(events),
        'support_events':len(d[['batch','step']].drop_duplicates()),'candidate_rows':len(d),
        'distinct_roots':d.root_id.nunique(),
        'status':'selection_probability_weighted_prototype_not_optimal_or_outcome_validated'}


def build_bundle(analysis,split='discovery'):
    if split!='discovery':
        raise ValueError('Rule discovery cannot consume validation/held-out data')
    analysis = Path(analysis)
    df,cfg,cat,scope = load(analysis,split)
    arm = focus_arm(cfg)
    evidence,sources,coverage = [],{},[]
    background = df[df.arm==scope['background_arm']]
    scales = background.groupby(['representation','stage'])[list(cat['features'])].std()
    for method in ['enrichment','differential','trends','pca']:
        for kind in ['effects','effect_rates']:
            path = analysis/split/method/(kind+'.csv')
            t = pd.read_csv(path)
            t['_row'] = np.arange(len(t))
            t = t[t.arm==arm]
            source = {'path':str(path.resolve()),'sha256':digest(path)}
            sources[method+':'+kind] = source
            for (rep,contrast),g in t.groupby(['representation','contrast'],sort=True):
                coverage.append({'method':method,'statistic':kind,'representation':rep,'contrast':contrast,
                    'n_rows':len(g),'q_below_005':int((g.q_value<.05).sum()),'minimum_q':g.q_value.min()})
                g = g[np.isfinite(g.effect)].copy()
                g['_rank'] = g.effect.abs()
                if method in ('differential','trends'):
                    scale = np.array([scales.loc[(rep,int(r.stage)),r.feature] for r in g.itertuples()])
                    g['_rank'] = np.divide(g._rank,scale,out=np.zeros(len(g)),where=scale>1e-8)
                # Include corrected signals and large effects, with fixed small per-family budget.
                chosen = pd.concat([g[g.q_value<.05].sort_values(['q_value','_rank'],ascending=[True,False]).head(2),
                                    g.sort_values('_rank',ascending=False,kind='stable').head(2)]).drop_duplicates('_row')
                for _,r in chosen.iterrows():
                    row = r.drop(labels=['_row','_rank']).to_dict()
                    idx = int(r['_row'])
                    evidence.append({'evidence_id':f'{method}:{kind}:row{idx}','method':method,
                        'statistic':kind,'source':source,'csv_data_row_zero_based':idx,**row})
    path = analysis/split/'pca/loadings.csv'
    t = pd.read_csv(path);t['_row'] = np.arange(len(t))
    for _,g in t.groupby(['representation','component'],sort=True):
        for _,r in g.loc[g.loading.abs().sort_values(ascending=False,kind='stable').head(2).index].iterrows():
            idx = int(r['_row'])
            evidence.append({'evidence_id':f'pca:loading:row{idx}','method':'pca_loading','statistic':'loading',
                'source':{'path':str(path.resolve()),'sha256':digest(path)},'csv_data_row_zero_based':idx,
                **r.drop(labels=['_row']).to_dict()})
    stages = {s['stage']:s for s in scope['stages']}
    pairs = sorted({(int(e['stage']),e['feature']) for e in evidence if
        e.get('representation')=='predicted_endpoint' and e.get('statistic')=='effects' and
        e.get('contrast') in PRIMARY and abs(e.get('effect',0))>1e-10 and
        cat['features'].get(e.get('feature'),{}).get('differentiable_supported')})
    joint = df[(df.arm==arm)&(df.representation=='predicted_endpoint')]
    targets = []
    for stage,feature in pairs:
        distribution = target_distribution(joint[joint.stage==stage],feature,cfg)
        if distribution is None:
            continue
        targets.append({'target_id':f'selection:{stage}:{feature}','feature':feature,
            'region':cat['features'][feature]['region'],'representation':'predicted_endpoint',
            'stage':stage,'stage_start':stages[stage]['stage_start'],'stage_end':stages[stage]['stage_end'],
            **distribution})
    focus = sorted({e['feature'] for e in evidence if e.get('feature') in cat['features']})
    events = pd.read_parquet(analysis/'selection_events.parquet',filters=[('split','==','discovery')])
    examples = []
    for step in [scope['steps'][0],scope['steps'][len(scope['steps'])//2],scope['steps'][-1]]:
        g = joint[(joint.batch==min(cfg['discovery_batches']))&(joint.step==step)]
        for label in [True,False]:
            sample = g[g.selected==label].sort_values('probability',ascending=False).head(1)
            for _,row in sample.iterrows():
                examples.append({k:row[k] for k in ['node_id','step','score_time','selected','offspring_count',
                                                  'probability','pic50_on','pic50_off','root_id']})
    bundle = {'schema_version':'2.0','dataset_id':digest(analysis/'ingest_manifest.json'),'split':split,
        'evidence_arm':arm,
        'scope':{**scope,'batches':sorted(df.batch.unique().tolist()),'n_feature_rows':len(df),
                 'candidate_frames_per_representation':df.groupby('representation').size().to_dict(),
                 'stage_event_counts':events[events.resampled].groupby(['arm','stage']).size().reset_index(name='n_batch_events').to_dict('records')},
        'definitions':{
            'objective':'Describe selection preference during observed resampling; no final success or post-window evidence.',
            'time':'All control windows use score_time. Proposal geometry can be one integration step later. Last stage includes window_end.',
            'expected_vs_realized':'p-weighted preference is the primary signal; offspring counts are one multinomial realization. Selection noise is retained.',
            'enrichment':'High-feature thresholds are fitted on discovery unguided candidates at the SAME score step, then frozen.',
            'statistics':'Each event is compared internally; equal events within batch-stage and equal independent batches. Sign-flip tests, shared deterministic bootstrap draws. BH across features and stages separately for each representation/arm/contrast/output file.',
            'rates':'Event-effect derivatives and stage-effect derivatives use participating score times. Parent-child geometry rates use actual geometry times. Shift/dt is a different quantity from the derivative of shift.',
            'pca':'Axes fitted only on discovery unguided time-matched selection-window states; PCA variance alone is not a beneficial direction.',
            'prototypes':'Equal-batch/equal-event probability-weighted IQRs describe expected selected candidates; they are not learned optima.',
            'negative_controls':'Rejected means zero offspring at this event, never terminal failure. Same-parent clones may have identical endpoint predictions.',
            'geometry':'Distance proxies, conditional atom masks, and fixed receptor regions; no verified hydrogen-bond or chemical mechanism.'},
        'evidence':evidence,'evidence_coverage':coverage,'targets':targets,'selection_examples':examples,
        'feature_catalog':{'features':{f:cat['features'][f] for f in focus},
            'regions':{cat['features'][f]['region']:{'pocket':cat['regions'].get(cat['features'][f]['region'],{}).get('pocket','ligand')} for f in focus},
            'full_catalog_path':str((analysis/'feature_catalog.json').resolve()),'full_catalog_sha256':digest(analysis/'feature_catalog.json')},
        'limitations':['No outcome optimization claim is supported by this selection-only evidence.',
                       'Scoring preferences and random replication are not causal coordinate gradients.',
                       'The same-parent and cross-representation controls must accompany rule proposals.']}
    bundle = attach_matched_context(analysis,bundle,df)
    dest = analysis/'agents'
    write_json(dest/'evidence_bundle.json',bundle)
    write_json(dest/'bundle_manifest.json',{'sha256':digest(dest/'evidence_bundle.json'),'input_sources':sources,
                                           'builder_sha256':digest(__file__)})
    return bundle
