"""Evaluate completed frozen-rule experiments, separately from reward discovery."""
import gzip
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
import torch
from ..io import read_json,write_json,write_table,digest
from ..trajectory_source import trajectory_paths,open_trajectory
from .scalar_guidance import RegionalReward


def paired_effect(values):
    v=np.asarray(values,float)
    if not len(v) or not np.isfinite(v).all():raise ValueError('Invalid paired batch effects')
    n=len(v);mean=float(v.mean())
    if n<2:return {'n_batches':n,'mean_difference':mean,'t_ci_low':None,'t_ci_high':None,'sign_flip_p':None}
    half=float(student_t.ppf(.975,n-1)*v.std(ddof=1)/np.sqrt(n))
    if n<=15:
        signs=np.array(list(itertools.product([-1,1],repeat=n)))
        p=float((np.abs(signs@v/n)>=abs(mean)-1e-12).mean())
    else:
        signs=np.random.default_rng(20261004).choice([-1,1],size=(32768,n))
        p=float((1+(np.abs(signs@v/n)>=abs(mean)-1e-12).sum())/(len(signs)+1))
    return {'n_batches':n,'mean_difference':mean,'t_ci_low':mean-half,'t_ci_high':mean+half,
            'sign_flip_p':p,'positive_batches':int((v>0).sum()),'negative_batches':int((v<0).sum())}


def summarize_records(records,baseline):
    d=pd.DataFrame(records)
    if d.duplicated(['arm','batch','slot']).any():raise ValueError('Duplicated output candidate')
    d['valid']=d.build_success.eq(True).astype(float)
    d['valid_connected']=(d.build_success.eq(True)&d.connected.eq(True)).astype(float)
    rows=[]
    for (arm,batch),g in d.groupby(['arm','batch'],sort=True):
        built=g[g.valid==1];clash=built.pairs_below_1_2A.dropna()
        rows.append({'arm':arm,'batch':batch,'n':len(g),'valid_fraction':g.valid.mean(),
            'connected_fraction':g.valid_connected.mean(),'ck2_rescore_all':g.pic50_on_rescore.mean(),
            'ck2_rescore_valid':built.pic50_on_rescore.mean(),
            'unique_smiles_fraction':built.smiles.nunique()/len(g),
            'clash_fraction_among_measured_valid':float((clash>0).mean()) if len(clash) else np.nan,
            'clash_measured_valid_n':len(clash),'qed_valid':built.qed.mean()})
    batches=pd.DataFrame(rows)
    if baseline not in set(batches.arm):raise ValueError('Baseline missing')
    base=batches[batches.arm==baseline].set_index('batch')
    paired=[]
    metrics=['ck2_rescore_all','ck2_rescore_valid','valid_fraction','connected_fraction','unique_smiles_fraction',
             'clash_fraction_among_measured_valid','qed_valid']
    for arm,g in batches.groupby('arm',sort=True):
        if arm==baseline:continue
        g=g.set_index('batch')
        if set(g.index)!=set(base.index) or not g.n.sort_index().equals(base.n.sort_index()):raise ValueError('Unmatched batch populations')
        for metric in metrics:
            difference=(g[metric]-base[metric]).sort_index()
            # Missing validity-conditioned outcomes remain visible, never zero-filled.
            finite=difference[np.isfinite(difference)]
            effect=paired_effect(finite) if len(finite) else {'n_batches':0,'mean_difference':None}
            paired.append({'arm':arm,'baseline':baseline,'metric':metric,'missing_batch_pairs':len(difference)-len(finite),**effect})
    summaries=[]
    for arm,g in d.groupby('arm',sort=True):
        b=batches[batches.arm==arm]
        summaries.append({'arm':arm,'n':len(g),'valid':int(g.valid.sum()),'connected':int(g.valid_connected.sum()),
            'unique_smiles':g.loc[g.valid==1,'smiles'].nunique(),
            **{m:float(b[m].mean()) for m in metrics}})
    return batches,pd.DataFrame(summaries),pd.DataFrame(paired)


def geometry_and_control(campaign):
    runtime=RegionalReward(read_json(campaign/'reward_program.json'),read_json(campaign/'reward_catalog.json'))
    cfg=read_json(campaign/'config.json');scale=cfg['coord_scale'];features=runtime.program['terms']
    rows=[];telemetry=[];schedules=[]
    for path in trajectory_paths(campaign):
        arm=path.parent.parent.name;batch=int(path.parent.name.split('_')[1]);frame=read_json(campaign/f'frame_batch_{batch:03d}.json')
        com=np.array(frame['target_com'])[:,None,:]
        with open_trajectory(path) as z:
            score_time=z['score_time'][:,0];mask=z['mask'];coords=z['predicted_coords']
            schedules.append({'arm':arm,'batch':batch,'resampled_steps':np.flatnonzero(z['resampled']).tolist()})
            for step in np.flatnonzero(np.round(score_time,6)<=.5):
                x=torch.as_tensor(coords[step]*scale+com,dtype=torch.float64)
                for term in features:
                    value=runtime.observable(term['feature'],x,torch.tensor(mask[step],dtype=torch.bool))
                    rows.append({'arm':arm,'batch':batch,'step':int(step),'score_time':float(score_time[step]),
                                 'term_id':term['id'],'mean':float(value.mean()),'median':float(value.quantile(.5)),
                                 'fraction_above_target':float((value>term['target']).double().mean()),'representation':'predicted_endpoint'})
        # Trusted local experiment file, explicit restricted tensor-only loading.
        with gzip.open(path.parent/'final_prediction.pt.gz','rb') as f:
            final=torch.load(f,map_location='cpu',weights_only=True)
        for term in features:
            values=runtime.observable(term['feature'],final['coords'].double(),final['mask'].bool())
            rows.append({'arm':arm,'batch':batch,'step':100,'score_time':1.,'term_id':term['id'],
                         'mean':float(values.mean()),'median':float(values.quantile(.5)),
                         'fraction_above_target':float((values>term['target']).double().mean()),'representation':'final_native_head_outcome'})
        trace=[json.loads(line) for line in (path.parent/'guidance_trace.jsonl').read_text().splitlines()]
        active=[r for r in trace if r['active']]
        lengths=np.array([r['injection_max_atom_A'] for r in trace])
        telemetry.append({'arm':arm,'batch':batch,'active_steps':[r['step'] for r in active],
                          'reward_evaluated_steps':[r['step'] for r in trace if r['reward_evaluated']],
                          'max_atom_step_A':float(lengths.max()),
                          'max_sum_of_step_maxima_A':float(lengths.sum(0).max()),
                          'nonzero_injection_steps':[r['step'] for r in trace if max(r['injection_max_atom_A'])>1e-12],
                          'nonzero_gradient_particle_steps':sum(sum(v>1e-12 for v in r.get('gradient_norm',[])) for r in trace),
                          'max_native_gradient_norm':max((max(r['gradient_norm']) for r in trace if 'gradient_norm' in r),default=0.),
                          'capped_particle_steps':sum(sum(v<1-1e-7 for v in r.get('cap_factor',[])) for r in trace)})
    return pd.DataFrame(rows),telemetry,schedules


def evaluate(campaign,output,baseline='unguided'):
    campaign,output=Path(campaign),Path(output)
    if read_json(campaign/'COMPLETE.json')['status']!='complete':raise ValueError('Incomplete campaign')
    records=read_json(campaign/'final_records.json')
    batch,summary,paired=summarize_records(records,baseline)
    geometry,telemetry,schedules=geometry_and_control(campaign)
    for s in schedules:
        if s['arm'].startswith('gradient') and s['resampled_steps']:raise ValueError('Gradient arm resampled')
        if s['arm']=='single' and s['resampled_steps']!=list(range(51)):raise ValueError('Unexpected SMC schedule')
    for row in telemetry:
        if row['max_atom_step_A']>.025+1e-6:raise ValueError('Displacement cap failed')
        if any(step>=20 for step in row['active_steps']):raise ValueError('Reward active outside frozen support')
    for name,data in [('batch_outcomes.csv',batch),('arm_outcomes.csv',summary),('paired_outcomes.csv',paired),('frozen_geometry.csv',geometry)]:
        write_table(output/name,data)
    if baseline=='unguided' and 'single' in set(batch.arm):
        _,_,against_smc=summarize_records(records,'single')
        write_table(output/'paired_outcomes_vs_single.csv',against_smc)
    geometry_pairs=[]
    for (term,step),g in geometry.groupby(['term_id','step']):
        b=g[g.arm==baseline].set_index('batch')['mean']
        for arm,a in g.groupby('arm'):
            if arm==baseline:continue
            diff=a.set_index('batch')['mean']-b
            geometry_pairs.append({'arm':arm,'baseline':baseline,'term_id':term,'step':int(step),
                                   'score_time':float(g.score_time.iloc[0]),**paired_effect(diff.sort_index())})
    write_table(output/'paired_geometry.csv',pd.DataFrame(geometry_pairs))
    write_json(output/'control_audit.json',{'telemetry':telemetry,'schedules':schedules})
    write_json(output/'comparison_provenance.json',{'source_campaign':str(campaign.resolve()),
        'config_sha256':digest(campaign/'config.json'),'final_records_sha256':digest(campaign/'final_records.json'),
        'source_code_commit':read_json(campaign/'config.json')['extension']['code_commit'],
        'baseline':baseline,'independent_unit':'paired batch; candidates and time points are not independent replicates',
        'interval':'two-sided paired-batch Student t 95% CI; exploratory with only four batches',
        'p_value':'exact two-sided sign-flip for <=15 batches; raw diagnostic p-values, no discovery claim',
        'geometry_scope':'actual historical selection time window plus final native-head outcomes; no post-window rule discovery'})
    return summary
