"""Snapshot-bound affinity distribution, lineage and control-response analysis.

No model/head calls. Historical full populations and post-selected endpoints
remain separate from common-initial/common-suffix experimental contrasts.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

from evomolsteer.io import read_json, digest, write_table
from evomolsteer.storage.transactions import atomic_json
from .common import binding, validate_protocol, validate_condition

LABELS = {'unguided':'对齐初态无引导','M2':'M2','top10_global_0':'Top10全局0–0.5',
          'S3':'S3固定→同时钟','blend_0p5':'恒定混合λ=0.5','C3':'C3曲率逐降',
          'blend_0p25':'恒定混合λ=0.25','C2':'C2恒定曲率','W1':'W1早弱晚强',
          'W3':'W3早强晚弱','S1':'S1同时钟→固定','linear_decay':'线性衰减剂量',
          'top10_global_0p1':'Top10全局0.1–0.5','top10_local_0p1':'Top10局部0.1–0.5',
          'NC_R11':'N/C R11','motif_R15_v2':'局部motif R15_v2','ancestor_R02_v3':'祖先R02_v3'}


def distribution(scores, valid, qualified, planned, threshold, steer_best):
    scores=np.asarray(scores,float); valid=np.asarray(valid,bool); qualified=np.asarray(qualified,bool)
    finite=np.isfinite(scores); values=scores[finite]; v=scores[finite & valid]; q=scores[finite & qualified]
    out={'n_planned':int(planned),'n_scored':len(values),'score_coverage':len(values)/planned if planned else None,
         'n_valid_connected':len(v),'n_qualified':len(q),
         'mean':float(values.mean()) if len(values) else None,
         'median':float(np.median(values)) if len(values) else None,
         'best_all':float(values.max()) if len(values) else None,
         'best_valid':float(v.max()) if len(v) else None,'best_qualified':float(q.max()) if len(q) else None}
    for name,probability in [('q10',.1),('q25',.25),('q75',.75),('q90',.9),('q95',.95),('q99',.99)]:
        out[name]=float(np.quantile(values,probability,method='linear')) if len(values) else None
    for name,level,strict in [('tail',threshold,False),('ge8p5',8.5,False),('above_steer',steer_best,True)]:
        count=int(np.sum(v>level if strict else v>=level))
        out[name+'_count']=count;out[name+'_rate']=count/planned if planned else None
    for fraction in [.05,.1]:
        n=max(1,math.ceil(len(values)*fraction))
        out[f'top{round(fraction*100)}_mean']=float(np.sort(values)[-n:].mean()) if len(values) else None
    return out


def mean_ci(roots, differences, repeats=10000):
    roots=np.asarray(roots);d=np.asarray(differences,float);unique=np.unique(roots)
    sums=np.asarray([d[roots==k].sum() for k in unique]);counts=np.asarray([(roots==k).sum() for k in unique])
    draws=np.random.default_rng(42).integers(0,len(unique),(repeats,len(unique)))
    means=sums[draws].sum(1)/counts[draws].sum(1);mean=float(d.mean())
    lo,hi=np.quantile(means,[.025,.975])
    p=(1+np.sum(np.abs(means-mean)>=abs(mean)))/(repeats+1)
    return float(lo),float(hi),float(p)


def holm(pvalues, family_size=None):
    p=np.asarray(pvalues,float);m=max(len(p),family_size or 0);order=np.argsort(p);q=np.empty(len(p));last=0.
    for rank,index in enumerate(order):
        last=max(last,(m-rank)*p[index]);q[index]=min(1.,last)
    return q


def nearest_teacher(x,mask,teachers):
    cloud=np.asarray(x,float)[np.asarray(mask,bool)]
    costs=[]
    for teacher in teachers:
        cost=cdist(cloud,teacher,'sqeuclidean');i,j=linear_sum_assignment(cost)
        costs.append(float(cost[i,j].mean()))
    return float(np.sqrt(min(costs)))


def field_mean(rows,key):
    v=[r[key] for r in rows if isinstance(r.get(key),(int,float)) and np.isfinite(r[key])]
    return float(np.mean(v)) if v else None


def run(repo,config_path,work,output,workers=4):
    repo,work,output=Path(repo).resolve(),Path(work).resolve(),Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    cfg=read_json(config_path);validate_protocol(cfg)
    collection=read_json(repo/'docs/experiments'/cfg['name']/'latest_collection.json')
    paths=collection['snapshot'].get('receipt_paths')
    if paths is None:raise ValueError('Analysis needs an explicit immutable snapshot receipt list')
    conditions={c['id']:c for c in cfg['conditions']}
    _,reference=validate_condition(repo,conditions['M2']);teachers=np.asarray(reference['frames'][-1]['teacher_endpoint_A'],float)
    configs={c['id']:validate_condition(repo,c)[0] for c in cfg['conditions']}
    snapshot_hashes={};records={};states={};control_rows={}
    for name in paths:
        path=work/name;record=read_json(path);condition=conditions[record['condition']]
        if binding(cfg,condition,record['attempt'])!=record['binding']:raise ValueError('Unmatched result receipt')
        key=(condition['id'],record['attempt']['id']);records[key]=record;snapshot_hashes[name]=digest(path)
    # Geometry and control are reduced once; no expansion into atom-by-time tables.
    def extract(item):
        key,r=item;a=r['attempt'];base={'condition':key[0],'attempt_id':key[1],'rank':a['rank'],'root':a['initial_id'],
            'historical_pic50':a['historical_pic50'],'suffix_seed':a['suffix_seed'],'status':r['status'],
            'pic50':r.get('pic50'),'valid_connected':bool(r.get('valid_connected')),
            'qualified':bool(r.get('qualified_structure')),'smiles':r.get('smiles'),
            'pb_status':r.get('pb_status'),'pb_pass':r.get('pb_pass'),
            'energy_status':r.get('energy_status'),'strain':r.get('mmff_relief_per_heavy'),'seconds':r.get('seconds')}
        if r['status']!='complete':return key,base,None,None
        folder=work/'methods'/key[0]/key[1]
        audit=read_json(folder/'guidance.json');rows=audit['steps'];active=[x for x in rows if x['active']]
        program=configs[key[0]];expected=[] if program is None else [i for i in range(100)
            if i/100>=program['window'][0]-1e-6 and (i+1)/100<=program['window'][1]+1e-6]
        if audit['active_steps']!=expected or len(rows)!=100:raise ValueError('Window audit differs: '+str(key))
        if not audit['no_smc'] or audit['affinity_head_gradient']:raise ValueError('Forbidden guidance detected')
        spec=cfg['initials'][a['initial_id']]
        if audit['initial_sha256']!=spec['sha256'] or audit['suffix_seed']!=a['suffix_seed']:raise ValueError('Input pairing differs')
        with np.load(folder/'states.npz') as archive:
            state={k:archive[k].copy() for k in archive.files}
        mask=state['mask'][0].astype(bool);terminal=state['terminal_coords_A'][0];half=state['half_coords_A'][0]
        base.update(active_steps=len(active),nonzero_steps=sum(x.get('nonzero_molecules',0)>0 for x in active),
            cumulative_rms_A=audit['cumulative_rms_A'],injection_early_A=sum(x.get('injected_rms_A_mean',0) for x in active if x['time']<.25),
            injection_late_A=sum(x.get('injected_rms_A_mean',0) for x in active if x['time']>=.25),
            cap_fraction=field_mean([dict(x,hit=float(x['cap_factor']<.999999)) for x in active if 'cap_factor'in x],'hit'),
            atom_clip_fraction=field_mean(active,'atom_clip_fraction'),reward_first=next((x.get('reward_mean') for x in active),None),
            reward_last=next((x.get('reward_mean') for x in active[::-1]),None),
            half_to_final_rms_A=float(np.sqrt(np.square(terminal[mask]-half[mask]).sum(1).mean())),
            final_nearest_M2_teacher_rms_A=nearest_teacher(terminal,mask,teachers),
            final_Rg_A=float(np.sqrt(np.square(terminal[mask]-terminal[mask].mean(0)).sum(1).mean())))
        realization=[x['injected_rms_A_mean']/x['requested_rms_A_mean'] for x in active
                     if x.get('requested_rms_A_mean',0)>1e-10]
        base['realized_requested_ratio_mean']=float(np.mean(realization)) if realization else None
        base['attenuated_step_fraction']=float(np.mean(np.asarray(realization)<.99999)) if realization else None
        return key,base,state,rows
    with ThreadPoolExecutor(max_workers=workers) as pool:
        extracted=list(pool.map(extract,records.items()))
    features=[]
    for key,feature,state,rows in extracted:
        features.append(feature)
        if state is not None:states[key]=state;control_rows[key]=rows
    df=pd.DataFrame(features).sort_values(['condition','rank']).reset_index(drop=True)
    for baseline in ['unguided','M2']:
        baseline_records=df[df.condition==baseline].set_index('attempt_id')
        df['delta_vs_'+baseline]=[float(r.pic50-baseline_records.loc[r.attempt_id,'pic50'])
            if pd.notna(r.pic50) and r.attempt_id in baseline_records.index else np.nan for r in df.itertuples()]
        diffs=[];halfs=[];equal=[]
        for r in df.itertuples():
            key=(r.condition,r.attempt_id);ref=(baseline,r.attempt_id)
            if key not in states or ref not in states:diffs.append(np.nan);halfs.append(np.nan);equal.append(None);continue
            x,y=states[key],states[ref];mask=x['mask'][0].astype(bool)&y['mask'][0].astype(bool)
            diffs.append(float(np.sqrt(np.square(x['terminal_coords_A'][0][mask]-y['terminal_coords_A'][0][mask]).sum(1).mean())))
            halfs.append(float(np.sqrt(np.square(x['half_coords_A'][0][mask]-y['half_coords_A'][0][mask]).sum(1).mean())))
            equal.append(bool(all(np.array_equal(x[k],y[k]) for k in ('atomics','bonds','charges'))))
        df['final_rms_vs_'+baseline]=diffs;df['half_rms_vs_'+baseline]=halfs;df['same_categories_vs_'+baseline]=equal
    metrics=[];pairs=[];roots=[]
    for cohort,spec in cfg['cohorts'].items():
        subset=df[df['rank']<=spec['n']]
        for condition in cfg['conditions']:
            group=subset[subset.condition==condition['id']];complete=group[group.status=='complete']
            row={'cohort':cohort,'condition':condition['id'],'label':LABELS[condition['id']],'source':'new_aligned',
                 'complete_cohort':len(complete)==spec['n'],
                 **distribution(complete.pic50,complete.valid_connected,complete.qualified,spec['n'],cfg['tail_threshold'],cfg['steer_best'])}
            row['n_failed']=int((group.status=='failed').sum());row['n_skipped']=int((group.status=='skipped_condition_error').sum())
            row['n_pending']=spec['n']-len(group);row['n_roots']=complete.root.nunique()
            row['root_equal_mean']=float(complete.groupby('root').pic50.mean().mean()) if len(complete) else None
            energy=complete.loc[complete.energy_status=='converged','strain'].dropna()
            row.update(n_energy=len(energy),strain_median=float(energy.median()) if len(energy) else None,
                strain_p90=float(energy.quantile(.9)) if len(energy) else None,
                pb_evaluated=int((complete.pb_status=='evaluated').sum()),pb_pass=int((complete.pb_pass==True).sum()))
            for key in ['cumulative_rms_A','cap_fraction','atom_clip_fraction','final_nearest_M2_teacher_rms_A','final_Rg_A',
                        'half_to_final_rms_A','final_rms_vs_unguided','half_rms_vs_unguided','final_rms_vs_M2','half_rms_vs_M2',
                        'injection_early_A','injection_late_A','realized_requested_ratio_mean','attenuated_step_fraction']:
                row[key+'_mean']=float(complete[key].mean()) if key in complete and complete[key].notna().any() else None
            metrics.append(row)
            for root,local in complete.groupby('root'):
                roots.append({'cohort':cohort,'condition':condition['id'],'root':root,'n':len(local),'mean':float(local.pic50.mean()),
                    'best_valid':float(local.loc[local.valid_connected,'pic50'].max()) if local.valid_connected.any() else None,
                    'delta_vs_unguided':float(local.delta_vs_unguided.mean()),'delta_vs_M2':float(local.delta_vs_M2.mean()),
                    'cumulative_rms_A':float(local.cumulative_rms_A.mean())})
            for baseline in ['unguided','M2']:
                if condition['id']==baseline:continue
                common=complete.merge(subset[(subset.condition==baseline)&(subset.status=='complete')],on='attempt_id',suffixes=('','_base'))
                if not len(common):continue
                delta=common.pic50-common.pic50_base;lo,hi,p=mean_ci(common.root,delta)
                pairs.append({'cohort':cohort,'condition':condition['id'],'baseline':baseline,'n_pairs':len(common),'n_roots':common.root.nunique(),
                    'delta_mean':float(delta.mean()),'delta_median':float(delta.median()),'win_fraction':float((delta>1e-6).mean()),
                    'unchanged_fraction_1e5':float((delta.abs()<=1e-5).mean()),'ci95_low':lo,'ci95_high':hi,'p_bootstrap':p,
                    'delta_q95':float(common.pic50.quantile(.95)-common.pic50_base.quantile(.95)),
                    'delta_q99':float(common.pic50.quantile(.99)-common.pic50_base.quantile(.99)),
                    'delta_best_valid':row['best_valid']-float(common.loc[common.valid_connected_base,'pic50_base'].max()) if row['best_valid'] is not None and common.valid_connected_base.any() else None})
    # Holm over the predeclared unique contrasts, not duplicate reversed baselines.
    unique={}
    for p in pairs:unique.setdefault((p['cohort'],tuple(sorted([p['condition'],p['baseline']]))),p['p_bootstrap'])
    adjusted=dict(zip(unique,holm(list(unique.values()),family_size=93)))
    for p in pairs:p['p_holm_exploratory']=adjusted[(p['cohort'],tuple(sorted([p['condition'],p['baseline']])))]
    hist=[];raw=repo/'data/raw/ck2_clk3_lineage_20261003/results/main1000_w050'
    original={}
    for arm,label in [('single','原始Steer全量'),('unguided','原始无引导全量')]:
        source=[r for f in sorted((raw/arm).glob('batch_*/final_records.json')) for r in read_json(f)]
        if len(source)!=1000:raise ValueError('Historical source is incomplete')
        scores=[r['pic50_on_rescore'] for r in source];valid=[bool(r.get('build_success') and r.get('connected')) for r in source]
        original[arm]=source
        hist.append({'source':arm,'cohort':'full1000','condition':label,**distribution(scores,valid,[False]*len(source),len(source),cfg['tail_threshold'],cfg['steer_best'])})
    selected={ (r['batch'],r['slot']):r for r in original['single'] }
    for cohort,spec in cfg['cohorts'].items():
        chosen=[selected[(a['batch'],a['final_slot'])] for a in cfg['attempts'][:spec['n']]]
        hist.append({'source':'single_selected','cohort':cohort,'condition':'原始Steer终态后筛选集合',
            **distribution([r['pic50_on_rescore'] for r in chosen],[bool(r.get('build_success') and r.get('connected')) for r in chosen],
                           [False]*len(chosen),len(chosen),cfg['tail_threshold'],cfg['steer_best'])})
    rootdf=pd.DataFrame(roots);metricdf=pd.DataFrame(metrics);pairdf=pd.DataFrame(pairs);histdf=pd.DataFrame(hist)
    correlations=[]
    for condition,g in df[df.status=='complete'].groupby('condition'):
        for key in ['historical_pic50','cumulative_rms_A','final_nearest_M2_teacher_rms_A','final_Rg_A','half_to_final_rms_A']:
            finite=g[['pic50',key]].dropna()
            if len(finite)>3 and finite[key].nunique()>1:
                rho,p=spearmanr(finite.pic50,finite[key]);correlations.append({'condition':condition,'observable':key,
                    'n':len(finite),'spearman':float(rho),'p_naive_not_independent':float(p)})
    for name,frame in [('attempt_metrics',df),('performance',metricdf),('paired_effects',pairdf),('root_effects',rootdf),
                       ('historical_benchmarks',histdf),('diagnostic_correlations',pd.DataFrame(correlations))]:
        write_table(output/(name+'.parquet'),frame);write_table(output/(name+'.csv'),frame)
    supplements(repo,cfg,output)
    cases=df[(df.condition!='unguided')&(df.status=='complete')]
    examples=pd.concat([g.nlargest(3,'delta_vs_unguided') for _,g in cases.groupby('condition')]+
                       [g.nsmallest(3,'delta_vs_unguided') for _,g in cases.groupby('condition')]).drop_duplicates(['condition','attempt_id'])
    write_table(output/'counterexamples.csv',examples)
    audit={'protocol_sha256':cfg['protocol_sha256'],'collection_unix':collection['collected_unix'],
        'snapshot_sha256':collection['snapshot']['sha256'],'receipt_count':len(records),'completed':int((df.status=='complete').sum()),
        'input_receipt_sha256':snapshot_hashes,'quantile_method':'numpy linear; all finite-scored attempts, with valid Best separate',
        'mean_inference':'paired root-cluster bootstrap B=10000 seed42, exploratory; 93 unique planned contrasts for Holm',
        'geometry_proxy':'World-frame Hungarian RMS to nearest of the fixed 28 M2 endpoint teachers. Terminal geometry diagnostic, not the actual runtime scalar.',
        'no_new_model_or_head_calls':True,'historical_selected_comparison':'Post-selection descriptive benchmark, not same-noise or equal-search-budget control.'}
    atomic_json(output/'analysis_audit.json',audit)
    lines=['# 优质初态对齐实验：分布比较','',
        f"本次快照含{audit['completed']}条完成结果。三个阈值组嵌套；统计按根聚类，不能视为三次独立验证。",'']
    for cohort in cfg['cohorts']:
        lines += [f'## {cohort}（计划N={cfg["cohorts"][cohort]["n"]}）','',
            '| 方法 | 已评分/计划 | 均值 | P75 | P90 | P95 | P99 | 有效Best | 高分数 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for r in metricdf[metricdf.cohort==cohort].sort_values('mean',ascending=False).to_dict('records'):
            f=lambda v:'—' if pd.isna(v) else f'{v:.5f}'
            lines.append(f"| {r['label']} | {r['n_scored']}/{r['n_planned']} | "+' | '.join(f(r[k]) for k in ['mean','q75','q90','q95','q99','best_valid'])+f" | {r['tail_count']} |")
        lines.append('')
    lines+=['## 历史参照','',
            '| 参照 | 集合 | N | 均值 | P90 | P95 | P99 | 有效Best | 高分数 |',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in hist:
        lines.append(f"| {r['condition']} | {r['cohort']} | {r['n_scored']} | "+' | '.join(f"{r[k]:.5f}" for k in ['mean','q90','q95','q99','best_valid'])+f" | {r['tail_count']} |")
    lines+=['', '完整参照包含原始Steer1000次与原始无引导1000次；Steer按终态后筛选的50/100/200集合单列。未完成条件不参与同N排名。P99在N=50时只由最顶端少数点确定。',
            '逐起点正反例、根节点效应、实际注入剂量、末态教师几何残差见同目录数据表。几何残差/剂量与亲和力的相关性是描述性证据，不是独立随机化的机制证明。']
    (output/'tables.zh-CN.md').write_bytes(('\n'.join(lines)+'\n').encode())
    plot(output,df,metricdf,original)
    return metricdf


def plot(output,df,metrics,original):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for condition in ['unguided','M2','top10_global_0','S1','S3']:
        v=np.sort(df.loc[(df.condition==condition)&(df.status=='complete'),'pic50'].to_numpy())
        if not len(v):continue
        axes[0].step(v,np.arange(1,len(v)+1)/len(v),where='post',label=f'{condition} n={len(v)}')
        axes[1].step(v,(len(v)-np.arange(len(v)))/len(v),where='pre',label=f'{condition} n={len(v)}')
    v=np.sort([r['pic50_on_rescore'] for r in original['single']])
    axes[0].step(v,np.arange(1,len(v)+1)/len(v),where='post',label='Historical Steer n=1000',color='black',ls='--')
    axes[1].step(v,(len(v)-np.arange(len(v)))/len(v),where='pre',label='Historical Steer n=1000',color='black',ls='--')
    axes[0].set(xlabel='Final predicted pIC50',ylabel='Empirical CDF',title='Full distribution')
    axes[1].set(xlabel='Final predicted pIC50',ylabel='Fraction with score >= x',yscale='log',xlim=(7.8,8.8),ylim=(.001,1),title='Upper tail (all finite scores)')
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(output/'affinity_distributions.png',dpi=180);plt.close(fig)
    root=df[(df.status=='complete')&(df.condition!='unguided')].pivot_table(index='condition',columns='root',values='delta_vs_unguided',aggfunc='mean')
    limit=max(.1,float(np.nanmax(np.abs(root.to_numpy()))))
    fig,ax=plt.subplots(figsize=(12,6));a=ax.imshow(root.to_numpy(),cmap='RdBu_r',vmin=-limit,vmax=limit,aspect='auto')
    ax.set_yticks(range(len(root)),root.index);ax.set_xticks(range(len(root.columns)),root.columns,rotation=75,ha='right',fontsize=7)
    ax.set_title('Mean paired pIC50 difference vs unguided, by initial root');fig.colorbar(a,ax=ax,label='pIC50 difference')
    fig.tight_layout();fig.savefig(output/'root_effects.png',dpi=180);plt.close(fig)


def supplements(repo,cfg,output):
    """Derived compact diagnostics; no additional model or trajectory reads."""
    output=Path(output);df=pd.read_parquet(output/'attempt_metrics.parquet')
    diversity=[];gains=[];parameters=[]
    for cohort,spec in cfg['cohorts'].items():
        for condition,group in df[(df['rank']<=spec['n'])&(df.status=='complete')].groupby('condition'):
            counts=group.loc[group.valid_connected,'smiles'].dropna().value_counts();prob=counts/counts.sum()
            diversity.append({'cohort':cohort,'condition':condition,'valid_n':int(counts.sum()),'unique_smiles':len(counts),
                'largest_graph_count':int(counts.max()) if len(counts) else 0,
                'largest_graph_fraction':float(prob.max()) if len(prob) else None,
                'effective_graph_count':float(np.exp(-(prob*np.log(prob)).sum())) if len(prob) else None})
    ng=df[(df.condition=='unguided')&(df.status=='complete')][['attempt_id','pic50']].rename(columns={'pic50':'unguided_score'})
    allpaired=df[df.status=='complete'].merge(ng,on='attempt_id')
    for condition,group in allpaired.groupby('condition'):
        for name,mask in [('unguided_below_7p5',group.unguided_score<7.5),
                          ('unguided_7p5_to_8',(group.unguided_score>=7.5)&(group.unguided_score<8)),
                          ('unguided_ge8',group.unguided_score>=8)]:
            local=group[mask]
            if len(local):gains.append({'condition':condition,'descriptive_stratum':name,'n':len(local),
                'mean_delta':float(local.delta_vs_unguided.mean()),'n_improved':int((local.delta_vs_unguided>0).sum()),
                'caveat':'Posthoc conditioning on the same baseline also produces regression-to-mean effects; not causal subgroup evidence.'})
    for condition in cfg['conditions']:
        p,r=validate_condition(repo,condition)
        row={'condition':condition['id'],'observed_n':int(((df.condition==condition['id'])&(df.status=='complete')).sum()),
             'program_sha256':condition.get('program_sha256'),'reference_sha256':condition.get('reference_sha256')}
        if p is not None:
            policy=p.get('temporal_policy',{});dose=policy.get('dose_factors',[1.]);blend=policy.get('teacher_blend',[p.get('temporal_teacher_blend',0.)])
            delta=policy.get('delta_A',[p.get('pointcloud_delta_A',1.)])
            row.update(window_start=p['window'][0],window_end=p['window'][1],reward=p.get('reward_view',p.get('family')),
                nominal_rms_ratio=p.get('native_rms_ratio',p.get('dose',{}).get('native_rms_ratio')),
                dose_first=dose[0],dose_last=dose[-1],blend_first=blend[0],blend_last=blend[-1],
                delta_first=delta[0],delta_last=delta[-1],time_ramp_power=p.get('time_ramp_power',0.),
                teacher_count=len(r['frames'][-1]['teacher_endpoint_A']) if 'frames'in r else None,
                mixture_temperature=p.get('mixture_temperature'),local_point_count=len(p.get('atom_slots',[])) or None)
        parameters.append(row)
    for name,frame in [('diversity',pd.DataFrame(diversity)),('baseline_strata',pd.DataFrame(gains)),('parameter_audit',pd.DataFrame(parameters))]:
        write_table(output/(name+'.csv'),frame);write_table(output/(name+'.parquet'),frame)


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',default='.');p.add_argument('--config',required=True)
    p.add_argument('--work',required=True);p.add_argument('--output',required=True);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();run(a.repo,a.config,a.work,a.output,a.workers)


if __name__=='__main__':main()
