"""Compact affinity-primary evidence, with adaptive and frozen results separated."""
from pathlib import Path
import itertools
import numpy as np
import pandas as pd
from ..io import read_json,write_json,digest
from .terminal_statistics import converged_energy


def paired_head(actual,native):
    keys=['batch','seed','slot'];columns=keys+['pic50_on_rescore']
    g=actual[actual.arm.eq('gradient')][columns];n=native[native.arm.eq('unguided')][columns]
    paired=g.merge(n,on=keys,validate='one_to_one',suffixes=('_g','_n'))
    if len(paired)!=len(g) or paired[['pic50_on_rescore_g','pic50_on_rescore_n']].isna().any().any():
        raise ValueError('All generated attempts and score pairs required')
    paired['delta']=paired.pic50_on_rescore_g-paired.pic50_on_rescore_n
    return paired


def batch_inference(delta_by_batch,seed=42):
    values=np.asarray(delta_by_batch,float)
    if not len(values) or not np.isfinite(values).all():raise ValueError('Finite independent batch contrasts required')
    rng=np.random.default_rng(seed);boot=values[rng.integers(0,len(values),(10000,len(values)))].mean(1)
    observed=abs(values.mean());p=float(np.mean([abs(np.mean(values*np.array(sign)))>=observed-1e-12
        for sign in itertools.product((-1,1),repeat=len(values))])) if len(values)<=16 else None
    return {'batch_count':len(values),'mean_head_delta':float(values.mean()),
        'batch_bootstrap_CI95':np.quantile(boot,[.025,.975]).tolist(),'symmetric_null_exact_signflip_p':p,
        'assumptions':'Equal generation batches; bootstrap sampling and sign-symmetric null. Not measured affinity or a causal LLM attribution.'}


def arm_summary(frame):
    valid=frame[frame.valid_connected];unique=valid.drop_duplicates('smiles');energy=converged_energy(frame)
    return {'attempts':len(frame),'valid_rate':float(frame.valid_connected.mean()),'PB_fast_rate':float(frame.pb_fast_pass.mean()),
        'score_coverage':float(frame.pic50_on_rescore.notna().mean()),'all_head_mean':float(frame.pic50_on_rescore.mean()),
        'valid_head_mean':float(valid.pic50_on_rescore.mean()),'unique_first_occurrence_head_mean':float(unique.pic50_on_rescore.mean()),
        'best_valid_head':float(valid.pic50_on_rescore.max()),'best_PB_fast_head':float(frame.loc[frame.pb_fast_pass,'pic50_on_rescore'].max()),
        'unique_Top5_mean':float(unique.nlargest(5,'pic50_on_rescore').pic50_on_rescore.mean()),
        'converged_energy_n':len(energy),'MMFF_relief_per_heavy_median':float(energy.median()),
        'MMFF_relief_per_heavy_p90':float(energy.quantile(.9)),
        'surround_relax_RMS_A_mean':float(frame.relax_rms_surround_A.mean())}


def report(evidence,output,require_complete=True):
    evidence=Path(evidence);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    outcomes=[read_json(p) for p in sorted(evidence.glob('round_*.outcome.json'))]
    if require_complete and [r['round'] for r in outcomes]!=list(range(1,31)):raise ValueError('All thirty new rounds required')
    rows=[];pairs=[];retained=[]
    baseline=pd.read_csv(evidence/'round_01/local/candidate_metrics.csv')
    for r in outcomes:
        local=evidence/f"round_{r['round']:02d}/local";actual=pd.read_csv(local/'candidate_metrics.csv')
        native=actual if actual.arm.eq('unguided').any() else baseline
        delta=paired_head(actual,native)
        if not np.isclose(delta.delta.mean(),r['all_head_change_vs_native'],atol=1e-9):raise ValueError('Outcome/paired score mismatch')
        for batch,group in delta.groupby('batch'):
            pairs.append({'round':r['round'],'split':r['split'],'batch':int(batch),'n':len(group),
                'mean_head_delta':float(group.delta.mean()),'mean_absolute_head_delta':float(group.delta.abs().mean()),
                'positive_fraction':float(group.delta.gt(0).mean()),'head_delta_p10':float(group.delta.quantile(.1)),
                'head_delta_p90':float(group.delta.quantile(.9))})
        g=actual[actual.arm.eq('gradient')]
        rows.append({'round':r['round'],'split':r['split'],'reward_view':r['reward_view'],'dose':r['native_rms_ratio'],
            'head_delta':r['all_head_change_vs_native'],**arm_summary(g)})
        retained.append({'round':r['round'],'outcome_sha256':digest(evidence/f"round_{r['round']:02d}.outcome.json"),
            'retention_sha256':digest(local/'retention.json'),'inference_commit':read_json(local/'retention.json')['inference_commit']})
    frozen={}
    for label,selected in [('validation',[27]),('heldout',[28,29,30]),('all_frozen',[27,28,29,30])]:
        selected=[n for n in selected if any(r['round']==n for r in outcomes)]
        if not selected:continue
        tables=[pd.read_csv(evidence/f'round_{n:02d}/local/candidate_metrics.csv') for n in selected]
        data=pd.concat(tables,ignore_index=True);contrasts=[v['mean_head_delta'] for v in pairs if v['round'] in selected]
        frozen[label]={'rounds':selected,'gradient':arm_summary(data[data.arm.eq('gradient')]),
            'native':arm_summary(data[data.arm.eq('unguided')]),'batch_contrast':batch_inference(contrasts)}
    pd.DataFrame(rows).to_parquet(output/'round_metrics.parquet',compression=None,index=False)
    pd.DataFrame(pairs).to_parquet(output/'paired_batch_metrics.parquet',compression=None,index=False)
    value={'schema_version':'affinity-primary-campaign-report-1.0','new_rounds_completed':len(outcomes),'maximum_rounds':30,
        'discovery_rounds':[r['round'] for r in outcomes if r['split']=='discovery'],'frozen':frozen,'retained_sources':retained,
        'historical_Steer100':{'all_head_mean':7.510350561141967,'best_valid_head':8.347940444946289,
            'scope':'Historical100 subset, not verified full-original maximum; unequal-budget and unpaired benchmark'},
        'interpretation':'Adaptive discovery is not heldout evidence. Scores are shared-head predictions, not measured affinity. Graph transitions free; all failures retained.'}
    write_json(output/'campaign_report.json',value)
    lines=['# 亲和力优先：新30轮实验报告','',f"已保留{len(outcomes)}/30轮。主种子42；动态学习窗口后继续原生推理至1；无Steer重采样。",'',
        '|轮次|集合|奖励|剂量|相对配对无引导均值Δ|有效率|有效最高预测值|',
        '|---:|---|---|---:|---:|---:|---:|']
    for r in rows:lines.append(f"|{r['round']}|{r['split']}|{r['reward_view']}|{r['dose']:.4g}|{r['head_delta']:.6f}|{r['valid_rate']:.1%}|{r['best_valid_head']:.6f}|")
    for label,result in frozen.items():
        c=result['batch_contrast'];g=result['gradient'];n=result['native'];lo,hi=c['batch_bootstrap_CI95']
        lines+=['',f"{label}：{g['attempts']}个引导样本，均值{g['all_head_mean']:.6f}，匹配无引导{n['all_head_mean']:.6f}，Δ{c['mean_head_delta']:+.6f}；批次bootstrap95%区间[{lo:.6f},{hi:.6f}]。",
            f"应变中位数{g['MMFF_relief_per_heavy_median']:.6f} vs {n['MMFF_relief_per_heavy_median']:.6f} kcal/mol/重原子；p90 {g['MMFF_relief_per_heavy_p90']:.6f} vs {n['MMFF_relief_per_heavy_p90']:.6f}。"]
    lines+=['','历史Steer100的均值7.510351、有效最高8.347940分开比较；它不是同预算配对留出组。去重值保留首个出现构象，与历史统计定义一致。',
        '平均Δ接近零也可能来自正负抵消，应结合实际位移、个体绝对Δ与奖励响应判断。控制器归一化梯度方向，因此整体放大奖励数值不等于提高注入剂量；需调整剂量比及实际限幅。',
        '坐标关联不是因果结合区域，真实FLOWR坐标VJP也不是affinity head梯度。应变是孤立配体MMFF松弛，不是结合自由能。']
    (output/'report.zh-CN.md').write_bytes(('\n'.join(lines)+'\n').encode('utf8'))
    return value
