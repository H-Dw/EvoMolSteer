"""Compare locally evaluated regional cohorts with reused frozen controls."""
import argparse
from pathlib import Path
import pandas as pd
from evomolsteer.io import read_json,write_json,digest
from evomolsteer.ablation.statistics import summarize,compare


def report(repo,study,output):
    repo=repo.resolve();study=study.resolve();output=output.resolve();output.mkdir(parents=True,exist_ok=True)
    bank=read_json(repo/'docs/experiments/m2_tools_full_drop_ablation_20261012/shared_control_bank/manifest.json')
    threshold=read_json(repo/'configs/experiments/m2_terminal_regions_20261014/analysis.json')['evaluation']['elite_threshold_pic50']
    controls={name:pd.read_csv(repo/bank['result']['outcomes'][arm]['metrics']) for name,arm in [('M2','gradient'),('unguided','unguided')]}
    historical_path=repo/'docs/experiments/guidance_vs_steer_20261007/steer_full1000/candidate_metrics.csv'
    historical=pd.read_csv(historical_path)
    state=read_json(study/'collection_status.json') if (study/'collection_status.json').exists() else None
    candidates={}
    for path in sorted((study/'evaluated').glob('M2REG_terminal_*/candidate_metrics.csv')):
        if state is not None and path.parent.name not in state['completed']:continue
        candidates[path.parent.name]=pd.read_csv(path)
    rows=[];comparisons={}
    frames=dict(controls,**candidates,historical_Steer=historical)
    for name,frame in frames.items():
        summary=summarize([frame],threshold)['policy_mean']
        rows.append({'condition':name,'n':len(frame),**summary,'p99_pic50':float(frame.pic50_on_rescore.quantile(.99)),
            'best_valid_pic50':float(frame.loc[frame.valid_connected,'pic50_on_rescore'].max())})
    for name,frame in candidates.items():
        comparisons[name]={k:compare([frame],[v],family=max(1,len(candidates)*2)) for k,v in controls.items()}
    names=list(candidates)
    if len(names)==4:
        local={name:frame for name,frame in candidates.items() if '_local_' in name}
        for name,frame in local.items():
            other=name.replace('_local_','_global_')
            comparisons[name]['local_minus_same_window_global']=compare([frame],[candidates[other]],family=4)
        for mode in ['local','global']:
            early=next(n for n in names if f'_{mode}_0_0p5' in n)
            late=next(n for n in names if f'_{mode}_0p1_0p5' in n)
            comparisons[late]['delayed_minus_full']=compare([candidates[late]],[candidates[early]],family=4)
    pd.DataFrame(rows).to_csv(output/'performance.csv',index=False)
    write_json(output/'comparison.json',{'comparisons':comparisons,'summary':rows,
        'historical_steer':'Unpaired reference with different search budget and dependent descendants',
        'threshold':threshold,'candidate_programs':len(candidates),'failed_conditions':state.get('failed',{}) if state else {},
        'source_control_manifest_sha256':digest(repo/'docs/experiments/m2_tools_full_drop_ablation_20261012/shared_control_bank/manifest.json'),
        'historical_metrics_sha256':digest(historical_path),'evaluation':'All attempts, failures retained; final predicted pIC50, not measured affinity'})
    text=['# M2 终态教师与局部区域实验结果','',
        f'已完成并通过输入配对核验的候选：{len(candidates)}/4。',
        '预测 pIC50 为主要指标，应变为收敛 MMFF94s 局部松弛能差/重原子数；失败保留在全部尝试分母。','',
        '| 条件 | N | 平均 pIC50 | p99 | 应变 p90 | 有效比例 | 有效高亲和力比例 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        strain=f"{r['strain_p90']:.5f}" if r['strain_p90'] is not None else '缺失'
        text.append(f"| {r['condition']} | {r['n']} | {r['mean_pic50']:.5f} | {r['p99_pic50']:.5f} | {strain} | {r['valid_rate']:.2%} | {r['elite_valid_rate']:.2%} |")
    text += ['', '原 Steer 为不同搜索预算的历史非配对参照；不能仅凭最大值或未匹配的输出数量判断因果效果。',
        'M2、无引导的固定400样本直接复用，运行版本与初始输入通过桥接/逐组核验。','']
    if len(candidates)<4:text.append('四个条件尚未全部完成；目前不能对 Agent 局部引导或延迟窗口作完整结论。')
    for name,values in comparisons.items():
        text += ['',f'**{name}**']
        for label,c in values.items():
            lo,hi=c['simultaneous_CI']; verdict='区间跨零，差异尚不明确' if lo<=0<=hi else ('正向差异' if lo>0 else '负向差异')
            text.append(f"- 相对 {label}：ΔpIC50={c['delta_pic50']:+.5f}，同时区间 [{lo:+.5f}, {hi:+.5f}]，{verdict}。")
    if state and state.get('failed'):
        text += ['','失败条件单独记录：'+', '.join(state['failed'])+'；没有替换成 M2 后冒充候选结果。']
    (output/'report.zh-CN.md').write_text('\n'.join(text)+'\n',encoding='utf-8',newline='\n')
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',default='.',type=Path)
    p.add_argument('--study',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();print(report(a.repo,a.study,a.output))
