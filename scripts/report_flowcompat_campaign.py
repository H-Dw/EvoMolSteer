"""Compact honest progress and independent confirmation from retained reports."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from evomolsteer.io import read_json,write_json
from evomolsteer.generation.path_evaluation import paired_effect,summarize_tail

def report(repo):
    root=Path(repo).resolve();cfg=root/'configs/experiments/flowcompat30_v1'
    docs=root/'docs/experiments/flowcompat30_20261009';state=read_json(cfg/'campaign.json')
    records=[]
    for entry in state['rounds']:
        n=entry['round'];d=read_json(docs/f'round{n:02d}/comparison.json')
        plan=read_json(docs/f'round{n:02d}_plan.json');m=d['results']['gradient']
        records.append({'round':n,'mean_pic50':m['all_mean_pic50'],
            'vs_R26':d.get('versus_gradient',{}).get('paired_mean_pic50'),
            'vs_native':d.get('versus_unguided',{}).get('paired_mean_pic50'),
            'valid_rate':m['valid_n']/m['n'],'pb_rate':m['pb_fast_rate'],
            'strain_median':m['strain_median_per_heavy'],'strain_p90':m['strain_p90_per_heavy'],
            'elite_n':m['elite_valid_n'],'maximum_pic50':m['valid_max_pic50'],
            'implementation_passed':entry['implementation_feedback'].get('implementation_passed'),
            'screening_admissible':entry['screening_admissible'],'parent':str(plan['parent']),
            'axis':str(plan['changed_axis']),'n':m['n'],'inference_commit':read_json(docs/f'round{n:02d}/retention.json')['inference_commit']})
    table=pd.DataFrame(records);table.to_csv(docs/'rounds_summary.csv',index=False)
    confirmation={'complete':False,'adopt_candidate':False}
    if state['rounds_completed']==30:
        controls=pd.concat([pd.read_csv(docs/f'round{n:02d}/candidate_metrics.csv') for n in [25,27,29]])
        candidates=pd.concat([pd.read_csv(docs/f'round{n:02d}/candidate_metrics.csv') for n in [26,28,30]])
        r26=controls[controls.arm.eq('gradient')];native=controls[controls.arm.eq('unguided')]
        a=paired_effect(candidates,r26);b=paired_effect(candidates,native)
        metrics={key:summarize_tail(value,read_json(docs/'protocol.json')['tail_threshold_pic50']) for key,value in [('candidate',candidates),('R26',r26),('native',native)]}
        steer=read_json(docs/'steer_reference.json')['all'];m=metrics['candidate'];baseline=metrics['R26']
        checks={'positive_independent_effect':a['paired_mean_pic50']>.01 and a['batch_bootstrap_CI95'][0]>0,
            'positive_vs_native':b['paired_mean_pic50']>0,
            'validity':m['valid_n']/m['n']>=baseline['valid_n']/baseline['n']-.03,
            'pose_quality':m['pb_fast_rate']>=baseline['pb_fast_rate']-.03,
            'strain_median':m['strain_median_per_heavy']<=1.3*max(baseline['strain_median_per_heavy'],steer['strain_median_per_heavy']),
            'strain_p90':m['strain_p90_per_heavy']<=1.3*max(baseline['strain_p90_per_heavy'],steer['strain_p90_per_heavy']),
            'screened_before_confirmation':read_json(docs/'frozen_validation.json')['screening_admissible'],
            'implementation':all(read_json(docs/f'round{n:02d}/implementation_feedback.json').get('implementation_passed',False) for n in [26,28,30])}
        confirmation={'complete':True,'adopt_candidate':all(checks.values()),'checks':checks,
            'metrics':metrics,'versus_R26':a,'versus_native':b,'historical_Steer_unpaired':steer,
            'frozen_candidate':read_json(docs/'frozen_validation.json')}
    write_json(docs/'confirmation.json',confirmation)
    lines=[f'# FLOWR 坐标引导探索：{state["rounds_completed"]}/30轮',
        '', '本轮数包括必要的无信息控制、方向反证和冻结验证，并非30次独立LLM设计。每轮均实际推理，保留提交与生成版本；主种子42，学习窗口由输入声明，窗口结束后原生续推至1。',
        '', '初始根谱系坍缩不能排除后续条件变异。两步共同祖先工具揭示在线评分与空间再预测创新的关联，区域方向仍需独立批次、绝对效应和反方向实验支撑。LLM解释受工具证据与注册公式约束，未训练新的亲和力模型。',
        '', '实际FLOWR端点VJP使用一次正常前向与一次反传，冻结自条件及点云对应。模块输出梯度只是奖励的敏感度，不能解释为亲和力或注意力的因果贡献。',
        '', '历史Steer的1000分子均值7.65148、最高8.70481；原始设计供体参与奖励构建且计算预算不同，因此作为非配对参照，不能据此宣布优于Steer。',
        '', '|轮|评分均值|相对R26|相对native|有效/PB|应变中位/P90|实施/筛选|',
        '|---|---:|---:|---:|---|---|---|']
    def number(value):return '—' if value is None or pd.isna(value) else f'{value:+.5f}'
    for r in records:
        lines.append(f'|{r["round"]}|{r["mean_pic50"]:.5f}|{number(r["vs_R26"])}|{number(r["vs_native"])}|{r["valid_rate"]:.0%}/{r["pb_rate"]:.0%}|{r["strain_median"]:.4f}/{r["strain_p90"]:.4f}|{r["implementation_passed"]}/{r["screening_admissible"]}|')
    lines+=['', '未通过实施门的终态数值仅作描述；未通过筛选的试验不覆盖R26默认。六个新批次确认后才可判断是否升级。', '',
        '独立验证完成：'+str(confirmation['complete'])+'；满足升级条件：'+str(confirmation['adopt_candidate'])+'.', '',
        '参考：[FK steering](https://arxiv.org/abs/2501.06848)说明群体重采样如何保留有望得到高奖励的路径；[Flow guidance](https://proceedings.mlr.press/v267/feng25s.html)说明一般flow guidance的条件与近似边界。当前实现属于有界条件端点控制，不声称精确FK分布或完整未来价值梯度。']
    (docs/'progress.zh-CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return confirmation

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',default='.')
    report(p.parse_args().repo)
