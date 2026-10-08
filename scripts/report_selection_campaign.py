"""Summarize the predeclared experiment; no validation-based reward retuning."""
import argparse
from pathlib import Path
import pandas as pd
from evomolsteer.io import read_json,write_json,digest
from evomolsteer.generation.path_evaluation import summarize_tail,paired_effect
from dispatch_path_round import verify_retention


def report(root):
    root=Path(root).resolve();docs=root/'docs/experiments/selection_path20_20261008'
    cfg=root/'configs/experiments/selection_path20_v1';campaign=read_json(cfg/'campaign.json')
    if campaign['rounds_completed']!=20:raise ValueError('All twenty registered rounds must finish')
    threshold=read_json(docs/'protocol.json')['tail_threshold_pic50'];records=[]
    for n in range(1,21):
        verify_retention(docs/f'round{n:02d}/retention.json')
        result=read_json(docs/f'round{n:02d}/comparison.json');plan=read_json(docs/f'round{n:02d}_plan.json')
        m=result['results']['gradient']
        baseline_native=result['results'].get('unguided')
        records.append({'round':n,'reason':plan['reason'],'batches':str(plan['batches']),
          'mean_pic50':m['all_mean_pic50'],'valid_mean_pic50':m['valid_mean_pic50'],
          'vs_R26':result.get('versus_gradient',{}).get('paired_mean_pic50'),
          'vs_native':result.get('versus_unguided',{}).get('paired_mean_pic50',m['all_mean_pic50']-baseline_native['all_mean_pic50'] if baseline_native else None),
          'valid_n':m['valid_n'],'n':m['n'],'pb_fast_rate':m['pb_fast_rate'],
          'max_pic50':m['valid_max_pic50'],'elite_graphs':m['elite_unique_graphs'],
          'strain_median':m['strain_median_per_heavy'],'strain_p90':m['strain_p90_per_heavy'],
          'screening_admissible':campaign['rounds'][n-1]['screening_admissible']})
    pd.DataFrame(records).to_csv(docs/'rounds.csv',index=False)
    def concat(numbers,arm):
        frames=[pd.read_csv(docs/f'round{n:02d}/candidate_metrics.csv') for n in numbers]
        result=pd.concat(frames,ignore_index=True);return result[result.arm.eq(arm)].copy()
    candidate=concat([18,20],'gradient');baseline=concat([17,19],'gradient');native=concat([17,19],'unguided')
    m={label:summarize_tail(d,threshold) for label,d in [('candidate',candidate),('R26',baseline),('native',native)]}
    vsbase=paired_effect(candidate,baseline);vsnative=paired_effect(candidate,native)
    steer=read_json(docs/'steer_reference.json');frozen=read_json(docs/'frozen_validation.json')
    strain_limit=1.3*max(m['R26']['strain_median_per_heavy'],steer['all']['strain_median_per_heavy'])
    checks={'paired_R26_mean_gt_0_01':vsbase['paired_mean_pic50']>.01,
      'batch_bootstrap_lower_gt_0':vsbase['batch_bootstrap_CI95'][0]>0,
      'native_mean_improved':vsnative['paired_mean_pic50']>0,
      'validity_retained':m['candidate']['valid_n']/m['candidate']['n']>=m['R26']['valid_n']/m['R26']['n']-.03,
      'PB_retained':m['candidate']['pb_fast_rate']>=m['R26']['pb_fast_rate']-.03,
      'secondary_strain_within_limit':m['candidate']['strain_median_per_heavy']<=strain_limit}
    outcome={'frozen_candidate':frozen,'confirmation_results':m,'versus_R26':vsbase,'versus_native':vsnative,
      'historical_Steer':steer,'acceptance_checks':checks,'accepted':all(checks.values()),
      'strain_limit_per_heavy':strain_limit,'active_workflow':read_json(cfg/'active_workflow.json'),
      'comparison_limits':['Historical Steer is unpaired, unequal compute and donor-influenced.',
        'Four fixed-seed independent batches provide limited uncertainty resolution; no wet-lab claim.',
        'All-attempt mean and chemically valid mean are both reported.'],
      'registered_attempts':2300,'completed_rounds':20}
    write_json(docs/'final_comparison.json',outcome)
    lines=['# R26选择路径20轮探索结果','',f"冻结候选来自第{frozen['selected_round']}轮；独立确认判定：{'通过注册门槛' if outcome['accepted'] else '未通过注册门槛，保持R26默认'}。",
      '', '## 同批次独立确认（每组200个）','',
      '|组别|全部尝试平均pIC50|有效分子平均pIC50|有效数|PB-fast|最大pIC50|极高亲和力不同图数|应变中位数/重原子|应变P90|',
      '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for label,values in [(k,m[k]) for k in ['native','R26','candidate']]+[('历史Steer（1000，非配对）',steer['all']),('历史Steer非donor（300）',steer['non_donors'])]:
        lines.append(f"|{label}|{values['all_mean_pic50']:.6f}|{values['valid_mean_pic50']:.6f}|{values['valid_n']}/{values['n']}|{values['pb_fast_rate']:.3f}|{values['valid_max_pic50']:.6f}|{values['elite_unique_graphs']}|{values['strain_median_per_heavy']:.6f}|{values['strain_p90_per_heavy']:.6f}|")
    lines+=['',f"候选相对R26配对均值变化：{vsbase['paired_mean_pic50']:+.6f}；四批变化：{vsbase['batch_means']}；批次bootstrap 95%区间：{vsbase['batch_bootstrap_CI95']}。",
      f"候选相对无引导配对均值变化：{vsnative['paired_mean_pic50']:+.6f}。极高亲和力门槛沿用先前冻结的{threshold:.6f}。",
      '', '## 各轮方向与最终结果','', '|轮次|独立变化/目的|平均pIC50|对R26|对无引导|有效/PB|应变中位数|筛选合格|', '|---|---|---:|---:|---:|---|---:|---|']
    for r in records:
        rb='—' if r['vs_R26'] is None else f"{r['vs_R26']:+.6f}";rn='—' if r['vs_native'] is None else f"{r['vs_native']:+.6f}"
        lines.append(f"|{r['round']}|{r['reason']}|{r['mean_pic50']:.6f}|{rb}|{rn}|{r['valid_n']}/{r['n']}; {r['pb_fast_rate']:.2f}|{r['strain_median']:.4f}|{r['screening_admissible']}|")
    lines+=['', '每轮失败及暂时合格后均恢复R26程序、基础Skills与数值源文件校验；第18/20轮只确认提前冻结的同一候选，没有根据确认标签调参。没有粒子重采样、亲和力头梯度或化学图限制。全部推理完成100步，动态支持内引导，之后原生续推。',
      '', '700个Steer选择事件的几何群体变化分析显示原生趋势68项BH显著、选择项0项，不能把原生积分规律称为优势区域。质量排序、ESS、模式覆盖、局部度量和稳健损失均为待验证机制。方法、来源、公式和限制见[研究设计](research_and_design.zh-CN.md)；精简数据见mining目录；真实Agent字面输入、响应和修复记录见agents目录。',
      '', '历史Steer不是同随机状态、同计算量对照；教师donor参与奖励设计，因此全体Steer与非donor子集分别报告。当前验证是固定主种子下四个批次，不能外推为实验结合活性结论。',
      '', '共完成20轮、2300次完整模型生成尝试。原始Steer/checkpoints保留；每轮报告通过校验、发布后清理生成轨迹，最终清理审计另存。']
    (docs/'report.zh-CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    print({'accepted':outcome['accepted'],'candidate_round':frozen['selected_round'],'vs_R26':vsbase,'vs_native':vsnative,'checks':checks})
    return outcome


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',default='.');report(p.parse_args().repo)
