"""Summarize the predeclared experiment; no validation-based reward retuning."""
import argparse
import hashlib
import subprocess
from pathlib import Path
import pandas as pd
from evomolsteer.io import read_json,write_json,digest
from evomolsteer.generation.path_evaluation import summarize_tail,paired_effect
from dispatch_path_round import verify_retention
from evomolsteer.generation.selection_workflow import activate_confirmed,restore_incumbent

ROUND_DIRECTIONS={
 1:'建立无引导/R26配对控制',2:'质量标签排序，β=2',3:'降低排序先验强度，β=1',
 4:'教师先验ESS下限50%',5:'教师先验ESS下限75%',6:'仅旧归一化实现的负对照',
 7:'几何模式计数去偏，强度0.5（早期实现）',8:'几何模式计数去偏，强度1',
 9:'优先选择每个几何模式的代表',10:'优先覆盖不同教师来源批次',
11:'收缩协方差坐标度量，混合0.1',12:'协方差混合0.25',13:'协方差混合0.5',
14:'逐原子稳健坐标误差，δ=1Å',15:'逐原子稳健误差曲率，δ=2Å',
16:'用修正后实现重测最佳筛选参数',17:'冻结候选，建立确认批次43/44的控制',
18:'冻结候选在批次43/44的确认',19:'确认批次45/46的控制',20:'同一冻结候选在批次45/46的确认'}


def confirmation_source_audit(root, docs):
    """Numerical source, reference and paired initial states must stay fixed."""
    source_paths=['src/evomolsteer/generation/'+name for name in
        ['selection_path_reward.py','endpoint_reward.py','endpoint_controller.py',
         'coordinate_contrast.py','scalar_guidance.py']]
    hashes={path:digest(root/path) for path in source_paths};rounds={}
    frozen=read_json(docs/'frozen_validation.json')
    expected_programs={
        'control':read_json(root/'configs/experiments/skill_ablation_v1/incumbent.json'),
        'candidate':read_json(root/f"configs/experiments/selection_path20_v1/round{frozen['selected_round']:02d}.json")}
    def numerical_program(program):
        return {key:value for key,value in program.items() if key not in ['round','program_id','derivation']}
    for n in [17,18,19,20]:
        report=read_json(docs/f'round{n:02d}/execution_report.json')
        commit=report['code_commit']
        program=read_json(docs/f'round{n:02d}/inference_config/reward_program.json')
        expected=expected_programs['control' if n in [17,19] else 'candidate']
        if numerical_program(program)!=numerical_program(expected):
            raise ValueError(f'Confirmation reward parameters differ: round {n}')
        if report['window']!=program['window']:
            raise ValueError('Confirmation guidance support differs')
        for path,sha in hashes.items():
            data=subprocess.check_output(['git','show',f'{commit}:{path}'],cwd=root)
            if hashlib.sha256(data).hexdigest()!=sha:
                raise ValueError(f'Confirmation numerical source changed: round {n}, {path}')
        if (not report['no_particle_resampling'] or report['affinity_head_gradient']
                or report['outside_window_injection'] or report['steps']!=100
                or report['additional_production_calls_per_step']!=0):
            raise ValueError('Confirmation generation audit failed')
        rounds[n]=report
    for control,candidate in [(17,18),(19,20)]:
        for key,sha in rounds[candidate]['initial_state_signatures'].items():
            batch=key.split('/')[-1]
            for arm in ['gradient','unguided']:
                if rounds[control]['initial_state_signatures'][f'{arm}/{batch}']!=sha:
                    raise ValueError('Confirmation paired initial states differ')
    return {'numerical_source_sha256':hashes,
            'inference_commits':{str(n):r['code_commit'] for n,r in rounds.items()},
            'same_initial_states':True,'same_frozen_reward_parameters':True,
            'no_resampling_or_affinity_gradient':True}


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
    baseline_vs_native=paired_effect(baseline,native)
    for effect in [vsbase,vsnative,baseline_vs_native]:
        effect['limitation']='Frozen proposal on four new fixed-seed batches; batch bootstrap has limited resolution and does not establish broader-seed or experimental affinity generalization.'
    steer=read_json(docs/'steer_reference.json');frozen=read_json(docs/'frozen_validation.json')
    strain_limit=1.3*max(m['R26']['strain_median_per_heavy'],steer['all']['strain_median_per_heavy'])
    tail_limit=1.3*max(m['R26']['strain_p90_per_heavy'],steer['all']['strain_p90_per_heavy'])
    checks={'matched_implementation_screen_admissible':frozen['screening_admissible'],
      'paired_R26_mean_gt_0_01':vsbase['paired_mean_pic50']>.01,
      'batch_bootstrap_lower_gt_0':vsbase['batch_bootstrap_CI95'][0]>0,
      'native_mean_improved':vsnative['paired_mean_pic50']>0,
      'validity_retained':m['candidate']['valid_n']/m['candidate']['n']>=m['R26']['valid_n']/m['R26']['n']-.03,
      'PB_retained':m['candidate']['pb_fast_rate']>=m['R26']['pb_fast_rate']-.03,
      'secondary_strain_within_limit':m['candidate']['strain_median_per_heavy']<=strain_limit,
      'secondary_strain_p90_within_limit':m['candidate']['strain_p90_per_heavy']<=tail_limit}
    source_audit=confirmation_source_audit(root,docs)
    active=(activate_confirmed(root,'selection_path20_v1',frozen,checks) if all(checks.values())
        else restore_incumbent(root,'selection_path20_v1','Frozen independent confirmation failed; retain historical R26.',20))
    outcome={'frozen_candidate':frozen,'confirmation_results':m,'versus_R26':vsbase,'versus_native':vsnative,
      'R26_versus_native':baseline_vs_native,
      'historical_Steer':steer,'acceptance_checks':checks,'accepted':all(checks.values()),
      'strain_limit_per_heavy':strain_limit,'strain_p90_limit_per_heavy':tail_limit,'active_workflow':active,
      'confirmation_source_audit':source_audit,
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
      f"R26相对无引导配对均值变化：{baseline_vs_native['paired_mean_pic50']:+.6f}；四批变化：{baseline_vs_native['batch_means']}；批次bootstrap 95%区间：{baseline_vs_native['batch_bootstrap_CI95']}。",
      '', '## 各轮方向与最终结果','', '|轮次|独立变化/目的|平均pIC50|对R26|对无引导|有效/PB|应变中位数|筛选合格|', '|---|---|---:|---:|---:|---|---:|---|']
    for r in records:
        rb='—' if r['vs_R26'] is None else f"{r['vs_R26']:+.6f}";rn='—' if r['vs_native'] is None else f"{r['vs_native']:+.6f}"
        lines.append(f"|{r['round']}|{ROUND_DIRECTIONS[r['round']]}|{r['mean_pic50']:.6f}|{rb}|{rn}|{r['valid_n']}/{r['n']}; {r['pb_fast_rate']:.2f}|{r['strain_median']:.4f}|{'是' if r['screening_admissible'] else '否'}|")
    lines+=['', '每轮失败及暂时合格后均恢复R26程序、基础Skills与数值源文件校验；第18/20轮只确认提前冻结的同一候选，没有根据确认标签调参。没有粒子重采样、亲和力头梯度或化学图限制。全部推理完成100步，动态支持内引导，之后原生续推。',
      '', '数值一致性审计改变了探索解释：第6轮没有新学习规则，也重现了第4轮约+0.02518的提升；第4轮相对这个负对照的差值仅-0.00000285。早期结果不能仅归因于ESS或排序规则。随后统一R26归一化方式，消除无信息的niche常量偏移和单位协方差效应；第7轮在额外修复前启动，仅作探索证据。最终候选只从第8–16轮选取，第16轮负责用当前实现重新评估最佳参数。修订在揭示独立确认标签前注册，原版本报告与提交均保留。',
      '', '700个Steer选择事件的几何群体变化分析显示原生趋势68项BH显著、选择项0项，不能把原生积分规律称为优势区域。质量排序、ESS、模式覆盖、局部度量和稳健损失均为待验证机制。方法、来源、公式和限制见[研究设计](research_and_design.zh-CN.md)；精简数据见mining目录；真实Agent字面输入、响应和修复记录见agents目录。',
      '', '奖励标量、条件坐标导数、协方差度量及对应源码字段见[奖励数学定义](reward_formulas.zh-CN.md)。选择项未显著表示当前检验未发现证据，不等于不存在可利用信号。教师权重ESS与独立家族数分别解释。应变使用MMFF弛豫能差除以重原子数，是几何能量诊断，不是结合自由能。',
      '', '第一组确认退化的直接证据、稀有模式权重与质量标签的竞争，以及后续可检验原因见[失败解释](failure_interpretation.zh-CN.md)。其中原因保持假设表述；确认过程中没有改变冻结参数。',
      '', '历史Steer不是同随机状态、同计算量对照；教师donor参与奖励设计，因此全体Steer与非donor子集分别报告。当前验证是固定主种子下四个批次，不能外推为实验结合活性结论。',
      '', '确认同时检查应变中位数与P90，二者均不超过R26和历史Steer相应值较大者的1.3倍。该次级质量门槛在确认标签揭示前补充，不在推理中限制化学图或筛选粒子。第2轮没有保留新增的后验ESS汇总，该诊断缺失已记录；分数、结构质量、配对及无重采样审计仍完整。',
      '', '共完成20轮、2300次完整模型生成尝试。原始Steer/checkpoints保留；每轮报告通过校验、发布后清理生成轨迹。最终[清理审计](final_cleanup.json)、[远端完成核验](completion_verification.json)与[本地验证记录](verification.json)均保存。',
      '', '![各轮探索与独立确认](rounds_analysis.png)']
    (docs/'report.zh-CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    print({'accepted':outcome['accepted'],'candidate_round':frozen['selected_round'],'vs_R26':vsbase,'vs_native':vsnative,'checks':checks})
    return outcome


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',default='.');report(p.parse_args().repo)
