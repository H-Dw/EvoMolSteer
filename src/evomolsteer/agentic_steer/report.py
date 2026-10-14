"""Report realized decisions, module enactment and state-specific outcomes."""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr,beta

from evomolsteer.io import read_json,write_json,write_table
from .workflow import summarize,numeric_feedback


def report(repo,cfg):
    study=Path(repo)/cfg['study'];panel=read_json(study/'panel.json');rows=summarize(repo,cfg)
    cards={x['node_id']:x for x in panel['candidate_cards']};all_bank=[read_json(p) for p in sorted((study/'trials').glob('N*.json'))]
    requested=set()
    for condition in ['C0','B_diversity']+[f'C{i}' for i in range(1,7)]:
        for phase in ['initial','feedback']:
            requested.update(t['trial_id'] for t in read_json(study/'conditions'/condition/(phase+'.job.json'))['trials'])
    bank=[x for x in all_bank if x['trial']['trial_id'] in requested]
    grouped=defaultdict(list)
    for result in bank:grouped[result['trial']['node_id']].append(result)
    nodes=[]
    for node,values in sorted(grouped.items()):
        count=len(values);hits=sum(x['elite_hit'] for x in values)
        nodes.append({'node_id':node,'attempts':count,'head':cards[node]['joint_head_pic50'],
            'mean_final_pic50':float(np.mean([x['pic50'] for x in values])),
            'best_qualified_pic50':max([x['pic50'] for x in values if x['qualified_structure']],default=None),
            'elite_hits':hits,'estimated_probability':hits/count,
            'binomial_ci_low':0. if hits==0 else float(beta.ppf(.025,hits,count-hits+1)),
            'binomial_ci_high':1. if hits==count else float(beta.ppf(.975,hits+1,count-hits)),
            'qualified':sum(x['qualified_structure'] for x in values)})
    correlation=spearmanr([x['head'] for x in nodes],[x['mean_final_pic50'] for x in nodes])
    audits=[];previous=None
    for level in range(1,7):
        condition=f'C{level}';folder=study/'conditions'/condition
        if not (folder/'feedback.execution.json').exists():continue
        initial=read_json(folder/'initial.execution.json');feedback=read_json(folder/'feedback.execution.json')
        initial_nodes={d['node_id'] for d in initial['directions']}
        observed=[{'node_id':x['trial']['node_id'],'replicate':x['trial']['replicate'],**{k:x[k] for k in ['pic50','qualified_structure','elite_hit']}}
            for x in bank if x['trial']['node_id'] in initial_nodes and x['trial']['replicate']<cfg['initial_replicates']]
        baseline_next={(x['node_id'],x['replicate']) for x in numeric_feedback(panel,cfg,observed)}
        actual_next={(x['node_id'],x['replicate']) for x in feedback['next_trials']}
        enacted_pairs=[p for p in initial['comparisons'] if p['left'] in initial_nodes and p['right'] in initial_nodes and p['left']!=p['right']]
        entry={'condition':condition,'added_module':cfg['modules'][level-1],
            'selected_nodes':sorted(initial_nodes),'selection_changed_from_previous':None if previous is None else initial_nodes!=previous,
            'initial_hypotheses':len(initial['hypotheses']),'stated_comparisons':len(initial['comparisons']),'enacted_comparisons':len(enacted_pairs),
            'module_initial_claims':[a for a in initial['audits'] if a['module']==cfg['modules'][level-1]],
            'module_feedback_claims':[a for a in feedback['audits'] if a['module']==cfg['modules'][level-1]],
            'feedback_updates':len(feedback['updates']),'followup_differs_from_numeric_feedback':actual_next!=baseline_next,
            'new_nodes_after_feedback':sorted({x[0] for x in actual_next}-initial_nodes),
            'code_executed':level==6,
            'feedback_changed_code':bool(level==6 and initial['code']!=feedback['code']),
            'scientific_limits':['Cumulative design does not identify an isolated module causal effect',
                'Output-field presence is enactment evidence, not confirmation of chemical truth',
                'Initial responses sometimes call forecast shape current shape; feedback receives explicit definitions']}
        audits.append(entry);previous=initial_nodes
    write_json(study/'module_response_audit.json',audits);write_table(study/'state_outcomes.csv',nodes)
    numbers={'unique_trials':len(bank),'unique_nodes':len(nodes),'logical_trials':sum(x['attempts'] for x in rows),
        'unique_seed_steps':sum(x['suffix_steps'] for x in bank),'elite_hits_in_bank':sum(x['elite_hit'] for x in bank),
        'spearman_head_vs_native_mean':float(correlation.statistic),'nominal_correlation_p':float(correlation.pvalue),
        'correlation_limit':'Selected development panel, unequal small replicate counts and adaptive allocation; not general calibration',
        'max_first_forecast_rms_A':max(x['first_forecast_rms_A'] for x in bank),
        'no_smc_all':all(not x['particle_resampling'] for x in bank),'no_gradient_all':all(not x['gradient_guidance'] for x in bank),
        'energy_converged':sum(x['energy_status']=='converged' for x in bank),
        'source_checkpoint':'flowr_root_v2.ckpt','threshold':panel['objective']['threshold'],
        'total_local_bytes':sum(p.stat().st_size for p in study.rglob('*') if p.is_file())}
    historical=pd.read_parquet(Path(repo)/cfg['terminal_metrics'])
    write_json(study/'historical_context.json',{'n':len(historical),'mean_pic50_all':float(historical.pic50_on_rescore.mean()),
        'best_pic50_all':float(historical.pic50_on_rescore.max()),'valid_connected_rate':float(historical.valid_connected.mean()),
        'pb_fast_pass_rate':float(historical.pb_fast_pass.mean()),
        'frozen_threshold_valid_hits':int(((historical.pic50_on_rescore>=numbers['threshold'])&historical.valid_connected).sum()),
        'comparison_limit':'Original 1000 full Steer samples; different computation, initial states and structure criterion. Context only.'})
    write_json(study/'summary.json',numbers)
    lines=['# Agentic Steer 首轮小预算实验', '',
        '本轮学习的是实际中间状态的后续生成潜力。每条试验从完整快照恢复坐标、类别、prior和self-conditioning，在原生100步网格执行剩余70步；没有后续SMC和gradient guidance。', '',
        f"发现集终态95%分位数冻结为 {numbers['threshold']:.4f} pIC50。命中还要求可解码、连通且没有0.8 Å以下严重受体重叠。此处结构合格是初筛定义，不等于完整PoseBusters合格。MMFF94s松弛能量/重原子仅作为次级指标。", '',
        f"真实DeepSeek deepseek-flash API、默认温度0。八个条件共{numbers['logical_trials']}次逻辑试验，复用后生成{len(bank)}条独立子流、{len(nodes)}个起点，共{numbers['unique_seed_steps']}个seed-step。母随机种子42；每个node/replicate有确定的独立子流。", '',
        '|条件|尝试|命中|全部预测pIC50均值|最佳基础结构合格pIC50|结构合格率|MMFF松弛/重原子均值|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        value=lambda key:'NA' if row[key] is None else f"{row[key]:.4f}"
        lines.append(f"|{row['condition']}|{row['attempts']}|{row['elite_hits']}|{value('mean_pic50_all')}|{value('best_qualified_pic50')}|{row['qualified_rate']:.1%}|{value('mean_mmff_relief_per_heavy')}|")
    lines+=['','## 六个模块如何实施','','|模块|实际初始选择|相对前一LLM条件改变|实际执行的候选对照|追加是否不同于数值反馈基线|','|---|---|---|---:|---|']
    for a in audits:lines.append(f"|{a['added_module']}|{', '.join(a['selected_nodes'])}|{a['selection_changed_from_previous']}|{a['enacted_comparisons']}|{a['followup_differs_from_numeric_feedback']}|")
    code_entry=next((a for a in audits if a['condition']=='C6'),None)
    lines+=['', '每个模块的原始主张、证据引用、实际动作及代码扰动结果分别保存；它们不自动成为事实。K1之外的科学功能也可能被较少模块的Agent自发完成，不能将其归因于新增指令。', '',
        f"K6实际执行了选择程序；反馈后程序是否改变：{code_entry['feedback_changed_code'] if code_entry else '未完成'}。程序执行和程序对反馈响应是两项不同验证。", '',
        '## 概率与解释边界','',
        '每个方向单独保存成功/失败、重复数与二项区间。两次均未命中仍有很宽的上界；不能据此认定该路径没有潜力。自适应追加存在选择与停止偏差，正式确认需要未参与选点的固定重复。不同条件复用的子流不是新的独立样本。', '',
        f"开发面板起点head与原生后续平均评分的Spearman相关为 {numbers['spearman_head_vs_native_mean']:.4f}，只描述本次选点面板，不代表全部历史样本。首预测最大恢复误差 {numbers['max_first_forecast_rms_A']:.3g} Å。", '',
        '历史教师来自终态回溯的多条真实祖先；历史SMC条件下的后代成功不等于关闭SMC后的成功概率。灭绝与未观察未来仍按删失处理。本次没有将LLM文字置信度变成繁殖概率。', '',
        f"原始1000个Steer终态的全部评分均值为 {historical.pic50_on_rescore.mean():.4f}、最高为 {historical.pic50_on_rescore.max():.4f}，仅作背景。计算预算、初始状态及结构判据不同，本轮不能作同预算优劣结论。", '',
        '初始回答出现过将端点forecast特征称为current特征、将面板构建代码当成预算策略的解释风险；反馈阶段额外提供了实际策略代码与表示定义。K6在执行前检查raw单位、注册字段、纯函数AST和数值扰动；raw API提议与实际执行排序分别保存。', '',
        '## 后续递增验证','',
        '先使用新的固定子流确认优先方向，并加入未被初始选中的方向作为实际对照；再做单模块完整/删除对比。获得命中后比较方向档案与全局top-k的保留差异，使用真实反馈/置乱反馈检验适应功能。最后才扩展到多个恢复时刻、完整树搜索和同预算Steer/BoN；本轮不能证明已超过原Steer。']
    probe_file=study/'probes/scientific_contract_v2/response_check.json'
    if probe_file.exists():
        probe=read_json(probe_file)
        lines+=['','## 小型纠正实验','',
            f"补充完成状态/化学失败标签、真实策略代码、欧氏距离反例及确定性均值后，新的真实API检查四项误解全部纠正：{probe['omissions_counterexamples_correct']}。选择程序数值与top4排序均发生变化：{probe['numeric_scores_changed']} / {probe['ranking_changed']}。这证明输入和Skills修改得到响应，不能仅凭响应宣称质量改善。"]
    refined=study/'refinement_v2/results.json'
    if refined.exists():
        result=read_json(refined)
        lines += ['',f"修正程序依据先前12条C6观察，又请求4条续生成（其中{result['new_unique_trials']}条新独立子流，其余复用）。新增请求的全部pIC50均值为{result['mean_pic50']:.4f}、命中{result['elite_hits']}/4、基础结构合格{result['qualified']}/4。它是第二轮开发筛查，使用了更多先验反馈，不与原12次条件作同预算优劣声明。"]
    (study/'report.zh-CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return numbers


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--repo',default='.');a=p.parse_args();print(report(a.repo,read_json(a.config)))


if __name__=='__main__':main()
