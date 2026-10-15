"""Rebuild the scientific report from immutable programs and actual outcomes."""
import argparse
import copy
import gzip
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from evomolsteer.io import read_json,write_json,write_table,digest
from .evaluation import score_records,energy
from .reward import MotifReward
from .semantic_contract import identity


def elemental_diagnostics(folder,p,t):
    """Geometry-only matching; also follow the actual boundary-controlled slots."""
    geometry=read_json(folder/'geometry.json');audit=read_json(folder/'guidance_audit.json');rows=[]
    boundary=next((s for s in reversed(audit['steps']) if 'boundary_generated_slots' in s),{})
    tracked=boundary.get('boundary_generated_slots')
    for stage,key in [('boundary','teacher_clock'),('final','terminal')]:
        state=geometry[key];q=copy.deepcopy(p);q.update(assignment_element_cost=0.,target_schedule='fixed',representation='bridge' if stage=='boundary' else 'endpoint')
        reward=MotifReward(q,t);xyz=torch.tensor(state['coords_A'],dtype=torch.double)
        atom=np.asarray(state['atomic_indices']);mask=torch.tensor(state.get('mask',geometry['terminal']['mask']),dtype=torch.bool)
        logits=torch.zeros((*atom.shape,max(t['atom_vocabulary'].values())+1),dtype=torch.double)
        _,detail=reward.evaluate(xyz,logits,mask);ids=detail['assignment'].numpy()
        desired=np.asarray(reward.labels);names=np.array(t['native_elements'])[p['atom_slots']]
        for i in range(len(xyz)):
            for element in sorted(set(names)):
                chosen=names==element;actual=atom[i,ids[i]]
                rows.append({'slot':i,'stage':stage,'source_element':element,'source_n':int(chosen.sum()),
                    'geometry_matched_identity_fraction':float((actual[chosen]==desired[chosen]).mean()),
                    'boundary_controlled_slot_identity_fraction':float((atom[i,np.asarray(tracked[i])[chosen]]==desired[chosen]).mean()) if tracked is not None else None,
                    'point_rms_A':float(detail['point_rms_A'][i])})
    return rows


def build(repo,config):
    repo=Path(repo).resolve();m=read_json(repo/config);study=repo/m['campaign_study'];state=read_json(study/'state.json');out=study/'summary'
    out.mkdir(parents=True,exist_ok=True);tau=state['original_steer']['tau'];metric_rows=[];decisions=[];diagnostics=[];bindings=[]
    sources={};programs=[]
    def add(label,folder,p=None,t=None,stage='from_zero',iteration=None):
        folder=Path(folder);rec=folder/'records.json';sources[rec.relative_to(repo).as_posix()]=digest(rec)
        metric=score_records(read_json(rec),tau);er=energy(folder,out/'quality'/f'{label}.energy.json')
        row={'condition':label,'stage':stage,'effective_iteration':iteration,**metric,
             'energy_coverage':er['coverage'],'median_heavy_relaxation_per_atom':er['median'],'p90_heavy_relaxation_per_atom':er['p90']}
        top=max((r for r in er['rows'] if r.get('qualified') and r.get('pic50') is not None),key=lambda r:r['pic50'],default={})
        row['highest_affinity_energy']=top.get('heavy_relaxation_relief_after_H')
        if p:
            row.update(operator=p['operator'],n_support=len(p['atom_slots']),n_teacher=len(t['coords_A']),
                control_start=p['window'][0],control_end=p['window'][1],rho=p['dose']['native_rms_ratio'],
                curvature_A=p['dose']['pointcloud_delta_A'],controller=p['controller'],target_schedule=p.get('target_schedule','fixed'),
                dose_schedule=p.get('time_schedule','constant'),representation=p['representation'])
            if (folder/'geometry.json').exists():diagnostics.extend(dict(condition=label,**r) for r in elemental_diagnostics(folder,p,t))
        metric_rows.append(row);return row
    reference=state['reference'];rp=reference['program'];templates=read_json(study/'templates.json')
    add('S0_REFERENCE_REPLAY',repo/reference['result_folder'],rp,templates[rp['template_id']],iteration=0)
    add('UNGUIDED_b0_n24',repo/m['generated_root']/'S0_REFERENCE_REPLAY/unguided',stage='control')
    for r in state['rounds']:
        label=r['round'];folder=study/label;decision=read_json(folder/'decision.json') if (folder/'decision.json').exists() else {}
        a=decision.get('Analyst',{});d=decision.get('Designer') or {}
        decisions.append({'effective_iteration':r.get('effective_iteration'),'source_round':label,'stage':r['stage'],'status':r['status'],
            'parent':r.get('parent'),'factor':r.get('factor'),'region':r.get('region'),'operator':r.get('operator'),
            'hypothesis':a.get('scientific_hypothesis'),'execution_prediction':a.get('predicted_execution_response'),
            'falsifier':a.get('falsifier'),'designer_rationale':d.get('scientific_rationale'),
            'parameter_changes':d.get('parameter_changes'),'evidence_refs':a.get('evidence_refs'),
            'rollback':r.get('rollback') or r.get('effective_rollback',False),'inference_commit':r.get('commit'),
            'new_attempts':r.get('new_attempts',0),'metrics':r.get('metrics'),'paired':r.get('paired')})
        if (folder/'delivery.json').exists():
            delivery=read_json(folder/'delivery.json');bindings.append({'condition':label,**delivery})
        p=r.get('program')
        if not p:continue
        teacher_path=next((folder/'candidates').glob('*/teacher.json'),None)
        t=read_json(teacher_path) if teacher_path else templates[p['template_id']]
        programs.append({'condition':label,'numerical_signature':identity(p,t),'program':p})
        if r.get('result_folder'):
            add(label,repo/r['result_folder'],p,t,r['stage'],r.get('effective_iteration'))
        for j,trial in enumerate(r.get('trials',[])):
            add(label+'_stream'+str(j),repo/trial['result_folder'],p,t,'suffix',r.get('effective_iteration'))
    # Controls are fixed artifacts, with the same failure denominator as guided.
    for name in ['controls_b%d_n%d'%(m['validation_batch'],m['confirmation_samples'])]:
        folder=repo/m['generated_root']/name/'unguided'
        if (folder/'records.json').exists():add(name,folder,stage='control')
    oldug=repo/'data/generated/partial_region_search_20261014/controls_batch0_n50/unguided'
    if (oldug/'records.json').exists():add('REUSED_UNGUIDED_b0_n50',oldug,stage='control_legacy_metadata')
    write_table(out/'conditions.csv',metric_rows);write_table(out/'elemental_diagnostics.parquet',diagnostics)
    write_json(out/'scientific_decisions.json',decisions);write_json(out/'agent_delivery.json',bindings);write_json(out/'executable_programs.json',programs)
    actual=[]
    for path in (repo/m['generated_root']).glob('*/*/records.json'):
        actual.append({'path':path.relative_to(repo).as_posix(),'n':len(read_json(path)),'sha256':digest(path)})
    costs={'effective_conditions':len(state['rounds']),'duplicate_zero_generation_proposals':len(state.get('duplicate_proposals',[])),
        'actual_generated_attempts':sum(x['n'] for x in actual),'condition_records':actual,
        'real_api_response_receipts':len(list(study.rglob('*.receipt.json'))),'main_role_pairs_with_delivery':len(bindings)}
    write_json(out/'cost_and_provenance.json',costs);write_json(out/'source_hashes.json',sources)
    compare=[r for r in metric_rows if r['stage'] in ('from_zero','confirmation')]
    peak=max(compare,key=lambda r:r['best_qualified_pic50'] or -np.inf,default={})
    exploration=max((r for r in compare if r['condition']!='S0_REFERENCE_REPLAY' and r['stage']=='from_zero'),key=lambda r:r['best_qualified_pic50'] or -np.inf,default={})
    target=state['original_steer']['best'];best=state['best']['program']['program_id']
    value={'target':target,'highest_new_from_zero_condition_including_replay':peak,'highest_exploration_condition':exploration,'active_program':best,'confirmed_best':state.get('confirmed_best'),
        'target_reached_in_from_zero':any(r['n_above_target']>0 for r in compare),'costs':costs}
    write_json(out/'conclusions.json',value)
    lines=['# 局部 motif auto-research v2：最终证据报告','',
        f"已完成 {len(state['rounds'])}/20 个有效研究条件；另有 {costs['duplicate_zero_generation_proposals']} 个相同函数提案复用已有结果，未重复生成。",
        f"原始 Steer 最优预测 pIC50 为 **{target:.6f}**。本轮新生成从头条件（含S0参考重放及冻结确认）中的最高基本合格分数为 **{peak.get('best_qualified_pic50',float('nan')):.6f}**（{peak.get('condition')}）；新探索程序最高为 **{exploration.get('best_qualified_pic50',float('nan')):.6f}**（{exploration.get('condition')}）。",
        f"活动保留程序为 **{best}**。冻结确认是否通过：**{'是' if state.get('confirmed_best') else '否/尚未通过'}**。所有亲和力均为同一模型的预测，不能表述为实验结合亲和力。",'',
        '## 1. 真实完成的条件','',
        '|有效轮|原始记录标识|阶段|单项改变|支持点数|平均 pIC50|最高合格|Uτ|合格/尝试|回滚|',
        '|---:|---|---|---|---:|---:|---:|---:|---|---|']
    for r in state['rounds']:
        mm=r.get('metrics',{});p=r.get('program',{});rb=r.get('rollback') or r.get('effective_rollback',False)
        lines.append(f"|{r.get('effective_iteration')}|{r['round']}|{r['stage']}|{r.get('factor')}|{len(p.get('atom_slots',[]))}|{mm.get('mean_all_pic50',float('nan')):.6f}|{mm.get('best_qualified_pic50',float('nan')) or float('nan'):.6f}|{mm.get('U_tau',float('nan')):.6f}|{mm.get('n_qualified')}/{mm.get('n')}|{rb}|")
    lines+=['','前两轮从同一历史状态续生成，仅用于机制诊断；其绝对分数不能与从头生成的效应比较。原始 R06、R08–R14 的重复提案及提交均保留，重编号映射见 `novelty_budget_revision.json`。',
        '', '## 2. 对照、统计和质量','',f'冻结阈值 τ={tau:.9f}，来自原始基本合格 Steer 分数的90%分位。Uτ=每次尝试中基本合格分子的 max(pIC50−τ,0) 的平均值，失败贡献0。晋升同时要求U增加超过数值容差且均值损失不超过配置容忍。最高分不单独决定晋升。',
        '', '|条件|n|均值|最高合格|Uτ|基本合格|MMFF重原子松弛/原子中位数|p90|', '|---|---:|---:|---:|---:|---|---:|---:|']
    for r in metric_rows:
        lines.append(f"|{r['condition']}|{r['n']}|{r['mean_all_pic50']:.6f}|{r['best_qualified_pic50'] or float('nan'):.6f}|{r['U_tau']:.6f}|{r['n_qualified']}/{r['n']}|{r.get('median_heavy_relaxation_per_atom') or float('nan'):.4f}|{r.get('p90_heavy_relaxation_per_atom') or float('nan'):.4f}|")
    lines+=['','MMFF值采用固定重原子先优化H，再释放全分子的两阶段差值；是局部松弛代理，不是结合能。基本合格仅指可解码、连接、最小蛋白距离≥0.8Å。PoseBusters若另有结果单独保存，不悄悄加入晋升gate。',
        '', '配对区间为固定初态池内bootstrap，未校正自适应搜索。调参池和新引导终态留出池分开；历史Steer教师先验并未对留出池屏蔽。旧对照若缺原生参数元数据，明确标记为有限元数据比较。原始Steer有1000次终态尝试及克隆依赖，本轮小样本最大值不构成等预算优越性检验。',
        '', '## 3. Analyst与Designer实际如何工作','',
        'Analyst通过最多3轮、每轮最多2个确定性工具查询读取合法N/C条件交换、角色/口袋场、完整选择窗口轨迹、实测控制响应和完整历史。新增空间区域使用同图或同根对照的、扣除整体位移的描述性局部场差异；不能当成显著性富集结果。全窗口拟合给出线性/二次系数、变化率和残差，不再拆小时间箱。',
        '', 'Designer读取同一不可变证据包、Analyst区域与可执行observable、父程序、确切数值选项、实际算子源码及父响应，只能编译Analyst绑定的支持与单项变化。确定性校验约束证据ID、区域、时间、有效参数路径、数值去重。两角色使用真实 deepseek-flash API、温度0，调用与审阅副本都有请求哈希和响应receipt。',
        '', '接入正确不等于科学推断正确：自由文本仍可能混淆历史Steer、旧guidance实验与当前父程序，或把匹配变化归因过强。报告以实际程序、数据及执行记录为准；保留语义审阅和逐轮解释修正，不能声称已证明LLM优于确定性搜索。',
        '', '## 4. 奖励与真实坐标梯度','',
        '固定桥接：z_h=((1−h)/(1−t))x_t+((h−t)/(1−t))yθ(x_t,t,p)。点式奖励为局部Hungarian匹配后的负pseudo-Huber距离均值。其它注册算子包含坐标目标带、口袋距离带、邻接方向、局部成对距离、质心与协方差匹配；实际执行的算子以逐轮程序为准。原子标签可用于脱离梯度的匹配代价，不直接求类别梯度。',
        '', '历史路径策略：q̂(u)=q_h+b₁(u−h)+b₂(u−h)²，使用整个学习窗口真实祖先坐标拟合且锚定q_h；h(t)=min(h,t+η×学习窗口宽度)，桥接状态与目标同时使用h(t)。这是一个新的时变目标策略，不是精确后续随机路径导数或成功概率模型。',
        '', '每步只使用当前FLOWR前向的真实坐标VJP；不对affinity head求导，不额外调用head，不做SMC。注入强度相对于原生预测步RMS，并记录逐原子/累计裁剪、匹配切换及实际位移。引导止于配置窗口，剩余推理原生完成到t=1，化学图自由变化。',
        '', '所有逐轮可复现选择依据、启用参数和否证条件见 `scientific_decisions.json`，程序见 `executable_programs.json`；这是可审核决策记录，不是内部思维链转录。',
        '', '## 5. 形成、保持与亲和力的区别','',
        '额外诊断以纯几何Hungarian评估每种元素的空间实现率，同时跟踪边界实际被控制的生成槽位。终态重新匹配和同槽位保持分开，避免把换了原子对应的高匹配率解释为保留。元素准确不等于芳香/受体角色已形成。逐样本精简表见 `elemental_diagnostics.parquet`。',
        '', '本轮代码和结果提交可逐轮回溯；新结构下载后逐文件校验，再删除远端派生结构。原始Steer与模型checkpoint保留。新增API功能验证、相同程序提案以及修复接口的失败调用均单列，不伪装成新生成收益。']
    for d in decisions:
        lines+=['',f"## 轮次 {d['effective_iteration']}（{d['source_round']}）的实际选择",'',
            f"父程序：`{d['parent']}`；因素：`{d['factor']}`；区域：`{d['region']}`。",'',
            'Analyst假设（保留原文，属于待验证解释）：','',d['hypothesis'] or '未提出可执行假设。','',
            'Designer映射：','',d['designer_rationale'] or '未执行。','',
            f"实际变化：`{d['parameter_changes']}`。结果与配对区间见本轮results.json；回滚：{d['rollback']}。"]
    (study/'interpretation.zh-CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return value


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',default='.');p.add_argument('--config',required=True);a=p.parse_args()
    value=build(a.repo,a.config);print({k:v for k,v in value.items() if k!='costs'})


if __name__=='__main__':main()
