"""Rebuild the scientific report from immutable programs and actual outcomes."""
import argparse
import copy
import gzip
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from evomolsteer.io import read_json,write_json,write_table,digest
from .evaluation import score_records,energy,paired
from .reward import MotifReward
from .semantic_contract import identity


def figures(rows,target,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    points=sorted([r for r in rows if r.get('effective_iteration') is not None and r['stage']!='suffix'],key=lambda r:r['effective_iteration'])
    x=[r['effective_iteration'] for r in points];fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    axes[0].plot(x,[r['mean_all_pic50'] for r in points],'o-',label='Mean, all scored attempts')
    axes[0].plot(x,[r['best_qualified_pic50'] for r in points],'s-',label='Best basic-qualified')
    axes[0].axhline(target,color='black',ls='--',label='Historical Steer best (n=1000)')
    axes[0].set(ylabel='Predicted pIC50',xlabel='Effective research iteration (0 = reference replay)')
    axes[0].legend(fontsize=8);axes[1].plot(x,[r['U_tau'] for r in points],'o-',color='tab:green')
    axes[1].set(ylabel='Tail utility per attempt',xlabel='Effective research iteration')
    for ax in axes:ax.grid(alpha=.2)
    fig.savefig(out/'search_metrics.png',dpi=180);fig.savefig(out/'search_metrics.svg');plt.close(fig)


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
    programs.append({'condition':'S0_REFERENCE_REPLAY','numerical_signature':identity(rp,templates[rp['template_id']]),'program':rp})
    counterparts=[('S0_REFERENCE_REPLAY',rp,templates[rp['template_id']],repo/reference['result_folder'])]
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
            if r['stage']=='from_zero':counterparts.append((label,p,t,repo/r['result_folder']))
        for j,trial in enumerate(r.get('trials',[])):
            add(label+'_stream'+str(j),repo/trial['result_folder'],p,t,'suffix',r.get('effective_iteration'))
    # Controls are fixed artifacts, with the same failure denominator as guided.
    for name in ['controls_b%d_n%d'%(m['validation_batch'],m['confirmation_samples'])]:
        folder=repo/m['generated_root']/name/'unguided'
        if (folder/'records.json').exists():add(name,folder,stage='control')
    oldug=repo/'data/generated/partial_region_search_20261014/controls_batch0_n50/unguided'
    if (oldug/'records.json').exists():add('REUSED_UNGUIDED_b0_n50',oldug,stage='control_legacy_metadata')
    write_table(out/'conditions.csv',metric_rows);write_table(out/'elemental_diagnostics.parquet',diagnostics)
    element_summary=pd.DataFrame(diagnostics).groupby(['condition','stage','source_element'],as_index=False)[
        ['geometry_matched_identity_fraction','boundary_controlled_slot_identity_fraction','point_rms_A']].mean()
    element_summary.to_csv(out/'elemental_summary.csv',index=False)
    figures(metric_rows,state['original_steer']['best'],out)
    quality=[]
    for path in sorted((out/'quality').glob('*.posebusters.json')):
        q=read_json(path);quality.append({'condition':path.name.split('.posebusters')[0],**{k:q[k] for k in ['n_attempts','n_evaluated','n_pass_all']}})
    write_table(out/'posebusters_conditions.csv',quality)
    write_json(out/'scientific_decisions.json',decisions);write_json(out/'agent_delivery.json',bindings);write_json(out/'executable_programs.json',programs)
    matching_effects=[]
    for label,p,t,folder in counterparts:
        if not p.get('assignment_element_cost'):continue
        q=copy.deepcopy(p);q['assignment_element_cost']=0.
        for other,op,ot,ofolder in counterparts:
            if op.get('assignment_element_cost') or identity(q,t)!=identity(op,ot):continue
            matching_effects.append({'condition':label,'zero_cost_counterpart':other,'cost':p['assignment_element_cost'],
                'n_support':len(p['atom_slots']),'paired':paired(folder,ofolder,tau),
                'limits':'Retrospective fixed-pool numerical counterpart; not independent validation or a search-adjusted causal test.'})
    write_json(out/'matching_cost_counterparts.json',matching_effects)
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
    if quality:
        lines+=['','## 附：PoseBusters dock诊断','', '|条件|总尝试|可评估|全部检查通过|','|---|---:|---:|---:|']
        lines+=[f"|{q['condition']}|{q['n_attempts']}|{q['n_evaluated']}|{q['n_pass_all']}|" for q in quality]
        lines+=['','以上采用同一蛋白、标准dock检查，无参考化学图限制。失败与不可解码均保留在总尝试分母中。']
    if matching_effects:
        lines+=['','## 附：匹配代价的数值对应对照','', '|条件|代价0对照|支持数|元素匹配代价|平均分差|尾部产出差|', '|---|---|---:|---:|---:|---:|']
        for c in matching_effects:
            lines.append(f"|{c['condition']}|{c['zero_cost_counterpart']}|{c['n_support']}|{c['cost']}|{c['paired']['mean_difference']:.6f}|{c['paired']['U_difference']:.6f}|")
        lines+=['','对照必须通过去除匹配代价后的数值签名、初态及原生协议检查；区间保存在matching_cost_counterparts.json。它们仍来自反复使用的调参池，不能作为独立证实。']
    lines+=['','![实际从头生成指标](summary/search_metrics.png)','', '图中的最大值来自不同的小样本池；原始Steer水平线是目标参考，不是等预算显著性比较。']
    (study/'interpretation.zh-CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if any(r['round']=='R15_v2' for r in state['rounds']):
        r15=next(r for r in state['rounds'] if r['round']=='R15_v2')
        note=['# 本轮关键发现与证据边界','',
            '本文件中的轮次均属于 local_motif20_v2_20261015，不能与历史R26、其它campaign的R11混用。',
            '', '1. **局部身份形成不等于亲和力受益。** R03的终态N几何匹配率95.8%，跟踪边界实际控制槽位的终态N率100%；但最高基本合格pIC50仅8.127753。R16_v2也形成了N，仍无高分尾部。不能把所有失败都解释为没有生成N。',
            '', '2. **更准确地贴合固定点云并不是当前首要缺口。** 参考S0边界点云RMS约0.039Å，R15_v2约0.129Å；后者终态预测亲和力反而更高。时变历史路径拟合与目标带各有真实数值作用，但没有提高本轮尾部。单条成功祖先路径不一定适合其它随机起点。',
            '', '3. **点云对应规则确实是一个可干预模块。** R15_v2只加入0.5的脱离梯度的元素匹配代价，几何支持、pseudo-Huber奖励、rho、窗口及原生协议保持不变。其训练池平均分比S0高0.014569、U高0.006450，但两项配对区间均跨0。相对同池无引导平均高0.274159，仍是反复使用过的调参池证据。',
            '', '4. **收益具有支持背景依赖。** 同一匹配代价用于3点N/C区域（R16）或4点O区域（R18）没有复制较宽20/25点支持的收益。它提示其它骨架/邻域共同参与适配；不能推出20点是最小充分区域，也不能推出元素代价对所有区域有效。当前只做了少数条件组合。',
            '', '5. **记录显示Skills输入有所响应，但尚未证明Agent具有独立性能贡献。** 加入可追溯的跨实验家族反证后，功能测试明确引用已有失败；下一次真实探索R15转向匹配规则并产生数值及生成差异。提示词、输入知识和选择同时改变，且没有等预算确定性选择器对照，不能将改善完全归因于LLM或某句Skills。',
            '', '6. **探索覆盖仍然有限。** 注册库有多个教师和六类算子，但实际从头探索主要围绕F_b011_s030；未执行的算子、其它起点/教师、bounded控制不能写成已经失败。Agent仍有对最高分教师的锚定。拟合残差、几何关联与原子角色都不足以替代真实终态干预。',
            '', '## 当前保留函数的数学含义','',
            '对局部教师点q及其观测元素e，π=argmin Σj[||z_h[πj]−qj||²−c log softmax(ℓθ)[πj,e_j]]。匹配在无梯度分支中计算，当前最佳c=0.5。之后只求R=−(1/K)Σj δ²(√(1+||z_h[πj]−qj||²/δ²)−1)的真实模型坐标VJP。',
            '', 'R15_v2的K=20，小于25个生成槽位；δ=1Å、rho=0.66、逐原子步上限0.15Å、累计RMS上限6Å，窗口[0.1,0.5]，100个积分步。π只改变被匹配的几何对应，不直接对logits或元素求导；化学图自由变化。后半程原生完成至1，最终分数另行统一评估。',
            '', '这些值是此次比较中冻结的研究配置，不是从源分子证明的通用常数。学习和引导范围来自配置；源码还存在注册算子的校准默认值，若迁移目标需重新登记、验证，不能把旧分子的原子槽位移到新目标。',
            '', f"最终活动程序：{best}；相对无引导的冻结确认通过：{bool(state.get('confirmed_best'))}。是否达到原始Steer最高预测分数：{value['target_reached_in_from_zero']}。具体确认均值、尾部、区间和结构质量以主报告及结果JSON为准。",
            '', '原始Steer最优候选的旧PoseBusters记录在internal_energy项失败；复用记录保留。旧H放置包含的应变不能直接与新MMFF两阶段值比较。已有同算法两阶段重原子松弛/原子代理：源F约5.8923，R15训练最高分样本约0.9566。亲和力仍是本研究的首要目标，该结构改善不能替代达到8.704806的要求。']
        (study/'scientific_readout.zh-CN.md').write_text('\n'.join(note)+'\n',encoding='utf-8')
    return value


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',default='.');p.add_argument('--config',required=True);a=p.parse_args()
    value=build(a.repo,a.config);print({k:v for k,v in value.items() if k!='costs'})


if __name__=='__main__':main()
