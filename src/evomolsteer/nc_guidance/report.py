"""Small, deterministic final-layout and delivery audits; no new model fitting."""
import argparse
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from evomolsteer.io import read_json,write_json,write_table,digest
from evomolsteer.partial_search.report import api_audit,paired_comparison
from .workflow import numerical_signature
from .search import atomic_indices


def local_observations(folder,program,teacher):
    geometry=read_json(folder/'geometry.json');x=np.asarray(geometry['terminal']['coords_A'])
    labels=atomic_indices(folder,geometry,teacher['atom_vocabulary']);selected=program['atom_slots']
    target=np.asarray(teacher['terminal_coords_A'])[selected];position={s:i for i,s in enumerate(selected)}
    rows=[];N=teacher['atom_vocabulary']['N'];C=teacher['atom_vocabulary']['C']
    for sample,(xyz,a) in enumerate(zip(x,labels)):
        cost=np.square(xyz[:,None]-target[None]).sum(-1);rr,cc=linear_sum_assignment(cost);match=rr[np.argsort(cc)]
        for slot,element in zip(program['typed_slots'],program['typed_elements']):
            point=np.asarray(teacher['terminal_coords_A'][slot]);distance=np.linalg.norm(xyz-point,axis=-1)
            weight=np.where(distance<2.5,np.exp(-.5*(distance/.8)**2),0.)
            massN=float(weight[a==N].sum());massC=float(weight[a==C].sum());den=massN+massC
            actual=int(a[match[position[slot]]]);valid=actual>=0
            rows.append({'sample':sample,'teacher_slot':slot,'desired_element':element,
                'matched_correct':bool(actual==teacher['atom_vocabulary'][element]) if valid else None,
                'matched_distance_A':float(distance[match[position[slot]]]),
                'local_N_fraction':massN/den if den>1e-12 else None,
                'nearest_N_A':float(distance[a==N].min()) if (a==N).any() else None,
                'local_density_definition':'Gaussian sigma .8 A, truncated at 2.5 A; descriptive, not reward'})
    return rows


def report(repo,cfg):
    repo=Path(repo);study=repo/cfg['campaign_study'];state=read_json(study/'state.json');out=study/'summary'
    rows=[];delivery=[];observations=[];seen={};steps=[];deferred=[];deferred_delivery=[];paired=[];operators=set()
    for entry in state['rounds']:
        if entry.get('status')=='designer_deferred':
            label=entry['round'];deferred.append(label);folder=study/label
            config=read_json(repo/cfg['config_dir']/(label+'.json'))
            info={'instruction_sha256':{role:hashlib.sha256((repo/config[role.lower()+'_skill']).read_text(encoding='utf-8').encode()).hexdigest() for role in ('Analyst','Designer')}}
            deferred_delivery.append({'round':label,'additional_generation_attempts':0,
                'api':{role:api_audit(folder,role,info['instruction_sha256'][role]) for role in ('Analyst','Designer')}})
            continue
        label=entry['round'];folder=study/label;config=read_json(repo/cfg['config_dir']/(label+'.json'))
        bp=next((folder/'candidates').glob('*/bindings.json'));bindings=read_json(bp)
        p=read_json(repo/bindings['programs'][0]['program']);teacher=read_json(repo/bindings['teacher'])
        operators.add(numerical_signature(p,teacher))
        key=f"{config['pilot_initial_batch']}:{config['pilot_samples']}:"+numerical_signature(p,teacher)
        duplicate=seen.get(key);seen.setdefault(key,label)
        generated=repo/cfg['generated_root']/label/label
        if not generated.exists():
            paths=[repo/path for path in state['cache'].values() if (repo/path/'guidance_audit.json').exists()]
            generated=next(path for path in paths if numerical_signature(read_json(path/'guidance_audit.json')['program'],teacher)==numerical_signature(p,teacher))
        audit=read_json(generated/'guidance_audit.json');metrics=entry['metrics'];controls=entry['controls']
        if config['pilot_initial_batch']==0:
            control_paths={'R13_reference':repo/'data/generated/partial_region_search_20261014'/('R13/R13_0.1' if metrics['n']==24 else 'R19/R19_0.1'),
                'unguided':repo/'data/generated/partial_region_search_20261014'/('controls_screen/unguided' if metrics['n']==24 else 'controls_batch0_n50/unguided')}
        else:
            base=repo/cfg['generated_root']/f"controls_b{config['pilot_initial_batch']}_n{metrics['n']}"
            control_paths={'R13_reference':base/'R13_0.1','unguided':base/'unguided'}
        for name,path in control_paths.items():paired.append({'round':label,**paired_comparison(generated,path,label+' vs '+name)})
        local=local_observations(generated,p,teacher)
        for row in local:observations.append({'round':label,**row})
        typed=pd.DataFrame(local)
        def accuracy(element):
            v=typed.loc[typed.desired_element.eq(element),'matched_correct'].dropna()
            return float(v.astype(float).mean()) if len(v) else None
        rows.append({'round':label,'strategy':entry['strategy'],'patch':p['patch_id'],'template':p['template_id'],
            'geometry_points':len(p['atom_slots']),'typed_points':len(p['typed_slots']),'target_N':','.join(str(s) for s,e in zip(p['typed_slots'],p['typed_elements']) if e=='N'),
            'target_C':','.join(str(s) for s,e in zip(p['typed_slots'],p['typed_elements']) if e=='C'),
            'start':p['window'][0],'end':p['window'][1],'n':metrics['n'],'n_qualified':metrics['n_qualified'],
            'atomic_logit_strength':p.get('atomic_logit_strength',0.),
            'type_weight':p['weights']['type'],
            'active_type_points':len(p['typed_slots']) if p['weights']['type']>0 or p.get('atomic_logit_strength',0)>0 else 0,
            'mean_pic50':metrics['mean_all_pic50'],'max_qualified_pic50':metrics['best_qualified_pic50'],
            'delta_mean_R13':metrics['mean_all_pic50']-controls['R13_reference']['mean_all_pic50'],
            'delta_mean_unguided':metrics['mean_all_pic50']-controls['unguided']['mean_all_pic50'],
            'delta_max_R13':metrics['best_qualified_pic50']-controls['R13_reference']['best_qualified_pic50'] if metrics['best_qualified_pic50'] is not None else None,
            'N_accuracy':accuracy('N'),'C_accuracy':accuracy('C'),'aromatic_N_role':metrics.get('aromatic_N_role_fraction'),
            'median_MMFF_relief_per_heavy':metrics['median_mmff_relief_per_heavy'],'above_8p704806':metrics['n_above_reference'],
            'rollback':entry['rollback'],'duplicate_of':duplicate,'result_reused':entry['reused_numeric_program'],
            'confirmation':entry['confirmation'],'inference_commit':entry['inference_commit']})
        apis={role:api_audit(folder,role,bindings['instruction_sha256'][role]) for role in ('Analyst','Designer')}
        positive={k:v for k,v in p['weights'].items() if v>0}
        delivery.append({'round':label,'api':apis,'program_hash_matches':digest(repo/bindings['programs'][0]['program'])==bindings['programs'][0]['sha256'],
            'actual_window_matches':audit['actual_active_steps']==p['expected_active_steps'],
            'actual_steps':audit['actual_steps'],'all_cloud_excluded':len(p['atom_slots'])<len(teacher['active_slots']),
            'direct_background_max':max(s.get('direct_unmatched_gradient_max',0.) for s in audit['steps']),
            'model_mediated_background_max':max(s.get('current_unmatched_gradient_max',0.) for s in audit['steps']),
            'type_term_enabled':p['weights']['type']>0,'active_terms':positive,
            'observed_type_gradient_max':max(s.get('type_gradient_rms',0.) for s in audit['steps']),
            'observed_type_mixture_input_max':max(s.get('type_mixture_input_rms',0.) for s in audit['steps']),
            'local_logit_guidance_enabled':p.get('atomic_logit_strength',0.)>0,
            'observed_logit_shift_max':max(s.get('logit_shift_max',0.) for s in audit['steps']),
            'unmatched_logit_change_max':max(s.get('unmatched_logit_change_max',0.) for s in audit['steps']),
            'duplicate_numeric_operator_in_same_pool':duplicate,'no_smc':audit['no_smc'],
            'no_affinity_gradient':audit['no_affinity_head_gradient'],'original_teacher_parent_copy_verified':teacher['parent_copy_verified'],
            'scientific_prose_status':'Hypotheses can still contain factual mistakes; operative fields and deterministic target maps are authoritative'})
        for s in audit['steps']:
            if s['active']:steps.append({'round':label,**s})
    table=write_table(out/'rounds.csv',rows);write_table(out/'local_site_observations.parquet',observations)
    write_table(out/'active_step_diagnostics.parquet',steps);write_json(out/'delivery_audit.json',delivery)
    write_json(out/'deferred_delivery_audit.json',deferred_delivery);write_json(out/'paired_comparisons.json',paired)
    result={'completed_rounds':len(state['rounds']),'generated_or_reused_rounds':len(rows),'deferred_rounds':deferred,'incumbent':state['best_round'],'best':state['best'],
        'best_local_round':state.get('best_nc_round'),'reference_pic50':8.704806,
        'target_reached':any(r['above_8p704806']>0 for r in rows),
        'numeric_duplicate_rounds':[r['round'] for r in rows if r['duplicate_of']],
        'distinct_numeric_operators':len(operators),'new_candidate_generation_attempts':sum(r['n'] for r in rows if not r['result_reused']),
        'all_direct_rewards_are_proper_subsets':all(a['all_cloud_excluded'] for a in delivery),
        'all_API_deliveries_verified':all(all(v.get(k,False) for k in ('receipt_and_file_hashes_match','input_file_hashes_match','skill_delivered_exactly')) for a in delivery+deferred_delivery for v in a['api'].values()),
        'confirmation_results':[r for r in rows if r['confirmation']],
        'limits':['Final pIC50 is a model prediction','Qualified means basic chemical validity, connectedness and absence of severe receptor overlap, not full PoseBusters',
            'Training maxima are selected over many trials','Counterfactual targets were scored on fixed coordinates, not generated or relaxed',
            'Local final typing uses untyped geometric correspondence; density and distance diagnostics complement this discrete assignment',
            'Only the declared teacher subset contributes to loss; genuine model Jacobians can propagate to other current coordinates',
            'No independent target generalization or experimental affinity validation is established']}
    write_json(out/'result.json',result)
    text='# 历史R13起点的局部N/C引导：逐轮结果\n\n'
    text+=f"已完成{len(state['rounds'])}轮决策，其中{len(rows)}轮生成或复用，暂缓{deferred}；活动最佳为{state['best_round']}。参考终态pIC50为8.704806；是否已有基本结构合格分子超过该值：{result['target_reached']}。所有评分为模型预测。\n\n"
    text+='|轮次|策略|几何点/类型点|最高合格pIC50|全部均值|均值差vsR13|N正确率|C正确率|合格/尝试|回滚|\n|---|---|---|---:|---:|---:|---:|---:|---|---|\n'
    def fmt(value,precision=6):return 'NA' if value is None else f'{value:.{precision}f}'
    for r in rows:
        text+=f"|{r['round']}|{r['strategy']}|{r['geometry_points']}/{r['typed_points']}|{fmt(r['max_qualified_pic50'])}|{fmt(r['mean_pic50'])}|{fmt(r['delta_mean_R13'])}|{fmt(r['N_accuracy'],3)}|{fmt(r['C_accuracy'],3)}|{r['n_qualified']}/{r['n']}|{r['rollback']}|\n"
    text+='\n先测试至多10点的稀疏区域，再前瞻加入保留R13原20/25点几何上下文、仅比较2个N/C位置的对照。所有分支均不使用全25点损失。后续推理完整完成至t=1，引导窗口由程序指定且不超过.5。\n\n'
    text+='Analyst读取有限合法布局、成对N/C补偿、化学角色、方向性几何和真实终态反馈，选择注册局部区域；Designer必须返回与该区域绑定的执行契约。程序产生N/C标签和空间分配，不依赖自然语言手工抄写原子列表。训练目标主指标为最高基本结构合格pIC50，均值、失败和应变分别报告。\n\n'
    text+='局部损失包括真实半程桥/端点几何的负pseudo-Huber误差、选定位点的N/C解码log概率，可选方向性供体代理及局部中性电荷/芳香邻接概率。各项通过真实模型对当前坐标的Jacobian回传，按RMS作混合与位移限幅。R18另行标记为混合控制：增加仅作用于两个选中位置的解码logit奖励梯度步，再由原生类别积分器采样；它与纯坐标控制分开评价。未对affinity head求导、没有SMC、没有手动改写离散元素或固定全图。局部直接损失不等于模型Jacobian对其它坐标的影响必为零。\n\n'
    text+='N/C正确率按终态受体坐标系的一对一几何匹配计算，并分开报告N与C；另存2.5Å局部核密度及最近N距离，避免只依赖一个离散匹配率。原始元素与期望元素已经显式分离；芳香N角色通过实际生成SDF检查，不把通用N概率当作已形成吡啶N或氢键。\n\n'
    duplicates=[r['round'] for r in rows if r['duplicate_of']]
    if duplicates:text+=f"数值等价的研究调用：{duplicates}。R07仅改变了点数上限元数据，实际与R04相同并曾重复生成；重复结果不能当作新增证据。签名已去除这类非计算字段。\n\n"
    text+='完整请求/响应、Skills快照、确定性区域和程序、源代码commit及失败记录均保留。delivery_audit.json核对实际API交付、程序哈希和真实控制日志；不把自由文字解释当作已证明的科学结论。R06在控制器暂停前已派发，恢复其实际已完成结果，没有重复推理，原执行commit保留。\n'
    text+='\nR19/R20确认的是本轮最强的局部程序R11：它复现历史R13几何，类型权重为0，并不是已经优化成功的新N/C奖励。每个程序的active_type_points和type_weight在rounds.csv单列，避免把仅保留作诊断的类型标签误当作施加了类型控制。\n\n'
    text+='配对确认结果（全部评分均值差；固定初态池的描述性区间）：\n\n'
    for c in paired:
        if c['round'] in ('R19','R20'):
            text+=f"- {c['comparison']}：{c['mean_pic50_difference']:+.6f}，95%区间[{c['paired95'][0]:+.6f}, {c['paired95'][1]:+.6f}]。\n"
    historical=out/'historical_steer_comparison.json'
    if historical.exists():
        h=read_json(historical)
        text+=f"\n历史单目标Steer：{h['n_attempts']}次终态尝试，基础合格{h['n_basic_qualified']}，全部均值{h['mean_all_pic50']:.6f}，最高基础合格{h['best_basic_qualified_pic50']:.6f}，同口径MMFF松弛/重原子中位数{h['median_MMFF_relief_per_heavy']:.6f}。这是1000个历史终态，与本轮单池50样本的预算和谱系依赖不同，不能据此做无条件方法优劣推断。\n"
    probe=study/'R18_identity_replay/identity_report.json'
    if probe.exists():
        d=read_json(probe);n=next(g for g in d['groups'] if g['desired_element']=='N');j=d['joint_layout']
        text+=f"\n固定R18诊断回放额外{d['extra_attempts']}次，数值程序未修改，终态坐标最大差{d['terminal_coordinate_max_abs_difference_A']:.1f}Å、pIC50最大差{d['pic50_max_abs_difference']:.1f}。半程实际目标N形成{n['correct_in_actual_boundary_state']}/{n['n']}，其中{n['boundary_correct_then_lost']}个在无引导后半程丢失；半程形成的完整成对布局{j['correct_at_boundary']}个，全部在后续丢失。该回放不计为第21轮优化，不计为新的独立seed或奖励性能增益。\n"
    text+='\n完整科学解释、剩余文字错误、回滚边界及可恢复入口见同级目录的../final_report.zh-CN.md。\n'
    (out/'report.zh-CN.md').write_text(text,encoding='utf-8');return table


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',default='.');p.add_argument('--config',required=True);a=p.parse_args()
    print(report(a.repo,read_json(a.config)).to_string(index=False))


if __name__=='__main__':main()
