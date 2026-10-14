"""Summarize actually executed/reused partial programs without counting no-ops as gains."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from evomolsteer.io import read_json,write_json,write_table,digest


def api_audit(folder,role,expected_skill_sha):
    stem='Analyst.selection' if role=='Analyst' and (folder/'Analyst.selection.json').exists() else role
    receipt=read_json(folder/(stem+'.receipt.json'));path=folder/(stem+'.request.json');request=read_json(path)
    messages=request['body']['messages'];system_sha=next(m['content_sha256'] for m in messages if m['role']=='system')
    return {'requested_model':receipt['requested_model'],'returned_model':receipt['returned_model'],
        'backend':receipt['backend'],'temperature':request['body']['temperature'],
        'receipt_and_file_hashes_match':receipt['request_sha256']==digest(path)
            and receipt['response_sha256']==digest(folder/(stem+'.json'))
            and receipt['wire_request_sha256']==digest(folder/receipt['wire_request_path']),
        'input_file_hashes_match':all(digest(folder/m['content_relative_path'])==m['content_sha256'] for m in messages),
        'skill_delivered_exactly':system_sha==expected_skill_sha}


def paired_comparison(guided_folder,control_folder,label):
    def load(folder):return pd.DataFrame(read_json(folder/'records.json')).set_index('slot').sort_index()
    guided,control=load(guided_folder),load(control_folder)
    ga,ca=[read_json(f/'guidance_audit.json')['input_pairing'] for f in (guided_folder,control_folder)]
    pairing=ga['initial_sha256']==ca['initial_sha256'] and ga['initial_slots']==ca['initial_slots']
    if not pairing or not guided.index.equals(control.index):raise ValueError('Different random input slots')
    x=guided.pic50.to_numpy(float);y=control.pic50.to_numpy(float);finite=np.isfinite(x)&np.isfinite(y)
    diff=x[finite]-y[finite];rng=np.random.default_rng(42)
    means=diff[rng.integers(0,len(diff),(10000,len(diff)))].mean(1)
    return {'comparison':label,'n_paired':int(finite.sum()),'input_pairing_verified':pairing,
        'mean_pic50_difference':float(diff.mean()),'paired95':np.quantile(means,[.025,.975]).tolist(),
        'qualified_guided':int(guided.qualified_structure.sum()),'qualified_control':int(control.qualified_structure.sum()),
        'uncertainty_scope':'Descriptive fixed initial pool and fixed reward; not search-adjusted or target-generalization inference'}


def report(repo,cfg):
    repo=Path(repo);study=repo/cfg['campaign_study'];state=read_json(study/'state.json');rows=[]
    for entry in state['rounds']:
        metrics=entry['conditions'];best=max(metrics,key=lambda m:m['best_qualified_pic50'] if m['best_qualified_pic50'] is not None else -np.inf)
        selection=study/entry['round']/'candidates'/cfg['candidate_id']/'Analyst.json'
        analyst=read_json(selection);new=entry['new_numerical_programs']
        rows.append({'round':entry['round'],'strategy':entry['strategy'],'selected_count':len(analyst['atom_slots']),
            'region_ids':','.join(analyst['region_ids']),'endpoint_mix':best.get('endpoint_mix',0.),
            'best_start':best['start'],'best_qualified_pic50':best['best_qualified_pic50'],
            'top5_qualified_mean':best['top5_qualified_mean_pic50'],'mean_all_pic50':best['mean_all_pic50'],
            'n':best['n'],'qualified':best['n_qualified'],'above_reference':best['n_above_reference'],
            'new_programs':new,'real_execution_changed':bool(new),'rollback':entry.get('rollback_required',False),
            'confirmation':entry.get('confirmation',False),'inference_commit':entry['commit']})
    table=write_table(study/'summary/rounds.csv',rows)
    baseline=pd.DataFrame(state['baseline']);write_table(study/'summary/screen_controls.csv',baseline)
    for entry in state['rounds']:
        if entry.get('confirmation'):write_table(study/'summary'/(entry['round']+'_controls.csv'),entry['controls'])
    checks=[]
    for row in rows:
        folder=study/row['round']/'candidates'/cfg['candidate_id'];binding=read_json(folder/'bindings.json')
        a=read_json(folder/'Analyst.json');d=read_json(folder/'Designer.json')
        checks.append({'round':row['round'],'same_slot_set':a['atom_slots']==d['atom_slots'],
            'same_region_ids':a['region_ids']==d['region_ids'],'bound_program_hashes':all(digest(repo/p['program'])==p['sha256'] for p in binding['programs']),
            'same_point_weights':bool(np.allclose(np.array([w['relative_weight'] for w in a['point_importance']])/sum(w['relative_weight'] for w in a['point_importance']),d['atom_weights'],atol=1e-8,rtol=0)),
            'new_execution':row['real_execution_changed'],
            'api':{role:api_audit(folder,role,binding['instruction_sha256'][role]) for role in ('Analyst','Designer')},
            'scientific_rationale_is_not_proof':True})
    write_json(study/'summary/binding_audit.json',checks)
    comparisons=[];confirmations=[]
    for entry in state['rounds']:
        if not entry.get('confirmation'):continue
        label=entry['round'];gc=read_json(repo/f'configs/experiments/partial_region_search_20261014/{label}.json')
        key=f"batch{gc['pilot_initial_batch']}_n{gc['pilot_samples']}"
        control_folder=repo/state['controls'][key]['local']
        guided_folder=repo/'data/generated/partial_region_search_20261014'/label/entry['conditions'][0]['condition']
        for m in entry['conditions']+entry['controls']:
            confirmations.append({'round':label,'condition':m['condition'],'n':m['n'],
                'mean_all_pic50':m['mean_all_pic50'],'best_qualified_pic50':m['best_qualified_pic50'],
                'qualified':m['n_qualified'],'median_mmff_relief_per_heavy':m['median_mmff_relief_per_heavy']})
        for control in entry['controls']:
            comparisons.append(paired_comparison(guided_folder,control_folder/control['condition'],label+' vs '+control['condition']))
    write_table(study/'summary/confirmation_conditions.csv',confirmations)
    write_json(study/'summary/paired_comparisons.json',comparisons)
    best_train=table.loc[~table.confirmation].sort_values('best_qualified_pic50',ascending=False).iloc[0].to_dict()
    validation=table.loc[table['round'].eq('R20')].iloc[0].to_dict() if table['round'].eq('R20').any() else None
    attempt_files=sorted((repo/'data/generated/partial_region_search_20261014').glob('*/*/records.json'))
    write_json(study/'summary/result.json',{'rounds':len(rows),'training_absolute_best':best_train,
        'active_tolerance_based_checkpoint':state['best_round'],'active_checkpoint':state['best'],
        'independent_initial_batch_validation':validation,'reference_pic50':8.704806,
        'target_reached':any(r['above_reference']>0 for r in rows),
        'n_new_numerical_programs':int(table.new_programs.sum()),'n_no_execution_change_rounds':int((~table.real_execution_changed).sum()),
        'actual_generation_conditions_including_controls':len(attempt_files),
        'actual_generation_attempts_including_controls':sum(len(read_json(p)) for p in attempt_files),
        'limitations':['Training winners were selected over multiple region and clock proposals',
            'Validation initial batch native guidance outcomes were unseen; its historical Steer finals were in prior evidence',
            'Basic structure qualification is not full PoseBusters; MMFF is a strain proxy',
            'All affinities are model predictions, not experimental binding',
            'The performance objective was revised at R04 to follow the user high-affinity discovery goal']})
    header='|轮次|策略|点数|终态项系数|最佳起点|最高有效 pIC50|全部均值|合格/尝试|新函数数|回滚|\n|---|---|---:|---:|---:|---:|---:|---:|---:|---|\n'
    body=''.join(f"|{r['round']}|{r['strategy']}|{r['selected_count']}|{r['endpoint_mix']:g}|{r['best_start']:g}|{r['best_qualified_pic50']:.5f}|{r['mean_all_pic50']:.5f}|{r['qualified']}/{r['n']}|{r['new_programs']}|{'是' if r['rollback'] else '否'}|\n" for r in rows)
    text=f"# 部分点云区域搜索结果\n\n已完成 {len(rows)} 轮角色调用与测试/复用记录。原始教师最终预测 pIC50 为 8.704806。训练池绝对最高有效分数为 {best_train['best_qualified_pic50']:.6f}，来自 {best_train['round']}；按 .01 容差保持的活动 checkpoint 是 {state['best_round']}。相同数值函数复用结果，不计为新分子实验。\n\n"+header+body
    if validation:
        text+=f"\nR20 在另一初始批次的 {validation['n']} 个样本中，最高有效分数为 {validation['best_qualified_pic50']:.6f}，全部均值为 {validation['mean_all_pic50']:.6f}，合格 {validation['qualified']}/{validation['n']}，超过教师的合格分子 {validation['above_reference']} 个。\n"
    text+='\n冻结确认与配对对照（最高值仅统计基本结构合格结果，均值统计全部有评分的尝试）：\n\n|确认|条件|全部均值|最高合格pIC50|合格/尝试|MMFF松弛/重原子中位数|\n|---|---|---:|---:|---:|---:|\n'
    for m in confirmations:
        text+=f"|{m['round']}|{m['condition']}|{m['mean_all_pic50']:.6f}|{m['best_qualified_pic50']:.6f}|{m['qualified']}/{m['n']}|{m['median_mmff_relief_per_heavy']:.4f}|\n"
    text+='\n'
    for c in comparisons:
        text+=f"- {c['comparison']}：配对均值差 {c['mean_pic50_difference']:+.6f}，描述性95%区间 [{c['paired95'][0]:+.6f}, {c['paired95'][1]:+.6f}]。\n"
    text+='\n这些区间仅描述固定初态池与固定奖励，没有校正搜索选择。R19包含原24个调参初态，不能视作独立复制。R20未复制亲和力提升，不能把R13宣布为经过独立验证的新性能基线。详细解释和可恢复入口见同目录 interpretation.zh-CN.md。\n'
    text+='\n每轮的 API 原始回答、确定性点集扩展、真实教师、Designer、函数及哈希均保留在对应目录。R01-R03 旧前五均值判据及回滚保留，R04 起以最高有效分数为主；这一目标修订不能用来把早期失败改写成成功。所有候选都从原始随机输入开始，种子 42，无 SMC，结束引导后完整生成至终态。\n\n部分点云通过注册区域和精确槽位查表获得；两个区域只产生一个并集奖励。数值比较和点列表由程序生成，避免 LLM 算术、抄录及符号错误。Designer 不得更换区域、点、点权重、教师或时间窗。R12 的双时刻检验在相同点上同时使用真实半程和真实终态两种坐标视图，使用共享空间分配和固定 .5 系数；它不同于教师温度消融，保持每步剂量且没有额外模型前向或 affinity-head 求导。\n\n所有亲和力都是预测值。历史 Steer 的 1000 终态只有 23 个原始谱系根，区域关联不是独立因果证据。初态验证没有使用其本轮引导终态结果调参，但该批历史 Steer 结果属于先验。基本结构检查不等于完整 PoseBusters，应变为 MMFF 松弛代理。目标是否实现应根据最终分数和有效结构报告，不能用低半程匹配误差替代。\n'
    (study/'summary/report.zh-CN.md').write_text(text,encoding='utf-8');return table


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',default='.');p.add_argument('--config',required=True);a=p.parse_args()
    print(report(a.repo,read_json(a.config)).to_string(index=False))


if __name__=='__main__':main()
