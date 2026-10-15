"""Report actual programs and paired generation outcomes, independently of prose."""
from pathlib import Path
import gzip
import hashlib
import json
import numpy as np
import pandas as pd
from evomolsteer.io import read_json,write_json,write_table,digest
from evomolsteer.partial_search.report import api_audit


def report(repo,master):
    repo=Path(repo);study=repo/master['campaign_study'];state=read_json(study/'state.json');out=study/'summary'
    rows=[];deliveries=[];best_sample=None;attempts=0
    for entry in state['rounds']:
        label=entry['round'];folder=repo/entry.get('study',master['campaign_study']+'/'+label)
        cfg_path=repo/master['config_dir']/(label+'.json')
        if cfg_path.exists():folder=repo/read_json(cfg_path)['study']
        checks={}
        for role in ('Analyst','Designer'):
            skill=folder/(role+'.SKILL.md')
            if skill.exists():checks[role]=api_audit(folder,role,hashlib.sha256(skill.read_text(encoding='utf-8').encode()).hexdigest())
        selected=read_json(folder/'selection.json')
        designer=read_json(folder/'Designer.json')
        item={'round':label,'api':checks,'declared_patch':designer['patch_id'],'selected_patch':selected['patch_id'],
              'expected_response':designer['expected_execution_change'],'status':entry['status']}
        if entry.get('metrics'):
            p=entry['program'];data=repo/entry['result_folder'];a=read_json(data/'guidance_audit.json')
            active=[r for r in a['steps'] if r['active']];m=entry['metrics'];records=read_json(data/'records.json')
            candidate=max((r for r in records if r['qualified_structure']),key=lambda r:r['pic50'],default=None)
            if candidate and (best_sample is None or candidate['pic50']>best_sample['pic50']):
                best_sample={'round':label,'source_folder':entry['result_folder'],**candidate}
            attempts+=entry.get('additional_attempts',0)
            expected=sorted(k for k,w in p['weights'].items() if w>0)
            masks=read_json(data/'geometry.json')['teacher_clock']['mask']
            item.update(executed_terms=expected,positive_term_binding=designer.get('enabled_terms',expected)==expected,
                        proper_subset=len(p['atom_slots'])<min(sum(m) for m in masks),actual_steps=a['actual_steps'],actual_active_steps=a['actual_active_steps'],
                        real_generator_vjp=a['real_generator_vjp'],no_smc=a['no_smc'],no_affinity_gradient=a['no_affinity_head_gradient'])
            rows.append({'round':label,'factor':entry['factor'],'status':entry['status'],'parent':read_json(folder/'proposal.json')['parent']['program_id'],
                'patch':p['patch_id'],'n':m['n'],'qualified':m['n_qualified'],'mean':m['mean_all_pic50'],'max_qualified':m['best_qualified_pic50'],
                'dose':p['dose']['native_rms_ratio'],'type_share':p['weights']['type'],'cat_noise':p['native_options']['cat_noise_level'],
                'final_corr_pred':p['native_options']['final_corr_pred'],'geometry':p['geometry_mode'],'points':len(p['atom_slots']),
                'mean_vs_unguided':entry['paired']['unguided']['mean_pic50_difference'],
                'mean_vs_R13_same_protocol':entry['paired']['R13']['mean_pic50_difference'],
                'unguided_mean':entry['controls']['unguided']['mean_all_pic50'],
                'unguided_max':entry['controls']['unguided']['best_qualified_pic50'],
                'strain_median':m['median_mmff_relief_per_heavy'],'formation':entry['realization']['joint_formation'],
                'retention':entry['realization']['retention_after_formation'],'control_RMS_path_A':m['cumulative_control_RMS_A'],
                'mean_type_half_cosine':float(np.mean([r['gradient_cosine_type_half'] for r in active if 'gradient_cosine_type_half' in r])) if any('gradient_cosine_type_half' in r for r in active) else None,
                'mean_atom_clip_fraction':float(np.mean([r.get('atom_clip_fraction',0) for r in active])),
                'rollback':entry['rollback'],'inference_commit':entry['inference_commit']})
        deliveries.append(item)
    table=write_table(out/'detailed_rounds.csv',rows);write_json(out/'delivery_checks.json',deliveries)
    actual=list((repo/master['generated_root']).glob('*/*/records.json'))
    write_json(out/'result.json',{'completed_decisions':len(state['rounds']),'best_activity_round':state['best_round'],
        'best_tail_round':state.get('tail_round'),'best_sample':best_sample,'generated_condition_attempts':sum(len(read_json(p)) for p in actual),
        'decision_generation_attempts':attempts,'target_pic50':master['target_pic50'],
        'target_reached':bool(best_sample and best_sample['pic50']>=master['target_pic50']),
        'limitations':['Model-predicted affinity, not experimental binding','Training selected over many operators and seeds reused within pool',
            'Basic structure qualification differs from full PoseBusters','Native-protocol changes require their own unguided controls',
            'Teacher prior uses original Steer data; held-out initial-pool confirmation is not target-level generalization']})
    if best_sample:
        source=repo/best_sample['source_folder']/f"{best_sample['slot']:03d}.sdf.gz"
        if source.exists():
            target=repo/master['generated_root']/'best_candidate.sdf'
            with gzip.open(source,'rb') as f:target.write_bytes(f.read())
            write_json(out/'best_candidate_source.json',{'source':str(source.relative_to(repo)),'sha256':digest(source),
                'export':str(target.relative_to(repo)),'record':best_sample})
    header='|轮次|单项探索|父程序|均值|最高合格|合格/尝试|相对同协议无引导均值|形成率|回滚|\n|---|---|---|---:|---:|---|---:|---:|---|\n'
    body=''
    for r in rows:
        body+=f"|{r['round']}|{r['factor']}|{r['parent']}|{r['mean']:.6f}|{r['max_qualified']:.6f}|{r['qualified']}/{r['n']}|{r['mean_vs_unguided']:+.6f}|{r['formation']:.3f}|{r['rollback']}|\n"
    text='**20轮局部Gradient Guidance顺序优化结果**\n\n'+f"已记录{len(state['rounds'])}个研究决定，{len(rows)}个生成/复用结果。活动最佳为{state['best_round']}，高亲和力尾部候选为{state.get('tail_round')}。\n\n"+header+body
    text+='\n最高值与均值分别报告，类型形成/保持不是亲和力目标的替代。R19/R20为冻结程序扩大样本和新初态池确认，不能与24个训练输入的极值直接比较。每个原生噪声/终态输出设置使用同协议无引导及R13对照；原始Steer1000仅作历史参照。\n'
    text+='\n早期R01/R03/R04/R06自由解释存在教师分数、其他控制器或参数语义混淆，原回答保留。后续输入增加精确父程序、参数语义与结构化参数前后值；执行数据和本报告由确定性程序提取。接口修复和未生成的早期暂缓保存在原目录，不计为额外优化轮。\n'
    (out/'final_report.zh-CN.md').write_text(text,encoding='utf-8')
    return read_json(out/'result.json')
