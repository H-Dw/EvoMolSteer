"""Small reproducible reports; literal role effects and molecular quality separate."""
import argparse
from pathlib import Path
import pandas as pd
from evomolsteer.io import read_json,write_json,digest
from evomolsteer.generation.affinity_reporting import paired_head,batch_inference,arm_summary
from assemble_skill_ablation import assemble
from evomolsteer.continuous.skill_ablation import effective_signature


def report(root,study):
    root=Path(root).resolve();study=Path(study).resolve()
    literal=assemble(root,study,complete=True)
    if not literal['all_executable_equal_incumbent']:
        raise ValueError('New designs require separate generation and confirmation; cannot reuse incumbent quality')
    experiment=study.parent;fresh=experiment/'generation/round_01/local'
    table=pd.read_csv(fresh/'candidate_metrics.csv')
    old=root/'docs/experiments/ck2_affinity_geometry30_20261007'
    for n in range(27,31):
        old_program=read_json(root/f'configs/experiments/ck2_affinity_geometry30_v1/backtrack_round{n:02d}.json')
        if effective_signature(old_program)[0]!=literal['incumbent_signature']:
            raise ValueError('Prior validation reward differs from current literal reward')
    prior=pd.concat([pd.read_csv(old/f'round_{n:02d}/local/candidate_metrics.csv') for n in range(27,31)],ignore_index=True)
    def quality(frame):
        pairs=paired_head(frame,frame);contrasts=pairs.groupby('batch').delta.mean()
        if pairs.duplicated(['batch','seed','slot']).any():raise ValueError('Repeated molecular pairs')
        return {'gradient':arm_summary(frame[frame.arm.eq('gradient')]),
            'native':arm_summary(frame[frame.arm.eq('unguided')]),'batch_contrast':batch_inference(contrasts),
            'positive_pair_fraction':float(pairs.delta.gt(0).mean()),'mean_absolute_pair_delta':float(pairs.delta.abs().mean())}
    cohorts={'fresh100':quality(table),'retained_frozen400':quality(prior),
        'combined500':quality(pd.concat([table,prior],ignore_index=True))}
    execution=read_json(fresh/'execution_report.json');coordinate=read_json(fresh/'coordinate_audit.json')
    window=read_json(fresh/'window/report.json')
    if execution['outside_window_injection'] or not execution['no_particle_resampling'] or not window['zero_equivalence_passed']:
        raise ValueError('Execution control failed')
    if not coordinate['coordinate_preflight']['passed']:raise ValueError('True-model preflight failed')
    rows=literal['responses'];pd.DataFrame([{k:v for k,v in r.items() if k!='updates'}|{'updates':str(r['updates'])} for r in rows]).to_csv(study/'condition_results.csv',index=False)
    modules=read_json(study/'treatment_modules.json')
    ablations=[]
    labels={'joint_geometry':'联合几何与相关特征建议','continuous_dynamics':'额外的连续动态/拟合建议',
        'localization':'额外的区域定位建议','lineage_credit':'额外的谱系/后代赋分建议',
        'coherent_modes':'额外的多峰完整构象建议','regional_emphasis':'探索区域加权的建议',
        'history_matching':'探索父子变化匹配的建议','dose_calibration':'额外的剂量校准建议'}
    for role,names in modules.items():
        for name in names:
            selected=[r for r in rows if r['variant']=='without_'+role.lower()+'_'+name]
            ablations.append({'role':role,'module':name,'description_zh':labels[name],'literal_design_changed':any(not r['execution_equivalent_to_incumbent'] for r in selected),
                'response_count':len(selected),'quality_effect':'Exactly equal executable reward in this fixed task; no incremental molecular-quality effect measurable',
                'decision':'Remove from active guidance; archive as experimental treatment',
                'generalization':'Does not prove the advice is useless for new targets or different evidence.'})
    write_json(study/'module_results.json',ablations)
    before=sum(p.stat().st_size for p in (experiment/'legacy_skills').glob('*.md'))
    after=sum(p.stat().st_size for p in (root/'skills').rglob('*.md'))
    value={'literal_conditions':len({r['variant'] for r in rows}),'designer_responses':len(rows),
        'analyst_responses':len(list(study.glob('agents/*/*/Analyst.response.json'))),
        'unique_executable_rewards':len(literal['unique_executable_signatures']),
        'all_execution_equivalent_to_incumbent':True,'cohorts':cohorts,
        'skill_bytes':{'historical':before,'current':after,'reduction_fraction':1-after/before},
        'execution_controls':{'zero_native_equivalent':window['zero_equivalence_passed'],
            'no_particle_resampling':execution['no_particle_resampling'],'outside_window_injection':execution['outside_window_injection'],
            'true_model_preflight':coordinate['coordinate_preflight']['passed']},
        'provenance':{'fresh_retention_sha256':digest(fresh/'retention.json'),
            'fresh_outcome_sha256':digest(experiment/'generation/round_01.outcome.json'),
            'evidence_sha256':digest(study/'evidence.json'),'incumbent_signature':literal['incumbent_signature']},
        'limitations':['Fixed evidence, one molecular target and subagent backend; two repeats for factorial role conditions, one for module removals.',
            'Designs were limited to the shared registered formulas and scalar updates; this is not an ablation of unrestricted novel reward coding.',
            'Identical executable designs share inference and are not separate molecular replicates.',
            '400 prior frozen pairs are reused explicitly; only 100 molecular pairs are freshly generated.',
            'Instruction changes can affect explanations without changing rewards; no universally negative module was demonstrated.',
            'Predicted pIC50 is not measured affinity; MMFF relaxation is an isolated-ligand strain proxy, not binding free energy.']}
    forward=experiment/'deployment_agents/forward_audit.json'
    if forward.exists():
        value['final_core_forward_test']=read_json(forward)
        if not value['final_core_forward_test']['same_executable_reward_as_validated_incumbent']:
            raise ValueError('Final deployed role response differs from tested reward')
    write_json(study/'quality_report.json',value)
    lines=['# Skills消融与精简结果','',
        f"完成{value['literal_conditions']}种条件、{value['analyst_responses']}次Analyst和{value['designer_responses']}次Designer独立上下文调用。所有可执行设计均与冻结奖励完全相同。",
        '','对照包括旧/精简Analyst与旧/精简Designer的2×2交叉（每格2次）、完整建议版，以及8项建议的逐项移除。Designer建议消融复用同一份Analyst结果。共享证据仅含发现集；Agent未接触留出结果。',
        '','编译器允许保留、修改或暂缓，不规定赢家、剂量或公式，也不修复Agent选择。程序等价依据执行字段、参考库和运行约束，不依据措辞相似或文件名。',
        '','|角色|移除建议|奖励是否改变|处理|','|---|---|---|---|']
    for r in ablations:lines.append(f"|{r['role']}|{r['description_zh']}（{r['module']}）|否|从活动Skills移除，保留实验档案|")
    lines+=['','这些建议在当前证据上没有带来可测的额外亲和力或应变收益；没有证明它们在其他任务上普遍无用，也没有观察到可归因于某一建议的负效应。解释仍能识别变化方向反转、拟合保真度失败和谱系支持不足，基础科学核查得以保留。',
        '',f"活动Skills总文本从{before:,}字节降到{after:,}字节，减少{1-after/before:.1%}。分子名称、固定区域、历史轮次参数和环境要求已移除。保留数据来源、表示/时钟、统计单位、关联与因果区分、证据绑定和可微路径核查。",'',
        '|数据|引导均值|无引导均值|配对Δ及批次95%区间|MMFF中位数 引导/无引导|MMFF p90 引导/无引导|有效率 引导/无引导|',
        '|---|---:|---:|---|---|---|---|']
    for label,result in cohorts.items():
        g,n,c=result['gradient'],result['native'],result['batch_contrast'];lo,hi=c['batch_bootstrap_CI95']
        lines.append(f"|{label}|{g['all_head_mean']:.6f}|{n['all_head_mean']:.6f}|{c['mean_head_delta']:+.6f} [{lo:.6f}, {hi:.6f}]|{g['MMFF_relief_per_heavy_median']:.6f}/{n['MMFF_relief_per_heavy_median']:.6f}|{g['MMFF_relief_per_heavy_p90']:.6f}/{n['MMFF_relief_per_heavy_p90']:.6f}|{g['valid_rate']:.1%}/{n['valid_rate']:.1%}|")
    lines+=['','400个冻结验证样本来自既有报告，本轮新增100个配对样本与100个零引导控制样本。完全相同的奖励没有被重复生成并宣称为独立消融样本；相同程序支持当前性能不受Skills精简影响，未声称性能进一步提升。新100样本只有2个独立生成批次，其区间不能作为充分的独立泛化证明。',
        '','奖励继续使用原FLOWR endpoint预测的真实坐标VJP和分数条件点云混合，不对affinity head求导。学习窗口来自参考数据；窗口之后原生推理至终点。零引导与原生一致、没有重采样、没有窗口外注入均已通过运行检查。',
        '','标准入口为 `scripts/role_agent.py`：导出Analyst请求→导入并验证实际响应→导出绑定该响应的Designer请求→验证/编译原样奖励；可使用subagent或同一API接口。原历史正确答案型检查只用于冻结档案重放，未用于本轮消融。',
        '','本研究固定公式注册表和可修改标量，未测试任意新奖励代码的原创探索；精简结论适用于当前工作流，不能据此否定区域/历史等算法研究方向。零收益结论针对额外指引的当前增量作用，基础富集、差异、趋势和谱系统计脚本均保留。',
        '','本轮实验请求、字面响应、hash、程序等价证书、模块移除结果和精简质量报告位于本目录。旧原始Steer与checkpoint保留；本轮生成轨迹经报告核验后清理。']
    if forward.exists():lines+=['','删除建议后的最终部署文本另进行了1次Analyst与1次Designer独立调用。实际注入文本与当前Skills逐字一致，Designer绑定实际Analyst输出，最终编译奖励仍与已验证奖励相同。详见`deployment_agents/forward_audit.json`；这项接口测试不增加分子样本量。']
    (experiment/'report.zh-CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    return value


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',default=str(Path(__file__).resolve().parents[1]));p.add_argument('--study',required=True)
    a=p.parse_args();r=report(a.repo,a.study);print({'conditions':r['literal_conditions'],'fresh_delta':r['cohorts']['fresh100']['batch_contrast']['mean_head_delta']})
