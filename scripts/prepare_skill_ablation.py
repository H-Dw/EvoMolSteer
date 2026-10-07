"""Freeze the same discovery evidence for independent role-instruction trials."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from evomolsteer.io import read_json,write_json,digest
from evomolsteer.continuous.skill_ablation import variants,export_request
from evomolsteer.generation.formula_registry import formula_registry


def prepare(root,output):
    root=Path(root).resolve();out=Path(output)
    if out.exists():raise FileExistsError('Evidence preparation is immutable')
    out.mkdir(parents=True)
    original=root/'docs/experiments/ck2_affinity_geometry30_20261007'
    cfg=root/'configs/experiments/ck2_affinity_geometry30_v1'
    sources={};evidence=[]
    for label,path in [('endpoint_effects',original/'endpoint_mining/whole_window_effects.parquet'),
        ('regional_effects',original/'regional_mining_v1/whole_window_effects.parquet'),
        ('increment_effects',next((original/'kinematics_mining_v3').glob('whole_window*.parquet'))),
        ('survival_effects',original/'terminal_lineage_v4/whole_window_survival_effects.parquet')]:
        sources[path.relative_to(root).as_posix()]=digest(path)
        for i,row in enumerate(pd.read_parquet(path).to_dict('records')):
            evidence.append({'evidence_id':label+':'+str(i),'kind':label,'data':row})
    for label,path in [('regional_curves',original/'regional_mining_v1/region_prior.json'),
        ('curve_support',original/'endpoint_mining/sparse_coordinate_prior_v2.json'),
        ('lineage_support',original/'terminal_lineage_v4/manifest.json'),
        ('information_audit',original/'information_audit_v1/information_summary.json')]:
        d=read_json(path);sources[path.relative_to(root).as_posix()]=digest(path)
        if label=='regional_curves':d={k:d[k] for k in ('times','supported_node_functions','statistics_unit','limitations')}
        if label=='curve_support':d={k:d[k] for k in ('selected_count','selection_rule','curve_fidelity_rule','curve_fidelity_audit')}
        if label=='lineage_support':d={k:v for k,v in d.items() if k not in ('sources','source_code_sha256')}
        evidence.append({'evidence_id':label,'kind':label,'data':d})
    registry={};rng=np.random.default_rng(812);numbers=rng.permutation(np.arange(5,27));incumbent=None
    for i,n in enumerate(numbers):
        program=cfg/f'backtrack_round{n:02d}.json';p=read_json(program)
        if p.get('derivative_path')!='flowr_endpoint_vjp':continue
        token='program_'+str(i);ref=next(f for f in cfg.glob('*reference.json.gz') if digest(f)==p['reference_sha256'])
        registry[token]={'program_path':program.relative_to(root).as_posix(),'program_sha256':digest(program),
            'reference_path':ref.relative_to(root).as_posix(),'reference_sha256':digest(ref),
            'reward_view':p['reward_view'],'window':p['window'],
            'parameters':{k:p[k] for k in ('native_rms_ratio','teacher_neighbors','teacher_score_beta',
                'teacher_endpoint_temperature_A2','mixture_temperature','time_ramp_power') if k in p}}
        r=read_json(original/f'round_{n:02d}.outcome.json')
        evidence.append({'evidence_id':'quality:'+token,'kind':'discovery_quality','data':{
            'program_id':token,'mean_pic50_delta':r['all_head_change_vs_native'],'best_valid_pic50':r['best_valid_head'],
            'MMFF_relief_per_heavy_median':r['terminal']['all_mmff_relief_per_heavy_median'],
            'MMFF_relief_per_heavy_p90':r['MMFF_p90'],'valid_rate_change':r['valid_rate_change'],
            'window_shape_improvement_fraction':r['shape_improvement_fraction'],
            'scope':'Adaptive reused discovery batches; not independent validation'}})
        if int(n)==26:incumbent=token
    packet={'schema_version':'skill-ablation-evidence-1.0','incumbent_program_id':incumbent,
        'task':{'objective':'Maintain or improve final predicted target affinity; strain must not materially regress',
            'requested_action':'Analyze observed selection, then choose one evidence-supported registered reward for a paired generation comparison. Retain, modify or defer is allowed.',
            'quality_basis':'All attempts and failures; compare means and physical tails separately',
            'learning_window':registry[incumbent]['window'],'primary_representation':'predicted_endpoint_world_A',
            'independent_units':'Generation batches; parent-debiased local moments where specified'},
        'execution_contract':{'derivative_path':'flowr_endpoint_vjp','affinity_head_gradient':False,
            'additional_production_forward_calls_per_step':0,'no_SMC':True,'no_graph_veto':True,
            'time_support':'Exact learned nodes only, native continuation after the window',
            'parameter_updates':'Only the bounded scalar fields accepted by the shared compiler',
            'note':'The compiler checks validity but does not prescribe the winning program or repair Agent choices'},
        'program_registry':registry,'evidence':evidence,'source_hashes':sources}
    write_json(out/'evidence.json',packet)
    write_json(out/'formula_registry.json',formula_registry(root))
    design={'replicates_per_factorial_condition':2,'replicates_per_advice_removal':1,
        'screen_generation_batches':[28,29],'confirmation_batches':list(range(30,38)),
        'master_seed':42,'steps':100,'screen_attempts_per_program':100,
        'confirmation_attempts_per_program':400,
        'noninferiority':{'mean_pic50_margin':.03,'MMFF_median_relative_increase':.10,'MMFF_p90_relative_increase':.10},
        'variants':variants(),'evaluation':'Paired independent-batch confidence intervals; exact effective-program equality permits shared inference with explicit certificates',
        'causal_scope':'Instruction interventions for this evidence/task and subagent simulator. No general LLM or target-general causal claim.'}
    write_json(out/'design.json',design)
    for v in design['variants']:
        count=2 if v['id'] in ('legacy_legacy','compact_legacy','legacy_compact','compact_compact') else 1
        for rep in range(count):export_request(root,out/'evidence.json',out/'agents'/v['id']/f'replicate_{rep}',v,'Analyst')
    return len(evidence)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',default=str(Path(__file__).resolve().parents[1]));p.add_argument('--output',required=True)
    a=p.parse_args();print({'evidence_rows':prepare(a.repo,a.output)})
