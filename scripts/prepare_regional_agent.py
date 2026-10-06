from pathlib import Path
import argparse
import pandas as pd
from evomolsteer.io import read_json,write_json,digest
parser=argparse.ArgumentParser();parser.add_argument('--repo',default=str(Path(__file__).resolve().parents[1]));parser.add_argument('--output',required=True);parser.add_argument('--completed-through',type=int,default=10);args=parser.parse_args()
root=Path(args.repo);ev=root/'docs/experiments/ck2_affinity_geometry30_20261007';out=Path(args.output)
if out.exists():raise FileExistsError('Immutable literal Agent input folder already exists: '+str(out))
if args.completed_through<10:raise ValueError('R9/R10 counterevidence required')
out.mkdir(parents=True)
skill=root/'skills/affinity-regional-pointcloud/SKILL.md';prior=ev/'regional_mining_v1/region_prior.json'
old=read_json(ev/'persistent_skill_test/input.json');data=read_json(prior)
t=pd.read_parquet(ev/'regional_mining_v1/whole_window_effects.parquet')
rows=[read_json(ev/f'round_{n:02d}.outcome.json') for n in range(1,args.completed_through+1)]
packet={'candidate_programs':[dict(round=r['round'],all_head_change_vs_native=r['all_head_change_vs_native'],score_coverage=r['head_coverage'],valid_rate_change=r['valid_rate_change'],unique_head_change_vs_native=r['unique_head_change_vs_native'],eta=r['native_rms_ratio']) for r in rows],
    'response_cases':old['response_cases'],'window':data['window'],'reference_sha256':data['reference_sha256'],
    'region_prior_sha256':digest(prior),'region_prior_relative_path':prior.relative_to(root).as_posix(),
    'enrichment_summary':t[t.measure=='enrichment_z'].to_dict('records'),
    'supported_node_functions':data['supported_node_functions'],'independent_batches':data['independent_batches'],
    'information_audit':read_json(ev/'information_audit_v1/information_summary.json'),
    'regression_case':{'round':9,'final_mean_gain':rows[8]['all_head_change_vs_native'],'eta':.3,'mean_step_RMS_A':rows[8]['actual_window_shape']['mean_injection_rms_A']},
    'dose_counterexample':{'round':10,'eta':.51,'final_mean_gain':rows[9]['all_head_change_vs_native'],'parent7_gain':rows[6]['all_head_change_vs_native']},
    'skill_sha256':digest(skill),'constraints':{'no_head_gradient':True,'production_extra_forward_calls_per_step':0,'no_SMC':True,'no_graph_gate':True,'first_regional_eta':.3}}
write_json(out/'input.json',packet)
prompt='''Read the supplied Skill literally and the complete input. Simulate the Analyst/Designer API using real data, without inference or source modification. Save response.json and response.md in this folder. Return JSON fields: skill_sha256,input_sha256,prompt_sha256,region_prior_sha256,primary_objective,selected_round,response_actions [{id,interventions}],per_step_affinity_calls,affinity_head_gradient,feature_priority,coordinate_features,production_extra_forward_calls_per_step,full_zero_validation_required,post_native_reward_response {kind,measured_post_native_endpoint_reward},one_time_numerical_audit {executed,additional_joint_FLOWR_forward_calls},regional_case {selected_region,selection_rule,whole_window_effect_z,first_node_effect_z,last_node_effect_z,constant_signed_attraction_allowed},regression_case {round,observed_final_gain,active_control_confirmed,intervention},dose_counterexample {observed_gain,best_parent_gain,stronger_is_not_necessarily_better},information_case {compressed_teacher_score_correlation,full_teacher_score_correlation,excess_RMS_proves_affinity},overlap_independent_causality,conditional_derivative_semantics,reward_design {reward_view,derivative_path,coordinate_representation,window,reference_sha256,native_rms_ratio,time_ramp_power,geometry_block_weights,teacher_neighbors,teacher_score_beta,teacher_endpoint_temperature_A2,coordinate_region_weights,coordinate_region_radius_A,coordinate_background_weight,mixture_temperature,pointcloud_delta_A},uncertainties. Exact20 weights correspond to landmarks. Do not invent GPU validation. Explain evidence-to-formula rationale in concise prose; no private reasoning transcripts.'''
(out/'prompt.txt').write_text(prompt+'\n',encoding='utf8')
print({'folder':str(out),'skill_sha256':digest(skill),'input_sha256':digest(out/'input.json'),'prompt_sha256':digest(out/'prompt.txt')})
