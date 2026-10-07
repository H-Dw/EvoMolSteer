"""Prepare a literal Skill/API simulation from completed campaign evidence."""
from pathlib import Path
import argparse
from evomolsteer.io import read_json,write_json,digest


def prepare(root):
    root=Path(root);e=root/'docs/experiments/ck2_affinity_geometry30_20261007';out=e/'persistent_skill_test'
    if out.exists():raise FileExistsError(out)
    old=read_json(e/'sparse_skill_test_v2/input.json');prior=e/'endpoint_mining/sparse_coordinate_prior_v2.json'
    history=e/'kinematics_mining_v3/kinematics_prior.json';h=read_json(history)
    records=[read_json(p) for p in sorted(e.glob('round_*.outcome.json'))];head=read_json(e/'round_06/local/head_window_response.json')
    zero=read_json(e/'round_06/local/window/report.json');control=read_json(e/'round_06/local/coordinate_audit.json')
    candidates=[{'round':r['round'],'all_head_change_vs_native':r['all_head_change_vs_native'],
        'valid_rate_change':r['valid_rate_change'],'score_coverage':r['head_coverage'],
        'all_head_mean':r['terminal']['all_pic50_on_rescore_mean'],'best_valid_head':r['best_valid_head'],
        'unique_Top5_mean':r['unique_Top5_mean'],'negative_MMFF_relative_change':r['negative_MMFF_relative_change']} for r in records]
    packet={'schema_version':'persistent-coordinate-agent-input-1.0','candidate_programs':candidates,'response_cases':old['response_cases'],
        'primary_objective':'predicted_affinity','reference_sha256':old['reference_sha256'],'prior_sha256':digest(prior),
        'prior_path':str(prior),'kinematics_prior_sha256':digest(history),'kinematics_prior_path':str(history),
        'history_feature_names':[f for f,w in zip(h['features'],h['history_feature_weights']) if w],
        'trajectory_counterevidence':{'round':6,'mean_window_last_node_gain':sum(v['mean_delta'] for v in head['window_end_nodes'])/2,
            'final_mean_gain':next(r for r in records if r['round']==6)['all_head_change_vs_native'],
            'mean_injection_rms_A':next(r for r in records if r['round']==6)['actual_window_shape']['mean_injection_rms_A'],
            'required_intervention':'persistent_coordinate_target_revision','actual_zero_passed':zero['zero_equivalence_passed'],
            'actual_FD_passed':control['coordinate_preflight']['passed']},
        'current_best_interpretation':'R7 full coordinate pointcloud improved final affinity at comparable dose; prototype masking is exploratory, not an already proven improvement.',
        'history_label_semantics':h['target_semantics'],'history_first_node_observed':False,
        'history_effects_path':str(e/'kinematics_mining_v3/whole_window_velocity_effects.parquet'),
        'landmark_context_path':str(e/'endpoint_mining/landmark_context.json'),
        'steer_benchmark':{'historical_subset_mean':7.510350561141967,'historical_subset_best_valid':8.347940444946289,
                           'full_original_max_verified':False,'paired_equal_budget':False},
        'reward_registry':{'first_family':'endpoint_supported_attractor','first_eta':.3,'first_history_strength':0.,
            'history_strength_candidates':[.25,1.,4.],'geometry_block_weights':[0.,1.,2.,1.],
            'salience':'observed_abs_effect','prototype_robust_delta':2.,'mixture_temperature':.5,'teacher_score_beta':2.}}
    skill=root/'docs/experiments/skill_ablation_20261007/legacy_skills/affinity-persistent-coordinate.md';out.mkdir(parents=True);write_json(out/'input.json',packet)
    text=('Simulate Analyst and Designer using exactly '+str(skill)+'. Read the input, two priors, velocity effect table, landmark context and implementation generation/persistent_reward.py. '+
          'Write literal UTF-8 response.json and concise response.md only. Bind skill_sha256, input_sha256, prompt_sha256, prior_sha256, kinematics_prior_sha256. '+
          'Return primary_objective, affinity_head_gradient, selected_round, score_coverage_interpretation, feature_priority=3D_coordinates, coordinate_features, per_step_affinity_calls=0 and response_actions in the same audit schema as sparse_skill_test_v2/response.json. '+
          'Add trajectory_case with intervention, observed_mean_window_gain, observed_final_gain, active_control_confirmed and no_effect_does_not_follow_from_small_mean=true. '+
          'Return history_feature_names, production_extra_forward_calls_per_step=0, full_zero_validation_required=true, first-order-only post_native_reward_response and one_time_numerical_audit. '+
          'reward_design includes reference_sha256, window from prior, reward_view=endpoint_supported_attractor, derivative_path=flowr_endpoint_vjp, coordinate_representation=predicted_endpoint_world_A, '+
          'native_rms_ratio, time_ramp_power, four geometry_block_weights, exact74 geometry_feature_weights, direction_mode=linear_node_effect, exact geometry_direction_functions, '+
          'history_strength, history_time_power, geometry_salience_mode, prototype_robust_delta, mixture_temperature, teacher_score_beta. '+
          'First isolate bounded attraction at matched eta.3/history0. Do not invent data, claim covariance/statistics as causal, or claim new GPU validation. Save concise explicit rationale and uncertainties, not private deliberation.')
    (out/'prompt.txt').write_text(text,encoding='utf8',newline='\n')
    return {'folder':str(out),'skill_sha256':digest(skill),'input_sha256':digest(out/'input.json'),'prompt_sha256':digest(out/'prompt.txt')}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1]);print(prepare(p.parse_args().repo))
