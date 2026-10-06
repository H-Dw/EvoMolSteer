"""Bind actual Agent decisions to bounded geometry and recorded lineage support."""
import math
from .affinity_skill import audit_behavior
from .sparse_design import validate_sparse_fields
from .endpoint_design import compile_endpoint
from ..io import read_json,digest,write_json

HISTORY_KEYS=('history_strength','history_time_power','prototype_robust_delta',
              'geometry_salience_mode','mixture_temperature','teacher_score_beta')


def audit_persistent_behavior(skill,task,prompt,response,prior_path,history_path,output=None):
    response_path=response;response=read_json(response_path);data=read_json(task)
    result=audit_behavior(skill,task,response)
    prior=read_json(prior_path);history=read_json(history_path);d=response['reward_design']
    if response['prompt_sha256']!=digest(prompt) or response['prior_sha256']!=digest(prior_path) or response['kinematics_prior_sha256']!=digest(history_path):
        raise ValueError('Exact prompt and two prior bindings required')
    validate_sparse_fields(d,prior)
    if (d['reward_view']!='endpoint_supported_attractor' or d['derivative_path']!='flowr_endpoint_vjp'
            or d['coordinate_representation']!='predicted_endpoint_world_A' or d['window']!=prior['window']
            or d['reference_sha256']!=prior['reference_sha256'] or history['reference_sha256']!=prior['reference_sha256']):
        raise ValueError('Actual endpoint derivative and dynamic recorded support required')
    case=response['trajectory_case'];expected=data['trajectory_counterevidence']
    if (case['intervention']!='persistent_coordinate_target_revision' or case['active_control_confirmed'] is not True
            or case['no_effect_does_not_follow_from_small_mean'] is not True):
        raise ValueError('Agent ignored active-control/window-to-final counterevidence')
    for name,target in [('observed_mean_window_gain',expected['mean_window_last_node_gain']),('observed_final_gain',expected['final_mean_gain'])]:
        if not math.isclose(case[name],target,rel_tol=0,abs_tol=1e-8):raise ValueError('Trajectory evidence fabricated')
    if response['history_feature_names']!=data['history_feature_names']:raise ValueError('Unsupported history fields promoted')
    if response['production_extra_forward_calls_per_step']!=0 or response['full_zero_validation_required'] is not True:raise ValueError('Extra production inference or missing full zero check')
    post=response['post_native_reward_response']
    if post['kind']!='first_order_only_at_old_x_t' or post['measured_post_native_endpoint_reward'] is not False:raise ValueError('Fresh endpoint reward mislabeled')
    if d['native_rms_ratio']!=.3 or d['history_strength']!=0.:raise ValueError('First bounded target must match dose with history disabled')
    numerical=response['one_time_numerical_audit']
    if numerical['executed'] is not False or numerical['additional_joint_FLOWR_forward_calls']!=4:raise ValueError('New loss numerical validation misreported')
    result.update(persistent_response_verified=True,prompt_sha256=digest(prompt),response_sha256=digest(response_path),
                  prior_sha256=digest(prior_path),kinematics_prior_sha256=digest(history_path),
                  trajectory_intervention_verified=True,history_feature_names=response['history_feature_names'])
    if output:write_json(output,result)
    return result


def compile_persistent(response,base,reference,reference_sha,prior,prior_sha,history,history_sha,history_relative):
    d=response['reward_design'];weights,mode,functions=validate_sparse_fields(d,prior)
    if prior['reference_sha256']!=reference_sha or history['reference_sha256']!=reference_sha:raise ValueError('Prior identity mismatch')
    p=compile_endpoint(response,base,reference,reference_sha)
    p.update(geometry_feature_weights=weights,geometry_direction_mode=mode,geometry_direction_functions=functions,
             geometry_direction_times=prior['times'],coordinate_prior_sha256=prior_sha,
             geometry_weight_normalization='within_block_supported_mass',history_prior_sha256=history_sha,
             history_prior_relative_path=history_relative,target_definition='bounded_supported_elite_prototypes_and_lineage_increments')
    p.update({key:d[key] for key in HISTORY_KEYS})
    # Runtime construction validates numerical ranges and both registered priors.
    from ..generation.persistent_reward import PersistentEndpointReward
    PersistentEndpointReward(p,reference,history)
    return p
