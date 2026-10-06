"""Behavioral binding and compilation of registered regional coordinate loss."""
import math
import numpy as np
from .affinity_skill import audit_behavior
from .endpoint_design import compile_endpoint
from ..io import digest,read_json,write_json

REGIONAL_KEYS=('coordinate_region_weights','coordinate_region_radius_A','coordinate_background_weight',
              'coordinate_region_node_salience','region_prior_sha256','region_prior_relative_path',
              'mixture_temperature','pointcloud_delta_A')


def strongest_mass(data):
    candidates=[r for r in data['enrichment_summary'] if r['component']=='soft_mass' and r['q']<.05
        and r['effect_z']>=.1 and r['positive_batch_fraction']>=12/14]
    if not candidates:raise ValueError('No reproducible positive soft-mass region')
    return max(candidates,key=lambda r:r['effect_z'])


def audit_regional_behavior(skill,task,prompt,response,prior_path,output=None):
    response_path=response;response=read_json(response);data=read_json(task);prior=read_json(prior_path)
    result=audit_behavior(skill,task,response)
    if response['prompt_sha256']!=digest(prompt) or response['region_prior_sha256']!=digest(prior_path):raise ValueError('Exact prompt/local-coordinate prior binding required')
    case=response['regional_case'];strong=strongest_mass(data);region=strong['region'];f=f'landmark_{region:02d}_soft_mass'
    exact=[strong['effect_z'],prior['supported_node_functions'][f]['values_z'][0],prior['supported_node_functions'][f]['values_z'][-1]]
    if case['selected_region']!=region or case['selection_rule']!='strongest_reproducible_positive_soft_mass' or case['constant_signed_attraction_allowed'] is not False:
        raise ValueError('Agent ignored localization/node-sign contradiction')
    if not np.allclose([case[k] for k in ('whole_window_effect_z','first_node_effect_z','last_node_effect_z')],exact,rtol=0,atol=1e-8):raise ValueError('Agent fabricated regional statistics')
    if response['overlap_independent_causality'] is not False:raise ValueError('Overlapping regions mistaken for independent causal motifs')
    reg=response['regression_case']
    if reg['round']!=9 or reg['active_control_confirmed'] is not True or reg['intervention']!='geometry_target_revision' or not math.isclose(reg['observed_final_gain'],data['regression_case']['final_mean_gain'],abs_tol=1e-8):raise ValueError('R9 active-control regression ignored')
    dose=response['dose_counterexample']
    if dose['stronger_is_not_necessarily_better'] is not True or not np.allclose([dose['observed_gain'],dose['best_parent_gain']],[data['dose_counterexample']['final_mean_gain'],data['dose_counterexample']['parent7_gain']],rtol=0,atol=1e-8):raise ValueError('Dose counterexample ignored')
    info=response['information_case'];metrics={m['metric']:m['whole_window_mean'] for m in data['information_audit']['metrics']}
    if info['excess_RMS_proves_affinity'] is not False or not np.allclose([info['compressed_teacher_score_correlation'],info['full_teacher_score_correlation']],
        [metrics['compressed_chosen_teacher_score_correlation'],metrics['full_chosen_teacher_score_correlation']],rtol=0,atol=1e-8):raise ValueError('Information audit incorrectly used as affinity proof')
    if response['conditional_derivative_semantics']!='fixed_attention_assignments_and_priors_endpoint_only':raise ValueError('Conditional geometry derivative mislabeled')
    d=response['reward_design'];weights=np.zeros(len(prior['region_weights']));weights[region]=1.
    required={'reward_view':'endpoint_regional_pointcloud','derivative_path':'flowr_endpoint_vjp','coordinate_representation':'predicted_endpoint_world_A',
        'window':prior['window'],'reference_sha256':prior['reference_sha256'],'native_rms_ratio':.3,'time_ramp_power':0.,
        'teacher_neighbors':4,'teacher_score_beta':2.,'teacher_endpoint_temperature_A2':4.,
        'coordinate_region_radius_A':5.,'coordinate_background_weight':.25,'mixture_temperature':.5,'pointcloud_delta_A':1.}
    if any(d.get(k)!=v for k,v in required.items()) or not np.array_equal(d['coordinate_region_weights'],weights):raise ValueError('First regional intervention must isolate region emphasis from R7')
    audit=response['one_time_numerical_audit'];post=response['post_native_reward_response']
    if (response['production_extra_forward_calls_per_step']!=0 or not response['full_zero_validation_required']
        or audit['executed'] is not False or audit['additional_joint_FLOWR_forward_calls']!=4
        or post['kind']!='first_order_only_at_old_x_t' or post['measured_post_native_endpoint_reward'] is not False):raise ValueError('Runtime validation or forward cost mislabeled')
    result.update(regional_response_verified=True,prompt_sha256=digest(prompt),response_sha256=digest(response_path),
        region_prior_sha256=digest(prior_path),selected_region=region,node_sign_contradiction_verified=True,
        active_control_counterevidence_verified=True,information_limitations_verified=True)
    if output:write_json(output,result)
    return result


def compile_regional(response,base,reference,reference_sha,prior,prior_sha,relative):
    if prior['reference_sha256']!=reference_sha or prior['window']!=reference['window']:raise ValueError('Exact regional reference support required')
    p=compile_endpoint(response,base,reference,reference_sha);d=response['reward_design']
    p.update({k:d[k] for k in REGIONAL_KEYS if k in d})
    p.update(region_prior_sha256=prior_sha,region_prior_relative_path=relative,
        target_definition='coherent_elite_pointcloud_with_detached_regional_attention',
        geometry_block_weights_role='Monitoring only; spatial coordinate attention determines point-cloud force')
    from ..generation.regional_point_reward import RegionalPointReward
    RegionalPointReward(p,reference)
    return p
