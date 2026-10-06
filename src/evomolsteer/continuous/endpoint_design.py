"""Compile a data-only Agent design into registered endpoint formulas."""
import copy,math
from ..generation.endpoint_reward import FAMILIES
from ..io import read_json,digest,write_json
from .affinity_skill import audit_behavior

REGISTERED_ENDPOINT_FAMILIES=set(FAMILIES)|{'endpoint_supported_attractor','endpoint_regional_pointcloud'}


def audit_endpoint_behavior(skill,task,prompt,response,reference_sha,output=None):
    result=audit_behavior(skill,task,response)
    if response.get('prompt_sha256')!=digest(prompt):raise ValueError('Literal prompt not bound by Agent')
    design=response['reward_design']
    if design.get('reference_sha256')!=reference_sha or design['derivative_path']!='flowr_endpoint_vjp':
        raise ValueError('Agent did not implement the representation intervention')
    if design['reward_view'] not in FAMILIES or response.get('production_extra_forward_calls_per_step')!=0:
        raise ValueError('Unregistered geometry or additional production forward')
    response_kind=response['post_native_reward_response']
    if response_kind['measured_post_native_endpoint_reward'] is not False or response_kind['kind']!='first_order_only_at_old_x_t':
        raise ValueError('Agent mislabeled lagged reward response')
    if not response['full_zero_validation_required']:raise ValueError('New derivative path requires fresh zero check')
    semantics=response['label_semantics'];label=str(semantics['recorded_label']).lower();head=str(semantics['head_reads']).lower()
    if 'joint' not in label or 'latent' not in head:raise ValueError('Incorrect score/endpoint label semantics')
    audit=response['one_time_numerical_audit']
    if audit['additional_joint_FLOWR_forward_calls']!=4 or audit['executed'] is not False:
        raise ValueError('One-time audit cost or execution status misreported')
    result.update(representation_intervention=True,derivative_path=design['derivative_path'],reward_view=design['reward_view'],
        reference_sha256=reference_sha,prompt_sha256=digest(prompt),post_native_response_first_order_only=True,
        full_zero_validation_required=True,one_time_numerical_forward_calls=4)
    if output:write_json(output,result)
    return result


def compile_endpoint(response,base,reference,reference_sha):
    d=response['reward_design']
    if response['primary_objective']!='predicted_affinity' or response['affinity_head_gradient'] is not False:
        raise ValueError('Current affinity-first, no-head-gradient objective required')
    if response.get('production_extra_forward_calls_per_step')!=0 or not response.get('full_zero_validation_required'):
        raise ValueError('One-forward production and fresh full-zero validation required')
    if d['reward_view'] not in REGISTERED_ENDPOINT_FAMILIES or d['derivative_path']!='flowr_endpoint_vjp':raise ValueError('Registered real-VJP family required')
    if d['window']!=reference['window'] or reference['schema_version']!='affinity-endpoint-library-1.0':raise ValueError('Dynamic evidence support mismatch')
    if d['coordinate_representation'] not in ('predicted_endpoint_world_A','native_model_endpoint_world_coordinates_A'):
        raise ValueError('No actual-state or identity-gradient substitution')
    eta=float(d['native_rms_ratio']);power=float(d['time_ramp_power']);weights=list(map(float,d['geometry_block_weights']))
    if not math.isfinite(eta) or not .1<=eta<=3 or not math.isfinite(power) or power<0:raise ValueError('Bounded finite dose/schedule required')
    if len(weights)!=4 or not all(math.isfinite(v) and v>=0 for v in weights) or sum(weights)<=0:raise ValueError('Invalid geometric block weights')
    p=copy.deepcopy(base)
    p.update(reward_view=d['reward_view'],derivative_path=d['derivative_path'],window=reference['window'],reference_sha256=reference_sha,
        native_rms_ratio=eta,time_ramp_power=power,geometry_block_weights=weights,
        family='coordinate_only_affinity_endpoint',target_definition='recorded_joint_head_endpoint_strata',
        objective_profile='affinity_primary_coordinate30',affinity_head_gradient=False,additional_per_step_affinity_calls=0,
        coordinate_representation='predicted_endpoint_world_A',record_reward_response=True,
        dose_reference='predictive_flow',preserve_native_rigid_pose=False,
        constraints={'max_atom_step_A':.15,'max_cumulative_rms_A':6.,'severe_receptor_clash_A':.8,'backtrack_attempts':7})
    for key in ('teacher_neighbors','teacher_score_beta','teacher_endpoint_temperature_A2'):
        if key in d:p[key]=d[key]
    if p['reward_view']=='endpoint_pointcloud':p['geometry_block_weights_role']='Monitoring only, not point-cloud force weights'
    return p
