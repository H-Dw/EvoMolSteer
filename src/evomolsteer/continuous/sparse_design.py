"""Data-only constrained compilation and observed Skill response checks."""
import math
from pathlib import Path
from .affinity_skill import audit_behavior
from .endpoint_design import compile_endpoint
from ..io import digest,read_json,write_json


def validate_sparse_fields(design,prior):
    weights=list(map(float,design['geometry_feature_weights']))
    if len(weights)!=len(prior['features']) or not all(math.isfinite(v) and 0<=v<=4 for v in weights) or sum(weights)<=0:
        raise ValueError('Finite supported feature weights required')
    if any(w>0 and not allowed for w,allowed in zip(weights,prior['feature_weights'])):raise ValueError('Unsupported or floor-scaled reward field')
    blocks=design['geometry_block_weights'];cuts=[0,3,9,14,len(weights)]
    if len(blocks)!=4 or not any(blocks[k]>0 and sum(weights[a:b])>0 for k,(a,b) in enumerate(zip(cuts[:-1],cuts[1:]))):
        raise ValueError('No effective nonzero coordinate block')
    mode=design['direction_mode']
    if mode not in ('node_contrast','legendre_effect','linear_node_effect'):raise ValueError('Unregistered trend mode')
    functions=design.get('geometry_direction_functions',{})
    field_functions=prior['node_effect_functions'] if mode=='linear_node_effect' else prior['direction_functions']
    expected={f:field_functions[f] for f,w in zip(prior['features'],weights) if w>0}
    if mode=='legendre_effect' and any(not prior.get('curve_fidelity_audit',{}).get(f,{}).get('legendre_runtime_allowed',False) for f in expected):
        raise ValueError('Recorded polynomial contradicts supported node behavior; use node contrast or empirical interpolation')
    if mode in ('legendre_effect','linear_node_effect') and functions!=expected:raise ValueError('Trend functions must be copied from recorded support, not invented')
    if mode=='node_contrast' and functions:raise ValueError('Node mode must not mislabel unused trend functions')
    return weights,mode,functions


def audit_sparse_behavior(skill,task,prompt,response,prior_path,output=None):
    response_path=Path(response) if isinstance(response,(str,Path)) else None
    if response_path is not None:response=read_json(response_path)
    result=audit_behavior(skill,task,response);prior=read_json(prior_path)
    if response['prompt_sha256']!=digest(prompt) or response['prior_sha256']!=digest(prior_path):raise ValueError('Exact Skill prompt/prior binding required')
    d=response['reward_design'];weights,mode,functions=validate_sparse_fields(d,prior)
    if d['reward_view']!='endpoint_direction' or d['derivative_path']!='flowr_endpoint_vjp' or d['window']!=prior['window']:
        raise ValueError('Sparse endpoint VJP and dynamic support required')
    if d['coordinate_representation']!='predicted_endpoint_world_A':raise ValueError('Wrong coordinate representation')
    task_data=read_json(task)
    declared=response.get('reference_sha256',d.get('reference_sha256'))
    if declared!=prior['reference_sha256'] or task_data['reference_sha256']!=prior['reference_sha256']:
        raise ValueError('Response/input/prior reference hash mismatch')
    if response.get('production_extra_forward_calls_per_step')!=0 or response.get('full_zero_validation_required') is not True:
        raise ValueError('Production cost and fresh zero-validation contract required')
    post=response['post_native_reward_response']
    if post['kind']!='first_order_only_at_old_x_t' or post['measured_post_native_endpoint_reward'] is not False:
        raise ValueError('Wrong response semantics')
    result.update(prior_sha256=digest(prior_path),prompt_sha256=digest(prompt),response_sha256=digest(response_path) if response_path is not None else None,
                  sparse_response_verified=True,selected_features=[f for f,w in zip(prior['features'],weights) if w>0],
                  direction_mode=mode,recorded_function_count=len(functions))
    if output:write_json(output,result)
    return result


def compile_sparse(response,base,reference,reference_sha,prior,prior_sha):
    if prior['reference_sha256']!=reference_sha or prior['features']!=reference['features']:raise ValueError('Sparse prior/reference mismatch')
    weights,mode,functions=validate_sparse_fields(response['reward_design'],prior)
    if response.get('reference_sha256',response['reward_design'].get('reference_sha256'))!=reference_sha:
        raise ValueError('Designer reference hash mismatch')
    p=compile_endpoint(response,base,reference,reference_sha)
    p.update(geometry_feature_weights=weights,geometry_direction_mode=mode,geometry_direction_functions=functions,
             coordinate_prior_sha256=prior_sha,geometry_weight_normalization='within_block_supported_mass')
    p['geometry_direction_times']=prior['times']
    p['geometry_curve_fidelity_audit']={f:prior['curve_fidelity_audit'][f] for f,w in zip(prior['features'],weights) if w>0}
    return p
