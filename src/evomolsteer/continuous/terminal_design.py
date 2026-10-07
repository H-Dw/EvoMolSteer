"""Behavior, source and numerical semantics of terminal-credit Agent designs."""
import numpy as np
from .affinity_skill import audit_behavior
from .endpoint_design import compile_endpoint
from ..io import read_json, digest, write_json


def audit_terminal_behavior(skill, task, prompt, response, reference, output=None):
    response_path = response
    r, data = read_json(response), read_json(task)
    result = audit_behavior(skill, task, r)
    if r['prompt_sha256'] != digest(prompt) or r['terminal_reference_sha256'] != digest(reference):
        raise ValueError('Literal prompt and terminal teacher binding required')
    c = r['lineage_case']
    exact = data['lineage_case']
    keys = ('terminal_score_time', 'nodes_fewer_than3_ancestors', 'nodes_fewer_than6_ancestors',
            'terminal_geometry_outside_window_used', 'whole_window_credit_strata_allowed',
            'decoded_final_rescore_label', 'future_resampling_in_labels', 'base_prior', 'near_duplicate_RMS_threshold_A',
            'teacher_cap_per_batch', 'original_teacher_cap_per_batch')
    if any(c.get(k) != exact[k] for k in keys) or c.get('extinction_affinity_label', 'missing') is not None:
        raise ValueError('Agent confused extinction, score semantics or missing credit strata')
    if c.get('initial_alive_median') != exact['initial_alive_median'] or c.get('teacher_counts_first_last') != exact['teacher_counts_first_last']:
        raise ValueError('Agent ignored root-collapse counterevidence')
    if (r['survival_is_affinity_causality'] is not False or r['descendant_count_prior_bonus'] is not False
            or r['early_diversity_narrowing_acknowledged'] is not True):
        raise ValueError('Biased survival credit incorrectly treated as a causal affinity map')
    d = r['reward_design']
    required = dict(reward_view='endpoint_pointcloud', derivative_path='flowr_endpoint_vjp',
        coordinate_representation='predicted_endpoint_world_A', window=data['window'],
        reference_sha256=data['terminal_reference_sha256'], native_rms_ratio=.3, time_ramp_power=0.,
        teacher_neighbors=4, teacher_score_beta=2., teacher_endpoint_temperature_A2=4.)
    if any(d.get(k) != v for k, v in required.items()):
        raise ValueError('First target-library comparison must isolate the credit revision')
    audit = r['one_time_numerical_audit']
    post = r['post_native_reward_response']
    if (r['production_extra_forward_calls_per_step'] != 0 or r['full_zero_validation_required'] is not True
            or audit.get('executed') is not False or audit.get('additional_joint_FLOWR_forward_calls') != 4
            or post.get('kind') != 'first_order_only_at_old_x_t' or post.get('measured_post_native_endpoint_reward') is not False):
        raise ValueError('Runtime cost/validation evidence misreported')
    result.update(terminal_credit_response_verified=True, response_sha256=digest(response_path),
        prompt_sha256=digest(prompt), reference_sha256=digest(reference),
        missing_extinction_not_low_affinity_verified=True, sparse_early_credit_not_filled_verified=True)
    if output:
        write_json(output, result)
    return result


def compile_terminal(response, base, reference, reference_sha):
    if reference.get('reference_variant') != 'terminal-descendant-endpoint-library-1.0':
        raise ValueError('Explicit registered terminal-credit reference required')
    p = compile_endpoint(response, base, reference, reference_sha)
    if p['reward_view'] != 'endpoint_pointcloud':
        raise ValueError('Terminal teachers only support intact endpoint point-cloud loss')
    p.update(mixture_temperature=.5, pointcloud_delta_A=1.,
        target_definition='coherent_terminal_descendant_credit_endpoint_ancestors',
        reference_variant=reference['reference_variant'])
    from ..generation.endpoint_reward import EndpointGeometryReward
    EndpointGeometryReward(p, reference)
    return p
