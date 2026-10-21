"""Experiment settings are data, never target-specific optimizer instructions."""
from .io import defaults

GROUP_ROOT = '/opt/MolSteer/generated_datasets/plip_schedule_council_5nn4_r12_h4_b100_crn_s42_20261022'

def configuration():
    c = defaults()
    c.update(schema='plip-council-crn-20versions-v1', rounds=5, versions=[f'v{i}' for i in range(1, 21)],
             batch_size=100, batches=1, physical_batch_size=400, hypotheses=4, total_result_budget=80200)
    c['llm'].pop('input_soft_tokens'); c['llm'].pop('input_hard_tokens')
    c['llm'].update(input_soft_bytes=65536, input_hard_bytes=262144, max_output_tokens=32768, timeout=600)
    c['council'] = {'max_output_tokens': 65536, 'decision_audit': True}
    c['execution'] = {'scheduler': 'teacher_ready_queue', 'gpu_owners': 1, 'quality_workers': 16,
                      'logical_batches': 5, 'logical_batch_size': 20, 'rng': 'native_twenty_row_common_streams_v1'}
    c['agent_objective'].update(policy='four_shared_schedules', round_barrier='four_groups_evaluated',
                                scope='unselected_hypothesis_100_candidates')
    c['reporting'].update(primary='unselected_hypothesis_mean_pic50', fixed_round=5)
    return c

def input_limit(config):
    return config['llm'].get('input_hard_bytes', config['llm'].get('input_hard_tokens', 131072))

def budget_context(config, observed_round):
    total = config['rounds']; remaining = max(0, total-observed_round); fraction = observed_round/total
    return {'completed_rounds': observed_round, 'total_rounds': total, 'remaining_sampling_rounds': remaining,
            'consumed_fraction': fraction,
            'decision_phase': 'review_only' if not remaining else 'direction_discovery' if fraction < .4 else
                              'evidence_refinement' if fraction < .8 else 'supported_consolidation',
            'instruction': 'Phase is a budget cue, not evidence that a direction is correct. Final-round proposals are unexecuted suggestions.'}
