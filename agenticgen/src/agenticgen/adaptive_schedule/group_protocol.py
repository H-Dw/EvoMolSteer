"""Experiment settings are data, never target-specific optimizer instructions."""
from .io import defaults

from agenticgen.runtime.paths import host_config
GROUP_ROOT = host_config()["result_root"]

def configuration():
    c = defaults()
    c.update(schema='agenticgen-core1hop-fixedstart-v2', rounds=5, versions=[f'v{i}' for i in range(6, 26)],
             population_size=25, slots_per_seed=5, batch_size=25, batches=1, physical_batch_size=100, hypotheses=4, total_result_budget=20050)
    c['controller'] = {'protocol': 'conditional_learning_v1', 'region': 'core_plus_one_hop',
        'restart_policy': 'shared_r0_before_step10', 'initial_step_ticks': 2, 'max_step_ticks': 3,
        'max_changed_intervals': 2, 'dynamic_memory': 'teacher_version_isolated'}
    c['previous_protocol_root'] = host_config()['previous_protocol_root']
    c['transition_from_version'] = 'v5'
    c['llm'].pop('input_soft_tokens'); c['llm'].pop('input_hard_tokens')
    c['llm'].update(input_soft_bytes=65536, input_hard_bytes=262144, max_output_tokens=32768, timeout=600)
    c['council'] = {'max_output_tokens': 65536, 'decision_audit': True}
    c['execution'] = {'scheduler': 'teacher_ready_queue', 'gpu_owners': 1, 'quality_workers': 16,
                      'logical_batches': 5, 'logical_batch_size': 5, 'native_rng_draw_rows': 20, 'rng': 'native_twenty_row_common_streams_v1'}
    c['agent_objective'].update(policy='four_shared_schedules', round_barrier='four_groups_evaluated',
                                scope='unselected_hypothesis_25_candidates',
        priority=['condition_coordinate_fidelity','mean_affinity_teacher_agreement','best_affinity_bonus'],
        thresholds=None, functional_metric='absolute_mean_pic50_minus_teacher',
        conflict_policy='report_continuous_Pareto_tradeoffs_no_hidden_weight_or_pass_gate')
    c['reporting'].update(primary='unselected_condition_rmsd_A', secondary='mean_affinity_absolute_gap',
        top10_percent_count=3, legacy_top10_count=10, fixed_round=5)
    c['retention']['restart_parent'] = 'shared_r0_step10'
    c['condition_comparison'] = {'ranking_tolerance_A': .02, 'sensitivity_A': [.01,.05],
        'frame': 'protein_fixed', 'ligand_fit_for_primary': False,
        'aggregation': 'mean_of_per_candidate_local_rmsd', 'terminal_state_required': True,
        'band': 'anchored_to_pool_minimum', 'chemical_pose': 'separate_exact_subgraph_diagnostic'}
    c['search_policy'] = {'earliest_expansion_version': 11, 'fine_resolution': .1,
        'range_levels': '1,5,10,15,20,...', 'max_levels_per_version': 1,
        'initial_upper': 1, 'coarse_default': 'upper/10', 'max_executed_range_probes': 1}
    c['rollback'] = {'consecutive_regressions': 2, 'comparison': 'adjacent_completed_versions',
        'pool': 'completed_new_protocol_versions', 'history_preserved': True}
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
