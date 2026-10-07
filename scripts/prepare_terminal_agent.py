"""Prepare an immutable real-data Skill behavior test; no fabricated LLM output."""
import argparse
from pathlib import Path
import pandas as pd
from evomolsteer.io import digest, read_json, write_json

p = argparse.ArgumentParser()
p.add_argument('--repo', default=str(Path(__file__).resolve().parents[1]))
p.add_argument('--output', required=True)
p.add_argument('--completed-through', type=int, default=12)
p.add_argument('--mining-folder', default='terminal_lineage_v4')
a = p.parse_args()
root = Path(a.repo)
ev = root/'docs/experiments/ck2_affinity_geometry30_20261007'
out = Path(a.output)
if out.exists(): raise FileExistsError(out)
out.mkdir(parents=True)
skill = root/'skills/affinity-terminal-lineage/SKILL.md'
folder = ev/a.mining_folder
manifest = read_json(folder/'manifest.json')
t = pd.read_parquet(folder/'batch_node_credit.parquet')
effects = pd.read_parquet(folder/'whole_window_survival_effects.parquet')
rows = [read_json(ev/f'round_{n:02d}.outcome.json') for n in range(1, a.completed_through+1)]
packet = dict(skill_sha256=digest(skill), terminal_reference_sha256=digest(folder/'terminal_reference.json.gz'),
    parent_reference_sha256=manifest['parent_reference_sha256'], window=manifest['window'],
    response_cases=read_json(ev/'regional_skill_test/input.json')['response_cases'],
    candidate_programs=[dict(round=r['round'], all_head_change_vs_native=r['all_head_change_vs_native'],
        score_coverage=r['head_coverage'], valid_rate_change=r['valid_rate_change'], unique_head_change_vs_native=r['unique_head_change_vs_native']) for r in rows],
    lineage_case=dict(terminal_score_time=manifest['terminal_metadata'][0]['terminal_score_time'],
        nodes_fewer_than3_ancestors=manifest['nodes_with_fewer_than3_ancestors'],
        nodes_fewer_than6_ancestors=manifest['nodes_with_fewer_than6_ancestors'],
        initial_alive_median=float(t[t.score_time == manifest['times'][0]].live_ancestors.median()),
        teacher_counts_first_last=[manifest['teacher_counts_per_node'][0], manifest['teacher_counts_per_node'][-1]],
        terminal_geometry_outside_window_used=False, whole_window_credit_strata_allowed=False,
        decoded_final_rescore_label=False, future_resampling_in_labels=True, extinction_affinity_label=None,
        base_prior='equal_batch_then_ancestor', near_duplicate_RMS_threshold_A=.001,
        teacher_cap_per_batch=manifest['teachers_per_batch_max'], original_teacher_cap_per_batch=2),
    survival_effects=effects.to_dict('records'), manifest=manifest,
    constraints=dict(no_head_gradient=True, production_extra_forward_calls_per_step=0, no_SMC=True, no_graph_gate=True))
write_json(out/'input.json', packet)
prompt = '''Read the complete supplied Skill and input. Simulate Analyst/Designer on real numeric evidence; no inference or code modification. Save response.json and concise response.md here. Return skill_sha256,input_sha256,prompt_sha256,terminal_reference_sha256,primary_objective,selected_round,response_actions [{id,interventions}],per_step_affinity_calls,affinity_head_gradient,feature_priority,coordinate_features,production_extra_forward_calls_per_step,full_zero_validation_required,post_native_reward_response {kind,measured_post_native_endpoint_reward},one_time_numerical_audit {executed,additional_joint_FLOWR_forward_calls},lineage_case {terminal_score_time,nodes_fewer_than3_ancestors,nodes_fewer_than6_ancestors,initial_alive_median,teacher_counts_first_last,terminal_geometry_outside_window_used,whole_window_credit_strata_allowed,decoded_final_rescore_label,future_resampling_in_labels,extinction_affinity_label,base_prior,near_duplicate_RMS_threshold_A,teacher_cap_per_batch,original_teacher_cap_per_batch},survival_is_affinity_causality,descendant_count_prior_bonus,early_diversity_narrowing_acknowledged,reward_design {reward_view,derivative_path,coordinate_representation,window,reference_sha256,native_rms_ratio,time_ramp_power,geometry_block_weights,teacher_neighbors,teacher_score_beta,teacher_endpoint_temperature_A2},uncertainties. Quote real ancestor-collapse and credit semantics, distinguish equal-batch base mass from nearest-K conditional priors, report original and revised per-batch teacher caps numerically, retain measured affinity parent, select the declared reference-only R7 comparison, and give concise evidence-to-formula reasoning. No private reasoning transcript or claimed GPU result.'''
(out/'prompt.txt').write_bytes((prompt+'\n').encode('utf8'))
print(dict(skill_sha256=digest(skill), input_sha256=digest(out/'input.json'), prompt_sha256=digest(out/'prompt.txt')))
