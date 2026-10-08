"""Bound Analyst/Designer simulation packets for the registered hypothesis pilot."""
import argparse
from pathlib import Path
from evomolsteer.io import read_json,write_json,digest

p=argparse.ArgumentParser();p.add_argument('--repo',default='.');p.add_argument('--role',choices=['Analyst','Designer'],required=True)
a=p.parse_args();root=Path(a.repo).resolve();docs=root/'docs/experiments/selection_path20_20261008';folder=docs/'agents';folder.mkdir(exist_ok=True)
role=a.role;payload={'role':role,'mining':read_json(docs/'mining_summary.json'),
  'representations':{'selection_accounting':'actual_current_and_proposal_world_A',
    'teacher_library':'predicted_endpoint_world_A','first_root_count':'after_first_selection',
    'warning':'Price/native trends describe actual current/proposal coordinates; the distinct R26 reward teachers are predicted endpoints. Do not merge the representations or interpret root loss as independently identified fitness.'},
  'failure_ledger':read_json(docs/'failure_ledger.json'),'protocol':read_json(docs/'protocol.json'),
  'baseline':read_json(root/'configs/experiments/skill_ablation_v1/incumbent.json'),
  'registered_pilot':{'reward_view':'endpoint_selection_path','selection_path':{'quality':'rank'},
    'change':'Only within-reference quality ordering in the original full point-cloud teacher prior; beta=2 and all R26 controller/geometry parameters unchanged.',
    'status':'Exploratory compatibility hypothesis; not causal affinity evidence or a trained contrastive encoder.'},
  'available_mechanisms':['rank quality prior','ESS base chance','coordinate niches','shrunk local SPD covariance','atomwise robust coordinate residual']}
if role=='Designer':payload['Analyst_response']=read_json(folder/'Analyst.response.json')
source=root/f'skills/{role.lower()}/SKILL.md';module=root/'skills/selection-pressure-path/SKILL.md'
instructions=source.read_text(encoding='utf-8')+'\n\n'+module.read_text(encoding='utf-8')+'\n\n'+(
 'Return JSON with input_sha256, instruction_sha256, selection_feature_q_discoveries, native_feature_q_discoveries, accounting_representation, teacher_representation, first_root_count_semantics, unknown_future_is_failure, clone_unit_for_significance, native_future_fitness_measured, decision (test_registered_pilot or defer), evidence_rules (list), and limitations. Assess the actual measured pilot evidence; no new generator run or code edit. If Designer also return reward_view, selection_path, conditional_gradient_formula, affinity_head_gradient, extra_production_forwards, support_source, and justification. Use descriptive evidence, no private chain of thought. Explicitly distinguish null BH-adjusted selection effects from native trends and the actual clone collapse. Numerical labels are not molecular energy.'
)
write_json(folder/f'{role}.input.json',payload);(folder/f'{role}.instructions.md').write_text(instructions,encoding='utf-8',newline='\n')
write_json(folder/f'{role}.request.json',{'role':role,'input_sha256':digest(folder/f'{role}.input.json'),
 'instruction_sha256':digest(folder/f'{role}.instructions.md'),'base_skill_sha256':digest(source),'module_skill_sha256':digest(module),
 'response_file':str(folder/f'{role}.response.json')})
print(read_json(folder/f'{role}.request.json'))
