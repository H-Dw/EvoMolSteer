from pathlib import Path
from evomolsteer.io import read_json,write_json,digest

root=Path(__file__).resolve().parents[1];folder=root/'docs/experiments/selection_path20_20261008/agents';checks={}
for role in ['Analyst','Designer']:
    request=read_json(folder/f'{role}.request.json');response=read_json(folder/f'{role}.response.json')
    assert request['input_sha256']==digest(folder/f'{role}.input.json')==response['input_sha256']
    assert request['instruction_sha256']==digest(folder/f'{role}.instructions.md')==response['instruction_sha256']
    assert response['selection_feature_q_discoveries']==0 and response['native_feature_q_discoveries']==68
    assert response['unknown_future_is_failure'] is False and response['clone_unit_for_significance'] is False
    assert response['native_future_fitness_measured'] is False and response['decision']=='test_registered_pilot'
    assert response['accounting_representation']=='actual_current_and_proposal_world_A'
    assert response['teacher_representation']=='predicted_endpoint_world_A' and response['first_root_count_semantics']=='after_first_selection'
    if role=='Designer':
        assert response['reward_view']=='endpoint_selection_path' and response['selection_path']=={'quality':'rank'}
        assert response['affinity_head_gradient'] is False and response['extra_production_forwards']==0
    checks[role]={'response_sha256':digest(folder/f'{role}.response.json'),'input_and_literal_instruction_binding':True,
                 'observed_null_selection_and_native_trend_distinguished':True,'clone_censoring_and_native_future_limits':True}
write_json(folder/'skill_response_audit.json',{'checks':checks,'scope':'Analyst simulation plus one representation clarification repair, then Designer simulation; literal evidence/response consistency, not causal prompt ablation.'})
print(checks)
