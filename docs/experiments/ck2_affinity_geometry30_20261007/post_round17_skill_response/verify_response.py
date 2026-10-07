"""Verify actual post-experiment Agent behavior against immutable supplied bytes."""
import hashlib,json,math
from pathlib import Path

def verify(folder):
    folder=Path(folder);sha=lambda b:hashlib.sha256(b).hexdigest()
    inp=json.loads((folder/'input.json').read_text(encoding='utf8'))
    r=json.loads((folder/'response.json').read_text(encoding='utf8'))
    assert r['input_sha256']==sha((folder/'input.json').read_bytes())
    assert r['prompt_sha256']==sha((folder/'prompt.txt').read_bytes())
    for key,d in inp['documents'].items():
        raw=d['content_utf8'].encode('utf8')
        assert len(raw)==d['bytes'] and sha(raw)==d['sha256']
        assert r['source_bindings'][key]=={k:v for k,v in d.items() if k!='content_utf8'}
    obj=lambda key:json.loads(inp['documents'][key]['content_utf8'])
    old,new=obj('round7_outcome'),obj('round17_outcome')
    assert r['selected_global_parent_round']==7
    assert old['all_head_change_vs_native']>new['all_head_change_vs_native']>0
    contrasts=r['numerical_contrasts']
    assert math.isclose(contrasts['R17_minus_native_mean'],new['all_head_change_vs_native'],abs_tol=1e-12)
    assert math.isclose(contrasts['R17_minus_R7_mean'],new['all_head_change_vs_native']-old['all_head_change_vs_native'],abs_tol=1e-12)
    instruction=r['response_to_skill']
    assert instruction['strength_insufficiency']['status']=='not_supported_as_primary_explanation'
    assert instruction['coordinate_target_revision']['status']=='leading_testable_hypothesis'
    assert instruction['input_feature_insufficiency']['status']=='possible_but_not_established'
    for key in ['head_gradient_used','graph_change_gate_used','particle_resampling_or_clone_added','source_or_active_program_changed']:
        assert instruction[key] is False
    assert instruction['extra_per_step_head_calls']==instruction['extra_per_step_forward_calls']==0
    control=r['active_control_audit'];assert control['active_steps']==50 and control['nonzero_control_fraction']==1
    assert math.isclose(control['RMS_ratio_R17_to_R7'],new['actual_window_shape']['mean_injection_rms_A']/old['actual_window_shape']['mean_injection_rms_A'],abs_tol=1e-12)
    assert control['finite_difference']['passed'] is True
    assert control['finite_difference']['one_time_extra_forward_calls']==4
    assert r['lineage_counterevidence']['extinction_label'] is None
    assert r['lineage_counterevidence']['survival_tests']['q_below005_count']==0
    assert r['historical_exact_value_binding']['equal_budget_paired_comparison'] is False
    assert r['historical_exact_value_binding']['original_full_experiment_global_max_verified'] is False
    assert {a['round'] for a in r['registered_ablation_assessment']}=={19,24}
    assert all(a['execution_result_in_this_input'] is None and a['cannot_test'] for a in r['registered_ablation_assessment'])
    report={'passed':True,'scope':'Actual subagent response on frozen post-R17 observations; semantic factual checks, not causal LLM attribution',
        'selected_global_parent_round':r['selected_global_parent_round'],'all_embedded_source_bytes_verified':True,
        'dynamic_source_policy':'Compare frozen embedded source bytes; later campaign-summary updates are not substituted',
        'response_sha256':sha((folder/'response.json').read_bytes()),'input_sha256':r['input_sha256'],'prompt_sha256':r['prompt_sha256'],
        'audit_code_sha256':sha(Path(__file__).read_bytes())}
    (folder/'behavior_audit.json').write_bytes((json.dumps(report,indent=2)+'\n').encode('utf8'))
    return report

if __name__=='__main__':print(json.dumps(verify(Path(__file__).resolve().parent)))
