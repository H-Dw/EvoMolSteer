from evomolsteer.generation.persistent_campaign import propose_persistent,apply_persistent_fields,PERSISTENT_KEYS


def row(n,gain):
    return dict(round=n,all_head_change_vs_native=gain,valid_rate_change=0,head_coverage=1,
                negative_MMFF_relative_change=.1,negative_MMFF_p90_relative_change=.1)


def test_negative_experiments_keep_global_parent_and_counterfactuals_are_distinct():
    rows=[row(7,.23),row(8,-.01),row(9,-.04)]
    programs={7:dict(reward_view='endpoint_pointcloud',native_rms_ratio=.3),
              9:dict(reward_view='endpoint_supported_attractor',native_rms_ratio=.3)}
    plans=[propose_persistent(n,rows,programs) for n in range(10,27)]
    assert all(p['parent_round']==7 and p['global_parent_retained'] for p in plans)
    assert plans[0]['template_round']==7 and plans[0]['native_rms_ratio']==.51
    assert plans[1]['template_round']==9 and plans[1]['history_strength']==.25
    assert len({repr(sorted(p.items())) for p in plans})==len(plans)


def test_frozen_validation_never_uses_better_heldout_round_and_dense_clears_history():
    rows=[row(7,.23),dict(row(26,.02),frozen_winner=7),row(27,1.)]
    assert propose_persistent(28,rows,{})['template_round']==7
    p=dict(reward_view='endpoint_pointcloud',geometry_feature_weights=[1],**{k:'unused' for k in PERSISTENT_KEYS})
    dense=apply_persistent_fields(p,{'action':'parent_refinement'})
    assert not any(k in dense for k in PERSISTENT_KEYS) and 'geometry_feature_weights' not in dense
    assert apply_persistent_fields(p,{'action':'frozen_validation'})==p


def test_frozen_program_preserves_history_formula_and_target_metadata():
    from evomolsteer.generation.affinity_campaign import update_program
    p=dict(window=[.1,.6],reference_sha256='immutable',reward_view='endpoint_supported_attractor',
           target_definition='bounded_supported_elite_prototypes_and_lineage_increments',history_strength=1.)
    assert update_program(p,{'action':'frozen_validation'},[.1,.6],'immutable')==p


def test_dense_switch_preserves_shared_score_prior_and_temperature():
    p=dict(reward_view='endpoint_pointcloud',teacher_score_beta=6.,mixture_temperature=1.5,history_strength=4.)
    result=apply_persistent_fields(p,{'action':'parent_refinement'})
    assert result['teacher_score_beta']==6. and result['mixture_temperature']==1.5
    assert 'history_strength' not in result
