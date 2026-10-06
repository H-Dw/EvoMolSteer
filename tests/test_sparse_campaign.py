from evomolsteer.generation.sparse_campaign import propose_sparse,apply_sparse_fields,SPARSE_KEYS


def row(n,delta,valid=0):
    return dict(round=n,all_head_change_vs_native=delta,reward_view='endpoint_direction',valid_rate_change=valid,
                head_coverage=1.,negative_MMFF_relative_change=0.,negative_MMFF_p90_relative_change=0.)


def test_sparse_policy_keeps_global_parent_and_changes_bad_targets():
    programs={5:dict(native_rms_ratio=.3),6:dict(native_rms_ratio=.3,geometry_feature_weights=[1],geometry_direction_mode='linear_node_effect')}
    rows=[row(5,.06),row(6,-.06)]
    plan=propose_sparse(9,rows,programs)
    assert plan['parent_round']==5 and plan['direction_mode']=='node_contrast' and plan['native_rms_ratio']==.3
    rows[-1]['all_head_change_vs_native']=.001
    assert propose_sparse(9,rows,programs)['native_rms_ratio']==.51
    assert propose_sparse(6,rows[:1],programs)['action']=='node_fidelity_intervention'


def test_dense_switch_removes_sparse_fields_but_frozen_validation_is_identical():
    p={k:'value' for k in SPARSE_KEYS};p.update(reward_view='endpoint_pointcloud',native_rms_ratio=.3)
    dense=apply_sparse_fields(p,{'sparse':False},{})
    assert all(k not in dense for k in SPARSE_KEYS)
    assert apply_sparse_fields(p,{'action':'frozen_validation'},{})==p
    rows=[row(5,.06),dict(row(26,.02),frozen_winner=5)]
    assert propose_sparse(27,rows,{})['template_round']==5


def test_registered_block_comparisons_change_dose_across_cycles():
    programs={5:dict(native_rms_ratio=.3),6:dict(native_rms_ratio=.3,geometry_feature_weights=[1],geometry_direction_mode='linear_node_effect')}
    rows=[row(5,.06),row(6,.04)]
    assert propose_sparse(10,rows,programs)['native_rms_ratio']==.3
    assert propose_sparse(18,rows,programs)['native_rms_ratio']==.51
    assert propose_sparse(26,rows,programs)['native_rms_ratio']==.867
