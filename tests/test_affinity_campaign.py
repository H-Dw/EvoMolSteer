from evomolsteer.generation.affinity_campaign import select_parent,propose

def rows():
    return [{'round':1,'reward_view':'motif_mixture','all_head_change_vs_native':.08,'valid_rate_change':0,'negative_MMFF_relative_change':-.1,'negative_MMFF_p90_relative_change':-.2},
            {'round':2,'reward_view':'affinity_landmark','all_head_change_vs_native':.03,'valid_rate_change':0,'negative_MMFF_relative_change':.1,'negative_MMFF_p90_relative_change':.1}]
def test_primary_gain_is_not_replaced_by_energy_objective():assert select_parent(rows())['round']==1
def test_flat_response_changes_delivered_dose():
    a=rows()[:1];a[0]['all_head_change_vs_native']=.001
    p=propose(2,a,{1:{'reward_view':'motif_mixture','native_rms_ratio':.3}})
    assert p['native_rms_ratio']>.3
def test_negative_response_revises_coordinate_target():
    a=rows()[:1];a[0]['all_head_change_vs_native']=-.04
    p=propose(2,a,{1:{'reward_view':'motif_mixture','native_rms_ratio':.3}})
    assert p['action']=='geometry_target_revision' and p['reward_view']!='motif_mixture'
