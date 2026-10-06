import pytest
from evomolsteer.continuous.affinity_skill import classify,affinity_choice

def test_affinity_priority_survives_physical_tradeoff():
    rows=[{'round':2,'all_head_change_vs_native':.08,'valid_rate_change':0,'energy_improvement':-.1},
          {'round':9,'all_head_change_vs_native':.03,'valid_rate_change':0,'energy_improvement':.1}]
    assert affinity_choice(rows)==2

def test_response_is_not_inferred_from_head_alone():
    assert classify(.007)=='flat_response'
    assert classify(-.04)=='negative_affinity'
    assert classify(.04)=='promising_gain'
    assert classify(.08)=='meaningful_gain'

def test_missing_head_coverage_not_hidden():
    with pytest.raises(ValueError):affinity_choice([{'round':1,'all_head_change_vs_native':.8,'valid_rate_change':0,'score_coverage':.4}])
