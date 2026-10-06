import pytest
from evomolsteer.generation.prototypes import write_json
from evomolsteer.generation.coordinate_exploration import proposal,regression_triggers,outcome_record


def test_continuation_requires_retained_preceding_result_and_keeps_one_factor(tmp_path):
    with pytest.raises(ValueError,match='Preceding'):proposal(15,tmp_path)
    write_json(tmp_path/'round_14.outcome.json',{})
    parent,change,_=proposal(15,tmp_path)
    assert parent==3 and change=={'mixture_temperature':.05}
    for value in (14,31):
        with pytest.raises(ValueError):proposal(value,tmp_path)


def test_escalation_requires_all_physical_and_head_gates(tmp_path):
    for quality,expected in [(0.01,.01),(-.01,.2)]:
        write_json(tmp_path/'round_23.outcome.json',{'all_head_change_vs_native':.01,
            'MMFF_relief_relative_change':quality,'surround_RMS_improvement_fraction':.01})
        assert proposal(24,tmp_path)[1]=={'native_rms_ratio':expected}


def test_regressions_trigger_backtracking_instead_of_success_or_stop():
    x={'shape_improvement_fraction':.02,'all_head_change_vs_native':.01,
       'MMFF_relief_relative_change':.04,'surround_RMS_improvement_fraction':-.1}
    assert regression_triggers(x)==['MMFF_relief_relative_change','surround_RMS_improvement_fraction']


def test_legacy_outcomes_use_retained_reports_without_overwriting(tmp_path):
    from evomolsteer.io import digest
    path=tmp_path/'round_03.outcome.json';write_json(path,{'round':3,'shape_improvement_fraction':.02})
    a={'all_pic50_on_rescore_mean':7.,'all_mmff_relief_per_heavy_median':.5,'all_relax_rms_surround_A_mean':.4}
    write_json(tmp_path/'round_01/local/terminal_report.json',{'results':{'unguided':a}})
    write_json(tmp_path/'round_03/local/terminal_report.json',{'results':{'gradient':{**a,'all_pic50_on_rescore_mean':7.1}}})
    before=digest(path);v=outcome_record(path)
    assert digest(path)==before and v['all_head_change_vs_native']==pytest.approx(.1) and v['surround_RMS_improvement_fraction']==0
