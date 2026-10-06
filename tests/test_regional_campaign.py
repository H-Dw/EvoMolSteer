import copy
from pathlib import Path
from evomolsteer.io import read_json
from evomolsteer.generation.regional_campaign import propose_regional,apply_regional_fields
from evomolsteer.generation.affinity_campaign import update_program


def fixture():
    root=Path(__file__).resolve().parents[1]
    prior=read_json(root/'docs/experiments/ck2_affinity_geometry30_20261007/regional_mining_v1/region_prior.json')
    rows=[dict(round=n,all_head_change_vs_native=g,head_coverage=1.,valid_rate_change=0.,negative_MMFF_relative_change=.1,negative_MMFF_p90_relative_change=.1) for n,g in [(7,.229),(9,-.04),(10,.207),(13,.2)]]
    p=read_json(root/'configs/experiments/ck2_affinity_geometry30_v1/backtrack_round07.json')
    programs={7:p,13:dict(p,reward_view='endpoint_regional_pointcloud',coordinate_region_weights=[1.]+[0.]*19)}
    return prior,rows,programs


def test_revised_search_retains_affinity_parent_and_does_not_escalate_failed_target():
    prior,rows,programs=fixture()
    for n in range(12,27):
        plan=propose_regional(n,rows,programs,prior)
        assert plan['parent_round']==7 and plan.get('template_round')!=9
    plan=propose_regional(14,rows,programs,prior)
    p=update_program(programs[7],plan,prior['window'],prior['reference_sha256'])
    p=apply_regional_fields(p,plan,prior)
    assert p['teacher_score_beta']==6. and p['mixture_temperature']==.5


def test_empirical_stage_salience_is_unsigned_amplitude_and_frozen_program_is_exact():
    prior,rows,programs=fixture();plan=propose_regional(22,rows,programs,prior)
    p=apply_regional_fields(programs[13],plan,prior)
    assert len(p['coordinate_region_node_salience'])==len(prior['times'])
    assert p['coordinate_region_node_salience'][0][0]>0 and p['coordinate_region_node_salience'][-1][0]>0
    assert apply_regional_fields(p,dict(action='frozen_validation'),prior)==p
    rows.append(dict(rows[0],round=26,frozen_winner=7));rows.append(dict(rows[0],round=27,all_head_change_vs_native=2.))
    assert propose_regional(28,rows,programs,prior)['template_round']==7
