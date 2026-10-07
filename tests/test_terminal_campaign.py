import ast
from pathlib import Path
from evomolsteer.io import read_json
from evomolsteer.generation.terminal_campaign import propose_terminal


def test_credit_policy_retains_global_parent_and_freezes_without_retuning():
    root = Path(__file__).resolve().parents[1]
    prior = read_json(root/'docs/experiments/ck2_affinity_geometry30_20261007/regional_mining_v1/region_prior.json')
    rows = [dict(round=n, all_head_change_vs_native=g, head_coverage=1., valid_rate_change=0.,
        negative_MMFF_relative_change=.1, negative_MMFF_p90_relative_change=.1) for n, g in [(7, .229), (13, .1), (17, -.04)]]
    p = read_json(root/'configs/experiments/ck2_affinity_geometry30_v1/backtrack_round07.json')
    programs = {7:p, 13:dict(p, reward_view='endpoint_regional_pointcloud'), 17:dict(p, reference_variant='terminal-descendant-endpoint-library-1.0')}
    for n in range(17, 27):
        plan = propose_terminal(n, rows, programs, prior)
        assert plan['parent_round'] == 7
    assert propose_terminal(17, rows, programs, prior)['terminal_credit']
    assert propose_terminal(19, rows, programs, prior)['teacher_score_beta'] == 0.
    assert propose_terminal(24, rows, programs, prior)['native_rms_ratio'] == .18
    rows += [dict(rows[0], round=26, frozen_winner=7), dict(rows[0], round=27, all_head_change_vs_native=2.)]
    assert propose_terminal(28, rows, programs, prior)['template_round'] == 7
    ast.parse((root/'scripts/resume_affinity_campaign.py').read_text())
