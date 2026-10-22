import copy
import pytest
from agenticgen.adaptive_schedule.group_protocol import configuration, input_limit, budget_context
from agenticgen.adaptive_schedule.group_run import evidence
from agenticgen.adaptive_schedule.io import defaults, read, write
from agenticgen.adaptive_schedule.wave_retention import prune_payload


def test_two_levels_of_budget_and_doubled_limits():
    c = configuration()
    assert c['rounds'] == 5 and c['versions'] == [f'v{i}' for i in range(1,21)]
    assert 200+2*100*4*c['rounds']*len(c['versions']) == c['total_result_budget'] == 80200
    assert input_limit(c) == 262144 and input_limit(defaults()) == 131072
    assert c['llm']['max_output_tokens'] == 32768 and c['council']['max_output_tokens'] == 65536
    assert budget_context(c,5)['decision_phase'] == 'review_only'
    assert budget_context(c,5)['remaining_sampling_rounds'] == 0


def test_group_lineage_and_incumbent_evidence_are_separate():
    template = {'teacher_id':'t'}
    trials = [{'id':'r00','record':{'pic50_on_rescore':1},'ticks':[5]*4} for _ in range(100)]
    old = evidence(template, 0, 'bootstrap', trials, trials)
    latest = [dict(t, id='r01', source_id='own', record={'pic50_on_rescore':2}) for t in trials]
    parent = [dict(t, source_id='retained_other_group', record={'pic50_on_rescore':3}) for t in trials]
    new = evidence(template,1,'G1',latest,parent,old)
    assert [t['id'] for t in new['chains'][0]['trials']] == ['r00','r01','parent']
    assert new['chains'][0]['incumbent_id'] != new['chains'][0]['latest_id']
    assert new['round_ids'] == ['r00','r01']


def test_structure_only_prune_is_idempotent_and_keeps_analysis(tmp_path):
    for name in ['terminal.pt.gz','trace.h5','molecules.sdf','complex.pdb','nodes.json','report.xml','prompt.txt']:
        (tmp_path/name).write_text('test')
    prune_payload(tmp_path); prune_payload(tmp_path)
    assert not (tmp_path/'terminal.pt.gz').exists()
    assert all((tmp_path/name).exists() for name in ['nodes.json','report.xml','prompt.txt'])
    assert read(tmp_path/'CLEANUP_COMPLETE.json')['removed_files'] == 4


def test_common_rng_replays_native_categorical_without_forcing_categories():
    torch = pytest.importorskip('torch')
    from agenticgen.adaptive_schedule.common_random import NativeStreams, capture, equal
    states = [capture() for _ in range(5)]
    before = capture(); stream = NativeStreams(states,4)
    p = torch.zeros(400,3,2); p[:100,:,0] = 1; p[100:,:,1] = 1
    result = stream.categorical(p)
    assert (result[:100] == 0).all() and (result[100:] == 1).all()
    assert equal(before,capture())
    eps = stream.normal('randn',(400,3,3))
    assert all(torch.equal(eps[:100],eps[i*100:(i+1)*100]) for i in range(1,4))
