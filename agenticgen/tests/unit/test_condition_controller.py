import copy
import json
import pytest
from agenticgen.adaptive_schedule.learning import (expand_template, belief_view, validate_memory,
    validate_controller_hypotheses, verified_large_steps)
from agenticgen.adaptive_schedule.objective import condition_summary, dominates
from agenticgen.adaptive_schedule.group_protocol import configuration
from agenticgen.adaptive_schedule.group_tools import PopulationTools
from agenticgen.adaptive_schedule.group_agent import plan_population
from test_plip_group_agent import population, hypotheses


def data25():
    d=population(); d['chains']=d['chains'][:25]
    t=d['teacher'];t.update(context_atoms=[0],bond_labels=[[0]],atom_labels=[1],charge_labels=[0])
    d['teacher']=expand_template(t)
    d.update(version='v6',controller_protocol='conditional_learning_v1',group_id='G1')
    for i,c in enumerate(d['chains']):
        c.update(candidate_id=f'b{i//5:03d}:s{i%5:03d}',batch=i//5,slot=i%5)
    return d


def test_one_hop_not_recursive_and_original_template_unchanged():
    t={'coords_A':[[0,0,0]]*4,'core_atoms':[0],'context_atoms':[0,1],
       'bond_labels':[[0,1,0,0],[1,0,1,0],[0,1,0,1],[0,0,1,0]]}
    original=copy.deepcopy(t); x=expand_template(t)
    assert x['core_atoms']==[0,1] and x['original_core_atoms']==[0]
    assert t==original
    with pytest.raises(ValueError):expand_template(x)


def test_fidelity_and_functional_conflict_not_affinity_only_success():
    d=data25(); trials=[copy.deepcopy(c['trials'][-1]) for c in d['chains']]
    baseline=condition_summary(trials,d['teacher'])
    for t in trials:
        t['nodes']['100']['local_endpoint_A']=[[.1,0,0]]
        t['record']['pic50_on_rescore']-=1
    altered=condition_summary(trials,d['teacher'])
    assert altered['condition_rmsd_A_all_attempts']['mean']<baseline['condition_rmsd_A_all_attempts']['mean']
    assert altered['mean_affinity_absolute_gap']>baseline['mean_affinity_absolute_gap']
    assert not dominates(altered,baseline) and not dominates(baseline,altered)
    assert altered['thresholds'] is None and altered['top10_percent_count']==3


def test_invalid_molecules_do_not_disappear_from_condition_distribution():
    d=data25(); trials=[copy.deepcopy(c['trials'][-1]) for c in d['chains']]
    trials[0]['record']['build_success']=False
    result=condition_summary(trials,d['teacher'])
    assert result['attempts']==25 and result['affinity']['valid']==24
    assert result['condition_rmsd_A_all_attempts']['n']==25
    assert result['condition_rmsd_A_valid_only']['n']==24


def test_memory_refutation_supersedes_and_scopes_are_enforced():
    m={'teacher_id':'t','version':'v6','published_after_round':1,
       'learning_update':[{'belief_id':'edge1','status':'supported','revision':'tentative'}]}
    newer=copy.deepcopy(m);newer['published_after_round']=2
    newer['learning_update'][0].update(status='refuted',revision='reverse response')
    assert belief_view([m,newer])[0]['status']=='refuted'
    validate_memory([m,newer],'t','v6',3)
    for teacher,version,round_no in [('other','v6',3),('t','v7',3),('t','v6',2)]:
        with pytest.raises(ValueError):validate_memory([m,newer],teacher,version,round_no)


def test_triple_tick_requires_isolated_condition_and_function_support():
    history=[{'ticks':[3,5,5,5]}, {'ticks':[5,5,5,5],'attempts':25,'paired_common_valid':25,
        'condition_error_delta':-.2,'affinity_gap_delta':-.1},
        {'ticks':[7,5,5,5],'attempts':25,'paired_common_valid':25,'condition_error_delta':-.1,'affinity_gap_delta':0}]
    assert verified_large_steps(history,[7,5,5,5])=={0:1}
    history[-1]['affinity_gap_delta']=.1
    assert verified_large_steps(history,[7,5,5,5])=={}
    history[-1].update(affinity_gap_delta=-.1,ticks=[7,6,5,5])
    assert verified_large_steps(history,[7,6,5,5])=={}


def test_new_objective_and_counts_reach_actual_prompt_and_registered_tools(tmp_path):
    d=data25();seen=[]
    def caller(messages,schema,output,**kwargs):
        payload=json.loads(messages[1]['content']);seen.append(payload)
        assert payload['candidates']==25 and payload['objective_priority'][0]=='condition_coordinate_fidelity'
        assert payload['continuous_objectives']['thresholds'] is None
        e=payload['evidence'][0]['evidence_id'];h=hypotheses(2,e)
        for item in h['hypotheses']:
            item.update(reference_ticks=[5]*4,experiment_purpose='direction_probe',search_mode='local',scale_reason='initial moderate probe',
                expected_condition_effect='test measured error direction',expected_functional_effect='test teacher gap',
                diagnostic_alternative='weak delivered control',expected_subgroup_response='unknown',revision_from_previous_result='unresolved')
        h['hypotheses'][0]['ticks']=[3,5,5,5];h['hypotheses'][1]['ticks']=[7,5,5,5]
        h['learning_update']=[dict(belief_id='edge',prior_judgment='unknown',intervention='none yet',
            observed_result='geometry and score differ',revision='still unknown',next_action='opposing probes',
            applicability='this background only',alternative_explanation='matching',status='unresolved',evidence_ids=[e])]
        kwargs['response_validator'](h);return h
    result=plan_population(d,tmp_path,configuration(),hypotheses=2,caller=caller)
    assert result['candidate_attempts_if_executed']==50
    assert all(j['earliest_common_restart']==10 and len(j['per_candidate_restart_steps'])==25 for j in result['jobs'])
    assert (tmp_path/'learning_update.json').exists()
    tools=PopulationTools(d,tmp_path/'check')
    edge=tools.calculate('response_edge',{'a':'r00','b':'latest'})
    assert len(edge['identities'])==25 and edge['paired']['delta']['mean']==pytest.approx(.1)
    total=sum(r['contribution_to_common_valid_mean'] for r in edge['identities'])
    assert total==pytest.approx(edge['paired']['delta']['mean'])
    assert sum(r['contribution_to_condition_mean_delta'] for r in edge['identities'])==pytest.approx(
        edge['B_condition_function']['condition_rmsd_A_all_attempts']['mean']-
        edge['A_condition_function']['condition_rmsd_A_all_attempts']['mean'])


def test_executed_array_rejected_without_executor_inventing_alternative():
    msg=hypotheses(2,'e')
    for h in msg['hypotheses']:h['reference_ticks']=[5]*4
    with pytest.raises(ValueError,match='already executed'):
        validate_controller_hypotheses(msg,[5]*4,2,{'e'},{},[{'ticks':msg['hypotheses'][0]['ticks']}])


def test_synthesis_cannot_repeat_current_sibling_schedule():
    from agenticgen.adaptive_schedule.council import validate_synthesis
    proposals={str(i):{'ticks':[i,5,5,5]} for i in range(1,5)}
    result={'ranked_proposal_ids':list(proposals),'selected':[{'proposal_id':i} for i in proposals]}
    validate_synthesis(result,proposals)
    with pytest.raises(ValueError,match='already executed'):
        validate_synthesis(result,proposals,[[3,5,5,5]])


def test_condition_report_preserves_small_population_and_selection_distinction(tmp_path):
    from agenticgen.adaptive_schedule.io import write
    from agenticgen.condition_reporting import version_report
    d=data25();teacher=d['teacher'];identity=teacher['teacher_id']
    reference=[copy.deepcopy(c['trials'][0]) for c in d['chains']]
    trials=[]
    for g in range(4):
        for item in reference:
            t=copy.deepcopy(item);t['ticks']=[3+g,5,5,5]
            t['nodes']['100']['local_endpoint_A']=[[.1*(g+1),0,0]]
            t['record']['pic50_on_rescore']+=.1*g
            trials.append(t)
    write(tmp_path/'config.json',configuration());write(tmp_path/'teachers.json',[teacher])
    write(tmp_path/'shared_r0'/identity/'best.json',reference)
    wave=tmp_path/'versions/v6'/identity/'round_01/wave'
    write(wave/'trials.json',trials);write(wave/'QUALITY_COMPLETE.json',{'attempts':100})
    write(wave/'SAMPLING_COMPLETE.json',{'seconds':1.2})
    write(tmp_path/'versions/v6'/identity/'incumbents/r01/best.json',trials[-25:])
    report=version_report(tmp_path,'v6')
    assert report['counts']['optimization_attempts']==100 and len(report['rows'])==6
    assert all(r['attempts']==25 and r['top10_percent_count']==3 for r in report['rows'])
    assert not report['rows'][-1]['shared_schedule']
    assert report['progress'][0]['candidate_attempts']==100
    assert all('threshold' not in k for k in report['progress'][0])


def test_small_common_stream_preserves_native_twenty_row_draws():
    torch=pytest.importorskip('torch')
    from agenticgen.adaptive_schedule.common_random import NativeStreams,capture,equal
    states=[capture() for _ in range(5)]
    small=NativeStreams(states,4,logical_rows=5);native=NativeStreams(states,4)
    before=capture()
    a=small.normal('randn',(100,3,3));b=native.normal('randn',(400,3,3))
    ids=[g*100+k*20+i for g in range(4) for k in range(5) for i in range(5)]
    assert torch.equal(a,b[ids]) and all(equal(x,y) for x,y in zip(small.states,native.states))
    p=torch.full((100,3,4),.25);p20=p.reshape(4,5,5,3,4).repeat(1,1,4,1,1).reshape(400,3,4)
    a=small.categorical(p);b=native.categorical(p20)
    assert torch.equal(a,b[ids]) and all(equal(x,y) for x,y in zip(small.states,native.states))
    assert equal(before,capture())
