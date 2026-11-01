"""Offline controls: reproduce the blocked frontier, budget gates and rooted rollback."""
import copy
from pathlib import Path

import jsonschema
import pytest

from agenticgen.uniguide import linear_scale as scale, linear_calibration as calibration
from agenticgen.uniguide import linear_runtime as runtime, checkpoint_learning as learning
from agenticgen.uniguide import linear_admission as admission
from agenticgen.uniguide.absolute_coordinate import POLICY, RMSD
from agenticgen.uniguide.coordinate_statistics import RANKING
from agenticgen.uniguide.tool_runtime import configure, assets
from agenticgen.uniguide.io import write, read, sha, value_sha
from test_linear_guidance import install_fake, row, action


def config(root, **extra):
    values=dict(root=str(root),protocol=POLICY,rounds=5,llm={},agent_tools=dict(skill_release='2.9.0'))
    values.update(extra)
    return configure(values)


def prepared(root,monkeypatch,errors=None,rounds=5):
    c=config(root,rounds=rounds)
    executor,calls,binding=install_fake(monkeypatch,root,errors=errors or (lambda a,n:.3 if a==[90]*4 else .5))
    subject=dict(teacher=dict(target='teacher'),condition=dict(condition_sha256='teachercondition'))
    calibration.ensure_limits(c,1)
    write(root/'BUDGET_SCOPE.json',dict(stage='calibration',teacher_id='teacher'))
    base=executor(None,c,subject['teacher'],subject['condition'],[90]*4,root/'baseline','calibration',50)
    d=calibration.derive_domain([base],'teacher','teachercondition',value_sha(binding(c,subject['teacher'],subject['condition'])),
                               c['guidance_control'],c['guidance_exploration'])
    folder=root/'calibration/frozen'/d['calibration_key']
    write(folder/'domain.json',d)
    write(folder/'COMPLETE.json',dict(domain_sha256=d['domain_sha256'],
          measurement_receipts={base['id']:sha(root/base['id']/'COMPLETE.json')}))
    write(root/'calibration/domains.json',{'teacher':d})
    return c,subject,d,base,executor,calls


def proposal(c,history,d,anchor,updates=((0,1),),identifier='candidate',context=None):
    context=context or learning.context(c,history,[],domain=d)
    from agenticgen.uniguide.exploration import classify
    return dict(action(anchor,updates,identifier),exploration_tier=classify(
        [dict(window_id=i,step=s) for i,s in updates],d['exploration_policy']),
        optimization_target=dict(calibration_source_id=d['initial_source'],
            incumbent_source_id=context['best_checkpoint_id'],reviewed_path_sha256=context['best_path_sha256'],
            purpose='seek_potential_global_improvement',path_relation=(
                'redirect_from_reviewed_best' if context['rebase_from_best_checkpoint'] else 'frontier_jump')))


def test_new_default_open_grid_is_not_the_confirmation_whitelist(tmp_path,monkeypatch):
    c,_,d,base,_,_=prepared(tmp_path,monkeypatch)
    assert c['agent_tools']['skill_release']=='2.9.0' and scale.open_domain(c)
    assert c['llm']['final_response_mode']=='strict_tool_v1'
    assert runtime.range_allowance(c,'teacher')==dict(remaining_attempts=0,remaining_configurations=0)
    a=proposal(c,[base],d,base,((0,2),(1,2)))
    resolved=scale.resolve_action(a,{base['id']:base},d,[base])
    assert resolved[scale.EFFECTIVE]==[110,110,90,90] and resolved['task_type']=='optimization'
    assert resolved['unconfirmed_scalar_levels']==[110] and not resolved['stability_guarantee']
    report=admission.audit_actions(c,[base],d,[a])
    assert report['actions'][0]['admission']['executable_now']
    assert report['actions'][0]['admission']['required_new_attempts']==50
    assert report['actions'][0]['admission']['distance_from_global_best_ticks']['total']==4
    assert len(assets(c))==4


def test_new_run_overlay_loads_through_the_complete_absolute_interface(tmp_path):
    from agenticgen.uniguide.absolute_runtime import validate_config
    configs=Path(__file__).resolve().parents[2]/'configs'
    c=read(configs/'uniguide_absolute_coordinate.json')
    c.update(read(configs/'uniguide_open_domain.json'))
    c.update(root=str(tmp_path/'new'),generation_context_root=str(tmp_path/'frozen'))
    validated=validate_config(c)
    assert validated['agent_tools']['skill_release']=='2.9.0'
    assert validated['llm']['final_response_mode']=='strict_tool_v1'
    assert scale.open_domain(validated)
    assert validated['checkpoint_learning']['enabled']
    assert validated['task_scheduling']=='ug04-930db032-serial'
    assert validated['guidance_control']['search_maximum']==200


def test_open_calibration_recenters_on_full_winner_without_reset_or_more_than_350(tmp_path,monkeypatch):
    def errors(a,n):
        return .1+(a[0]-(50 if n<50 else 100))**2/10000
    executor,calls,_=install_fake(monkeypatch,tmp_path,errors=errors)
    c=config(tmp_path)
    subjects=[dict(teacher=dict(target='teacher'),condition=dict(condition_sha256='teachercondition'))]
    d=calibration.run(c,subjects,lambda *a:None,executor)['teacher']
    states=read(tmp_path/'calibration/frozen'/d['calibration_key']/'SEARCH_STATE.json')
    assert states['policy']==scale.OPEN_POLICY
    assert d[scale.REFERENCE]!=50
    assert any(x.get('prior_full_winner') not in (None,50) for x in states['decisions'])
    assert 0<calibration.used(tmp_path,'teacher')<=350
    assert len([x for x in calls if x[2]==50])<=6
    assert len({(tuple(x[1]),x[2]) for x in calls})==len(calls)
    before=len(calls)
    assert calibration.run(c,subjects,lambda *a:pytest.fail('reloaded'),executor)['teacher']==d
    assert len(calls)==before
    assert d['reference_support']['finite_budget_cannot_guarantee_interior_optimum']


def test_independent_alternatives_do_not_imply_the_portfolio_is_affordable(tmp_path,monkeypatch):
    c,_,d,base,_,_=prepared(tmp_path,monkeypatch)
    limits=read(tmp_path/'ATTEMPT_LIMITS.json');limits['stage_caps']['formal']=50
    write(tmp_path/'ATTEMPT_LIMITS.json',limits)
    actions=[proposal(c,[base],d,base,((i,1),),f'candidate{i}') for i in range(3)]
    before=(tmp_path/'attempts.json').read_bytes()
    alternatives=admission.audit_actions(c,[base],d,actions)
    assert all(r['admission']['executable_now'] for r in alternatives['actions'])
    admission.require_executable(alternatives,portfolio=False)
    portfolio=admission.audit_actions(c,[base],d,actions,portfolio=True)
    with pytest.raises(ValueError,match='not executable'):
        admission.require_executable(portfolio,portfolio=True)
    assert (tmp_path/'attempts.json').read_bytes()==before


def test_root_and_actual_parent_path_survive_worse_improving_rounds(tmp_path,monkeypatch):
    c,_,d,base,_,_=prepared(tmp_path,monkeypatch)
    def child(name,score,parent):
        return dict(base,id=name,**{RANKING:score,RMSD:score},proposal=dict(anchor_id=parent['id'],updates=[dict(window_id=0,step=1)],
             improvement_reason='A different complete path could improve the best.',uncertainty='Unmeasured.'))
    a=child('a',.5,base);b=child('b',.4,base)
    rounds=[dict(iteration=i,groups=[r],checkpoint=learning.checkpoint([r],base,baseline=base),
                 decision_summary=dict(reason='Saved visible decision summary.')) for i,r in enumerate([a,b],1)]
    ctx=learning.context(c,[base,a,b],rounds,domain=d)
    assert ctx['best_checkpoint_id']==base['id'] and ctx['rebase_from_best_checkpoint']
    assert ctx['calibration_baseline']['performance']==pytest.approx(.3)
    assert ctx['best_decision_path'][0]['parent_source_id'] is None
    assert len(ctx['unsuccessful_recent_paths'])==2
    x=child('x',.2,base);y=child('y',.1,x)
    z=child('z',.15,y);w=child('w',.17,y)
    recent=[dict(iteration=i,groups=[r],checkpoint=learning.checkpoint([r],y,baseline=base)) for i,r in enumerate([z,w],3)]
    ctx=learning.context(c,[base,x,y,z,w],recent,domain=d)
    assert ctx['best_checkpoint_id']=='y' and ctx['rebase_from_best_checkpoint']
    assert [n['source_id'] for n in ctx['best_decision_path']]==[base['id'],'x','y']
    assert ctx['best_decision_path'][1]['saved_proposal_summary']['anchor_id']==base['id']
    assert learning.checkpoint([dict(a,**{RANKING:base[RANKING],RMSD:base[RMSD]})],base,baseline=base)['best_compatible_source_id']==base['id']
    x['proposal']['anchor_id']='missing'
    with pytest.raises(ValueError,match='parent is absent'):
        learning.context(c,[base,x,y],[],domain=d)


def test_admission_does_not_hide_corrupt_partial_measurements_or_retry_failures(tmp_path,monkeypatch):
    from agenticgen.uniguide.analysis_tools import EvidenceIntegrityError
    c,s,d,base,executor,_=prepared(tmp_path,monkeypatch)
    a=proposal(c,[base],d,base)
    resolved=scale.resolve_action(a,{base['id']:base},d,[base])
    failed=[dict(resolved,failure=dict(error='original numerical failure'))]
    assert admission.audit_actions(c,[base],d,[a],failed)['actions'][0]['status']=='rejected'
    executor(None,c,s['teacher'],s['condition'],resolved[scale.EFFECTIVE],tmp_path/'partial','calibration',20)
    before=(tmp_path/'attempts.json').read_bytes()
    write(tmp_path/'partial/records.json',[])
    with pytest.raises(EvidenceIntegrityError,match='source changed'):
        admission.audit_actions(c,[base],d,[a])
    assert (tmp_path/'attempts.json').read_bytes()==before


def test_purpose_incumbent_and_review_binding_cannot_be_omitted_or_invented(tmp_path,monkeypatch):
    c,_,d,base,_,_=prepared(tmp_path,monkeypatch)
    a=proposal(c,[base],d,base)
    for field,value in [('purpose','explain_window_contribution'),('incumbent_source_id','invented'),('reviewed_path_sha256','0'*64)]:
        bad=copy.deepcopy(a);bad['optimization_target'][field]=value
        with pytest.raises(ValueError,match='not executable'):
            admission.require_executable(admission.audit_actions(c,[base],d,[bad]),portfolio=False)
    bad=copy.deepcopy(a);bad.pop('optimization_target')
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad,scale.action_schema({base['id']:base},d))


def test_actual_council_prompt_contains_calibration_performance_and_path_before_round_one(tmp_path,monkeypatch):
    import json
    from agenticgen.uniguide import linear_council
    c,_,d,base,_,_=prepared(tmp_path,monkeypatch)
    skills=tmp_path/'skills';skills.mkdir()
    for name,path in assets(c).items():(skills/name).write_bytes(path.read_bytes())
    a=proposal(c,[base],d,base)
    snapshots=[]
    def provider(messages,schema,out,**opts):
        packet=json.loads(messages[1]['content']);snapshots.append(packet)
        assert schema['properties']['decision']['enum']==['propose','hold']
        assert packet['calibration_baseline']['source_id']==base['id']
        assert packet['calibration_baseline']['performance']==pytest.approx(.3)
        assert packet['global_best_performance']==pytest.approx(.3)
        assert packet['checkpoint_learning']['calibration_selection']['origin'].startswith('deterministic')
        names={x['function']['name'] for x in opts['registry'].definitions}
        assert names=={'summarize_population','compare_paired_runs','audit_action_batch'}
        if 'proposals' in schema['properties']:
            answer=dict(decision='propose',proposals=[a],reason='A new region could improve the calibration-rooted best.')
        else:
            assert packet['candidate_admission']['actions'][0]['admission']['executable_now']
            answer=dict(decision='propose',selected=[a['id']],reason='Select the executable opportunity.')
        opts['response_validator'](answer);write(out,answer);return answer
    before=(tmp_path/'attempts.json').read_bytes()
    decision=linear_council.advise(c,[base],d,tmp_path/'advice',skills,provider=provider)
    assert len(snapshots)==2 and decision['selected']==[a['id']]
    assert (tmp_path/'attempts.json').read_bytes()==before


def test_runtime_keeps_calibration_representative_when_all_rounds_are_worse(tmp_path,monkeypatch):
    from agenticgen.uniguide import absolute_runtime
    c,s,d,base,executor,calls=prepared(tmp_path,monkeypatch,rounds=3)
    monkeypatch.setattr(absolute_runtime,'calibrated',lambda config:[s])
    monkeypatch.setattr(runtime,'source_binding',lambda:{'control':value_sha(c['guidance_control'])})
    contexts=[]
    def advisor(cfg,h,domain,folder,skills,failures=(),final=False):
        for role in ('proposal','synthesis'):write(folder/(role+'.json'),dict(reason='Offline fixture.'))
        ctx=cfg['guidance_checkpoint_context'];contexts.append(ctx)
        actions=[] if final else [proposal(cfg,h,domain,base,((len(contexts)-1,1),),context=ctx)]
        return dict(decision='hold' if final else 'propose',actions=actions,reason='A potentially better complete path.')
    result=runtime.run_version(c,'GD01',lambda *a:None,executor,advisor,lambda *a:None)['current']
    assert result['teachers']['teacher']['representative']==base['id']
    assert not result['teachers']['teacher']['agent_improved_calibration']
    assert result['teachers']['teacher']['global_best_includes_calibration']
    assert result['rounds'][2]['checkpoint_learning']['rebase_from_best_checkpoint']
    assert read(tmp_path/'versions/GD01/teacher/CALIBRATION_BASELINE.json')['source_id']==base['id']
    assert read(tmp_path/'versions/GD01/teacher/GLOBAL_BEST.json')['representative']==base['id']
    assert len(calls)==4 and all(ctx['calibration_baseline']['performance']==pytest.approx(.3) for ctx in contexts)


def test_runtime_preflight_rejects_unaffordable_portfolio_before_any_sampling(tmp_path,monkeypatch):
    from agenticgen.uniguide import absolute_runtime
    c,s,d,base,executor,calls=prepared(tmp_path,monkeypatch)
    # ensure_limits is frozen: simulate an already charged 700/750 formal ledger instead.
    existing=read(tmp_path/'attempts.json')
    existing['already_measured']=dict(stage='formal',teacher_id='teacher',count=700)
    write(tmp_path/'attempts.json',existing)
    monkeypatch.setattr(absolute_runtime,'calibrated',lambda config:[s])
    monkeypatch.setattr(runtime,'source_binding',lambda:{'control':value_sha(c['guidance_control'])})
    def advisor(cfg,h,domain,folder,skills,failures=(),final=False):
        return dict(decision='propose',actions=[proposal(cfg,h,domain,base,((i,1),),str(i),cfg['guidance_checkpoint_context']) for i in (0,1)])
    before=(tmp_path/'attempts.json').read_bytes()
    with pytest.raises(ValueError,match='not executable'):
        runtime.run_version(c,'GD01',lambda *a:None,executor,advisor,lambda *a:None)
    assert len(calls)==1 and (tmp_path/'attempts.json').read_bytes()==before


def test_native_strict_rollback_prompt_carries_actual_best_and_parent_decisions(tmp_path,monkeypatch):
    import json
    import httpx
    from agenticgen.uniguide import linear_council
    from agenticgen.providers.deepseek_tools import call_with_tools
    from agenticgen.providers.strict_tools import FINAL_TOOL
    from test_strict_final_submission import wire, call
    c,s,d,base,executor,_=prepared(tmp_path,monkeypatch,
        errors=lambda a,n:.3 if a==[90]*4 else .2 if a==[100,90,90,90] else .1 if a==[100,100,90,90] else .4)
    write(tmp_path/'BUDGET_SCOPE.json',dict(stage='formal',teacher_id='teacher'))
    h=[base];completed=[]
    for i,updates in enumerate([((0,1),),((1,1),)],1):
        anchor=h[-1];ctx=learning.context(c,h,completed,domain=d)
        a=proposal(c,h,d,anchor,updates,str(i),ctx)
        resolved=scale.resolve_action(a,{r['id']:r for r in h},d,h)
        folder=tmp_path/f'child{i}';write(folder/'proposal.json',resolved)
        measured=executor(None,c,s['teacher'],s['condition'],resolved[scale.EFFECTIVE],folder,'formal',50)
        measured=dict(measured,proposal=resolved,selected_proposal=resolved)
        completed.append(dict(iteration=i,groups=[measured],checkpoint=learning.checkpoint([measured],anchor,baseline=base),
                              decision_summary=dict(reason='Saved choice of a potentially better joint path.')))
        h.append(measured)
    best=h[-1]
    for i in (3,4):
        bad=dict(best,id=f'bad{i}',**{RANKING:.4,RMSD:.4},selected_proposal=best['proposal'])
        completed.append(dict(iteration=i,groups=[bad],checkpoint=learning.checkpoint([bad],best,baseline=base)))
    ctx=learning.context(c,h,completed,domain=d)
    assert ctx['rebase_from_best_checkpoint']
    c['guidance_checkpoint_context']=ctx
    a=proposal(c,h,d,best,((2,3),),'redirect',ctx)
    a['evidence_ids']=[base['id'],h[1]['id'],best['id']]
    skills=tmp_path/'skills';skills.mkdir()
    for name,path in assets(c).items():(skills/name).write_bytes(path.read_bytes())
    seen=[]
    def handle(request):
        body=json.loads(request.content);seen.append(body)
        packet=json.loads(body['messages'][1]['content'])
        assert packet['calibration_baseline']['performance']==pytest.approx(.3)
        assert packet['global_best_performance']==pytest.approx(.1)
        assert [n['source_id'] for n in packet['checkpoint_learning']['best_decision_path']]==[r['id'] for r in h]
        assert packet['allowed_anchor_ids']==[best['id']]
        assert all(t['function']['strict'] for t in body['tools'])
        final=next(t['function']['parameters'] for t in body['tools'] if t['function']['name']==FINAL_TOOL)
        if 'proposals' in final['properties']:
            nested=final['properties']['proposals']['items']['properties']['optimization_target']
            assert set(nested['required'])==set(a['optimization_target'])
            answer=dict(decision='propose',proposals=[a],reason='Redirect after the measured best path review.')
        else:
            answer=dict(decision='propose',selected=['redirect'],reason='One executable potential improvement.')
        return httpx.Response(200,json=wire([call(args=answer)]))
    monkeypatch.setenv('DEEPSEEK_API_KEY','fixture-not-a-key')
    before=(tmp_path/'attempts.json').read_bytes()
    result=linear_council.advise(c,h,d,tmp_path/'advice',skills,
        provider=lambda *a,**kw:call_with_tools(*a,**kw,transport=httpx.MockTransport(handle)))
    assert result['selected']==['redirect'] and len(seen)==2
    assert (tmp_path/'attempts.json').read_bytes()==before
