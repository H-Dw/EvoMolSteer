import copy
import json
import pytest
import httpx
import test_early_path_search as old
from agenticgen.uniguide import stage_search as stage, linear_scale as scale, checkpoint_learning, linear_council, linear_runtime
from agenticgen.uniguide.geometry_protocol import choose
from agenticgen.uniguide.io import read,write,sha

PART=dict(early=[0,1],middle=[2,3],late=[4,5],reason="Ordered control stages.")

@pytest.fixture
def setup(tmp_path,monkeypatch):
    def config(root):
        from pathlib import Path
        folder=Path(__file__).resolve().parents[2]/'configs'
        c=read(folder/'uniguide_absolute_coordinate.json')
        c.update(read(folder/'uniguide_stage_path_search.json'))
        c.update(root=str(root),generation_context_root=str(root.parent/'frozen'))
        return old.validate_config(c)
    monkeypatch.setattr(old,'config',config)
    return old.setup(tmp_path,monkeypatch)

@pytest.mark.parametrize('n',[3,4,5,6,7,10])
def test_dynamic_partition(n):
    e=max(2,n//3) if n>3 else 1;m=max(1,(n-e)//2)
    part=dict(early=list(range(e)),middle=list(range(e,e+m)),late=list(range(e+m,n)),reason='Temporal partition')
    domain=dict(search_grids=[[0,10]]*n)
    assert stage.validate_partition(part,domain)==part
    for key in stage.STAGES:
        broken=copy.deepcopy(part);broken[key]=[]
        with pytest.raises(Exception):stage.validate_partition(broken,domain)
    broken=copy.deepcopy(part);broken['late']+=part['early']
    with pytest.raises(Exception):stage.validate_partition(broken,domain)


def test_admission_focus_directions_lookup(setup,tmp_path):
    from agenticgen.uniguide.analysis_tools import EvidenceTools
    from agenticgen.uniguide.linear_admission import audit_actions
    c,s,d,h,failures,executor,calls=setup
    learning=checkpoint_learning.context(c,h,[],d);best=choose(h)
    focus=dict(stage='early',operation='start',source_id=best['id'])
    actions=old.five(c,h,d)
    def audit(a,**kw):return audit_actions(c,h,d,a,portfolio=True,learning=learning,search_focus=focus,stage_partition=PART,**kw)
    assert audit(actions)['batch_admission']['executable_now']
    invalid=audit_actions(c,h,d,actions,portfolio=True,learning=learning,stage_partition=PART)
    assert not invalid['batch_admission']['executable_now']
    with pytest.raises(ValueError,match='partition'):stage.audit(actions,h,d,[],250,dict(learning,search_path_context=dict(learning['search_path_context'],stage_partition=PART)),focus,PART)
    registry=EvidenceTools(tmp_path,h,d,tmp_path/'lookup')
    registry.action_config=c;registry.action_learning=learning
    result=registry.execute('lookup_schedule',dict(target_scale_schedule=best[scale.EFFECTIVE]))
    assert result['required_new_attempts']==0 and result['source_id']==best['id']
    duplicate=copy.deepcopy(actions);duplicate[0]['target_scale_schedule']=best[scale.EFFECTIVE]
    report=audit(duplicate)
    assert not report['batch_admission']['executable_now']
    assert best['id'] in report['actions'][0]['reason']
    # Later rounds still need two EARLY directions, not just Middle alternatives.
    later=dict(learning,search_path_context=dict(learning['search_path_context'],stage_partition=PART,first_round=False,previous_focus=focus))
    with pytest.raises(ValueError,match='Early'):
        stage.audit(actions[2:],h,d,[],250,later,dict(focus,operation='refine'))
    with pytest.raises(ValueError,match='focus'):
        stage.audit(actions[:2],h,d,[],250,later,dict(focus,stage='late',operation='advance'))

@pytest.mark.parametrize('repair_success',[True,False])
@pytest.mark.parametrize('later',[True,False])
def test_duplicate_repair_single_role(setup,tmp_path,monkeypatch,repair_success,later):
    from agenticgen.uniguide.tool_runtime import assets
    from agenticgen.providers.deepseek_tools import call_with_tools,SessionStopped
    from agenticgen.providers.strict_tools import FINAL_TOOL
    c,s,d,h,_,_,_=setup;actions=old.five(c,h,d);seen=[];best=choose(h)
    if later:
        learning=checkpoint_learning.context(c,h,[],d)
        state=stage.context(h,[],d,best,partition=PART)
        state.update(first_round=False,previous_focus=dict(stage='early',operation='start',source_id=best['id']))
        learning['search_path_context']=state;c['guidance_checkpoint_context']=learning
        stage.freeze_partition(tmp_path,d,PART)
    monkeypatch.setenv('DEEPSEEK_API_KEY','unit-not-a-key')
    def handle(request):
        body=json.loads(request.content);seen.append(body)
        payload=dict(decision='propose',reason='Measured opportunities.',proposals=copy.deepcopy(actions),stage_partition=PART,search_focus=dict(stage='early',operation='start',source_id=best['id']))
        if later:
            payload.pop('stage_partition');payload['search_focus']['operation']='refine'
        if len(seen)==1 or not repair_success:payload['proposals'][0]['target_scale_schedule']=best[scale.EFFECTIVE]
        return httpx.Response(200,json=dict(model='deepseek-flash',usage=dict(completion_tokens=20),choices=[dict(finish_reason='tool_calls',message=dict(role='assistant',content=None,reasoning_content='opaque',tool_calls=[dict(id='final'+str(len(seen)),type='function',function=dict(name=FINAL_TOOL,arguments=json.dumps(payload)))]))]))
    def run():return linear_council.advise(c,h,d,tmp_path/'advice',next(iter(assets(c).values())).parent,provider=lambda *a,**kw:call_with_tools(*a,**kw,transport=httpx.MockTransport(handle)))
    if repair_success:
        result=run();assert result['actions']==actions and result['stage_partition']==PART
        assert not list((tmp_path/'advice').glob('synthesis*'))
    else:
        with pytest.raises(SessionStopped):run()
    assert len(seen)==2
    assert best['id'] in json.dumps(seen[1])
    prompt=read(tmp_path/'advice/proposal.prompt_source.json')
    combined=' '.join(m['content'] for m in prompt['messages'])
    assert 'B1/B2' not in combined and 'keep all six windows' not in combined
    assert 'not mechanism, ablation or falsification' not in combined
    assert ('stage_partition' in prompt['response_schema']['required']) is (not later)


def test_stage_replay_suffix_and_branch_credit():
    def row(name,profile,score):
        r=old.scored(name,[score]*50,2.);r[scale.EFFECTIVE]=profile;return r
    a=row('a',[10]*6,.5);b=row('b',[20,10,10,10,10,10],.6)
    action=dict(id='adapt',anchor_id='b',target_scale_schedule=[20,10,10,10,20,10],evidence_ids=['b'])
    bad=row('bad',action['target_scale_schedule'],.7);bad['selected_proposal']=action
    focus=dict(stage='early',operation='start',source_id='a')
    r=dict(iteration=1,stage_partition=PART,search_focus=focus,stage_actions=[action],groups=[bad],checkpoint_learning=dict(best_checkpoint_id='a'))
    domain=dict(search_grids=[list(range(0,401,10))]*6)
    state=stage.context([a,b,bad],[r],domain,a)
    family=next(f for f in state['prefix_families'] if f['best_source_id']=='b')
    assert family['credit']==0
    win_action=dict(id='win',anchor_id='a',target_scale_schedule=[10]*4+[30,10],evidence_ids=['a'])
    win=row('win',win_action['target_scale_schedule'],.4);win['selected_proposal']=win_action
    r2=dict(r,iteration=2,groups=[win],stage_actions=[win_action],search_focus=dict(focus,stage='late',operation='advance'))
    state=stage.context([a,b,bad,win],[r,r2],domain,win)
    assert state['pending_early_revisit']['source_id']=='win'
    family=next(f for f in state['prefix_families'] if f['best_source_id']=='b')
    assert family['reactivation_options']
    improved=row('improved',action['target_scale_schedule'],.55);improved['selected_proposal']=action
    state=stage.context([a,b,improved],[dict(r,groups=[improved])],domain,a)
    assert next(f for f in state['prefix_families'] if f['best_source_id']=='improved')['credit']==1


@pytest.mark.parametrize('fail_attempt',[False,True])
def test_five_round_execution_and_resume(setup,tmp_path,monkeypatch,fail_attempt):
    from agenticgen.uniguide import absolute_runtime
    c,s,d,h,_,executor,calls=setup
    if fail_attempt:
        executor,formal_calls=old.executor_fixture(tmp_path,monkeypatch,fail=lambda actual,phase:phase=='formal' and actual[0]==350)
        formal_calls.extend(calls);calls=formal_calls
    monkeypatch.setattr(absolute_runtime,'calibrated',lambda cfg:[s])
    monkeypatch.setattr(linear_runtime,'source_binding',lambda:{'fixture':'stages'})
    def advisor(cfg,history,domain,folder,skills,failures=(),final=False):
        learning=cfg['guidance_checkpoint_context'];state=learning['search_path_context']
        actions=[] if final else old.five(cfg,history,domain,int(folder.name[-3:]),learning)
        if not final and folder.name.endswith('005'):
            from test_diverse_box_search import action
            profile=list(choose(history)[scale.EFFECTIVE]);profile[-2:]=[0,0]
            actions[-1]=action(cfg,history,domain,profile,'final_suffix',learning)
        focus=dict(stage='early',operation='start' if state['first_round'] else 'refine',source_id=choose(history)['id'])
        write(folder/'proposal.json',dict(proposals=actions))
        return dict(decision='hold' if final else 'propose',actions=actions,search_focus=focus,stage_partition=PART,reason='Measured opportunity.')
    result=linear_runtime.run_version(c,'GD01',lambda *a:None,executor,advisor,lambda *a:None)
    assert len(calls)==35
    assert result['current']['complete'] is (not fail_attempt)
    assert read(tmp_path/'ATTEMPT_LIMITS.json')['per_teacher_stage_caps']==dict(calibration=500,formal=1250)
    assert all(r['stage_partition']==PART for r in result['current']['rounds'])
    teacher=next(iter(result['current']['teachers'].values()))
    assert teacher['search_path_context']['stage_partition']==PART
    assert teacher['search_path_context']['stage_partition_sha256']==stage.value_sha(PART)
    pending=teacher['search_path_context']['pending_early_revisit']
    assert pending['status']=='unfulfilled_at_round_limit' and not pending['additional_rounds_authorized']
    saved=read(tmp_path/'versions/GD01'/s['teacher']['target']/'GLOBAL_BEST.json')
    assert saved['search_path_context']['pending_early_revisit']==pending
    assert linear_runtime.run_version(c,'GD01',None,None,None,None)==result and len(calls)==35


def test_mandatory_revisit_and_exact_budget_exemptions(setup):
    c,s,d,h,_,_,_=setup;best=choose(h);actions=old.five(c,h,d)
    learning=checkpoint_learning.context(c,h,[],d)
    focus=dict(stage='early',operation='refine',source_id=best['id'])
    learning['search_path_context'].update(stage_partition=PART,first_round=False,previous_focus=focus,pending_early_revisit=dict(source_id=best['id']))
    result=stage.audit(actions,h,d,[],250,learning,focus)
    assert result['early_revisit_required'] and result['early_revisit_proposed']
    joint=copy.deepcopy(actions)
    for a in joint:a['target_scale_schedule'][-1]=400
    with pytest.raises(ValueError,match='Late background'):stage.audit(joint,h,d,[],250,learning,focus)
    result=stage.audit([],h,d,[],0,learning,focus)
    assert not result['early_revisit_required'] and result['exemption_reason']
    same=copy.deepcopy(actions[:2]);same[1]['target_scale_schedule']=list(same[0]['target_scale_schedule']);same[1]['target_scale_schedule'][0]-=10
    with pytest.raises(ValueError,match='directions'):stage.audit(same,h,d,[],250,learning,focus)


def test_paused_prefix_cannot_change_parent_to_bypass(setup):
    c,s,d,h,_,_,_=setup;best=choose(h);actions=old.five(c,h,d)
    learning=checkpoint_learning.context(c,h,[],d);focus=dict(stage='early',operation='refine',source_id=best['id'])
    side=next(row for row in h if row[scale.EFFECTIVE][:4]!=best[scale.EFFECTIVE][:4])
    family=dict(prefix=side[scale.EFFECTIVE][:4],is_incumbent_prefix=False,credit=0,reactivation_options=[])
    learning['search_path_context'].update(stage_partition=PART,first_round=False,previous_focus=focus,prefix_families=[family])
    a=copy.deepcopy(actions[-1]);a['target_scale_schedule']=side[scale.EFFECTIVE][:4]+[390,380]
    with pytest.raises(ValueError,match='Paused'):stage.audit(actions[:2]+[a],h,d,[],250,learning,focus)
    family['reactivation_options']=[dict(suffix=[390,380],evidence_ids=[side['id']])]
    a['evidence_ids']=[side['id']]
    assert stage.audit(actions[:2]+[a],h,d,[],250,learning,focus)['observed_early_directions']>=2
    other=copy.deepcopy(a);other['id']='renamed';other['target_scale_schedule'][-1]=370
    with pytest.raises(ValueError,match='At most one'):stage.audit(actions[:2]+[a,other],h,d,[],250,learning,focus)


def test_partition_freeze_and_invalid_lookup(setup,tmp_path):
    c,s,d,h,_,_,_=setup;best=choose(h)
    stage.freeze_partition(tmp_path,d,PART)
    state=stage.restore_partition(tmp_path,d,stage.context(h,[],d,best))
    assert state['stage_partition']==PART
    changed=dict(PART,early=[0,1,2],middle=[3])
    with pytest.raises(Exception):stage.freeze_partition(tmp_path,d,changed)
    invalid=dict(best,attempts=10)
    with pytest.raises(ValueError):stage.lookup(best[scale.EFFECTIVE],[invalid])
    assert stage.lookup(best[scale.EFFECTIVE],[],[dict(id='reserved',failure=dict(error='interrupted'),**{scale.EFFECTIVE:best[scale.EFFECTIVE]})])['status']=='reserved'


@pytest.mark.parametrize('n',[3,4,5,6,7,10])
def test_dynamic_stage_audit_and_focus_transitions(n):
    e=max(2,n//3) if n>3 else 1;m=max(1,(n-e)//2)
    part=dict(early=list(range(e)),middle=list(range(e,e+m)),late=list(range(e+m,n)),reason='Ordered temporal controls')
    d=dict(search_grids=[[0,10,20]]*n,control_settings=dict(maximum_total_tick_change=2*n))
    root=old.scored('root',[.4]*50,2.);root[scale.EFFECTIVE]=[10]*n
    h=[root];learning=dict(best_checkpoint_id='root',search_path_context=stage.context(h,[],d,root))
    focus=dict(stage='early',operation='start',source_id='root')
    def action(i,value):
        target=[10]*n;target[i]=value
        return dict(id=str(i)+str(value),anchor_id='root',target_scale_schedule=target,evidence_ids=['root'])
    actions=[action(0,0),action(1 if n>3 else 0,20)]
    report=stage.audit(actions,h,d,[],250,learning,focus,part)
    assert report['observed_early_directions']==2 and report['required_initial_single_window_probes']==2
    state=stage.context(h,[],d,root,partition=part)
    state.update(first_round=False,previous_focus=focus)
    learning['search_path_context']=state
    middle=dict(focus,stage='middle',operation='advance')
    with pytest.raises(ValueError,match='focus'):stage.audit(actions,h,d,[],250,learning,middle)
    assert stage.audit(actions+[action(part['middle'][0],20)],h,d,[],250,learning,middle)['focus_changed']
    state['previous_focus']=middle
    stage.validate_focus(dict(focus,operation='backtrack'),state)
    with pytest.raises(ValueError,match='transition'):stage.validate_focus(dict(focus,operation='advance'),state)
    # Boundary and ordering defects are rejected even if each stage is nonempty.
    missing=copy.deepcopy(part);missing['late'].pop()
    with pytest.raises(ValueError):stage.validate_partition(missing,d)
    reversed_part=copy.deepcopy(part);reversed_part['early'],reversed_part['late']=part['late'],part['early']
    with pytest.raises(ValueError):stage.validate_partition(reversed_part,d)


def test_too_few_windows_and_exhausted_legal_space():
    import itertools
    with pytest.raises(ValueError,match='at least three'):
        stage.validate_partition(dict(early=[0],middle=[1],late=[0],reason='Unavailable'),dict(search_grids=[[0,10]]*2))
    d=dict(search_grids=[[0,10]]*3,control_settings=dict(maximum_total_tick_change=3))
    h=[]
    for i,p in enumerate(itertools.product((0,10),repeat=3)):
        row=old.scored(str(i),[.4+i*.01]*50,2.);row[scale.EFFECTIVE]=list(p);h.append(row)
    part=dict(early=[0],middle=[1],late=[2],reason='Three controls')
    learning=dict(best_checkpoint_id='0',search_path_context=stage.context(h,[],d,h[0]))
    report=stage.audit([],h,d,[],250,learning,dict(stage='early',operation='start',source_id='0'),part)
    assert report['required_early_directions']==0 and report['exemption_reason']


def test_failed_side_branch_pause_and_same_round_adoption_reopens():
    def row(name,p,score):
        r=old.scored(name,[score]*50,2.);r[scale.EFFECTIVE]=p;return r
    root=row('root',[10]*6,.5);side=row('side',[20]+[10]*5,.6)
    fail=dict(id='failed',anchor_id='side',target_scale_schedule=[20]+[10]*3+[20,10],evidence_ids=['side'])
    action=dict(id='win',anchor_id='root',target_scale_schedule=[10]*4+[30,10],evidence_ids=['root'])
    win=row('win',action['target_scale_schedule'],.4);win['selected_proposal']=action
    r=dict(iteration=1,stage_partition=PART,search_focus=dict(stage='early',operation='start',source_id='root'),stage_actions=[fail,action],groups=[win],checkpoint_learning=dict(best_checkpoint_id='root'))
    domain=dict(search_grids=[list(range(0,401,10))]*6)
    state=stage.context([root,side,win],[r],domain,win)
    family=next(f for f in state['prefix_families'] if f['best_source_id']=='side')
    assert family['credit']==0 and family['reactivation_options'][0]['suffix']==[30,10]
    assert state['pending_early_revisit']['trigger_round']==1
    # Re-adopting the original suffix with a new improved prefix still triggers review.
    back=dict(id='back',anchor_id='win',target_scale_schedule=[30]+[10]*5,evidence_ids=['root'])
    result=row('back',back['target_scale_schedule'],.3);result['selected_proposal']=back
    r2=dict(r,iteration=2,stage_actions=[back],groups=[result],checkpoint_learning=dict(best_checkpoint_id='win'))
    state=stage.context([root,side,win,result],[r,r2],domain,result)
    assert state['pending_early_revisit']['suffix']==[10,10]
    r2['stage_partition_sha256']='tampered'
    with pytest.raises(ValueError,match='hash'):stage.context([root,side,win,result],[r,r2],domain,result)


def test_all_release_hashes_and_stage_runtime_text():
    from pathlib import Path
    root=Path(__file__).resolve().parents[2]/'src/agenticgen/resources/skill_releases/guidance'
    for manifest in root.glob('*/RELEASE.json'):
        for name,digest in read(manifest).get('files_sha256',{}).items():
            assert sha(manifest.parent/name)==digest,(manifest.parent.name,name)
    new=root/'2.15.0'
    for role,skill in [('analysis','history'),('proposal','proposal'),('computation','evidence-computation')]:
        entry=new/'skills'/('jump-guidance-'+skill)/'SKILL.md'
        runtime=new/'runtime/linear_absolute_coordinate'/(role+'.md')
        assert entry.read_bytes()==runtime.read_bytes()
        import re
        assert not re.search(r'\bB\d+\b|six windows|seven windows',entry.read_text(encoding='utf-8'))


def test_lookup_rejects_legacy_nonfinite_and_policy_mismatch(setup):
    from agenticgen.uniguide import condition_evaluation as fitted
    c,s,d,h,_,_,_=setup;best=choose(h)
    for key in [fitted.CONFIG,fitted.MEAN]:
        invalid=dict(best);invalid.pop(key)
        with pytest.raises(ValueError):stage.lookup(best[scale.EFFECTIVE],[invalid])
    invalid=dict(best);invalid[fitted.MEAN]=float('nan')
    with pytest.raises(ValueError):stage.lookup(best[scale.EFFECTIVE],[invalid])
    wrong=dict(d,path_search_policy='early-forward-backtracking-v1')
    with pytest.raises(ValueError,match='policy'):stage.validate_binding(c,wrong)
    stage.validate_binding(c,d)


def test_paired_suffix_evidence_reopens_without_global_improvement():
    def row(name,p,score):
        r=old.scored(name,[score]*50,2.);r[scale.EFFECTIVE]=p;return r
    root=row('root',[10]*6,.4);side=row('side',[20]+[10]*5,.6);other=row('other',[30]+[10]*5,.8)
    failed=dict(id='fail',anchor_id='side',target_scale_schedule=[20]+[10]*3+[20,10],evidence_ids=['side'])
    r=dict(iteration=1,stage_partition=PART,search_focus=dict(stage='early',operation='start',source_id='root'),stage_actions=[failed],groups=[],checkpoint_learning=dict(best_checkpoint_id='root'))
    d=dict(search_grids=[list(range(0,401,10))]*6)
    state=stage.context([root,side,other],[r],d,root)
    assert state['pending_early_revisit'] is None
    action=dict(id='improve',anchor_id='other',target_scale_schedule=[30]+[10]*3+[30,10],evidence_ids=['other'])
    improved=row('improve',action['target_scale_schedule'],.7);improved['selected_proposal']=action
    r2=dict(r,iteration=2,stage_actions=[action],groups=[improved])
    state=stage.context([root,side,other,improved],[r,r2],d,root)
    assert not state['adopted_suffixes'] and state['pending_early_revisit'] is None
    family=next(f for f in state['prefix_families'] if f['best_source_id']=='side')
    assert family['reactivation_options']==[dict(suffix=[30,10],evidence_ids=['other','improve'])]
