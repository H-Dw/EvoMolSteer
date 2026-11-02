"""Seven-window executable config, split terminal controls and bounded selection."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import httpx
import pytest
import torch
from test_dynamic_matching_jump import dynamic_condition
from test_absolute_coordinate_protocol import condition

from test_diverse_box_search import prepared,install_executor,action
from agenticgen.uniguide import box_search,box_gates,box_exclusion,linear_scale as scale
from agenticgen.uniguide import linear_calibration as calibration,linear_runtime as runtime
from agenticgen.uniguide import linear_council,profile_diversity
from agenticgen.uniguide.absolute_runtime import validate_config
from agenticgen.uniguide.guidance_windows import SEVEN,expand
from agenticgen.uniguide.geometry_sampling import scale_binding
from agenticgen.uniguide.io import read,write,sha
from agenticgen.uniguide.coordinate_statistics import RANKING


BASE=[50,100,40,40,30,50,50]


def config(root):
    folder=Path(__file__).resolve().parents[2]/'configs'
    c=read(folder/'uniguide_absolute_coordinate.json')
    c.update(read(folder/'uniguide_diverse_seven_box_search.json'))
    c.update(root=str(root),generation_context_root=str(root.parent/'frozen'))
    return validate_config(c)


def setup(root,monkeypatch):
    return prepared(root,monkeypatch,config_factory=config,baseline_actual=BASE)


def five(c,h,d,iteration=1,learning=None):
    profiles=[]
    for index in (0,1,5,6):
        p=list(BASE);p[index]=([0,1,2,5,10][iteration-1] if index==0 else p[index]+10*iteration)
        profiles.append(p)
    profiles.append([50,100,40,40,30,200+10*iteration,300+10*iteration])
    return [action(c,h,d,p,f'round{iteration}_direction{i}',learning) for i,p in enumerate(profiles)]


def test_seven_box_schedule_75_76_boundary_and_native_endpoints(tmp_path):
    c=config(tmp_path);actual=[1,20,30,40,50,250,350]
    result=expand(actual,scale_binding(actual,window_bounds=c['windows'],early_shape=box_search.early_shape(c)))
    assert c['windows']==[list(p) for p in SEVEN]
    assert result[0]==result[99]==0
    assert result[1:11]==[1]*10
    assert result[51:76]==[250]*25 and result[76:99]==[350]*23
    for index,disabled,retained in ((5,75,76),(6,76,75)):
        local=dict(c,guidance_box_mask=[i!=index for i in range(7)])
        assert not box_gates.step_enabled(local,disabled) and box_gates.step_enabled(local,retained)
        assert box_gates.apply([350]*10,disabled,local)==[0]*10
        assert box_gates.apply([350]*10,retained,local)==[350]*10


def test_seven_dimensions_are_required_and_release_six_cannot_be_mixed(tmp_path,monkeypatch):
    c,s,d,b,_,_=setup(tmp_path,monkeypatch)
    assert scale.action_schema([b['id']],d)['properties']['target_scale_schedule']['minItems']==7
    assert len(d['search_grids'])==7
    assert sum(len(g)-1 for g in d['search_grids'])==283
    assert scale.effective(scale.encode([1,100,40,40,30,400,0],d),d)[-2:]==[400,0]
    with pytest.raises(ValueError):scale.encode(BASE[:-1],d)
    bad=copy.deepcopy(c);bad['agent_tools']['skill_release']='2.12.0'
    with pytest.raises(ValueError,match='layout'):validate_config(bad)
    bad=copy.deepcopy(c);bad['box_search']['window_grids']=bad['box_search']['window_grids'][:-1]
    with pytest.raises(ValueError,match='grid'):validate_config(bad)


@pytest.mark.parametrize('native_box',[5,6])
def test_split_native_box_disables_actual_autograd_and_chemical_hook(tmp_path,dynamic_condition,native_box):
    from agenticgen.uniguide.geometry_sampling import GeometryControl
    from agenticgen.uniguide import absolute_coordinate as coordinates
    from agenticgen.uniguide.geometry_dynamics import policy
    c=config(tmp_path/'run');c['guidance_box_mask']=[i!=native_box for i in range(7)]
    actual=[0 if i==native_box else 50 for i in range(7)]
    calls=[]
    class Model:
        coord_scale=1.
        integrator=SimpleNamespace(noise_scheduler=lambda t:t.new_tensor(0.),coord_noise_level=.2)
        def __call__(self,current,*a,**kw):
            calls.append(torch.is_grad_enabled());return dict(coords=current['coords']*1.01+.01)
        def _get_predictions(self,out):return out,None
    u=object.__new__(GeometryControl);u.model=Model();u.absolute_config=c;u.geometry_condition=dynamic_condition
    u.scale_binding=scale_binding(actual,window_bounds=c['windows'],early_shape=box_search.early_shape(c))
    u.schedule=u.expand_schedule(actual);u.sampling_policy=policy(c)
    u.prefix_policy={'enabled':False};u.com=torch.zeros(10,3);u.rows=[];u.resume=None;u.folder=tmp_path
    u.last_assignment=None;u.random_streams=SimpleNamespace(states=[])
    u.absolute_state=coordinates.AssignmentState(u);u.store=lambda *a:None
    state=dict(coords=torch.tensor([[[0.,0.,0.],[1.4,0.,0.],[9.,9.,9.]]]).repeat(10,1,1),mask=torch.ones(10,3,dtype=torch.bool))
    for step in (0,50,51,75,76,98,99):
        u.forward(state,{},[torch.full((10,),step*.00999)]*3,None,None,None,step,.00999)
        active=box_gates.step_enabled(c,step)
        assert calls[-1]==active
        if not active:
            assert not torch.count_nonzero(u.delta)
            u.rows=[{}]
            assert u.apply_chemical_inpainting(state,step) is state
            assert u.rows[-1]['chemical_inpainting_applied']==[False]*10


@pytest.mark.parametrize('teacher',['6pyy_H7S_A_402','3w54_RNB_A_504'])
@pytest.mark.parametrize('harmful',[False,True])
def test_seven_box_preexperiment_selects_only_affordable_full_proofs(tmp_path,monkeypatch,teacher,harmful):
    c=config(tmp_path);backgrounds=box_exclusion.search_backgrounds(teacher,7)
    specs=box_exclusion.profiles(teacher,backgrounds)
    assert len(specs)==33 and all(len(s['actual'])==7 for s in specs.values())
    assert all(p[-1]==p[-2] for p in backgrounds.values())
    executor,calls,_=install_executor(tmp_path,monkeypatch,
        score=lambda a,m,n:.4+sum((.1 if harmful and i in (5,6) else -.02)*m[i] for i in range(7)))
    calibration.ensure_limits(c,1)
    def provider(messages,schema,out,**kw):
        packet=json.loads(messages[1]['content']);data=packet['evidence']
        assert packet['maximum_excluded']==1
        assert schema['properties']['excluded_box_ids']['items']['maximum']==6
        chosen=data['eligible_box_ids']
        answer=dict(excluded_box_ids=chosen,evidence_ids=[data['evidence_id']],reason='Use actual full proofs only.')
        if packet['phase']=='final':
            answer['confirmed_profile_id']=next(n for n,ids in data['confirmed_options'].items() if ids==chosen)
        kw['response_validator'](answer);write(out,answer);return answer
    selected,_=box_exclusion.Search(c,dict(teacher=dict(target=teacher),condition=dict(condition_sha256='fixture')),None,executor,provider).run()
    assert selected['consumed_attempts']==sum(r['delta'] for r in calls)==500
    assert len(selected['excluded_box_ids'])==(1 if harmful else 0)
    assert len(selected['guidance_box_mask'])==7
    limits=read(tmp_path/'ATTEMPT_LIMITS.json')
    assert limits['stage_caps']=={'calibration':500,'formal':1250}


def test_actual_strict_proposal_and_synthesis_accept_seven_scale_five_direction_payloads(tmp_path,monkeypatch):
    from agenticgen.uniguide.tool_runtime import assets
    from agenticgen.providers.deepseek_tools import call_with_tools
    from agenticgen.providers.strict_tools import FINAL_TOOL
    c,s,d,b,_,_=setup(tmp_path,monkeypatch);proposals=five(c,[b],d)
    monkeypatch.setenv('DEEPSEEK_API_KEY','unit-not-a-key')
    skills=tmp_path/'skills';skills.mkdir()
    for name,path in assets(c).items():(skills/name).write_bytes(path.read_bytes())
    def handle(request):
        body=json.loads(request.content);packet=json.loads(body['messages'][1]['content'])
        assert all(t['function']['strict'] for t in body['tools'])
        assert len(packet['window_evidence']['coverage'])==7
        assert packet['execution_contract']['windows'][-2:]==[dict(window_id=5,step_bounds=[51,75]),dict(window_id=6,step_bounds=[76,99])]
        answer=(dict(decision='propose',proposals=proposals,reason='Five complete new combinations.')
            if packet['final_submission_contract']['payload_field']=='proposals' else
            dict(decision='propose',selected=[p['id'] for p in proposals],reason='Execute every proposal.'))
        return httpx.Response(200,json=dict(model='deepseek-flash',usage=dict(completion_tokens=20),choices=[dict(finish_reason='tool_calls',
            message=dict(role='assistant',content=None,reasoning_content='opaque',tool_calls=[dict(id='final_fixture',type='function',function=dict(name=FINAL_TOOL,arguments=json.dumps(answer)))]))]))
    result=linear_council.advise(c,[b],d,tmp_path/'advice',skills,provider=lambda *a,**kw:call_with_tools(*a,**kw,transport=httpx.MockTransport(handle)))
    assert len(result['actions'])==5 and all(len(a['target_scale_schedule'])==7 for a in result['actions'])


def test_five_rounds_execute_25_seven_dimensional_groups(tmp_path,monkeypatch):
    from agenticgen.uniguide import absolute_runtime
    c,s,d,b,executor,calls=setup(tmp_path,monkeypatch)
    monkeypatch.setattr(absolute_runtime,'calibrated',lambda cfg:[s])
    monkeypatch.setattr(runtime,'source_binding',lambda:{'fixture':'seven-dimensional'})
    def advisor(cfg,h,domain,folder,skills,failures=(),final=False):
        proposals=[] if final else five(cfg,h,domain,int(folder.name[-3:]),cfg['guidance_checkpoint_context'])
        return dict(decision='hold' if final else 'propose',actions=proposals,reason='All complete directions.')
    result=runtime.run_version(c,'GD01',lambda *a:None,executor,advisor,lambda *a:None)['current']
    assert result['complete'] and len(result['rounds'])==5
    assert sum(r['delta'] for r in calls if r['stage']=='formal')==1250
    assert all(r['hypothesis_execution']['all_hypotheses_attempted'] for r in result['rounds'])
    assert all(len(r['actual'])==7 for r in calls)


def test_current_background_coverage_distinguishes_last_two_boxes(tmp_path,monkeypatch):
    from agenticgen.uniguide.window_evidence import context
    c,s,d,b,_,_=setup(tmp_path,monkeypatch)
    alternate=dict(b,id='other',**{scale.EFFECTIVE:[*BASE[:-1],80]})
    coverage=context(c,[b,alternate],d,b)['coverage']
    assert coverage[5]['source_ids_at_current_background']==['baseline']
    assert coverage[6]['observed_levels_at_current_background']==[50,80]
    suggestions=profile_diversity.bank([b],d,BASE)
    assert any(p['target_scale_schedule'][-1]!=p['target_scale_schedule'][-2] for p in suggestions)


def test_new_release_keeps_parent_bytes_and_binds_seven_windows():
    base=Path(__file__).resolve().parents[2]/'src/agenticgen/resources/skill_releases/guidance'
    manifest=read(base/'2.13.0/RELEASE.json')
    for name,digest in manifest['old_files_sha256'].items():assert sha(base/'2.12.0'/name)==digest
    for name,digest in manifest['files_sha256'].items():assert sha(base/'2.13.0'/name)==digest
    assert manifest['guidance_windows']==[list(p) for p in SEVEN]
