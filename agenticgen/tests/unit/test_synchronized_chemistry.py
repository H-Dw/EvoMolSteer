"""The chemical state projection shares the declared gradient phase, not its scale."""
import copy
import pytest
import torch

from agenticgen.uniguide.geometry_chemistry import policy,match_fractions
from agenticgen.uniguide.geometry_dynamics import policy as dynamics
from agenticgen.uniguide import guidance_endpoints,absolute_coordinate
from agenticgen.uniguide.native_tools import tensor_digest
from agenticgen.uniguide.tool_runtime import assets
from agenticgen.uniguide.io import read,value_sha
from agenticgen.uniguide.absolute_runtime import validate_config
from test_zero_start_guidance import config
from test_geometry_chemical_state import fixture
from test_absolute_coordinate_protocol import condition
from test_dynamic_matching_jump import dynamic_condition


def test_new_config_preserves_phase_and_old_release_keeps_legacy_phase(tmp_path):
    c=config(tmp_path/'new')
    assert c['agent_tools']['skill_release']=='2.10.1'
    assert c['chemical_inpainting']==dict(enabled=True,activation='guidance_phase')
    assert policy(c)['step_bounds']==guidance_endpoints.update_bounds(c)==[1,98]
    assert not policy(c)['activation_depends_on_scale']
    assert all('1–98' in p.read_text(encoding='utf8') for p in assets(c).values())
    old=dict(c,agent_tools=dict(c['agent_tools'],skill_release='2.10.0'),chemical_inpainting={'enabled':True})
    assert policy(validate_config(old))['step_bounds']==[10,99]
    assert policy({})['step_bounds']==[10,99]
    with pytest.raises(ValueError,match='synchronized|guidance phase'):
        validate_config(dict(c,chemical_inpainting={'enabled':True}))
    with pytest.raises(ValueError,match='2.10.1'):
        validate_config(dict(old,chemical_inpainting=c['chemical_inpainting']))
    with pytest.raises(ValueError):policy(dict(c,chemical_inpainting=dict(enabled=True,activation='unknown')))


def test_all_100_steps_project_only_1_through_98_without_coordinate_or_rng_change(tmp_path):
    c,state,_=fixture(tmp_path)
    c.sampling_policy=dynamics(config(tmp_path/'new'))
    c.schedule=[0.]*100  # Zero extra coordinate guidance keeps the declared chemistry phase.
    rng=torch.get_rng_state().clone();original=tensor_digest(state)
    for step in range(100):
        c.rows.append(dict(step=step))
        result=c.apply_chemical_inpainting(state,step)
        active=1<=step<=98
        assert c.rows[-1]['chemical_inpainting_applied']==[active]*2
        assert result['coords'] is state['coords'] and result['self_condition']==state['self_condition']
        assert result['mask'] is state['mask'] and tensor_digest(state)==original
        if active:
            assert all(x==[1.,1.] for x in match_fractions(result,torch.tensor([[3,2]]*2),c.chemical_features).values())
            for key in ('atomics','charges','hybridization'):
                assert torch.equal(result[key][:,[0,1,4]],state[key][:,[0,1,4]])
        else:
            assert result is state
    assert torch.equal(rng,torch.get_rng_state())


def test_early_dynamic_assignment_and_chemical_binding_share_phase(dynamic_condition,tmp_path):
    c,state,_=fixture(tmp_path/'legacy')
    settings=config(tmp_path/'new')
    c.folder=tmp_path/'synchronized';c.absolute_config=settings;c.geometry_condition=dynamic_condition
    c.scale_binding={'guidance_windows':settings['windows']};c.sampling_policy=dynamics(settings)
    c.resume=None;c.check_rows=lambda name,value,ids:value
    c.absolute_state=absolute_coordinate.AssignmentState(c)
    c.configure_chemistry()
    binding=read(c.folder/'CHEMICAL_CONDITION.json')
    assert binding['step_bounds']==[1,98] and binding['activation']=='guidance_phase'
    assert binding['chemical_condition_sha256']==value_sha({k:v for k,v in binding.items() if k!='chemical_condition_sha256'})
    points=torch.tensor([[[9.,9.,9.],[0.,0.,0.],[1.4,0.,0.],[8.,8.,8.],[7.,7.,7.]]]).repeat(2,1,1)
    rng=torch.get_rng_state().clone()
    for step,xyz in [(0,points),(1,points[:,[1,0,2,3,4]]),(98,points),(99,points[:,[1,0,2,3,4]])]:
        c.active_step=step;c.rows.append(dict(step=step))
        ids=c.absolute_state.evaluate(xyz,state['mask'].bool(),[0,1])[3]
        projected=c.apply_chemical_inpainting(state,step)
        if step in (1,98):
            assert all(x==[1.,1.] for x in match_fractions(projected,ids,c.chemical_features).values())
        else:assert projected is state
    assert c.absolute_state.rows[0]['first_step']==0
    assert torch.equal(rng,torch.get_rng_state())


def test_disabled_switch_retains_phase_and_new_binding_rejects_legacy_resume(tmp_path):
    c,state,_=fixture(tmp_path/'legacy')
    cfg=config(tmp_path/'new')
    off=dict(cfg,chemical_inpainting=dict(cfg['chemical_inpainting'],enabled=False))
    assert policy(off)['step_bounds']==[1,98] and not policy(off)['enabled']
    c.sampling_policy=dynamics(off)
    assert c.apply_chemical_inpainting(state,1) is state
    assert c.rows[-1]['chemical_inpainting_applied']==[False,False]
    c.sampling_policy=dynamics(cfg);c.folder=tmp_path/'sync';c.absolute_config=cfg;c.configure_chemistry()
    c.scale_binding={};c.resume=None
    bound=c.checkpoint_metadata()['geometry_binding']
    assert bound['chemical_binding']['step_bounds']==[1,98]
    c.resume=dict(geometry_binding=copy.deepcopy(bound))
    c.resume['geometry_binding']['chemical_binding']['step_bounds']=[10,99]
    with pytest.raises(ValueError,match='identical distance condition'):c.initialize()


def test_both_actual_role_prompts_contain_the_synchronized_contract(tmp_path,monkeypatch):
    import json
    import test_linear_guidance as helpers
    from agenticgen.uniguide import linear_calibration,linear_runtime,linear_council
    from agenticgen.uniguide.io import write
    original=helpers.row
    def six_row(*args,**kwargs):
        result=original(*args,**kwargs)
        result['window_diagnostics']=(result['window_diagnostics']*2)[:6]
        return result
    monkeypatch.setattr(helpers,'row',six_row)
    executor,_,_=helpers.install_fake(monkeypatch,tmp_path)
    c=config(tmp_path)
    subject=dict(teacher=dict(target='teacher'),condition=dict(condition_sha256='teachercondition'))
    d=linear_calibration.run(c,[subject],lambda *a:None,executor)['teacher']
    history,failures=linear_runtime.history(c,subject['teacher'],subject['condition'],d)
    skills=tmp_path/'role_skills';skills.mkdir()
    for name,path in assets(c).items():(skills/name).write_bytes(path.read_bytes())
    snapshots=[]
    def provider(messages,schema,out,**options):
        packet=json.loads(messages[1]['content']);snapshots.append(packet)
        chemical=packet['conditioning_protocol']['chemical_inpainting']
        assert chemical['step_bounds']==[1,98] and chemical['activation']=='guidance_phase'
        assert not chemical['activation_depends_on_scale']
        result=dict(decision='hold',reason='No new execution in this offline contract fixture.')
        result['proposals' if 'proposals' in schema['properties'] else 'selected']=[]
        options['response_validator'](result);write(out,result);return result
    before=(tmp_path/'attempts.json').read_bytes()
    linear_council.advise(c,history,d,tmp_path/'advice',skills,failures,provider=provider)
    assert len(snapshots)==2 and (tmp_path/'attempts.json').read_bytes()==before
