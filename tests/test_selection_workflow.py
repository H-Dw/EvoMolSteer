import copy
import importlib.util
import sys
from pathlib import Path
import pytest
from evomolsteer.io import digest,read_json,write_json
from evomolsteer.generation.selection_workflow import restore_incumbent,activate_confirmed


def test_rollback_restores_bytes_skills_and_disabled_routing(tmp_path):
    base=tmp_path/'baseline.json';base.write_bytes(b'{"window":[0.05,0.45],"reference_sha256":"immutable"}\n')
    skill=tmp_path/'SKILL.md';skill.write_text('generic evidence skill')
    algorithm=tmp_path/'reward.py';algorithm.write_text('unchanged numerical algorithm')
    folder=tmp_path/'configs/experiments/trial';folder.mkdir(parents=True)
    write_json(folder/'baseline_snapshot.json',{'program_path':'baseline.json','program_sha256':digest(base),
      'baseline_skills':{'Analyst':{'path':'SKILL.md','sha256':digest(skill)}},'baseline_algorithm_files':{'reward.py':digest(algorithm)}})
    (folder/'active_program.json').write_text('failed candidate')
    restored=restore_incumbent(tmp_path,'trial','negative affinity',3)
    assert (folder/'active_program.json').read_bytes()==base.read_bytes()
    assert restored['candidate_enabled'] is False and restored['after_round']==3
    assert read_json(folder/'active_skills.json')['Analyst']['sha256']==digest(skill)
    algorithm.write_text('accidentally changed baseline')
    with pytest.raises(ValueError,match='Baseline algorithm changed'):
        restore_incumbent(tmp_path,'trial','test',4)


def driver_module():
    scripts=Path(__file__).resolve().parents[1]/'scripts'
    sys.path.insert(0,str(scripts))
    spec=importlib.util.spec_from_file_location('selection_driver_under_test',scripts/'run_selection_path_campaign.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_frozen_confirmation_rejects_reward_mutation(tmp_path):
    module=driver_module();root=Path(__file__).resolve().parents[1]
    d=object.__new__(module.Driver);d.root=tmp_path;d.cfg=tmp_path/'configs/experiments/selection_path20_v1'
    d.docs=tmp_path/'docs/experiments/selection_path20_20261008';d.cfg.mkdir(parents=True);d.docs.mkdir(parents=True)
    base=root/'configs/experiments/skill_ablation_v1/incumbent.json'
    target=tmp_path/'configs/experiments/skill_ablation_v1/incumbent.json';target.parent.mkdir(parents=True);target.write_bytes(base.read_bytes())
    source=d.cfg/'round02.json';source.write_bytes(base.read_bytes())
    write_json(d.docs/'frozen_validation.json',{'selected_round':2,'program_sha256':digest(source)})
    p=read_json(source);p['native_rms_ratio']*=2;write_json(source,p)
    with pytest.raises(ValueError,match='Frozen source mutated'):d.freeze(18)


def test_single_module_schedule_does_not_inherit_failed_parameters():
    trials=driver_module().TRIALS
    assert set(trials)==set(range(2,16))
    assert trials[2][0]=={'quality':'rank'}
    assert trials[11][0]=={'precision_mix':.1} and trials[11][1]=={}
    assert all(len(spec)==1 for spec,extra,reason in trials.values())


def test_reconnect_checks_round_without_dispatching_again(tmp_path):
    module=driver_module();d=object.__new__(module.Driver);d.cfg=tmp_path
    write_json(tmp_path/'campaign.json',{'rounds_completed':2})
    d.work='/protected/experiment';d.remote=lambda command:'{"round":4,"campaign":"selection_path_r04"}'
    with pytest.raises(ValueError,match='Active inference identity mismatch'):d.resume_existing(3)
    with pytest.raises(ValueError,match='next unretained round'):d.resume_existing(2)


def adoption_fixture(root):
    folder=root/'configs/experiments/trial';folder.mkdir(parents=True)
    reference=folder/'selection_reference.json.gz';reference.write_bytes(b'fixed teachers')
    base=root/'baseline.json';write_json(base,{'window':[.03,.47],'reference_sha256':digest(reference)})
    skill=root/'skills/selection-pressure-path/SKILL.md';skill.parent.mkdir(parents=True);skill.write_text('generic selection evidence')
    numerical=root/'reward.py';numerical.write_text('immutable R26')
    write_json(folder/'baseline_snapshot.json',{'program_path':'baseline.json','program_sha256':digest(base),
      'baseline_skills':{'Analyst':{'path':skill.relative_to(root).as_posix(),'sha256':digest(skill)}},
      'baseline_algorithm_files':{'reward.py':digest(numerical)}})
    source=folder/'round12.json';write_json(source,{'window':[.03,.47],'reward_view':'endpoint_selection_path',
      'reference_sha256':digest(reference),'selection_path':{'precision_mix':.25}})
    frozen={'selected_round':12,'program_sha256':digest(source),'reference_sha256':digest(reference),
      'screening_admissible':True,'before_confirmation_labels':True}
    return folder,source,frozen


def test_adoption_requires_confirmation_and_rolls_back_on_rejection(tmp_path):
    folder,source,frozen=adoption_fixture(tmp_path)
    with pytest.raises(ValueError,match='confirmation checks'):
        activate_confirmed(tmp_path,'trial',frozen,{'mean':True,'strain':False})
    assert read_json(folder/'active_workflow.json')['default']=='historical_R26'
    state=activate_confirmed(tmp_path,'trial',frozen,{'mean':True,'strain':True})
    assert state['candidate_enabled'] is True
    assert digest(folder/'active_program.json')==frozen['program_sha256']
    assert read_json(folder/'active_program.json')['window']==[.03,.47]
    assert 'experimental_module' in read_json(folder/'active_skills.json')


def test_adoption_rejects_mutated_frozen_proposal(tmp_path):
    folder,source,frozen=adoption_fixture(tmp_path)
    source.write_text('changed after seeing confirmation labels')
    with pytest.raises(ValueError,match='Frozen proposal changed'):
        activate_confirmed(tmp_path,'trial',frozen,{'mean':True})
    assert read_json(folder/'active_workflow.json')['candidate_enabled'] is False


def test_read_status_reconnect_does_not_repeat_generation():
    module=driver_module();d=object.__new__(module.Driver);d.work='/experiment'
    calls=[]
    def remote(command):
        calls.append(command)
        if len(calls)==1:raise module.paramiko.SSHException('transport closed')
        return '0'
    d.remote=remote;events=[];d.emit=lambda *args,**kwargs:events.append(kwargs)
    reconnects=[];d.connect=lambda:reconnects.append(True)
    assert d.read_inference_status(13)=='0'
    assert len(reconnects)==1 and calls[0]==calls[1]
    assert all('dispatch' not in command and 'git' not in command for command in calls)
    assert events[0]['no_generation_rerun'] is True
