import copy
import importlib.util
import sys
from pathlib import Path
import pytest
from evomolsteer.io import digest,read_json,write_json
from evomolsteer.generation.selection_workflow import restore_incumbent


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
