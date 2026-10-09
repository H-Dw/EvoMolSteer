"""Dataset selection, explicit split membership, and child configuration."""
from dataclasses import asdict
from pathlib import Path

import pytest

from evomolsteer.generation import steer_campaign
from evomolsteer.generation.dataset_profiles import load_dataset_profile
from evomolsteer.generation.target_catalog import FORMAT
from evomolsteer.io import read_json
from evomolsteer.storage.transactions import atomic_json
from test_target_campaign import config


def test_default_hiqbind_uses_explicit_test_manifest_only(tmp_path,monkeypatch,capsys):
    cfg=config(tmp_path,2)
    manifest=Path(cfg.input_dataset)/'targets.json'
    atomic_json(manifest,{'format':FORMAT,'targets':[
        {'target_id':'hiq_system_B','target_protein':'target_001/reference_pocket10.pdb',
         'target_ligand':'target_001/reference.sdf'}]})
    values=asdict(cfg);values.update(target_manifest=str(manifest),expected_targets=1)
    actual_loader=steer_campaign.load_dataset_profile
    monkeypatch.setattr(steer_campaign,'load_dataset_profile',lambda dataset:{**values,**{
        'samples':actual_loader(dataset)['samples'],'batch_size':actual_loader(dataset)['batch_size']}})
    steer_campaign.main(['--dry-run'])
    import json
    plan=json.loads(capsys.readouterr().out)
    assert [r['target_id'] for r in plan['targets']]==['hiq_system_B']
    assert plan['samples_per_target']==100 and plan['batch_size']==20 and plan['batches_per_target']==5
    assert plan['first_job_config']['samples']==100 and plan['first_job_config']['batch_size']==20
    assert plan['selection_window']==[0.,.5]
    assert not Path(cfg.output_dataset).exists()


def test_missing_hiqbind_inputs_do_not_start_or_fallback(tmp_path,monkeypatch,capsys):
    cfg=config(tmp_path,2);values=asdict(cfg)
    values['target_manifest']=str(tmp_path/'missing.json')
    calls=[]
    monkeypatch.setattr(steer_campaign,'load_dataset_profile',lambda dataset:(calls.append(dataset) or values))
    monkeypatch.setattr(steer_campaign,'run_campaign',lambda *a,**k:pytest.fail('Inference must not start'))
    with pytest.raises(SystemExit) as error:steer_campaign.main(['--dry-run'])
    assert error.value.code==2 and calls==['hiqbind']
    assert 'explicit target manifest' in capsys.readouterr().err
    assert not Path(cfg.output_dataset).exists()


def test_crossdocked_is_selected_only_when_explicit(tmp_path,monkeypatch,capsys):
    cfg=config(tmp_path,2);selected=[]
    monkeypatch.setattr(steer_campaign,'load_dataset_profile',lambda dataset:(selected.append(dataset) or asdict(cfg)))
    steer_campaign.main(['--dataset','crossdocked100','--dry-run'])
    assert selected==['crossdocked100']
    assert 'target_000/reference' in capsys.readouterr().out


def test_explicit_config_and_cli_overrides_replace_profile(tmp_path,monkeypatch,capsys):
    cfg=config(tmp_path,2);path=tmp_path/'custom.json';atomic_json(path,asdict(cfg))
    monkeypatch.setattr(steer_campaign,'load_dataset_profile',lambda *a:pytest.fail('Explicit config must bypass profiles'))
    steer_campaign.main(['--config',str(path),'--samples','40','--batch-size','20','--dry-run'])
    import json
    plan=json.loads(capsys.readouterr().out)
    assert plan['samples_per_target']==40 and plan['first_job_config']['samples']==40
    assert plan['batches_per_target']==2
    with pytest.raises(SystemExit):steer_campaign.main(['--config',str(path),'--dataset','hiqbind','--dry-run'])


def test_profiles_keep_distinct_output_roots_and_same_sampling_policy():
    hiq=load_dataset_profile();cross=load_dataset_profile('crossdocked100')
    assert hiq['output_dataset']!=cross['output_dataset']
    assert hiq['target_manifest'] and hiq['expected_targets']==297
    assert cross['expected_targets']==100
    for name in ('samples','batch_size','steps','window_start','window_end','archive_every','remove_archived_targets'):
        assert hiq[name]==cross[name]
