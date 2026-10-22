import contextlib
import pytest
from agenticgen.adaptive_schedule import protocol_state as p
from agenticgen.adaptive_schedule.io import write,read,sha


def setup(tmp_path,monkeypatch):
    from agenticgen.runtime import locks
    from agenticgen.adaptive_schedule import group_run,wave_retention,io
    monkeypatch.setattr(locks,'lock',lambda root:contextlib.nullcontext())
    monkeypatch.setattr(group_run,'SKILLS',{})
    monkeypatch.setattr(wave_retention,'verify_snapshot',lambda folder:None)
    monkeypatch.setattr(p,'commit',lambda:'b'*40)
    monkeypatch.setattr(p,'source_fingerprint',lambda:{'adaptive_schedule/wave_retention.py':'new','native.py':'same'})
    stage=tmp_path/'protocols/old'
    write(stage/'config.json',{'scientific':'unchanged'})
    write(tmp_path/'teachers.json',[])
    write(tmp_path/'historical_references.json',{})
    write(stage/'PREPARED.json',{'commit':'a'*40,'config_sha256':sha(stage/'config.json'),
        'teachers_sha256':sha(tmp_path/'teachers.json'),'shared_r0_sha256':{},
        'source_fingerprint':{'adaptive_schedule/wave_retention.py':'old','native.py':'same'}})
    write(tmp_path/'ACTIVE_PROTOCOL.json',{'directory':'protocols/old','prepared_sha256':sha(stage/'PREPARED.json')})
    return stage


def test_append_only_repair_preserves_config_and_old_provenance(tmp_path,monkeypatch):
    stage=setup(tmp_path,monkeypatch);digest=sha(stage/'PREPARED.json')
    result=p.repair_preflight(tmp_path,'Correct extracted teacher evidence while preserving all native science')
    assert sha(stage/'PREPARED.json')==digest
    assert p.protocol_dir(tmp_path)!=stage
    assert result['previous_protocol_prepared_sha256']==digest
    assert result['config_sha256']==sha(p.config_path(tmp_path))
    assert result['native_probe_directory'].endswith('aaaaaaaa')
    assert not p.acceptance_path(tmp_path).exists()


@pytest.mark.parametrize('barrier',['native_change','binding','accepted','inputs'])
def test_repair_rejects_scientific_changes_and_started_work(tmp_path,monkeypatch,barrier):
    stage=setup(tmp_path,monkeypatch)
    if barrier=='native_change':monkeypatch.setattr(p,'source_fingerprint',lambda:{'native.py':'changed'})
    if barrier=='binding':write(tmp_path/'versions/v6/binding.json',{})
    if barrier=='accepted':write(stage/'ACCEPTANCE_COMPLETE.json',{})
    if barrier=='inputs':write(tmp_path/'teachers.json',[{'changed':True}])
    before=read(tmp_path/'ACTIVE_PROTOCOL.json')
    with pytest.raises(ValueError):p.repair_preflight(tmp_path,'Concrete documented diagnostic repair rationale')
    assert read(tmp_path/'ACTIVE_PROTOCOL.json')==before
