import copy
import json
import pytest
import jsonschema

from agenticgen.adaptive_schedule.evidence_pack import pack, unpack
from agenticgen.adaptive_schedule.learning import action_domain, validate_controller_hypotheses
from agenticgen.adaptive_schedule.search_policy import policy
from agenticgen.adaptive_schedule.group_agent import response_schema
from agenticgen.adaptive_schedule.group_protocol import configuration
from agenticgen.runtime.restart_failed_version import safe_base, inventory, audit_document


def test_shared_evidence_is_exactly_reversible_without_losing_candidates():
    block={'candidates':[{'id':i,'value':i*.001,'failed':i==3} for i in range(25)]}
    original={f'E{i}':{'source_id':str(i),'A':block,'B':block} for i in range(8)}
    before=copy.deepcopy(original); packed,blocks=pack(original)
    assert original==before and unpack(packed,blocks)==original
    assert len(json.dumps([packed,blocks])) < len(json.dumps(original))/2
    assert pack(original)==(packed,blocks)


def test_missing_cyclic_and_reserved_evidence_is_rejected():
    with pytest.raises(ValueError):unpack({'$evidence_ref':'B0'}, {})
    with pytest.raises(ValueError):unpack({'$evidence_ref':'B0'}, {'B0':{'$evidence_ref':'B0'}})
    with pytest.raises(ValueError):pack({'$evidence_ref':'spoof'})


def test_real_failure_anchor_uses_current_group_not_R0():
    anchor=[7,5,5,5];p=policy();domain=action_domain(anchor,{},p)
    assert domain[0]==[5,6,7,8,9]
    class Tools:registry={}
    schema=response_schema(Tools(),2,{'e'},final_only=True,controller=True,
        action_anchor=anchor,local_domain=domain)
    h=schema['properties']['hypotheses']['items']['properties']
    with pytest.raises(jsonschema.ValidationError):jsonschema.validate([5]*4,h['reference_ticks'])
    with pytest.raises(jsonschema.ValidationError):jsonschema.validate([3,5,5,5],h['ticks'])
    jsonschema.validate([5,5,5,5],h['ticks'])
    msg={'hypotheses':[{'id':'H1','ticks':[4,5,5,5],'reference_ticks':anchor,'evidence_ids':['e']}]}
    with pytest.raises(ValueError,match='anchor tick 7, proposed tick 4, delta -3'):
        validate_controller_hypotheses(msg,anchor,1,{'e'},{},[],p)
    assert action_domain(anchor,{0:-1},p)[0]==[4,5,6,7,8,9]


def test_expanded_probe_not_restricted_to_local_schema():
    class Tools:registry={}
    schema=response_schema(Tools(),2,{'e'},final_only=True,controller=True,tick_upper=50,
        action_anchor=[5]*4,local_domain=[[3,4,5,6,7]]*4)
    jsonschema.validate([45,5,5,5],schema['properties']['hypotheses']['items']['properties']['ticks'])


def test_budget_change_does_not_inflate_output_or_sampling_budget():
    c=configuration()
    assert c['llm']['input_hard_bytes']==524288
    assert c['llm']['max_output_tokens']==32768 and c['council']['max_output_tokens']==65536
    assert c['population_size']==25 and c['rounds']==5 and c['physical_batch_size']==100


def test_deletion_paths_and_audit_selection(tmp_path):
    for name in ['../v6','v6/../../shared_r0','v0','v6;rm']:
        with pytest.raises(ValueError):safe_base(tmp_path,name)
    assert safe_base(tmp_path,'v6')==tmp_path/'versions/v6'
    assert audit_document('teacher/round_01/council/analysts/G1/x.request.json')
    assert not audit_document('teacher/round_01/wave/final_records.json')
    assert not audit_document('teacher/round_01/wave/terminal.pt.gz')
    target=tmp_path/'real';target.mkdir();(target/'x').write_text('x')
    assert inventory(target)['x']['bytes']==1


@pytest.fixture
def staged(tmp_path,monkeypatch):
    import contextlib
    import agenticgen.runtime.restart_failed_version as m
    from agenticgen.adaptive_schedule.io import write,sha
    root=tmp_path;folder=root/'protocols/new';base=safe_base(root,'v6');op=root/'operations/rerun'
    write(folder/'PREPARED.json',{})
    write(folder/'config.json',{'fixed':'settings'})
    write(root/'ACTIVE_PROTOCOL.json',{'directory':'protocols/new','prepared_sha256':sha(folder/'PREPARED.json')})
    write(folder/'ACCEPTANCE_COMPLETE.json',{'source_fingerprint':{'a':'b'},'config_sha256':sha(folder/'config.json'),'teachers':2})
    write(root/'shared_r0/sentinel.json',{'keep':True})
    write(base/'teacher/round_01/council/G1.request.json',{'prompt':'original'})
    write(base/'teacher/round_01/wave/final_records.json',[{'score':1}])
    (base/'teacher/round_01/wave/terminal.pt.gz').write_bytes(b'failed-structure')
    write(op/'journal.json',{'phase':'staged','version':'v6','commit':'c','new_protocol':'protocols/new',
        'protected_sha256':{'shared_r0/sentinel.json':sha(root/'shared_r0/sentinel.json')},
        'failed_attempt_files':inventory(base),'failed_candidate_count':200})
    monkeypatch.setattr(m,'idle_locks',lambda r:contextlib.nullcontext())
    monkeypatch.setattr(m,'no_worker',lambda r:None)
    monkeypatch.setattr(m,'commit',lambda:'c')
    monkeypatch.setattr(m,'source_fingerprint',lambda:{'a':'b'})
    return root,base,op,m


def test_full_restart_deletes_old_results_but_keeps_failure_audit_and_R0(staged):
    from agenticgen.adaptive_schedule.io import read
    root,base,op,m=staged
    result=m.finish(root,op)
    assert result['deleted_failed_candidate_count']==200 and not base.exists()
    assert (op/'failure_audit/teacher/round_01/council/G1.request.json').exists()
    assert not list(op.rglob('final_records.json')) and not list(op.rglob('*.pt.gz'))
    assert read(root/'shared_r0/sentinel.json')=={'keep':True}
    assert m.finish(root,op)['phase']=='deleted_ready_for_new_approval'


def test_missing_native_validation_never_deletes(staged):
    root,base,op,m=staged
    (root/'protocols/new/ACCEPTANCE_COMPLETE.json').unlink()
    with pytest.raises(FileNotFoundError):m.finish(root,op)
    assert base.exists()


def test_interrupted_delete_is_resumable(staged,monkeypatch):
    root,base,op,m=staged;real=m.shutil.rmtree
    def interruption(path):
        (path/'teacher/round_01/wave/terminal.pt.gz').unlink()
        raise OSError('simulated interruption')
    monkeypatch.setattr(m.shutil,'rmtree',interruption)
    with pytest.raises(OSError):m.finish(root,op)
    assert (op/'discarded_version').exists() and not base.exists()
    monkeypatch.setattr(m.shutil,'rmtree',real)
    assert m.finish(root,op)['rerun_rounds']==5


def test_modified_failed_output_or_R0_refuses_cleanup(staged):
    root,base,op,m=staged
    (root/'shared_r0/sentinel.json').write_text('changed')
    with pytest.raises(ValueError,match='Protected artifact'):m.finish(root,op)
    assert base.exists()

@pytest.fixture
def failed_protocol(tmp_path,monkeypatch):
    import contextlib
    import agenticgen.runtime.restart_failed_version as m
    import agenticgen.adaptive_schedule.io as io
    import agenticgen.adaptive_schedule.wave_retention as retention
    import agenticgen.adaptive_schedule.group_run as run
    from agenticgen.adaptive_schedule.io import write,sha
    root=tmp_path; old=root/'protocols/old';base=safe_base(root,'v6')
    config=configuration();config['llm'].update(input_soft_bytes=65536,input_hard_bytes=262144)
    write(old/'config.json',config);write(root/'teachers.json',[{'teacher_id':'t'}])
    fingerprint={'adaptive_schedule/group_agent.py':'old'}
    prepared={'config_sha256':sha(old/'config.json'),'source_fingerprint':fingerprint,
              'teachers_sha256':sha(root/'teachers.json')}
    for name in ['PREPARED.json','ACCEPTANCE_COMPLETE.json']:write(old/name,prepared)
    write(root/'ACTIVE_PROTOCOL.json',{'directory':'protocols/old','prepared_sha256':sha(old/'PREPARED.json')})
    for name in ['config.json','PREPARED.json','ACCEPTANCE_COMPLETE.json','historical_references.json']:write(root/name,{})
    write(base/'binding.json',dict(prepared,skill_hashes={}))
    write(base/'pipeline.json',{'teachers':{'t':{'phase':'failed'}}})
    write(base/'t/round_01/wave/COMPLETE.json',{'attempts':100})
    write(root/'shared_r0/t/COMPLETE.json',{'kept':True})
    monkeypatch.setattr(m,'idle_locks',lambda r:contextlib.nullcontext())
    monkeypatch.setattr(m,'no_worker',lambda r:None)
    monkeypatch.setattr(m,'commit',lambda:'12345678')
    monkeypatch.setattr(m,'source_fingerprint',lambda:{'adaptive_schedule/group_agent.py':'new'})
    monkeypatch.setattr(m,'check_authorization',lambda p,v:{'path':'auth','sha256':'a'})
    monkeypatch.setattr(io,'repository',lambda:root)
    monkeypatch.setattr(retention,'verify_snapshot',lambda p:None)
    monkeypatch.setattr(run,'SKILLS',{})
    return root,base,old,m


def test_staging_preserves_failed_outputs_and_demands_new_acceptance(failed_protocol):
    from agenticgen.adaptive_schedule.io import read
    root,base,old,m=failed_protocol;before=inventory(base);protected=inventory(old)
    result=m.stage(root,'v6',root/'auth')
    assert inventory(base)==before and inventory(old)==protected
    assert not result['candidate_outputs_deleted'] and result['requires_native_validation']
    new=root/read(root/'ACTIVE_PROTOCOL.json')['directory']
    assert read(new/'config.json')['llm']['input_hard_bytes']==524288
    assert not (new/'ACCEPTANCE_COMPLETE.json').exists()


def test_staging_rejects_scientific_change_or_completed_version(failed_protocol,monkeypatch):
    root,base,old,m=failed_protocol
    monkeypatch.setattr(m,'source_fingerprint',lambda:{'adaptive_schedule/wave_native.py':'changed'})
    with pytest.raises(ValueError,match='sampling/reward/RNG'):m.stage(root,'v6',root/'auth')
    (root/'FINALIZED_v6.json').write_text('{}')
    with pytest.raises(ValueError,match='Finalized'):m.stage(root,'v6',root/'auth')
    assert base.exists()
