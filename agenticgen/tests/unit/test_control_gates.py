import contextlib
import pytest
from agenticgen.adaptive_schedule.io import write, read, sha, value_sha
from agenticgen.adaptive_schedule.group_protocol import configuration
from agenticgen.adaptive_schedule.search_policy import policy
from agenticgen.adaptive_schedule import version_gate as gate
from agenticgen.adaptive_schedule.protocol_state import stage_protocol, load_config, acceptance_path


def test_protocol_migration_preserves_prepared_r0_and_requires_new_acceptance(tmp_path,monkeypatch):
    import agenticgen.runtime.locks as locks
    monkeypatch.setattr(locks,'lock',lambda root: contextlib.nullcontext())
    old=configuration()
    for key in ['condition_comparison','search_policy','rollback']:old.pop(key)
    old.update(schema='agenticgen-core1hop-fixedstart-v1',versions=[f'v{i}' for i in range(6,21)],total_result_budget=15050)
    write(tmp_path/'config.json',old);write(tmp_path/'teachers.json',[{'teacher_id':'t'}])
    write(tmp_path/'PREPARED.json',{'config_sha256':sha(tmp_path/'config.json'),'teachers_sha256':sha(tmp_path/'teachers.json')})
    write(tmp_path/'ACCEPTANCE_COMPLETE.json',{'original_native_acceptance':True,'config_sha256':sha(tmp_path/'config.json')})
    write(tmp_path/'shared_r0/t/COMPLETE.json',{'files_sha256':{}})
    protected={str(p.relative_to(tmp_path)):sha(p) for p in tmp_path.rglob('*') if p.is_file()}
    stage_protocol(tmp_path,'Authorized ranking/range/budget protocol extension, preserving original R0.')
    assert all(sha(tmp_path/p)==digest for p,digest in protected.items())
    assert load_config(tmp_path)['versions'][-1]=='v25'
    assert not acceptance_path(tmp_path).exists()
    with pytest.raises(ValueError,match='already staged'):stage_protocol(tmp_path,'Do not repeat an already recorded migration transaction.')


def gate_fixture(tmp_path,monkeypatch,previous='v10',upper=5):
    write(tmp_path/'config.json',configuration())
    for n in range(6,int(previous[1:])+1):
        write(tmp_path/'versions'/f'v{n}'/'search_policy.json',policy())
    version='v'+str(int(previous[1:])+1)
    current=policy(upper);write(tmp_path/'versions'/version/'search_policy.json',current)
    monkeypatch.setattr(gate,'version_metrics',lambda root: [])
    approval={'search_policy_sha256':value_sha(current),
        'condition_comparison_sha256':value_sha(configuration()['condition_comparison'])}
    return version,approval


def test_expansion_requires_reviewed_evidence_and_exact_policy_hash(tmp_path,monkeypatch):
    version,approval=gate_fixture(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='substantive boundary'):gate.check_control_gate(tmp_path,version,approval)
    approval['range_expansion']={'boundary_response':'Measured boundary responses support a new range question.',
        'delivered_movement_and_clipping':'Recorded applied displacements and clipping must be reviewed.',
        'unresolved_hypothesis':'A stronger permitted signal may distinguish control saturation.',
        'document':'review.md','document_sha256':'testdigest'}
    from agenticgen.adaptive_schedule import group_review
    calls=[];monkeypatch.setattr(group_review,'committed_file',lambda *args:calls.append(args))
    gate.check_control_gate(tmp_path,version,approval)
    assert len(calls)==1
    changed=policy(5,.7);write(tmp_path/'versions'/version/'search_policy.json',changed)
    with pytest.raises(ValueError,match='exact version'):gate.check_control_gate(tmp_path,version,approval)


def test_regression_gate_requires_historical_best_not_just_approval_flag(tmp_path,monkeypatch):
    version,approval=gate_fixture(tmp_path,monkeypatch,previous='v8',upper=1)
    rows=[dict(id='v6',condition=1.,functional_gap=1.,best=8),
          dict(id='v7',condition=1.,functional_gap=1.1,best=9),
          dict(id='v8',condition=1.,functional_gap=1.2,best=10)]
    monkeypatch.setattr(gate,'version_metrics',lambda root: rows)
    write(tmp_path/'versions/v6/binding.json',{'skill_hashes':{'analyst':'old_a','synthesizer':'old_s'},'source_fingerprint':{'native':'same'},'search_policy_sha256':value_sha(policy())})
    write(tmp_path/'versions/v6/search_policy.json',policy())
    with pytest.raises(ValueError,match='historical-best'):gate.check_control_gate(tmp_path,version,approval)
    approval['rollback']={'restored_version':'v6','restored_binding_sha256':sha(tmp_path/'versions/v6/binding.json'),
        'code_compatibility_document':'review.md','code_compatibility_sha256':'compat',
        'source_binding_sha256':value_sha({'native':'same'})}
    from agenticgen.adaptive_schedule import group_review
    calls=[];monkeypatch.setattr(group_review,'committed_file',lambda *args:calls.append(args))
    gate.check_control_gate(tmp_path,version,approval)
    assert [c[1] for c in calls]==['old_a','old_s','compat']
