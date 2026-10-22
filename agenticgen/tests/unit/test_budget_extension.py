"""Synthetic administrative fixtures; never substitute for native acceptance or GPT review."""
import contextlib
from pathlib import Path

import pytest

from agenticgen.adaptive_schedule.io import read, write, sha, value_sha
from agenticgen.adaptive_schedule.group_protocol import configuration
from agenticgen.runtime import extend_budget as m


@pytest.fixture
def boundary(tmp_path, monkeypatch):
    import agenticgen.adaptive_schedule.wave_retention as retention
    root = tmp_path/'results'; repo = tmp_path/'repo'; repo.mkdir(); old = root/'protocols/old'
    config = configuration(); config.update(versions=[f'v{i}' for i in range(6,26)], total_result_budget=20050)
    write(old/'config.json', config); write(root/'teachers.json', [{'teacher_id':'a'}, {'teacher_id':'b'}])
    r0 = {}
    for t in ['a', 'b']:
        write(root/'shared_r0'/t/'COMPLETE.json', {'keep':'R0'}); r0[t] = sha(root/'shared_r0'/t/'COMPLETE.json')
    fp = {'adaptive_schedule/group_protocol.py':'old', 'adaptive_schedule/group_run.py':'old_budget_assertion',
          'adaptive_schedule/search_policy.py':'old_version_bound', 'adaptive_schedule/wave_native.py':'unchanged'}
    prepared = {'config_sha256':sha(old/'config.json'), 'source_fingerprint':fp,
                'teachers_sha256':sha(root/'teachers.json'), 'shared_r0_sha256':r0, 'commit':'old'}
    for name in ['PREPARED.json', 'ACCEPTANCE_COMPLETE.json']:write(old/name,prepared)
    for name in ['config.json','PREPARED.json','ACCEPTANCE_COMPLETE.json','historical_references.json']:write(root/name,{})
    write(root/'ACTIVE_PROTOCOL.json', {'directory':'protocols/old','prepared_sha256':sha(old/'PREPARED.json')})
    base=root/'versions/v6';write(base/'binding.json', prepared);write(base/'report.json', {'formal':True})
    write(base/'pipeline.json', {'teachers':{t:{'phase':'complete','round':5} for t in ['a','b']}})
    for t in ['a','b']:
        for r in range(1,6):
            wave=base/t/f'round_{r:02d}'/'wave'
            for name in ['COMPLETE.json','QUALITY_COMPLETE.json','CLEANUP_COMPLETE.json']:write(wave/name,{'attempts':100})
            write(base/t/f'round_{r:02d}'/'council/COMPLETE.json',{})
            write(base/t/f'RETENTION_COMPLETE_{r:02d}.json',{})
    sessions=[];coverage=[]
    for i in range(52):
        name=f'versions/v6/evidence/{i}.request.json';write(root/name,{'synthetic':i});files={name:sha(root/name)}
        s={'id':str(i),'files_sha256':files,'evidence_sha256':value_sha(files)};sessions.append(s)
        coverage.append(dict(id=str(i),evidence_sha256=s['evidence_sha256'],evidence_files=[name],
            assessment='Synthetic fixture only, not actual scientific review.',
            decision_update='Synthetic fixture does not authorize scientific sampling.',
            counterexample_or_limitation='Synthetic fixture has no scientific interpretation.'))
    shard=base/'review/shard.json';write(shard,{'sessions':sessions})
    manifest=base/'review/manifest.json';write(manifest,{'expected_sessions':52,'binding_sha256':sha(base/'binding.json'),
        'shards':[{'path':shard.relative_to(root).as_posix(),'sha256':sha(shard)}]})
    finalized=root/'FINALIZED_v6.json';write(finalized,{'version':'v6','rounds':5,'outcomes':1000,
        'review_manifest_sha256':sha(manifest),'report_sha256':sha(base/'report.json')})
    write(repo/'coverage.json',{'sessions':coverage})
    progress=repo/'progress.json';write(progress,{'version':'v6','reviewed_sessions':52,'expected_sessions':52,
        'finalized_sha256':sha(finalized),'manifest_sha256':sha(manifest),
        'coverage':[{'path':'coverage.json','sha256':sha(repo/'coverage.json')}]})
    auth=repo/'authorization.json';write(auth,read(Path(__file__).parents[2]/'docs/authorizations/20261023_v30_budget.json'))
    monkeypatch.setattr(m,'idle_locks',lambda r:contextlib.nullcontext())
    monkeypatch.setattr(m,'no_worker',lambda r:None)
    monkeypatch.setattr(m,'assert_committed_runtime',lambda:None)
    monkeypatch.setattr(m,'repository',lambda:repo)
    monkeypatch.setattr(m,'committed',lambda p:{'path':Path(p).relative_to(repo).as_posix(),'sha256':sha(p)})
    monkeypatch.setattr(m,'commit',lambda:'new12345')
    monkeypatch.setattr(m,'source_fingerprint',lambda:{**fp,'adaptive_schedule/group_protocol.py':'new',
        'adaptive_schedule/group_run.py':'new_budget_assertion','adaptive_schedule/search_policy.py':'new_version_bound',
        'runtime/extend_budget.py':'added'})
    monkeypatch.setattr(retention,'verify_snapshot',lambda p:None)
    return root,old,progress,auth


def accept(root):
    folder=m.protocol_dir(root);p=read(folder/'PREPARED.json');probe=root/p['native_probe_directory']
    write(probe/'results.json',[{'synthetic':True}]);write(folder/'validation/gradient_checks.json',{'passed':True})
    write(folder/'ACCEPTANCE_COMPLETE.json',{'source_fingerprint':m.source_fingerprint(),'config_sha256':sha(folder/'config.json'),
        'teachers':2,'recovery_checks':2,'common_randomness':True,'different_hypothesis_schedules':True,'affinity_vjp':False,
        'results_sha256':sha(probe/'results.json'),'gradient_checks_sha256':sha(folder/'validation/gradient_checks.json')})


def test_append_preserves_old_results_and_blocks_until_native_acceptance(boundary):
    root,old,progress,auth=boundary;old_before=m.inventory(old);results=m.inventory(root/'versions');r0=m.inventory(root/'shared_r0')
    result=m.stage(root,'v6',auth,progress);op=Path(result['operation'])
    assert m.inventory(old)==old_before and m.inventory(root/'versions')==results and m.inventory(root/'shared_r0')==r0
    assert read(m.protocol_dir(root)/'config.json')['versions'][-1]=='v30'
    with pytest.raises(FileNotFoundError):m.finish(root,op)
    accept(root)
    with pytest.raises(ValueError,match='not complete'):m.verify_ready(root,'v7')
    assert m.finish(root,op)['launch_approved'] is False
    assert m.verify_ready(root,'v7')['phase']=='accepted_for_review_check'
    assert m.verify_ready(root,'v30')['next_version']=='v7'
    with pytest.raises(ValueError):m.verify_ready(root,'v31')
    with pytest.raises(ValueError):m.verify_ready(root,'v6')


@pytest.mark.parametrize('mutation', ['active','future','quality','finalized','scientific_source','scientific_config','review'])
def test_stage_rejects_unsafe_boundaries_without_switching(boundary,monkeypatch,mutation):
    root,old,progress,auth=boundary;pointer=(root/'ACTIVE_PROTOCOL.json').read_bytes()
    if mutation=='active':monkeypatch.setattr(m,'no_worker',lambda r:(_ for _ in ()).throw(ValueError('active worker')))
    if mutation=='future':write(root/'versions/v7/binding.json',{})
    if mutation=='quality':(root/'versions/v6/a/round_03/wave/QUALITY_COMPLETE.json').unlink()
    if mutation=='finalized':write(root/'versions/v6/report.json',{'changed':True})
    if mutation=='scientific_source':monkeypatch.setattr(m,'source_fingerprint',lambda:{'adaptive_schedule/wave_native.py':'different'})
    if mutation=='scientific_config':
        import agenticgen.adaptive_schedule.group_protocol as protocol
        c=configuration();c['max_atom_step_A']=1
        monkeypatch.setattr(protocol,'configuration',lambda:c)
    if mutation=='review':
        p=read(progress);p['reviewed_sessions']=51;write(progress,p)
    with pytest.raises(ValueError):m.stage(root,'v6',auth,progress)
    assert (root/'ACTIVE_PROTOCOL.json').read_bytes()==pointer


def test_aborted_sources_require_individual_review(boundary):
    root,old,progress,auth=boundary
    write(root/'operations/failed_version_rerun_v6_old/failure_audit/prompt.json',{'failed':True})
    with pytest.raises(ValueError,match='Aborted attempt review'):m.stage(root,'v6',auth,progress)


def test_changed_reviewed_prompt_rejects_stage(boundary):
    root,old,progress,auth=boundary
    write(root/'versions/v6/evidence/51.request.json',{'altered':True})
    with pytest.raises(ValueError,match='Reviewed source'):m.stage(root,'v6',auth,progress)


def test_old_result_or_acceptance_mutation_rejects_finish(boundary):
    root,old,progress,auth=boundary;op=Path(m.stage(root,'v6',auth,progress)['operation']);accept(root)
    p=root/'shared_r0/a/COMPLETE.json';original=p.read_bytes();write(p,{'modified':True})
    with pytest.raises(ValueError,match='Protected artifact'):m.finish(root,op)
    p.write_bytes(original);folder=m.protocol_dir(root)
    write(folder/'validation/gradient_checks.json',{'passed':False})
    with pytest.raises(ValueError,match='evidence changed'):m.finish(root,op)
