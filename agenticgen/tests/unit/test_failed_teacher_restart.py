"""Administrative restart fixtures; never scientific or LLM performance evidence."""
import contextlib
from pathlib import Path
import pytest
from agenticgen.adaptive_schedule.io import read, write, sha, value_sha
from agenticgen.adaptive_schedule.group_protocol import configuration
from agenticgen.runtime import restart_failed_teacher as m


@pytest.fixture
def failed_teacher(tmp_path, monkeypatch):
    from agenticgen.adaptive_schedule import group_run
    from agenticgen.adaptive_schedule import wave_retention
    root=tmp_path/'results';repo=tmp_path/'repo';repo.mkdir();old=root/'protocols/old';base=root/'versions/v7'
    c=configuration();c['llm']['max_output_tokens']=32768;write(old/'config.json',c)
    write(root/'teachers.json',[{'teacher_id':'a'},{'teacher_id':'b'}])
    fp={'adaptive_schedule/wave_native.py':'same','adaptive_schedule/group_protocol.py':'old'}
    newfp={**fp,'adaptive_schedule/group_protocol.py':'new','runtime/restart_failed_teacher.py':'added'}
    skills={k:sha(repo/n) for k,n in []}
    for k,n in group_run.SKILLS.items():
        p=repo/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('frozen generic skill');skills[k]=sha(p)
    prepared={'source_fingerprint':fp,'config_sha256':sha(old/'config.json'), 'teachers_sha256':sha(root/'teachers.json'),'commit':'old'}
    for n in ['PREPARED.json','ACCEPTANCE_COMPLETE.json']:write(old/n,prepared)
    write(base/'binding.json',{**prepared,'skill_hashes':skills});write(base/'deployment.json',{'commit':'old','time':1})
    write(base/'search_policy.json',{'upper_amplitude':1});write(base/'review_approval.json',{'full_prior_review':52})
    for n in ['config.json','PREPARED.json','ACCEPTANCE_COMPLETE.json','historical_references.json']:write(root/n,{})
    write(root/'ACTIVE_PROTOCOL.json',{'directory':'protocols/old','prepared_sha256':sha(old/'PREPARED.json')})
    write(root/'versions/v6/binding.json',{'complete_version':True});write(root/'FINALIZED_v6.json',{'done':True})
    jobs={}
    for t in ['a','b']:
        write(root/'shared_r0'/t/'COMPLETE.json',{'R0':True})
        boot={'jobs':[{'id':str(i),'ticks':[i,5,5,5]} for i in range(4)]};write(base/t/'bootstrap/hypotheses.json',boot)
        jobs[t]={'teacher_id':t,'observed_round':0,'jobs':[{'ticks':j['ticks'],'source_proposal_id':'bootstrap:'+j['id'],'proposal':j} for j in boot['jobs']]}
        for r in range(1,6 if t=='a' else 4):
            wave=base/t/f'round_{r:02d}'/'wave'
            for n in ['COMPLETE.json','QUALITY_COMPLETE.json','CLEANUP_COMPLETE.json']:write(wave/n,{'attempts':100})
            write(base/t/f'round_{r:02d}'/'council/COMPLETE.json',{})
            write(base/t/f'RETENTION_COMPLETE_{r:02d}.json',{})
            write(base/t/f'round_{r:02d}'/'council/turn.receipt.json',{'provider_request_id':t+str(r),'usage':{'completion_tokens':32768}})
    write(base/'pipeline.json',{'binding':{'initial_jobs_sha256':value_sha(jobs)},
        'teachers':{'a':{'phase':'complete','round':5},'b':{'phase':'failed','round':3}},
        'events':[{'kind':'gpu_started','teacher':'a','time':1},{'kind':'gpu_started','teacher':'b','time':2}]})
    auth=repo/'auth.json';write(auth,{'authority':'human_user','version':'v7','rerun_all_five_rounds':True,
        'delete_failed_candidate_outputs':True,'preserve_completed_teachers':True,'failed_teachers':['b'],'max_output_tokens':65536})
    monkeypatch.setattr(m,'idle_locks',lambda root:contextlib.nullcontext());monkeypatch.setattr(m,'no_worker',lambda root:None)
    monkeypatch.setattr(m,'assert_committed_runtime',lambda:None);monkeypatch.setattr(m,'repository',lambda:repo)
    monkeypatch.setattr(m,'committed',lambda p:{'path':'auth.json','sha256':sha(p)})
    monkeypatch.setattr(m,'commit',lambda:'new12345');monkeypatch.setattr(m,'source_fingerprint',lambda:newfp)
    monkeypatch.setattr(wave_retention,'verify_snapshot',lambda p:None)
    return root,auth,jobs


def accept(root):
    folder=m.protocol_dir(root);p=read(folder/'PREPARED.json');probe=root/p['native_probe_directory']
    write(probe/'results.json',{'synthetic':True});write(folder/'validation/gradient_checks.json',{'passed':True})
    write(folder/'ACCEPTANCE_COMPLETE.json',{'source_fingerprint':m.source_fingerprint(),'config_sha256':sha(folder/'config.json'),
        'teachers':2,'recovery_checks':2,'common_randomness':True,'different_hypothesis_schedules':True,'affinity_vjp':False,
        'results_sha256':sha(probe/'results.json'),'gradient_checks_sha256':sha(folder/'validation/gradient_checks.json')})


def test_only_failed_teacher_restarts_all_five_and_keeps_completed_costs(failed_teacher):
    root,auth,jobs=failed_teacher;base=root/'versions/v7';kept=m.inventory(base/'a');r0=m.inventory(root/'shared_r0')
    op=Path(m.stage(root,'v7',auth)['operation'])
    assert (base/'b').exists()
    with pytest.raises(FileNotFoundError):m.finish(root,op)
    accept(root)
    with pytest.raises(ValueError):m.verify_ready(root,'v7')
    result=m.finish(root,op)
    assert result['deleted_candidate_count']==300 and not (base/'b').exists()
    assert m.inventory(base/'a')==kept and m.inventory(root/'shared_r0')==r0
    assert (op/'failure_audit/b/round_03/council/turn.receipt.json').exists()
    assert m.preserved_jobs(root,'v7')=={'a':jobs['a']} and m.preserved_jobs(root,'v8')=={}
    assert m.preserved_provenance(root,'v7')['a']['deployment']['commit']=='old'
    m.seed_pipeline(root,'v7',jobs,5)
    from agenticgen.adaptive_schedule.teacher_pipeline import run_teacher_pipeline
    seen=[]
    def infer(t,r,j):seen.append((t,r));return {}
    run_teacher_pipeline(jobs,base,rounds=5,infer=infer,
        analyze=lambda t,r,a,c:{'teacher_id':t,'observed_round':r},
        commit=lambda *args:{'quality_complete':True,'retention_complete':True})
    assert seen==[('b',r) for r in range(1,6)] and m.inventory(base/'a')==kept
    assert read(base/'pipeline.json')['events'][0]['teacher']=='a'
    assert m.verify_ready(root,'v8')['version']=='v7'
    assert m.finish(root,op)['phase']=='deleted_ready_for_new_approval'


@pytest.mark.parametrize('mutation',['alive','nonterminal','finalized','barrier','scope','science','config','skill','bootstrap'])
def test_stage_rejects_unsafe_scope_without_mutation(failed_teacher,monkeypatch,mutation):
    root,auth,jobs=failed_teacher;base=root/'versions/v7';pointer=(root/'ACTIVE_PROTOCOL.json').read_bytes()
    if mutation=='alive':monkeypatch.setattr(m,'no_worker',lambda r:(_ for _ in ()).throw(ValueError('alive')))
    if mutation=='nonterminal':
        p=read(base/'pipeline.json');p['teachers']['b']['phase']='analyzing';write(base/'pipeline.json',p)
    if mutation=='finalized':write(root/'FINALIZED_v7.json',{})
    if mutation=='barrier':(base/'a/round_05/wave/QUALITY_COMPLETE.json').unlink()
    if mutation=='scope':
        a=read(auth);a['failed_teachers']=['a','b'];write(auth,a)
    if mutation=='science':monkeypatch.setattr(m,'source_fingerprint',lambda:{'adaptive_schedule/wave_native.py':'changed'})
    if mutation=='config':
        import agenticgen.adaptive_schedule.group_protocol as gp
        c=configuration();c['max_atom_step_A']=1;monkeypatch.setattr(gp,'configuration',lambda:c)
    if mutation=='skill':
        from agenticgen.adaptive_schedule.group_run import SKILLS
        (m.repository()/SKILLS['analyst']).write_text('altered')
    if mutation=='bootstrap':write(base/'b/bootstrap/hypotheses.json',{'jobs':[]})
    with pytest.raises(ValueError):m.stage(root,'v7',auth)
    assert (root/'ACTIVE_PROTOCOL.json').read_bytes()==pointer


def test_finish_resumes_interrupted_deletion_and_rejects_new_files(failed_teacher,monkeypatch):
    root,auth,jobs=failed_teacher;base=root/'versions/v7';op=Path(m.stage(root,'v7',auth)['operation']);accept(root)
    original=Path.unlink;count=[0]
    def interrupted(p,*args,**kwargs):
        if p.is_relative_to(base):
            count[0]+=1
            if count[0]==3:raise OSError('simulated interruption')
        return original(p,*args,**kwargs)
    monkeypatch.setattr(Path,'unlink',interrupted)
    with pytest.raises(OSError):m.finish(root,op)
    monkeypatch.setattr(Path,'unlink',original)
    write(base/'b/unexpected.json',{})
    with pytest.raises(ValueError,match='Unknown'):m.finish(root,op)
    (base/'b/unexpected.json').unlink()
    assert m.finish(root,op)['deleted_candidate_count']==300


def test_tampered_preserved_teacher_and_native_acceptance_block_finish(failed_teacher):
    root,auth,jobs=failed_teacher;op=Path(m.stage(root,'v7',auth)['operation']);accept(root)
    p=root/'versions/v7/a/round_05/wave/COMPLETE.json';old=p.read_bytes();write(p,{'changed':True})
    with pytest.raises(ValueError,match='Protected'):m.finish(root,op)
    p.write_bytes(old);write(m.protocol_dir(root)/'validation/gradient_checks.json',{'passed':False})
    with pytest.raises(ValueError,match='acceptance'):m.finish(root,op)
