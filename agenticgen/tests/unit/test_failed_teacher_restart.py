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
    # Preserve the historical V7 32K -> 64K fixture while defaults evolve.
    from agenticgen.adaptive_schedule import group_protocol
    v7_config=configuration();v7_config['llm']['max_output_tokens']=65536
    monkeypatch.setattr(group_protocol,'configuration',lambda:v7_config)
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


@pytest.fixture
def focused_restart(failed_teacher,monkeypatch):
    root,auth,jobs=failed_teacher
    (root/'versions/v7').rename(root/'versions/v10')
    old=root/'protocols/old';base=root/'versions/v10'
    config=read(old/'config.json');config['llm']['max_output_tokens']=65536;write(old/'config.json',config)
    p=read(old/'PREPARED.json');p['config_sha256']=sha(old/'config.json')
    p['strategy_review_operation']='operations/adopted_best_round'
    p['master_review_policy']={'mode':'best_round_strategy','full_review_prohibited':True,
                             'authorization':{'path':'review-auth.json','sha256':'reviewhash'}}
    for name in ['PREPARED.json','ACCEPTANCE_COMPLETE.json']:write(old/name,p)
    binding=read(base/'binding.json');binding.update(config_sha256=p['config_sha256']);write(base/'binding.json',binding)
    write(root/'operations/adopted_best_round/journal.json',{'phase':'accepted_for_review_check',
        'commit':p['commit'],'new_protocol':'protocols/old','acceptance_sha256':sha(old/'ACCEPTANCE_COMPLETE.json'),
        'authorization':p['master_review_policy']['authorization'],'protected_sha256':{}})
    write(root/'ACTIVE_PROTOCOL.json',{'directory':'protocols/old','prepared_sha256':sha(old/'PREPARED.json')})
    a=read(auth);a.update(version='v10',previous_max_output_tokens=65536,max_output_tokens=98304);write(auth,a)
    from agenticgen.adaptive_schedule import group_protocol
    monkeypatch.setattr(group_protocol,'configuration',configuration)
    return root,auth,jobs


def test_96k_restart_preserves_best_round_review_and_completed_teacher(focused_restart):
    root,auth,jobs=focused_restart;kept=m.inventory(root/'versions/v10/a')
    op=Path(m.stage(root,'v10',auth)['operation']);p=read(m.prepared_path(root))
    assert p['master_review_policy']['full_review_prohibited'] is True
    assert 'strategy_review_operation' not in p and p['previous_strategy_review_operation']=='operations/adopted_best_round'
    assert read(m.config_path(root))['llm']['max_output_tokens']==98304
    accept(root);m.finish(root,op)
    assert m.verify_ready(root,'v10')['version']=='v10'
    assert m.inventory(root/'versions/v10/a')==kept
    p['master_review_policy']['mode']='full_review';write(m.prepared_path(root),p)
    with pytest.raises(ValueError):m.verify_ready(root,'v10')


@pytest.mark.parametrize('change',['previous_budget','scope','adoption','source','reward'])
def test_96k_restart_rejects_unapproved_changes(focused_restart,monkeypatch,change):
    root,auth,_=focused_restart
    if change in ('previous_budget','scope'):
        a=read(auth);a['previous_max_output_tokens' if change=='previous_budget' else 'max_output_tokens']=32768 if change=='previous_budget' else 131072;write(auth,a)
    elif change=='adoption':
        path=root/'operations/adopted_best_round/journal.json';j=read(path);j['phase']='staged';write(path,j)
    elif change=='source':
        monkeypatch.setattr(m,'source_fingerprint',lambda:{'adaptive_schedule/wave_native.py':'changed'})
    else:
        from agenticgen.adaptive_schedule import group_protocol
        c=configuration();c['max_atom_step_A']=1;monkeypatch.setattr(group_protocol,'configuration',lambda:c)
    with pytest.raises(ValueError):m.stage(root,'v10',auth)


@pytest.fixture
def citation_restart(focused_restart,monkeypatch):
    root,auth,jobs=focused_restart; old=root/'protocols/old';base=root/'versions/v10'
    c=read(old/'config.json');c['llm']['max_output_tokens']=98304;write(old/'config.json',c)
    p=read(old/'PREPARED.json');p['config_sha256']=sha(old/'config.json')
    p.pop('strategy_review_operation');p['teacher_restart_operation']='operations/previous_restart'
    for n in ('PREPARED.json','ACCEPTANCE_COMPLETE.json'):write(old/n,p)
    write(root/'ACTIVE_PROTOCOL.json',{'directory':'protocols/old','prepared_sha256':sha(old/'PREPARED.json')})
    b=read(base/'binding.json');b['config_sha256']=p['config_sha256']
    b['preserved_teacher_provenance']={'a':{'binding':{'original':True},'deployment':{'commit':'original'},'files_sha256':{}}}
    write(base/'binding.json',b)
    write(root/'operations/previous_restart/journal.json',{'phase':'deleted_ready_for_new_approval',
          'commit':'old','new_protocol':'protocols/old','new_acceptance_sha256':sha(old/'ACCEPTANCE_COMPLETE.json'),
          'protected_sha256':{}})
    a=read(auth);a.update(repair_kind='private_evidence_alias_v1',previous_commit='old',previous_max_output_tokens=98304);write(auth,a)
    new={**p['source_fingerprint'],'runtime/restart_failed_teacher.py':'new',
         'adaptive_schedule/group_agent.py':'new','adaptive_schedule/evidence_aliases.py':'new'}
    monkeypatch.setattr(m,'source_fingerprint',lambda:new)
    return root,auth,jobs


def test_citation_restart_keeps_config_and_original_completed_teacher_source(citation_restart):
    root,auth,_=citation_restart;old=(root/'protocols/old/config.json').read_bytes()
    op=Path(m.stage(root,'v10',auth)['operation'])
    assert m.config_path(root).read_bytes()==old
    accept(root);m.finish(root,op)
    assert m.preserved_provenance(root,'v10')['a']['deployment']['commit']=='original'
    assert read(m.prepared_path(root))['authorized_analysis_budget_changes']=={}


@pytest.mark.parametrize('change',['source','previous_commit','budget','prior_transaction'])
def test_citation_restart_rejects_unbound_changes(citation_restart,monkeypatch,change):
    root,auth,_=citation_restart
    if change=='source':monkeypatch.setattr(m,'source_fingerprint',lambda:{**m.read(m.prepared_path(root))['source_fingerprint'],'adaptive_schedule/group_protocol.py':'changed'})
    elif change=='prior_transaction':
        p=root/'operations/previous_restart/journal.json';j=read(p);j['phase']='staged';write(p,j)
    else:
        a=read(auth);a['previous_commit' if change=='previous_commit' else 'max_output_tokens']='wrong' if change=='previous_commit' else 131072;write(auth,a)
    with pytest.raises(ValueError):m.stage(root,'v10',auth)


@pytest.fixture
def interface_restart(citation_restart,monkeypatch):
    root,auth,jobs=citation_restart
    (root/'versions/v10').rename(root/'versions/v11')
    base=root/'versions/v11';a=read(auth)
    a.update(version='v11',repair_kind='bounded_interface_repair_v1',previous_commit='normal-v11')
    write(auth,a);write(base/'deployment.json',{'commit':'normal-v11'})
    fp=read(root/'protocols/old/PREPARED.json')['source_fingerprint']
    monkeypatch.setattr(m,'source_fingerprint',lambda:{**fp,'runtime/restart_failed_teacher.py':'new',
        'providers/deepseek_api.py':'new','adaptive_schedule/group_agent.py':'new'})
    return root,auth,jobs


def test_interface_restart_accepts_compatible_older_protocol_but_preserves_completed_source(interface_restart):
    root,auth,jobs=interface_restart;base=root/'versions/v11';kept=m.inventory(base/'a')
    config=(root/'protocols/old/config.json').read_bytes()
    op=Path(m.stage(root,'v11',auth)['operation']);accept(root);m.finish(root,op)
    assert m.inventory(base/'a')==kept and m.config_path(root).read_bytes()==config
    assert not (base/'b').exists() and (op/'failure_audit/b/round_03/council/turn.receipt.json').exists()
    assert m.preserved_jobs(root,'v11')=={'a':jobs['a']}
    assert read(m.prepared_path(root))['interface_repair_policy']['private_max_format_repairs']==2
    assert read(m.prepared_path(root))['authorized_analysis_budget_changes']=={}
    assert m.verify_ready(root,'v12')['version']=='v11'


@pytest.mark.parametrize('mutation',['source','commit','budget','skills'])
def test_interface_restart_rejects_out_of_scope_change(interface_restart,monkeypatch,mutation):
    root,auth,_=interface_restart
    if mutation=='source':
        fp=m.source_fingerprint();monkeypatch.setattr(m,'source_fingerprint',lambda:{**fp,'adaptive_schedule/wave_native.py':'changed'})
    elif mutation=='skills':
        from agenticgen.adaptive_schedule.group_run import SKILLS
        (m.repository()/SKILLS['analyst']).write_text('changed')
    else:
        a=read(auth);a['previous_commit' if mutation=='commit' else 'max_output_tokens']='other' if mutation=='commit' else 65536
        write(auth,a)
    with pytest.raises(ValueError):m.stage(root,'v11',auth)


@pytest.fixture
def commitment_restart(interface_restart,monkeypatch):
    root,auth,jobs=interface_restart
    (root/'versions/v11').rename(root/'versions/v21');base=root/'versions/v21';old=root/'protocols/old'
    p=read(old/'PREPARED.json');probe=root/'operations/fixed_start_acceptance_old'
    write(probe/'results.json',{'synthetic_native_fixture':True})
    write(old/'validation/gradient_checks.json',{'passed':True})
    accepted={**p,'teachers':2,'recovery_checks':2,'common_randomness':True,
        'different_hypothesis_schedules':True,'affinity_vjp':False,
        'results_sha256':sha(probe/'results.json'),'gradient_checks_sha256':sha(old/'validation/gradient_checks.json')}
    write(old/'ACCEPTANCE_COMPLETE.json',accepted)
    prev=root/'operations/previous_restart/journal.json';j=read(prev)
    j['new_acceptance_sha256']=sha(old/'ACCEPTANCE_COMPLETE.json');write(prev,j)
    shortage=base/'b/round_03/council/insufficient_unexecuted_proposals.json'
    write(shortage,{'eligible_count':3})
    state=read(base/'pipeline.json');state['teachers']['b']['error']={'message':'Fewer than four unexecuted proposed schedules'};write(base/'pipeline.json',state)
    a=read(auth);a.update(version='v21',repair_kind='prior_selected_schedules_v1',failed_round=3,
        shortage_receipt_sha256=sha(shortage));write(auth,a)
    current={**p['source_fingerprint'],**{k:'new' for k in m.COMMITMENT_SOURCE_CHANGES}}
    monkeypatch.setattr(m,'source_fingerprint',lambda:current)
    logical=Path('/opt/MolSteer/operations/agenticgen_deploy_new12345/tests_receipt.json')
    physical=root/'remote-tests.json';write(physical,{'commit':'new12345','passed':True,'scope':'prior_selected_schedules_v1'})
    monkeypatch.setattr(m,'read',lambda path:read(physical if Path(path)==logical else path))
    monkeypatch.setattr(m,'sha',lambda path:sha(physical if Path(path)==logical else path))
    return root,auth,jobs,logical


def test_commitment_repair_reuses_exact_native_evidence_and_protects_completed_teacher(commitment_restart,monkeypatch):
    root,auth,jobs,receipt=commitment_restart
    kept=m.inventory(root/'versions/v21/a');r0=m.inventory(root/'shared_r0')
    old_acceptance=sha(root/'protocols/old/ACCEPTANCE_COMPLETE.json')
    op=Path(m.stage(root,'v21',auth)['operation'])
    with pytest.raises(FileNotFoundError):m.finish(root,op)
    reuse=m.reuse_native_acceptance(root,receipt)
    assert not reuse['fresh_native_sampling'] and reuse['source_acceptance_sha256']==old_acceptance
    m.finish(root,op)
    assert m.inventory(root/'versions/v21/a')==kept and m.inventory(root/'shared_r0')==r0
    assert m.preserved_jobs(root,'v21')=={'a':jobs['a']}
    assert (op/'failure_audit/b/round_03/council/insufficient_unexecuted_proposals.json').is_file()
    assert not (root/'versions/v21/b').exists()
    # Future skills-only revisions retain the original repair receipt identity.
    monkeypatch.setattr(m,'commit',lambda:'later1234')
    assert m.verify_ready(root,'v22')['version']=='v21'
    write(root/'remote-tests.json',{'commit':'new12345','passed':False,'scope':'prior_selected_schedules_v1'})
    with pytest.raises(ValueError):m.native_ready(root)


@pytest.mark.parametrize('mutation',['shortage_hash','round','failure','native_source','config','skills'])
def test_commitment_repair_rejects_other_faults_or_scientific_changes(commitment_restart,monkeypatch,mutation):
    root,auth,_,_=commitment_restart
    if mutation in ('shortage_hash','round'):
        a=read(auth);a['shortage_receipt_sha256' if mutation=='shortage_hash' else 'failed_round']='wrong' if mutation=='shortage_hash' else 2;write(auth,a)
    elif mutation=='failure':
        path=root/'versions/v21/pipeline.json';p=read(path);p['teachers']['b']['error']={'message':'Different failure'};write(path,p)
    elif mutation=='native_source':
        fp=m.source_fingerprint();monkeypatch.setattr(m,'source_fingerprint',lambda:{**fp,'adaptive_schedule/wave_native.py':'changed'})
    elif mutation=='config':
        from agenticgen.adaptive_schedule import group_protocol
        c=configuration();c['max_atom_step_A']=1;monkeypatch.setattr(group_protocol,'configuration',lambda:c)
    else:
        from agenticgen.adaptive_schedule.group_run import SKILLS
        (m.repository()/SKILLS['analyst']).write_text('changed')
    with pytest.raises(ValueError):m.stage(root,'v21',auth)
