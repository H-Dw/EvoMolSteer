"""Audited five-round restart of failed teachers, preserving completed teachers.

This operation never repairs a scientific proposal or approves an experiment.
The prior authorized review scope must pass again against the new fixed revision.
"""
import copy
from pathlib import Path
import shutil

from agenticgen.adaptive_schedule.io import read, write, sha, commit, source_fingerprint, value_sha, repository
from agenticgen.adaptive_schedule.protocol_state import protocol_dir, prepared_path, acceptance_path, config_path
from agenticgen.runtime.launch import idle_locks
from .restart_failed_version import safe_base, inventory, verify_protected, no_worker, audit_document
from .extend_budget import committed, assert_committed_runtime

ALLOWED_SOURCE_CHANGES = {'runtime/restart_failed_teacher.py', 'adaptive_schedule/group_run.py',
    'adaptive_schedule/group_review.py', 'adaptive_schedule/group_protocol.py', 'cli.py',
    'providers/deepseek_api.py'}
GLOBAL_RESET = {'binding.json', 'deployment.json', 'pipeline.json', 'PIPELINE_COMPLETE.json', 'review_approval.json'}
CITATION_SOURCE_CHANGES = {'runtime/restart_failed_teacher.py',
    'adaptive_schedule/group_agent.py', 'adaptive_schedule/evidence_aliases.py'}
INTERFACE_SOURCE_CHANGES = {'runtime/restart_failed_teacher.py',
    'adaptive_schedule/group_agent.py', 'providers/deepseek_api.py'}
COMMITMENT_SOURCE_CHANGES = {'runtime/restart_failed_teacher.py',
    'adaptive_schedule/group_agent.py', 'adaptive_schedule/council.py',
    'adaptive_schedule/committed_wave.py'}


def completed_barriers(base, teacher, config):
    from agenticgen.adaptive_schedule.wave_retention import verify_snapshot
    tb = base/teacher
    for r in range(1, config['rounds']+1):
        wave = tb/f'round_{r:02d}'/'wave'
        for name in ['COMPLETE.json', 'QUALITY_COMPLETE.json', 'CLEANUP_COMPLETE.json']:
            if not (wave/name).is_file():raise ValueError('Completed teacher barrier missing')
        for name in ['COMPLETE.json', 'QUALITY_COMPLETE.json']:
            if read(wave/name).get('attempts') != 4*config['population_size']:
                raise ValueError('Completed teacher count differs')
        if not (tb/f'round_{r:02d}'/'council/COMPLETE.json').is_file() or not (tb/f'RETENTION_COMPLETE_{r:02d}.json').is_file():
            raise ValueError('Completed teacher analysis/retention missing')
    verify_snapshot(tb/'incumbents'/f"r{config['rounds']:02d}")


def stage(root, version, authorization):
    from agenticgen.adaptive_schedule.group_protocol import configuration
    from agenticgen.adaptive_schedule.group_run import SKILLS
    from agenticgen.adaptive_schedule.wave_retention import verify_snapshot
    root = Path(root).resolve(); base = safe_base(root, version)
    with idle_locks(root):
        no_worker(root); assert_committed_runtime()
        if (root/f'FINALIZED_{version}.json').exists():raise ValueError('Finalized version cannot be reset')
        if any(int(p.parent.name[1:]) > int(version[1:]) for p in (root/'versions').glob('v*/binding.json')):
            raise ValueError('Later formal version exists')
        auth_ref = committed(authorization); auth = read(authorization)
        citation_repair = auth.get('repair_kind') == 'private_evidence_alias_v1'
        interface_repair = auth.get('repair_kind') == 'bounded_interface_repair_v1'
        commitment_repair = auth.get('repair_kind') == 'prior_selected_schedules_v1'
        budget_pair = (auth.get('previous_max_output_tokens',32768),auth.get('max_output_tokens'))
        allowed_budget = budget_pair == (98304,98304) if (citation_repair or interface_repair or commitment_repair) else budget_pair in {(32768,65536),(65536,98304)}
        if not (auth.get('authority') == 'human_user' and auth.get('version') == version and
                auth.get('rerun_all_five_rounds') is True and auth.get('delete_failed_candidate_outputs') is True and
                auth.get('preserve_completed_teachers') is True and
                allowed_budget):
            raise ValueError('Authorization does not cover this teacher restart')
        state = read(base/'pipeline.json'); teachers = {t['teacher_id'] for t in read(root/'teachers.json')}
        if set(state['teachers']) != teachers or any(t['phase'] not in {'failed','complete'} for t in state['teachers'].values()):
            raise ValueError('Both teachers must be terminal')
        failed = sorted(t for t, s in state['teachers'].items() if s['phase'] == 'failed')
        kept = sorted(teachers-set(failed))
        if not failed or failed != sorted(auth.get('failed_teachers',[])):raise ValueError('Failure scope differs from authorization')
        old_dir = protocol_dir(root); old = read(old_dir/'PREPARED.json'); previous = read(old_dir/'config.json')
        binding = read(base/'binding.json')
        for frozen in [old, read(old_dir/'ACCEPTANCE_COMPLETE.json'), binding]:
            if frozen['config_sha256'] != sha(old_dir/'config.json') or frozen['source_fingerprint'] != old['source_fingerprint']:
                raise ValueError('Old source/config binding changed')
        if old['teachers_sha256'] != sha(root/'teachers.json'):raise ValueError('Teacher manifest changed')
        if old.get('budget_extension_operation'):
            prior = read(root/old['budget_extension_operation']/'journal.json')
            if prior['phase'] != 'accepted_for_review_check' or root/prior['new_protocol'] != old_dir:
                raise ValueError('Prior budget adoption incomplete')
            verify_protected(root, prior['protected_sha256'])
        if old.get('strategy_review_operation'):
            prior = read(root/old['strategy_review_operation']/'journal.json')
            if (prior['phase'] != 'accepted_for_review_check' or root/prior['new_protocol'] != old_dir
                    or prior['commit'] != old['commit']
                    or prior['acceptance_sha256'] != sha(old_dir/'ACCEPTANCE_COMPLETE.json')
                    or old.get('master_review_policy',{}).get('authorization') != prior['authorization']):
                raise ValueError('Prior best-round policy adoption incomplete')
            verify_protected(root, prior['protected_sha256'])
        if old.get('teacher_restart_operation'):
            prior = read(root/old['teacher_restart_operation']/'journal.json')
            if (prior['phase'] != 'deleted_ready_for_new_approval' or root/prior['new_protocol'] != old_dir
                    or prior['commit'] != old['commit']
                    or prior['new_acceptance_sha256'] != sha(old_dir/'ACCEPTANCE_COMPLETE.json')):
                raise ValueError('Prior failed-teacher restart incomplete')
            verify_protected(root, prior['protected_sha256'])
        if citation_repair and (read(base/'deployment.json').get('commit') != auth.get('previous_commit')
                               or old.get('commit') != auth.get('previous_commit')):
            raise ValueError('Citation repair must bind the actual failed fixed commit')
        if (interface_repair or commitment_repair) and read(base/'deployment.json').get('commit') != auth.get('previous_commit'):
            # A normal version can reuse a compatible older protocol. Its actual
            # deployment is bound separately; source/config equality is checked above.
            raise ValueError('Interface repair must bind the actual failed fixed commit')
        if commitment_repair:
            for teacher in failed:
                failure = state['teachers'][teacher]
                receipt = base/teacher/f"round_{failure['round']:02d}"/'council/insufficient_unexecuted_proposals.json'
                if (failure['round'] != auth.get('failed_round') or
                        sha(receipt) != auth.get('shortage_receipt_sha256') or
                        'Fewer than four unexecuted proposed schedules' not in str(failure.get('error'))):
                    raise ValueError('Committed-wave repair requires the exact diagnosed shortage')
        current = source_fingerprint()
        changed = sorted(k for k in current.keys()|old['source_fingerprint'].keys() if current.get(k) != old['source_fingerprint'].get(k))
        allowed = COMMITMENT_SOURCE_CHANGES if commitment_repair else INTERFACE_SOURCE_CHANGES if interface_repair else CITATION_SOURCE_CHANGES if citation_repair else ALLOWED_SOURCE_CHANGES
        if not changed or not set(changed) <= allowed:raise ValueError('Unsupported source/science change')
        for role, relative in SKILLS.items():
            if sha(repository()/relative) != binding['skill_hashes'][role]:raise ValueError('Frozen skills changed')
        new = copy.deepcopy(previous); new['llm']['max_output_tokens'] = auth['max_output_tokens']
        if previous['llm']['max_output_tokens'] != auth.get('previous_max_output_tokens',32768) or configuration() != new:
            raise ValueError('Only authorized output budget may change')
        initial = {}
        for teacher in teachers:
            boot = read(base/teacher/'bootstrap/hypotheses.json')
            initial[teacher] = {'teacher_id':teacher, 'observed_round':0,
                'jobs':[{'ticks':j['ticks'],'source_proposal_id':'bootstrap:'+j['id'],'proposal':j} for j in boot['jobs']]}
        if state['binding']['initial_jobs_sha256'] != value_sha(initial):raise ValueError('Original bootstrap provenance differs')
        for teacher in kept:
            if state['teachers'][teacher]['round'] != previous['rounds']:raise ValueError('Completed teacher round differs')
            completed_barriers(base, teacher, previous)
        all_files = inventory(base)
        removed = {n:m for n,m in all_files.items() if n in GLOBAL_RESET or Path(n).parts[0] in failed}
        protected = {}
        for name in ['protocols','shared_r0','baselines','versions']:
            for n, meta in inventory(root/name).items():
                rel = name+'/'+n
                if name == 'versions' and n.startswith(version+'/') and n[len(version)+1:] in removed:continue
                protected[rel] = meta['sha256']
        for p in list(root.glob('FINALIZED_*.json')) + [root/n for n in ['config.json','PREPARED.json',
                'ACCEPTANCE_COMPLETE.json','teachers.json','historical_references.json']]:
            protected[p.relative_to(root).as_posix()] = sha(p)
        for folder in (root/'operations').glob('*/failure_audit'):
            protected.update({folder.relative_to(root).as_posix()+'/'+n:m['sha256'] for n,m in inventory(folder).items()})
        for teacher in teachers:verify_snapshot(root/'shared_r0'/teacher)
        for name,digest in read(root/'historical_references.json').items():
            if sha(name) != digest:raise ValueError('Historical input changed')
        op = root/'operations'/f'failed_teacher_rerun_{version}_{commit()[:8]}'
        folder = root/'protocols'/f'analysis_output_repair_{version}_{commit()[:8]}'
        if op.exists() or folder.exists():raise ValueError('Restart operation exists; inspect journal')
        write(op/'journal.json', {'phase':'staged','version':version,'commit':commit(),'authorization':auth_ref,
            'old_protocol':old_dir.relative_to(root).as_posix(),'new_protocol':folder.relative_to(root).as_posix(),
            'old_pointer':read(root/'ACTIVE_PROTOCOL.json'),'source_changes':changed,'protected_sha256':protected,
            'failed_attempt_files':removed,'failed_teachers':failed,'completed_teachers':kept,
            'completed_states':{t:state['teachers'][t] for t in kept},'preserved_initial_jobs':{t:initial[t] for t in kept},
            'completed_events':[e for e in state.get('events',[]) if e.get('teacher') in kept],
            'old_binding':binding,'old_deployment':read(base/'deployment.json'),'rerun_rounds':5,
            'repair_kind':auth.get('repair_kind','output_budget'),
            'failed_candidate_count':sum(read(p)['attempts'] for t in failed for p in (base/t).glob('round_*/wave/COMPLETE.json'))})
        write(folder/'config.json',new)
        prepared = {**old,'config_sha256':sha(folder/'config.json'),'source_fingerprint':current,'commit':commit(),
            'previous_protocol':old_dir.relative_to(root).as_posix(),'previous_protocol_prepared_sha256':sha(old_dir/'PREPARED.json'),
            'teacher_restart_operation':op.relative_to(root).as_posix(),'native_probe_directory':f'operations/teacher_restart_acceptance_{commit()[:8]}',
            'requires_native_acceptance':True,'r0_resampling_allowed':False,
            'authorized_analysis_budget_changes':{'max_output_tokens':auth['max_output_tokens']}}
        if citation_repair:
            prepared['authorized_analysis_budget_changes'] = {}
            prepared['citation_protocol'] = 'private-evidence-alias-v1'
        if interface_repair:
            prepared['authorized_analysis_budget_changes'] = {}
            prepared['interface_repair_policy'] = {'name':'bounded_interface_repair_v1',
                'private_max_format_repairs':2, 'format_policy':'json-interface-v5'}
        if commitment_repair:
            prepared['authorized_analysis_budget_changes'] = {}
            prepared['committed_schedule_context'] = 'prior_selected_schedules_v1'
            prepared['requires_native_acceptance'] = False
            prepared['native_acceptance_reuse_required'] = True
            prepared['native_probe_directory'] = old.get('native_probe_directory',
                'operations/fixed_start_acceptance_'+old['commit'][:8])
        # These completed predecessor transactions remain in the protected history;
        # the new native acceptance is checked by the scoped restart gate below.
        for key in ['budget_extension_operation','failure_restart_operation','strategy_review_operation']:
            if key in prepared:prepared['previous_'+key] = prepared.pop(key)
        write(folder/'PREPARED.json',prepared)
        verify_protected(root,protected)
        write(root/'ACTIVE_PROTOCOL.json',{'directory':folder.relative_to(root).as_posix(),'prepared_sha256':sha(folder/'PREPARED.json')})
        return {'phase':'staged','operation':str(op),'failed_teachers':failed,'preserved_teachers':kept,'outputs_deleted':False}


def native_ready(root):
    folder=protocol_dir(root); p=read(prepared_path(root)); a=read(acceptance_path(root)); probe=root/p['native_probe_directory']
    if p.get('native_acceptance_reuse_required'):
        verify_native_reuse(root, p, a)
    if (a['source_fingerprint'] != source_fingerprint() or a['config_sha256'] != sha(config_path(root)) or
            a.get('teachers') != 2 or a.get('recovery_checks') != 2 or a.get('common_randomness') is not True or
            a.get('different_hypothesis_schedules') is not True or a.get('affinity_vjp') is not False or
            a['results_sha256'] != sha(probe/'results.json') or
            a['gradient_checks_sha256'] != sha(folder/'validation/gradient_checks.json') or
            read(folder/'validation/gradient_checks.json').get('passed') is not True):
        raise ValueError('Fresh native acceptance required')


def verify_native_reuse(root, prepared, accepted):
    """Narrow context repair only: retain original native evidence and require real scoped tests."""
    root = Path(root)
    old_folder = (root/prepared['previous_protocol']).resolve()
    if not old_folder.is_relative_to(root.resolve()/'protocols'):
        raise ValueError('Native source escapes protocol history')
    old = read(old_folder/'ACCEPTANCE_COMPLETE.json')
    reuse = accepted.get('native_evidence_reuse', {})
    if (reuse.get('source_acceptance_sha256') != sha(old_folder/'ACCEPTANCE_COMPLETE.json') or
            reuse.get('source_prepared_sha256') != sha(old_folder/'PREPARED.json') or
            old['source_fingerprint'] != read(old_folder/'PREPARED.json')['source_fingerprint'] or
            old['config_sha256'] != sha(config_path(root))):
        raise ValueError('Native reuse source/config changed')
    current = source_fingerprint()
    changed = {k for k in current.keys()|old['source_fingerprint'].keys()
               if current.get(k) != old['source_fingerprint'].get(k)}
    if not changed or not changed <= COMMITMENT_SOURCE_CHANGES:
        raise ValueError('Native reuse cannot cover sampling, gradient, scoring or configuration changes')
    if reuse.get('source_changes') != sorted(changed):raise ValueError('Native compatibility scope changed')
    test = Path(reuse.get('tests_receipt', ''))
    expected = Path('/opt/MolSteer/operations')/('agenticgen_deploy_'+prepared['commit'][:8])/'tests_receipt.json'
    if test != expected or sha(test) != reuse.get('tests_receipt_sha256'):
        raise ValueError('Scoped remote test receipt changed')
    check = read(test)
    if check.get('commit') != prepared['commit'] or check.get('passed') is not True or check.get('scope') != 'prior_selected_schedules_v1':
        raise ValueError('Scoped remote contract/transaction checks required')
    # Only source identity and explicit reuse provenance change; evidence remains exact.
    if {k:v for k,v in accepted.items() if k not in ('source_fingerprint','native_evidence_reuse')} != {k:v for k,v in old.items() if k != 'source_fingerprint'}:
        raise ValueError('Native evidence was changed during reuse')


def reuse_native_acceptance(root, tests_receipt):
    root=Path(root); p=read(prepared_path(root)); old=root/p['previous_protocol']
    if not p.get('native_acceptance_reuse_required'):raise ValueError('This repair does not authorize native reuse')
    accepted=read(old/'ACCEPTANCE_COMPLETE.json')
    current=source_fingerprint()
    changed=sorted(k for k in current.keys()|accepted['source_fingerprint'].keys()
                   if current.get(k)!=accepted['source_fingerprint'].get(k))
    accepted.update(source_fingerprint=current, native_evidence_reuse={
        'source_acceptance_sha256':sha(old/'ACCEPTANCE_COMPLETE.json'),
        'source_prepared_sha256':sha(old/'PREPARED.json'),'source_changes':changed,
        'tests_receipt':str(tests_receipt),'tests_receipt_sha256':sha(tests_receipt),
        'fresh_native_sampling':False,'scope':'context eligibility; native sampling/gradient/quality code unchanged'})
    verify_native_reuse(root,p,accepted)
    dest=protocol_dir(root)/'validation/gradient_checks.json';dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(old/'validation/gradient_checks.json',dest)
    write(acceptance_path(root),accepted)
    native_ready(root)
    return {'fresh_native_sampling':False,'source_acceptance_sha256':sha(old/'ACCEPTANCE_COMPLETE.json'),
            'acceptance_sha256':sha(acceptance_path(root))}


def operation_journal(root):
    p=prepared_path(root)
    if not p.exists() or not read(p).get('teacher_restart_operation'):return None
    op=(root/read(p)['teacher_restart_operation']).resolve()
    if op.parent != root/'operations' or op.is_symlink():raise ValueError('Unsafe restart operation')
    return op,read(op/'journal.json')


def finish(root, operation):
    root=Path(root).resolve(); op=Path(operation).resolve()
    if op.parent != root/'operations' or op.is_symlink():raise ValueError('Unsafe restart operation')
    with idle_locks(root):
        no_worker(root); assert_committed_runtime(); j=read(op/'journal.json'); base=safe_base(root,j['version'])
        if j['commit'] != commit() or root/j['new_protocol'] != protocol_dir(root):raise ValueError('Use staged fixed runtime')
        if (root/f"FINALIZED_{j['version']}.json").exists():raise ValueError('Version finalized during restart')
        native_ready(root); verify_protected(root,j['protected_sha256'])
        if j['phase'] == 'deleted_ready_for_new_approval':return {'phase':j['phase'],'operation':str(op)}
        expected=j['failed_attempt_files']; archive=op/'failure_audit'
        if j['phase'] == 'staged':
            for n,m in expected.items():
                if sha(base/n) != m['sha256']:raise ValueError('Failed source changed')
                if audit_document(n):
                    dest=archive/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(base/n,dest)
                    if sha(dest) != m['sha256']:raise ValueError('Audit copy differs')
            write(op/'deletion_manifest.json',{'files':expected,'audit_copies':[n for n in expected if audit_document(n)],
                'result_rows_excluded':j['failed_candidate_count'],'failed_teachers':j['failed_teachers'],
                'completed_teachers_preserved':j['completed_teachers'],'new_results_start_at_round':1})
            j['phase']='audit_verified_ready_to_delete';write(op/'journal.json',j)
        if j['phase'] != 'audit_verified_ready_to_delete':raise ValueError('Unknown restart phase')
        for n in expected:
            if audit_document(n) and sha(archive/n) != expected[n]['sha256']:raise ValueError('Archived evidence changed')
        remaining=inventory(base)
        for n,m in remaining.items():
            if (n in GLOBAL_RESET or Path(n).parts[0] in j['failed_teachers']) and expected.get(n) != m:
                raise ValueError('Unknown/changed candidate content before deletion')
        for n,m in expected.items():
            p=base/n
            if p.exists():
                if not p.resolve().is_relative_to(base) or p.is_symlink() or sha(p) != m['sha256']:
                    raise ValueError('Unsafe deletion target')
                p.unlink()
        for teacher in j['failed_teachers']:
            tb=base/teacher
            if tb.exists():
                for p in sorted((p for p in tb.rglob('*') if p.is_dir()),key=lambda p:len(p.parts),reverse=True):p.rmdir()
                tb.rmdir()
        verify_protected(root,j['protected_sha256'])
        j.update(phase='deleted_ready_for_new_approval',deleted_files=len(expected),candidate_outputs_deleted=True,
                 new_acceptance_sha256=sha(acceptance_path(root)))
        write(op/'journal.json',j)
        write(root/'status.json',{'phase':'awaiting_rerun_approval','version':j['version'],'operation':str(op)})
        return {'phase':j['phase'],'operation':str(op),'deleted_candidate_count':j['failed_candidate_count'],
                'preserved_teachers':j['completed_teachers'],'rerun_rounds':5}


def verify_ready(root, version):
    root=Path(root).resolve(); found=operation_journal(root)
    if not found:return None
    op,j=found; p=read(prepared_path(root))
    if (j['phase'] != 'deleted_ready_for_new_approval' or root/j['new_protocol'] != protocol_dir(root) or
            j['commit'] != p['commit'] or p['source_fingerprint'] != source_fingerprint() or
            version not in read(config_path(root))['versions'] or int(version[1:]) < int(j['version'][1:]) or
            j['new_acceptance_sha256'] != sha(acceptance_path(root))):
        raise ValueError('Teacher restart/native gate not complete')
    previous = read(root/j['old_protocol']/'PREPARED.json')
    if p.get('master_review_policy') != previous.get('master_review_policy'):
        raise ValueError('Restart must preserve the adopted review scope')
    native_ready(root); verify_protected(root,j['protected_sha256'])
    return j


def preserved_provenance(root, version):
    root=Path(root).resolve(); found=operation_journal(root)
    if not found or found[1]['version'] != version:return {}
    j=verify_ready(root,version)
    earlier = j['old_binding'].get('preserved_teacher_provenance',{})
    return {t:copy.deepcopy(earlier[t]) if t in earlier else {'binding':j['old_binding'],'deployment':j['old_deployment'],
        'reason':'Completed five-round teacher preserved byte-identically; only failed teacher reruns',
        'files_sha256':{n:d for n,d in j['protected_sha256'].items() if n.startswith(f'versions/{version}/{t}/')}}
        for t in j['completed_teachers']}


def preserved_jobs(root, version):
    root=Path(root).resolve(); found=operation_journal(root)
    if not found or found[1]['version'] != version:return {}
    return copy.deepcopy(verify_ready(root,version)['preserved_initial_jobs'])


def seed_pipeline(root, version, initial_jobs, rounds):
    root=Path(root).resolve(); found=operation_journal(root)
    if not found or found[1]['version'] != version:return
    j=verify_ready(root,version); path=safe_base(root,version)/'pipeline.json'
    if path.exists():return  # scheduler validates frozen initial jobs on every resume
    binding={'initial_jobs_sha256':value_sha(initial_jobs),'rounds':rounds,'teachers':list(initial_jobs),
             'gpu_owners':1,'global_llm_concurrency':4}
    teachers={t:copy.deepcopy(j['completed_states'][t]) if t in j['completed_states'] else
              {'round':1,'phase':'ready','job':job} for t,job in initial_jobs.items()}
    write(path,{'binding':binding,'teachers':teachers,'events':copy.deepcopy(j['completed_events']), 'last_gpu_teacher':None,
                'preserved_teacher_operation':found[0].relative_to(root).as_posix()})
