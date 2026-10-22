"""Human-authorized full five-round rerun after a tested analysis repair.

Stage an append-only protocol, run native validation, then remove the failed
attempt's candidate outputs. Keep prompts, responses, costs and deletion hashes.
No launch, fabricated review, or scientific proposal is performed by this module.
"""
import copy
import os
from pathlib import Path
import re
import shutil

from agenticgen.adaptive_schedule.io import read, write, sha, commit, source_fingerprint
from agenticgen.adaptive_schedule.protocol_state import protocol_dir, config_path, acceptance_path
from agenticgen.runtime.launch import idle_locks

ALLOWED_SOURCE_CHANGES = {
    'adaptive_schedule/council.py', 'adaptive_schedule/evidence_pack.py',
    'adaptive_schedule/group_agent.py', 'adaptive_schedule/group_protocol.py',
    'adaptive_schedule/learning.py', 'runtime/restart_failed_version.py', 'cli.py'}


def safe_base(root, version):
    root = Path(root).resolve()
    if not re.fullmatch(r'v[1-9][0-9]*', version):
        raise ValueError('Invalid version identity')
    base = root/'versions'/version
    if base.resolve().parent != root/'versions' or base.is_symlink():
        raise ValueError('Version path escaped its exact directory')
    return base


def inventory(folder):
    result = {}
    for p in sorted(folder.rglob('*')):
        if p.is_symlink() or not p.resolve().is_relative_to(folder.resolve()):
            raise ValueError('Symlink/escaped entry in cleanup tree')
        if p.is_file():
            result[p.relative_to(folder).as_posix()] = {'sha256':sha(p), 'bytes':p.stat().st_size}
    return result


def verify_protected(root, expected):
    for name, digest in expected.items():
        if sha(root/name) != digest:
            raise ValueError('Protected artifact changed: '+name)


def no_worker(root):
    for name in ['workers.json', 'launch.json']:
        p = root/name
        if p.exists() and (Path('/proc')/str(read(p)['pid'])).exists():
            raise ValueError('Worker remains alive; stop/check it before full rerun')


def check_authorization(path, version):
    from agenticgen.adaptive_schedule.io import repository
    import subprocess
    path = Path(path).resolve(); repo = repository().resolve()
    if not path.is_relative_to(repo):
        raise ValueError('Authorization must be committed in the deployed repository')
    relative = path.relative_to(repo).as_posix()
    if subprocess.check_output(['git','-C',str(repo),'show',commit()+':'+relative]) != path.read_bytes():
        raise ValueError('Authorization differs from committed source')
    a = read(path)
    if not (a.get('authority') == 'human_user' and a.get('version') == version and
            a.get('rerun_all_five_rounds') is True and a.get('delete_failed_candidate_outputs') is True and
            a.get('max_input_hard_bytes') == 524288):
        raise ValueError('Authorization does not cover this full rerun')
    return {'path':relative, 'sha256':sha(path)}


def stage(root, version, authorization):
    from agenticgen.adaptive_schedule.group_protocol import configuration
    from agenticgen.adaptive_schedule.wave_retention import verify_snapshot
    from agenticgen.adaptive_schedule.group_run import SKILLS
    from agenticgen.adaptive_schedule.io import repository
    root=Path(root).resolve(); base=safe_base(root,version)
    with idle_locks(root):
        no_worker(root)
        if (root/f'FINALIZED_{version}.json').exists():
            raise ValueError('Finalized version cannot be reset')
        auth=check_authorization(authorization,version)
        state=read(base/'pipeline.json')
        if not state['teachers'] or any(t['phase']!='failed' for t in state['teachers'].values()):
            raise ValueError('This full rerun requires both affected teachers to be failed')
        old_dir=protocol_dir(root); old=read(old_dir/'PREPARED.json')
        acceptance=read(old_dir/'ACCEPTANCE_COMPLETE.json'); binding=read(base/'binding.json')
        if old.get('teachers_sha256') != sha(root/'teachers.json'):
            raise ValueError('Frozen teacher manifest changed')
        for frozen in (old,acceptance,binding):
            if frozen['config_sha256']!=sha(old_dir/'config.json') or frozen['source_fingerprint']!=old['source_fingerprint']:
                raise ValueError('Original frozen source/config binding changed')
        current=source_fingerprint()
        changed=sorted(k for k in old['source_fingerprint'].keys()|current.keys() if old['source_fingerprint'].get(k)!=current.get(k))
        if not changed or not set(changed)<=ALLOWED_SOURCE_CHANGES:
            raise ValueError('Repair changed sampling/reward/RNG/quality or another unsupported module')
        for role,relative in SKILLS.items():
            if sha(repository()/relative)!=binding['skill_hashes'][role]:
                raise ValueError('This repair cannot change frozen skills')
        previous=read(old_dir/'config.json'); new=copy.deepcopy(previous)
        new['llm'].update(input_soft_bytes=131072,input_hard_bytes=524288)
        # This also guards against an unrelated source configuration change.
        if configuration()!=new:
            raise ValueError('Only the explicitly authorized input budgets may change')
        protected={p.relative_to(root).as_posix():sha(p) for p in
            [root/n for n in ['config.json','PREPARED.json','ACCEPTANCE_COMPLETE.json','teachers.json','historical_references.json']]
            +[p for p in (root/'protocols').rglob('*') if p.is_file()]}
        for t in read(root/'teachers.json'):
            folder=root/'shared_r0'/t['teacher_id']; verify_snapshot(folder)
            protected.update({p.relative_to(root).as_posix():sha(p) for p in folder.rglob('*') if p.is_file()})
        for name,digest in read(root/'historical_references.json').items():
            if sha(name)!=digest:raise ValueError('Historical input changed')
        op=root/'operations'/f'failed_version_rerun_{version}_{commit()[:8]}'
        folder=root/'protocols'/f'analysis_repair_{version}_{commit()[:8]}'
        if op.exists() or folder.exists():raise ValueError('Rerun operation exists; inspect its journal')
        before=inventory(base)
        write(op/'journal.json',{'phase':'staged','version':version,'commit':commit(),'authorization':auth,
            'old_protocol':old_dir.relative_to(root).as_posix(),'old_pointer':read(root/'ACTIVE_PROTOCOL.json'),
            'new_protocol':folder.relative_to(root).as_posix(),'source_changes':changed,
            'protected_sha256':protected,'failed_attempt_files':before,
            'failed_attempt_excluded_from_scientific_results':True,'rerun_rounds':5,
            'failed_candidate_count':sum(read(p)['attempts'] for p in base.glob('*/round_*/wave/COMPLETE.json'))})
        write(folder/'config.json',new)
        prepared={**old,'config_sha256':sha(folder/'config.json'),'source_fingerprint':current,'commit':commit(),
            'previous_protocol':old_dir.relative_to(root).as_posix(),'previous_protocol_prepared_sha256':sha(old_dir/'PREPARED.json'),
            'failure_restart_operation':op.relative_to(root).as_posix(),'requires_native_acceptance':True,
            'native_probe_directory':f'operations/failed_attempt_acceptance_{commit()[:8]}',
            'r0_resampling_allowed':False,'authorized_analysis_budget_changes':{'input_soft_bytes':131072,'input_hard_bytes':524288}}
        write(folder/'PREPARED.json',prepared)
        write(root/'ACTIVE_PROTOCOL.json',{'directory':folder.relative_to(root).as_posix(),'prepared_sha256':sha(folder/'PREPARED.json')})
        return {'operation':str(op),'phase':'staged','candidate_outputs_deleted':False,'requires_native_validation':True}


def audit_document(name):
    p=Path(name)
    return (p.suffix in {'.json','.txt','.md'} and
            ('council' in p.parts or 'bootstrap' in p.parts or len(p.parts)==1 or p.name=='job.json'))


def finish(root, operation):
    """Resumable deletion after real validation; no old hypotheses enter the rerun."""
    root=Path(root).resolve(); op=Path(operation).resolve()
    if op.parent != root/'operations' or op.is_symlink():raise ValueError('Unsafe operation path')
    with idle_locks(root):
        no_worker(root);j=read(op/'journal.json');base=safe_base(root,j['version'])
        if j['commit']!=commit():raise ValueError('Use the staged fixed commit')
        if j['phase']=='deleted_ready_for_new_approval':return {'phase':j['phase'],'operation':str(op)}
        if (root/f"FINALIZED_{j['version']}.json").exists():raise ValueError('Version finalized after staging')
        if protocol_dir(root)!=root/j['new_protocol']:raise ValueError('Active protocol changed')
        a=read(acceptance_path(root))
        if a['source_fingerprint']!=source_fingerprint() or a['config_sha256']!=sha(config_path(root)) or a.get('teachers')!=2:
            raise ValueError('Fresh native acceptance is required before deleting candidates')
        verify_protected(root,j['protected_sha256'])
        tomb=op/'discarded_version';archive=op/'failure_audit'
        if j['phase']=='staged':
            if inventory(base)!=j['failed_attempt_files']:raise ValueError('Failed outputs changed since staging')
            for name,meta in j['failed_attempt_files'].items():
                if audit_document(name):
                    dest=archive/name;dest.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copyfile(base/name,dest)
                    if sha(dest)!=meta['sha256']:raise ValueError('Failure audit copy verification failed')
            write(op/'deletion_manifest.json',{'files':j['failed_attempt_files'],
                'audit_copies':[n for n in j['failed_attempt_files'] if audit_document(n)],
                'result_rows_excluded':j['failed_candidate_count'],'new_results_start_at_round':1})
            j['phase']='audit_verified_ready_to_delete';write(op/'journal.json',j)
        if base.exists() and tomb.exists():raise ValueError('Ambiguous cleanup state')
        if base.exists():
            if inventory(base)!=j['failed_attempt_files']:raise ValueError('Version changed before deletion')
            os.replace(base,tomb)
        if tomb.exists():
            remaining=inventory(tomb)
            if any(j['failed_attempt_files'].get(k)!=v for k,v in remaining.items()):
                raise ValueError('Unknown content in discard directory')
            if tomb.resolve().parent!=op or tomb.is_symlink():raise ValueError('Unsafe discard directory')
            shutil.rmtree(tomb)
        # All documents live outside versions, so normal report/review globbing
        # cannot count aborted outcomes or inherit its dynamic search history.
        verify_protected(root,j['protected_sha256'])
        j.update(phase='deleted_ready_for_new_approval',candidate_outputs_deleted=True,
            deleted_files=len(j['failed_attempt_files']),shared_r0_preserved=True,
            new_acceptance_sha256=sha(acceptance_path(root)))
        write(op/'journal.json',j)
        write(root/'status.json',{'phase':'awaiting_rerun_approval','version':j['version'],'operation':str(op)})
        return {'phase':j['phase'],'operation':str(op),'deleted_failed_candidate_count':j['failed_candidate_count'],
                'rerun_rounds':5,'shared_r0_preserved':True}
