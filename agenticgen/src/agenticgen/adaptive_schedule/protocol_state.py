"""Append-only protocol adoption before V6; preserve original R0 provenance."""
from pathlib import Path
from .io import read, write, sha, value_sha, commit, source_fingerprint


def protocol_dir(root):
    root=Path(root); pointer=root/'ACTIVE_PROTOCOL.json'
    if not pointer.exists(): return root
    marker=read(pointer); folder=(root/marker['directory']).resolve()
    if not folder.is_relative_to(root.resolve()) or sha(folder/'PREPARED.json') != marker['prepared_sha256']:
        raise ValueError('Protocol pointer escaped root or preparation changed')
    return folder


def config_path(root): return protocol_dir(root)/'config.json'
def load_config(root): return read(config_path(root))
def prepared_path(root): return protocol_dir(root)/'PREPARED.json'
def acceptance_path(root): return protocol_dir(root)/'ACCEPTANCE_COMPLETE.json'


def stage_protocol(root, reason):
    """No sampled scientific versions may exist. New native acceptance is required."""
    from .group_protocol import configuration
    from agenticgen.runtime.locks import lock
    root=Path(root)
    if not reason or len(reason.strip()) < 30: raise ValueError('Concrete migration rationale required')
    with lock(root):
        if (root/'ACTIVE_PROTOCOL.json').exists():
            raise ValueError('Protocol already staged; inspect provenance instead of repeating')
        if any((root/'versions').glob('*/binding.json')):
            raise ValueError('Cannot change protocol after a scientific version has started')
        old=read(root/'config.json'); prepared=read(root/'PREPARED.json')
        if prepared['config_sha256'] != sha(root/'config.json') or prepared['teachers_sha256'] != sha(root/'teachers.json'):
            raise ValueError('Original prepared inputs changed')
        prior_acceptance=read(root/'ACCEPTANCE_COMPLETE.json')
        if prior_acceptance.get('config_sha256') != prepared['config_sha256']:
            raise ValueError('Original native acceptance does not bind the original configuration')
        new=configuration()
        allowed={'schema','versions','total_result_budget','condition_comparison','search_policy','rollback'}
        changed={k for k in old.keys()|new.keys() if old.get(k) != new.get(k)}
        if not changed <= allowed: raise ValueError('Migration would alter sampling/R0 settings: '+str(changed-allowed))
        from .wave_retention import verify_snapshot
        r0={}
        for teacher in read(root/'teachers.json'):
            folder=root/'shared_r0'/teacher['teacher_id']
            verify_snapshot(folder)
            r0[teacher['teacher_id']]=sha(folder/'COMPLETE.json')
        folder=root/'protocols'/'continuous_rmsd_v2'
        if folder.exists(): raise ValueError('Migration directory exists; inspect incomplete transaction')
        write(folder/'config.json',new)
        record={'config_sha256':sha(folder/'config.json'),'teachers_sha256':sha(root/'teachers.json'),
            'source_fingerprint':source_fingerprint(),'commit':commit(),'reason':reason,
            'previous_prepared_sha256':sha(root/'PREPARED.json'),
            'previous_config_sha256':sha(root/'config.json'),
            'previous_acceptance_sha256':sha(root/'ACCEPTANCE_COMPLETE.json'),
            'shared_r0_sha256':r0, 'changed_config_keys':sorted(changed),
            'requires_native_acceptance':True,'r0_resampling_allowed':False}
        write(folder/'PREPARED.json',record)
        write(root/'ACTIVE_PROTOCOL.json',{'directory':folder.relative_to(root).as_posix(),
            'prepared_sha256':sha(folder/'PREPARED.json')})
        return record


def repair_preflight(root, reason):
    """Append-only diagnostic repair before any new-protocol scientific binding.

    Only the diagnostic/administrative modules listed below may differ. Sampling,
    reward, RNG, scoring, configuration and skills are not adopted through here.
    """
    from agenticgen.runtime.locks import lock
    from .wave_retention import verify_snapshot
    from .group_run import SKILLS
    from .io import repository
    import subprocess
    root=Path(root)
    if not reason or len(reason.strip()) < 30:
        raise ValueError('Concrete preflight repair rationale required')
    allowed={'adaptive_schedule/teacher_evidence.py','adaptive_schedule/wave_retention.py',
             'adaptive_schedule/protocol_state.py','adaptive_schedule/group_run.py',
             'adaptive_schedule/group_validation.py','cli.py'}
    with lock(root):
        old_folder=protocol_dir(root)
        if old_folder == root or acceptance_path(root).exists():
            raise ValueError('Only an unaccepted staged protocol may be repaired')
        if any((root/'versions').glob('**/binding.json')):
            raise ValueError('Scientific version already started')
        old=read(old_folder/'PREPARED.json'); current=source_fingerprint()
        changed={k for k in old['source_fingerprint'].keys()|current.keys()
                 if old['source_fingerprint'].get(k)!=current.get(k)}
        if not changed or not changed <= allowed:
            raise ValueError('Preflight repair includes non-diagnostic source changes: '+str(changed-allowed))
        for path in SKILLS.values():
            previous=subprocess.check_output(['git','-C',str(repository()),'show',old['commit']+':'+path])
            if previous != (repository()/path).read_bytes():
                raise ValueError('Frozen controller skills changed during preflight repair')
        if old['config_sha256']!=sha(old_folder/'config.json') or old['teachers_sha256']!=sha(root/'teachers.json'):
            raise ValueError('Prepared scientific inputs changed')
        refs=read(root/'historical_references.json')
        for name,digest in refs.items():
            if sha(name)!=digest: raise ValueError('Historical file changed: '+name)
        for teacher,digest in old['shared_r0_sha256'].items():
            r0=root/'shared_r0'/teacher
            verify_snapshot(r0)
            if sha(r0/'COMPLETE.json')!=digest: raise ValueError('Shared R0 changed')
        probe=root/old.get('native_probe_directory', 'operations/fixed_start_acceptance_'+old['commit'][:8])
        if not probe.resolve().is_relative_to(root.resolve()) or list(probe.glob('*/ACCEPTED.json')):
            raise ValueError('Cannot reuse accepted/pruned or external preflight probes')
        folder=root/'protocols'/('continuous_rmsd_v2_repair_'+commit()[:8])
        if folder.exists(): raise ValueError('Repair exists; inspect incomplete transaction')
        write(folder/'config.json',read(old_folder/'config.json'))
        updated={**old,'source_fingerprint':current,'commit':commit(),'reason':reason,
            'previous_protocol':old_folder.relative_to(root).as_posix(),
            'previous_protocol_prepared_sha256':sha(old_folder/'PREPARED.json'),
            'diagnostic_source_changes':sorted(changed),
            'historical_references_sha256':sha(root/'historical_references.json'),
            'native_probe_directory':probe.relative_to(root).as_posix(),
            'native_probe_files_before_repair':{p.relative_to(probe).as_posix():sha(p)
                for p in sorted(probe.rglob('*')) if p.is_file()},
            'requires_native_acceptance':True,'scientific_sampling_started':False}
        write(folder/'PREPARED.json',updated)
        write(root/'ACTIVE_PROTOCOL.json',{'directory':folder.relative_to(root).as_posix(),
            'prepared_sha256':sha(folder/'PREPARED.json')})
        return updated
