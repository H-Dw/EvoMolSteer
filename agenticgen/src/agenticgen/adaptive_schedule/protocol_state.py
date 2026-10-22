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
