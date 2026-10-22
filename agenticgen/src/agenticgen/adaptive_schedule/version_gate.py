"""Enforce reviewed range changes and historical-best rollback, without approving them."""
from .io import read, sha, value_sha
from .search_policy import load_version_policy, policy
from .selection import rollback_advice


def version_metrics(root):
    """Read finalized immutable reports only; report refresh is not a prerequisite."""
    from statistics import fmean
    config=read_config(root); rows=[]
    for version in config['versions']:
        finalized=root/f'FINALIZED_{version}.json'
        if not finalized.exists(): continue
        base=root/'versions'/version; path=base/'report.json'
        if sha(path) != read(finalized)['report_sha256']: raise ValueError('Finalized report changed')
        report=read(path)
        selected=[x['primary']['selected'] for x in report.get('representatives',{}).values()]
        full=(len(selected)==len(read(root/'teachers.json')) and all(selected)
              and report['counts']['optimization_attempts']==len(selected)*config['rounds']*4*config['population_size'])
        approval=read(base/'review_approval.json')
        rows.append(dict(id=version,condition=fmean(x['condition'] for x in selected) if full else None,
            functional_gap=fmean(x['functional_gap'] for x in selected) if full else None,
            best=fmean(x['best'] for x in selected) if full else None,
            rollback_completed=bool(approval.get('rollback',{}).get('restored_version'))))
    return rows


def read_config(root):
    from .protocol_state import load_config
    return load_config(root)


def check_control_gate(root, version, approval):
    from .group_review import committed_file
    from .group_run import SKILLS
    from .io import commit
    active=load_version_policy(root,version)
    if approval.get('search_policy_sha256') != value_sha(active):
        raise ValueError('Approval must bind the exact version range/coarse/fine policy')
    if approval.get('condition_comparison_sha256') != value_sha(read_config(root)['condition_comparison']):
        raise ValueError('Approval must bind the frozen RMSD ranking/alignment policy')
    rows=version_metrics(root)
    advice=rollback_advice(rows)
    if rows and advice['status'] != 'complete': raise ValueError('Incomplete prior performance prevents normal version advance')
    previous=policy() if version=='v6' else load_version_policy(root,f'v{int(version[1:])-1}')
    if active['upper_amplitude'] < previous['upper_amplitude'] and not advice.get('required'):
        raise ValueError('Range shrink must accompany the required historical-state rollback')
    if active['upper_amplitude'] > previous['upper_amplitude']:
        evidence=approval.get('range_expansion',{})
        for key in ('boundary_response','delivered_movement_and_clipping','unresolved_hypothesis'):
            if len(evidence.get(key,'').strip()) < 30: raise ValueError('Range expansion needs substantive '+key)
        committed_file(evidence['document'],evidence['document_sha256'],commit())
    if advice.get('required'):
        restore=advice['restore_version']; old=root/'versions'/restore
        rollback=approval.get('rollback',{})
        if rollback.get('restored_version') != restore or rollback.get('restored_binding_sha256') != sha(old/'binding.json'):
            raise ValueError('Two consecutive regressions require the historical-best version state')
        if active != load_version_policy(root,restore):
            raise ValueError('Rollback must restore range, coarse scale and fine grid together')
        binding=read(old/'binding.json')
        for role,path in SKILLS.items():
            committed_file(path,binding['skill_hashes'][role],commit())
        committed_file(rollback['code_compatibility_document'],rollback['code_compatibility_sha256'],commit())
        if rollback.get('source_binding_sha256') != value_sha(binding['source_fingerprint']):
            raise ValueError('Rollback code compatibility must bind the restored source inventory')
    elif approval.get('rollback'):
        raise ValueError('Do not reset regression history without a required rollback')
    return {'search_policy':active,'rollback':advice}
