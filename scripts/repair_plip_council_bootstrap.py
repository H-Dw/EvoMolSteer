"""Audited pre-R1 adoption for the recorded diagnostics/action-contract repair.

No model, sampler, reward, random stream, skill or scientific setting may change.
Completed R0 and previous requests are preserved and checked before/after adoption.
"""
import argparse
from pathlib import Path
import shutil
import subprocess
import time

from evomolsteer.adaptive_schedule.io import read, write, sha, source_fingerprint, repository, commit
from evomolsteer.adaptive_schedule.runner import lock


ALLOWED = {'../deepseek_api.py', 'group_agent.py'}


def inventory(root):
    paths = list((root/'shared_r0').rglob('*')) + list((root/'baselines').glob('*.json'))
    paths += [root/name for name in ['config.json', 'teachers.json', 'historical_references.json']]
    return {p.relative_to(root).as_posix(): sha(p) for p in paths if p.is_file()}


def check(root, test_receipt):
    prepared, accepted = read(root/'PREPARED.json'), read(root/'ACCEPTANCE_COMPLETE.json')
    new = source_fingerprint(); old = prepared['source_fingerprint']
    changed = {k for k in old.keys() | new.keys() if old.get(k) != new.get(k)}
    if changed != ALLOWED:
        raise ValueError('This repair permits exactly the two reviewed interface files')
    if accepted['source_fingerprint'] != old or accepted['recovery_checks'] != 8:
        raise ValueError('Original native acceptance is inconsistent')
    if (prepared['config_sha256'] != sha(root/'config.json') or
            prepared['teachers_sha256'] != sha(root/'teachers.json')):
        raise ValueError('Scientific configuration changed')
    base = root/'versions/v1'
    if (base/'pipeline.json').exists() or list(base.glob('*/round_*/job.json')):
        raise ValueError('This repair is restricted to the recorded failure before R1')
    if list(base.glob('*/bootstrap/hypotheses.json')) or list((root/'versions').glob('v[2-9]*')):
        raise ValueError('Already selected hypotheses or later versions require another audit')
    binding = read(base/'binding.json')
    if binding['source_fingerprint'] != old:
        raise ValueError('Frozen version binding differs')
    from evomolsteer.adaptive_schedule.group_run import SKILLS
    for role, relative in SKILLS.items():
        if sha(base/f'{role}_SKILL.md') != binding['skill_hashes'][role] or sha(repository()/relative) != binding['skill_hashes'][role]:
            raise ValueError('Frozen optimizer skill changed')
    for teacher in read(root/'teachers.json'):
        r0 = root/'shared_r0'/teacher['teacher_id']
        if read(r0/'COMPLETE.json')['attempts'] != 100 or read(r0/'QUALITY_COMPLETE.json')['attempts'] != 100:
            raise ValueError('Both completed R0 populations are required')
        if sha(r0/'quality_records.json') != read(r0/'QUALITY_COMPLETE.json')['records_sha256']:
            raise ValueError('R0 quality changed')
    tests = read(test_receipt)
    if tests['exit_code'] != 0 or tests['commit'] != commit() or sha(tests['log']) != tests['log_sha256']:
        raise ValueError('Passed tests must bind this exact deployed revision')
    return prepared, accepted, binding, new


def repair(root, test_receipt, *, apply=False):
    root, test_receipt = Path(root), Path(test_receipt)
    with lock(root):
        prepared, accepted, binding, new = check(root, test_receipt)
        if not apply:
            return {'eligible': True, 'changed_sources': sorted(ALLOWED), 'apply': False}
        op = root/'operations'/f'bootstrap_interface_repair_{time.time_ns()}'
        before = inventory(root)
        journal = {'commit': commit(), 'time': time.time(), 'changed_sources': sorted(ALLOWED),
            'reason': 'Serialize jsonschema Unset diagnostics and expose already enforced action bounds to the LLM.',
            'scientific_code_unchanged': True, 'skills_unchanged': True,
            'prior_prepared': prepared, 'prior_acceptance': accepted, 'prior_binding': binding,
            'test_receipt_sha256': sha(test_receipt), 'protected_files_sha256': before,
            'phase': 'prepared'}
        write(op/'journal.json', journal)
        old_revision = read(root/'versions/v1/deployment.json')['commit']
        diff = subprocess.check_output(['git', '-C', str(repository()), 'diff', old_revision, commit(), '--',
            'src/evomolsteer/deepseek_api.py', 'src/evomolsteer/adaptive_schedule/group_agent.py'])
        (op/'reviewed_interface_diff.patch').write_bytes(diff)
        archives = []
        for bootstrap in (root/'versions/v1').glob('*/bootstrap'):
            archive = bootstrap/'saved_attempts'/op.name
            children = [p for p in bootstrap.iterdir() if p.name != 'saved_attempts']
            archive.mkdir(parents=True, exist_ok=True)
            for p in children:
                # All paths are fixed children within this known result root.
                if not p.resolve().is_relative_to(root.resolve()):
                    raise ValueError('Bootstrap archive escaped result root')
                shutil.move(str(p), str(archive/p.name))
            archives.append(str(archive))
        write(root/'PREPARED.json', dict(prepared, source_fingerprint=new, commit=commit()))
        write(root/'ACCEPTANCE_COMPLETE.json', dict(accepted, source_fingerprint=new,
            compatibility_audit=str(op/'journal.json')))
        write(root/'versions/v1/binding.json', dict(binding, source_fingerprint=new))
        if inventory(root) != before:
            raise ValueError('Protected R0 or historical reference changed')
        journal.update(phase='complete', archives=archives, protected_hashes_equal=True,
            native_acceptance_reused_because='Every native sampling, reward, RNG and retention source hash is unchanged.')
        write(op/'journal.json', journal)
        return {'operation': str(op), 'protected_hashes_equal': True, 'archives': archives}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--test-receipt', required=True, type=Path)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    print(repair(args.root, args.test_receipt, apply=args.apply))
