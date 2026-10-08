"""Recover omitted report copies from an already published experiment ledger.

This changes storage only: it cannot create a new implementation result. The
existing complete ledger record and all retained file hashes must first match.
"""
import argparse
import hashlib
import subprocess
from pathlib import Path
from evomolsteer.io import digest, read_json, write_json
from dispatch_path_round import verify_retention


def recover(repo, ledger, output_dataset):
    repo, ledger, output = map(lambda p: Path(p).resolve(), (repo, ledger, output_dataset))
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    relative = ledger.relative_to(repo).as_posix()
    blob = subprocess.check_output(['git', '-C', str(repo), 'show', commit + ':' + relative])
    if hashlib.sha256(blob).hexdigest() != digest(ledger):
        raise ValueError('Recovery requires an unchanged, published ledger version')
    recovered = []
    for entry in read_json(ledger)['rounds']:
        if entry['status'] != 'complete':
            continue
        folder = output / f"round{entry['round']:02d}"
        target, manifest = folder / 'implementation_feedback.json', folder / 'retention.json'
        if target.exists():
            if read_json(target) != entry['implementation_feedback']:
                raise ValueError('Existing implementation artifact differs from its measured ledger record')
            continue
        verify_retention(manifest)
        write_json(target, entry['implementation_feedback'])
        recovery = folder / 'feedback_storage_recovery.json'
        write_json(recovery, {'storage_only': True, 'measured_before_efficacy': True,
                             'source_ledger': relative, 'source_ledger_sha256': digest(ledger),
                             'source_commit': commit, 'new_feedback_sha256': digest(target),
                             'reason': 'Copy measured ledger result omitted from standalone report retention; no numerical rerun or new gate result'})
        retained = read_json(manifest)
        for path in [target, recovery]:
            retained['files'].append({'path': path.name, 'sha256': digest(path), 'bytes': path.stat().st_size})
        write_json(manifest, retained)
        verify_retention(manifest)
        recovered.append(entry['round'])
    return {'recovered_rounds': recovered}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True)
    parser.add_argument('--input-dataset', required=True, help='Published complete-round campaign JSON')
    parser.add_argument('--output-dataset', required=True, help='Existing retained round-report directory')
    args = parser.parse_args()
    print(recover(args.repo, args.input_dataset, args.output_dataset))
