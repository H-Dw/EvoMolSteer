import datetime
import fcntl
import hashlib
import json
import subprocess
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


base = Path('/workspace/uniguide_audits')
checkout = base / 'candidate-batch-bf960310'
root = Path('/workspace/agenticgen_geometry_acceptance_20261028_bf960310')
complete = root / 'acceptance/COMPLETE.json'
assert complete.exists(), 'Engineering acceptance has not completed'
receipt = json.loads(complete.read_text())
expected = 'bf9603103cecb37e2b390d7c812472c2eda008b9'
head = subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip()
status = subprocess.check_output(['git', '-C', str(checkout), 'status', '--porcelain'], text=True)
assert head == expected and not status
protected = json.loads(Path(str(checkout) + '-source-before.json').read_text())['source_hashes']
after = {p: sha(p) for p in protected}
assert protected == after
processes = []
for path in Path('/proc').glob('[0-9]*/cmdline'):
    try:
        args = path.read_bytes().split(b'\0')
        if b'/workspace/uniguide_audits/verify_candidate_batch_bf960310.py' in args:
            processes.append(path.parent.name)
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        pass
assert not processes, 'Verifier is still exiting'
with (root / '.executor.lock').open('r') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
charges = json.loads((root / 'attempts.json').read_text())
assert sum(row['count'] for row in charges.values()) == 40
targets = {}
for path in sorted(root.glob('acceptance/*/mixed_failure_validation/VERIFIED.json')):
    target = path.parent.parent.name
    verified = json.loads(path.read_text())
    assert verified['healthy_full_100_steps_exact'] and verified['nonfinite_units_preserved']
    targets[target] = dict(verified_sha256=sha(path), engineering_attempts=verified['engineering_attempts'])
assert len(targets) == 2
previous_preexperiment = Path('/workspace/agenticgen_lambda_preexperiment_20261028')
preexperiment = {}
for target in targets:
    directory = previous_preexperiment / target
    ledger = directory / 'attempts.json'
    boundaries = list(directory.glob('calibration/frozen/*/INSUFFICIENT_EVIDENCE.json'))
    preexperiment[target] = dict(
        attempts_sha256=sha(ledger) if ledger.exists() else None,
        has_completed_domains=(directory / 'calibration/domains.json').exists(),
        insufficient_evidence=[dict(path=str(p), sha256=sha(p), evidence=json.loads(p.read_text())) for p in boundaries])
payload = dict(
    time_asia_shanghai=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    execution_commit=head, checkout_clean=True, engineering_process_exited=True, engineering_lock_free=True,
    acceptance_sha256=sha(complete), tests_sha256=sha(base / 'candidate-batch-bf960310-tests.log'),
    fetch_audit_sha256=sha(base / 'candidate-batch-bf960310-fetch.json'),
    verifier_sha256=sha(base / 'verify_candidate_batch_bf960310.py'), audit_script_sha256=sha(__file__),
    engineering_attempts=40, production_attempts_added=0, fitted_multipliers=False, llm_calls=0,
    targets=targets, protected_files=len(protected), source_hashes_after=after,
    existing_preexperiment=preexperiment,
    original_gpu_repository_head=subprocess.check_output(['git','-C','/workspace/EvoMolSteer','rev-parse','HEAD'],text=True).strip(),
    original_gpu_repository_clean=not subprocess.check_output(['git','-C','/workspace/EvoMolSteer','status','--porcelain'],text=True).strip(),
    resumed_UG05=False, launched_new_calibration=False,
    scope='Final changed-path verification plus explicitly source-bound reuse of the 82dc89c1 full engineering receipt')
output = root / 'acceptance/FINAL_AUDIT.json'
assert not output.exists(), 'Final audit already exists'
output.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + '\n')
print(json.dumps(dict(audit=str(output), time=payload['time_asia_shanghai'], protected_files=len(protected), targets=targets)))
