"""Portable identities and atomic, checksummed per-attempt receipts."""
import hashlib
import gzip
import json
import time
from pathlib import Path

from evomolsteer.io import read_json, digest
from evomolsteer.storage.transactions import atomic_json

TERMINAL = {'complete', 'failed', 'skipped_condition_error'}


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def binding(config, condition, attempt):
    return identity({'protocol': config['protocol_sha256'], 'condition': condition, 'attempt': attempt})


def ordered_attempts(attempts):
    pilot = [1, 50, 100, 200]
    return sorted(attempts, key=lambda x: (pilot.index(x['rank']) if x['rank'] in pilot else 4, x['rank']))


def result_path(work, condition, attempt):
    return Path(work) / 'methods' / condition['id'] / attempt['id'] / 'result.json'


def read_result(work, config, condition, attempt):
    path = result_path(work, condition, attempt)
    if not path.exists():
        return None
    record = read_json(path)
    if record.get('binding') != binding(config, condition, attempt):
        raise ValueError('Existing result has a different frozen input: ' + str(path))
    if record.get('status') not in TERMINAL:
        raise ValueError('Nonterminal result receipt: ' + str(path))
    if record['status'] == 'complete':
        for name, expected in record.get('artifacts', {}).items():
            if digest(path.parent / name) != expected:
                raise ValueError('Result artifact checksum differs: ' + str(path.parent / name))
        if not record.get('artifacts'):
            raise ValueError('Complete result lacks artifacts')
    return record


def failure(work, config, condition, attempt, error, *, status='failed', phase='inference'):
    path = result_path(work, condition, attempt)
    if path.exists():
        # Explicit retry archives the old receipt before a new attempt is started.
        return read_result(work, config, condition, attempt)
    record = {'binding': binding(config, condition, attempt), 'condition': condition['id'],
              'attempt': attempt, 'status': status, 'phase': phase, 'error': str(error),
              'finished_unix': time.time(), 'pic50': None, 'valid_connected': False,
              'qualified_structure': False, 'artifacts': {}}
    atomic_json(path, record)
    return record


def validate_protocol(config):
    frozen = {k: v for k, v in config.items() if k != 'protocol_sha256'}
    if identity(frozen) != config['protocol_sha256']:
        raise ValueError('Frozen protocol checksum differs')
    if len(config['attempts']) != 200 or len(config['conditions']) != 17:
        raise ValueError('Expected 200 attempts and 17 frozen conditions')
    if config['steps'] != 100 or config['master_seed'] != 42:
        raise ValueError('Unexpected integration or master-seed protocol')
    if len({x['suffix_seed'] for x in config['attempts']}) != 200:
        raise ValueError('Suffix RNG collision')


def validate_condition(repo, condition):
    if condition['kind'] == 'unguided':
        return None, None
    program, reference = Path(repo) / condition['program'], Path(repo) / condition['reference']
    if digest(program) != condition['program_sha256'] or digest(reference) != condition['reference_sha256']:
        raise ValueError('Frozen reward or teacher checksum differs: ' + condition['id'])
    if reference.suffix == '.gz':
        with gzip.open(reference, 'rt', encoding='utf-8') as stream:
            teacher = json.load(stream)
    else:
        teacher = read_json(reference)
    spec = read_json(program)
    inner = spec.get('teacher_sha256', spec.get('reference_sha256'))
    if inner != condition['reference_sha256']:
        raise ValueError('Program names a different teacher: ' + condition['id'])
    return spec, teacher
