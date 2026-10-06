"""Read-only, hash-grounded verification of retained motif-campaign evidence."""
import hashlib
import json
from pathlib import Path

from ..io import digest, read_json
from .coordinate_exploration import validate_retention
from .motif_campaign import validate_initial_pairing
from .window_reference import load_reference


def effective_program(program):
    """Keep every field except explicit round/derivation metadata.

    Unknown future control fields therefore cannot silently escape the frozen
    validation contract. This does not impose chemical-graph equality.
    """
    return {k: v for k, v in program.items()
            if k not in {'round', 'program_id', 'derivation'}}


def verify_frozen(programs, parent):
    if parent not in (9, 10, 11, 12):
        raise ValueError('Final winner must come from strict-scope discovery')
    frozen = effective_program(programs[parent])
    for number in (13, 14, 15):
        if number in programs and effective_program(programs[number]) != frozen:
            raise ValueError('Frozen validation/heldout reward changed')
    return hashlib.sha256(json.dumps(frozen, sort_keys=True,
        separators=(',', ':')).encode('utf8')).hexdigest()


def audit(evidence, config, require_complete=True):
    """Audit summaries only; never read, recreate or delete generated structures."""
    evidence, config = Path(evidence), Path(config)
    campaign = read_json(config / 'campaign.json')
    outcomes = [read_json(p) for p in sorted(evidence.glob('round_*.outcome.json'))]
    numbers = [r['round'] for r in outcomes]
    expected = list(range(1, 16)) if require_complete else list(range(1, len(numbers)+1))
    if numbers != expected or not numbers:
        raise ValueError('Sequential retained rounds required; final audit needs fifteen')
    rounds = {r['round']: r for r in campaign['rounds']}
    references = {digest(p): p for p in config.glob('*reference.json.gz')}
    programs, checks = {}, []
    baseline = read_json(evidence / 'round_01/local/window/report.json')
    for number in numbers:
        entry = rounds[number]
        local = evidence / f'round_{number:02d}/local'
        retention_sha = validate_retention(local, entry['n_per_arm'] * len(entry['arms']))
        source_program = config / f'backtrack_round{number:02d}.json'
        actual_program = local / 'inference_config/reward_program.json'
        p = read_json(actual_program)
        if read_json(source_program) != p or digest(source_program) != digest(actual_program):
            raise ValueError('Source/actual program bytes differ')
        programs[number] = p
        coordinate = read_json(local / 'coordinate_audit.json')
        execution = read_json(local / 'execution_report.json')
        window = read_json(local / 'window/report.json')
        actual_config = read_json(local / 'inference_config/config.json')
        extension = actual_config['extension']
        if (coordinate['program_sha256'] != digest(actual_program)
                or coordinate['reference_sha256'] != p['reference_sha256']
                or execution['code_commit'] != entry['inference_commit']):
            raise ValueError('Actual inference provenance differs')
        if (not coordinate['coordinate_preflight']['passed']
                or not execution['no_particle_resampling']
                or execution['outside_window_injection']
                or extension['particle_resampling']
                or extension['post_window_injection']
                or extension['apply_guidance']):
            raise ValueError('Derivative/no-SMC/control-support contract failed')
        if (extension['control_domain'] != p['window']
                or extension['evidence_domain'] != p['window']
                or execution['window'] != p['window']):
            raise ValueError('Evidence and control windows differ')
        if execution['integration_steps'] != 100 or any(
                b['inference_steps'] != 100 or b['outside_injection']
                for b in execution['batch_results']):
            raise ValueError('Full native continuation contract failed')
        expected_batches = sorted(entry['batches'])
        active = [b for b in coordinate['batch_results'] if b['arm'] == 'gradient']
        if sorted(b['batch'] for b in active) != expected_batches or any(
                not b['no_injection_after_window'] or b['nonzero_injection_steps'] != b['active_steps']
                for b in active):
            raise ValueError('Incomplete window-dose coverage')
        if 'gradient_zero' in entry['arms'] and window['zero_equivalence_passed'] is not True:
            raise ValueError('Zero-dose/native equivalence failed')
        pairing = validate_initial_pairing(window,
            window if 'unguided' in entry['arms'] else baseline)
        ref = load_reference(references[p['reference_sha256']])
        strict = bool(ref.get('scale_score_times')) and max(ref['scale_score_times']) < p['window'][1]-1e-6
        if number >= 9 and not strict:
            raise ValueError('Final-scope normalization includes outside-window states')
        checks.append({'round': number, 'attempts': entry['n_per_arm'] * len(entry['arms']),
            'retention_sha256': retention_sha, 'program_sha256': digest(actual_program),
            'reference_sha256': p['reference_sha256'], 'inference_commit': entry['inference_commit'],
            'initial_pairing': pairing, 'full_native_steps': 100,
            'nonzero_guidance_steps_per_batch': [b['nonzero_injection_steps'] for b in active],
            'strict_window_scale': strict,
            'zero_equivalence': window.get('zero_equivalence_passed')})
    frozen = None
    if 13 in programs:
        parent = read_json(evidence / 'round_13.plan.json')['parent_round']
        frozen = {'parent_round': parent, 'effective_program_sha256': verify_frozen(programs, parent),
            'checked_rounds': [r for r in (13, 14, 15) if r in programs]}
    return {'schema_version': 'motif-campaign-verification-1.0', 'passed': True,
        'rounds_verified': len(numbers), 'attempts': sum(r['attempts'] for r in checks),
        'rounds': checks, 'frozen_validation': frozen,
        'scope': 'Execution and retained-byte verification, not evidence of biological affinity or energy improvement.',
        'legacy_scale_exception': 'R1–8 are preserved historical screens; final winner is restricted to strict-scope R9–12.'}
