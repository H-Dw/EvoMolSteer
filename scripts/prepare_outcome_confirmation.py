"""Freeze a learned candidate and add independent, clock-matched controls."""
import argparse
import copy
from pathlib import Path
from evomolsteer.io import digest, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT/'configs/experiments/terminal_outcome15_v1'
ADMIN = {'program_id', 'round', 'agent_request_sha256', 'derivation', 'outcome_binding'}
SOURCES = ['controller.py', 'flowcompat_v2_controller.py', 'flowcompat_controller.py',
    'endpoint_controller.py', 'flowcompat_control.py', 'endpoint_reward.py',
    'affinity_geometry_reward.py', 'branch_mixture_reward.py', 'coordinate_conditioning.py']


def runtime_program(program):
    return {key: value for key, value in program.items() if key not in ADMIN}


def prepare(number, candidate):
    if number not in (14, 15):
        raise ValueError('Reserved frozen independent confirmation rounds required')
    candidate = (ROOT/Path(candidate)).resolve()
    plan_path = CFG/f'round{number:02d}.jobs.json'
    plan = read_json(plan_path)
    if len(plan['jobs']) != 1 or (ROOT/plan['jobs'][0]['program']).resolve() != candidate:
        raise ValueError('A single compiled candidate job must precede controls')
    p = read_json(candidate)
    reference = ROOT/plan['jobs'][0]['reference']
    if p['reference_sha256'] != digest(reference):
        raise ValueError('Compiled reference mismatch')
    if p.get('outcome_binding', {}).get('updates') != {}:
        raise ValueError('Confirmation cannot change any reward parameter')
    lock = CFG/'confirmation.freeze.json'
    sources = {name: digest(ROOT/'src/evomolsteer/generation'/name) for name in SOURCES}
    if number == 14:
        if lock.exists():
            raise FileExistsError(lock)
        write_json(lock, {'selected_program_path': candidate.relative_to(ROOT).as_posix(),
            'selected_program_sha256': digest(candidate), 'runtime_program': runtime_program(p),
            'development_parent_program_sha256': p['outcome_binding']['base_program_sha256'],
            'reference_sha256': digest(reference), 'source_sha256': sources,
            'selection_data': 'Completed development batches only; confirmation outcomes unused',
            'confirmation_batches': {'14': [133, 134], '15': [135, 136]}})
    frozen = read_json(lock)
    if runtime_program(p) != frozen['runtime_program'] or sources != frozen['source_sha256'] or digest(reference) != frozen['reference_sha256']:
        raise ValueError('Frozen candidate or generation implementation changed')
    batches = frozen['confirmation_batches'][str(number)]
    if plan['jobs'][0]['batches'] != batches:
        raise ValueError('Independent batch panel required')
    plan['jobs'][0]['arms'] = 'gradient,unguided'
    for label, old_path, ref_path in (
        ('R11_051', CFG/'round02_R11.json', CFG/'references/R11_instant_051.json.gz'),
        ('R26_051', CFG/'round01_R26.json', CFG/'references/R26_instant_051.json.gz')):
        control = copy.deepcopy(read_json(old_path))
        control.update(round=number, program_id=f'{label}_frozen_control_round{number:02d}')
        if control['reference_sha256'] != digest(ref_path) or control['window'] != p['window'] or control['score_window'] != p['score_window']:
            raise ValueError('Historical control must use identical clocks')
        target = CFG/f'round{number:02d}_{label}_control.json'
        if target.exists():
            raise FileExistsError(target)
        write_json(target, control)
        plan['jobs'].append({'cohort': label, 'campaign': f'outcome_r{number:02d}_{label}',
            'program': target.relative_to(ROOT).as_posix(), 'reference': ref_path.relative_to(ROOT).as_posix(),
            'n': 50*len(batches), 'batches': batches, 'arms': 'gradient'})
    plan['freeze_lock_sha256'] = digest(lock)
    write_json(plan_path, plan)
    print(plan_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--round', type=int, required=True)
    parser.add_argument('--program', required=True)
    args = parser.parse_args()
    prepare(args.round, args.program)
