"""Create one serial inference job only from a validated compiled reward."""
import argparse
from pathlib import Path
from evomolsteer.io import digest, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT/'configs/experiments/terminal_outcome15_v1'


def finalize(number, program, reference, cohort, batches, arms='gradient'):
    program, reference = (ROOT/Path(program)).resolve(), (ROOT/Path(reference)).resolve()
    if not 1 <= number <= 15 or not all(p.is_relative_to(CFG) for p in (program, reference)):
        raise ValueError('Registered local campaign paths and fifteen-round budget required')
    p = read_json(program)
    if p['round'] != number or p['seed'] != 42 or p['reference_sha256'] != digest(reference):
        raise ValueError('Compiled program/reference/round mismatch')
    if not p.get('outcome_binding') or p['outcome_binding']['label_source'] != 'decoded_final':
        raise ValueError('Actual validated final-label Agent compilation required')
    target = CFG/f'round{number:02d}.jobs.json'
    if target.exists():
        raise FileExistsError(target)
    if len(set(batches)) != len(batches) or not batches:
        raise ValueError('Distinct whole batches required')
    job = {'cohort': cohort, 'campaign': f'outcome_r{number:02d}_{cohort}',
           'program': program.relative_to(ROOT).as_posix(),
           'reference': reference.relative_to(ROOT).as_posix(),
           'n': 50*len(batches), 'batches': batches, 'arms': arms}
    write_json(target, {'round': number, 'seed': 42, 'jobs': [job]})
    return target


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--round', type=int, required=True)
    for flag in ('program', 'reference', 'cohort'):
        p.add_argument('--'+flag, required=True)
    p.add_argument('--batches', default='131,132')
    p.add_argument('--arms', default='gradient')
    a = p.parse_args()
    print(finalize(a.round, a.program, a.reference, a.cohort,
                   list(map(int, a.batches.split(','))), a.arms))
