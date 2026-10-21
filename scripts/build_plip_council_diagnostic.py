#!/usr/bin/env python3
"""Export a completed four-hypothesis capacity probe for analysis-only validation.

No sampling, no changed scientific records, no claimed final geometry or quality.
The probe's saved per-step amplitudes must prove the expected group layout.
"""
import argparse
from pathlib import Path

from evomolsteer.adaptive_schedule.io import read, write, sha
from evomolsteer.adaptive_schedule.group_tools import interval_summary, PopulationTools
from evomolsteer.adaptive_schedule.grid import INTERVALS


def export(source, templates, output):
    source, output = Path(source), Path(output)
    benchmark = read(source/'benchmark.json')
    if benchmark['status'] != 'ok' or benchmark['batch_size'] != 400 or benchmark['attempts'] != 400:
        raise ValueError('A complete four-by-100 probe is required')
    teacher = next(t for t in read(templates) if t['teacher_id'] == benchmark['teacher'])
    records, nodes, steps = (read(source/name) for name in ['final_records.json', 'nodes.json', 'steps.json'])
    if len(records) != 400 or len(steps) != 100:
        raise ValueError('Incomplete capacity evidence')
    def select(row, index):
        return {k: v[index] if isinstance(v, list) and len(v) == 400 else v
                for k, v in row.items() if k not in {'rng_before', 'rng_after'}}
    schedules = [[4, 5, 5, 5], [6, 5, 5, 5], [5, 4, 5, 5], [5, 6, 5, 5]]
    groups = []
    for n, ticks in enumerate(schedules):
        chains = []
        for i in range(100):
            ix = n*100+i
            chain_steps = [select(row, ix) for row in steps]
            for j, (start, end) in enumerate(INTERVALS):
                for row in chain_steps:
                    if start <= row['step'] <= end and abs(row['lambda']-ticks[j]/10) > 1e-6:
                        raise ValueError('Probe schedules do not match declared group layout')
            chains.append({'candidate_id': f'c{i:03d}', 'batch': i//20, 'slot': i%20,
                'incumbent_id': 'r00', 'latest_id': 'r00', 'restartable': False,
                'trials': [{'id': 'r00', 'ticks': ticks, 'record': records[ix],
                    'nodes': {s: select(row, ix) for s, row in nodes.items()},
                    'intervals': interval_summary(chain_steps), 'source_branch': str(source)}]})
        data = {'teacher': teacher, 'group_id': f'G{n+1}', 'observed_round': 0,
                'round_ids': ['r00'], 'chains': chains}
        PopulationTools(data, output/'unused_validation')
        file = output/f'G{n+1}.json'
        if file.exists() and read(file) != data:
            raise ValueError('Do not overwrite frozen diagnostic evidence')
        write(file, data)
        groups.append({'group_id': f'G{n+1}', 'evidence': file.name, 'anchor_ticks': ticks})
    manifest = {'groups': groups, 'diagnostic_only': True, 'source_root': str(source),
        'unavailable': ['PLIP', 'PoseBusters', 'final step 100 local geometry', 'past optimization rounds'],
        'source_hashes': {name: sha(source/name) for name in ['benchmark.json', 'final_records.json', 'nodes.json', 'steps.json']}}
    write(output/'manifest.json', manifest)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--templates', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    export(args.source, args.templates, args.output)
