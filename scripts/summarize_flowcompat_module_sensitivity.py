"""Summarize measured decoder sensitivities from compact retained reports.

This is a post-generation diagnostic. It does not fit an affinity model, change
an inference contract, or rank differently scaled module outputs by importance.
Only within-module, matched batch/time scalar differences are compared.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from evomolsteer.io import digest, read_json, write_json


def load_measurements(folder, inputs):
    table_path = folder / 'decoder_sensitivity.parquet'
    program_path = folder / 'inference_config/reward_program.json'
    if not table_path.exists():
        return None
    for path in [table_path, program_path, folder / 'retention.json']:
        inputs[str(path)] = digest(path)
    program = read_json(program_path)
    lower, upper = map(float, program['window'])
    table = pd.read_parquet(table_path)
    table = table[table.arm.eq('gradient')].copy()
    keys = ['batch', 'time', 'module']
    if table.duplicated(keys).any():
        raise ValueError('Module observations must have unique batch/time keys')
    if not table['finite'].all() or not np.isfinite(table.gradient_RMS).all():
        raise ValueError('A retained nonfinite sensitivity must be investigated')
    if (table.gradient_RMS < 0).any() or (table.elements <= 0).any():
        raise ValueError('Invalid gradient RMS or tensor size')
    if ((table.time < lower - 2e-7) | (table.time >= upper + 2e-7)).any():
        raise ValueError('Observed sensitivities exceed the declared window')
    table['time_key'] = table.time.round(7)
    if table.duplicated(['batch', 'time_key', 'module']).any():
        raise ValueError('Time-key rounding merged distinct actual observations')
    return table, [lower, upper]


def summarize(input_dataset, output_dataset):
    source, output = Path(input_dataset).resolve(), Path(output_dataset).resolve()
    if source == output or output.is_relative_to(source):
        raise ValueError('Keep the derived dataset outside its input dataset')
    output.mkdir(parents=True, exist_ok=True)
    inputs, tables, windows = {}, {}, {}
    for folder in sorted(source.glob('round[0-9][0-9]')):
        if not (folder / 'retention.json').exists():
            continue
        loaded = load_measurements(folder, inputs)
        if loaded is not None:
            number = int(folder.name[-2:])
            tables[number], windows[number] = loaded
    if not tables:
        raise ValueError('No retained measured module sensitivities')
    rows = []
    for number, table in tables.items():
        # Empty-mechanism controls 2/8 cover screening batches. Frozen controls
        # 25/27/29 cover the respective held-out candidate pairs 26/28/30.
        control_number = {26: 25, 28: 27, 30: 29}.get(number, 8 if number >= 8 else 2)
        control = tables.get(control_number)
        paired = None
        if control is not None:
            if windows[number] != windows[control_number]:
                raise ValueError('Paired module windows differ')
            paired = table.merge(control, on=['batch', 'time_key', 'module'],
                                 suffixes=('', '_control'), validate='one_to_one')
            if len(paired) != len(table):
                raise ValueError('Baseline must cover every compared module observation')
            if not paired.elements.eq(paired.elements_control).all():
                raise ValueError('Compared module output shapes differ')
        for module, group in table.groupby('module', sort=True):
            per_batch = group.groupby('batch', sort=True).gradient_RMS.mean()
            concentration = group.groupby('batch', sort=True).top_quarter_slot_gradient_energy_fraction.mean()
            record = {
                'round': number, 'module_output': module,
                'independent_generation_batches': len(per_batch),
                'observed_batch_times': len(group),
                'gradient_RMS_batch_mean': float(per_batch.mean()),
                'gradient_RMS_batch_min': float(per_batch.min()),
                'gradient_RMS_batch_max': float(per_batch.max()),
                'top_quarter_slot_energy_fraction_batch_mean': float(concentration.mean()),
                'comparison_round': control_number if paired is not None else None,
                'relative_RMS_difference': None,
                'slot_energy_fraction_difference': None,
            }
            if paired is not None:
                pair = paired[paired.module.eq(module)]
                difference = (pair.gradient_RMS - pair.gradient_RMS_control).groupby(pair.batch).mean().mean()
                baseline = pair.groupby('batch').gradient_RMS_control.mean().mean()
                record['relative_RMS_difference'] = float(difference / baseline) if baseline > 0 else None
                shift = pair.top_quarter_slot_gradient_energy_fraction - pair.top_quarter_slot_gradient_energy_fraction_control
                record['slot_energy_fraction_difference'] = float(shift.groupby(pair.batch).mean().mean())
            rows.append(record)
    pd.DataFrame(rows).to_csv(output / 'module_sensitivity_summary.csv', index=False)
    write_json(output / 'interpretation.json', {
        'schema_version': 'flowcompat-module-sensitivity-summary-1.0',
        'rounds': sorted(tables), 'learned_windows': windows,
        'measurement': 'Existing reward backward hooks; no extra neural model forward',
        'comparison': 'Same module output and matched batch/time RMS; not a vector-gradient difference',
        'limitations': [
            'Different tensor scales and representations prohibit ranking modules by these RMS values.',
            'These are sensitivities of the chosen reward, not affinity-head gradients or causal module contributions.',
            'Top-quarter slots may refer to different output axes; this summary does not map them to molecular regions.',
            'Paired trajectories differ after injection; scalar changes mix reward and state changes.',
            'Generation batches share target/checkpoint/master seed and are not biological replications.',
            'Post-generation reporting is not retrospectively claimed as evidence consumed by the bound Agents.',
        ],
    })
    write_json(output / 'manifest.json', {
        'input_dataset': str(source), 'input_files': inputs, 'script_sha256': digest(__file__),
        'outputs': {p.name: digest(p) for p in sorted(output.iterdir())
                    if p.is_file() and p.name != 'manifest.json'},
    })
    return {'retained_rounds': len(tables), 'module_summary_rows': len(rows)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dataset', required=True)
    parser.add_argument('--output-dataset', required=True)
    args = parser.parse_args()
    print(summarize(args.input_dataset, args.output_dataset))
