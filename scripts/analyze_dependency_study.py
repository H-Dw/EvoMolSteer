"""Compact local ablation quality tables with matched-batch uncertainty."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from evomolsteer.io import read_json, write_json, digest
from evomolsteer.generation.terminal_statistics import converged_energy

def metrics(path, arm):
    frame = pd.read_csv(path)
    frame = frame.loc[frame.arm.eq(arm)].copy()
    required = ['batch', 'slot', 'pic50_on_rescore', 'valid_connected', 'pb_fast_pass', 'energy_status']
    if not set(required) <= set(frame) or frame.duplicated(['batch', 'slot']).any():
        raise ValueError('Complete identified candidate metrics required')
    if not np.isfinite(frame.pic50_on_rescore).all():
        raise ValueError('All-attempt affinity coverage required')
    frame['energy_status'] = frame.energy_status.fillna('not_applicable')
    energy = converged_energy(frame)
    valid = frame[frame.valid_connected]
    result = {'n': len(frame), 'affinity_mean_all': float(frame.pic50_on_rescore.mean()),
        'affinity_mean_valid': float(valid.pic50_on_rescore.mean()) if len(valid) else None,
        'affinity_p95': float(frame.pic50_on_rescore.quantile(.95)), 'affinity_max': float(frame.pic50_on_rescore.max()),
        'valid': int(frame.valid_connected.sum()), 'pb_fast_pass': int(frame.pb_fast_pass.sum()),
        'mmff_converged_n': len(energy), 'strain_median_per_heavy': float(energy.median()) if len(energy) else None,
        'strain_p90_per_heavy': float(energy.quantile(.9)) if len(energy) else None,
        'energy_status_counts': frame.energy_status.value_counts().to_dict(),
        'valid_unique_graphs': int(valid.smiles.nunique()) if len(valid) else 0,
        'elite_valid': int((valid.pic50_on_rescore >= 8.258901977539063).sum()),
        'source_sha256': digest(path)}
    return frame, result

def paired(left, right):
    merged = left[['batch', 'slot', 'pic50_on_rescore']].merge(right[['batch', 'slot', 'pic50_on_rescore']],
        on=['batch', 'slot'], validate='one_to_one', suffixes=('_left', '_right'))
    if len(merged) != len(left) or len(merged) != len(right):
        raise ValueError('Incomplete matched cohorts')
    merged['delta'] = merged.pic50_on_rescore_left - merged.pic50_on_rescore_right
    batches = merged.groupby('batch').delta.mean()
    if len(batches) < 2: raise ValueError('At least two independent batches required')
    rng = np.random.default_rng(42)
    boot = batches.to_numpy()[rng.integers(0, len(batches), (10000, len(batches)))].mean(1)
    return {'delta_affinity_mean': float(merged.delta.mean()), 'independent_batches': len(batches),
        'batch_deltas': {str(k): float(v) for k, v in batches.items()},
        'batch_positive_fraction': float((batches > 0).mean()),
        'batch_bootstrap_CI95': np.quantile(boot, [.025, .975]).tolist(),
        'CI_limitation': 'Two-batch screen intervals are descriptive and too weak for robust inference' if len(batches) < 6 else
            'Conditional on this fixed target, checkpoint and master seed; only generation batches resampled'}

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True); p.add_argument('--output', required=True)
    a = p.parse_args(); plan = read_json(a.manifest); output = Path(a.output); output.mkdir(parents=True, exist_ok=True)
    frames, outcomes = {}, {}
    for item in plan['cohorts']:
        frames[item['name']], outcomes[item['name']] = metrics(item['metrics'], item['arm'])
    comparison = {}
    controls = plan.get('controls', ['native', 'R26'])
    for name in frames:
        if name not in ['native', 'R26']:
            comparison[name] = {control: paired(frames[name], frames[control]) for control in controls if control != name}
    write_json(output / 'quality_summary.json', {'schema_version': 'dependency-quality-1.0', 'outcomes': outcomes,
        'paired_comparisons': comparison, 'primary': 'All-attempt FLOWR predicted pIC50, not experimental affinity',
        'secondary': 'Converged same-graph local MMFF94s relief per heavy atom; not physical binding energy',
        'elite_threshold': 8.258901977539063, 'protocol_manifest_sha256': digest(a.manifest)})
    pd.DataFrame([{'condition': k, **{n: v for n, v in row.items() if not isinstance(v, dict)}} for k, row in outcomes.items()]).to_csv(output / 'quality_table.csv', index=False)
    # Cache only the reusable comparison columns, rather than every old patch diagnostic.
    keep = ['batch', 'slot', 'pic50_on_rescore', 'valid_connected', 'pb_fast_pass', 'energy_status', 'mmff_relief_per_heavy', 'smiles']
    for name, frame in frames.items():
        frame[[k for k in keep if k in frame]].to_parquet(output / (name + '.metrics.parquet'), index=False, compression=None)
