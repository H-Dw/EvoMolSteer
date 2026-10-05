"""Compare frozen terminal reports without retaining generated structures.

Two fixed-seed batches are development evidence. Per-batch effects are exposed;
this module deliberately does not manufacture a population confidence interval
from the individual particles or collapse different quality axes into a score.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from ..io import read_json, write_json, digest


METRICS = [
    ('valid_rate', 'Connected valid yield', 'higher'),
    ('pb_fast_pass_rate', 'PB dock_fast yield', 'higher'),
    ('unique_smiles', 'Unique canonical graphs', 'higher'),
    ('unique_scaffolds', 'Unique scaffolds', 'higher'),
    ('unique_graph_diversity', 'Unique graph diversity', 'higher'),
    ('valid_pic50_on_rescore_mean', 'Valid-pose target head', 'higher'),
    ('unique_valid_pic50_on_rescore_mean', 'Unique-first-pose target head', 'higher'),
    ('valid_pic50_off_rescore_mean', 'Valid-pose off-target head', 'descriptive'),
    ('valid_mmff_relief_per_heavy_median', 'MMFF relief / heavy atom', 'lower'),
    ('valid_relax_rms_surround_A_mean', 'Surround relaxation RMS (A)', 'lower'),
    ('valid_surround_clash_atom_fraction_mean', 'Surround severe clash fraction', 'lower'),
    ('valid_terminal_patch_q_over_r2_mean', 'Final regional q/r2', 'lower'),
]


def compare(manifest_path, output):
    manifest = read_json(manifest_path)
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    rows, batchrows, diagnostics, sources = [], [], [], []
    signatures = {}
    for item in manifest['groups']:
        report = read_json(item['terminal_report']); arm = item['arm']; label = item['label']
        if report['seed'] != 42:
            raise ValueError('Comparison requires frozen master seed 42')
        result = report['results'][arm]
        rows.append({'group': label, **result})
        batchrows.extend({'group': label, **r} for r in report['batch_results'] if r['arm'] == arm)
        sources.append({'path': item['terminal_report'], 'sha256': digest(item['terminal_report'])})
        if item.get('execution_report'):
            execution = read_json(item['execution_report'])
            if not execution['no_particle_resampling'] or execution['outside_window_injection']:
                raise ValueError('Experimental execution contract failed')
            signatures[label] = {k.split('/')[1]: v for k, v in execution['initial_state_signatures'].items() if k.split('/')[0] == arm}
            diagnostics.extend({'group': label, **r} for r in execution['endpoint_diagnostics_at_window_end'] if r['arm'] == arm)
            sources.append({'path': item['execution_report'], 'sha256': digest(item['execution_report'])})
    d = pd.DataFrame(rows).set_index('group'); batches = pd.DataFrame(batchrows)
    native = manifest['native']; steer = manifest['steer']; tests = manifest['tests']
    pairs = []
    for name in tests:
        if signatures.get(name) != signatures.get(native) or not signatures.get(native):
            raise ValueError('Experimental arms do not share the exact native initial states')
        left = batches[batches.group == name].set_index('batch')
        right = batches[batches.group == native].set_index('batch')
        if list(left.index) != list(right.index):
            raise ValueError('Different batch coverage')
        for batch in left.index:
            for metric, _, direction in METRICS:
                x, y = left.loc[batch, metric], right.loc[batch, metric]
                pairs.append({'group': name, 'native': native, 'batch': int(batch), 'metric': metric,
                              'difference': float(x-y) if pd.notna(x) and pd.notna(y) else None,
                              'preferred_direction': direction})
    decisions = {}
    regional = pd.DataFrame(diagnostics)
    for name in tests:
        qualities = [d.loc[name, k] - d.loc[native, k] >= -.05 for k in ['valid_rate', 'pb_fast_pass_rate']]
        local = regional[regional.group == name]; base = regional[regional.group == native]
        deficit = float(local.mean_deficit.mean()) if len(local) else None
        native_deficit = float(base.mean_deficit.mean()) if len(base) else None
        decisions[name] = {
            'quality_yield_gate_passed': bool(all(qualities)),
            'window_regional_deficit': deficit, 'native_window_regional_deficit': native_deficit,
            'regional_deficit_improved': deficit < native_deficit if deficit is not None and native_deficit is not None else None,
            'final_target_head_at_least_steer': bool(d.loc[name, 'valid_pic50_on_rescore_mean'] >= d.loc[steer, 'valid_pic50_on_rescore_mean']),
            'final_mmff_relief_at_most_steer': bool(d.loc[name, 'valid_mmff_relief_per_heavy_median'] <= d.loc[steer, 'valid_mmff_relief_per_heavy_median']),
            'unique_yield_above_steer': bool(d.loc[name, 'unique_yield'] > d.loc[steer, 'unique_yield']),
            'surround_relaxation_at_most_steer': bool(d.loc[name, 'valid_relax_rms_surround_A_mean'] <= d.loc[steer, 'valid_relax_rms_surround_A_mean']),
            'interpretation': 'Development screen only. Check energy coverage and every batch effect; no causal or generalization claim.'}
    d.reset_index().to_csv(out/'arm_metrics.csv', index=False)
    batches.to_csv(out/'batch_metrics.csv', index=False)
    pd.DataFrame(pairs).to_csv(out/'paired_batch_differences.csv', index=False)
    regional.to_csv(out/'window_regional_metrics.csv', index=False)
    report = {'schema_version': 'terminal-comparison-1.0', 'master_seed': 42,
              'maximum_rounds': manifest.get('maximum_rounds',5), 'round': manifest['round'], 'groups': manifest['groups'],
              'new_arms_exact_initial_state_pairing': True, 'decisions': decisions, 'sources': sources,
              'uncertainty': 'Fixed seed and reused development batches; particles are not independent experimental replicates. No inferential CI is issued.'}
    write_json(out/'comparison.json', report)
    lines = [f"# Seed 42 terminal comparison — round {manifest['round']}", '',
             'All arms complete 100 integration steps. New guided arms use no particle resampling; control ends at the learned window boundary.', '',
             '| Metric | ' + ' | '.join(d.index) + ' |', '|---|' + '|'.join(['---:']*len(d)) + '|']
    for metric, title, _ in METRICS:
        lines.append('| ' + title + ' | ' + ' | '.join(f'{d.loc[k, metric]:.6g}' if pd.notna(d.loc[k, metric]) else 'NA' for k in d.index) + ' |')
    lines.extend(['', 'MMFF is same-graph local relaxation relief, not binding free energy. FLOWR affinity-head predictions share the training/selection oracle. PB dock_fast is a structural subset.', '',
                  'The historical Steer arm is an original result, not a new randomized treatment. Fixed-seed development batches cannot establish generalization or causality.', '',
                  'Per-batch effects, missing-energy counts and candidate failures remain in the linked source reports; no composite score substitutes for these checks.'])
    (out/'comparison.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return report
