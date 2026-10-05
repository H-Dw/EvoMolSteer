"""Compact cross-round report from retained evidence, without raw trajectories."""
from pathlib import Path
from ..io import read_json, digest, write_json, write_table


def summarize(campaign, evidence, original_terminal, output, decision):
    campaign, evidence, original_terminal, output = map(Path, (campaign, evidence, original_terminal, output))
    if output.exists():
        raise FileExistsError(output)
    config = read_json(campaign)
    baseline = evidence / 'round_01/local'
    native = read_json(baseline / 'terminal_report.json')['results']['unguided']
    execution = read_json(baseline / 'execution_report.json')
    shape = read_json(baseline / 'window/report.json')['results']['unguided']['symmetric_shape_A']
    sources = [campaign, original_terminal, baseline / 'terminal_report.json', baseline / 'execution_report.json', baseline / 'window/report.json']
    rows = []

    def record(label, round_number, terminal, window=None):
        fields = ('n', 'valid_connected', 'pb_fast_pass', 'unique_smiles', 'unique_scaffolds',
                  'all_pic50_on_rescore_mean', 'all_pic50_on_rescore_n',
                  'valid_pic50_on_rescore_mean', 'valid_pic50_on_rescore_n',
                  'unique_valid_pic50_on_rescore_mean', 'unique_valid_pic50_on_rescore_n',
                  'all_mmff_relief_per_heavy_median', 'all_mmff_relief_per_heavy_n',
                  'all_relax_rms_surround_A_mean', 'all_relax_rms_surround_A_n')
        row = {'label': label, 'round': round_number, **{f: terminal[f] for f in fields}}
        distance = window['symmetric_shape_A'] if window else None
        row.update(actual_x_window_end_symmetric_shape_A=distance,
                   shape_improvement_fraction_vs_native=1-distance/shape if distance is not None else None)
        rows.append(row)

    record('Original Steer', None, read_json(original_terminal)['results']['single'])
    record('Native', 0, native, {'symmetric_shape_A': shape})
    completed = [r for r in config['rounds'] if r['status'] == 'completed']
    for r in completed:
        local = evidence / f"round_{r['round']:02d}/local"
        terminal_path, execution_path, window_path = (local / n for n in ('terminal_report.json', 'execution_report.json', 'window/report.json'))
        e = read_json(execution_path)
        if not e['no_particle_resampling'] or e['outside_window_injection'] or e['integration_steps'] != 100:
            raise ValueError('Inference contract violated')
        if e['seed'] != config['master_seed'] or e['window'] != execution['window']:
            raise ValueError('Different random seed or learned window')
        for batch in r['batches']:
            if e['initial_state_signatures'][f'gradient/{batch}'] != execution['initial_state_signatures'][f'unguided/{batch}']:
                raise ValueError('Unpaired initial states')
        t, w = read_json(terminal_path), read_json(window_path)
        if t['results']['gradient']['n'] != r['n_per_arm']:
            raise ValueError('Missing failure-inclusive candidates')
        record(r['campaign'], r['round'], t['results']['gradient'], w['results']['gradient'])
        coordinate_path = local / 'coordinate_audit.json'
        audit = read_json(coordinate_path)
        if audit['window'] != execution['window'] or not audit['coordinate_preflight']['passed'] or not all(b['no_injection_after_window'] for b in audit['batch_results']):
            raise ValueError('Coordinate gradient or time gate failed')
        # execution_report uses the historical common diagnostic reference;
        # these hashes identify the actual coordinate reward and learning data.
        rows[-1].update(inference_commit=e['code_commit'], coordinate_program_sha256=audit['program_sha256'],
                        learning_reference_sha256=audit['reference_sha256'])
        sources.extend((terminal_path, execution_path, window_path, coordinate_path))
    if len(completed) != config['rounds_completed'] or config['rounds_started'] > config['maximum_rounds']:
        raise ValueError('Campaign count mismatch')
    output.mkdir(parents=True)
    write_table(output / 'round_metrics.csv', rows)
    result = {'schema_version': 'coordinate-campaign-summary-1.0', 'seed': config['master_seed'],
              'window': execution['window'], 'maximum_rounds': config['maximum_rounds'],
              'rounds_completed': len(completed), 'decision': decision, 'rows': rows,
              'limitations': ['Two reused development batches, no independent generalization or causal inference',
                              'Same FLOWR oracle; predicted head is not measured binding affinity',
                              'Validity denominators differ; all, valid and unique metrics are separate',
                              'MMFF relief is a strain/relaxation proxy, not receptor binding energy',
                              'Shape matching uses actual intermediate coordinates; chemical assignment errors are reported separately per round'],
              'sources': [{'path': str(p.resolve()), 'sha256': digest(p)} for p in sources]}
    write_json(output / 'summary.json', result)
    lines = ['# seed42 坐标奖励实验汇总', '', f"已完成 {len(completed)}/{config['maximum_rounds']} 轮；窗口 {execution['window']}。", '',
             f"|组别|actual x_{execution['window'][-1]:g}形状距离 Å ↓|较 native 改善 %|有效/PB|唯一分子|all/valid/unique head|MMFF松弛/重原子 ↓|周围松弛 RMS Å ↓|",
             '|---|---:|---:|---:|---:|---|---:|---:|']
    for row in rows:
        distance = row['actual_x_window_end_symmetric_shape_A']
        improve = row['shape_improvement_fraction_vs_native']
        heads = '/'.join(f"{row[k]:.4f}" for k in ('all_pic50_on_rescore_mean','valid_pic50_on_rescore_mean','unique_valid_pic50_on_rescore_mean'))
        lines.append(f"|{row['label']}|{'—' if distance is None else f'{distance:.6f}'}|{'—' if improve is None else f'{100*improve:.3f}'}|{row['valid_connected']}/{row['pb_fast_pass']}|{row['unique_smiles']}|{heads}|{row['all_mmff_relief_per_heavy_median']:.6f}|{row['all_relax_rms_surround_A_mean']:.6f}|")
    lines += ['', decision, '', '本表不把100个同批候选当作100个独立重复，不给出由粒子数夸大的显著性。各项完整分母、失败对照及源文件SHA保存在CSV/JSON和逐轮报告中。', '']
    (output / 'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    return result
