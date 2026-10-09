"""Export an evidence-to-function ablation table from frozen Agent artifacts."""
import argparse
from pathlib import Path
from evomolsteer.io import read_json, write_json, digest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', required=True); p.add_argument('--quality', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args(); study = Path(a.study); quality = read_json(a.quality)
    rows = []; markdown = ['|条件|被移除的信息／指引|函数族|L/C/P|η / ramp|平均 pIC50|相对 R26|PB / N|应变中位数|',
                          '|---|---|---|---|---|---:|---:|---:|---:|']
    for condition in read_json(study / 'protocol.json')['conditions']:
        name = condition['condition']; folder = study / name
        designer = read_json(folder / 'Designer.response.json')
        compiled = read_json(folder / 'compiled.json')
        program = read_json(Path(__file__).resolve().parents[1] / compiled['program'])
        row = {'condition': name, 'removed': condition['removed'], 'information_level': condition['information_level'],
            'Analyst_directions': read_json(folder / 'Analyst.response.json')['directions'],
            'Designer_family': designer['family'], 'Designer_rationale': designer['rationale'],
            'Designer_failure_modes': designer['failure_modes'], 'program': compiled['program'],
            'program_sha256': digest(Path(__file__).resolve().parents[1] / compiled['program']),
            'reference_sha256': compiled['reference_sha256'], 'quality': quality['outcomes'].get(name)}
        rows.append(row)
        q = row['quality']; function = designer['family']
        weights = '/'.join(str(program.get(k, '—')) for k in ['ligand_anchor_weight','contact_weight','repulsion_weight'])
        dose = str(program['native_rms_ratio']) + ' / ' + str(program['time_ramp_power'])
        if q:
            mean = f"{q['affinity_mean_all']:.4f}"; delta = f"{q['affinity_mean_all']-quality['outcomes']['R26']['affinity_mean_all']:+.4f}"
            pb = f"{q['pb_fast_pass']}/{q['n']}"; strain = f"{q['strain_median_per_heavy']:.4f}"
        else: mean = delta = pb = strain = '待完成'
        removed = ', '.join(condition['removed']) or '无'
        if name.startswith('D'): removed += '; ' + condition['information_level']
        markdown.append(f'|{name}|{removed}|{function}|{weights}|{dose}|{mean}|{delta}|{pb}|{strain}|')
    output = Path(a.output); output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'evidence_to_function.json', {'rows': rows, 'quality_sha256': digest(a.quality),
        'limits': 'Primary Agent calls are stochastic workflow realizations, not isolated coefficient or causally identified instruction effects'})
    (output / 'ablation_table.md').write_text('\n'.join(markdown)+'\n', encoding='utf-8')
