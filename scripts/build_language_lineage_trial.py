"""Score-blind language recognition of held-out, observed Steer ancestry.

Prepare donor narratives and anonymous current/predicted state cards. Terminal
quality, parent slots, offspring, weights and scores stay in the oracle folder.
This tests path recognition; extinct alternatives have censored outcomes.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evomolsteer.continuous.affinity_geometry import numpy_geometry, names
from evomolsteer.continuous.elite_path_credit import terminal_ancestors
from evomolsteer.continuous.endpoint_regions import regional_moments
from evomolsteer.io import digest, read_json, write_json


def fingerprint(arrays):
    h = hashlib.sha256()
    for a in arrays:
        v = np.ascontiguousarray(a)
        h.update(v.dtype.str.encode()); h.update(str(v.shape).encode()); h.update(v.tobytes())
    return h.hexdigest()


def features(x, y, atoms, bonds, points, origin, previous_x=None, previous_y=None, regions=(0, 5, 15)):
    out = {}
    for prefix, cloud in [('current', x), ('forecast', y)]:
        z = numpy_geometry(cloud[None], points, origin)[0]
        f = dict(zip(names(points), z))
        for name in ['global_centroid_x', 'global_centroid_y', 'global_centroid_z',
                     'shape_xx', 'shape_yy', 'shape_zz', 'pair_kernel_2.5', 'pair_kernel_7.0']:
            out[prefix + '_' + name] = float(f[name])
        out[prefix + '_Rg'] = float(np.sqrt(sum(f['shape_' + a] for a in ['xx', 'yy', 'zz'])))
        local = regional_moments(cloud[None], points).reshape(len(points), 10)
        for r in regions:
            out[f'{prefix}_region{r:02d}_mass'] = float(local[r, 0])
            for j, axis in enumerate('xyz'):
                out[f'{prefix}_region{r:02d}_d{axis}'] = float(local[r, j + 1])
            out[f'{prefix}_region{r:02d}_occupancy3'] = float(f[f'landmark_{r:02d}_occupancy3'])
        d = np.linalg.norm(cloud[:, None] - cloud[None], axis=-1)
        out[prefix + '_minimum_pair_distance'] = float(d[~np.eye(len(x), dtype=bool)].min())
    out['remaining_slot_RMS'] = float(np.sqrt(((y-x)**2).sum(-1).mean()))
    out['current_step_RMS'] = None if previous_x is None else float(np.sqrt(((x-previous_x)**2).sum(-1).mean()))
    out['forecast_revision_RMS'] = None if previous_y is None else float(np.sqrt(((y-previous_y)**2).sum(-1).mean()))
    out['predicted_N_count'] = int((atoms == 4).sum())
    out['predicted_O_count'] = int((atoms == 5).sum())
    out['predicted_edges'] = int((np.triu(bonds, 1) != 0).sum())
    return out


def fmt(v, n=3):
    return '未定义' if v is None else f'{v:.{n}f}'


def sentence(f, *, changes=None, regions=(0, 5, 15)):
    lines = []
    for p, label in [('current', '当前噪声/生成槽位云'), ('forecast', '此刻模型预测的终点云')]:
        center = ','.join(fmt(f[p + '_global_centroid_' + a]) for a in 'xyz')
        cov = ','.join(fmt(f[p + '_shape_' + a]) for a in ['xx', 'yy', 'zz'])
        lines.append(f'{label}：相对参考重心偏移({center}) Å；回转半径 {fmt(f[p+"_Rg"])} Å；三个固定轴的二阶矩({cov}) Å²；2.5 Å短程对核 {fmt(f[p+"_pair_kernel_2.5"])}，7 Å长程对核 {fmt(f[p+"_pair_kernel_7.0"])}；最短槽位间距 {fmt(f[p+"_minimum_pair_distance"])} Å。')
        details = []
        for r in regions:
            offset = ','.join(fmt(f[f'{p}_region{r:02d}_d{a}']) for a in 'xyz')
            details.append(f'R{r:02d}的高斯质量 {fmt(f[f"{p}_region{r:02d}_mass"])}、局部偏移({offset}) Å、3 Å壳层核 {fmt(f[f"{p}_region{r:02d}_occupancy3"])}')
        lines.append('区域分布：' + '；'.join(details) + '。')
    lines.append(f'当前到预测终点的同槽位剩余RMS为 {fmt(f["remaining_slot_RMS"])} Å；本次沿真实父子边的当前位移RMS为 {fmt(f["current_step_RMS"])} Å，终点预测修订RMS为 {fmt(f["forecast_revision_RMS"])} Å。')
    lines.append(f'预测硬标签中N/O槽位数为 {f["predicted_N_count"]}/{f["predicted_O_count"]}，非零硬键边数为 {f["predicted_edges"]}；这不等于当前噪声态已形成有效分子。')
    if changes:
        selected = ['current_Rg', 'forecast_Rg', 'forecast_pair_kernel_2.5', 'forecast_pair_kernel_7.0',
                    *[f'forecast_region{r:02d}_{kind}' for r in regions for kind in ['mass', 'dx', 'dy', 'dz', 'occupancy3']]]
        lines.append('相对前一0.01节点：' + '；'.join(f'{k}变化 {fmt(changes[k])}、变化率 {fmt(changes[k]/.01)} /t' for k in selected) + '。')
    return '\n'.join(lines)


def trajectory(root, batch):
    path = root / 'single' / f'batch_{batch:03d}' / 'trajectory.npz'
    with np.load(path, allow_pickle=False) as z:
        # Every scientific field needed by the oracle is retained in memory,
        # never copied into anonymous public records.
        arrays = {k: z[k] for k in z.files}
    centers = np.asarray(read_json(root / f'frame_batch_{batch:03d}.json')['target_com'], float).reshape(-1, 3)
    if not np.allclose(centers, centers[0], rtol=0, atol=1e-8):
        raise ValueError('This trial requires a shared aligned coordinate frame within each batch')
    com = centers[:1]
    return arrays, com, path


def prepare(repo, config):
    out = repo / config['output']
    if (out / 'manifest.json').exists():
        raise FileExistsError('Use a new output; do not alter an active blind trial')
    blind, oracle = out / 'blind', out / 'oracle'
    blind.mkdir(parents=True, exist_ok=True); oracle.mkdir(exist_ok=True)
    root = repo / config['dataset'] / 'results' / config['campaign']
    cfg = read_json(root / 'config.json')
    ref = json.loads(gzip.decompress((repo / config['reference']).read_bytes()))
    points, origin = np.asarray(ref['landmarks_A']), np.asarray(ref['origin_A'])
    metrics = pd.read_parquet(repo / config['terminal_metrics'])
    train = metrics[metrics.batch.isin(config['donor_batches'])].copy()
    valid = train.valid_connected & train.pb_fast_pass & train.energy_status.eq('converged')
    strain_cap = float(train.loc[valid, 'mmff_relief_per_heavy'].quantile(.9))
    good = train[valid & (train.mmff_relief_per_heavy <= strain_cap)].sort_values('pic50_on_rescore', ascending=False)
    # Separate terminal ancestry modes by independent batch, so near clones
    # cannot fill the entire donor library.
    donors = good.drop_duplicates('batch').head(config['donors']).sample(frac=1, random_state=42)
    threshold = float(train.loc[valid, 'pic50_on_rescore'].quantile(.95))
    narrative = ['# 优质路径的逐0.01自然语言先验',
        '这是训练批次中质量合格终态的真实祖先轨迹，不含任何预测亲和力数值。路径代号随机分配，不代表质量排名。',
        '时间t表示模型输入/评分的当前时刻；proposal在t+0.01，重采样复制proposal后成为下一节点。所有前后比较按真实父子索引连接，槽位号不是永久分子身份。',
        '描述分别保留当前云和预测终点云。终点修订不是原子的实际运动；早期硬原子、硬键不等于有效化学图。没有每步MMFF能量标签，不能从距离推断作用能。',
        '既往Analyst的训练证据：短程pair_kernel_2.5常呈正向选择关联，部分固定轴形状分量较小；R00的壳层占据与其他距离场可以异向。后续时序Analyst发现局部方向会反复换号，不能把全窗口均值、最终教师或某一条路径当成所有早期seed的单一目标。这些是关联和假设，不能保证未来高亲和力。',
        '区域坐标锚点如下（高斯半径5 Å；壳层核以3 Å为中心、宽度1 Å；均为无量纲坐标密度代理）：']
    for r in config['regions']:
        narrative.append(f'- R{r:02d}：固定参考点 {points[r].tolist()} Å。')
    traces, donor_ids, sources = [], [], []
    for number, row in enumerate(donors.itertuples(index=False), 1):
        a, com, path = trajectory(root, row.batch)
        sources.append({'path': str(path.relative_to(repo)), 'sha256': digest(path)})
        anc = terminal_ancestors(a['selected_indices'])[:, row.slot]
        donor_ids.append({'label': f'P{number}', 'batch': row.batch, 'terminal_slot': row.slot,
                          'pic50': row.pic50_on_rescore, 'strain': row.mmff_relief_per_heavy, 'smiles': row.smiles})
        narrative.append(f'\n## 路径P{number}\n该路径由连续的真实父子边构成，包含复制和独立后续生成；下列描述不给出其选择次数或评分。')
        previous_f = None
        for step in range(config['last_step'] + 1):
            slot = int(anc[step])
            x = a['current_coords'][step, slot].astype(float) * cfg['coord_scale'] + com
            y = a['predicted_coords'][step, slot].astype(float) * cfg['coord_scale'] + com
            if step:
                parent = int(anc[step-1])
                if int(a['selected_indices'][step-1, slot]) != parent or not np.array_equal(a['current_coords'][step, slot], a['proposal_coords'][step-1, parent]):
                    raise ValueError('Ancestor coordinate/parent identity failed')
                px = a['current_coords'][step-1, parent].astype(float) * cfg['coord_scale'] + com
                py = a['predicted_coords'][step-1, parent].astype(float) * cfg['coord_scale'] + com
            else:
                px = py = None
            f = features(x, y, a['predicted_atomics'][step, slot], a['predicted_bonds'][step, slot], points, origin, px, py, config['regions'])
            delta = None if previous_f is None else {k: f[k]-previous_f[k] for k in f if isinstance(f[k], (float, int)) and isinstance(previous_f[k], (float, int))}
            narrative.append(f'\n### t={step*.01:.2f}\n' + sentence(f, changes=delta, regions=config['regions']))
            traces.append({'path': f'P{number}', 't': step*.01, 'batch': row.batch, 'slot': slot, **f})
            previous_f = f
    (blind / 'quality_paths.zh-CN.md').write_text('\n\n'.join(narrative) + '\n', encoding='utf-8')
    # Short stage-specific prototype files avoid forcing an agent to read all
    # 255 long descriptions into one context. They contain exactly the same
    # natural-language observations, never terminal labels or fitted scores.
    prior_header = '\n\n'.join(narrative[:10])
    (blind / 'description_basis.zh-CN.md').write_text(prior_header, encoding='utf-8')
    for step in range(1, config['last_step'] + 1):
        text = [f'# t={step*.01:.2f} 的五条描述先验']
        for number in range(1, len(donors)+1):
            rows = [r for r in traces if r['path'] == f'P{number}']
            previous, current = rows[step-1], rows[step]
            delta = {k:current[k]-previous[k] for k in current if isinstance(current[k],(float,int)) and isinstance(previous[k],(float,int))}
            text.append(f'## P{number}\n' + sentence(current, changes=delta, regions=config['regions']))
        (blind / f'prototypes_{step:03d}.zh-CN.md').write_text('\n\n'.join(text),encoding='utf-8')
    pd.DataFrame(traces).to_parquet(oracle / 'donor_path_features.parquet', compression=None, index=False)
    write_json(oracle / 'donors.json', {'donors': donor_ids, 'strain_cap': strain_cap, 'elite_threshold': threshold})
    all_steps = list(range(1,config['last_step']+1))
    pools = {step: [] for step in all_steps}
    oracle_rows, raw = [], {step: [] for step in config['start_steps']}
    for batch in config['test_batches']:
        a, com, path = trajectory(root, batch)
        sources.append({'path': str(path.relative_to(repo)), 'sha256': digest(path)})
        ancestors = terminal_ancestors(a['selected_indices'])
        m = metrics[metrics.batch.eq(batch)].sort_values('slot').reset_index(drop=True)
        if not np.array_equal(m.slot.to_numpy(), np.arange(50)):
            raise ValueError('Complete identified final attempts required')
        for step in all_steps:
            unique = {}
            for slot in range(50):
                key = fingerprint([a[f'current_{channel}'][step, slot] for channel in ['coords', 'atomics', 'bonds', 'charges']])
                unique.setdefault(key, []).append(slot)
            for key, aliases in unique.items():
                slot = aliases[0]
                anonymous = 'C' + hashlib.sha256(f'language-blind-42/{step}/{batch}/{key}'.encode()).hexdigest()[:10]
                x = a['current_coords'][step, slot].astype(float) * cfg['coord_scale'] + com
                y = a['predicted_coords'][step, slot].astype(float) * cfg['coord_scale'] + com
                parent = int(a['selected_indices'][step-1, slot])
                px = a['current_coords'][step-1, parent].astype(float) * cfg['coord_scale'] + com
                py = a['predicted_coords'][step-1, parent].astype(float) * cfg['coord_scale'] + com
                f = features(x, y, a['predicted_atomics'][step, slot], a['predicted_bonds'][step, slot], points, origin, px, py, config['regions'])
                previous_f = features(px, py, a['predicted_atomics'][step-1,parent], a['predicted_bonds'][step-1,parent],points,origin,regions=config['regions'])
                delta = {k:f[k]-previous_f[k] for k in f if isinstance(f[k],(float,int)) and isinstance(previous_f[k],(float,int))}
                parent_key = fingerprint([a[f'current_{channel}'][step-1,parent] for channel in ['coords','atomics','bonds','charges']])
                parent_id = 'C'+hashlib.sha256(f'language-blind-42/{step-1}/{batch}/{parent_key}'.encode()).hexdigest()[:10]
                pools[step].append({'id': anonymous, 'parent_id': parent_id, 'features': f, 'changes':delta})
                if step in raw:
                    raw[step].append({'id': anonymous, 'current_coords_A': x.tolist(), 'predicted_endpoint_coords_A': y.tolist(),
                    'predicted_atom_codes': a['predicted_atomics'][step, slot].tolist(),
                    'predicted_bond_codes': a['predicted_bonds'][step, slot].tolist()})
                descendants = np.flatnonzero(np.isin(ancestors[step], aliases))
                finals = m.iloc[descendants]
                quality = finals.valid_connected & finals.pb_fast_pass & finals.energy_status.eq('converged') & (finals.mmff_relief_per_heavy <= strain_cap)
                q = finals.loc[quality]
                oracle_rows.append({'id': anonymous, 'step': step, 'batch': batch, 'slot': slot, 'aliases': aliases,
                    'current_state_sha256': key, 'descendant_slots': descendants.tolist(),
                    'n_descendants': len(descendants), 'n_quality_descendants': len(q),
                    'future_observed': bool(len(descendants)), 'quality_future_max_pic50': float(q.pic50_on_rescore.max()) if len(q) else None,
                    'quality_future_mean_pic50': float(q.pic50_on_rescore.mean()) if len(q) else None,
                    'has_elite_descendant': bool(len(q) and (q.pic50_on_rescore >= threshold).any()),
                    'intermediate_pic50': float(a['pic50_on'][step, slot]), 'parent_id':parent_id,'features': f})
    for step, records in pools.items():
        records.sort(key=lambda r: r['id'])
        cards = [f'# t={step*.01:.2f} 的匿名状态\n共{len(records)}个不同当前状态。列表顺序仅按匿名ID，未按任何评分排序。每张卡只包含此刻结构和已发生的前一0.01变化，没有未来状态。']
        for record in records:
            cards.append('## ' + record['id'] + '\n已发生的父子边来自 '+record['parent_id']+'。\n' + sentence(record['features'], changes=record['changes'], regions=config['regions']))
        (blind / f'states_{step:03d}.zh-CN.md').write_text('\n\n'.join(cards) + '\n', encoding='utf-8')
        if step in raw:
            write_json(blind / f'structures_{step:03d}.json', {'t': step*.01, 'regions_A': {f'R{r:02d}': points[r].tolist() for r in config['regions']}, 'states': sorted(raw[step], key=lambda r:r['id'])})
    write_json(oracle / 'candidate_truth.json', {'elite_threshold': threshold, 'strain_cap': strain_cap, 'candidates': oracle_rows})
    protocol = {'schema_version': 'language-path-blind-1.1', 'start_steps': config['start_steps'], 'available_steps':all_steps,
        'ranks_per_start': 5, 'pool_sizes': {str(s):len(r) for s,r in pools.items()},
        'prior': 'quality_paths.zh-CN.md', 'sequential_files': [f'states_{s:03d}.zh-CN.md' for s in config['start_steps']],
        'blinding': 'Do not read any files outside the supplied blind directory. No affinity, selection probability, clone multiplicity, future survival or source identity is supplied.',
        'test_scope': 'Retrospective held-out path recognition, not a new generation or counterfactual affinity test',
        'clock': 't is current input/score time; prior-step proposal becomes current after selection',
        'continuation': 'At each start, rank five anonymous distinct states. Then advance by 0.01; only children of retained directions are eligible, keep at most five. If none are observed, stop: do not manufacture descendants. These are observed Steer continuations, not new generation. Read each future state file only after sealing the preceding ranking.',
        'rank_output': {'t': 'float', 'ranking': [{'rank': '1..5', 'candidate_id': 'existing anonymous ID', 'matched_description': 'observable evidence and prototype/time citations', 'counterevidence': 'observed mismatches', 'confidence': 'low/medium/high'}], 'method_summary': 'concise public decision rule, not private reasoning'}}
    write_json(blind / 'protocol.json', protocol)
    write_json(out / 'manifest.json', {'config': config, 'sources': sources, 'code_sha256': digest(__file__),
        'donor_batches': config['donor_batches'], 'test_batches': config['test_batches'],
        'source_separation': not bool(set(config['donor_batches']) & set(config['test_batches'])),
        'blind_files': {p.name:digest(p) for p in blind.iterdir() if p.is_file()},
        'node_semantics': 'Exact state clones merged within batch and time; equality includes coords, atomics, bonds and charges',
        'oracle_not_delivered': True, 'future_labels_are_Steer_conditioned': True})
    print(json.dumps({'output':str(out),'pool_sizes':protocol['pool_sizes'],'donor_paths':len(donors)},ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    a = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    prepare(repo, read_json(repo/a.config))
