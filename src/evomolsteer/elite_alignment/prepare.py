"""Build final-ranked cohorts and lossless, deduplicated initial-state inputs."""
import argparse
import gzip
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from evomolsteer.io import read_json, digest
from evomolsteer.storage.transactions import atomic_json
from evomolsteer.agentic_steer.native import slice_tree
from .common import identity, validate_protocol, validate_condition

NAME = 'elite_initial_alignment_20261017'


def prepare(repo):
    repo = Path(repo).resolve()
    directory = repo / 'configs/experiments' / NAME
    directory.mkdir(parents=True, exist_ok=True)
    work = repo / 'data/generated' / NAME
    raw = repo / 'data/raw/ck2_clk3_lineage_20261003/results/main1000_w050/single'
    records = [r for p in sorted(raw.glob('batch_*/final_records.json')) for r in read_json(p)]
    if len(records) != 1000 or not all(np.isfinite(r['pic50_on_rescore']) for r in records):
        raise ValueError('Ranking universe must contain all 1000 finite final scores')
    records.sort(key=lambda r: (-r['pic50_on_rescore'], r['batch'], r['slot']))
    source_hashes = {str(p.relative_to(repo)).replace('\\', '/'): digest(p) for p in raw.glob('batch_*/final_records.json')}
    initials, attempts, cache = {}, [], {}
    for rank, row in enumerate(records[:200], 1):
        batch, root = row['batch'], row['root_slot']
        key = f'b{batch:03d}_root{root:03d}'
        if key not in initials:
            source = raw / f'batch_{batch:03d}' / 'initial_state.pt.gz'
            if batch not in cache:
                with gzip.open(source, 'rb') as stream:
                    cache[batch] = torch.load(stream, map_location='cpu', weights_only=False)
                with gzip.open(source.parent / 'restart_step_000.pt.gz', 'rb') as stream:
                    restart = torch.load(stream, map_location='cpu', weights_only=False)
                for field in ('coords', 'atomics', 'bonds', 'charges', 'mask'):
                    if not torch.equal(cache[batch]['prior'][field], restart['current'][field]):
                        raise ValueError('Saved prior differs from actual step-zero current')
            saved = cache[batch]
            b = len(saved['prior']['coords'])
            state = {k: slice_tree(saved[k], root, b, 'cpu') for k in ('prior', 'pocket_target')}
            state.update(target_com=np.asarray(saved['target_com'])[root:root+1], coord_scale=saved['coord_scale'])
            output = work / 'inputs/initials' / (key + '.pt.gz')
            output.parent.mkdir(parents=True, exist_ok=True)
            if not output.exists():
                with output.open('wb') as target, gzip.GzipFile(fileobj=target, mode='wb', filename='', mtime=0) as stream:
                    torch.save(state, stream)
            else:
                with gzip.open(output, 'rb') as stream:
                    previous = torch.load(stream, map_location='cpu', weights_only=False)
                for field in ('coords', 'atomics', 'bonds', 'charges', 'mask'):
                    if not torch.equal(previous['prior'][field], state['prior'][field]):
                        raise ValueError('An existing exported initial differs')
            initials[key] = {'file': str(output.relative_to(work)).replace('\\', '/'), 'sha256': digest(output),
                             'source': str(source.relative_to(repo)).replace('\\', '/'), 'source_sha256': digest(source),
                             'batch': batch, 'root_slot': root}
        seed = int(np.random.SeedSequence([42, batch, root, row['slot']]).generate_state(1)[0])
        attempts.append({'id': f'F_b{batch:03d}_s{row["slot"]:03d}', 'rank': rank, 'batch': batch,
                         'final_slot': row['slot'], 'root_slot': root, 'initial_id': key, 'suffix_seed': seed,
                         'historical_pic50': row['pic50_on_rescore'],
                         'historical_valid_connected': bool(row.get('build_success') and row.get('connected')),
                         'cohorts': [f'top{q:02d}' for q, n in ((5, 50), (10, 100), (20, 200)) if rank <= n]})

    conditions = [{'id': 'unguided', 'kind': 'unguided', 'window': None}]
    programs = repo / 'docs/experiments/m2_tools_full_drop_ablation_20261012/programs'

    def add(cid, program, reference, commit, kind='legacy'):
        program, reference = Path(program), Path(reference)
        p = read_json(repo / program)
        conditions.append({'id': cid, 'kind': kind, 'program': program.as_posix(), 'program_sha256': digest(repo / program),
                           'reference': reference.as_posix(), 'reference_sha256': digest(repo / reference),
                           'historical_commit': commit, 'window': p['window']})

    signatures = {'M2': ('c62e296818e05c479dc68bee5d34f24ef0871cab3bc3e6204f442a9b6dfe40cd', 'ce85fde5c5ef'),
                  'blend_0p5': ('ab41341eeb9b52bacac60af419cf96810790098b59bb821a5c7fb4f776099085', '6b72efa5d5f5'),
                  'blend_0p25': ('6d42e0d667e4a642f83cc5672dcbdef92dc144e93502c4fa8427cb09dff74cf1', '317ac8bbe963'),
                  'linear_decay': ('aa26106a518fdde2575c25176e3009ee6fc1edb0c6f6f353030d3b91e66e81b6', '0f452b8ff2d2')}
    static = Path('configs/experiments/steer_dependency_v1/static.json.gz')
    temporal = Path('docs/experiments/compact_skills_verification_20261012/evidence/temporal_reference.json.gz')
    add('M2', (programs / (signatures['M2'][0] + '.json')).relative_to(repo), static, signatures['M2'][1])
    top = Path('configs/experiments/m2_terminal_regions_20261014/programs')
    add('top10_global_0', top / 'terminal_topk_global_0_0.5.json', top / 'reference_0_0.5.json.gz', 'e4549af1a960')
    jobs = Path('configs/experiments/temporal_guidance_run_20261013/jobs')
    temporal_cases = [('S3', '3255dfbc980c', 'a3c8685c9935'), ('C3', '20b868bcf02b', '55ec551c6544'),
                      ('C2', '572c774209da', '6663f2c1a36d'), ('W1', 'e8a8ad052151', '5f7a5ec3d97a'),
                      ('W3', '1f217373730e', '2be38b93ed9e'), ('S1', '4edf30248b26', '07d255cfe84d')]
    for cid, sig, commit in temporal_cases:
        job = read_json(repo / jobs / f'DSF_{sig}_b6376_g_indexed_batch.json')
        add(cid, job['program'], job['reference'], commit)
    for cid in ('blend_0p5', 'blend_0p25', 'linear_decay'):
        sig, commit = signatures[cid]
        add(cid, (programs / (sig + '.json')).relative_to(repo), temporal if cid.startswith('blend') else static, commit)
    add('top10_global_0p1', top / 'terminal_topk_global_0.1_0.5.json', top / 'reference_0.1_0.5.json.gz', 'e4549af1a960')
    add('top10_local_0p1', top / 'terminal_topk_local_0.1_0.5.json', top / 'reference_0.1_0.5.json.gz', 'e4549af1a960')
    for cid, study, roundid, commit in [('NC_R11', 'nc_autoresearch20_20261015', 'R11', '78d31a914fed'),
                                       ('motif_R15_v2', 'local_motif20_v2_20261015', 'R15_v2', 'dbbac1158f89'),
                                       ('ancestor_R02_v3', 'ancestral_modes35_v3_20261016', 'R02_v3', '46218d7950f7')]:
        folder = repo / 'docs/experiments' / study / roundid
        binding = read_json(next((folder / 'candidates').glob('*/bindings.json')))
        item = next(x for x in binding['programs'] if read_json(repo / x['program'])['program_id'] == roundid)
        add(cid, item['program'], binding['teacher'], commit, kind='native_local')
    # Put controls first, then frozen historical strategies in the agreed order.
    order = ['unguided', 'M2', 'top10_global_0', 'S3', 'blend_0p5', 'C3', 'blend_0p25', 'C2',
             'W1', 'W3', 'S1', 'linear_decay', 'top10_global_0p1', 'top10_local_0p1', 'NC_R11', 'motif_R15_v2', 'ancestor_R02_v3']
    conditions.sort(key=lambda x: order.index(x['id']))
    config = {'schema_version': 'elite-initial-alignment-1', 'name': NAME, 'master_seed': 42, 'steps': 100,
              'checkpoint': '/root/private_data/MolSteer/flowr_root/checkpoints/flowr_root_v2.ckpt',
              'checkpoint_sha256': 'f28e863b2b208718f3d3f85f09837c2f907a71123436db6f98a25d2f1858b6a0',
              'flowr_root': '/root/private_data/MolSteer/flowr_root',
              'protein': '/root/private_data/MolSteer/flowr_root/experiments/ck2_clk3_lineage_20261003/inputs/3PE1_protein_aligned.pdb',
              'remote_work': '/root/private_data/MolSteer/flowr_root/experiments/evomolsteer_' + NAME,
              'native_options': {'cat_noise_euler_guard': False, 'cat_noise_level': 1., 'coord_noise_level': .2,
                                 'final_corr_pred': True, 'use_sde_simulation': True},
              'sample_timeout_seconds': 1800, 'startup_timeout_seconds': 1800, 'poll_seconds': 3,
              'max_worker_restarts': 1, 'minimum_free_bytes': 128 * 1024**2,
              'tail_threshold': 8.258901977539063, 'steer_best': 8.704806327819824,
              'ranking_sources': source_hashes, 'initials': initials, 'attempts': attempts, 'conditions': conditions,
              'cohorts': {f'top{q:02d}': {'n': n, 'cutoff': attempts[n-1]['historical_pic50'],
                                         'unique_roots': len({x['initial_id'] for x in attempts[:n]})}
                          for q, n in ((5, 50), (10, 100), (20, 200))},
              'interpretation': 'Retrospectively selected roots; cloned roots and nested cohorts are dependent; new paired suffix RNG, not historical replay.'}
    config['protocol_sha256'] = identity(config)
    validate_protocol(config)
    for condition in conditions:
        validate_condition(repo, condition)
    output = directory / 'config.json'
    atomic_json(output, config)
    atomic_json(directory / 'preparation.json', {'initial_count': len(initials), 'attempts': len(attempts),
                'conditions': len(conditions), 'formal_inferences': len(attempts) * len(conditions), 'cohorts': config['cohorts'],
                'input_bytes': sum((work / x['file']).stat().st_size for x in initials.values()), 'new_inference': False})
    return output


def main():
    p = argparse.ArgumentParser(); p.add_argument('--repo', default='.')
    args = p.parse_args(); print(prepare(args.repo))


if __name__ == '__main__':
    main()
