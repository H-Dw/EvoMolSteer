"""Read saved endpoints on CPU and fit the same nine matched condition atoms.

Correspondence is chosen in the original protein frame and held fixed during
proper Kabsch fitting. No iterative rematching, model inference, or source edit.
"""
from pathlib import Path
import argparse
import collections
import csv
import datetime
import gzip
import hashlib
import io
import json
import math
import statistics
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, data):
    Path(path).write_bytes((json.dumps(data, ensure_ascii=True, indent=2,
                                      allow_nan=False) + '\n').encode('utf-8'))


def fit(moving, reference):
    """Row vectors, equal weights, rotation determinant +1; no scaling."""
    p, q = np.asarray(moving, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    assert p.shape == q.shape and p.ndim == 2 and p.shape[1] == 3
    assert np.isfinite(p).all() and np.isfinite(q).all()
    pc, qc = p - p.mean(0), q - q.mean(0)
    u, _, vt = np.linalg.svd(pc.T @ qc)
    correction = np.diag([1., 1., np.linalg.det(u @ vt)])
    rotation = u @ correction @ vt
    translation = q.mean(0) - p.mean(0) @ rotation
    aligned = p @ rotation + translation
    rmsd = float(np.sqrt(np.mean(np.sum((aligned - q) ** 2, axis=1))))
    return rmsd, rotation, translation, aligned


def self_test():
    # Local RNG only, unrelated to the stored scientific streams.
    rng = np.random.default_rng(20261101)
    p = rng.normal(size=(9, 3))
    rotation = Rotation.from_rotvec([.4, -.7, .2]).as_matrix().T
    q = p @ rotation + [3., -2., 1.]
    error, r, t, _ = fit(p, q)
    assert error < 1e-12 and np.max(np.abs(p @ r + t - q)) < 1e-12
    mirror = p * [-1., 1., 1.]
    error, r, _, _ = fit(p, mirror)
    assert abs(np.linalg.det(r) - 1.) < 1e-12 and error > .1
    error, _, _, _ = fit(p, p * 2.)
    assert error > .1  # Scaling is forbidden.
    return dict(rigid_transform_recovered=True, reflection_forbidden=True,
                scaling_forbidden=True, fixed_correspondence=True)


def outliers(values):
    finite = np.asarray([v for v in values if v is not None], dtype=float)
    if len(finite) < 10:
        return [False] * len(values), True, None
    q1, q3 = np.quantile(finite, [.25, .75], method='linear')
    upper = float(q3 + max(3 * (q3 - q1), 1.))
    flags = [v is not None and v > upper for v in values]
    return flags, sum(flags) <= .1 * len(finite) + 1e-12, upper


def stats(values, excluded=None, eligible=True):
    good = [(i, float(v)) for i, v in enumerate(values) if v is not None and math.isfinite(v)]
    if not good:
        return dict(n=0, missing=len(values), mean=None, best=None, filtered_mean=None,
                    top10_mean=None, top10_n=0, excluded=0, filter_eligible=eligible)
    excluded = excluded or [False] * len(values)
    vals = [v for _, v in good]
    kept = [v for i, v in good if not excluded[i]]
    k = math.ceil(.1 * len(vals))
    return dict(n=len(vals), missing=len(values)-len(vals), mean=statistics.mean(vals),
                best=min(vals), filtered_mean=statistics.mean(kept) if kept and eligible else None,
                top10_mean=statistics.mean(sorted(vals)[:k]), top10_n=k,
                excluded=sum(excluded), filter_eligible=eligible)


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True)
    ap.add_argument('--rows', required=True)
    ap.add_argument('--paired', required=True)
    ap.add_argument('--best', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--storage-src', required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    tracked = {}

    def protect(path, expected=None):
        path = Path(path)
        assert 'launcher.log' not in path.name and not path.name.startswith('core.')
        item = dict(size=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns, sha256=sha(path))
        if expected:
            assert item['sha256'] == expected, str(path)
        if str(path) in tracked:
            assert tracked[str(path)] == item
        tracked[str(path)] = item
        return path

    for path in [args.source, args.rows, args.paired, args.best]:
        protect(path)
    source, paired, best = read(args.source), read(args.paired), read(args.best)
    teachers_path = protect('/workspace/agenticgen_uniguide_flowr_20261026/teachers.json')
    references = {t['target']: np.asarray(t['core_A'], dtype=np.float64) for t in read(teachers_path)}
    assert all(ref.shape == (9, 3) and np.linalg.matrix_rank(ref-ref.mean(0)) >= 2
               for ref in references.values())
    with gzip.open(args.rows, 'rt', encoding='utf-8') as stream:
        old_rows = [json.loads(line) for line in stream]
    group_results, view_rows, max_raw_difference = [], {}, 0.
    max_scipy_difference, max_increase, rank_deficient = 0., 0., 0
    validation = self_test()

    def evaluate(coords, valid, target, previous=None, independent=False):
        nonlocal max_raw_difference, max_scipy_difference, max_increase, rank_deficient
        ids = np.flatnonzero(valid)
        xyz = np.asarray(coords, dtype=np.float64)
        if len(ids) < 9 or not np.isfinite(xyz[ids]).all():
            return dict(raw_rmsd=None, aligned_rmsd=None, missing_reason='invalid_endpoint')
        ref = references[target]
        cost = ((ref[:, None] - xyz[ids][None]) ** 2).sum(-1)
        rows, columns = linear_sum_assignment(cost)
        slots = ids[columns[np.argsort(rows)]]
        matched = xyz[slots]
        raw = float(np.sqrt(np.mean(cost[rows, columns])))
        if previous is not None:
            max_raw_difference = max(max_raw_difference, abs(raw-previous))
            assert abs(raw-previous) < 2e-5
        error, rotation, translation, aligned = fit(matched, ref)
        assert np.max(np.abs(rotation.T @ rotation - np.eye(3))) < 1e-10
        assert abs(np.linalg.det(rotation)-1.) < 1e-10
        max_increase = max(max_increase, error-raw)
        assert error <= raw + 1e-10
        rank = int(np.linalg.matrix_rank(matched-matched.mean(0)))
        rank_deficient += int(rank < 2)
        if independent and rank >= 2:
            rot, _ = Rotation.align_vectors(ref-ref.mean(0), matched-matched.mean(0))
            independent_error = float(np.sqrt(np.mean(np.sum(
                (rot.apply(matched-matched.mean(0))-(ref-ref.mean(0)))**2, axis=1))))
            max_scipy_difference = max(max_scipy_difference, abs(independent_error-error))
            assert abs(independent_error-error) < 1e-9
        return dict(raw_rmsd=raw, aligned_rmsd=error, condition_slots=slots.tolist(),
                    rotation_row_vectors=rotation.tolist(), translation_angstrom=translation.tolist(),
                    matched_coordinates_angstrom=matched.tolist(),
                    aligned_condition_coordinates_angstrom=aligned.tolist(), moving_rank=rank)

    def add_group(metadata, candidates):
        raw = [r['raw_rmsd'] for r in candidates]
        aligned = [r['aligned_rmsd'] for r in candidates]
        ff, eligible, upper = outliers(aligned)
        rf, re, ru = outliers(raw)
        for i, candidate in enumerate(candidates):
            candidate.update(aligned_excluded=ff[i], aligned_filter_eligible=eligible)
        group_results.append(dict(**metadata, stats=stats(aligned, ff, eligible),
                                  raw_stats=stats(raw, rf, re), aligned_upper_fence=upper,
                                  raw_upper_fence=ru))

    for number, group in enumerate(source['groups'], 1):
        folder = Path(group['folder'])
        protect(folder/'records.json', group['record_sha256'])
        terminal_path = protect(folder/'terminal.pt.gz', group['terminal_sha256'])
        protect(folder/'COMPLETE.json')
        terminal = torch.load(io.BytesIO(gzip.decompress(terminal_path.read_bytes())),
                              map_location='cpu', weights_only=False)
        coords = terminal['coords'].double().numpy()
        masks = terminal['mask'].bool().numpy()
        assert len(coords) == len(masks) == group['population']
        rows = [dict(group=group['folder'], index=i, **evaluate(x, m, group['teacher'], independent=i==0))
                for i, (x, m) in enumerate(zip(coords, masks))]
        add_group({k:group[k] for k in ['experiment','teacher','phase','folder','population',
                                      'record_sha256','terminal_sha256','schedule']}, rows)
        view_rows.update({(r['group'], r['index']):r for r in rows})
        assert abs(group_results[-1]['raw_stats']['mean']-group['stats']['rmsd']['mean']) < 1e-10
        if number % 100 == 0:
            print(json.dumps(dict(groups_completed=number), ensure_ascii=True), flush=True)

    selected = []
    for original in old_rows:
        result = view_rows[(original['group'], original['index'])]
        assert abs(result['raw_rmsd']-original['rmsd']) < 1e-10
        selected.append(dict(**{k:original[k] for k in ['experiment','teacher','phase',
                               'physical_id','group','index','seed','slot']},
                             **{k:v for k,v in result.items() if k not in ['group','index']}))

    # Read the 100 historical reference endpoints using the original lossless decoder.
    sys.path.insert(0, args.storage_src)
    from evomolsteer.storage.trajectory import TrajectoryPackage
    reference_rows = []
    for experiment in source['experiments']:
        if experiment['phase'] != '历史参考':
            continue
        path = protect(experiment['reference_source'], experiment['reference_sha256'])
        archive = read(path)
        refrows, cache = [], {}
        for row in archive['records']:
            if not row['matched_formal_population']:
                continue
            h5 = next(p for p in archive['source_hashes']
                      if f"batch_{row['batch']:03d}/terminal.h5" in p)
            if h5 not in cache:
                protect(h5, archive['source_hashes'][h5])
                with TrajectoryPackage(h5) as package:
                    cache[h5] = (package.read('coords', 0), package.read('mask', 0).astype(bool))
            coords, mask = cache[h5]
            value = evaluate(coords[row['slot']], mask[row['slot']], experiment['teacher'],
                             row['condition_rmsd_A'], independent=True)
            value.update(experiment=experiment['experiment'], teacher=experiment['teacher'],
                         phase=experiment['phase'], group=str(path), index=len(refrows),
                         physical_id=f"{h5}:{row['slot']}", seed=row['seed'], slot=row['slot'])
            refrows.append(value)
        assert len(refrows) == 50
        add_group(dict(experiment=experiment['experiment'], teacher=experiment['teacher'],
                       phase=experiment['phase'], folder=str(path), population=50,
                       record_sha256=sha(path), terminal_sha256=None, schedule=None), refrows)
        reference_rows.extend(refrows)
    selected.extend(reference_rows)
    assert len(selected) == source['all_unique_saved_candidates_with_historical_reference'] == 19135

    buckets = collections.defaultdict(list)
    for row in selected:
        buckets[(row['experiment'], row['teacher'], row['phase'])].append(row)
    experiments = []
    old_experiments = {(e['experiment'],e['teacher'],e['phase']):e for e in source['experiments']}
    for key, rows in buckets.items():
        previous = old_experiments[key]
        assert len(rows) == previous['unique_saved_candidates']
        experiments.append(dict(experiment=key[0], teacher=key[1], phase=key[2],
                                population=len(rows), raw_stats=previous['stats']['rmsd'],
                                stats=stats([r['aligned_rmsd'] for r in rows],
                                            [r['aligned_excluded'] for r in rows],
                                            all(r['aligned_filter_eligible'] for r in rows))))
    assert len(experiments) == len(source['experiments']) == 79
    by_group = {g['folder']:g for g in group_results}
    paired_results = []
    for row in paired:
        if row.get('source'):
            group = by_group[row['source']]
        else:
            group = next(g for g in group_results if g['experiment']==row['experiment']
                         and g['teacher']==row['teacher'])
        paired_results.append(dict(experiment=row['experiment'], teacher=row['teacher'],
                                   population=group['population'], folder=group['folder'],
                                   raw_stats=group['raw_stats'], stats=group['stats']))
    best_results = [dict(experiment=t['strategy'], teacher=t['teacher'], suite=t['suite'],
                         selected_round=t['selected_round'],
                         **{k:by_group[t['best_configuration']['folder']][k]
                            for k in ['folder','population','raw_stats','stats']})
                    for t in best['selected']]
    # Verify all inputs and scientific endpoints are still byte-identical.
    for path, expected in tracked.items():
        p = Path(path)
        assert expected == dict(size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns, sha256=sha(p)), path
    validation.update(source_files_unchanged=len(tracked), completed_groups=len(source['groups']),
                      historical_reference_groups=len(reference_rows)//50,
                      unique_saved_candidates=len(selected), experiment_rows=len(experiments),
                      max_raw_rmsd_difference=max_raw_difference,
                      max_independent_scipy_difference=max_scipy_difference,
                      max_alignment_rmsd_increase=max_increase,
                      rank_deficient_candidate_views=rank_deficient,
                      reflection_used=False, scale_fitting_used=False, rematching_after_alignment=False,
                      new_sampling=0, new_model_forwards=0, new_llm_calls=0)
    result = dict(timestamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
                  definition='Original nine-PLIP-heavy-atom geometric Hungarian correspondence in protein frame, then fixed-correspondence proper equal-weight Kabsch fit and RMSD on those same nine atoms; no reflection/scaling/rematching',
                  interpretation='Internal shape preservation after removing global translation and rotation; not pocket pose retention, chemical identity, or an independent validation',
                  outlier_policy='Per complete configuration: minimum10 finite, upper Q3+max(3IQR,1Angstrom), maximum10percent; recomputed for aligned values; top10 is best ceil(0.1*n) from unfiltered finite values',
                  experiments=experiments, groups=group_results, paired_fixed_scale=paired_results,
                  completed_job_best_configurations=best_results, validation=validation)
    write(out/'alignment_results.json', result)
    write(out/'source_preservation.json', tracked)
    write(out/'validation.json', validation)
    with gzip.open(out/'aligned_candidates.jsonl.gz', 'wt', encoding='utf-8') as stream:
        for row in selected:
            stream.write(json.dumps(row, ensure_ascii=True, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(out), **validation), ensure_ascii=True), flush=True)


if __name__ == '__main__':
    main()
