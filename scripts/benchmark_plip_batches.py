#!/usr/bin/env python3
"""Diagnostic native inference capacity probe; never changes a prepared study.

Uses the real 100-step loop, coordinate VJP, trace/checkpoint writes and final
rescoring. Wider batches concatenate historical twenty-slot blocks. Their RNG
is deliberately NOT claimed equivalent to the original five independent streams.
No diagnostic molecule is admitted to the scientific result matrix.
"""
import argparse
import contextlib
import gc
import json
import os
from pathlib import Path
import shutil
import statistics
import threading
import time
from types import SimpleNamespace


def join_blocks(values, torch):
    first = values[0]
    if torch.is_tensor(first):
        if first.ndim and first.shape[0] == 20:
            return torch.cat(values, 0)
        if not all(torch.equal(first, v) for v in values[1:]):
            raise ValueError("Non-batch tensor differs")
        return first.clone()
    if isinstance(first, dict):
        return {k: join_blocks([v[k] for v in values], torch) for k in first}
    if isinstance(first, list) and len(first) == 20:
        return [x for v in values for x in v]
    if any(v != first for v in values[1:]):
        raise ValueError("Non-batch metadata differs")
    return first


def median_cases(rows):
    result = []
    for teacher, size in sorted({(r['teacher'], r['batch_size']) for r in rows}):
        group = [r for r in rows if (r['teacher'], r['batch_size']) == (teacher, size)]
        ok = [r for r in group if r['status'] == 'ok']
        result.append({'teacher': teacher, 'batch_size': size, 'passed': len(ok) == len(group),
                       'repeats': len(group),
                       'seconds_median': statistics.median(r['seconds'] for r in ok) if ok else None,
                       'peak_allocated_GiB': max((r['peak_allocated_GiB'] for r in ok), default=None),
                       'peak_reserved_GiB': max((r['peak_reserved_GiB'] for r in ok), default=None),
                       'peak_device_GiB': max((r['peak_device_GiB'] for r in ok), default=None)})
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--sizes', type=int, nargs='+', default=[20, 40, 60, 80, 100])
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--device-sysfs', type=Path, default=Path('/sys/class/drm/renderD133/device'))
    args = p.parse_args()
    if args.repeats < 1 or any(b not in range(20, 101, 20) for b in args.sizes):
        p.error('Use positive repeats and batch sizes 20,40,60,80,100')
    if args.output.exists():
        p.error('Output must be a new directory; preserve all prior probes')
    import fcntl
    import numpy as np
    import torch
    from evomolsteer.adaptive_schedule.flowr import (
        Control, load_initial, load_model, load_pt, device, cpu, tensor_digest, save_pt)
    from evomolsteer.adaptive_schedule.io import read, write, sha, commit, source_fingerprint
    from evomolsteer.generation.rng import set_rng, rng_state
    from evomolsteer.generation.controller import final_metrics

    args.output.mkdir(parents=True)
    # Both locks remain held throughout, excluding formal launch during diagnostics.
    with contextlib.ExitStack() as stack:
        for name in ['supervisor.lock', 'worker.lock']:
            f = stack.enter_context((args.source_root / name).open('a'))
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = read(args.source_root / 'config.json')
        teachers = [t for t in read(args.source_root / 'teachers.json') if t['rank'] in (1, 2)]
        if [t['rank'] for t in teachers] != [1, 2]:
            raise ValueError('Exactly the original Top1 and Top2 are required')
        write(args.output / 'manifest.json', {
            'diagnostic_only': True, 'source_root': str(args.source_root), 'commit': commit(),
            'source_fingerprint': source_fingerprint(), 'config_sha256': sha(args.source_root / 'config.json'),
            'teacher_ids': [t['teacher_id'] for t in teachers], 'sizes': args.sizes, 'repeats': args.repeats,
            'torch': torch.__version__, 'hip': torch.version.hip, 'device': torch.cuda.get_device_name(0),
            'device_total_GiB': torch.cuda.get_device_properties(0).total_memory / 2**30,
            'rng_policy': 'diagnostic dense batch uses batch-zero entry RNG; not a scientific replacement',
            'timing': 'native encoding + 100 full steps + gradients + trace/checkpoints + final rescore; excludes PLIP/LLM',
            'sysfs_device': str(args.device_sysfs.resolve())})
        model = load_model(config, teachers[0]['target'], args.output)
        initials = [load_initial(config, teachers[0]['target'], b)[0] for b in range(5)]
        rows = []

        def run_case(template, size, repeat, warmup=False):
            folder = args.output / f"top{template['rank']}_b{size}_r{repeat}"
            folder.mkdir()
            initial = dict(initials[0])
            for key in ['prior', 'pocket_target', 'pocket_off', 'target_com']:
                initial[key] = join_blocks([s[key] for s in initials[:size // 20]], torch)
            prior, pt, po = (device(initial[k]) for k in ['prior', 'pocket_target', 'pocket_off'])
            for pocket in [pt, po]:
                pocket['complex'] = [SimpleNamespace(com=x.numpy()) for x in initial['target_com']]
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            peak_device = [0]
            stop = threading.Event()

            def monitor():
                while not stop.is_set():
                    try:
                        peak_device[0] = max(peak_device[0], int((args.device_sysfs / 'mem_info_vram_used').read_text()))
                    except OSError:
                        pass
                    stop.wait(.02)

            watcher = threading.Thread(target=monitor, daemon=True)
            watcher.start()
            control = Control(model, template, initial, [[5]*4 for _ in range(size)], folder, folder / 'context.pt.gz')
            model._adaptive = control
            set_rng(initial['rng'])
            torch.cuda.synchronize()
            started = time.perf_counter()
            row = {'teacher': template['teacher_id'], 'batch_size': size, 'repeat': repeat, 'warmup': warmup}
            try:
                output = model._generate_selective(prior=prior, pocket_data_target=pt, pocket_data_untarget=po,
                    steps=100, times=[torch.zeros(size, device='cuda') for _ in range(3)],
                    apply_guidance=False, coord_noise_level=.2)
                native_rng = cpu(rng_state())
                save_pt(folder / 'terminal.pt.gz', output)
                trace = SimpleNamespace(pt=pt, po=po, batch=0, seed=42, arm='capacity_probe',
                                        roots=np.arange(size), parents=np.arange(size), events=list(range(100)))
                records = final_metrics(model, cpu(output), trace, folder)
                write(folder / 'final_records.json', records)
                torch.cuda.synchronize()
                row.update(status='ok', seconds=time.perf_counter()-started, attempts=len(records),
                           output_sha256=tensor_digest(output), rng_sha256=tensor_digest(native_rng),
                           all_coordinates_finite=bool(torch.isfinite(output['coords']).all()),
                           all_gradients_finite=True, affinity_vjp=False,
                           max_guidance_A=max(max(r['max_atom_guidance_A']) for r in control.rows),
                           output_first20_sha256=tensor_digest({k:v[:20] for k,v in output.items() if torch.is_tensor(v)}))
                if len(records) != size or not row['all_coordinates_finite']:
                    raise ValueError('Incomplete or nonfinite result')
                old = args.source_root / 'shared_r0' / template['teacher_id'] / 'batch_000'
                # Discover the frozen R0 via history, without assuming a folder convention.
                history = args.source_root / 'versions/v1' / template['teacher_id'] / 'batch_000/histories.json'
                if size == 20 and history.exists():
                    old = args.source_root / read(history)[0]['history'][0]['branch']
                    row['matches_frozen_r0_terminal'] = tensor_digest(output) == tensor_digest(load_pt(old/'terminal.pt.gz'))
                    row['matches_frozen_r0_rng'] = tensor_digest(native_rng) == tensor_digest(load_pt(old/'rng_terminal.pt.gz'))
            except torch.OutOfMemoryError:
                row.update(status='oom', seconds=time.perf_counter()-started)
            finally:
                control.close()
                stop.set()
                watcher.join()
                row.update(peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                           peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,
                           peak_device_GiB=peak_device[0]/2**30)
                write(folder / 'benchmark.json', row)
                del model._adaptive
                model.integrator._adaptive_observer = None
            # Only our diagnostic payloads, after recording hashes; keep all JSON evidence.
            removed = []
            for path in folder.rglob('*'):
                if path.is_file() and path.suffix.lower() not in {'.json', '.xml'}:
                    if not path.resolve().is_relative_to(folder.resolve()):
                        raise ValueError('Payload escaped diagnostic folder')
                    removed.append({'path': str(path.relative_to(folder)), 'bytes': path.stat().st_size, 'sha256': sha(path)})
                    path.unlink()
            write(folder / 'diagnostic_cleanup.json', removed)
            return row

        # Absorb kernel initialization without counting it as a measured repeat.
        warm = run_case(teachers[0], 20, -1, True)
        write(args.output / 'warmup.json', warm)
        for size in args.sizes:
            for template in teachers:
                for repeat in range(args.repeats):
                    row = run_case(template, size, repeat)
                    rows.append(row)
                    write(args.output / 'results.json', rows)
                    write(args.output / 'summary.json', median_cases(rows))
                    print(json.dumps(row), flush=True)
                    if row['status'] != 'ok':
                        break
                if row['status'] != 'ok':
                    break
            if row['status'] != 'ok':
                break
        write(args.output / 'COMPLETE.json', {'measured_cases': len(rows), 'time': time.time(),
               'all_requested_cases_passed': len(rows)==len(args.sizes)*2*args.repeats and all(r['status']=='ok' for r in rows)})


if __name__ == '__main__':
    main()
