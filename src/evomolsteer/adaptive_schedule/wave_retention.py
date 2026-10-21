"""Transactional quality barrier and compact one-incumbent-per-chain snapshots."""
import shutil
import time
from pathlib import Path

from .io import read, write, sha, value_sha, better
from .grid import CHECKPOINT_STEPS, HEAD_STEPS


def quality_task(task):
    from .vendor.profiling import profile_task
    started = time.time()
    try:
        result = profile_task(task)
    except Exception as exc:
        write(Path(task['out'])/'execution_error.json', {'type': type(exc).__name__, 'message': str(exc),
                                                       'node_id': task['node_id'], 'time': time.time()})
        raise
    write(Path(task['out'])/'timing.json', {'seconds': time.time()-started})
    return result


def evaluate_quality(folder, template, pool):
    from rdkit import Chem
    from .quality import metrics_folder
    records = read(folder/'final_records.json')
    if (folder/'QUALITY_COMPLETE.json').exists():
        if read(folder/'QUALITY_COMPLETE.json')['records_sha256'] != sha(folder/'quality_records.json'):
            raise ValueError('Quality records changed')
        return read(folder/'quality_records.json')
    sdf = folder/'molecules_all_built.sdf'
    molecules = ({int(m.GetProp('slot')): m for m in Chem.SDMolSupplier(
        str(sdf), removeHs=True) if m is not None and m.HasProp('slot')}
        if sdf.exists() and sdf.stat().st_size else {})
    tasks = []
    for i, r in enumerate(records):
        out = folder/'profiles'/f'slot_{i:03d}'
        if not r.get('build_success'):
            write(out/'failure.json', {'reason': 'builder_failed', 'node_id': r['node_id']})
        elif i not in molecules:
            raise ValueError('Built molecule absent; preserve inputs for repair')
        elif not (out/'complete.json').exists():
            tasks.append({'molblock': Chem.MolToMolBlock(molecules[i]), 'protein': template['protein'],
                          'node_id': r['node_id'], 'out': str(out)})
    list(pool.map(quality_task, tasks, chunksize=1))
    enriched = [{**r, **metrics_folder(folder/'profiles'/f'slot_{i:03d}', template)} for i, r in enumerate(records)]
    write(folder/'quality_records.json', enriched)
    write(folder/'QUALITY_COMPLETE.json', {'attempts': len(records), 'records_sha256': sha(folder/'quality_records.json'),
                                          'time': time.time()})
    return enriched


def verify_snapshot(folder):
    complete = read(folder/'COMPLETE.json')
    for name, digest in complete['files_sha256'].items():
        path = (folder/name).resolve()
        if not path.is_relative_to(folder.resolve()) or sha(path) != digest:
            raise ValueError('Incumbent snapshot checksum failed: '+name)
    return complete


def prune_payload(folder):
    """Only named structural payload formats, after caller has published a snapshot."""
    removed = read(folder/'cleanup.json') if (folder/'cleanup.json').exists() else []
    known = {r['path'] for r in removed}
    for p in sorted(folder.rglob('*')):
        if not p.is_file() or not (p.name.endswith('.pt.gz') or p.suffix.lower() in {'.h5', '.sdf', '.pdb'}):
            continue
        if not p.resolve().is_relative_to(folder.resolve()):
            raise ValueError('Cleanup escaped the branch')
        name = p.relative_to(folder).as_posix()
        if name not in known:
            removed.append({'path': name, 'bytes': p.stat().st_size, 'sha256': sha(p)})
            write(folder/'cleanup.json', removed)
            known.add(name)
        p.unlink()
    write(folder/'CLEANUP_COMPLETE.json', {'removed_files': len(removed), 'bytes': sum(r['bytes'] for r in removed)})


def materialize_snapshot(parent, wave, folder, best, winners):
    import h5py
    import numpy as np
    from rdkit import Chem
    from .flowr import load_pt, save_pt
    from .wave_native import mix_checkpoint, join, selected_document
    folder.mkdir(parents=True, exist_ok=True)
    identity = value_sha(best)
    if (folder/'COMPLETE.json').exists():
        complete = verify_snapshot(folder)
        if complete['identity_sha256'] != identity:
            raise ValueError('Published snapshot belongs to different incumbents')
        return
    start = read(wave/'manifest.json')['start_step']
    for step in CHECKPOINT_STEPS:
        p = load_pt(parent/f'checkpoint_{step:03d}.pt.gz')
        w = load_pt(wave/f'checkpoint_{step:03d}.pt.gz') if step >= start else p
        effective = winners if step >= start else [None]*100
        save_pt(folder/f'checkpoint_{step:03d}.pt.gz', mix_checkpoint(p, w, effective))
    for step in HEAD_STEPS:
        p = load_pt(parent/f'head_{step:03d}.pt.gz')
        w = load_pt(wave/f'head_{step:03d}.pt.gz') if step >= start else p
        sources = [w if step >= start and x is not None else p for x in winners]
        sizes = [400 if step >= start and x is not None else 100 for x in winners]
        indices = [x if step >= start and x is not None else i for i, x in enumerate(winners)]
        # External RNG is incidental; logical stream states live in checkpoints.
        keys = ['prediction', 'raw_head', 'assignment', 'gradient_world']
        save_pt(folder/f'head_{step:03d}.pt.gz', {k: join([s[k] for s in sources], sizes, indices) for k in keys})
    p, w = load_pt(parent/'terminal.pt.gz'), load_pt(wave/'terminal.pt.gz')
    sources = [w if x is not None else p for x in winners]
    sizes = [400 if x is not None else 100 for x in winners]
    indices = [x if x is not None else i for i, x in enumerate(winners)]
    save_pt(folder/'terminal.pt.gz', join(sources, sizes, indices))
    # Every retained trace contains a complete prefix and suffix, so deleted
    # ancestors are never needed for recovery or analysis.
    with h5py.File(parent/'trace.h5', 'r') as a, h5py.File(wave/'trace.h5', 'r') as b, h5py.File(folder/'trace.h5', 'w') as out:
        for key in a:
            ds = out.create_dataset(key, shape=a[key].shape, dtype=a[key].dtype,
                                    chunks=a[key].chunks, compression='gzip', compression_opts=1)
            for step in range(100):
                values = a[key][step]
                if step >= start:
                    other = b[key][step]
                    for i, winner in enumerate(winners):
                        if winner is not None:
                            values[i] = other[winner]
                ds[step] = values
    nodes_a, nodes_b = read(parent/'nodes.json'), read(wave/'nodes.json')
    steps_a, steps_b = read(parent/'steps.json'), {r['step']: r for r in read(wave/'steps.json')}
    def merged_doc(a, b, step):
        rows = [selected_document(b, x, 400) if step >= start and x is not None else selected_document(a, i, 100)
                for i, x in enumerate(winners)]
        keys = set.intersection(*(set(r) for r in rows)) - {'step'}
        return {'step': step, **{k: [r[k] for r in rows] for k in keys}}
    write(folder/'nodes.json', {str(s): merged_doc(nodes_a[str(s)], nodes_b.get(str(s)), s) for s in CHECKPOINT_STEPS})
    write(folder/'steps.json', [merged_doc(steps_a[s], steps_b.get(s), s) for s in range(100)])
    mols = []
    for source in [parent, wave]:
        sdf = source/'molecules_all_built.sdf'
        mols.append({int(m.GetProp('slot')): m for m in Chem.SDMolSupplier(str(sdf), removeHs=False)
                     if m is not None and m.HasProp('slot')} if sdf.exists() and sdf.stat().st_size else {})
    writer = Chem.SDWriter(str(folder/'molecules_all_built.sdf'))
    try:
        for i, winner in enumerate(winners):
            source, row = (wave, winner) if winner is not None else (parent, i)
            mol = mols[int(winner is not None)].get(row)
            if best[i]['record'].get('build_success') and mol is None:
                raise ValueError('Selected built structure is missing')
            if mol is not None:
                mol.SetIntProp('slot', i); writer.write(mol)
            shutil.copytree(source/'profiles'/f'slot_{row:03d}', folder/'profiles'/f'slot_{i:03d}', dirs_exist_ok=True)
    finally:
        writer.close()
    write(folder/'best.json', best)
    write(folder/'final_records.json', [x['record'] for x in best])
    files = {p.relative_to(folder).as_posix(): sha(p) for p in folder.rglob('*') if p.is_file() and p.name != 'COMPLETE.json'}
    write(folder/'COMPLETE.json', {'identity_sha256': identity, 'files_sha256': files, 'attempts': 100})
    verify_snapshot(folder)


def commit_best(base, parent, wave, trials, round_no, shared_r0):
    """Recoverable publication, followed by idempotent structure-only cleanup."""
    receipt_path = base/f'commit_{round_no:02d}.json'
    folder = base/'incumbents'/f'r{round_no:02d}'
    if receipt_path.exists():
        receipt = read(receipt_path)
        verify_snapshot(folder)
    else:
        old = read(parent/'best.json')
        best, winners = list(old), [None]*100
        for g in range(4):
            for i, trial in enumerate(trials[g]):
                if better(trial['record'], best[i]['record']):
                    best[i], winners[i] = trial, g*100+i
        materialize_snapshot(parent, wave, folder, best, winners)
        receipt = {'round': round_no, 'snapshot': str(folder), 'snapshot_sha256': sha(folder/'COMPLETE.json'),
                   'quality_complete': True, 'retention_complete': True, 'improved': sum(x is not None for x in winners)}
        write(receipt_path, receipt)
        write(base/'incumbent.json', receipt)
    prune_payload(wave)
    if parent.resolve() != shared_r0.resolve():
        prune_payload(parent)
    write(base/f'RETENTION_COMPLETE_{round_no:02d}.json', receipt)
    return receipt
