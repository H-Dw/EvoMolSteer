"""Single retained R0 population, computed in the same four-group execution context."""
from pathlib import Path
from .io import read, write, sha
from .grid import CHECKPOINT_STEPS, HEAD_STEPS


def compact_population(source, destination, population=25):
    import h5py
    import torch
    from rdkit import Chem
    from .flowr import load_pt, save_pt
    from .wave_native import take
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    binding = read(source/'manifest.json')
    if binding['rows'] != population*4 or len({tuple(t) for t in binding['schedules']}) != 1:
        raise ValueError('R0 compaction requires four identical CRN groups')
    original = read(source/'final_records.json')
    if len(original) != population*4:
        raise ValueError('Incomplete physical R0')
    selection = list(range(population))
    terminal = load_pt(source/'terminal.pt.gz')
    duplicate_errors = []
    for group in range(1,4):
        offset = group*population
        duplicate_errors.append(float((terminal['coords'][:population]-terminal['coords'][offset:offset+population]).abs().max()))
        if not torch.allclose(terminal['coords'][:population],terminal['coords'][offset:offset+population],atol=1e-5,rtol=1e-5):
            raise ValueError('R0 repeated common-random groups have different coordinates')
        for key in ['atomics','charges','bonds','mask']:
            if not torch.equal(terminal[key][:population],terminal[key][offset:offset+population]):
                raise ValueError('R0 repeated common-random groups have different '+key)
        for a,b in zip(original[:population],original[offset:offset+population]):
            x,y=a.get('pic50_on_rescore'),b.get('pic50_on_rescore')
            if (x is None)!=(y is None) or (x is not None and abs(x-y)>1e-4):
                raise ValueError('R0 repeated common-random scores differ')
    write(destination/'PHYSICAL_REPEAT_CHECK.json',{'physical_rows':population*4,
        'unique_rows':population,'coordinate_max_abs_by_repeat':duplicate_errors,
        'categorical_equal':True,'score_abs_tolerance':1e-4,
        'interpretation':'Numerical replay acceptance only, not a condition reconstruction success threshold.'})
    for step in CHECKPOINT_STEPS:
        state = load_pt(source/f'checkpoint_{step:03d}.pt.gz')
        for key in ['current','cond','times','previous_assignment']:
            state[key] = take(state[key], selection, population*4)
        state['rows'] = population
        save_pt(destination/f'checkpoint_{step:03d}.pt.gz', state)
    for step in HEAD_STEPS:
        head = load_pt(source/f'head_{step:03d}.pt.gz')
        save_pt(destination/f'head_{step:03d}.pt.gz', take(head, selection, population*4))
    save_pt(destination/'terminal.pt.gz', take(load_pt(source/'terminal.pt.gz'), selection, population*4))
    with h5py.File(source/'trace.h5','r') as src, h5py.File(destination/'trace.h5','w') as dst:
        for key in src:
            shape = (src[key].shape[0],population,*src[key].shape[2:])
            ds = dst.create_dataset(key,shape=shape,dtype=src[key].dtype,chunks=True,compression='gzip',compression_opts=1)
            for step in range(shape[0]):
                ds[step] = src[key][step,:population]
    def document(row):
        return {k:v[:population] if isinstance(v,list) and len(v)==population*4 else v for k,v in row.items()}
    write(destination/'nodes.json',{k:document(v) for k,v in read(source/'nodes.json').items()})
    write(destination/'steps.json',[document(v) for v in read(source/'steps.json')])
    sdf = source/'molecules_all_built.sdf'
    writer = Chem.SDWriter(str(destination/'molecules_all_built.sdf'))
    try:
        if sdf.exists() and sdf.stat().st_size:
            for mol in Chem.SDMolSupplier(str(sdf),removeHs=False):
                if mol is not None and mol.HasProp('slot') and int(mol.GetProp('slot')) < population:
                    writer.write(mol)
    finally:
        writer.close()
    write(destination/'final_records.json', original[:population])
    write(destination/'manifest.json',dict(binding,rows=population,groups=1,schedules=binding['schedules'][:population],
        physical_rows=population*4,unique_start_count=population,physical_source=str(source),
        physical_source_complete_sha256=sha(source/'COMPLETE.json')))
    write(destination/'SAMPLING_COMPLETE.json',dict(read(source/'SAMPLING_COMPLETE.json'),
        rows=population,physical_rows=population*4,unique_start_count=population))
    write(destination/'EVALUATION_COMPLETE.json',{'attempts':population,'physical_repeat_rows_not_independent':population*3})
    write(destination/'COMPLETE.json',{'records_sha256':sha(destination/'final_records.json'),'attempts':population})


def prepare_r0(model, root, config, template, folder):
    from .wave_native import execute_wave
    from .wave_retention import verify_snapshot
    if (folder/'R0_VERIFIED.json').exists():
        if read(folder/'R0_VERIFIED.json')['complete_sha256'] != sha(folder/'COMPLETE.json'):
            raise ValueError('Frozen R0 completion changed')
        verify_snapshot(folder)
        return
    if not (folder/'COMPLETE.json').exists():
        source = root/'preparation'/template['teacher_id']/'r0_physical'
        execute_wave(model,config,template,[config['initial_ticks']]*(config['population_size']*4),source,root/'contexts'/'population_context.pt.gz')
        compact_population(source,folder,config['population_size'])


def finish_r0(folder):
    from .wave_retention import verify_snapshot, prune_payload
    if (folder/'R0_VERIFIED.json').exists():
        verify_snapshot(folder)
        return
    population = read(folder/'manifest.json')['population_size']
    if not (folder/'QUALITY_COMPLETE.json').exists() or len(read(folder/'best.json')) != population:
        raise ValueError('Complete R0 quality and best index required before physical-repeat cleanup')
    files = {p.relative_to(folder).as_posix():sha(p) for p in folder.rglob('*')
             if p.is_file() and p.name not in {'COMPLETE.json','R0_VERIFIED.json'}}
    write(folder/'COMPLETE.json',{'records_sha256':sha(folder/'final_records.json'),
        'attempts':population,'files_sha256':files,'physical_rows':population*4})
    verify_snapshot(folder)
    write(folder/'R0_VERIFIED.json',{'complete_sha256':sha(folder/'COMPLETE.json')})
    source=Path(read(folder/'manifest.json')['physical_source'])
    prune_payload(source)
