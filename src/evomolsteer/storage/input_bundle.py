"""Portable analysis inputs: all trajectory arrays, shared geometry, provenance.

This is an analysis input package, not a checkpoint/restart archive. All 100-step
trajectory fields are retained losslessly; only the observed selection events
are consumed by the analysis pipeline. Original experiment archives remain kept.
"""
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import shutil

from ..io import digest,read_json,write_json
from ..scope import discover_scope
from ..trajectory_source import trajectory_paths
from .trajectory import pack_trajectory,TrajectoryPackage


def _convert(source,destination):
    return pack_trajectory(source,destination,codec='gzip_shuffle')


def build_input_bundle(root,destination,cfg,workers=4):
    root,destination = Path(root).resolve(),Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    if destination==root or destination.is_relative_to(root):
        raise ValueError('Keep the normalized package outside the original source')
    campaign = root/'results'/cfg['campaign']
    scope = discover_scope(campaign,cfg)
    paths = [p for p in trajectory_paths(campaign) if (p.parent/'COMPLETE.json').exists()
             and p.parent.parent.name in cfg['selection_arms']+[cfg['background_arm']]
             and (cfg.get('include_batches') is None or int(p.parent.name.split('_')[1]) in cfg['include_batches'])]
    if any(p.suffix!='.npz' for p in paths):
        raise ValueError('Bundle creation expects the original NPZ trajectories')
    destination.mkdir(parents=True)
    manifest = {'format':'evomolsteer.analysis_inputs.v1','complete':False,'source_root':str(root),
        'campaign':cfg['campaign'],'selection_scope':scope,'trajectory_fields':'all original fields and steps',
        'codec':'gzip_shuffle','copied_files':{},'packages':[],
        'not_included':['restart/endpoint checkpoints','RNG states','final exports','other campaigns'],
        'purpose':'Geometry and selection analysis; original full experiment archive remains authoritative'}
    manifest_path = destination/'input_bundle_manifest.json'
    write_json(manifest_path,manifest)
    copy_paths = list((root/'inputs').glob('*'))+[campaign/'config.json']
    copy_paths += [campaign/f'frame_batch_{int(p.parent.name.split("_")[1]):03d}.json' for p in paths]
    copy_paths += [p.parent/'COMPLETE.json' for p in paths]
    for source in sorted(set(copy_paths)):
        if not source.is_file():
            continue
        relative = source.relative_to(root)
        target = destination/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
        expected = digest(source)
        if digest(target)!=expected:
            raise ValueError('Copied input checksum mismatch')
        manifest['copied_files'][relative.as_posix()] = {'sha256':expected,'bytes':target.stat().st_size}
    with ProcessPoolExecutor(max_workers=min(max(int(workers),1),len(paths))) as pool:
        jobs = {pool.submit(_convert,str(p),str((destination/p.relative_to(root)).with_suffix('.h5'))):p for p in paths}
        for n,job in enumerate(as_completed(jobs),1):
            r = job.result()
            r['package'] = Path(r['package']).relative_to(destination).as_posix()
            r['source'] = jobs[job].relative_to(root).as_posix()
            manifest['packages'].append(r)
            print(f'Packed and bit-verified {n}/{len(paths)}: {r["package"]}',flush=True)
    manifest['packages'].sort(key=lambda r:r['source'])
    manifest['source_trajectory_bytes'] = sum(r['source_bytes'] for r in manifest['packages'])
    manifest['package_trajectory_bytes'] = sum(r['package_bytes'] for r in manifest['packages'])
    manifest['verified_arrays'] = sum(r['verified_arrays'] for r in manifest['packages'])
    if discover_scope(destination/'results'/cfg['campaign'],cfg)!=scope:
        raise AssertionError('Packed input changes observed selection scope')
    manifest['complete'] = True
    write_json(manifest_path,manifest)
    return manifest


def verify_input_bundle(destination):
    destination = Path(destination).resolve()
    manifest = read_json(destination/'input_bundle_manifest.json')
    if manifest.get('format')!='evomolsteer.analysis_inputs.v1' or not manifest['complete']:
        raise ValueError('Incomplete or unknown input bundle')
    count = 0
    for relative,record in manifest['copied_files'].items():
        path = (destination/relative).resolve()
        if not path.is_relative_to(destination) or digest(path)!=record['sha256']:
            raise ValueError('Input file checksum mismatch')
    for record in manifest['packages']:
        path = (destination/record['package']).resolve()
        if not path.is_relative_to(destination) or digest(path)!=record['package_sha256']:
            raise ValueError('Trajectory package checksum mismatch')
        with TrajectoryPackage(path) as p:
            count += p.verify()
    return {'verified_arrays':count,'packages':len(manifest['packages']),'complete':True}
