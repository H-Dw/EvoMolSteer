"""Idempotent conversion of closed trajectories, with verified source deletion."""
from pathlib import Path
from contextlib import nullcontext
import shutil
import numpy as np
from ..io import read_json,digest
from .trajectory import pack_trajectory,TrajectoryPackage
from .transactions import atomic_json,exclusive_lock,checked_path,unlink_verified


def convert_trajectory(source,destination,*,input_dataset,delete_source=False):
    """Convert one closed NPZ. A verified receipt precedes any source deletion.

    Repeating the same call recovers publication/deletion interrupted by a
    process exception. An existing unrelated package is never overwritten.
    Callers must close the producer before invoking this function.
    """
    source=checked_path(source,input_dataset)
    destination=Path(destination).resolve()
    if source.suffix!='.npz' or destination.suffix!='.h5' or source==destination:
        raise ValueError('Expected distinct .npz input and .h5 output paths')
    destination.parent.mkdir(parents=True,exist_ok=True)
    receipt=destination.with_suffix('.conversion.json')
    with exclusive_lock(destination.with_suffix('.conversion.lock')):
        record=read_json(receipt) if receipt.exists() else None
        if record is not None and (record.get('source')!=str(source) or record.get('destination')!=str(destination)):
            raise ValueError('Receipt belongs to a different conversion')
        if record is None:
            if destination.exists():raise FileExistsError(destination)
            if destination.with_suffix(destination.suffix+'.partial').exists():
                raise FileExistsError('Unowned partial output exists')
            if not source.is_file():raise FileNotFoundError(source)
            record={'format':'evomolsteer.conversion.v1','state':'preparing',
                    'source':str(source),'destination':str(destination),'source_sha256':digest(source),
                    'source_bytes':source.stat().st_size,'delete_requested':bool(delete_source)}
            atomic_json(receipt,record)
        if source.exists() and digest(source)!=record['source_sha256']:
            raise ValueError('Source no longer matches this transaction')
        if not destination.exists():
            if not source.exists():raise FileNotFoundError('Both source and normalized trajectory are absent')
            partial=destination.with_suffix(destination.suffix+'.partial')
            # Only this receipt's uncommitted workspace can be retried.
            if partial.exists():
                checked_path(partial,destination.parent).unlink()
            report=pack_trajectory(source,destination,codec='gzip_shuffle')
        else:
            report=record.get('report')
            with TrajectoryPackage(destination) as package:
                if package.file.attrs.get('source_sha256')!=record['source_sha256']:
                    raise ValueError('Published package has a different source')
                with np.load(source,allow_pickle=False) if source.exists() else nullcontext() as original:
                    count=package.verify(original)
            if report is None:
                if not source.exists():raise ValueError('Source removed without a verified publication receipt')
                report={'verified_arrays':count,'package_sha256':digest(destination),'package_bytes':destination.stat().st_size}
        if 'report' in record and digest(destination)!=record['report']['package_sha256']:
            raise ValueError('Published package changed; source retained')
        # This receipt makes the package recoverable after a crash before unlink.
        record.update(state='verified',report=report,delete_requested=bool(delete_source))
        atomic_json(receipt,record)
        if delete_source and source.exists():
            unlink_verified(source,record['source_sha256'],input_dataset)
        record.update(state='complete',source_deleted=not source.exists())
        atomic_json(receipt,record)
        return record


def convert_dataset(input_dataset,output_dataset,*,delete_source=False):
    """Mirror a completed generation dataset, replacing only trajectory NPZs.

    Shared inputs/configuration and restart artifacts are copied when roots
    differ. Only verified trajectory NPZ files in the explicit input are retired.
    Complete batch and campaign markers are required; for a live producer use
    StepTrajectoryWriter or the single-file commit API instead.
    """
    source,target=Path(input_dataset).resolve(),Path(output_dataset).resolve()
    if not source.is_dir():raise NotADirectoryError(source)
    if target!=source and (target.is_relative_to(source) or source.is_relative_to(target)):
        raise ValueError('Use equal or disjoint dataset directories')
    trajectories=sorted(source.glob('results/*/*/batch_*/trajectory.npz'))
    if not trajectories and not (target/'conversion_manifest.json').exists():
        raise ValueError('No completed trajectory dataset found')
    if any(not (p.parent/'COMPLETE.json').exists() for p in trajectories):
        raise ValueError('A batch is still open; no conversion/deletion was started')
    if any(not (p.parents[2]/'COMPLETE.json').exists() for p in trajectories):
        raise ValueError('A campaign is still open; use the per-step producer interface')
    if target!=source and target.exists() and any(target.iterdir()) and not (target/'conversion_manifest.json').exists():
        raise FileExistsError('Output dataset is not empty and has no conversion manifest')
    target.mkdir(parents=True,exist_ok=True)
    manifest_path=target/'conversion_manifest.json'
    with exclusive_lock(target/'.dataset_conversion.lock'):
        manifest=read_json(manifest_path) if manifest_path.exists() else {
            'format':'evomolsteer.dataset_conversion.v1','source_root':str(source),'output_root':str(target),
            'complete':False,'trajectories':{}}
        if manifest['source_root']!=str(source) or manifest['output_root']!=str(target):
            raise ValueError('Conversion manifest has different dataset roots')
        manifest['complete']=False;atomic_json(manifest_path,manifest)
        recognized={p.resolve() for p in trajectories}|{(source/name).resolve() for name in manifest['trajectories']}
        if target!=source:
            for path in sorted(source.rglob('*')):
                if not path.is_file() or path.resolve() in recognized or path.name in {'conversion_manifest.json','trajectory_retirement.json','SHA256SUMS'} or path.suffix=='.lock':
                    continue
                checked_path(path,source)
                dest=target/path.relative_to(source)
                if dest.exists():
                    if digest(path)!=digest(dest):raise ValueError('Existing output differs: '+str(dest))
                else:
                    dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
        # Include previous receipts to finish a deletion interrupted between batches.
        relatives=sorted(set(p.relative_to(source).as_posix() for p in trajectories)|set(manifest['trajectories']))
        for relative in relatives:
            path=source/relative;dest=(target/relative).with_suffix('.h5')
            record=convert_trajectory(path,dest,input_dataset=source,delete_source=False)
            manifest['trajectories'][relative]=record
            atomic_json(manifest_path,manifest)
        if delete_source:
            if source!=target:
                atomic_json(source/'trajectory_retirement.json',{'output_dataset':str(target),'state':'retiring',
                    'replacement_manifest':str(manifest_path)})
            for relative in relatives:
                record=convert_trajectory(source/relative,(target/relative).with_suffix('.h5'),
                                          input_dataset=source,delete_source=True)
                manifest['trajectories'][relative]=record;atomic_json(manifest_path,manifest)
        manifest['complete']=True;atomic_json(manifest_path,manifest)
        if delete_source and source!=target:
            atomic_json(source/'trajectory_retirement.json',{'output_dataset':str(target),'state':'complete',
                'replacement_manifest':str(manifest_path)})
        # An inherited NPZ checksum inventory no longer describes the output.
        old_inventory=source/'SHA256SUMS'
        historical=target/'SHA256SUMS.before_conversion'
        if old_inventory.exists() and not historical.exists():shutil.copyfile(old_inventory,historical)
        checksum_lines=[]
        for path in sorted(target.rglob('*')):
            if path.is_file() and path.name!='SHA256SUMS' and path.suffix!='.lock':
                checksum_lines.append(digest(path)+'  '+path.relative_to(target).as_posix())
        (target/'SHA256SUMS').write_text('\n'.join(checksum_lines)+'\n',encoding='utf-8')
        return manifest
