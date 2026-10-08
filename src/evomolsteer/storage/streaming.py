"""Commit each generated step immediately; consolidate a batch after completion."""
from pathlib import Path
import gzip
import os
import numpy as np
from ..io import read_json,digest
from .arrays import array_digest
from .trajectory import pack_arrays,TrajectoryPackage
from .transactions import atomic_json,exclusive_lock,checked_path,unlink_verified


class StepTrajectoryWriter:
    """Input: one step's numeric fields, without a leading time axis.

    Every append publishes a bit-verified HDF5 stage before returning. The caller
    can then release its raw arrays. Finalization publishes trajectory.h5 before
    deleting committed stage files. No raw NPZ or ever-growing raw trace is made.
    Durable stages survive interrupted generation; no unfinished batch is marked
    COMPLETE. A replay of the same step is allowed only if every input bit matches.
    """
    def __init__(self,batch_directory,*,expected_steps,metadata=None,codec='gzip_shuffle'):
        self.root=Path(batch_directory).resolve()
        if type(expected_steps) is not int or expected_steps<1:
            raise ValueError('expected_steps must be a positive integer')
        if codec not in ('none','gzip_shuffle'):
            raise ValueError('Unknown lossless trajectory codec')
        self.codec=codec
        self.root.mkdir(parents=True,exist_ok=True)
        self.stages=self.root/'trajectory.stages'
        self.manifest_path=self.root/'trajectory.manifest.json'
        self.final_path=self.root/'trajectory.h5'
        self.lock=self.root/'.trajectory_writer.lock'
        self.expected_steps=int(expected_steps)
        with exclusive_lock(self.lock):
            if self.manifest_path.exists():
                manifest=read_json(self.manifest_path)
                if manifest.get('format')!='evomolsteer.streaming.v1' or manifest['expected_steps']!=expected_steps or manifest['metadata']!=(metadata or {}) or manifest.get('codec','gzip_shuffle')!=codec:
                    raise ValueError('Existing stream has a different schema/configuration')
            else:
                if self.final_path.exists() or self.stages.exists() or (self.root/'trajectory.npz').exists():
                    raise FileExistsError('Unowned trajectory already exists in '+str(self.root))
                self.stages.mkdir()
                atomic_json(self.manifest_path,{'format':'evomolsteer.streaming.v1','state':'recording',
                    'expected_steps':self.expected_steps,'metadata':metadata or {},'schema':None,'steps':[],
                    'raw_npz_created':False,'codec':codec,
                    'stage_container':'gzip_wrapped_hdf5' if codec!='none' else 'unfiltered_hdf5'})

    def append(self,step,fields):
        if type(step) is not int or not 0<=step<self.expected_steps:
            raise ValueError('Invalid step index')
        arrays={k:np.asarray(v)[None,...] for k,v in fields.items()}
        if not arrays:raise ValueError('Empty generation step')
        schema={k:{'dtype':v.dtype.str,'shape':list(v.shape[1:])} for k,v in arrays.items()}
        hashes={k:array_digest(v) for k,v in arrays.items()}
        with exclusive_lock(self.lock):
            manifest=read_json(self.manifest_path)
            if manifest['schema'] is not None and manifest['schema']!=schema:
                raise ValueError('Step field names, dtype or shape changed')
            if step<len(manifest['steps']):
                record=manifest['steps'][step]
                if hashes!=record['array_sha256']:raise ValueError('Replay differs from the committed step')
                if manifest['state']=='complete':
                    if digest(self.final_path)!=manifest['final']['package_sha256']:raise ValueError('Final package changed')
                elif digest(self.root/record['file'])!=record['sha256']:
                    raise ValueError('Committed step file changed')
                return record
            if manifest['state']!='recording' or step!=len(manifest['steps']):
                raise ValueError('Steps must be committed consecutively before finalization')
            if self.codec=='none':
                dest=self.stages/f'step_{step:06d}.h5'
                partial=dest.with_suffix('.h5.partial')
                if dest.exists():
                    with TrajectoryPackage(dest) as p:p.verify(arrays)
                else:
                    if partial.exists():checked_path(partial,self.stages).unlink()
                    pack_arrays(arrays,dest,codec='none')
                record={'step':step,'file':dest.relative_to(self.root).as_posix(),'sha256':digest(dest),
                        'bytes':dest.stat().st_size,'array_sha256':hashes}
                manifest['schema']=schema;manifest['steps'].append(record)
                atomic_json(self.manifest_path,manifest)
                return record
            dest=self.stages/f'step_{step:06d}.h5.gz'
            raw=dest.with_suffix('')
            partial=raw.with_suffix(raw.suffix+'.partial')
            wrapped_partial=dest.with_suffix(dest.suffix+'.partial')
            # A crash may leave this stream's next stage published before journal commit.
            if dest.exists():
                with TrajectoryPackage(dest) as p:p.verify(arrays)
                if raw.exists():
                    with TrajectoryPackage(raw) as p:p.verify(arrays)
                    raw_sha=digest(raw)
            else:
                if partial.exists():checked_path(partial,self.stages).unlink()
                if raw.exists():
                    with TrajectoryPackage(raw) as p:p.verify(arrays)
                    raw_sha=digest(raw)
                else:raw_sha=pack_arrays(arrays,raw,codec='gzip_shuffle')['package_sha256']
                if wrapped_partial.exists():checked_path(wrapped_partial,self.stages).unlink()
                # A single-step HDF5 has substantial repetitive metadata. This
                # fast, lossless envelope compresses that metadata as well.
                verified_container=raw.read_bytes()
                if digest(raw)!=raw_sha:raise ValueError('Temporary normalized stage changed')
                with wrapped_partial.open('xb') as handle:
                    handle.write(gzip.compress(verified_container,compresslevel=1,mtime=0))
                    handle.flush();os.fsync(handle.fileno())
                # Array equivalence was established above. A byte-exact envelope
                # roundtrip is stronger than parsing and rechecking those arrays.
                if gzip.decompress(wrapped_partial.read_bytes())!=verified_container:
                    raise ValueError('Lossless stage envelope roundtrip failed')
                wrapped_partial.replace(dest)
            if raw.exists():
                unlink_verified(raw,raw_sha,self.stages)
            record={'step':step,'file':dest.relative_to(self.root).as_posix(),'sha256':digest(dest),
                    'bytes':dest.stat().st_size,'array_sha256':hashes}
            manifest['schema']=schema;manifest['steps'].append(record)
            atomic_json(self.manifest_path,manifest)
            return record

    def finalize(self):
        with exclusive_lock(self.lock):
            manifest=read_json(self.manifest_path)
            if len(manifest['steps'])!=self.expected_steps:
                raise ValueError('Cannot finalize an incomplete trajectory')
            if manifest['state']=='complete':
                with TrajectoryPackage(self.final_path) as p:p.verify()
                if digest(self.final_path)!=manifest['final']['package_sha256']:raise ValueError('Final package changed')
                return manifest['final']
            # If publication was already recorded, resume only the verified cleanup.
            if manifest['state']=='consolidated':
                with TrajectoryPackage(self.final_path) as p:p.verify()
                if digest(self.final_path)!=manifest['final']['package_sha256']:raise ValueError('Final package changed')
            else:
                parts={}
                for record in manifest['steps']:
                    path=checked_path(self.root/record['file'],self.stages)
                    if digest(path)!=record['sha256']:raise ValueError('Stage checksum changed')
                    with TrajectoryPackage(path) as p:
                        p.verify()
                        for key in p.keys:parts.setdefault(key,[]).append(p.read(key))
                arrays={key:np.concatenate(values,axis=0) for key,values in parts.items()}
                del parts
                if self.final_path.exists():
                    with TrajectoryPackage(self.final_path) as p:count=p.verify(arrays)
                    report={'package':str(self.final_path),'package_sha256':digest(self.final_path),
                            'package_bytes':self.final_path.stat().st_size,'verified_arrays':count,
                            'recovered_publication':True}
                else:
                    partial=self.final_path.with_suffix('.h5.partial')
                    if partial.exists():checked_path(partial,self.root).unlink()
                    report=pack_arrays(arrays,self.final_path,codec=self.codec)
                manifest.update(state='consolidated',final=report)
                atomic_json(self.manifest_path,manifest)
            # Never delete a stage before durable publication + successful full comparison.
            for record in manifest['steps']:
                path=self.root/record['file']
                if path.exists():unlink_verified(path,record['sha256'],self.stages)
            if self.stages.exists():self.stages.rmdir()  # unexpected files prevent completion
            manifest.update(state='complete',stage_files_retired=True)
            atomic_json(self.manifest_path,manifest)
            return manifest['final']
