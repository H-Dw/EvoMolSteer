"""Compact transport for terminal evaluation, not a trajectory-mining dataset.

Final molecules/scores, inputs and logs are copied byte-for-byte. Initial states
and all selection records are exact NPZ arrays; full trajectory hashes preserve
provenance. Original trajectories remain untouched until report-gated retirement.
"""
import shutil,hashlib
from pathlib import Path
import numpy as np
from ..io import read_json,write_json,digest
from .trajectory import TrajectoryPackage

INITIAL_FIELDS=('current_coords','current_atomics','current_bonds')

def execution_arrays(source):
    source=Path(source)
    if source.suffix=='.h5':
        with TrajectoryPackage(source) as p:
            arrays={key:p.read(key,0)[None] for key in INITIAL_FIELDS}
            arrays.update({key:p.read(key) for key in ['selected_indices','resampled']})
    else:
        with np.load(source,allow_pickle=False) as p:
            arrays={key:p[key][:1].copy() for key in INITIAL_FIELDS}
            arrays.update({key:p[key].copy() for key in ['selected_indices','resampled']})
    return arrays

def state_signature(arrays):
    h=hashlib.sha256()
    for key in INITIAL_FIELDS:h.update(arrays[key][0].tobytes())
    return h.hexdigest()

def export_evaluation_view(dataset,campaign,destination):
    root=Path(dataset).resolve();run=(root/'results'/campaign).resolve();out=Path(destination).resolve()
    if not run.is_relative_to(root/'results') or run.name!=campaign:raise ValueError('One campaign required')
    if out.exists() or out.is_relative_to(root):raise ValueError('Fresh destination outside original dataset required')
    if read_json(run/'COMPLETE.json')['status']!='complete':raise ValueError('Complete generation required')
    batches=[];copied={}
    for directory in [root/'inputs',run]:
        for p in sorted(directory.rglob('*')):
            if p.is_symlink():raise ValueError('Symlinks not supported')
            if not p.is_file():continue
            relative=p.relative_to(root);target=out/relative;target.parent.mkdir(parents=True,exist_ok=True)
            if p.name in ['trajectory.h5','trajectory.npz']:
                arrays=execution_arrays(p);snapshot=target.with_name('execution_state.npz')
                np.savez_compressed(snapshot,**arrays)
                with np.load(snapshot,allow_pickle=False) as z:
                    for key,value in arrays.items():
                        if value.dtype!=z[key].dtype or value.tobytes()!=z[key].tobytes():raise ValueError('Execution snapshot changed bytes')
                batches.append({'batch_path':p.parent.relative_to(run).as_posix(),
                    'source_trajectory_sha256':digest(p),'source_bytes':p.stat().st_size,
                    'snapshot_sha256':digest(snapshot),'initial_state_signature':state_signature(arrays),
                    'steps':len(arrays['selected_indices'])})
            else:
                shutil.copyfile(p,target)
                if digest(p)!=digest(target):raise ValueError('Evaluation payload copy mismatch')
                copied[relative.as_posix()]=digest(p)
    m={'schema_version':'terminal-execution-view-1.0','complete':True,'campaign':campaign,'batches':batches,
       'converter_source_sha256':digest(Path(__file__)),
       'copied_file_sha256':copied,'usage':'Terminal chemistry/energy evaluation and execution audit only; not intermediate coordinate mining',
       'retained_arrays':'Exact initial coordinates/atomics/bonds and all selected_indices/resampled records',
       'omitted':'Intermediate structure arrays, which are disposable in these inference tests; original Steer learning trajectories are unaffected',
       'source_deleted':False}
    write_json(out/'results'/campaign/'EVALUATION_VIEW.json',m)
    return m
