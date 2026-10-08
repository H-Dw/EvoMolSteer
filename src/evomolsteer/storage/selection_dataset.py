"""Readable selection-window datasets with all-step ancestry and exact terminals.

The window is defined by scoring time. Candidate arrays are captured before
selection, including zero-offspring siblings. No archive envelope or restart
pickle is produced. Retained numeric arrays preserve their original dtype/bits.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..io import digest, read_json, clean
from .arrays import array_digest
from .streaming import StepTrajectoryWriter
from .trajectory import pack_arrays, TrajectoryPackage
from .transactions import atomic_json, checked_path


FORMAT = 'evomolsteer.selection_learning.v1'
LINEAGE_FIELDS = ('selected_indices', 'offspring_count', 'root_slot', 'parent_slot',
                  'resampled', 'score_time', 'state_time', 'step_size')
EVENT_FIELDS = ('pic50_on', 'pic50_off', 'weight_on', 'weight_off',
                'selection_probability', 'root_slot', 'parent_slot', 'offspring_count')
STRUCTURE_FIELDS = tuple(rep+'_'+kind for rep in ('current','proposal','predicted')
                         for kind in ('coords','atomics','bonds','charges'))+('mask',)


def selection_steps(score_times, start, end):
    times = np.asarray(score_times)
    if times.ndim != 1 or len(times) < 1 or times.dtype.kind != 'f' or not np.isfinite(times).all():
        raise ValueError('A finite one-dimensional floating scoring grid is required')
    if np.any(np.diff(times) <= 0) or not np.isfinite([start,end]).all() or not 0 <= start <= end <= 1:
        raise ValueError('Invalid scoring grid or selection window')
    # FLOWR compares a time tensor with bounds converted to its floating dtype.
    bounds = np.asarray([start,end], dtype=times.dtype)
    steps = np.flatnonzero((times >= bounds[0]) & (times <= bounds[1]))
    if not len(steps):
        raise ValueError('The requested window contains no scoring event')
    return steps


def _package_record(root, path):
    return {'path':Path(path).relative_to(root).as_posix(), 'sha256':digest(path),
            'bytes':Path(path).stat().st_size}


def _publish_arrays(arrays, path, codec):
    """Idempotent publication; an existing different package is never replaced."""
    if path.exists():
        with TrajectoryPackage(path) as package:
            package.verify(arrays)
        return _package_record(path.parent,path)
    pack_arrays(arrays,path,codec=codec)
    return _package_record(path.parent,path)


def terminal_descendants(indices, steps):
    """Candidate -> terminal multiplicity. Extinct siblings retain an explicit 0."""
    indices = np.asarray(indices)
    b = indices.shape[1]
    mass = np.ones(b,dtype=np.int64)
    result = {}
    wanted = set(map(int,steps))
    for i in range(len(indices)-1,-1,-1):
        next_mass = np.zeros(b,dtype=np.int64)
        np.add.at(next_mass,indices[i],mass)
        mass = next_mass
        if i in wanted:
            result[i] = mass.copy()
    return np.stack([result[int(i)] for i in steps])


class SelectionLearningWriter:
    """Commit a full inference's lineage but geometry only inside the window.

    ``score_times`` must be the exact grid used by the native integration loop.
    ``append`` takes one complete pre-selection candidate population and its
    decision. ``finish`` requires all integration steps and the native t=1 output.
    """
    def __init__(self,batch_directory,*,score_times,window,metadata=None,codec='none'):
        self.root=Path(batch_directory).resolve()
        self.times=np.asarray(score_times)
        self.steps=selection_steps(self.times,*window)
        self.codec=codec
        self.metadata=metadata or {}
        self.manifest_path=self.root/'learning_manifest.json'
        self.schema={'format':FORMAT,'score_times':self.times.tolist(),'selection_steps':self.steps.tolist(),
                     'window':list(window),'codec':codec,'metadata':self.metadata}
        self.root.mkdir(parents=True,exist_ok=True)
        if self.manifest_path.exists():
            manifest=read_json(self.manifest_path)
            if any(manifest[k]!=v for k,v in self.schema.items()):
                raise ValueError('Existing learning dataset has different inputs/schedule')
        else:
            if (self.root/'trajectory.h5').exists():
                raise FileExistsError('Unowned trajectory exists')
            atomic_json(self.manifest_path,{**self.schema,'state':'recording'})
        self.window_writer=StepTrajectoryWriter(self.root,expected_steps=len(self.steps),codec=codec,
            metadata={**self.metadata,'geometry_scope':'score_time_window','source_steps':self.steps.tolist()})
        self.lineage_writer=StepTrajectoryWriter(self.root/'lineage',expected_steps=len(self.times),codec=codec,
            metadata={**self.metadata,'geometry_scope':'no_geometry','source_steps':list(range(len(self.times)))})

    def append(self,step,fields):
        if type(step) is not int or not 0<=step<len(self.times):
            raise ValueError('Invalid original integration step')
        missing=set(LINEAGE_FIELDS+EVENT_FIELDS+STRUCTURE_FIELDS)-set(fields)
        if missing:
            raise ValueError('Incomplete candidate observation: '+str(sorted(missing)))
        fields={k:np.asarray(v) for k,v in fields.items()}
        observed=float(fields['score_time'][0])
        if not np.isclose(observed,float(self.times[step]),rtol=0,atol=1e-7):
            raise ValueError('Observed score time disagrees with the native grid')
        ids=fields['selected_indices']
        if ids.ndim!=1 or ids.dtype.kind not in 'iu' or np.any(ids<0) or np.any(ids>=len(ids)):
            raise ValueError('Invalid parent selection indices')
        counts=np.bincount(ids.astype(np.intp),minlength=len(ids))
        if not np.array_equal(counts,fields['offspring_count']):
            raise ValueError('Offspring counts disagree with the selection')
        if any(fields[k].shape[0]!=len(ids) for k in EVENT_FIELDS+STRUCTURE_FIELDS):
            raise ValueError('Candidate fields do not share the same population axis')
        in_window=step in self.steps
        if self.metadata.get('arm') in ('single','joint') and bool(fields['resampled'])!=in_window:
            raise ValueError('Steer arm resampled outside its configured window or missed an event')
        if self.metadata.get('arm')=='unguided' and bool(fields['resampled']):
            raise ValueError('Unguided background unexpectedly resampled')
        if not bool(fields['resampled']) and not np.array_equal(ids,np.arange(len(ids))):
            raise ValueError('A non-resampled step must preserve slot order')
        if not np.isfinite(fields['pic50_on']).all() or not np.isfinite(fields['pic50_off']).all():
            raise ValueError('Nonfinite affinity score')
        if not np.isclose(fields['selection_probability'].sum(),1,atol=1e-6):
            raise ValueError('Incomplete selection probability mass')
        # Source array precision and bits are unchanged, including rejected geometry.
        lineage={k:fields[k] for k in LINEAGE_FIELDS}
        lineage['source_step']=np.asarray(step,dtype=np.int32)
        self.lineage_writer.append(step,lineage)
        if in_window:
            local=int(np.searchsorted(self.steps,step))
            self.window_writer.append(local,{**fields,'source_step':np.asarray(step,dtype=np.int32)})

    def finish(self,terminal_arrays):
        if not terminal_arrays or 'coords' not in terminal_arrays:
            raise ValueError('Native t=1 output coordinates are required')
        # Require the full inference first: an interrupted t=.5 run is not complete.
        if len(read_json(self.lineage_writer.manifest_path)['steps'])!=len(self.times):
            raise ValueError('Full inference has not reached its final step')
        window=self.window_writer.finalize()
        lineage=self.lineage_writer.finalize()
        arrays={k:np.asarray(v)[None] for k,v in terminal_arrays.items()}
        terminal=_publish_arrays(arrays,self.root/'terminal.h5',self.codec)
        with TrajectoryPackage(self.window_writer.final_path) as package:
            columns={'local_step':np.repeat(np.arange(len(self.steps),dtype=np.int32),len(package.read('selected_indices',0))),
                     'source_step':np.repeat(self.steps.astype(np.int32),len(package.read('selected_indices',0)))}
            b=len(package.read('selected_indices',0))
            if np.asarray(terminal_arrays['coords']).shape[0]!=b:
                raise ValueError('Terminal output dropped candidate slots')
            columns['slot']=np.tile(np.arange(b,dtype=np.int32),len(self.steps))
            for key in EVENT_FIELDS:
                columns[key]=package.read(key).reshape(-1)
            for key in ('resampled','state_time'):
                columns[key]=np.repeat(package.read(key),b)
            columns['score_time']=np.repeat(package.read('score_time')[:,0],b)
            columns['selected']=columns['offspring_count']>0
            # Uncompressed, typed, small lookup table; no duplicated coordinates.
            pq.write_table(pa.table(columns),self.root/'selection_events.parquet',compression=None)
        with TrajectoryPackage(self.lineage_writer.final_path) as package:
            all_indices=package.read('selected_indices')
        counts=terminal_descendants(all_indices,self.steps)
        pq.write_table(pa.table({'source_step':np.repeat(self.steps.astype(np.int32),b),
                                'slot':np.tile(np.arange(b,dtype=np.int32),len(self.steps)),
                                'terminal_descendant_count':counts.reshape(-1)}),
                       self.root/'terminal_ancestry.parquet',compression=None)
        records=[_package_record(self.root,self.window_writer.final_path),
                 _package_record(self.root,self.lineage_writer.final_path),terminal,
                 _package_record(self.root,self.root/'selection_events.parquet'),
                 _package_record(self.root,self.root/'terminal_ancestry.parquet')]
        manifest={**self.schema,'state':'complete','files':records,'particles':b,
                  'window_events':len(self.steps),'total_integration_steps':len(self.times),
                  'last_selected_proposal_time':self._last_proposal_time(),
                  'terminal_coordinate_frame':'world Angstrom, native output; window coordinates are model units',
                  'retained_geometry':'all window candidates, including zero-offspring; exact native terminal output',
                  'not_retained':['post-window intermediate geometry','restart/endpoint/RNG pickles'],
                  'losslessness_scope':'all arrays in the declared retained schema, original dtype and bit patterns',
                  'future_labels':'terminal_ancestry.parquet is an outcome diagnostic, not an online selection feature'}
        atomic_json(self.manifest_path,manifest)
        return manifest

    def _last_proposal_time(self):
        with TrajectoryPackage(self.window_writer.final_path) as package:
            return float(package.read('state_time',len(self.steps)-1))


def verify_learning_batch(batch):
    batch=Path(batch).resolve()
    manifest=read_json(batch/'learning_manifest.json')
    if manifest.get('format')!=FORMAT or manifest.get('state')!='complete':
        raise ValueError('Learning batch is incomplete or unsupported')
    verified=0
    for r in manifest.get('campaign_context',[]):
        path=checked_path(batch.parents[1]/r['path'],batch.parents[1])
        if digest(path)!=r['sha256']:
            raise ValueError('Campaign context checksum mismatch: '+r['path'])
    for r in manifest['files']:
        path=checked_path(batch/r['path'],batch)
        if path.stat().st_size!=r['bytes'] or digest(path)!=r['sha256']:
            raise ValueError('Learning file checksum mismatch: '+r['path'])
        if path.suffix=='.h5':
            with TrajectoryPackage(path) as package:
                verified+=package.verify()
    with TrajectoryPackage(batch/'trajectory.h5') as window, TrajectoryPackage(batch/'lineage/trajectory.h5') as lineage:
        steps=np.asarray(manifest['selection_steps'])
        if not np.array_equal(window.read('source_step'),steps) or not np.array_equal(lineage.read('source_step'),np.arange(manifest['total_integration_steps'])):
            raise ValueError('Original step mapping is inconsistent')
        for key in LINEAGE_FIELDS:
            if array_digest(window.read(key))!=array_digest(lineage.read(key)[steps]):
                raise ValueError('Window and full lineage disagree: '+key)
        if (batch/'trajectory.stages').exists() or (batch/'lineage/trajectory.stages').exists():
            raise ValueError('Consolidated batch still has temporary stages')
    return {'verified':True,'arrays':verified,'window_events':manifest['window_events'],
            'total_steps':manifest['total_integration_steps'],'particles':manifest['particles']}


def read_scoring_event(batch,source_step,slot=None,*,geometry=True,terminal_outcome=False):
    """JSON-ready evidence: one event/population, never the complete large trace."""
    batch=Path(batch)
    manifest=read_json(batch/'learning_manifest.json')
    if manifest.get('format')!=FORMAT or manifest.get('state')!='complete':
        raise ValueError('Learning batch is incomplete')
    if source_step not in manifest['selection_steps']:
        raise ValueError('Requested step is outside the stored scoring window')
    local=manifest['selection_steps'].index(source_step)
    needed=['trajectory.h5']+(['lineage/trajectory.h5'] if terminal_outcome else [])
    for name in needed:
        record=next(r for r in manifest['files'] if r['path']==name)
        if digest(batch/name)!=record['sha256']:
            raise ValueError('Scoring event source checksum mismatch: '+name)
    with TrajectoryPackage(batch/'trajectory.h5') as package:
        if slot is None:
            fields=[k for k in package.keys if geometry or not any(x in k for x in ('coords','atomics','bonds','charges','hybridization','mask'))]
            values={k:package.read(k,local) for k in fields}
            values['children_next_slots']=[np.flatnonzero(package.read('selected_indices',local)==s).tolist() for s in range(manifest['particles'])]
        else:
            if not 0<=slot<manifest['particles']:
                raise ValueError('Invalid particle slot')
            values=package.node(local,slot)
            if not geometry:
                values={k:v for k,v in values.items() if not any(x in k for x in ('coords','atomics','bonds','charges','hybridization','mask'))}
        dtypes={k:package.file['fields'][k].attrs['dtype'] for k in values if k in package.keys}
    result={'format':FORMAT,'metadata':manifest['metadata'],'source_step':source_step,'local_step':local,
            'slot':slot,'window':manifest['window'],'array_dtypes':dtypes,'observation':values,
            'coordinate_convention':'window model units; world = model * coord_scale + target COM (frame_batch JSON)',
            'predicted_coords_semantics':'t=1 forecast made at score_time, not actual current state'}
    if 'batch' in manifest['metadata']:
        campaign=batch.resolve().parents[1]
        for r in manifest.get('campaign_context',[]):
            path=checked_path(campaign/r['path'],campaign)
            if digest(path)!=r['sha256']:
                raise ValueError('Scoring event coordinate context checksum mismatch: '+r['path'])
        frame_path=campaign/f'frame_batch_{int(manifest["metadata"]["batch"]):03d}.json'
        if frame_path.exists():
            frame=read_json(frame_path)
            result['frame']={k:v[slot] if slot is not None and isinstance(v,list) and len(v)==manifest['particles'] else v
                             for k,v in frame.items()}
        config_path=campaign/'config.json'
        if config_path.exists():
            cfg=read_json(config_path)
            result['vocabularies']={k:cfg[k] for k in ('atom_vocabulary','charge_vocabulary','hybridization_vocabulary') if k in cfg}
            result['input_files']=cfg.get('input_files',{})
    if terminal_outcome:
        with TrajectoryPackage(batch/'lineage/trajectory.h5') as lineage:
            ids=lineage.read('selected_indices')
        ancestors=np.arange(manifest['particles'])
        for i in range(len(ids)-1,source_step-1,-1):
            ancestors=ids[i,ancestors]
        slots=range(manifest['particles']) if slot is None else [slot]
        result['terminal_outcome']={str(s):np.flatnonzero(ancestors==s).tolist() for s in slots}
        result['terminal_outcome_warning']='Observed descendants of this selection run; no counterfactual outcome for eliminated siblings'
    return clean(result)


def main(argv=None):
    p=argparse.ArgumentParser(description='Read one selection event for downstream Analyst/LLM input')
    p.add_argument('--batch-directory',required=True)
    p.add_argument('--step',type=int,required=True,help='Original integration index, not a time bin')
    p.add_argument('--slot',type=int)
    p.add_argument('--scores-only',action='store_true')
    p.add_argument('--terminal-outcome',action='store_true')
    p.add_argument('--verify',action='store_true')
    p.add_argument('--output')
    a=p.parse_args(argv)
    if a.verify:verify_learning_batch(a.batch_directory)
    result=read_scoring_event(a.batch_directory,a.step,a.slot,geometry=not a.scores_only,terminal_outcome=a.terminal_outcome)
    if a.output:atomic_json(a.output,result)
    else:print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))


if __name__=='__main__':main()
