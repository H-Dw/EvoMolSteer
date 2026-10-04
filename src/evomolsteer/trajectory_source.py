"""Common read-only access to original NPZ or verified normalized trajectories."""
from contextlib import contextmanager
from pathlib import Path
import numpy as np
from .io import read_json,digest


def validate_input_bundle(root,cfg):
    """Fail closed for unfinished/modified bundles; original NPZ roots are unchanged."""
    root = Path(root).resolve()
    retirement=root/'trajectory_retirement.json'
    if retirement.exists():
        raise ValueError('Original trajectories were retired; use dataset '+read_json(retirement)['output_dataset'])
    conversion=root/'conversion_manifest.json'
    if conversion.exists() and not read_json(conversion).get('complete'):
        raise ValueError('Dataset conversion is not yet complete')
    path = root/'input_bundle_manifest.json'
    if not path.exists():
        return
    manifest = read_json(path)
    if manifest.get('format')!='evomolsteer.analysis_inputs.v1' or not manifest.get('complete'):
        raise ValueError('Incomplete or unknown analysis input bundle')
    if manifest['campaign']!=cfg['campaign']:
        raise ValueError('Input bundle campaign mismatch')
    expected = {}
    for record in manifest['packages']:
        relative = Path(record['package'])
        if cfg.get('include_batches') is not None and int(relative.parent.name.split('_')[1]) not in cfg['include_batches']:
            continue
        expected[relative] = record['package_sha256']
    for relative,record in manifest['copied_files'].items():
        expected[Path(relative)] = record['sha256']
    for relative,sha in expected.items():
        source = (root/relative).resolve()
        if not source.is_relative_to(root) or digest(source)!=sha:
            raise ValueError('Normalized input checksum mismatch: '+str(relative))


def trajectory_paths(campaign):
    paths = sorted(list(Path(campaign).glob('*/batch_*/trajectory.npz'))+
                   list(Path(campaign).glob('*/batch_*/trajectory.h5')))
    if len({p.parent for p in paths}) != len(paths):
        raise ValueError('Ambiguous input: both NPZ and HDF5 for the same batch')
    return paths


@contextmanager
def open_trajectory(path):
    path = Path(path)
    if path.suffix == '.npz':
        with np.load(path,allow_pickle=False) as z:
            yield z
    elif path.suffix == '.h5':
        from .storage.trajectory import TrajectoryPackage
        with TrajectoryPackage(path) as p:
            # Keep a mapping interface; load arrays only when requested.
            class Fields:
                files = p.keys
                def __getitem__(self,key):
                    return p.read(key)
            yield Fields()
    else:
        raise ValueError('Unsupported trajectory source')


def analysis_arrays(source):
    """Load only consumed fields, retaining all ancestry checks and hard states."""
    keys = ['pic50_on','pic50_off','weight_on','weight_off','selection_probability',
            'selected_indices','offspring_count','root_slot','parent_slot','score_time',
            'step_size','state_time','resampled','mask','predicted_coords','predicted_atomics']
    keys += [rep+'_'+field for rep in ('current','proposal') for field in ('coords','atomics','bonds','charges')]
    return {key:source[key] for key in keys}
