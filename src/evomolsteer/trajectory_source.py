"""Common read-only access to original NPZ or verified normalized trajectories."""
from contextlib import contextmanager
from pathlib import Path
import numpy as np
from .io import read_json,digest


def pocket_input_path(root,role):
    """Resolve a biological input role in new or historical datasets."""
    folder=Path(root)/'inputs'
    legacy={'target_protein':'3PE1_protein_aligned.pdb','target_ligand':'3PE1_ligand_aligned.sdf',
            'off_target_protein':'6KHF_protein_aligned.pdb','off_target_ligand':'6KHF_ligand_aligned.sdf'}
    if role not in legacy:raise ValueError('Unknown pocket role: '+role)
    spec=folder/'pocket_inputs.json'
    name=read_json(spec)['files'][role] if spec.exists() else legacy[role]
    path=(folder/name).resolve()
    if not path.is_relative_to(folder.resolve()) or not path.is_file():
        raise ValueError('Pocket input is missing/outside its declared dataset: '+role)
    return path


def pocket_input_hashes(root):
    """Portable file identities; an aliased off-target is stored only once."""
    folder=(Path(root)/'inputs').resolve()
    return {p.relative_to(folder).as_posix():digest(p) for p in
            (pocket_input_path(root,role) for role in
             ('target_protein','target_ligand','off_target_protein','off_target_ligand'))}


def validate_input_bundle(root,cfg):
    """Fail closed for unfinished/modified bundles; original NPZ roots are unchanged."""
    root = Path(root).resolve()
    learning=root/'learning_dataset_manifest.json'
    if learning.exists():
        manifest=read_json(learning)
        if manifest.get('format')!='evomolsteer.selection_learning_dataset.v1' or manifest.get('state')!='complete' or manifest['campaign']!=cfg['campaign']:
            raise ValueError('Incomplete/mismatched selection learning dataset')
        expected=dict(manifest['copied_files'])
        for record in manifest['batches']:
            relative=Path(record['path'])
            if cfg.get('include_batches') is not None and int(relative.parent.name.split('_')[1]) not in cfg['include_batches']:
                continue
            batch_path=(root/relative).resolve()
            if not batch_path.is_relative_to(root) or digest(batch_path)!=record['sha256']:
                raise ValueError('Selection learning batch manifest checksum mismatch')
            expected[record['path']]=record
            batch_manifest=read_json(batch_path)
            if batch_manifest.get('state')!='complete':
                raise ValueError('Incomplete learning batch')
            for item in batch_manifest['files']:
                expected[(relative.parent/item['path']).as_posix()]=item
        for relative,record in expected.items():
            path=(root/relative).resolve()
            if not path.is_relative_to(root) or digest(path)!=record['sha256']:
                raise ValueError('Selection learning checksum mismatch: '+relative)
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


def analysis_arrays(source, chemistry=False):
    """Load only consumed fields, retaining all ancestry checks and hard states."""
    keys = ['pic50_on','pic50_off','weight_on','weight_off','selection_probability',
            'selected_indices','offspring_count','root_slot','parent_slot','score_time',
            'step_size','state_time','resampled','mask','predicted_coords','predicted_atomics']
    keys += [rep+'_'+field for rep in ('current','proposal') for field in ('coords','atomics','bonds','charges')]
    if chemistry:
        keys += ['predicted_bonds', 'predicted_charges']
    return {key:source[key] for key in keys}
