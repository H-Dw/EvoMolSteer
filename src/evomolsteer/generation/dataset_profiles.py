"""Explicit collection profiles for the source-checkout generation controller."""
from pathlib import Path

from ..io import read_json

DEFAULT_DATASET='hiqbind'
DATASET_CONFIGS={
    'hiqbind':'generation_hiqbind_steer.json',
    'crossdocked100':'generation_crossdocked100_steer.json',
}


def load_dataset_profile(dataset=DEFAULT_DATASET,*,repository=None):
    """Read a selected profile; missing inputs never trigger a different dataset."""
    if dataset not in DATASET_CONFIGS:
        raise ValueError('Unknown dataset profile: '+str(dataset))
    root=Path(repository) if repository is not None else Path(__file__).resolve().parents[3]
    return read_json(root/'configs'/DATASET_CONFIGS[dataset])


def validate_profile_inputs(dataset,values):
    """HiQBind membership comes from an explicit prepared test manifest."""
    if dataset!='hiqbind':return
    manifest=values.get('target_manifest')
    if not manifest or not Path(manifest).is_file() or not Path(values['input_dataset']).is_dir():
        raise FileNotFoundError(
            'HiQBind needs a prepared test input directory and an explicit target manifest. '
            f"input_dataset={values['input_dataset']}; target_manifest={manifest}. "
            'Provide both --input-dataset and --target-manifest or use a prepared --config. '
            'No inference or automatic dataset fallback was started.')
