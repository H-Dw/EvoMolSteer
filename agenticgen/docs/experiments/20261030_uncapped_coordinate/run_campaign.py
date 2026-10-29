"""One authorized, uncapped two-teacher campaign; no implicit recovery."""
import fcntl
import os
from pathlib import Path
import runpy
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

from agenticgen.uniguide.absolute_runtime import validate_config
from agenticgen.uniguide.absolute_coordinate import validate
from agenticgen.uniguide.io import read, write

TEACHERS = {'6pyy_H7S_A_402': 'single_b004_final_015',
            '3w54_RNB_A_504': 'single_b000_final_003'}


def validate_scope(config, subjects):
    config = validate_config(config)
    if config['rounds'] != 5 or config['chemical_inpainting'] != {'enabled': True}:
        raise ValueError('This campaign requires chemical state inpainting and five rounds')
    if config.get('sampling_recovery'):
        raise ValueError('Uncapped guidance requires fresh calibration; no capped sampling transfer')
    if len(subjects) != 2 or {s['teacher']['target']: s['teacher']['node_id'] for s in subjects} != TEACHERS:
        raise ValueError('Authorized two teachers differ')
    for subject in subjects:
        condition = validate(subject['condition'])
        teacher = subject['teacher']
        ids = [atom['id'] for atom in condition['atoms']]
        core = [condition['reference_coordinates_angstrom'][ids.index(i)]
                for i in condition['controlled_atom_ids']]
        expected_ids = [f'atom_{i:04d}' for i in teacher['core_sdf_atoms']]
        if len(core) != 9 or condition['controlled_atom_ids'] != expected_ids or core != teacher['core_A']:
            raise ValueError('Condition must be the original nine PLIP atoms and teacher coordinates')
    return config


def main():
    repo, config_path = map(Path, sys.argv[1:3])
    config = read(config_path)
    root = Path(config['root'])
    try:
        config = validate_scope(config, read(root/'SUBJECTS.json'))
        if not os.environ.get('DEEPSEEK_API_KEY'):
            raise ValueError('Interactive-shell API environment was not loaded')
        with (Path(config['generation_context_root'])/'.executor.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            runner = repo/'agenticgen/docs/experiments/20261029_template_distance_formal/run_campaign.py'
            runpy.run_path(str(runner), run_name='__main__')
    except Exception as error:
        if not (root/'CAMPAIGN_FAILED.json').exists():
            write(root/'CAMPAIGN_FAILED.json', dict(timestamp=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                error=str(error), no_automatic_retry=True, source_artifacts_preserved=True))
        raise


if __name__ == '__main__':
    main()
