"""Validate this campaign's scope, then use the existing bounded orchestration."""
import fcntl
import os
import runpy
import sys
from pathlib import Path
from agenticgen.uniguide.io import read


def main():
    repo, config_path = map(Path, sys.argv[1:3])
    config = read(config_path)
    if config['rounds'] != 5 or config['chemical_inpainting'] != {'enabled': True}:
        raise ValueError('This authorization is five rounds with chemical state inpainting')
    if not os.environ.get('DEEPSEEK_API_KEY'):
        raise ValueError('DEEPSEEK_API_KEY must be loaded from the interactive shell')
    for subject in read(Path(config['root'])/'SUBJECTS.json'):
        c = subject['condition']
        if len(c['controlled_atom_ids']) != 9 or set(c['controlled_atom_ids']) != {a['atom_id'] for a in c['interact_atoms']}:
            raise ValueError('Both teachers require identical condition and interact atom sets')
    # Hold the shared GPU lock for this one-shot chain. Never launch a second copy.
    with (Path(config['generation_context_root'])/'.executor.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        runner = repo/'agenticgen/docs/experiments/20261029_template_distance_formal/run_campaign.py'
        runpy.run_path(str(runner), run_name='__main__')


if __name__ == '__main__':main()
