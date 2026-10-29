"""Scope-checked absolute-coordinate campaign, using the bounded shared runner."""
import fcntl
import os
from pathlib import Path
import runpy
import sys
from datetime import datetime
from zoneinfo import ZoneInfo
from agenticgen.uniguide.io import read, write
from agenticgen.uniguide.absolute_coordinate import POLICY, validate


def main():
    repo, config_path=map(Path,sys.argv[1:3])
    config=read(config_path);root=Path(config['root'])
    try:
        if (config['protocol']!=POLICY or config['rounds']!=5
                or config['chemical_inpainting']!={'enabled':True}
                or config['agent_tools']['skill_release']!='2.4.0'):
            raise ValueError('Expected absolute coordinates, chemical state inpainting and five rounds')
        if not os.environ.get('DEEPSEEK_API_KEY'):
            raise ValueError('Interactive-shell API environment was not loaded')
        expected={'6pyy_H7S_A_402':'single_b004_final_015','3w54_RNB_A_504':'single_b000_final_003'}
        subjects=read(root/'SUBJECTS.json')
        if {s['teacher']['target']:s['teacher']['node_id'] for s in subjects}!=expected:
            raise ValueError('Authorized teachers differ')
        for subject in subjects:
            c=validate(subject['condition'])
            if len(c['controlled_atom_ids'])!=9:raise ValueError('Nine original condition atoms required')
        with (Path(config['generation_context_root'])/'.executor.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            runpy.run_path(str(repo/'agenticgen/docs/experiments/20261029_template_distance_formal/run_campaign.py'),run_name='__main__')
    except Exception as error:
        if not (root/'CAMPAIGN_FAILED.json').exists():
            write(root/'CAMPAIGN_FAILED.json',dict(timestamp=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                error=str(error),no_automatic_retry=True,source_artifacts_preserved=True))
        raise


if __name__=='__main__':main()
