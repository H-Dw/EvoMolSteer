"""Public prepare/verify/run/collect/analyze interface for historical path replay."""
import argparse
import json
from pathlib import Path

from evomolsteer.io import read_json
from .common import validate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','verify','run','collect','analyze','status'])
    p.add_argument('--repo',default='.',type=Path);p.add_argument('--config',required=True,type=Path)
    p.add_argument('--work',type=Path);p.add_argument('--output');p.add_argument('--retry-failed',action='store_true')
    a=p.parse_args()
    if a.action=='prepare':
        from .prepare import prepare
        print(prepare(a.repo,a.config));return
    cfg=read_json(a.config);validate(cfg);work=a.work or Path(cfg['remote_work'])
    if a.action=='verify':
        from .queue import verify
        print(json.dumps(verify(a.repo,cfg,work),indent=2))
    elif a.action=='run':
        from .queue import run
        print(json.dumps(run(a.repo,a.config,work,a.retry_failed),indent=2))
    elif a.action=='collect':
        from .collection import collect
        if a.output is None: p.error('collect requires --output (use - for streaming)')
        collect(cfg,work,a.output)
    elif a.action=='analyze':
        from .analysis import analyze
        print(json.dumps(analyze(a.repo,cfg,work,a.output),indent=2))
    else:
        from .queue import summary
        print(json.dumps({'queue':read_json(work/'queue_status.json') if (work/'queue_status.json').exists() else None,
                          'worker':read_json(work/'worker_state.json') if (work/'worker_state.json').exists() else None,
                          'conditions':summary(work,cfg)},indent=2))


if __name__=='__main__': main()
