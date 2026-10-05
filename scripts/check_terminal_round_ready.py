"""Fail closed before inference unless the previous round's outputs are retired."""
import argparse
import json
from pathlib import Path


def check(manifest,campaign,program=None):
    m=json.loads(Path(manifest).read_text())
    if m['master_seed']!=42 or not 1<=m['round']<=m['maximum_rounds']<=5:raise ValueError('Round/seed contract')
    if campaign not in m['campaigns']:raise ValueError('Campaign not authorized by current round manifest')
    if program is not None:
        p=json.loads(Path(program).read_text())
        if p['round']!=m['round'] or p['seed']!=m['master_seed']:raise ValueError('Reward program does not match frozen round/seed')
    audit=json.loads(Path(m['cleanup_audit']).read_text())
    if audit['status']!='deleted':raise ValueError('Cleanup not complete')
    if any(Path(t['path']).exists() for t in audit['targets']):raise ValueError('Previous generated output still exists')
    if not Path(audit['surviving_report']).is_file():raise ValueError('Previous result report missing')
    if not all(Path(p).exists() for p in m['protected_inputs']):raise ValueError('Protected reference/checkpoint missing')
    if (Path(m['generated_root'])/'results'/campaign).exists():raise ValueError('Refusing existing candidate output')
    return {'ready':True,'round':m['round'],'campaign':campaign,'seed':42,'retired_bytes':audit['bytes']}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--campaign',required=True);p.add_argument('--program')
    a=p.parse_args();print(json.dumps(check(a.manifest,a.campaign,a.program)))
