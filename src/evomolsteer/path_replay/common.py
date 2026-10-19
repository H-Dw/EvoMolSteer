"""Frozen protocol identities, B50 membership and atomic batch receipts."""
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from evomolsteer.io import digest, read_json
from evomolsteer.storage.transactions import atomic_json

NAME = 'path_replay_b50_20261018'
FLOWR_COMMIT = '739a33c2aab74d074dd24605b9a2e34dc809ae9c'
FLOWR_FILES = {
    'flowr/models/fm_pocket.py': 'beaace3ab92a8377bc8eab95333c8f40ce370ca9224f7a78927c5a424bb38426',
    'flowr/models/integrator.py': 'e9abd83167f708da4712db2ac49ab24ac44a39c59c46a54c20789c7873c58000',
    'flowr/scriptutil.py': 'ad9060d2a3e234b84362efa7d945d61186958548af1a5281f2805cc3e285a046',
    'flowr/gen/generate_from_pdb_selective.py': '1f0d52e9867f0f9125bec8de07c358b81afeed381a032444faad28e26e15aa97',
    'flowr/gen/utils.py': '2185b671e62bea6012dd38f4dfb8e6ba877285769b8609cc252c28760be440e0',
}
TERMINAL = {'complete', 'failed', 'skipped_condition_error'}


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def load_state(path):
    import torch
    with gzip.open(path, 'rb') as f:
        return torch.load(f, map_location='cpu', weights_only=False)


def ancestry(selected):
    """Return pre-selection parent slots [step, final seat] and t0 roots."""
    selected=np.asarray(selected)
    if selected.ndim!=2 or selected.dtype.kind not in 'iu': raise ValueError('Invalid lineage indices')
    steps,batch=selected.shape
    if np.any(selected<0) or np.any(selected>=batch): raise ValueError('Parent outside historical pool')
    before=np.empty_like(selected); slots=np.arange(batch)
    for step in reversed(range(steps)):
        slots=selected[step,slots]; before[step]=slots
    return before,slots.copy()


def program_identity(program):
    """Remove only known provenance fields, retaining every operative parameter."""
    return identity({k:v for k,v in program.items() if k not in
                     {'program_id','analyst_sha256','designer_sha256','evidence_bundle_sha256'}})


def validate(config):
    if identity({k:v for k,v in config.items() if k!='protocol_sha256'}) != config['protocol_sha256']:
        raise ValueError('Frozen replay protocol changed')
    if config['schema_version']!='historical-path-replay-1' or config['batch_size']!=50 or config['steps']!=100:
        raise ValueError('Expected fixed B50 / 100-step protocol')
    if config['flowr_commit']!=FLOWR_COMMIT or config['flowr_files']!=FLOWR_FILES:
        raise ValueError('Historical FLOWR source identity changed')
    attempts=config['attempts']; ids=[x['id'] for x in attempts]
    if len(ids)!=1000 or len(set(ids))!=1000 or sorted(x['rank'] for x in attempts)!=list(range(1,1001)):
        raise ValueError('Expected all 1000 uniquely ranked original final paths')
    groups=config['groups']
    if len(groups)!=20 or any(len(g['ids'])!=50 for g in groups): raise ValueError('B50 grouping changed')
    if [x for g in groups for x in g['ids']]!=ids: raise ValueError('Group order differs from immutable path order')
    if {x['id'] for x in attempts[:200]}!={x['id'] for x in attempts if x['rank']<=200}:
        raise ValueError('The shared first four groups must be exactly historical top20%')
    sizes={x['id']:x['n'] for x in config['conditions']}
    if sizes!={'unguided':200,'M2':200,'NC_R11':200,'ancestor_R02_v3':1000,
               'motif_R15_B50':1000,'motif_R15_v2':1000}: raise ValueError('Condition budget differs')
    aliases=[c for c in config['conditions'] if c.get('alias_of')]
    if len(aliases)!=1 or aliases[0]['alias_of']!='motif_R15_B50': raise ValueError('Unexpected result alias')


def condition_program(repo, condition):
    from evomolsteer.elite_alignment.common import validate_condition
    return validate_condition(repo,condition)


def groups_for(config, condition):
    if 'group_indices' in condition:
        indices=condition['group_indices']
        if len(set(indices))!=len(indices) or condition['n']!=50*len(indices):
            raise ValueError('Research condition has inconsistent B50 group selection')
        by_index={g['index']:g for g in config['groups']}
        return [by_index[i] for i in indices]
    return config['groups'][:condition['n']//config['batch_size']]


def batch_path(work, condition, group):
    return Path(work)/'methods'/condition['id']/f"group_{group['index']:03d}"/'COMPLETE.json'


def binding(config, condition, group):
    return identity({'protocol':config['protocol_sha256'],'condition':condition,'group':group})


def read_batch(work, config, condition, group):
    path=batch_path(work,condition,group)
    if not path.exists(): return None
    r=read_json(path)
    if r.get('binding')!=binding(config,condition,group) or r.get('status') not in TERMINAL:
        raise ValueError('Replay batch receipt identity mismatch: '+str(path))
    if r['status']=='complete':
        if not r.get('artifacts'): raise ValueError('Complete replay batch has no artifacts')
        for key,sha in r['artifacts'].items():
            if digest(path.parent/key)!=sha: raise ValueError('Replay artifact changed: '+key)
        if len(r['records'])!=50 or [x['id'] for x in r['records']]!=group['ids']:
            raise ValueError('Result rows differ from frozen B50 membership')
    return r


def fail_batch(work, config, condition, group, error, status='failed', phase='inference'):
    path=batch_path(work,condition,group)
    if path.exists(): return read_batch(work,config,condition,group)
    r={'status':status,'binding':binding(config,condition,group),'condition':condition['id'],
       'group':group['index'],'phase':phase,'error':str(error),'finished_unix':time.time(),
       'records':[{'id':k,'pic50':None,'status':status,'valid_connected':False} for k in group['ids']]}
    atomic_json(path,r); return r
