"""Read-only comparison contracts against a completed generation collection."""
import hashlib
import json
from pathlib import Path
import tarfile

from ..io import digest,read_json
from ..storage.transactions import checked_path


def _native_manifest(root,state):
    for row in state['catalog']:
        record=state['targets'][row['target_id']]
        directory=checked_path(root/record['dataset'],root)
        path=directory/'learning_dataset_manifest.json'
        if path.is_file():return read_json(path)
        if record['state']=='archived':
            archive=next(a for a in state['archives'] if a['path']==record['archive'])
            if not archive['metadata'].get('verified'):raise ValueError('Reference archive is not verified')
            key=next(k for k,v in archive['directories'].items() if v==record['dataset'])
            wanted='targets/'+key+'/learning_dataset_manifest.json'
            with tarfile.open(checked_path(root/archive['path'],root),'r|gz') as stream:
                for member in stream:
                    if member.name==wanted:
                        if not member.isfile() or member.size>8*1024*1024:raise ValueError('Invalid reference manifest')
                        return json.load(stream.extractfile(member))
    raise ValueError('Reference collection has no native completion manifest')


def validate_reference_campaign(cfg,selected):
    """Reject a mismatched baseline before claiming targets or starting a GPU."""
    if not cfg.reference_campaign:return None
    root=Path(cfg.reference_campaign).resolve();output=Path(cfg.output_dataset).resolve()
    if root==output or root.is_relative_to(output) or output.is_relative_to(root):
        raise ValueError('Reference and new generation outputs must be disjoint')
    state=read_json(root/'campaign_state.json')
    if state['status']!='complete' or any(r['state'] not in ('complete','archived') for r in state['targets'].values()):
        raise ValueError('Comparison reference must be a fully completed collection')
    expected=state['config']
    fields=('samples','batch_size','steps','seed','window_start','window_end','storage_codec','capture_bond_probabilities')
    for name in fields:
        if getattr(cfg,name)!=expected[name]:raise ValueError('Reference inference setting differs: '+name)
    if expected.get('arm','single')!='single':raise ValueError('Expected a single-target Steer reference')
    def identity(rows):
        return [{'target_id':r['target_id'],'files':{k:v['sha256'] for k,v in sorted(r['files'].items())}} for r in rows]
    frozen=identity(state['catalog']);current=identity(selected)
    if current!=frozen:raise ValueError('Target identities, order or input checksums differ from the Steer reference')
    manifest=_native_manifest(root,state)
    checkpoint_sha=digest(cfg.checkpoint)
    if manifest.get('state')!='complete' or manifest.get('checkpoint_sha256')!=checkpoint_sha:
        raise ValueError('Checkpoint differs from the native Steer reference')
    return {'format':'evomolsteer.paired_generation.v1','reference':str(root),
            'reference_state_sha256':digest(root/'campaign_state.json'),
            'target_catalog_sha256':hashlib.sha256(json.dumps(frozen,sort_keys=True).encode()).hexdigest(),
            'targets':len(selected),'matched_settings':{k:getattr(cfg,k) for k in fields},
            'checkpoint_sha256':checkpoint_sha,'reference_sampler_sha256':manifest.get('flowr_sampler_sha256'),
            'current_sampler_sha256':digest(Path(cfg.flowr_root)/'flowr/models/fm_pocket.py'),
            'comparison':'same target inputs, candidate slots, seed, SDE settings and decoder policy; resampling disabled',
            'randomness_note':'same initial priors/seeds; later paths are not identical after Steer consumes selection RNG'}
