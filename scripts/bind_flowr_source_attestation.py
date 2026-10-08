"""Attach unchanged-source evidence to one completed owned generation config.

Only metadata is added after inference; model sources, tensors, reward program,
and existing parsed configuration fields remain unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path

from evomolsteer.io import digest, read_json, write_json


def bind(dataset, campaign, before_record, after_record):
    root = Path(dataset).resolve()
    if not campaign or campaign in ['.', '..'] or any(c in campaign for c in '/\\:'):
        raise ValueError('One campaign name is required')
    run = (root / 'results' / campaign).resolve()
    if not run.is_relative_to(root / 'results'):
        raise ValueError('Owned completed generation dataset required')
    if read_json(run / 'COMPLETE.json').get('status') != 'complete':
        raise ValueError('Attach source evidence only after complete inference')
    before, after = read_json(before_record), read_json(after_record)
    schema = 'flowr-upstream-source-attestation-1.0'
    if (before.get('schema_version') != schema or after.get('schema_version') != schema
            or before['input_dataset'] != after['input_dataset'] or before['files'] != after['files']
            or after.get('source_unchanged') is not True or after.get('reference_sha256') != digest(before_record)):
        raise ValueError('Before/after source evidence differs or is not bound')
    config_path = run / 'config.json'
    config = read_json(config_path)
    if config.get('experiment', {}).get('campaign') != campaign:
        raise ValueError('Generation configuration belongs to a different campaign')
    if 'upstream_source_attestation' in config:
        raise FileExistsError('A completed source binding cannot be overwritten')
    canonical_before = hashlib.sha256((json.dumps(config, ensure_ascii=False, indent=2,
        sort_keys=True, allow_nan=False) + '\n').encode('utf-8')).hexdigest()
    config['upstream_source_attestation'] = {
        'schema_version': 'bound-flowr-upstream-source-attestation-1.0',
        'before': before, 'after': after,
        'before_record_sha256': digest(before_record), 'after_record_sha256': digest(after_record),
        'generation_config_sha256_before_binding': digest(config_path),
        'generation_config_canonical_sha256_before_binding': canonical_before,
        'binder_sha256': digest(__file__),
        'scope': 'Declared upstream source files only; metadata added after inference, no numerical change',
    }
    write_json(config_path, config)
    return {'bound': True, 'campaign': campaign, 'declared_source_files': len(before['files'])}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--campaign', required=True)
    parser.add_argument('--before-record', required=True)
    parser.add_argument('--after-record', required=True)
    args = parser.parse_args()
    print(bind(args.dataset, args.campaign, args.before_record, args.after_record))
