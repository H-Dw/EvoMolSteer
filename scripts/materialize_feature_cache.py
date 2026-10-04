"""Rebuild a retired cache from frozen raw inputs, without changing results."""
import argparse
from pathlib import Path
import shutil
import uuid
from evomolsteer.io import digest,read_json,write_json
from evomolsteer.ingest import ingest


def materialize(analysis):
    analysis=Path(analysis).resolve();record_path=analysis/'feature_cache_manifest.json'
    record=read_json(record_path);target=analysis/'features.parquet'
    if target.exists():
        if digest(target)!=record['sha256']:raise ValueError('Existing cache differs from the registered cache')
        print('Verified existing cache');return
    cfg=read_json(analysis/'config.json');source=read_json(analysis/'ingest_manifest.json')['source_root']
    tmp=analysis/('_cache_rebuild_'+uuid.uuid4().hex)
    if not tmp.resolve().is_relative_to(analysis):raise ValueError('Unsafe temporary location')
    rebuild={**cfg,'output_policy':{**cfg['output_policy'],'retain_feature_cache':True,'keep_worker_shards':False}}
    ingest(source,tmp,rebuild)
    candidate=tmp/'features.parquet'
    if digest(candidate)!=record['sha256']:
        raise ValueError('Rebuilt cache hash differs; retained temporary run for inspection: '+str(tmp))
    shutil.move(str(candidate),str(target))
    files=[p for p in tmp.rglob('*') if p.is_file()]
    directories=sorted([p for p in tmp.rglob('*') if p.is_dir()],key=lambda p:len(p.parts),reverse=True)
    if any(p.is_symlink() or not p.resolve().is_relative_to(tmp.resolve()) for p in files+directories):
        raise ValueError('Unexpected temporary path; retained rebuild diagnostics')
    for p in files:p.unlink()
    for p in directories:p.rmdir()
    tmp.rmdir()
    write_json(record_path,{**record,'status':'available','rebuilt_from_raw_and_hash_verified':True})
    print('Rebuilt feature cache with identical SHA-256:',record['sha256'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--analysis',required=True);a=p.parse_args();materialize(a.analysis)
