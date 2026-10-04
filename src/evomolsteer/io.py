"""Deterministic storage, provenance and safe archive handling."""
from pathlib import Path
import hashlib, json, tarfile
import numpy as np
import pandas as pd

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def feature_cache_path(analysis):
    path=Path(analysis)/'features.parquet'
    if not path.exists():
        raise FileNotFoundError('Feature cache is unavailable or retired. Rebuild it with scripts/materialize_feature_cache.py --analysis '+str(Path(analysis)))
    return path

def clean(value):
    if isinstance(value,dict): return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [clean(v) for v in value]
    if isinstance(value,np.ndarray): return clean(value.tolist())
    if isinstance(value,np.generic): return clean(value.item())
    if isinstance(value,float) and not np.isfinite(value): return None
    return value

def write_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')

def write_table(path,rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    df=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
    if path.suffix=='.parquet': df.to_parquet(path,index=False)
    else: df.to_csv(path,index=False,float_format='%.12g',lineterminator='\n')
    return df

def safe_extract(archive,destination,expected_sha256=None):
    archive,destination=Path(archive),Path(destination).resolve()
    sha=digest(archive)
    if expected_sha256 and sha!=expected_sha256: raise ValueError('Archive SHA256 mismatch')
    destination.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive,'r:*') as tf:
        for m in tf.getmembers():
            target=(destination/m.name).resolve()
            if not target.is_relative_to(destination) or not (m.isfile() or m.isdir()):
                raise ValueError('Unsafe archive member: '+m.name)
        tf.extractall(destination,filter='data')
    return sha

def validate_checksums(root):
    root=Path(root).resolve(); path=root/'SHA256SUMS'
    if not path.exists(): raise ValueError('Full experiment checksum manifest missing')
    count=0
    for line in path.read_text().splitlines():
        sha,name=line.split('  ',1); f=(root/name).resolve()
        if not f.is_relative_to(root) or not f.is_file() or digest(f)!=sha:
            raise ValueError('Internal checksum failed: '+name)
        count+=1
    return count
