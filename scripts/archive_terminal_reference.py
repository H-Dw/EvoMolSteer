"""Package only terminal decoding outputs from a protected historical experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def archive(dataset,campaign,output):
    root=Path(dataset).resolve();out=Path(output).resolve()
    if out.exists() or out.is_relative_to(root):raise ValueError('Fresh output outside reference required')
    folder=root/'results'/campaign
    if not (folder/'COMPLETE.json').exists():raise ValueError('Reference incomplete')
    paths=list((root/'inputs').glob('*'))
    paths+=list(folder.glob('*.json'))
    for arm in ['single','unguided']:
        for batch in sorted((folder/arm).glob('batch_*')):
            for name in ['final_records.json','molecules_all_built.sdf','molecules_raw_decodable.sdf','COMPLETE.json']:
                p=batch/name
                if not p.is_file():raise FileNotFoundError(p)
                paths.append(p)
    paths=sorted(set(p for p in paths if p.is_file()))
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    out.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(out,'w:gz') as archive:
        for p in paths:archive.add(p,arcname=p.relative_to(root).as_posix(),recursive=False)
    manifest={'schema':'terminal-reference-1','source':str(root),'campaign':campaign,
              'archive_sha256':sha(out),'files':[{'path':p.relative_to(root).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size} for p in paths]}
    out.with_suffix(out.suffix+'.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'files':len(paths),'bytes':out.stat().st_size,'sha256':manifest['archive_sha256']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--campaign',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();archive(a.dataset,a.campaign,a.output)
