"""Inventory/delete an explicit set of obsolete experiment outputs, never source data."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


ALLOWED = [Path('/root/private_data/MolSteer/flowr_root/experiments/evomolsteer_single_guidance_20261004'),
           Path('/root/private_data/MolSteer/flowr_root/experiments/evomolsteer_multistage_20261005'),
           Path('/root/private_data/MolSteer/flowr_root/experiments/evomolsteer_online_20261004/dataset'),
           Path('/root/private_data/MolSteer/EvoMolSteer/results/continuous_single_v3'),
           Path('/opt/evomolsteer-multistage-20261005')]
PROTECTED = [Path('/root/private_data/MolSteer/flowr_root/experiments/ck2_clk3_lineage_20261003'),
             Path('/root/private_data/MolSteer/flowr_root/checkpoints'),
             Path('/root/private_data/MolSteer/flowr_root/experiments/evomolsteer_online_20261004/runtime_deps')]


def inventory():
    result=[]
    for p in ALLOWED:
        resolved=p.resolve()
        if resolved!=p or p.is_symlink() or any(resolved.is_relative_to(q) or q.is_relative_to(resolved) for q in PROTECTED):
            raise ValueError('Unsafe cleanup target: '+str(p))
        if not p.exists():continue
        files=[]
        for f in sorted(p.rglob('*')):
            if f.is_symlink():raise ValueError('Symlink requires separate review: '+str(f))
            if f.is_file():files.append({'path':f.relative_to(p).as_posix(),'bytes':f.stat().st_size})
        result.append({'path':str(p),'bytes':sum(f['bytes'] for f in files),'files':files})
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--apply-plan')
    a=p.parse_args();records=inventory()
    output=Path(a.output).resolve()
    if any(output.is_relative_to(t) for t in ALLOWED):raise ValueError('Audit must survive cleanup')
    if a.apply_plan:
        plan=json.loads(Path(a.apply_plan).read_text())
        if plan['targets']!=records:raise ValueError('Cleanup inventory changed since review')
        if not plan.get('local_backups_verified'):raise ValueError('Local backup verification required')
        for record in records:shutil.rmtree(record['path'])
        result={'status':'deleted','targets':records,'bytes_removed':sum(r['bytes'] for r in records),
                'protected_paths':[str(t) for t in PROTECTED]}
    else:result={'status':'inventory','targets':records,'local_backups_verified':False,
                 'protected_paths':[str(t) for t in PROTECTED]}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'targets':[{'path':r['path'],'bytes':r['bytes'],'files':len(r['files'])} for r in records]}))
