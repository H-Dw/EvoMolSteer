"""Atomic remote launch identity; duplicate dispatch reuses the same inference."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def dispatch(repo,work,number,campaign,program,reference,previous,arms='gradient',batches='0,1'):
    repo,work,program,reference=map(lambda p:Path(p).resolve(),(repo,work,program,reference))
    if not 1<=number<=30 or Path(campaign).name!=campaign or not campaign.startswith(f'coordinate_r{number:02d}_'):
        raise ValueError('Bound coordinate round/campaign required')
    if not program.is_relative_to(repo/'configs') or not reference.is_relative_to(repo/'configs'):
        raise ValueError('Committed program/reference paths required')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    identity={'round':number,'campaign':campaign,'program_sha256':sha(program),'reference_sha256':sha(reference),
        'previous_report':previous,'arms':arms,'batches':batches}
    folder=work/f'round{number:02d}.dispatch';metadata=folder/'launch.json'
    try:folder.mkdir()
    except FileExistsError:
        if not metadata.is_file():raise RuntimeError('Incomplete dispatch claim; inspect before recovery')
        m=json.loads(metadata.read_text())
        if any(m[k]!=v for k,v in identity.items()):raise ValueError('Existing dispatch has a different immutable identity')
        if not (work/f'round{number:02d}.exit').exists():
            try:os.kill(m['pid'],0)
            except ProcessLookupError:raise RuntimeError('Dispatch process vanished without exit evidence')
        return m
    if (work/f'round{number:02d}.exit').exists():raise RuntimeError('Existing legacy execution must be adopted explicitly, never rerun')
    commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    env={**os.environ,'EXPERIMENT_ROOT':str(work)}
    with (work/f'round{number:02d}.log').open('wb') as log:
        child=subprocess.Popen(['bash',str(repo/'scripts/scnet_coordinate_round.sh'),str(number),campaign,
            str(program),str(reference),previous,arms,batches],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            env=env,start_new_session=True)
    m={**identity,'pid':child.pid,'inference_commit':commit,'status':'launched'}
    temporary=folder/'launch.json.partial';temporary.write_text(json.dumps(m)+'\n')
    os.replace(temporary,metadata);return m


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('repo','work','campaign','program','reference','previous'):p.add_argument('--'+name,required=True)
    p.add_argument('--round',type=int,required=True);p.add_argument('--arms',default='gradient');p.add_argument('--batches',default='0,1')
    a=p.parse_args();print(json.dumps(dispatch(a.repo,a.work,a.round,a.campaign,a.program,a.reference,a.previous,a.arms,a.batches)))
