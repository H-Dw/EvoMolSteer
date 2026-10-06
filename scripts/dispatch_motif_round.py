"""Atomic identity and process launch for the independently budgeted campaign."""
import argparse,hashlib,json,os,subprocess
from pathlib import Path


def dispatch(repo,work,number,campaign,program,reference,previous,arms='gradient',batches='0,1',n=100):
    repo,work,program,reference=map(lambda v:Path(v).resolve(),(repo,work,program,reference))
    if not 1<=number<=15 or campaign!=f'coordinate_r{number:02d}_motif15' or n not in (50,100):raise ValueError('New fifteen-round identity contract')
    if not program.is_relative_to(repo/'configs/experiments/ck2_motif_seed42_v1') or not reference.is_relative_to(program.parent):raise ValueError('Committed motif program/reference required')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    identity={'round':number,'campaign':campaign,'program_sha256':sha(program),'reference_sha256':sha(reference),'previous_report':previous,'arms':arms,'batches':batches,'n_per_arm':n}
    folder=work/f'round{number:02d}.dispatch';metadata=folder/'launch.json'
    try:folder.mkdir()
    except FileExistsError:
        if not metadata.is_file():raise RuntimeError('Incomplete launch identity; inspect before retry')
        m=json.loads(metadata.read_text())
        if any(m[k]!=v for k,v in identity.items()):raise ValueError('Existing launch identity mismatch')
        if not (work/f'round{number:02d}.exit').exists():
            try:os.kill(m['pid'],0)
            except ProcessLookupError:raise RuntimeError('Process vanished without exit evidence')
        return m
    if (work/f'round{number:02d}.exit').exists():raise RuntimeError('Existing execution is immutable')
    commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    with (work/f'round{number:02d}.log').open('wb') as log:
        child=subprocess.Popen(['bash',str(repo/'scripts/scnet_motif_round.sh'),str(number),campaign,str(program),str(reference),previous,arms,batches,str(n)],
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'EXPERIMENT_ROOT':str(work)},start_new_session=True)
    result={**identity,'pid':child.pid,'inference_commit':commit,'status':'launched'}
    temp=folder/'launch.json.partial';temp.write_text(json.dumps(result)+'\n');os.replace(temp,metadata);return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('repo','work','campaign','program','reference','previous'):p.add_argument('--'+k,required=True)
    p.add_argument('--round',type=int,required=True);p.add_argument('--arms',default='gradient');p.add_argument('--batches',default='0,1');p.add_argument('--n',type=int,default=100)
    a=p.parse_args();print(json.dumps(dispatch(a.repo,a.work,a.round,a.campaign,a.program,a.reference,a.previous,a.arms,a.batches,a.n)))
