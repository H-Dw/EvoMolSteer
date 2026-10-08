"""Guarded dispatch for one immutable twenty-round selection-path experiment."""
import argparse,hashlib,json,os,subprocess,shutil
from pathlib import Path
from dispatch_path_round import verify_retention
from retire_experiment_outputs import retire


def dispatch(repo,work,program,reference,number,previous=None,batches='41,42',arms='gradient',n=100,retry_failed=False):
    repo,work,program,reference=map(lambda p:Path(p).resolve(),[repo,work,program,reference])
    p=json.loads(program.read_text());campaign=f'selection_path_r{number:02d}'
    if not 1<=number<=20 or p['round']!=number or p['seed']!=42 or n<50 or n%50:
        raise ValueError('Fresh twenty-round seed/population contract')
    batch_ids=list(map(int,batches.split(',')))
    if len(batch_ids)!=n//50 or len(set(batch_ids))!=len(batch_ids) or min(batch_ids)<0:raise ValueError('Batch contract')
    if program.parent!=repo/'configs/experiments/selection_path20_v1' or not reference.is_relative_to(repo/'configs'):
        raise ValueError('Committed configuration required')
    if hashlib.sha256(reference.read_bytes()).hexdigest()!=p['reference_sha256']:raise ValueError('Reference checksum')
    if subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--untracked-files=no'],text=True).strip():
        raise ValueError('Dirty remote source')
    for f in [program,reference]:subprocess.check_call(['git','-C',str(repo),'ls-files','--error-unmatch',str(f.relative_to(repo))],stdout=subprocess.DEVNULL)
    work.mkdir(parents=True,exist_ok=True);active=work/'active.json'
    old=json.loads(active.read_text()) if active.exists() else None
    if old:
        if number!=old['round']+1 and not (retry_failed and number==old['round']):raise ValueError('Serial round required')
        exit_file=work/f'round{old["round"]:02d}.exit'
        if not exit_file.exists():raise ValueError('Prior inference active')
        if retry_failed:
            if number!=old['round'] or int(exit_file.read_text())==0:raise ValueError('Only failed execution retry')
            attempt=len(list(work.glob(f'round{number:02d}.failed*.log')))+1
            (work/f'round{number:02d}.log').rename(work/f'round{number:02d}.failed{attempt}.log')
            exit_file.rename(work/f'round{number:02d}.failed{attempt}.exit')
            report=repo/'docs/experiments/selection_path20_20261008/protocol.json'
        else:
            if not previous:raise ValueError('Retained previous result required')
            report=Path(previous).resolve();m=verify_retention(report)
            if m['campaign']!=old['campaign']:raise ValueError('Prior report mismatch')
    else:
        if number!=1:raise ValueError('New campaign begins at one')
        report=repo/'docs/experiments/selection_path20_20261008/protocol.json'
    targets=[str(work/'generated')]
    if old:targets.extend(str(work/(old['campaign']+s)) for s in ['.tar.gz','.tar.gz.json'])
    plan={'allowed_bases':[str(work)],'targets':targets,'result_report':str(report),
        'protected':[str(repo),'/root/private_data/MolSteer/flowr_root/experiments/ck2_clk3_lineage_20261003',
            '/root/private_data/MolSteer/flowr_root/checkpoints']}
    plan_file=work/f'cleanup_before_round{number:02d}.plan.json';plan_file.write_text(json.dumps(plan))
    audit=work/f'cleanup_before_round{number:02d}.json';retire(plan_file,audit,True)
    available=shutil.disk_usage(work).free
    if available<700*1024**2:raise ValueError('Insufficient space after report-verified cleanup; no generation launched')
    ready={'status':'active','round':number,'maximum_rounds':20,'master_seed':42,'campaigns':[campaign],
        'cleanup_audit':str(audit),'generated_root':str(work/'generated'),'free_bytes_before_inference':available,'protected_inputs':plan['protected'][1:]}
    (work/'round_ready.json').write_text(json.dumps(ready))
    flowr=repo.parent/'flowr_root'
    if not (flowr/'flowr/models/fm_pocket.py').is_file():raise ValueError('FLOWR root missing')
    commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    with (work/f'round{number:02d}.log').open('wb') as log:
        child=subprocess.Popen(['bash',str(repo/'scripts/scnet_inference_round.sh'),str(number),campaign,str(program),str(reference),arms,str(n),batches],
            stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,
            env={**os.environ,'FLOWR_ROOT':str(flowr),'EXPERIMENT_ROOT':str(work)})
    result={'round':number,'campaign':campaign,'pid':child.pid,'status':'launched','inference_commit':commit,'n':n,'batches':batch_ids,'arms':arms}
    active.write_text(json.dumps(result));(work/f'round{number:02d}.launch.json').write_text(json.dumps(result));return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['repo','work','program','reference']:p.add_argument('--'+name,required=True)
    p.add_argument('--round',required=True,type=int);p.add_argument('--previous');p.add_argument('--batches',default='41,42')
    p.add_argument('--arms',default='gradient');p.add_argument('--n',type=int,default=100);p.add_argument('--retry-failed',action='store_true')
    a=p.parse_args();print(json.dumps(dispatch(a.repo,a.work,a.program,a.reference,a.round,a.previous,a.batches,a.arms,a.n,a.retry_failed)))
