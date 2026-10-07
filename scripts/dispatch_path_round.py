"""Sequential, report-gated inference launch; one immutable experiment at a time."""
import argparse,json,os,subprocess,hashlib
from pathlib import Path
from retire_experiment_outputs import retire

def verify_retention(path):
    path=Path(path).resolve();m=json.loads(path.read_text())
    if m.get('status')!='complete' or not m.get('files'):raise ValueError('Complete retained reports required')
    for item in m['files']:
        p=(path.parent/item['path']).resolve()
        if not p.is_relative_to(path.parent) or not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']:
            raise ValueError('Retained report checksum mismatch')
    return m

def dispatch(repo,work,program,reference,number,previous=None,batches='30,31',arms='gradient',n=100):
    repo,work,program,reference=map(lambda x:Path(x).resolve(),(repo,work,program,reference))
    p=json.loads(program.read_text());campaign=f'elite_path_r{number:02d}'
    if not 4<=number<=20 or p['round']!=number or p['seed']!=42 or n%50 or n<50:
        raise ValueError('Round, seed or population contract')
    batch_ids=[int(v) for v in batches.split(',')]
    if len(batch_ids)!=n//50 or len(set(batch_ids))!=len(batch_ids) or min(batch_ids)<0:raise ValueError('Batch contract')
    if program.parent!=repo/'configs/experiments/elite_path20_v1' or not reference.is_relative_to(repo/'configs'):
        raise ValueError('Committed configuration required')
    if hashlib.sha256(reference.read_bytes()).hexdigest()!=p['reference_sha256']:raise ValueError('Reference checksum mismatch')
    if subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--untracked-files=no'],text=True).strip():
        raise ValueError('Dirty source checkout')
    for file in [program,reference]:subprocess.check_call(['git','-C',str(repo),'ls-files','--error-unmatch',str(file.relative_to(repo))],stdout=subprocess.DEVNULL)
    work.mkdir(parents=True,exist_ok=True)
    active=work/'active.json'
    old=json.loads(active.read_text()) if active.exists() else None
    if old:
        if number!=old['round']+1:raise ValueError('Sequential round required')
        exit_file=work/f'round{old["round"]:02d}.exit'
        if not exit_file.is_file():raise ValueError('Previous inference still active')
        if not previous:raise ValueError('Previous report missing')
        report=Path(previous).resolve();m=verify_retention(report)
        if m['campaign']!=old['campaign']:raise ValueError('Wrong previous report')
    else:
        if number!=4:raise ValueError('Initial inference must be round 4')
        report=repo/'docs/experiments/elite_path20_20261007/round03_library_manifest.json'
    targets=[str(work/'generated')]
    if old:targets.extend(str(work/(old['campaign']+s)) for s in ['.tar.gz','.tar.gz.json'])
    plan={'allowed_bases':[str(work)],'protected':[str(repo),'/root/private_data/MolSteer/flowr_root/experiments/ck2_clk3_lineage_20261003',
        '/root/private_data/MolSteer/flowr_root/checkpoints'],'result_report':str(report),'targets':targets}
    plan_file=work/f'cleanup_before_round{number:02d}.plan.json';plan_file.write_text(json.dumps(plan))
    audit=work/f'cleanup_before_round{number:02d}.json';retire(plan_file,audit,True)
    ready={'status':'active','round':number,'maximum_rounds':20,'master_seed':42,'campaigns':[campaign],
        'cleanup_audit':str(audit),'generated_root':str(work/'generated'),
        'protected_inputs':plan['protected'][1:]}
    (work/'round_ready.json').write_text(json.dumps(ready))
    commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    with (work/f'round{number:02d}.log').open('wb') as log:
        child=subprocess.Popen(['bash',str(repo/'scripts/scnet_path_round.sh'),str(number),campaign,str(program),str(reference),arms,str(n),batches],
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'EXPERIMENT_ROOT':str(work)})
    result={'round':number,'campaign':campaign,'pid':child.pid,'inference_commit':commit,'arms':arms,'batches':batch_ids,'n':n,'status':'launched'}
    active.write_text(json.dumps(result));(work/f'round{number:02d}.launch.json').write_text(json.dumps(result))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['repo','work','program','reference']:p.add_argument('--'+key,required=True)
    p.add_argument('--round',type=int,required=True);p.add_argument('--previous')
    p.add_argument('--batches',default='30,31');p.add_argument('--arms',default='gradient');p.add_argument('--n',type=int,default=100)
    a=p.parse_args();print(json.dumps(dispatch(a.repo,a.work,a.program,a.reference,a.round,a.previous,a.batches,a.arms,a.n)))
