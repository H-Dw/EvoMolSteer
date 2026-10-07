"""Replay an unchanged frozen intervention to repair an evaluation failure.

This does not advance the scientific optimization counter or edit the program.
The current completed round is retired only after verified report retention.
"""
import argparse,json,os,subprocess,hashlib
from pathlib import Path
from dispatch_path_round import verify_retention
from retire_experiment_outputs import retire

def replay(repo,work,number,previous):
    repo,work,previous=map(lambda p:Path(p).resolve(),[repo,work,previous])
    active=json.loads((work/'active.json').read_text())
    if int((work/f'round{active["round"]:02d}.exit').read_text())!=0:
        raise ValueError('Current inference must complete before replay')
    m=verify_retention(previous)
    if m['campaign']!=active['campaign'] or number>=active['round']:
        raise ValueError('Replay must repair an earlier retained intervention')
    program=repo/f'configs/experiments/elite_path20_v1/round{number:02d}.json'
    p=json.loads(program.read_text())
    references=list((repo/'configs/experiments/elite_path20_v1').glob('*.json.gz'))+list((repo/'configs/experiments/skill_ablation_v1').glob('*.json.gz'))
    reference=next(v for v in references if hashlib.sha256(v.read_bytes()).hexdigest()==p['reference_sha256'])
    if p['round']!=number or p['seed']!=42:raise ValueError('Frozen replay contract')
    if subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--untracked-files=no'],text=True).strip():
        raise ValueError('Replay requires a clean committed checkout')
    target=work/f'replay_round{number:02d}'
    if target.exists():raise FileExistsError('Replay already dispatched; diagnose it rather than overwrite')
    target.mkdir()
    protected=[str(repo),str(repo.parent/'flowr_root/experiments/ck2_clk3_lineage_20261003'),str(repo.parent/'flowr_root/checkpoints')]
    plan={'allowed_bases':[str(work)],'protected':protected,'result_report':str(previous),
          'targets':[str(work/'generated')]+[str(work/(active['campaign']+s)) for s in ['.tar.gz','.tar.gz.json']]+[str(target/'generated')]}
    path=target/'cleanup.plan.json';path.write_text(json.dumps(plan));audit=target/'cleanup.json';retire(path,audit,True)
    campaign=f'elite_path_r{number:02d}'
    ready={'status':'active','round':number,'maximum_rounds':20,'master_seed':42,'campaigns':[campaign],
           'cleanup_audit':str(audit),'generated_root':str(target/'generated'),'protected_inputs':protected[1:]}
    (target/'round_ready.json').write_text(json.dumps(ready))
    commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    with (target/f'round{number:02d}.log').open('wb') as log:
        child=subprocess.Popen(['bash',str(repo/'scripts/scnet_path_round.sh'),str(number),campaign,str(program),str(reference),'gradient','50','30'],
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
            env={**os.environ,'EXPERIMENT_ROOT':str(target),'FLOWR_ROOT':str(repo.parent/'flowr_root')})
    result={'status':'replay_launched','scientific_round':number,'optimization_counter_advanced':False,
            'campaign':campaign,'pid':child.pid,'inference_commit':commit,'program_sha256':hashlib.sha256(program.read_bytes()).hexdigest(),
            'reason':'Repair missing local PoseBusters checks; fixed scientific program, checkpoint, seed and population',
            'work':str(target)}
    (target/'launch.json').write_text(json.dumps(result));return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['repo','work','previous']:p.add_argument('--'+key,required=True)
    p.add_argument('--round',type=int,required=True);a=p.parse_args()
    print(json.dumps(replay(a.repo,a.work,a.round,a.previous)))
