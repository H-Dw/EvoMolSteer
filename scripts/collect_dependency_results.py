import argparse,json,os,subprocess,time
from pathlib import Path
import paramiko
from evomolsteer.io import read_json,write_json

def run(repo,workspace,names,seconds):
    """Bounded collection for this single active study, never remote analysis."""
    study=repo/'docs/experiments/steer_dependency_20261009/luna_suite'
    transport=repo/'test/data/steer_dependency/luna'
    transport.mkdir(parents=True,exist_ok=True)
    deadline=time.monotonic()+seconds
    evaluated=set()
    while time.monotonic()<deadline:
        client=paramiko.SSHClient();client.load_system_host_keys();client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect('ksai.scnet.cn',port=10544,username='root',password=os.environ['MOLSTEER_SCNET_PASSWORD'],look_for_keys=False,allow_agent=False,timeout=25)
        try:
            with client.open_sftp() as sftp:
                for name in names:
                    if name in evaluated:continue
                    try:
                        with sftp.open(workspace+'/'+name+'.status.json') as f:status=json.load(f)
                    except FileNotFoundError:continue
                    if status['status']=='failed':raise RuntimeError(name+': '+status.get('error','failed'))
                    if status['status']!='complete':continue
                    for suffix in ['.tar.gz','.tar.gz.json']:
                        dest=transport/(name+suffix)
                        if not dest.exists():sftp.get(workspace+'/archives/'+dest.name,str(dest))
                    write_json(study/'transport_records'/(name+'.remote.json'),status)
                    output=repo/'results/steer_dependency_20261009/luna'/name
                    subprocess.run([str(repo/'.venv/Scripts/python.exe'),str(repo/'scripts/evaluate_dependency_archive.py'),'--name',name,'--transport',str(transport),'--reports',str(study/'transport_records'),'--output',str(output),'--reference',str(repo/'configs/experiments/ck2_terminal_seed42_v1/local_reference.json.gz'),'--workers','2'],check=True,cwd=repo)
                    subprocess.run([str(repo/'.venv/Scripts/python.exe'),str(repo/'scripts/analyze_dependency_reward_response.py'),'--root',str(transport/('restored'+name)/'results'/name),'--reference',str(repo/'configs/experiments/steer_dependency_v1/structure.json.gz'),'--output',str(study/'transport_records'/(name+'.reward_response.json'))],check=True,cwd=repo)
                    manifest=read_json(study/'screen_manifest.json');condition=name.removeprefix('Luna_')
                    if not any(r['name']==condition for r in manifest['cohorts']):
                        manifest['cohorts'].append({'name':condition,'arm':'gradient','metrics':output.relative_to(repo).as_posix()+'/candidate_metrics.csv'})
                        write_json(study/'screen_manifest.json',manifest)
                    subprocess.run([str(repo/'.venv/Scripts/python.exe'),str(repo/'scripts/analyze_dependency_study.py'),'--manifest',str(study/'screen_manifest.json'),'--output',str(repo/'results/steer_dependency_20261009/luna/screen')],check=True,cwd=repo)
                    evaluated.add(name)
                    print(json.dumps({'evaluated':name,'remaining':len(names)-len(evaluated)}),flush=True)
        finally:client.close()
        if len(evaluated)==len(names):return
        time.sleep(min(60,max(0,deadline-time.monotonic())))
    raise TimeoutError('Unfinished: '+str(set(names)-evaluated))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',required=True);p.add_argument('--workspace',required=True);p.add_argument('--names',required=True);p.add_argument('--watch-seconds',type=int,default=0)
    a=p.parse_args();run(Path(a.repo),a.workspace,a.names.split(','),a.watch_seconds)
