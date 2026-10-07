"""Sequential local proposal/evaluation and remote-only FLOWR inference.

Each next single-axis proposal is created only after the preceding report is
retained. Parameter trials are registered adaptive tests, not independent LLM
inventions. No credentials, generated structures, or network settings enter Skills.
"""
import argparse,copy,json,os,shlex,subprocess,time,shutil,sys
from pathlib import Path
import paramiko
import pandas as pd
from evomolsteer.io import read_json,digest,write_json
from evomolsteer.storage.generation_archive import verify_generation_archive
from evomolsteer.storage.remote_transfer import download_resumable
from evomolsteer.generation.path_evaluation import retain_round,summarize_tail
from evomolsteer.continuous.terminal_path_library import build_library

HYPOTHESES={
    7:'Increase coordinate dose alone to distinguish a weak displacement from an unhelpful direction. A larger dose that lowers affinity rejects this dose direction; it does not excuse the scalar construction.',
    8:'Include all available identity-bearing teachers instead of the nearest four. This tests whether hard local truncation discards a distant high-utility path; the finite archive, not 1000 fabricated teachers, defines actual support.',
    9:'Strengthen decoded terminal utility contrast at fixed coordinate dose. This changes direction toward higher labeled paths, rather than increasing controller amplitude.',
    10:'Soften teacher competition to avoid a single dominating posterior and improve coordinate transitions between supported paths.',
    11:'Keep every learned step active but shift dose toward later, less-censored geometry. The ramp is normalized by the supplied learning window, not a hardcoded molecular time.',
    12:'Blend the prior with the previous endpoint posterior propagated through exact shared path IDs. This tests temporal path coherence without selection, resampling or chemical graph restrictions.',
    13:'Test a larger continuity blend independently from the best retained kernel. Excessive path commitment could prevent discovery; remaining base mass preserves alternative supported paths.',
    15:'Narrow the coordinate kernel at fixed dose to test whether broad geometry neighborhoods dilute the decoded future utility contrast.',
    16:'Replace matched point displacement with scaled permutation-invariant coordinate geometry. This tests whether matching individual atom positions obscures a useful spatial configuration; no atom type rewards are introduced.',
    17:'Attenuate dose when local observed utility contrast is nearly flat, avoiding amplification of weak evidence by RMS normalization. This is an evidence-amplitude test, not a molecular identity gate.'}

def evaluate_isolated(root,dataset,campaign,evaluated,cfg,logfile):
    """Fresh process owns all third-party logging streams for one evaluation."""
    with Path(logfile).open('w',encoding='utf-8') as log:
        subprocess.run([sys.executable,str(Path(root)/'scripts/evaluate_terminal.py'),
            '--dataset',str(dataset),'--campaign',campaign,'--reference',str(Path(root)/'configs/experiments/ck2_terminal_seed42_v1/local_reference.json.gz'),
            '--output',str(evaluated),'--arms',cfg['arms'],'--batches',','.join(map(str,cfg['batch_indices'])),'--workers','1'],
            cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
    checked=pd.read_csv(Path(evaluated)/'candidate_metrics.csv')
    if 'pb_error' in checked and checked.pb_error.notna().any():
        raise ValueError('PoseBusters evaluation error; retain raw structures and diagnose before advancing')

class Driver:
    def __init__(self,args):
        self.args=args;self.root=Path(args.repo).resolve();self.cfg=self.root/'configs/experiments/elite_path20_v1'
        self.docs=self.root/'docs/experiments/elite_path20_20261007';self.state=self.root/'test/elite_path20_driver'
        self.state.mkdir(parents=True,exist_ok=True);self.work=args.remote_work;self.remote_repo=args.remote_repo
        self.connect()
        self.threshold=read_json(self.root/'results/elite_path20/round02_credit/manifest.json')['threshold_pic50']
        self.control=pd.read_csv(self.docs/'round04/candidate_metrics.csv')
    def connect(self):
        if hasattr(self,'ssh'):self.ssh.close()
        self.ssh=paramiko.SSHClient();self.ssh.load_system_host_keys();self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        a=self.args
        self.ssh.connect(a.host,port=a.port,username='root',password=os.environ[a.password_env],look_for_keys=False,allow_agent=False,timeout=30)
        self.ssh.get_transport().set_keepalive(30)
    def download(self,remote_path,local_path,number):
        for attempt in range(3):
            try:
                with self.ssh.open_sftp() as sftp:return download_resumable(sftp,remote_path,local_path)
            except (OSError,EOFError,paramiko.SSHException) as error:
                if attempt==2:raise
                self.emit('download_retry',round=number,attempt=attempt+1,error_type=type(error).__name__,
                    retained_partial_bytes=Path(local_path).stat().st_size if Path(local_path).exists() else 0)
                self.connect()
    def emit(self,phase,**data):
        item={'phase':phase,**data};write_json(self.state/'driver_state.json',item);print(json.dumps(item),flush=True)
    def local(self,args):
        r=subprocess.run(args,cwd=self.root,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError(r.stdout[-6000:])
        return r.stdout.strip()
    def remote(self,command):
        _,out,err=self.ssh.exec_command(command,timeout=180)
        stdout=out.read().decode();stderr=err.read().decode();code=out.channel.recv_exit_status()
        if code:raise RuntimeError(f'Remote exit {code}: {stdout[-1500:]} {stderr[-2000:]}')
        return stdout.strip()
    def push(self,message):
        self.local(['git','add','configs/experiments/elite_path20_v1','docs/experiments/elite_path20_20261007'])
        # Source modules are staged explicitly before the first driver launch.
        self.local(['git','diff','--cached','--check'])
        if self.local(['git','diff','--cached','--name-only']):self.local(['git','commit','-m',message])
        self.local(['git','push','origin','main'])
        commit=self.local(['git','rev-parse','HEAD']);q=shlex.quote
        self.remote(f'timeout 150 git -C {q(self.remote_repo)} -c http.proxy=http://127.0.0.1:17897 pull --ff-only origin main')
        actual=self.remote(f'git -C {q(self.remote_repo)} rev-parse HEAD')
        if actual!=commit:raise ValueError('Remote commit differs')
        return commit
    def metrics(self,number):
        d=pd.read_csv(self.docs/f'round{number:02d}/candidate_metrics.csv')
        return summarize_tail(d[(d.arm=='gradient')&(d.batch==30)],self.threshold)
    def rank(self,m):
        control=summarize_tail(self.control[(self.control.arm=='gradient')&(self.control.batch==30)],self.threshold)
        admissible=m['all_mean_pic50']>=control['all_mean_pic50']-.02 and m['valid_n']/m['n']>=.90
        return (int(admissible),m['elite_unique_graphs'],m['all_mean_pic50'],m['valid_p95_pic50'],-m['strain_median_per_heavy'])
    def best(self,kernel_only=False):
        numbers=[]
        for p in self.cfg.glob('round*.json'):
            n=int(p.stem[5:])
            if n>17 or not (self.docs/f'round{n:02d}/retention.json').exists():continue
            if kernel_only and read_json(p)['reward_view']!='endpoint_path_value':continue
            numbers.append(n)
        if not numbers:raise ValueError('No evaluated parent')
        return max(numbers,key=lambda n:(self.rank(self.metrics(n)),-n))
    def freeze(self,number):
        if number==6:
            parent=5;p=read_json(self.cfg/'round05.json');response=read_json(self.docs/'round06_agents/Designer.response.json')
            request=read_json(self.docs/'round06_agents/Designer.request.json')
            if response['decision']!='test_formula' or response['reward_view']!='endpoint_path_value':raise ValueError('Kernel design deferred')
            if any(response[k]!=request[k] for k in ['input_sha256','instruction_sha256']):raise ValueError('Designer binding mismatch')
            p['reward_view']='endpoint_path_value';change={'reward_view':'endpoint_path_value'};reason=response['justification']
        elif number<=17:
            parent=self.best(kernel_only=True);p=read_json(self.cfg/f'round{parent:02d}.json')
            axis={7:'native_rms_ratio',8:'teacher_neighbors',9:'teacher_score_beta',10:'mixture_temperature',
                11:'time_ramp_power',12:'path_history_mix',13:'path_history_mix',15:'teacher_endpoint_temperature_A2',
                16:'path_kernel_space',17:'path_contrast_amplitude'}
            if number==14:
                path=self.cfg/'terminal_path_budget3.json.gz'
                report=build_library(self.root/'data/raw/ck2_clk3_lineage_20261003',self.root/'results/elite_path20/round01_graph',
                    self.root/'results/elite_path20/round02_credit',self.docs.parent/'guidance_vs_steer_20261007/steer_full1000/candidate_metrics.csv',
                    self.root/'configs/experiments/skill_ablation_v1/endpoint_reference.json.gz',path,budget=3)
                write_json(self.docs/'round14_library_manifest.json',report);change={'reference_sha256':digest(path)}
                reason='Expand graph-distinct terminal path coverage, keeping potential and every dose parameter fixed; not a MAP-Elites implementation.'
            else:
                key=axis[number];old=p.get(key,0. if key=='path_history_mix' else 'pointcloud' if key=='path_kernel_space' else False if key=='path_contrast_amplitude' else 0.)
                value={7:float(p['native_rms_ratio'])*2,8:1000,9:float(p['teacher_score_beta'])*2,
                    10:float(p['mixture_temperature'])*2,11:1.,12:.5,13:.8,15:float(p['teacher_endpoint_temperature_A2'])/2,
                    16:'geometry',17:True}[number]
                if value==old:raise ValueError('Identical single-axis proposal')
                change={key:value};reason=HYPOTHESES[number]+f' All other executable fields restored from evaluated kernel parent {parent}.'
            p.update(change)
        else:
            frozen=read_json(self.docs/'frozen_validation.json') if (self.docs/'frozen_validation.json').exists() else {'winner_round':self.best()}
            if number==18:write_json(self.docs/'frozen_validation.json',frozen)
            parent=4 if number==19 else frozen['winner_round'];p=read_json(self.cfg/f'round{parent:02d}.json')
            change={};reason='Frozen independent confirmation; no reward tuning on these validation batches.'
        # Historical Skill signatures remain parent provenance, never current bindings.
        inherited={k:p.pop(k) for k in list(p) if k.endswith('_sha256') and k!='reference_sha256'}
        p.update(round=number,program_id=f'elite_path20_r{number:02d}',objective_profile='sequential_path20',
            derivation={'parent_round':parent,'single_axis_change':change,'scientific_justification':reason,
                'proposal_policy':'Registered sequential adaptive tests; not independent LLM designs per parameter trial',
                'historical_parent_provenance':inherited})
        path=self.cfg/f'round{number:02d}.json';path.write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
        reference=next((v for v in [self.cfg/'terminal_path_reference.json.gz',self.cfg/'terminal_path_budget3.json.gz',
            self.root/'configs/experiments/skill_ablation_v1/endpoint_reference.json.gz'] if v.exists() and digest(v)==p['reference_sha256']),None)
        if reference is None:raise ValueError('Reference missing')
        batches='34,35' if number==20 else '32,33' if number>=18 else '30'
        arms='unguided,gradient' if number in [18,20] else 'gradient';n=100 if number>=18 else 50
        write_json(self.docs/f'round{number:02d}_plan.json',{'round':number,'parent':parent,'change':change,'reason':reason,
            'batches':batches,'arms':arms,'n_per_arm':n,'program_sha256':digest(path),'reference_sha256':digest(reference),
            'parent_screen_metrics':self.metrics(parent) if (self.docs/f'round{parent:02d}/candidate_metrics.csv').exists() else None})
        return path,reference,batches,arms,n
    def collect(self,number):
        campaign=f'elite_path_r{number:02d}';archive=self.state/f'round{number:02d}.tar.gz'
        for suffix in ['', '.json']:self.download(self.work+'/'+campaign+'.tar.gz'+suffix,str(archive)+suffix,number)
        dataset=self.root/f'test/data/elite_path20/round{number:02d}';evaluated=self.root/f'results/elite_path20/round{number:02d}'
        verified=verify_generation_archive(archive,str(archive)+'.json',dataset)
        cfg=read_json(dataset/'results'/campaign/'config.json')['experiment']
        self.emit('local_evaluation',round=number,archive_verified=True)
        # PoseBusters/RDKit bind process-global streams. A fresh interpreter per
        # round prevents a later round from reusing a closed redirected stream.
        evaluate_isolated(self.root,dataset,campaign,evaluated,cfg,self.state/f'round{number:02d}.evaluation.log')
        baseline=self.docs/'round04/candidate_metrics.csv' if number<=17 else self.docs/'round18/candidate_metrics.csv' if number==19 else None
        result=retain_round(dataset,campaign,evaluated,self.docs/f'round{number:02d}',self.threshold,baseline)
        write_json(self.docs/f'round{number:02d}/archive_verification.json',verified)
        # Verification report is added to the retention checksum list.
        retention=read_json(self.docs/f'round{number:02d}/retention.json')
        p=self.docs/f'round{number:02d}/archive_verification.json'
        retention['files'].append({'path':p.name,'sha256':digest(p),'bytes':p.stat().st_size});write_json(p.parent/'retention.json',retention)
        c=read_json(self.cfg/'campaign.json')
        if c['rounds_completed']!=number-1:raise ValueError('Local sequence changed')
        c['rounds_completed']=number;c['rounds'].append({'round':number,'kind':'inference','status':'complete',
            'module':read_json(self.docs/f'round{number:02d}_plan.json')['change'],'results':result['results'],
            'retention_report':f'docs/experiments/elite_path20_20261007/round{number:02d}/retention.json'})
        (self.cfg/'campaign.json').write_text(json.dumps(c,indent=2)+'\n',encoding='utf-8',newline='\n')
        self.emit('round_complete',round=number,summary=result)
        self.push(f'Retain sequential path round {number} and its paired terminal evidence')
        self.retire_local(dataset,archive)
    def retire_local(self,dataset,archive):
        p=Path(dataset).resolve();base=(self.root/'test/data/elite_path20').resolve()
        if not p.is_relative_to(base) or p==base:raise ValueError('Unsafe local retirement')
        shutil.rmtree(p)
        for f in [Path(archive),Path(str(archive)+'.json')]:
            if not f.resolve().is_relative_to(self.state.resolve()):raise ValueError('Unsafe archive retirement')
            f.unlink()
    def run(self):
        for number in range(self.args.start,self.args.last+1):
            if read_json(self.cfg/'campaign.json')['rounds_completed']!=number-1:raise ValueError('Sequential completed modules required')
            path,reference,batches,arms,n=self.freeze(number)
            commit=self.push(f'Freeze single-module path exploration round {number}')
            q=shlex.quote;previous=self.remote_repo+f'/docs/experiments/elite_path20_20261007/round{number-1:02d}/retention.json'
            command=' '.join(q(v) for v in ['/opt/miniforge3/envs/molsteer-flowr-dtk/bin/python',self.remote_repo+'/scripts/dispatch_path_round.py',
                '--repo',self.remote_repo,'--work',self.work,'--round',str(number),'--program',self.remote_repo+'/'+path.relative_to(self.root).as_posix(),
                '--reference',self.remote_repo+'/'+reference.relative_to(self.root).as_posix(),'--previous',previous,'--batches',batches,'--arms',arms,'--n',str(n)])
            self.remote('source /opt/MolSteer/scripts/scnet/activate_dtk.sh\n'+command)
            self.emit('inference_started',round=number,commit=commit,batches=batches,arms=arms)
            while True:
                status=self.remote(f'if test -f {q(self.work+f"/round{number:02d}.exit")}; then cat {q(self.work+f"/round{number:02d}.exit")}; else echo running; fi')
                if status!='running':
                    if status!='0':raise RuntimeError('Inference failed; retain logs and diagnose before retry: '+self.remote(f'tail -n 25 {q(self.work+f"/round{number:02d}.log")}'))
                    break
                time.sleep(30)
            self.collect(number)
        self.emit('requested_sequence_complete',last=self.args.last)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',default='.');p.add_argument('--host',default='ksai.scnet.cn');p.add_argument('--port',type=int,default=10544)
    p.add_argument('--password-env',default='MOLSTEER_SCNET_PASSWORD');p.add_argument('--start',type=int,default=6);p.add_argument('--last',type=int,default=20)
    p.add_argument('--collect-only',type=int,help='Resume local collection of an already completed remote round; do not start inference')
    p.add_argument('--remote-repo',default='/root/private_data/MolSteer/EvoMolSteer')
    p.add_argument('--remote-work',default='/root/private_data/MolSteer/flowr_root/experiments/evomolsteer_elite_path20_20261007')
    a=p.parse_args()
    if not 6<=a.start<=a.last<=20:raise ValueError('Twenty-round budget')
    d=Driver(a)
    try:
        if a.collect_only is not None:
            if not 6<=a.collect_only<=20 or read_json(d.cfg/'campaign.json')['rounds_completed']!=a.collect_only-1:
                raise ValueError('Collection must resume the next unfinished round')
            if d.remote(f'cat {shlex.quote(d.work+f"/round{a.collect_only:02d}.exit")}')!='0':
                raise ValueError('Remote inference not complete')
            d.collect(a.collect_only)
        else:d.run()
    finally:d.ssh.close()
