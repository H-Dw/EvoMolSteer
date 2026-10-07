"""Paired inference for literal Skill designs; analysis remains local.

Connection and repository settings are execution arguments, never Skill text.
Each job is committed before dispatch and retired after retained report checks.
"""
import argparse
import copy
import json
import shlex
import subprocess
from pathlib import Path
from resume_motif_campaign import MotifDriver
from evomolsteer.io import read_json,write_json,digest
from evomolsteer.continuous.skill_ablation import effective_signature

CONFIG='configs/experiments/skill_ablation_v1'
EVIDENCE='docs/experiments/skill_ablation_20261007/generation'
ANCHOR='docs/experiments/ck2_affinity_geometry30_20261007/round_30/comparison/comparison.json'


class SkillDriver(MotifDriver):
    def __init__(self,args):
        args.config_root=CONFIG;args.evidence_root=EVIDENCE
        args.state_root='test/skill_ablation_driver';args.output_prefix='skill_ablation';args.record_response=True
        super().__init__(args)
        self.bootstrap()

    def bootstrap(self):
        q=shlex.quote;work=self.args.remote_work
        if not work.endswith('/experiments/evomolsteer_skill_ablation_20261007'):
            raise ValueError('Explicit isolated ablation workspace required')
        self.remote('mkdir -p '+q(work))
        state=dict(round=0,maximum_rounds=30,master_seed=42,status='ready',campaigns=[],
            generated_root=work+'/generated',protected_inputs=[self.args.remote_repo,
            '/root/private_data/MolSteer/flowr_root/experiments/ck2_clk3_lineage_20261003',
            '/root/private_data/MolSteer/flowr_root/checkpoints/flowr_root_v2.ckpt'])
        with self.ssh.open_sftp() as sftp:
            try:sftp.stat(work+'/round_ready.json')
            except FileNotFoundError:
                with sftp.file(work+'/round_ready.json','w') as f:f.write(json.dumps(state))

    def push(self,number):
        self.command(number,'git_add',['git','add','--',str(self.cfg),str(self.evidence)])
        if subprocess.run(['git','diff','--cached','--quiet'],cwd=self.root).returncode:
            self.command(number,'git_commit',['git','commit','-m',f'Retain literal Skill ablation generation job {number}'])
        self.command(number,'github_push',['git','-c','http.proxy='+self.args.local_proxy,'push','origin','main'])

    def freeze(self,job):
        c=read_json(self.campaign);number=len(c['rounds'])+1
        if number!=c['rounds_completed']+1:raise ValueError('Previous job incomplete')
        source=self.root/job['program_path'];p=copy.deepcopy(read_json(source))
        signature,_=effective_signature(p)
        if signature!=job['effective_signature']:raise ValueError('Literal program changed')
        p.update(round=number,program_id=f'skill_ablation_job{number:02d}',seed=c['master_seed'])
        if 'unguided' in job['arms']:p.pop('head_native_labels_relative_path',None)
        else:p['head_native_labels_relative_path']=EVIDENCE+'/round_01/local/head_scores_window.parquet'
        path=self.cfg/f'backtrack_round{number:02d}.json';write_json(path,p)
        name=f'coordinate_r{number:02d}_skill_ablation'
        row={'round':number,'campaign':name,'status':'frozen','arms':job['arms'],'batches':job['batches'],
            'n_per_arm':100,'seed':c['master_seed'],'parent_round':0,'split':job['split'],
            'reason':job['label'],'reference_sha256':p['reference_sha256'],'effective_signature':signature,
            'literal_program_path':job['program_path'],'job_id':job['job_id']}
        c['rounds'].append(row);write_json(self.campaign,c)
        write_json(self.evidence/f'round_{number:02d}.plan.json',row)

    def launch_frozen(self,r):
        number=r['round'];self.push(number)
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        p=self.cfg/f'backtrack_round{number:02d}.json'
        ref=next(v for v in self.cfg.glob('*reference.json.gz') if digest(v)==r['reference_sha256'])
        previous=ANCHOR if number==1 else f'{EVIDENCE}/round_{number-1:02d}/comparison/comparison.json'
        q=shlex.quote;repo=self.args.remote_repo;work=self.args.remote_work
        args=['--repo',repo,'--work',work,'--round',str(number),'--campaign',r['campaign'],
            '--profile','skill_ablation','--program',repo+'/'+p.relative_to(self.root).as_posix(),
            '--reference',repo+'/'+ref.relative_to(self.root).as_posix(),'--previous',previous,
            '--arms',','.join(r['arms']),'--batches',','.join(map(str,r['batches'])),'--n','100']
        command=f'set -eu\ngit -C {q(repo)} -c http.proxy={q(self.args.remote_proxy)} pull --ff-only --quiet\ntest "$(git -C {q(repo)} rev-parse HEAD)" = {q(commit)}\n'
        command+=f'/opt/miniforge3/envs/molsteer-flowr-dtk/bin/python {q(repo+"/scripts/dispatch_motif_round.py")} '+' '.join(q(v) for v in args)
        launched=json.loads(self.remote(command));c=read_json(self.campaign)
        c['rounds'][-1].update(status='running',inference_commit=launched['inference_commit'])
        c['rounds_started']=number;write_json(self.campaign,c);self.event('ablation_inference_launched',round=number,pid=launched['pid'])

    def retire_remote(self,r):
        self.push(r['round']);q=shlex.quote;repo=self.args.remote_repo;work=self.args.remote_work
        self.remote(f'git -C {q(repo)} -c http.proxy={q(self.args.remote_proxy)} pull --ff-only --quiet')
        report=repo+f'/{EVIDENCE}/round_{r["round"]:02d}/comparison/comparison.json'
        plan={'allowed_bases':[work],'protected':[repo,'/root/private_data/MolSteer/flowr_root/experiments/ck2_clk3_lineage_20261003',
            '/root/private_data/MolSteer/flowr_root/checkpoints'],'result_report':report,
            'targets':[work+'/generated',work+'/'+r['campaign']+'.tar.gz',work+'/'+r['campaign']+'.tar.gz.json']}
        path=self.state/f'retire_remote{r["round"]}.json';write_json(path,plan)
        with self.ssh.open_sftp() as sftp:sftp.put(str(path),work+'/retire_after.plan.json')
        audit=work+f'/cleanup_after_round{r["round"]:02d}.json'
        self.remote(f'if test ! -f {q(audit)}; then cd {q(repo)}; PYTHONPATH={q(repo+"/src")} /opt/miniforge3/envs/molsteer-flowr-dtk/bin/python scripts/retire_experiment_outputs.py --plan {q(work+"/retire_after.plan.json")} --report {q(audit)} --apply; fi')
        with self.ssh.open_sftp() as sftp:sftp.get(audit,str(self.evidence/f'cleanup_after_round{r["round"]:02d}.remote.json'))
        self.push(r['round'])

    def run(self):
        while True:
            c=read_json(self.campaign);jobs=read_json(self.cfg/'jobs.json')['jobs']
            if c['rounds'] and c['rounds'][-1]['status']=='running':
                self.finish(c['rounds'][-1]);self.retire_remote(c['rounds'][-1]);continue
            if c['rounds'] and c['rounds'][-1]['status']=='frozen':self.launch_frozen(c['rounds'][-1]);continue
            if c['rounds'] and c['rounds'][-1]['status']=='completed' and not (self.evidence/f'cleanup_after_round{c["rounds"][-1]["round"]:02d}.remote.json').exists():
                self.retire_remote(c['rounds'][-1]);continue
            number=c['rounds_completed']
            if number>=min(len(jobs),self.args.through):
                self.event('ablation_jobs_complete',completed=number);return
            self.freeze(jobs[number])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',default=str(Path(__file__).resolve().parents[1]))
    p.add_argument('--host',default='ksai.scnet.cn');p.add_argument('--port',type=int,default=10544);p.add_argument('--user',default='root')
    p.add_argument('--remote-repo',default='/root/private_data/MolSteer/EvoMolSteer')
    p.add_argument('--remote-work',default='/root/private_data/MolSteer/flowr_root/experiments/evomolsteer_skill_ablation_20261007')
    p.add_argument('--local-proxy',default='http://127.0.0.1:7897');p.add_argument('--remote-proxy',default='http://127.0.0.1:17897')
    p.add_argument('--through',type=int,default=30);a=p.parse_args();driver=SkillDriver(a)
    try:driver.run()
    finally:driver.ssh.close()
