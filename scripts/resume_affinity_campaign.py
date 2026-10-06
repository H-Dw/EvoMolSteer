"""New bounded30 affinity-primary campaign; local evidence, remote inference only."""
import argparse,copy,json,shlex,subprocess
from pathlib import Path
from resume_motif_campaign import MotifDriver
from evomolsteer.io import read_json,digest,write_json
from evomolsteer.generation.affinity_campaign import propose,update_program,select_parent
from evomolsteer.continuous.affinity_skill import classify,audit_behavior
from evomolsteer.generation.endpoint_campaign import propose_endpoint
from evomolsteer.continuous.endpoint_design import audit_endpoint_behavior
from evomolsteer.continuous.sparse_design import audit_sparse_behavior
from evomolsteer.generation.sparse_campaign import propose_sparse,apply_sparse_fields

CONFIG='configs/experiments/ck2_affinity_geometry30_v1'
EVIDENCE='docs/experiments/ck2_affinity_geometry30_20261007'
SKILL='skills/affinity-coordinate-search/SKILL.md'
ENDPOINT_SKILL='skills/affinity-endpoint-pullback/SKILL.md'
SPARSE_SKILL='skills/affinity-sparse-coordinate/SKILL.md'
SPARSE_REVIEW='sparse_trace_review.json'
OLD_REPORT='docs/experiments/ck2_motif_seed42_20261006/round_15/comparison/comparison.json'

class AffinityDriver(MotifDriver):
    def __init__(self,args):
        args.config_root=CONFIG;args.evidence_root=EVIDENCE;args.state_root='test/affinity_campaign_driver'
        args.output_prefix='affinity';args.record_response=True
        super().__init__(args)
        self.check_skill();self.bootstrap_remote()
    def check_skill(self):
        if self.args.sparse_profile:
            folder=self.evidence/'sparse_skill_test_v2'
            audit_sparse_behavior(self.root/SPARSE_SKILL,folder/'input.json',folder/'prompt.txt',folder/'response.json',
                                  self.evidence/'endpoint_mining/sparse_coordinate_prior_v2.json')
            review=read_json(self.evidence/'agent_review'/SPARSE_REVIEW)
            if review.get('execution_ready') is not True:raise ValueError('Sparse Agent integration review is not ready for real runtime validation')
            for path,expected in review['source_hashes'].items():
                if digest(self.root/path)!=expected:raise ValueError('Sparse Agent review does not cover current source: '+path)
            return
        if self.args.endpoint_profile:
            folder=self.evidence/'endpoint_skill_test'
            audit_endpoint_behavior(self.root/ENDPOINT_SKILL,folder/'input.json',folder/'prompt.txt',
                read_json(folder/'response.json'),digest(self.cfg/'endpoint_reference.json.gz'))
            review=read_json(self.evidence/'agent_review/endpoint_design_review.json')
            if review.get('execution_ready') is not True:raise ValueError('Endpoint Agent integration review is not ready for runtime validation')
            for path,expected in review['source_hashes'].items():
                if digest(self.root/path)!=expected:raise ValueError('Endpoint Agent review does not cover current source: '+path)
            return
        packet=self.evidence/'skill_test/input.json';response=read_json(self.evidence/'skill_test/response.json')
        audit_behavior(self.root/SKILL,packet,response)
        review=read_json(self.evidence/'agent_review/geometry_design_review.json')
        if review.get('execution_ready') is not True:raise ValueError('Geometry Agent review has unresolved execution defects')
        for path,expected in review['source_hashes'].items():
            if digest(self.root/path)!=expected:raise ValueError('Agent review does not cover current geometry code')
    def bootstrap_remote(self):
        repo,work=self.args.remote_repo,self.args.remote_work;q=shlex.quote
        prefix='/root/private_data/MolSteer/flowr_root/experiments/'
        if work!=prefix+'evomolsteer_affinity_geometry30_20261007':raise ValueError('Explicit new experiment directory required')
        self.remote('mkdir -p '+q(work))
        state={'round':0,'maximum_rounds':30,'master_seed':42,'status':'ready','campaigns':[],'generated_root':work+'/generated',
               'protected_inputs':['/root/private_data/MolSteer/flowr_root/experiments/ck2_clk3_lineage_20261003/inputs',
                                   '/root/private_data/MolSteer/flowr_root/checkpoints/flowr_root_v2.ckpt']}
        with self.ssh.open_sftp() as sftp:
            try:sftp.stat(work+'/round_ready.json')
            except FileNotFoundError:
                with sftp.open(work+'/round_ready.json','w') as f:f.write(json.dumps(state))
    def freeze(self,number):
        self.check_skill();c=read_json(self.campaign)
        if number!=c['rounds_completed']+1 or c['rounds_started']!=c['rounds_completed'] or not 1<=number<=30:
            raise ValueError('New sequential30 budget')
        rows=[read_json(p) for p in sorted(self.evidence.glob('round_*.outcome.json'))]
        programs={r['round']:read_json(self.cfg/f"backtrack_round{r['round']:02d}.json") for r in rows}
        plan=(propose_sparse(number,rows,programs) if self.args.sparse_profile else
              propose_endpoint(number,rows,programs) if self.args.endpoint_profile else propose(number,rows,programs));parent=plan['parent_round']
        if self.args.sparse_profile and number==6:
            base=read_json(self.evidence/'sparse_skill_test_v2/compiled_design.json')
        elif self.args.endpoint_profile and number==5:
            base=read_json(self.evidence/'endpoint_skill_test/compiled_design.json')
        else:
            template=plan.get('template_round',parent)
            base=programs[template] if template else read_json(self.root/'configs/experiments/ck2_motif_seed42_v1/backtrack_round09.json')
        family=plan.get('reward_view',base['reward_view'])
        reference=self.cfg/('endpoint_reference.json.gz' if family.startswith('endpoint_') else
                            'survival_reference.json.gz' if family=='motif_mixture' else 'geometry_reference.json.gz')
        from evomolsteer.generation.window_reference import load_reference
        ref=load_reference(reference);p=update_program(base,plan,ref['window'],digest(reference))
        if self.args.sparse_profile:p=apply_sparse_fields(p,plan,read_json(self.evidence/'endpoint_mining/sparse_coordinate_prior_v2.json'))
        for k in ('compiled_design_sha256','designer_sha256','agent_review_sha256','boundary_agent_review_sha256','derivation'):p.pop(k,None)
        skill=SPARSE_SKILL if self.args.sparse_profile else ENDPOINT_SKILL if self.args.endpoint_profile else SKILL
        audit_folder='sparse_skill_test_v2' if self.args.sparse_profile else 'endpoint_skill_test' if self.args.endpoint_profile else 'skill_test'
        review_name=SPARSE_REVIEW if self.args.sparse_profile else 'endpoint_design_review.json' if self.args.endpoint_profile else 'geometry_design_review.json'
        p.update(round=number,program_id=f'ck2_affinity_geometry30_round{number:02d}',seed=42,
            skill_sha256=digest(self.root/skill),skill_behavior_audit_sha256=digest(self.evidence/audit_folder/'behavior_audit.json'),
            geometry_review_sha256=digest(self.evidence/'agent_review'/review_name),
            derivation={'plan':plan,'source':'Existing Steer coordinate library, recorded scores, Agent-vetted registered adaptive policy',
                        'no_private_reasoning_transcript':True})
        if self.args.endpoint_profile or self.args.sparse_profile:
            p['endpoint_designer_response_sha256']=digest(self.evidence/audit_folder/'response.json')
            p['endpoint_compiled_baseline_sha256']=digest(self.evidence/audit_folder/'compiled_design.json')
        if 7<=number<=26:p['head_native_labels_relative_path']=EVIDENCE+'/round_06/local/head_scores_window.parquet'
        else:p.pop('head_native_labels_relative_path',None)
        arms=['gradient'];batches=[0,1];split='discovery'
        if number==1 or (self.args.endpoint_profile and number==5) or (self.args.sparse_profile and number==6):arms=['unguided','gradient_zero','gradient']
        if number>=27:
            arms=['unguided','gradient'];batches=[20+2*(number-27),21+2*(number-27)];split='validation' if number==27 else 'heldout'
            if number==27:arms.insert(1,'gradient_zero')
        name=f'coordinate_r{number:02d}_affinity30';write_json(self.cfg/f'backtrack_round{number:02d}.json',p)
        write_json(self.evidence/f'round_{number:02d}.plan.json',{'round':number,'parent_round':parent,'split':split,'plan':plan,
            'program_sha256':digest(self.cfg/f'backtrack_round{number:02d}.json'),'reference_sha256':digest(reference),
            'previous_outcome_sha256':digest(self.evidence/f'round_{number-1:02d}.outcome.json') if number>1 else None,
            'skill_sha256':p['skill_sha256'],'skill_response_verified':True,
            'decision_basis':'Agent-verified affinity-primary policy. Registered adaptive code chooses program interventions; not thirty independent LLM inventions.'})
        c['rounds'].append({'round':number,'campaign':name,'parent_round':parent,'reason':plan['reason'],'status':'frozen',
                           'arms':arms,'batches':batches,'n_per_arm':100,'seed':42,'split':split,'reference_sha256':digest(reference)})
        write_json(self.campaign,c)
    def push(self,number):
        self.command(number,'git_add',['git','add','--',str(self.cfg),str(self.evidence)])
        if subprocess.run(['git','diff','--cached','--quiet'],cwd=self.root).returncode:
            self.command(number,'git_commit',['git','commit','-m',f'Preserve affinity evidence and freeze new30 round {number}'])
        self.command(number,'github_push',['git','-c','http.proxy='+self.args.local_proxy,'push','origin','main'])
    def launch_frozen(self,r):
        number=r['round'];self.push(number);commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        p=self.cfg/f'backtrack_round{number:02d}.json';reference=next(v for v in self.cfg.glob('*reference.json.gz') if digest(v)==r['reference_sha256'])
        previous=OLD_REPORT if number==1 else f'{EVIDENCE}/round_{number-1:02d}/comparison/comparison.json'
        repo,work=self.args.remote_repo,self.args.remote_work;q=shlex.quote
        args=['--repo',repo,'--work',work,'--round',str(number),'--campaign',r['campaign'],'--profile','affinity30',
              '--program',repo+'/'+p.relative_to(self.root).as_posix(),'--reference',repo+'/'+reference.relative_to(self.root).as_posix(),
              '--previous',previous,'--arms',','.join(r['arms']),'--batches',','.join(map(str,r['batches'])),'--n',str(r['n_per_arm'])]
        command=f'set -eu\ngit -C {q(repo)} -c http.proxy={q(self.args.remote_proxy)} pull --ff-only --quiet\ntest "$(git -C {q(repo)} rev-parse HEAD)" = {q(commit)}\n'
        command+=f'/opt/miniforge3/envs/molsteer-flowr-dtk/bin/python {q(repo+"/scripts/dispatch_motif_round.py")} '+' '.join(q(v) for v in args)
        launched=json.loads(self.remote(command));c=read_json(self.campaign)
        c['rounds'][-1].update(status='running',inference_commit=launched['inference_commit']);c['rounds_started']=number;write_json(self.campaign,c)
        self.event('round_launched',round=number,remote_pid=launched['pid'])
    def finish(self,r):
        result=super().finish(r);n=r['round'];local=self.evidence/f'round_{n:02d}/local'
        import pandas as pd
        data=pd.read_csv(local/'candidate_metrics.csv');g=data[data.arm=='gradient'];valid=g[g.valid_connected];pb=g[g.pb_fast_pass]
        uniq=valid.drop_duplicates('smiles');program=read_json(self.cfg/f'backtrack_round{n:02d}.json')
        result.update(reward_view=program['reward_view'],native_rms_ratio=program['native_rms_ratio'],head_coverage=float(g.pic50_on_rescore.notna().mean()),
            response_class=classify(result['all_head_change_vs_native']),best_valid_head=float(valid.pic50_on_rescore.max()),
            best_PB_head=float(pb.pic50_on_rescore.max()),unique_Top5_mean=float(uniq.nlargest(5,'pic50_on_rescore').pic50_on_rescore.mean()),
            valid_head=float(valid.pic50_on_rescore.mean()),objective_profile='affinity_primary_coordinate30',skill_response_verified=True)
        if n==26:
            rows=[read_json(p) for p in self.evidence.glob('round_*.outcome.json') if read_json(p)['round']<26]+[result]
            result['frozen_winner']=select_parent(rows,True)['round']
        write_json(self.evidence/f'round_{n:02d}.outcome.json',result);self.summarize();return result
    def summarize(self):
        rows=[read_json(p) for p in sorted(self.evidence.glob('round_*.outcome.json'))]
        lines=['# New affinity-primary geometry campaign','',f'{len(rows)}/30 rounds retained. Seed42, dynamic learned window, complete100 native steps; no SMC.',
            '','|Round|Split|Reward|Dose|Mean head delta|Best valid head|Unique Top5|Response|','|---:|---|---|---:|---:|---:|---:|---|']
        def score(v):return f'{v:.6f}' if v is not None else 'NA'
        for r in rows:
            if 'reward_view' in r:lines.append(f"|{r['round']}|{r['split']}|{r['reward_view']}|{r['native_rms_ratio']:.4g}|{score(r['all_head_change_vs_native'])}|{score(r['best_valid_head'])}|{score(r['unique_Top5_mean'])}|{r['response_class']}|")
        lines+=['','Historical Steer100 mean7.510351, best-valid8.347940; not an equal-budget paired heldout arm.',
                'Mean/maximum/top5 and physical tails are distinct outcomes. Recorded head predictions are not measured affinity.',
                '26 adaptive discovery rounds; four subsequent frozen checks. Registered search implements the independently tested Skill policy.']
        (self.evidence/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    def close(self):
        from evomolsteer.generation.affinity_reporting import report
        report(self.evidence,self.evidence/'final_report')
        self.command(30,'git_add_compact_statistics',['git','add','-f','--',str(self.evidence/'final_report/round_metrics.parquet'),
                     str(self.evidence/'final_report/paired_batch_metrics.parquet')])
        c=read_json(self.campaign);c['status']='budget_complete';write_json(self.campaign,c);self.summarize();self.push(30)
        repo,work=self.args.remote_repo,self.args.remote_work;q=shlex.quote
        self.remote(f'git -C {q(repo)} -c http.proxy={q(self.args.remote_proxy)} pull --ff-only --quiet')
        label=c['rounds'][-1]['campaign'];report=repo+f'/{EVIDENCE}/round_30/comparison/comparison.json'
        plan={'allowed_bases':[work],'protected':[repo,'/root/private_data/MolSteer/flowr_root/experiments/ck2_clk3_lineage_20261003','/root/private_data/MolSteer/flowr_root/checkpoints'],
              'result_report':report,'targets':[work+'/generated',work+'/'+label+'.tar.gz',work+'/'+label+'.tar.gz.json']}
        path=self.state/'final_remote_cleanup.plan.json';write_json(path,plan)
        with self.ssh.open_sftp() as sftp:sftp.put(str(path),work+'/final_remote_cleanup.plan.json')
        self.remote(f'cd {q(repo)}; PYTHONPATH={q(repo+"/src")} /opt/miniforge3/envs/molsteer-flowr-dtk/bin/python scripts/retire_experiment_outputs.py --plan {q(work+"/final_remote_cleanup.plan.json")} --report {q(work+"/final_remote_cleanup.json")} --apply')
        with self.ssh.open_sftp() as sftp:sftp.get(work+'/final_remote_cleanup.json',str(self.evidence/'final_remote_cleanup.json'))
        self.push(30)
        final_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        self.remote(f'git -C {q(repo)} -c http.proxy={q(self.args.remote_proxy)} pull --ff-only --quiet\ntest "$(git -C {q(repo)} rev-parse HEAD)" = {q(final_commit)}')
        self.event('campaign_complete',rounds_completed=30)
    def run(self):
        c=read_json(self.campaign)
        if c['maximum_rounds']!=30 or c['rounds_completed']>30:raise ValueError('Independent new30 contract')
        if not c['rounds']:self.freeze(1)
        while True:
            c=read_json(self.campaign);r=c['rounds'][-1]
            if r['status']=='frozen':self.launch_frozen(r);continue
            if r['status']=='running':self.finish(r)
            elif r['status']!='completed':raise ValueError('Unknown execution state')
            else:
                saved=read_json(self.evidence/f"round_{r['round']:02d}.outcome.json")
                if 'reward_view' not in saved or (r['round']==26 and 'frozen_winner' not in saved):
                    # Recover a crash between generic retention and affinity metadata;
                    # no new inference is launched and the frozen identity is unchanged.
                    self.finish(r)
            if r['round']==30:self.close();return
            if r['round']>=self.args.until_round:self.push(r['round']);self.event('requested_round_boundary',round=r['round']);return
            self.freeze(r['round']+1)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',default=str(Path(__file__).resolve().parents[1]));p.add_argument('--until-round',type=int,default=30)
    p.add_argument('--endpoint-profile',action='store_true',help='Use independently verified endpoint Skill and real FLOWR coordinate VJP from round5')
    p.add_argument('--sparse-profile',action='store_true',help='Use independently retested, node-faithful coordinate Skill from round6')
    p.add_argument('--host',default='ksai.scnet.cn');p.add_argument('--port',type=int,default=10544);p.add_argument('--user',default='root')
    p.add_argument('--remote-repo',default='/root/private_data/MolSteer/EvoMolSteer')
    p.add_argument('--remote-work',default='/root/private_data/MolSteer/flowr_root/experiments/evomolsteer_affinity_geometry30_20261007')
    p.add_argument('--local-proxy',default='http://127.0.0.1:7897');p.add_argument('--remote-proxy',default='http://127.0.0.1:17897')
    d=None
    try:d=AffinityDriver(p.parse_args());d.run()
    except Exception as e:
        if d:d.event('execution_error',error_type=type(e).__name__,message=str(e)[-1500:])
        raise
    finally:
        if d:
            if d.forward_owned:d.ssh.get_transport().cancel_port_forward('127.0.0.1',17897)
            d.ssh.close()
