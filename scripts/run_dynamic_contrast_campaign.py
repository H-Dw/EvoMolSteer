"""Fresh ten-round sequential cohort exploration, local analysis/remote inference.

Every proposal waits for its parent's retained report. No confirmation labels
enter adaptive selection. Credentials and network configuration remain runtime.
"""
import argparse,copy,json,os,shlex,shutil,sys,time
from pathlib import Path
import pandas as pd
from evomolsteer.io import read_json,write_json,digest
from evomolsteer.storage.generation_archive import verify_generation_archive
from evomolsteer.generation.path_evaluation import retain_round,summarize_tail
from evomolsteer.continuous.dynamic_regions import mine
from run_sequential_path_campaign import Driver as TransportDriver,evaluate_isolated


class Driver(TransportDriver):
    def __init__(self,args):
        self.args=args;self.root=Path(args.repo).resolve();self.cfg=self.root/'configs/experiments/dynamic_contrast10_v1'
        self.docs=self.root/'docs/experiments/dynamic_contrast10_20261008';self.state=self.root/'test/dynamic_contrast10_driver'
        self.state.mkdir(parents=True,exist_ok=True);self.work=args.remote_work;self.remote_repo=args.remote_repo
        self.threshold=read_json(self.docs/'protocol.json')['tail_threshold_pic50'];self.connect()

    def push(self,message):
        self.local(['git','add',str(self.cfg.relative_to(self.root)),str(self.docs.relative_to(self.root))])
        self.local(['git','diff','--cached','--check'])
        if self.local(['git','diff','--cached','--name-only']):self.local(['git','commit','-m',message])
        self.local(['git','-c','http.proxy=http://127.0.0.1:7897','push','origin','main'])
        commit=self.local(['git','rev-parse','HEAD']);q=shlex.quote
        self.remote(f'timeout 180 git -C {q(self.remote_repo)} -c http.proxy=http://127.0.0.1:17897 pull --ff-only origin main')
        if self.remote(f'git -C {q(self.remote_repo)} rev-parse HEAD')!=commit:raise ValueError('Remote exact commit mismatch')
        return commit

    def metrics(self,number):
        d=pd.read_csv(self.docs/f'round{number:02d}/candidate_metrics.csv')
        return summarize_tail(d[d.arm=='gradient'],self.threshold)

    def best(self):
        numbers=[n for n in range(2,8) if (self.docs/f'round{n:02d}/retention.json').exists()]
        if not numbers:raise ValueError('No completed adaptive candidate')
        def rank(n):
            m=self.metrics(n);return (int(m['valid_n']/m['n']>=.90 and m['pb_fast_rate']>=.90),
                m['all_mean_pic50'],m['valid_p95_pic50'],m['elite_unique_graphs'],-m['strain_median_per_heavy'],-n)
        return max(numbers,key=rank)

    def resolve_reference(self,p):
        candidates=list(self.cfg.glob('*.json.gz'))+[self.root/'configs/experiments/skill_ablation_v1/endpoint_reference.json.gz']
        for f in candidates:
            if digest(f)==p['reference_sha256']:return f
        raise ValueError('Exact immutable teacher reference missing')

    def freeze(self,number):
        existing=self.cfg/f'round{number:02d}.json'
        planfile=self.docs/f'round{number:02d}_plan.json'
        if existing.exists():
            plan=read_json(planfile);return existing,self.resolve_reference(read_json(existing)),plan['batches'],plan['arms'],plan['n_per_arm']
        if number==1:
            p=read_json(self.root/'configs/experiments/skill_ablation_v1/incumbent.json');parent='historical_R26'
            change={};reason='Fresh paired native and validated historical incumbent control, before any new regional reward trial.'
        elif number==2:
            p=read_json(self.cfg/'round01.json');parent=1
            binding=read_json(self.docs/'round02_agents/Designer.request.json');response=read_json(self.docs/'round02_agents/Designer.response.json')
            for k in ['input_sha256','instruction_sha256']:
                if response[k]!=binding[k]:raise ValueError('Designer response binding mismatch')
            if response['decision']!='test_formula' or response['reward_view']!='endpoint_dynamic_region':raise ValueError('Designer deferred or changed registry')
            reference=self.cfg/'rest_reference.json.gz'
            change={'reward_view':'endpoint_dynamic_region','regional_weight':.05,'regional_temperature':1.,
                'regional_bound':2.,'regional_response':'contrast','reference_sha256':digest(reference)}
            reason=response['justification'];p.update(change)
        elif number<=7:
            parent=self.best();p=read_json(self.cfg/f'round{parent:02d}.json')
            if number in [3,4]:
                change={'regional_weight':.15 if number==3 else .40}
                reason='Vary only regional scalar strength at unchanged physical controller dose, to test whether the new direction is useful or overwhelms the incumbent.'
            elif number in [5,6]:
                current=json.loads(__import__('gzip').decompress(self.resolve_reference(p).read_bytes()))['dynamic_cohort']
                hard=.75 if number==5 else current['hard_mix'];margin=.10 if number==5 else .20
                out=self.root/f'results/dynamic_contrast10/round{number:02d}_mining'
                manifest=mine(self.root/'data/raw/ck2_clk3_lineage_20261003','main1000_w050',
                    self.root/'configs/experiments/skill_ablation_v1/endpoint_reference.json.gz',out,
                    hard_mix=hard,margin_fraction=margin,gap_weight=current['gap_weight'])
                reference=self.cfg/f'round{number:02d}_reference.json.gz';shutil.copy2(out/'reference.json.gz',reference)
                retain_mining(out,self.docs/f'round{number:02d}_mining')
                change={'reference_sha256':digest(reference)}
                reason=('Change only offline control sampling: 75% spatially close observed lower-score mass plus 25% rest mass; match nuisance centroid/shape, not mined region fields.' if number==5 else
                    'Change only the event-relative ambiguity margin to 20% weighted IQR; guard uncertain negatives and re-estimate the same regional contrast pipeline.')
            else:
                change={'regional_response':'positive'}
                reason='Remove background repulsion in the added regional scalar while preserving measured positive targets and the evaluated incumbent. Test whether lower-score density subtraction was harmful.'
            p.update(change)
        else:
            frozen=self.docs/'frozen_validation.json'
            if not frozen.exists():
                if number!=8:raise ValueError('Freeze before observing confirmation')
                winner=self.best();source=self.cfg/f'round{winner:02d}.json'
                write_json(frozen,{'candidate_round':winner,'program_sha256':digest(source),'reference_sha256':read_json(source)['reference_sha256'],
                    'screening_batch':36,'master_seed':42,'confirmation_batches':[37,38,39,40],
                    'selection_rule':read_json(self.docs/'protocol.json')['selection_rule']})
            frozen=read_json(frozen)
            if digest(self.cfg/f'round{frozen["candidate_round"]:02d}.json')!=frozen['program_sha256']:raise ValueError('Frozen candidate changed')
            parent=1 if number==8 else frozen['candidate_round'];p=read_json(self.cfg/f'round{parent:02d}.json')
            change={};reason='Frozen independent confirmation; no reward or cohort policy tuning on this panel.'
        inherited={k:p.pop(k) for k in list(p) if k.endswith('_sha256') and k!='reference_sha256'}
        p.update(round=number,program_id=f'dynamic_contrast_r{number:02d}',objective_profile='dynamic_contrast10',
            derivation={'parent_round':parent,'single_module_change':change,'scientific_justification':reason,
                'historical_parent_provenance':inherited,'no_private_reasoning_transcript':True})
        existing.write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
        batches='36' if number<8 else '39,40' if number==10 else '37,38'
        arms='unguided,gradient' if number in [1,8,10] else 'gradient';n=50 if number<8 else 100
        write_json(planfile,{'round':number,'parent':parent,'change':change,'reason':reason,'batches':batches,'arms':arms,
            'n_per_arm':n,'program_sha256':digest(existing),'reference_sha256':p['reference_sha256'],
            'parent_screen_metrics':self.metrics(parent) if isinstance(parent,int) and (self.docs/f'round{parent:02d}/retention.json').exists() else None})
        return existing,self.resolve_reference(p),batches,arms,n

    def collect(self,number):
        campaign=f'dynamic_contrast_r{number:02d}';archive=self.state/f'round{number:02d}.tar.gz'
        for suffix in ['', '.json']:self.download(self.work+'/'+campaign+'.tar.gz'+suffix,str(archive)+suffix,number)
        dataset=self.root/f'test/data/dynamic_contrast10/round{number:02d}';evaluated=self.root/f'results/dynamic_contrast10/round{number:02d}'
        verified=verify_generation_archive(archive,str(archive)+'.json',dataset)
        cfg=read_json(dataset/'results'/campaign/'config.json')['experiment']
        self.emit('local_evaluation',round=number,archive_verified=True)
        evaluate_isolated(self.root,dataset,campaign,evaluated,cfg,self.state/f'round{number:02d}.evaluation.log')
        baseline=self.docs/'round01/candidate_metrics.csv' if 2<=number<=7 else self.docs/'round08/candidate_metrics.csv' if number==9 else None
        summary=retain_round(dataset,campaign,evaluated,self.docs/f'round{number:02d}',self.threshold,baseline)
        retentionfile=self.docs/f'round{number:02d}/retention.json';ret=read_json(retentionfile)
        p=retentionfile.parent/'archive_verification.json';write_json(p,verified)
        ret['files'].append({'path':p.name,'sha256':digest(p),'bytes':p.stat().st_size});write_json(retentionfile,ret)
        c=read_json(self.cfg/'campaign.json')
        if c['rounds_completed']!=number-1:raise ValueError('Local sequential state changed')
        c['rounds_completed']=number;c['rounds'].append({'round':number,'status':'complete','kind':'actual_FLOWR_inference',
            'results':summary,'retention_report':str(retentionfile.relative_to(self.root)).replace('\\','/')})
        write_json(self.cfg/'campaign.json',c);self.emit('round_complete',round=number,summary=summary)
        self.push(f'Retain dynamic contrast round {number} and paired terminal outcomes')
        base=(self.root/'test/data/dynamic_contrast10').resolve()
        if not dataset.resolve().is_relative_to(base) or dataset.resolve()==base:raise ValueError('Unsafe local generated retirement')
        shutil.rmtree(dataset)
        for f in [archive,Path(str(archive)+'.json')]:
            if not f.resolve().is_relative_to(self.state.resolve()):raise ValueError('Unsafe local archive retirement')
            f.unlink()

    def run(self):
        for number in range(self.args.start,self.args.last+1):
            if read_json(self.cfg/'campaign.json')['rounds_completed']!=number-1:raise ValueError('Prior report required before next proposal')
            path,reference,batches,arms,n=self.freeze(number)
            commit=self.push(f'Freeze sequential dynamic cohort exploration round {number}')
            q=shlex.quote;command=['/opt/miniforge3/envs/molsteer-flowr-dtk/bin/python',self.remote_repo+'/scripts/dispatch_contrast_round.py',
                '--repo',self.remote_repo,'--work',self.work,'--round',str(number),
                '--program',self.remote_repo+'/'+path.relative_to(self.root).as_posix(),
                '--reference',self.remote_repo+'/'+reference.relative_to(self.root).as_posix(),
                '--batches',batches,'--arms',arms,'--n',str(n)]
            if number>1:command+=['--previous',self.remote_repo+f'/docs/experiments/dynamic_contrast10_20261008/round{number-1:02d}/retention.json']
            self.remote('source /opt/MolSteer/scripts/scnet/activate_dtk.sh\n'+' '.join(q(v) for v in command))
            self.emit('inference_started',round=number,commit=commit,batches=batches,arms=arms)
            while True:
                status=self.remote(f'if test -f {q(self.work+f"/round{number:02d}.exit")}; then cat {q(self.work+f"/round{number:02d}.exit")}; else echo running; fi')
                if status!='running':
                    if status!='0':raise RuntimeError('Inference failed; retain and diagnose: '+self.remote(f'tail -n 30 {q(self.work+f"/round{number:02d}.log")}'))
                    break
                time.sleep(30)
            self.collect(number)
        self.emit('requested_sequence_complete',last=self.args.last)


def retain_mining(source,output):
    """Small deterministic summaries retained in Git; raw codes/moments rebuildable."""
    source,output=Path(source),Path(output);output.mkdir(parents=True,exist_ok=False)
    for name in ['whole_window_effects','event_cohorts','feature_trends']:
        pd.read_parquet(source/(name+'.parquet')).to_csv(output/(name+'.csv'),index=False,lineterminator='\n')
    for name in ['effect_functions.json','manifest.json']:shutil.copy2(source/name,output/name)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',default='.');p.add_argument('--host',default='ksai.scnet.cn');p.add_argument('--port',type=int,default=10544)
    p.add_argument('--password-env',default='MOLSTEER_SCNET_PASSWORD');p.add_argument('--start',type=int,default=1);p.add_argument('--last',type=int,default=10)
    p.add_argument('--collect-only',type=int)
    p.add_argument('--remote-repo',default='/root/private_data/MolSteer/EvoMolSteer')
    p.add_argument('--remote-work',default='/root/private_data/MolSteer/flowr_root/experiments/evomolsteer_dynamic_contrast10_20261008')
    a=p.parse_args()
    if not 1<=a.start<=a.last<=10:raise ValueError('Fresh ten-round budget')
    d=Driver(a)
    try:
        if a.collect_only is not None:d.collect(a.collect_only)
        else:d.run()
    finally:d.ssh.close()
