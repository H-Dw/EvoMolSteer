"""Deploy a pushed fixed commit and six new roots, preserving all historical inputs."""
import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from datetime import datetime, timezone, timedelta
import xml.etree.ElementTree as ET

STORAGE=Path('/root/private_data/uniguide_gradient_scope_20261101')
ROOT=Path('/workspace/agenticgen_gradient_scope_20261101')
OLD=Path('/workspace/agenticgen_guidance_comparison5_20261031')
PREVIOUS=Path('/workspace/agenticgen_guidance_selected_retest_20261031')
DOC='agenticgen/docs/experiments/20261101_gradient_scope'
PYTHON='/opt/miniforge3/envs/molsteer-flowr-dtk/bin/python'
def stamp():return datetime.now(timezone(timedelta(hours=8))).isoformat()
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=True,indent=2)+'\n')
def command(args,cwd=None):
    p=subprocess.run(args,cwd=cwd,capture_output=True,text=True,timeout=120)
    if p.returncode:raise RuntimeError(p.stderr[-1200:])
    return p.stdout


def main():
    parser=argparse.ArgumentParser();parser.add_argument('commit');args=parser.parse_args()
    commit=args.commit;assert len(commit)==40 and all(c in '0123456789abcdef' for c in commit)
    repo=STORAGE/'runs'/('scope-'+commit[:8]);audit=STORAGE/'audits'/('scope-'+commit[:8]);audit.mkdir(parents=True,exist_ok=True)
    def status(stage,**fields):write(audit/'DEPLOY_STATUS.json',dict(timestamp=stamp(),stage=stage,commit=commit,**fields))
    try:
        assert not ROOT.exists() and not repo.exists()
        oldplan=read(OLD/'PLAN.json');oldrepo=Path(read(PREVIOUS/'PLAN.json')['repo'])
        assert sha(PREVIOUS/'FINALIZED.json')=='a326bd92455f79f846423315aaaff829d5193454812dbc97c92f19900d407059'
        status('fetching')
        repo.mkdir(parents=True);command(['git','init','-q',str(repo)])
        command(['git','remote','add','origin','https://github.com/H-Dw/EvoMolSteer.git'],repo)
        alt=repo/'.git/objects/info/alternates';alt.parent.mkdir(parents=True,exist_ok=True);alt.write_text(str(oldrepo/'.git/objects')+'\n')
        transports=[]
        for prefix in ['', 'https://ghfast.top/','https://gh-proxy.com/','https://ghproxy.net/','https://gh.llkk.cc/']:
            try:
                p=subprocess.run(['git','fetch',prefix+'https://github.com/H-Dw/EvoMolSteer.git',commit],cwd=repo,capture_output=True,text=True,timeout=20)
                transports.append(dict(transport=prefix or 'direct',code=p.returncode))
                if p.returncode==0:break
            except subprocess.TimeoutExpired:transports.append(dict(transport=prefix or 'direct',timeout=20))
        else:raise RuntimeError('All GitHub transports failed; require verified pushed-only bundle')
        dirs=command(['git','-C',str(oldrepo),'sparse-checkout','list']).splitlines()+[DOC]
        command(['git','sparse-checkout','init','--cone'],repo);command(['git','sparse-checkout','set',*sorted(set(dirs))],repo)
        command(['git','checkout','-B','codex/uniguide-flowr-agent',commit],repo)
        assert sha(repo/DOC/'deploy_scope.py')==sha(__file__)
        write(audit/'DEPLOYMENT.json',dict(timestamp=stamp(),commit=commit,repo=str(repo),transports=transports))
        env=os.environ.copy();env.update(PYTHONPATH=f'{repo}/agenticgen/src:{repo}/src:/workspace/uniguide_deps:/opt/MolSteer/flowr_root',
            OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
        for k in ['DEEPSEEK_API_KEY','OPENAI_API_KEY']:env.pop(k,None)
        cpu=dict(env,CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',ROCR_VISIBLE_DEVICES='')
        source=ast.parse((repo/'agenticgen/docs/experiments/20261031_selected_contract_retest/deploy_retest.py').read_text())
        tests=next(ast.literal_eval(n.value) for n in source.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='TESTS' for t in n.targets))
        # Thousands of tiny fixture writes belong on the run disk, not NFS.
        test_tmp=Path('/workspace/uniguide_scope_cpu_tests_20261101')/commit[:8]
        test_tmp.parent.mkdir(parents=True,exist_ok=True)
        cmd=[PYTHON,'-m','pytest','-q','-c','agenticgen/pyproject.toml']+['agenticgen/tests/unit/test_'+n+'.py' for n in tests+('gradient_atom_scope','fixed_scope','scope_archive')]+['--basetemp='+str(test_tmp),'--junitxml='+str(audit/'regression.xml')]
        status('cpu_regression')
        with (audit/'regression.log').open('xb') as log:
            p=subprocess.run(cmd,cwd=repo,env=cpu,stdout=log,stderr=subprocess.STDOUT,timeout=600)
        suite=ET.parse(audit/'regression.xml').getroot().find('testsuite')
        counts={k:int(suite.get(k)) for k in ['tests','failures','errors','skipped']}
        write(audit/'REGRESSION.json',dict(timestamp=stamp(),code=p.returncode,**counts))
        assert p.returncode==0 and not counts['skipped']
        ROOT.mkdir();configs=Path(str(repo)+'-configs');configs.mkdir();jobs=[]
        for hops in range(3):
            for original in oldplan['jobs'][:2]:
                cfg=copy.deepcopy(read(original['config']));baseline=Path(cfg['root']);target=original['teacher']
                domainfile=baseline/'calibration/domains.json';domain=read(domainfile)[target]
                job_id=f'hop{hops}_'+('6pyy' if target.startswith('6pyy') else '3w54')
                cfg.update(root=str(ROOT/'jobs'/job_id),rounds=0,experiment_mode='fixed_gradient_atom_scope',
                    guidance_atom_scope=dict(mode='condition_hops',hops=hops),
                    fixed_guidance_schedule=[domain['calibrated_guidance_reference_scale']]*4,
                    fixed_scale_reference=dict(baseline_root=str(baseline),domains_file_sha256=sha(domainfile),
                        source_complete_sha256=sha(baseline/domain['initial_source']/'COMPLETE.json')),
                    checkpoint_learning={'enabled':False},prefix_inpainting={'enabled':False})
                cfg['agent_tools']['skill_release']='2.8.2'
                path=configs/(job_id+'.json');write(path,cfg)
                jobs.append(dict(id=job_id,teacher=target,hops=hops,experiment='gradient_atom_scope',rounds=0,
                                 skill_release='2.8.2',config=str(path),config_sha256=sha(path),schedule=cfg['fixed_guidance_schedule']))
        protected=read(PREVIOUS/'SOURCE_PRESERVATION.json')
        status('binding_historical_files')
        for root in [OLD,PREVIOUS]:
            for path in root.rglob('*'):
                if path.is_file() and not path.is_symlink() and not path.name.endswith('.tar.gz') and path.name!='launcher.log' and path.suffix!='.lock':
                    protected[str(path)]=sha(path)
        write(ROOT/'SOURCE_PRESERVATION.json',protected)
        plan=dict(timestamp=stamp(),root=str(ROOT),repo=str(repo),commit=commit,jobs=jobs,
                  global_executor_lock='/workspace/agenticgen_uniguide_flowr_20261026/.executor.lock',
                  scientific_attempt_cap=300,calibration_attempts=0,llm_calls=0,iterations=0,
                  preservation_files=len(protected),persistent_storage=str(STORAGE))
        runtime_files=[* (repo/'agenticgen/src/agenticgen/uniguide').glob('*.py'),
                       * (repo/'agenticgen/src/agenticgen/providers').glob('*.py'),repo/'agenticgen/scripts/run_uniguide_geometry.py',
                       repo/DOC/'run_suite.py',repo/DOC/'archive_results.py']
        plan['runtime_source_sha256']={str(p.relative_to(repo)):sha(p) for p in runtime_files}
        write(ROOT/'PLAN.json',plan);write(audit/'PLAN.json',plan)
        archive_roots=[str(p) for p in Path('/workspace').glob('agenticgen_*') if p.is_dir()]
        archive=dict(timestamp=stamp(),roots=sorted(archive_roots),staging='/workspace/uniguide_result_exports_20261101',
                     destination='/root/private_data',activate_after='2026-11-01T08:00:00+08:00',
                     exclude_model_weights=True,exclude_launcher_logs=True,exclude_core_dumps=True,exclude_existing_backup_tar=True,
                     expected_new_plan=str(ROOT/'PLAN.json'),storage_choice_pending=True)
        archive['model_weight_paths']=sorted({read(j['config'])['checkpoint'] for j in jobs})
        write(STORAGE/'ARCHIVE_PLAN.json',archive)
        status('launching_suite')
        # DTK is loaded by the caller; API keys are removed and no provider is invoked.
        cmd=[PYTHON,'-u',str(repo/DOC/'run_suite.py'),'--plan',str(ROOT/'PLAN.json')]
        with (ROOT/'launcher.log').open('xb') as log:
            child=subprocess.Popen(cmd,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        launch=dict(timestamp=stamp(),commit=commit,repo=str(repo),root=str(ROOT),audit=str(audit),plan=str(ROOT/'PLAN.json'),
                    plan_sha256=sha(ROOT/'PLAN.json'),pid=child.pid,command=cmd,jobs=jobs,archive_plan=str(STORAGE/'ARCHIVE_PLAN.json'),archive_plan_sha256=sha(STORAGE/'ARCHIVE_PLAN.json'))
        write(audit/'LAUNCH.json',launch);status('background_running',pid=child.pid);print(json.dumps(launch),flush=True)
    except Exception as e:
        status('failed_preflight',error=repr(e),automatic_retry=False);raise


if __name__=='__main__':main()
