"""Immutable deployment of seven-box, five-round guidance with fresh calibration."""
import argparse
import copy
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import xml.etree.ElementTree as ET

ROOT=Path('/workspace/agenticgen_seven_box_search_20261102')
STORAGE=Path('/root/private_data/uniguide_seven_box_search_20261102')
OLD=Path('/workspace/agenticgen_guidance_comparison5_20261031')
SELECTED=Path('/workspace/agenticgen_guidance_selected_retest_20261031')
SCOPE=Path('/workspace/agenticgen_gradient_scope_20261101')
DOC='agenticgen/docs/experiments/20261102_seven_box_formal'
TARGETS=('6pyy_H7S_A_402','3w54_RNB_A_504')
REPO_SUFFIX=''
REPAIR_META=None
EXTRA_PROTECTED=(Path('/workspace/agenticgen_box_exclusion_20261102'),)
PYTHON='/opt/miniforge3/envs/molsteer-flowr-dtk/bin/python'
TESTS=('box_exclusion','local_derivative_acceptance','synchronized_chemistry','zero_start_guidance','open_domain_search',
       'strict_final_submission','guidance_comparison5','comparison8_contracts',
       'prefix_promotion_recovery','dynamic_matching_jump','linear_guidance',
       'adaptive_calibration','geometry_tensor_contract','geometry_chemical_state',
       'uniguide_geometry','uniguide_geometry_batch','uniguide_geometry_dynamics',
       'guidance_skill_bundle','guidance_skill_release_v2','guidance_analysis_tools',
       'deepseek_tools','coordinate_statistics','absolute_uncapped_statistics',
       'absolute_coordinate_protocol','ug04_task_scheduling','gradient_atom_scope','diverse_box_search','seven_box_search')


def now():return datetime.now(timezone(timedelta(hours=8))).isoformat()
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):digest.update(block)
    return digest.hexdigest()
def write(path,value,exclusive=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x' if exclusive else 'w',encoding='utf8') as stream:
        json.dump(value,stream,ensure_ascii=True,indent=2);stream.write('\n')
def command(args,cwd=None,timeout=120):
    result=subprocess.run(args,cwd=cwd,capture_output=True,text=True,timeout=timeout)
    if result.returncode:raise RuntimeError(f'{args[0]} failed ({result.returncode}): '+result.stderr[-800:])
    return result.stdout


def validate_runtime_config(repo, config, environment):
    """Keep torch/DTK imports inside the explicitly activated runtime, never the launcher."""
    script=('import json,sys; from agenticgen.uniguide.absolute_runtime import validate_config; '
            'print(json.dumps(validate_config(json.load(sys.stdin)),ensure_ascii=True))')
    result=subprocess.run(['bash','-c',environment+shlex.join([PYTHON,'-c',script])],
        cwd=repo,input=json.dumps(config,ensure_ascii=True),capture_output=True,text=True,timeout=120)
    if result.returncode:
        raise RuntimeError('Activated runtime configuration validation failed: '+result.stderr[-800:])
    return json.loads(result.stdout)


def deploy(commit,bundle=None,bundle_sha=None,repair_from=None):
    if repair_from: raise ValueError('Scientific repair requires the audited single-teacher repair launcher; never restart an entire healthy suite')
    if re.fullmatch('[0-9a-f]{40}',commit) is None:raise ValueError('Exact pushed commit required')
    root=ROOT
    repo=Path('/workspace/uniguide_runs')/('seven-box-search-'+commit[:8]+REPO_SUFFIX)
    audit=STORAGE/'audits'/('seven-box-search-'+commit[:8]+REPO_SUFFIX);audit.mkdir(parents=True,exist_ok=True)
    def status(stage,**values):
        state=dict(timestamp=now(),stage=stage,commit=commit)
        state.update(values);write(audit/'DEPLOY_STATUS.json',state)
    try:
        if root.exists() or repo.exists():raise ValueError('Preserve existing root/checkout; inspect before any new attempt')
        repair=REPAIR_META
        if repair:write(audit/'ENGINEERING_REPAIR_PARENT.json',repair,True)
        if shutil.disk_usage('/workspace').free < (10<<30):raise ValueError('Insufficient run disk space')
        if shutil.disk_usage('/root/private_data').free < (256<<20):raise ValueError('Insufficient audit storage')
        old_plan=read(OLD/'PLAN.json');selected_plan=read(SELECTED/'PLAN.json')
        old_repo=Path(selected_plan['repo'])
        if command(['git','rev-parse','HEAD'],cwd=old_repo).strip()!=selected_plan['commit']:
            raise ValueError('Original fixed checkout changed')
        if command(['git','status','--porcelain','--untracked-files=no'],cwd=old_repo).strip():
            raise ValueError('Original checkout is dirty')
        status('fetching_pushed_commit')
        repo.mkdir(parents=True)
        command(['git','init','-q',str(repo)])
        command(['git','remote','add','origin','https://github.com/H-Dw/EvoMolSteer.git'],cwd=repo)
        alternate=repo/'.git/objects/info/alternates';alternate.parent.mkdir(parents=True,exist_ok=True)
        alternate.write_text(str((old_repo/'.git/objects').resolve())+'\n')
        transports=[]
        for prefix in ('','https://ghfast.top/','https://gh-proxy.com/','https://ghproxy.net/','https://gh.llkk.cc/'):
            try:
                result=subprocess.run(['git','fetch',prefix+'https://github.com/H-Dw/EvoMolSteer.git',commit],
                    cwd=repo,capture_output=True,text=True,timeout=35)
                transports.append(dict(transport=prefix or 'direct',code=result.returncode))
                if result.returncode==0:break
            except subprocess.TimeoutExpired:transports.append(dict(transport=prefix or 'direct',timeout=35))
        else:
            if not bundle or sha(bundle)!=bundle_sha:raise ValueError('GitHub fetch failed; no verified pushed-only bundle')
            command(['git','bundle','verify',bundle],cwd=repo)
            command(['git','fetch',bundle,'refs/heads/codex/uniguide-flowr-agent'],cwd=repo)
            transports.append(dict(transport='verified_pushed_only_bundle',sha256=bundle_sha))
        command(['git','sparse-checkout','init','--cone'],cwd=repo)
        command(['git','sparse-checkout','set','src','agenticgen/src','agenticgen/scripts','agenticgen/configs',
                 'agenticgen/tests',DOC,'agenticgen/docs/experiments/20261102_box_exclusion_formal','agenticgen/docs/experiments/20261031_guidance_comparison5',
                 'agenticgen/docs/experiments/20261031_guidance_comparison8'],cwd=repo)
        command(['git','checkout','-B','codex/uniguide-flowr-agent',commit],cwd=repo)
        if command(['git','rev-parse','HEAD'],cwd=repo).strip()!=commit:raise ValueError('Wrong execution commit')
        if sha(repo/DOC/'deploy.py')!=sha(__file__):raise ValueError('Deployment helper is not the pushed source')
        write(audit/'DEPLOYMENT.json',dict(timestamp=now(),repo=str(repo),commit=commit,transports=transports,
              source_alternate_read_only=str(alternate),deployment_sha256=sha(__file__)),True)
        status('remote_cpu_regression')
        setup=('source /opt/MolSteer/scripts/scnet/activate_dtk.sh >/dev/null 2>&1\n'
               f'export PYTHONPATH={repo}/agenticgen/src:{repo}/src:/workspace/uniguide_deps:/opt/MolSteer/flowr_root\n'
               'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1\n')
        cpu=setup+'export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=\nunset DEEPSEEK_API_KEY\n'
        tests=[PYTHON,'-m','pytest','-q','-c','agenticgen/pyproject.toml',
               *['agenticgen/tests/unit/test_'+name+'.py' for name in TESTS],
               '--basetemp=/workspace/pytest-seven-box-search-'+commit[:8]+REPO_SUFFIX,
               '--junitxml='+str(audit/'regression.xml')]
        with (audit/'regression.log').open('xb') as stream:
            result=subprocess.run(['bash','-c',cpu+shlex.join(tests)],cwd=repo,stdout=stream,stderr=subprocess.STDOUT,timeout=900)
        element=ET.parse(audit/'regression.xml').getroot().find('testsuite')
        counts={k:int(element.get(k)) for k in ('tests','failures','errors','skipped')}
        write(audit/'REGRESSION.json',dict(timestamp=now(),returncode=result.returncode,cpu_only=True,
              scientific_attempts=0,llm_calls=0,**counts),True)
        if result.returncode:raise ValueError('CPU regression failed; no scientific task started')
        status('verifying_frozen_inputs_and_history')
        snapshot=Path('/root/private_data/uniguide_comparison5_20261031/audits/comparison5-072e6f78/INPUT_SNAPSHOT.json')
        protected={}
        for item in read(snapshot)['copies'].values():
            if sha(item['destination'])!=item['sha256']:raise ValueError('Frozen durable input changed')
            protected[item['destination']]=item['sha256']
        histories=[]
        for prior in (OLD,SELECTED,SCOPE,Path('/workspace/agenticgen_six_window_sync_20261101'),Path('/workspace/agenticgen_six_window_sync_20261101_repair1'),*EXTRA_PROTECTED):
            preservation=prior/'SOURCE_PRESERVATION.json'
            if preservation.exists():
                for name,digest in read(preservation).items():
                    if sha(name)!=digest:raise ValueError('Historical protected file changed: '+name)
                    protected[name]=digest
            for path in prior.rglob('*'):
                if (path.is_file() and not path.is_symlink() and '__pycache__' not in path.parts
                    and not path.name.endswith('.lock') and not any(part in ('monitoring','quality_queue') for part in path.relative_to(prior).parts)):
                    protected[str(path)]=sha(path)
            ledgers=[]
            for path in (prior/'jobs').glob('*/attempt_*/attempts.json'):
                ledger=read(path)
                ledgers.append(dict(path=str(path),sha256=sha(path),
                    stages={stage:sum(x['count'] for x in ledger.values() if x['stage']==stage) for stage in ('calibration','formal')}))
            histories.append(dict(root=str(prior),ledgers=ledgers))
        overlay=read(repo/'agenticgen/configs/uniguide_diverse_seven_box_search.json')
        jobs=[];expected={};templates={j['teacher']:j for j in old_plan['jobs'] if j['id'].startswith(('01_','02_'))}
        for index,target in enumerate(TARGETS,1):
            original=templates[target];c=copy.deepcopy(read(original['config']));c.update(copy.deepcopy(overlay))
            job_id=f'{index:02d}_seven_box_'+('6pyy' if target.startswith('6pyy') else '3w54')
            attempt='attempt_002' if repair else 'attempt_001'
            c.update(root=str(root/'jobs'/job_id/attempt),rounds=5)
            c['conditions']=[r for r in c['conditions'] if r['teacher_id']==target]
            for name in ('postprocessing_recovery','experiment_mode','gradient_atom_scope','initial_selection'):c.pop(name,None)
            c['llm']=dict(c['llm'],final_response_mode='strict_tool_v1',attempts=3,max_format_repairs=1)
            checked=validate_runtime_config(repo,c,cpu)
            if (checked['rounds']!=5 or checked['chemical_inpainting'].get('activation')!='guidance_phase'
                or checked['guidance_control']['maximum_attempts_per_teacher']!=500 or not checked['box_exclusion']['enabled'] or checked['agent_tools']['skill_release']!='2.13.0' or checked['guidance_control']['maximum_selected']!=5 or checked['windows']!=[[0,10],[11,20],[21,30],[31,40],[41,50],[51,75],[76,99]]):
                raise ValueError('The deployment does not match the authorized seven-window configuration')
            if sha(c['checkpoint'])!=c['checkpoint_sha256']:raise ValueError('Original model checkpoint changed')
            protected[c['checkpoint']]=c['checkpoint_sha256']
            for name,digest in c['native_hashes'].items():
                p=Path(c['flowr'])/name
                if sha(p)!=digest:raise ValueError('Frozen native source changed')
                protected[str(p)]=digest
            old_subject=read(Path(original['root'])/'SUBJECTS.json')[0]
            expected[target]={k:old_subject['condition'][k] for k in ('condition_sha256','controlled_atom_ids')}
            path=Path(str(repo)+'-configs')/(job_id+'.json');write(path,c,True)
            jobs.append(dict(id=job_id,experiment='seven_box_negative_exclusion',teacher=target,config=str(path),
                config_sha256=sha(path),root=c['root'],rounds=5,skill_release='2.13.0',commit=commit,attempt=attempt))
        root.mkdir(parents=True,exist_ok=False)
        write(root/'SOURCE_PRESERVATION.json',protected,True)
        write(root/'HISTORICAL_ACCOUNTING.json',dict(timestamp=now(),roots=histories,engineering_repair=repair,
              imported_measurements=0,old_attempts_not_refunded=True,
              new_calibration_cap=500*len(jobs),new_formal_cap=1250*len(jobs)),True)
        sources=[*(repo/'agenticgen/src').rglob('*.py'),*(repo/'src').rglob('*.py')]
        sources.extend(repo/DOC/name for name in ('deploy.py','run_suite.py','inspect_suite.py','launch.sh'))
        plan=dict(timestamp=now(),root=str(root),repo=str(repo),commit=commit,jobs=jobs,rounds=5,engineering_repair=repair,
             generation_context_root=old_plan['generation_context_root'],
             global_executor_lock=old_plan['global_executor_lock'],wait_for_global_lock=True,
             maximum_calibration_attempts=500*len(jobs),maximum_formal_attempts=1250*len(jobs),fresh_calibration=True,
             expected_conditions=expected,original_input_snapshot=str(snapshot),original_input_snapshot_sha256=sha(snapshot),
             source_preservation_count=len(protected),plan_document_sha256=sha(repo/DOC/'plan.zh-CN.md'),
             persistent_storage=str(STORAGE),durable_snapshot_root=str(STORAGE/('result_snapshots'+(REPO_SUFFIX+'-'+commit[:8] if repair else ''))),
             trajectory_storage=str(root),trajectory_storage_is_overlay=True,
             runtime_source_sha256={p.relative_to(repo).as_posix():sha(p) for p in sources},
             failure_policy='preserve_skip_continue; audited engineering repairs preserve cumulative ledger; total scientific cap 3500')
        write(root/'PLAN.json',plan,True)
        write(audit/'READY.json',dict(timestamp=now(),commit=commit,repo=str(repo),root=str(root),
              plan=str(root/'PLAN.json'),plan_sha256=sha(root/'PLAN.json'),protected_files=len(protected),
              input_copies_verified=len(read(snapshot)['copies']),regression=counts,scientific_attempts=0,llm_calls=0),True)
        status('launching_background_suite')
        launch=['bash','-i',str(repo/DOC/'launch.sh'),str(repo),str(root/'PLAN.json')]
        with (root/'launcher.log').open('xb') as stream:
            process=subprocess.Popen(launch,cwd=repo,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        receipt=dict(timestamp=now(),pid=process.pid,command=launch,commit=commit,jobs=len(jobs),rounds=5,
                     plan_sha256=sha(root/'PLAN.json'),launcher_log_kept_remote_only=True)
        write(audit/'LAUNCH.json',receipt,True);status('background_launched',**receipt)
    except Exception as error:
        write(audit/'DEPLOY_FAILED.json',dict(timestamp=now(),error_type=type(error).__name__,error=str(error),
              preserved=True,automatic_restart=False))
        status('deployment_failed_preserved',error_type=type(error).__name__)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--commit',required=True)
    parser.add_argument('--bundle');parser.add_argument('--bundle-sha')
    parser.add_argument('--repair-from')
    args=parser.parse_args();deploy(args.commit,args.bundle,args.bundle_sha,args.repair_from)
