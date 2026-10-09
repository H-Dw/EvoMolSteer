"""Cooperating single-target Steer workers and verified ten-target archives."""
import argparse
from dataclasses import asdict,dataclass,field,replace
import json
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import sys
import socket
import traceback
import uuid

from ..io import digest,read_json
from ..storage.transactions import atomic_json,checked_path
from ..trajectory_source import validate_input_bundle
from .steer_launcher import (SteerLearningConfig,INPUT_FORMAT,verify_dataset,
                             DEFAULT_STEER_SAMPLES,DEFAULT_STEER_BATCH_SIZE)
from .target_catalog import discover_targets
from .target_progress import TargetProgress,TargetClaimConflict,FORMAT as PROGRESS_FORMAT
from .campaign_journal import CampaignJournal
from .target_archiving import TargetArchiveCoordinator

FORMAT='evomolsteer.steer_target_campaign.v1'


@dataclass(frozen=True)
class TargetCampaignConfig:
    flowr_root:str
    checkpoint:str
    input_dataset:str
    output_dataset:str
    python_executable:str=sys.executable  # FLOWR worker interpreter
    target_manifest:str|None=None
    expected_targets:int|None=None
    target_ids:tuple[str,...]=()
    protein_glob:str='**/*_pocket10.pdb'
    protein_suffix:str='_pocket10.pdb'
    ligand_suffix:str='.sdf'
    campaign:str='single_w050'
    samples:int=DEFAULT_STEER_SAMPLES
    batch_size:int=DEFAULT_STEER_BATCH_SIZE
    steps:int=100
    window_start:float=0.
    window_end:float=.5
    seed:int=42
    storage_codec:str='none'
    capture_bond_probabilities:bool=False
    compress:bool=True
    archive_every:int=10  # targets, not candidate molecules or particle batches
    compression_level:int=6
    remove_archived_targets:bool=True
    continue_on_error:bool=True
    runtime_environment:dict[str,str]=field(default_factory=dict)

    def resolved(self):
        cfg=replace(self,**{key:str(Path(getattr(self,key)).expanduser().resolve()) for key in
                           ('flowr_root','checkpoint','input_dataset','output_dataset')},
                    python_executable=str(Path(shutil.which(self.python_executable) or self.python_executable).expanduser().resolve()),
                    target_ids=tuple(self.target_ids))
        if cfg.target_manifest:cfg=replace(cfg,target_manifest=str(Path(cfg.target_manifest).resolve()))
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',cfg.campaign):raise ValueError('One campaign directory name required')
        if cfg.expected_targets is not None and (type(cfg.expected_targets) is not int or cfg.expected_targets<1):
            raise ValueError('expected_targets must be a positive integer')
        if type(cfg.archive_every) is not int or cfg.archive_every<1 or type(cfg.compression_level) is not int or not 1<=cfg.compression_level<=9:
            raise ValueError('Invalid archive group size or gzip level')
        if any(type(getattr(cfg,key)) is not bool for key in
               ('compress','remove_archived_targets','continue_on_error','capture_bond_probabilities')):
            raise ValueError('Policy flags must be bool')
        if not isinstance(cfg.runtime_environment,dict) or any(not isinstance(k,str) or not isinstance(v,str) for k,v in cfg.runtime_environment.items()):
            raise ValueError('runtime_environment must contain string keys and values')
        source,output=Path(cfg.input_dataset),Path(cfg.output_dataset)
        if source==output or source.is_relative_to(output) or output.is_relative_to(source):
            raise ValueError('Input and output collections must be disjoint')
        return cfg


def _generator_script():return Path(__file__).resolve().parents[3]/'scripts/generate_steer_learning.py'


def _job_config(cfg,row,output,input_manifest):
    return SteerLearningConfig(flowr_root=cfg.flowr_root,checkpoint=cfg.checkpoint,
        input_dataset=cfg.input_dataset,output_dataset=str(output),input_manifest=str(input_manifest),
        python_executable=cfg.python_executable,campaign=cfg.campaign,samples=cfg.samples,batch_size=cfg.batch_size,
        steps=cfg.steps,window_start=cfg.window_start,window_end=cfg.window_end,seed=cfg.seed,arms=('single',),
        storage_codec=cfg.storage_codec,capture_bond_probabilities=cfg.capture_bond_probabilities)


def _result(root,cfg,row=None):
    manifest=read_json(root/'learning_dataset_manifest.json')
    if manifest.get('state')!='complete':raise ValueError('Inner generation dataset is incomplete')
    verify_dataset(root,cfg.campaign,arms=('single',),samples=cfg.samples)
    validate_input_bundle(root,{'campaign':cfg.campaign})
    if row:
        copied=read_json(root/'inputs/pocket_inputs.json')
        for role,value in row['files'].items():
            if digest(root/'inputs'/copied['files'][role])!=value['sha256']:
                raise ValueError('Generated target inputs differ from the frozen catalog')
    rows=read_json(root/'results'/cfg.campaign/'final_records.json')
    if len(rows)!=cfg.samples:raise ValueError('Result collector omitted candidate slots')
    summary={'attempted_slots':len(rows),'built_slots':sum(bool(r['build_success']) for r in rows),
             'failed_slots':sum(not r['build_success'] for r in rows),'objective':'target predicted affinity',
             'off_target_is_target_alias':bool(manifest['off_target_is_target_alias']),
             'unique_built_smiles':len({r.get('smiles') for r in rows if r['build_success'] and r.get('smiles')})}
    import math
    for field_name in ('pic50_on_upstream','pic50_on_rescore'):
        values=[r[field_name] for r in rows if isinstance(r.get(field_name),(int,float)) and math.isfinite(r[field_name])]
        summary[field_name]={'n':len(values),'mean':sum(values)/len(values) if values else None,
                             'maximum':max(values) if values else None}
    atomic_json(root/'target_result.json',summary)
    return summary


def run_target_job(command,log,environment):
    """Explicitly call the existing script; propagate its failure and streamed log."""
    env=os.environ.copy();env.update(environment);env['PYTHONUNBUFFERED']='1'
    proc=None
    try:
        with Path(log).open('w',encoding='utf-8') as stream:
            proc=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                env=env,text=True,encoding='utf-8',errors='replace',bufsize=1,
                start_new_session=os.name!='nt',
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            for line in proc.stdout:
                stream.write(line);stream.flush();print(line,end='',flush=True)
            code=proc.wait()
        if code:raise RuntimeError(f'Target generation exited with {code}; see {log}')
    except BaseException:
        if proc is not None and proc.poll() is None:
            if os.name=='nt':proc.terminate()
            else:os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                if os.name=='nt':proc.kill()
                else:os.killpg(proc.pid,signal.SIGKILL)
                proc.wait()
        raise


def _claim_target(root,cfg,row,script,journal,progress,worker):
    with journal.transaction() as state:
        record=state['targets'][row['target_id']]
        # A lost terminal marker never justifies re-generating an archived or
        # verified completed dataset. Failed markers, however, require explicit
        # human inspection/reset before a new attempt can be claimed.
        if record['state'] in ('complete','archived') and not progress.statuses(row['key']):
            progress.migrate(row['key'],'finished')
        claim=progress.claim(row['key'])
        if claim is None:return None,progress.statuses(row['key'])
        number=len(record['attempts'])+1
        job=root/'jobs'/row['key']/f'attempt_{number:03d}'
        output=root/'targets'/(row['key'] if number==1 else row['key']+f'__attempt_{number:03d}')
        command=[sys.executable,'-u',str(script),'--config',str(job/'generation.json')]
        attempt={'number':number,'config':(job/'generation.json').relative_to(root).as_posix(),
                 'log':(job/'generation.log').relative_to(root).as_posix(),'command':command,'state':'running',
                 'owner_token':claim.token,'worker':worker}
        record['attempts'].append(attempt)
        record.update(state='running',dataset=output.relative_to(root).as_posix(),owner_token=claim.token)
        return {'claim':claim,'job':job,'output':output,'attempt':number,'command':command},[]


def _complete_target(row,task,summary,journal,progress):
    with journal.transaction() as state:
        record=state['targets'][row['target_id']]
        if record.get('owner_token')!=task['claim'].token:
            raise TargetClaimConflict('Concurrent target execution: campaign ownership changed before completion')
        progress.finish(task['claim'])
        record.update(state='complete',summary=summary)
        record['attempts'][-1]['state']='complete'


def _fail_target(row,task,error,journal,progress,detail):
    with journal.transaction() as state:
        progress.fail(task['claim'],detail)
        record=state['targets'][row['target_id']]
        record.update(state='failed',error=str(error))
        for attempt in record['attempts']:
            if attempt.get('owner_token')==task['claim'].token:attempt.update(state='failed',error=str(error))


def _run_target(cfg,row,task,journal,progress,worker):
    job,output=task['job'],task['output']
    try:
        # The empty running marker has already been created and committed.
        job.mkdir(parents=True)
        input_manifest=job/'pocket_inputs.json'
        atomic_json(input_manifest,{'format':INPUT_FORMAT,'files':{role:str(Path(cfg.input_dataset)/value['path'])
                    for role,value in row['files'].items()}})
        child=_job_config(cfg,row,output,input_manifest).resolved()
        atomic_json(job/'generation.json',asdict(child))
        for value in row['files'].values():
            if digest(Path(cfg.input_dataset)/value['path'])!=value['sha256']:
                raise ValueError('Target input changed during this campaign')
        print(json.dumps({'event':'target_start','target_id':row['target_id'],'attempt':task['attempt'],
                          'worker':worker['id']}),flush=True)
        run_target_job(task['command'],job/'generation.log',cfg.runtime_environment)
        summary=_result(output,cfg,row)
        _complete_target(row,task,summary,journal,progress)
        return True
    except BaseException as error:
        detail=('target_id: '+row['target_id']+'\nworker: '+json.dumps(worker)+'\n'
                +'attempt: '+str(task['attempt'])+'\n\n--- full controller traceback ---\n'+traceback.format_exc())
        log=job/'generation.log'
        if log.is_file():
            checked_path(log,journal.root)
            detail+='\n--- complete generation stdout/stderr log ---\n'+log.read_text(encoding='utf-8',errors='replace')
        _fail_target(row,task,error,journal,progress,detail)
        print(json.dumps({'event':'target_error','target_id':row['target_id'],'error':str(error),
                          'error_file':str(progress.path(row['key'],'error'))}),flush=True)
        if not isinstance(error,Exception) or not cfg.continue_on_error:raise
        return False


def run_campaign(config,*,dry_run=False,max_targets=None,retry_failed=False,flush_archives=False):
    cfg=config.resolved();root=Path(cfg.output_dataset)
    if max_targets is not None and (type(max_targets) is not int or max_targets<1):raise ValueError('max_targets must be positive')
    catalog=discover_targets(cfg.input_dataset,manifest=cfg.target_manifest,protein_glob=cfg.protein_glob,
        protein_suffix=cfg.protein_suffix,ligand_suffix=cfg.ligand_suffix,expected_targets=cfg.expected_targets)
    if len(set(cfg.target_ids))!=len(cfg.target_ids) or set(cfg.target_ids)-{r['target_id'] for r in catalog}:
        raise ValueError('Unknown/duplicate selected target IDs')
    selected=[r for r in catalog if not cfg.target_ids or r['target_id'] in cfg.target_ids]
    script=_generator_script()
    if not script.is_file():raise FileNotFoundError('Source checkout needs scripts/generate_steer_learning.py')
    # Validate the inner config without creating its input manifest or starting inference.
    probe=root/'jobs'/selected[0]['key']/'pocket_inputs.json'
    inner=_job_config(cfg,selected[0],root/'targets'/selected[0]['key'],probe)
    import math
    if not 0<=cfg.window_start<=cfg.window_end<1 or not all(math.isfinite(t) for t in (cfg.window_start,cfg.window_end)):
        raise ValueError('Invalid score-time window')
    if any(type(v) is not int or v<1 for v in (cfg.samples,cfg.batch_size,cfg.steps)) or cfg.samples%cfg.batch_size or cfg.steps<2:
        raise ValueError('Positive complete batches and at least two integration steps required')
    if type(cfg.seed) is not int or not 0<=cfg.seed<2**32 or cfg.storage_codec not in ('none','gzip_shuffle'):
        raise ValueError('Invalid seed/storage policy')
    for path in (Path(cfg.checkpoint),Path(cfg.python_executable),Path(cfg.flowr_root)/'flowr/models/fm_pocket.py',Path(cfg.flowr_root)/'flowr/gen/generate_from_pdb_selective.py'):
        if not path.is_file():raise FileNotFoundError(path)
    plan={'format':FORMAT,'catalog_targets':len(catalog),'selected_targets':len(selected),
          'samples_per_target':cfg.samples,'batch_size':cfg.batch_size,
          'batches_per_target':cfg.samples//cfg.batch_size,'objective':'target predicted affinity','arms':['single'],
          'selection_window':[cfg.window_start,cfg.window_end],'integration_end':1.,'compress':cfg.compress,
          'archive_every_targets':cfg.archive_every,'remove_archived_targets':cfg.compress and cfg.remove_archived_targets,
          'generation_progress':str(root/'generation_progress'),'progress_format':PROGRESS_FORMAT,
          'skip_markers':['running','finished','error'],
          'generator_script':str(script),'first_job_config':asdict(inner),'targets':selected}
    if dry_run:return plan
    root.mkdir(parents=True,exist_ok=True)
    progress=TargetProgress(root,selected)
    journal=CampaignJournal(root,json.loads(json.dumps(asdict(cfg))),selected,progress,FORMAT)
    archive=TargetArchiveCoordinator(root,cfg,journal,progress)
    worker={'id':uuid.uuid4().hex,'pid':os.getpid(),'host':socket.gethostname()}
    stats={'identity':worker,'claimed':0,'finished':0,'errors':0,'skipped':0,'skipped_markers':{s:0 for s in ('running','finished','error')}}
    if retry_failed:
        print(json.dumps({'event':'retry_failed_ignored','reason':'Existing error markers always skip; review and explicitly reset a target before retrying'}),flush=True)
    archive.run()  # repair a reserved group/retirement without an inference lock
    counts=progress.counts()
    if not counts['available']:
        stats['skipped']=len(selected)
        stats['skipped_markers']={s:counts[s] for s in stats['skipped_markers']}
        queue=[]
    else:queue=selected
    for row in queue:
        if max_targets is not None and stats['claimed']>=max_targets:break
        task,markers=_claim_target(root,cfg,row,script,journal,progress,worker)
        if task is None:
            stats['skipped']+=1
            for suffix in markers:stats['skipped_markers'][suffix]+=1
            print(json.dumps({'event':'target_skipped','target_id':row['target_id'],'markers':markers}),flush=True)
            continue
        stats['claimed']+=1
        succeeded=_run_target(cfg,row,task,journal,progress,worker)
        stats['finished' if succeeded else 'errors']+=1
        archive.run()
    archive.run(flush=flush_archives)
    result=journal.summary();stats['no_available_targets']=stats['claimed']==0
    if stats['no_available_targets']:
        print(json.dumps({'event':'no_available_targets','message':'All selected targets have running, finished or error markers; no inference was started',
                          'progress':progress.counts()}),flush=True)
    result['worker']=stats
    atomic_json(progress.root/'workers'/f"{worker['id']}.json",stats)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description='Single-target Steer collection; default tar.gz after every ten completed targets')
    p.add_argument('--config');p.add_argument('--flowr-root');p.add_argument('--checkpoint')
    p.add_argument('--input-dataset');p.add_argument('--output-dataset');p.add_argument('--python',dest='python_executable')
    p.add_argument('--target-manifest');p.add_argument('--expected-targets',type=int);p.add_argument('--campaign')
    p.add_argument('--samples',type=int,help=f'Candidate slots per target (default {DEFAULT_STEER_SAMPLES})')
    p.add_argument('--batch-size',type=int,help=f'Competing particles per batch (default {DEFAULT_STEER_BATCH_SIZE})')
    p.add_argument('--steps',type=int);p.add_argument('--seed',type=int)
    p.add_argument('--window-start',type=float);p.add_argument('--window-end',type=float)
    p.add_argument('--storage-codec',choices=['none','gzip_shuffle']);p.add_argument('--archive-every',type=int);p.add_argument('--compression-level',type=int)
    p.add_argument('--compress',action=argparse.BooleanOptionalAction,default=None)
    p.add_argument('--remove-archived-targets',action=argparse.BooleanOptionalAction,default=None)
    p.add_argument('--continue-on-error',action=argparse.BooleanOptionalAction,default=None)
    p.add_argument('--max-targets',type=int)
    p.add_argument('--retry-failed',action='store_true',help='Deprecated; never bypasses existing error markers')
    p.add_argument('--flush-archives',action='store_true')
    p.add_argument('--dry-run',action='store_true');a=p.parse_args(argv)
    values=read_json(a.config) if a.config else {}
    run_keys={'config','dry_run','max_targets','retry_failed','flush_archives'}
    values.update({k:v for k,v in vars(a).items() if k not in run_keys and v is not None})
    result=run_campaign(TargetCampaignConfig(**values),dry_run=a.dry_run,max_targets=a.max_targets,
                        retry_failed=a.retry_failed,flush_archives=a.flush_archives)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if not a.dry_run and result['status'] in ('partial_failure','archive_error','markers_only'):raise SystemExit(2)


if __name__=='__main__':main()
