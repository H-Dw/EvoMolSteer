"""Sequential single-target Steer jobs, automatic configs and ten-target archives."""
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

from ..io import digest,read_json
from ..storage.transactions import atomic_json,exclusive_lock,checked_path
from ..storage.target_archive import archive_target_directories,verify_target_archive,retire_archived_directories
from ..trajectory_source import validate_input_bundle
from .steer_launcher import SteerLearningConfig,INPUT_FORMAT,verify_dataset
from .target_catalog import discover_targets,FORMAT as CATALOG_FORMAT

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
    samples:int=1000
    batch_size:int=100
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


def _emit_reports(root,state):
    totals={'targets':len(state['targets']),'successful_targets':sum(r['state'] in ('complete','archived') for r in state['targets'].values()),
            'failed_targets':sum(r['state']=='failed' for r in state['targets'].values()),'archive_groups':len(state['archives']),
            'archive_bytes':sum(a['metadata']['archive_bytes'] for a in state['archives'])}
    atomic_json(root/'results_summary.json',{'format':FORMAT,'status':state['status'],'totals':totals,
        'targets':{key:{k:v for k,v in row.items() if k in ('state','summary','dataset','archive','attempts')} for key,row in state['targets'].items()},
        'archives':state['archives']})


def _publish_group(root,cfg,state,rows):
    index=len(state['archives'])
    path=root/'archives'/f'targets_{index:04d}.tar.gz'
    directories={row['key']:root/state['targets'][row['target_id']]['dataset'] for row in rows}
    sidecar=Path(str(path)+'.json')
    ids={row['key']:row['target_id'] for row in rows}
    # Recover a published archive after a crash before collection-state update.
    if path.exists():
        verified=verify_target_archive(path,sidecar if sidecar.exists() else None)
        if verified['manifest']['targets']!={k:{'target_id':v} for k,v in ids.items()}:
            raise ValueError('Published group differs from the pending target list')
        verified.pop('manifest');metadata=verified
        if not sidecar.exists():atomic_json(sidecar,metadata)
    else:
        if Path(str(path)+'.partial').exists():
            # Preserve any interrupted container; a new packing attempt can proceed.
            abandoned=Path(str(path)+'.partial')
            n=0
            while Path(str(abandoned)+f'.abandoned_{n}').exists():n+=1
            abandoned.rename(Path(str(abandoned)+f'.abandoned_{n}'))
        metadata=archive_target_directories(directories,path,target_ids=ids,compression_level=cfg.compression_level)
    item={'path':path.relative_to(root).as_posix(),'target_ids':list(ids.values()),'directories':{k:p.relative_to(root).as_posix() for k,p in directories.items()},
          'metadata':metadata,'retirement':'pending' if cfg.remove_archived_targets else 'kept'}
    state['archives'].append(item)
    for row in rows:
        record=state['targets'][row['target_id']];record.update(state='archived',archive=item['path'])
    # Make the verified archive locator durable before retiring any working copy.
    atomic_json(root/'campaign_state.json',state)
    if cfg.remove_archived_targets:
        item['retirement']=retire_archived_directories(path,metadata,directories,owned_root=root/'targets')
        atomic_json(root/'campaign_state.json',state)
    print(json.dumps({'event':'target_group_archived','target_count':len(rows),
                      'source_bytes':metadata['source_bytes'],'archive_bytes':metadata['archive_bytes'],
                      'reduction_percent':metadata['reduction_percent'],'path':str(path)}),flush=True)


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
          'generator_script':str(script),'first_job_config':asdict(inner),'targets':selected}
    if dry_run:return plan
    state_path=root/'campaign_state.json'
    if root.exists() and any(root.iterdir()) and not state_path.is_file():raise FileExistsError('Collection output is not owned by this controller')
    root.mkdir(parents=True,exist_ok=True)
    with exclusive_lock(root/'.target_campaign.lock'):
        configuration=json.loads(json.dumps(asdict(cfg)))
        if state_path.exists():
            state=read_json(state_path)
            if state.get('format')!=FORMAT or state['config']!=configuration or state['catalog']!=selected:
                raise ValueError('Resume configuration or target input hashes changed; use a new output collection')
        else:
            state={'format':FORMAT,'config':configuration,'catalog':selected,'status':'running','archives':[],
                   'targets':{r['target_id']:{'state':'pending','attempts':[]} for r in selected}}
            atomic_json(root/'targets_resolved.json',{'format':CATALOG_FORMAT,'dataset':cfg.input_dataset,'targets':selected})
            atomic_json(state_path,state)
        for item in state['archives']:
            verify_target_archive(root/item['path'],item['metadata'])
            if item['retirement']=='pending':
                directories={k:root/p for k,p in item['directories'].items()}
                item['retirement']=retire_archived_directories(root/item['path'],item['metadata'],directories,owned_root=root/'targets')
                atomic_json(state_path,state)
        executed=0
        for row in selected:
            record=state['targets'][row['target_id']]
            if record['state']=='archived':continue
            if record['state']=='complete':
                _result(root/record['dataset'],cfg,row)
            elif record['state']=='failed' and not retry_failed:
                continue
            else:
                if max_targets is not None and executed>=max_targets:break
                executed+=1
                # A fully finished child can be collected after parent interruption.
                if record.get('dataset') and (root/record['dataset']/'learning_dataset_manifest.json').is_file() and read_json(root/record['dataset']/'learning_dataset_manifest.json').get('state')=='complete':
                    record.update(state='complete',summary=_result(root/record['dataset'],cfg,row))
                    atomic_json(state_path,state)
                else:
                    number=len(record['attempts'])+1
                    job=root/'jobs'/row['key']/f'attempt_{number:03d}';job.mkdir(parents=True)
                    output=root/'targets'/(row['key'] if number==1 else row['key']+f'__attempt_{number:03d}')
                    input_manifest=job/'pocket_inputs.json'
                    atomic_json(input_manifest,{'format':INPUT_FORMAT,'files':{role:str(Path(cfg.input_dataset)/value['path']) for role,value in row['files'].items()}})
                    child=_job_config(cfg,row,output,input_manifest).resolved()
                    path=job/'generation.json';atomic_json(path,asdict(child))
                    command=[sys.executable,'-u',str(script),'--config',str(path)]
                    attempt={'number':number,'config':path.relative_to(root).as_posix(),'log':(job/'generation.log').relative_to(root).as_posix(),'command':command,'state':'running'}
                    record['attempts'].append(attempt);record.update(state='running',dataset=output.relative_to(root).as_posix())
                    atomic_json(state_path,state)
                    print(json.dumps({'event':'target_start','target_id':row['target_id'],'attempt':number}),flush=True)
                    try:
                        for value in row['files'].values():
                            if digest(Path(cfg.input_dataset)/value['path'])!=value['sha256']:
                                raise ValueError('Target input changed during this campaign')
                        run_target_job(command,job/'generation.log',cfg.runtime_environment)
                        record.update(state='complete',summary=_result(output,cfg,row));attempt['state']='complete'
                    except BaseException as error:
                        attempt.update(state='failed',error=str(error));record.update(state='failed',error=str(error))
                        atomic_json(state_path,state);_emit_reports(root,state)
                        if not isinstance(error,Exception) or not cfg.continue_on_error:raise
                    atomic_json(state_path,state)
            waiting=[r for r in selected if state['targets'][r['target_id']]['state']=='complete']
            if cfg.compress:
                while len(waiting)>=cfg.archive_every:
                    _publish_group(root,cfg,state,waiting[:cfg.archive_every]);waiting=waiting[cfg.archive_every:]
            _emit_reports(root,state)
        waiting=[r for r in selected if state['targets'][r['target_id']]['state']=='complete']
        exhausted=all(r['state'] in ('complete','archived','failed') for r in state['targets'].values())
        if cfg.compress and waiting and (exhausted or flush_archives):_publish_group(root,cfg,state,waiting)
        state['status']='complete' if all(r['state'] in ('complete','archived') for r in state['targets'].values()) else ('partial_failure' if exhausted else 'pending')
        atomic_json(state_path,state);_emit_reports(root,state)
        return read_json(root/'results_summary.json')


def main(argv=None):
    p=argparse.ArgumentParser(description='Single-target Steer collection; default tar.gz after every ten completed targets')
    p.add_argument('--config');p.add_argument('--flowr-root');p.add_argument('--checkpoint')
    p.add_argument('--input-dataset');p.add_argument('--output-dataset');p.add_argument('--python',dest='python_executable')
    p.add_argument('--target-manifest');p.add_argument('--expected-targets',type=int);p.add_argument('--campaign')
    p.add_argument('--samples',type=int);p.add_argument('--batch-size',type=int);p.add_argument('--steps',type=int);p.add_argument('--seed',type=int)
    p.add_argument('--window-start',type=float);p.add_argument('--window-end',type=float)
    p.add_argument('--storage-codec',choices=['none','gzip_shuffle']);p.add_argument('--archive-every',type=int);p.add_argument('--compression-level',type=int)
    p.add_argument('--compress',action=argparse.BooleanOptionalAction,default=None)
    p.add_argument('--remove-archived-targets',action=argparse.BooleanOptionalAction,default=None)
    p.add_argument('--continue-on-error',action=argparse.BooleanOptionalAction,default=None)
    p.add_argument('--max-targets',type=int);p.add_argument('--retry-failed',action='store_true');p.add_argument('--flush-archives',action='store_true')
    p.add_argument('--dry-run',action='store_true');a=p.parse_args(argv)
    values=read_json(a.config) if a.config else {}
    run_keys={'config','dry_run','max_targets','retry_failed','flush_archives'}
    values.update({k:v for k,v in vars(a).items() if k not in run_keys and v is not None})
    result=run_campaign(TargetCampaignConfig(**values),dry_run=a.dry_run,max_targets=a.max_targets,
                        retry_failed=a.retry_failed,flush_archives=a.flush_archives)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if not a.dry_run and result['status']=='partial_failure':raise SystemExit(2)


if __name__=='__main__':main()
