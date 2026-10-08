"""Specified pocket inputs -> native Steer inference -> readable learning archive."""
import argparse
from dataclasses import asdict,dataclass,replace
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from ..io import digest,read_json
from ..storage.transactions import atomic_json,exclusive_lock
from ..storage.selection_dataset import verify_learning_batch


INPUT_FORMAT='evomolsteer.pocket_inputs.v1'
DATASET_FORMAT='evomolsteer.selection_learning_dataset.v1'
ROLES=('target_protein','target_ligand','off_target_protein','off_target_ligand')
STAGED_NAMES=dict(zip(ROLES,('target_protein.pdb','target_ligand.sdf','off_target_protein.pdb','off_target_ligand.sdf')))


@dataclass(frozen=True)
class SteerLearningConfig:
    flowr_root:str
    checkpoint:str
    input_dataset:str
    output_dataset:str
    python_executable:str=sys.executable
    input_manifest:str|None=None
    campaign:str='steer_learning'
    samples:int=1000
    batch_size:int=100
    steps:int=100
    window_start:float=0.0
    window_end:float=0.5
    seed:int=42
    arms:tuple[str,...]=('single',)
    storage_codec:str='none'
    capture_bond_probabilities:bool=False

    def resolved(self):
        py=shutil.which(self.python_executable) or self.python_executable
        cfg=replace(self,**{k:str(Path(getattr(self,k)).expanduser().resolve()) for k in
                            ('flowr_root','checkpoint','input_dataset','output_dataset')},
                    python_executable=str(Path(py).expanduser().resolve()),arms=tuple(self.arms))
        if self.input_manifest:
            cfg=replace(cfg,input_manifest=str(Path(self.input_manifest).expanduser().resolve()))
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',cfg.campaign):
            raise ValueError('campaign must be a simple directory name')
        for key in ('samples','batch_size','steps'):
            if type(getattr(cfg,key)) is not int or getattr(cfg,key)<1:
                raise ValueError(key+' must be a positive integer')
        if cfg.steps<2 or cfg.samples%cfg.batch_size:
            raise ValueError('At least two steps and complete candidate batches are required')
        if type(cfg.seed) is not int or not 0<=cfg.seed<2**32:
            raise ValueError('A uint32 master seed is required')
        if not all(math.isfinite(t) for t in (cfg.window_start,cfg.window_end)) or not 0<=cfg.window_start<=cfg.window_end<1:
            raise ValueError('Selection bounds must satisfy 0 <= start <= end < 1')
        if not cfg.arms or len(set(cfg.arms))!=len(cfg.arms) or set(cfg.arms)-{'single','joint','unguided'} or not set(cfg.arms)&{'single','joint'}:
            raise ValueError('At least one Steer arm is required; allowed arms: single,joint,unguided')
        if cfg.storage_codec not in ('none','gzip_shuffle') or type(cfg.capture_bond_probabilities) is not bool:
            raise ValueError('Invalid storage/probability capture policy')
        for relative in ('flowr/models/fm_pocket.py','flowr/gen/generate_from_pdb_selective.py'):
            if not (Path(cfg.flowr_root)/relative).is_file():
                raise FileNotFoundError('FLOWR.ROOT checkout missing '+relative)
        for key in ('checkpoint','python_executable'):
            if not Path(getattr(cfg,key)).is_file():
                raise FileNotFoundError(getattr(cfg,key))
        source,target=Path(cfg.input_dataset),Path(cfg.output_dataset)
        if not source.is_dir():
            raise NotADirectoryError(source)
        if target==source or target.is_relative_to(source) or source.is_relative_to(target):
            raise ValueError('Use disjoint input and output datasets')
        resolve_inputs(cfg)
        return cfg


def resolve_inputs(cfg):
    """Named roles from a portable manifest, or the historical four-file dataset."""
    root=Path(cfg.input_dataset)
    manifest=Path(cfg.input_manifest) if cfg.input_manifest else next(
        (p for p in (root/'pocket_inputs.json',root/'inputs/pocket_inputs.json') if p.is_file()),None)
    if manifest is not None:
        spec=read_json(manifest)
        if spec.get('format')!=INPUT_FORMAT or set(spec.get('files',{}))-set(ROLES):
            raise ValueError('Unsupported pocket input manifest/roles')
        roles={k:(Path(v).expanduser() if Path(v).is_absolute() else manifest.parent/v).resolve()
               for k,v in spec['files'].items()}
    else:
        from .launcher import INPUT_FILES
        folder=root/'inputs' if (root/'inputs').is_dir() else root
        roles=dict(zip(ROLES,(folder/name for name in INPUT_FILES)))
        if not any(roles[k].is_file() for k in ROLES[2:]):
            roles={k:roles[k] for k in ROLES[:2]}
    if not all(k in roles for k in ROLES[:2]):
        raise ValueError('Target protein PDB and reference ligand SDF are required')
    optional={k for k in ROLES[2:] if k in roles}
    if optional and optional!=set(ROLES[2:]):
        raise ValueError('Provide both off-target files or neither')
    duplicated=not optional
    if duplicated:
        if 'joint' in cfg.arms:
            raise ValueError('Joint selectivity optimization requires an explicit off-target pocket')
        roles.update(off_target_protein=roles['target_protein'],off_target_ligand=roles['target_ligand'])
    for role,path in roles.items():
        if not path.is_file() or path.stat().st_size==0:
            raise FileNotFoundError(str(role)+': '+str(path))
    return roles,duplicated


def verify_dataset(root,campaign,*,arms=None,samples=None):
    root=Path(root).resolve()
    folder=root/'results'/campaign
    completion=read_json(folder/'COMPLETE.json')
    if completion.get('status')!='complete':
        raise ValueError('Generation is not complete')
    cfg=read_json(folder/'generation_request.json')
    arms=tuple(arms or cfg['arms'])
    samples=cfg['samples'] if samples is None else samples
    expected=set(range(samples//cfg['batch_size']))
    batches=[]
    for arm in arms:
        paths=sorted((folder/arm).glob('batch_*'))
        if {int(p.name.split('_')[1]) for p in paths}!=expected:
            raise ValueError('Missing or unexpected batches for '+arm)
        count=0
        for batch in paths:
            if not (batch/'COMPLETE.json').exists():
                raise ValueError('Incomplete batch '+str(batch))
            result=verify_learning_batch(batch)
            rows=read_json(batch/'final_records.json')
            if len(rows)!=result['particles'] or {r['slot'] for r in rows}!=set(range(result['particles'])):
                raise ValueError('Final records omitted or duplicated slots')
            count+=len(rows)
            batches.append({'arm':arm,'batch':int(batch.name.split('_')[1]),**result})
        if count!=samples:
            raise ValueError('Generated sample count differs from request')
    return {'verified':True,'batches':batches,'records':samples*len(arms)}


def launch(cfg,*,dry_run=False):
    cfg=cfg.resolved()
    roles,duplicate=resolve_inputs(cfg)
    root=Path(cfg.output_dataset)
    folder=root/'results'/cfg.campaign
    request=folder/'generation_request.json'
    command=[cfg.python_executable,'-u','-m','evomolsteer.generation.steer_worker','--request',str(request)]
    plan={'config':asdict(cfg),'command':command,'cwd':cfg.flowr_root,'input_roles':{k:str(v) for k,v in roles.items()},
          'off_target_is_target_alias':duplicate,'mode':'selective_smc','gradient_guidance':False,
          'continue_to_time':1.0,'archive_container':'directory','raw_trajectory_npz_created':False,
          'retained':'window candidates + full lineage + native terminal + scores/failures + input/provenance'}
    if root.exists() and any(root.iterdir()):
        raise FileExistsError('Use a new empty output dataset; original data are not overwritten')
    if dry_run:
        return plan
    root.mkdir(parents=True,exist_ok=True)
    with exclusive_lock(root/'.steer_launch.lock'):
        inputs=root/'inputs'
        inputs.mkdir()
        files={}
        hashes={}
        for role,source in roles.items():
            if duplicate and role in ROLES[2:]:
                files[role]=files[role.removeprefix('off_')]
                continue
            dest=inputs/STAGED_NAMES[role]
            original=digest(source)
            shutil.copyfile(source,dest)
            if digest(dest)!=original or digest(source)!=original:
                raise ValueError('Input changed during copying')
            files[role]=dest.name
            hashes[dest.name]={'sha256':original,'bytes':dest.stat().st_size}
        atomic_json(inputs/'pocket_inputs.json',{'format':INPUT_FORMAT,'files':files,
                    'off_target_is_target_alias':duplicate,'input_sha256':hashes})
        folder.mkdir(parents=True)
        atomic_json(request,asdict(cfg))
        state=root/'learning_dataset_manifest.json'
        manifest={'format':DATASET_FORMAT,'state':'starting','campaign':cfg.campaign,'config':asdict(cfg),
                  'input_files':files,'off_target_is_target_alias':duplicate,'input_sha256':hashes,
                  'checkpoint_sha256':digest(cfg.checkpoint),
                  'flowr_sampler_sha256':digest(Path(cfg.flowr_root)/'flowr/models/fm_pocket.py'),
                  'command':command,'archive_container':'directory','gradient_guidance':False,
                  'selection_window':[cfg.window_start,cfg.window_end],
                  'restart_capability':'not a restart/RNG archive','window_definition':'inclusive scoring time',
                  'controller_sha256':digest(Path(__file__).with_name('controller.py')),
                  'trace_sha256':digest(Path(__file__).with_name('steer_trace.py'))}
        atomic_json(state,manifest)
        env=os.environ.copy()
        env['PYTHONPATH']=os.pathsep.join([str(Path(__file__).resolve().parents[2]),cfg.flowr_root,env.get('PYTHONPATH','')])
        env['PYTHONUNBUFFERED']='1'
        proc=None
        try:
            with (folder/'generation.log').open('w',encoding='utf-8') as log:
                proc=subprocess.Popen(command,cwd=cfg.flowr_root,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                    text=True,encoding='utf-8',errors='replace',bufsize=1,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                manifest.update(state='running',pid=proc.pid)
                atomic_json(state,manifest)
                for line in proc.stdout:
                    log.write(line);log.flush();print(line,end='',flush=True)
                code=proc.wait()
            if code!=0:
                raise RuntimeError(f'FLOWR exited with {code}; see {folder / "generation.log"}')
            verification=verify_dataset(root,cfg.campaign,arms=cfg.arms,samples=cfg.samples)
            context=list(inputs.iterdir())+list(folder.glob('frame_batch_*.json'))
            context += [folder/name for name in ('config.json','runtime.json','generation_request.json','final_records.json','COMPLETE.json')]
            context += list((folder/'provenance').glob('*.py'))
            manifest.update(state='complete',verification=verification,
                copied_files={p.relative_to(root).as_posix():{'sha256':digest(p),'bytes':p.stat().st_size}
                              for p in sorted(set(context)) if p.is_file()},
                batches=[{'path':p.relative_to(root).as_posix(),'sha256':digest(p)}
                         for arm in cfg.arms for p in sorted((folder/arm).glob('batch_*/learning_manifest.json'))],
                files_bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()))
            atomic_json(state,manifest)
            # Include the final manifest itself, without a stale pre-publication size.
            for _ in range(3):
                size=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
                if manifest['files_bytes']==size:break
                manifest['files_bytes']=size
                atomic_json(state,manifest)
            return manifest
        except BaseException as error:
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=15)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
            manifest.update(state='failed',error=str(error))
            atomic_json(state,manifest)
            raise


def main(argv=None):
    p=argparse.ArgumentParser(description='Native Steer sampling with a directly readable selection-learning directory')
    p.add_argument('--config')
    p.add_argument('--flowr-root');p.add_argument('--checkpoint');p.add_argument('--input-dataset');p.add_argument('--output-dataset')
    p.add_argument('--python',dest='python_executable');p.add_argument('--input-manifest');p.add_argument('--campaign')
    p.add_argument('--samples',type=int);p.add_argument('--batch-size',type=int);p.add_argument('--steps',type=int)
    p.add_argument('--window-start',type=float);p.add_argument('--window-end',type=float);p.add_argument('--seed',type=int)
    p.add_argument('--arms',help='Comma separated single,joint,unguided; default single')
    p.add_argument('--storage-codec',choices=['none','gzip_shuffle'])
    p.add_argument('--capture-bond-probabilities',action=argparse.BooleanOptionalAction,default=None)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args(argv)
    values=read_json(a.config) if a.config else {}
    values.update({k:v for k,v in vars(a).items() if k not in ('config','dry_run') and v is not None})
    if isinstance(values.get('arms'),str):values['arms']=tuple(values['arms'].split(','))
    import json
    print(json.dumps(launch(SteerLearningConfig(**values),dry_run=a.dry_run),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
