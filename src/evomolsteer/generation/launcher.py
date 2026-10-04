"""Typed dataset interface and shell-free launch of a separate FLOWR runtime."""
import argparse
from dataclasses import asdict,dataclass,replace
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from ..io import read_json,digest
from ..storage.transactions import atomic_json,exclusive_lock

INPUT_FILES=('3PE1_protein_aligned.pdb','3PE1_ligand_aligned.sdf',
             '6KHF_protein_aligned.pdb','6KHF_ligand_aligned.sdf')


@dataclass(frozen=True)
class FlowrRunConfig:
    flowr_root:str
    checkpoint:str
    input_dataset:str
    output_dataset:str
    python_executable:str=sys.executable
    campaign:str='steer_streaming'
    samples:int=1000
    batch_size:int=50
    steps:int=100
    window_start:float=0.
    window_end:float=.5
    seed:int=42
    arms:tuple[str,...]=('unguided','single','joint')
    verify_passive:bool=True

    def resolved(self):
        py=shutil.which(self.python_executable) or self.python_executable
        cfg=replace(self,**{k:str(Path(getattr(self,k)).expanduser().resolve()) for k in
                           ['flowr_root','checkpoint','input_dataset','output_dataset']},
                    python_executable=str(Path(py).expanduser().resolve()),arms=tuple(self.arms))
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',cfg.campaign):
            raise ValueError('campaign must be one simple directory name')
        for key in ['samples','batch_size','steps']:
            if type(getattr(cfg,key)) is not int or getattr(cfg,key)<1:raise ValueError(key+' must be a positive integer')
        if cfg.steps<2 or type(cfg.seed) is not int or not 0<=cfg.seed<2**32:
            raise ValueError('At least two steps and a valid uint32 seed are required')
        if not all(math.isfinite(x) for x in (cfg.window_start,cfg.window_end)) or not 0<=cfg.window_start<=cfg.window_end<=1:
            raise ValueError('Invalid selection window')
        if not cfg.arms or len(set(cfg.arms))!=len(cfg.arms) or set(cfg.arms)-{'unguided','single','joint'}:
            raise ValueError('Invalid or duplicated sampling arms')
        if type(cfg.verify_passive) is not bool:raise TypeError('verify_passive must be boolean')
        for name in ['flowr/models/fm_pocket.py','flowr/gen/generate_from_pdb_selective.py']:
            if not (Path(cfg.flowr_root)/name).is_file():raise FileNotFoundError('FLOWR.ROOT checkout missing '+name)
        for name in ['checkpoint','python_executable']:
            if not Path(getattr(cfg,name)).is_file():raise FileNotFoundError(getattr(cfg,name))
        if not Path(cfg.input_dataset).is_dir():raise NotADirectoryError(cfg.input_dataset)
        for name in INPUT_FILES:
            if not (cfg.input_directory/name).is_file():raise FileNotFoundError(cfg.input_directory/name)
        return cfg

    @property
    def input_directory(self):
        p=Path(self.input_dataset)
        return p/'inputs' if (p/'inputs').is_dir() else p


def command_for(cfg,request):
    return [cfg.python_executable,'-u','-m','evomolsteer.generation.worker','--request',str(request)]


def launch(config,*,dry_run=False):
    """Return the run manifest. A nonzero model process never counts as success."""
    cfg=config.resolved()
    root=Path(cfg.output_dataset);campaign=root/'results'/cfg.campaign
    request=campaign/'generation_request.json'
    command=command_for(cfg,request)
    if dry_run:
        return {'config':asdict(cfg),'command':command,'cwd':cfg.flowr_root,'mode':'selective_smc',
                'online_storage':True,'raw_trajectory_npz_created':False}
    if campaign.exists():raise FileExistsError('Use a new campaign directory: '+str(campaign))
    root.mkdir(parents=True,exist_ok=True)
    with exclusive_lock(root/('.launch_'+cfg.campaign+'.lock')):
        if campaign.exists():raise FileExistsError(campaign)
        # Validate existing inputs before copying anything into a reused dataset.
        incoming=[p for p in sorted(cfg.input_directory.iterdir()) if p.is_file()]
        for source in incoming:
            target=root/'inputs'/source.name
            if target.exists() and digest(source)!=digest(target):
                raise ValueError('Output dataset has different pocket inputs: '+str(target))
        (root/'inputs').mkdir(exist_ok=True)
        input_hashes={}
        for source in incoming:
            target=root/'inputs'/source.name
            if not target.exists():shutil.copyfile(source,target)
            input_hashes[source.name]=digest(target)
        campaign.mkdir(parents=True)
        atomic_json(request,asdict(cfg))
        manifest={'format':'evomolsteer.generation.v1','status':'starting','config':asdict(cfg),
                  'command':command,'input_sha256':input_hashes,'checkpoint_sha256':digest(cfg.checkpoint),
                  'flowr_sampler_sha256':digest(Path(cfg.flowr_root)/'flowr/models/fm_pocket.py'),
                  'controller_sha256':digest(Path(__file__).with_name('controller.py')),
                  'output_dataset':str(root),'mode':'selective_smc','delete_policy':'verified stages only; no raw NPZ is created'}
        state=campaign/'generation_manifest.json';atomic_json(state,manifest)
        env=os.environ.copy()
        src=Path(__file__).resolve().parents[2]
        env['PYTHONPATH']=os.pathsep.join([str(src),cfg.flowr_root,env.get('PYTHONPATH','')])
        env['PYTHONUNBUFFERED']='1'
        proc=None
        try:
            with (campaign/'generation.log').open('w',encoding='utf-8') as log:
                proc=subprocess.Popen(command,cwd=cfg.flowr_root,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                    text=True,encoding='utf-8',errors='replace',bufsize=1,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                manifest.update(status='running',pid=proc.pid);atomic_json(state,manifest)
                for line in proc.stdout:
                    log.write(line);log.flush();print(line,end='',flush=True)
                code=proc.wait()
            if code!=0:raise RuntimeError(f'FLOWR.ROOT exited with {code}; see {campaign / "generation.log"}')
            if not (campaign/'COMPLETE.json').exists():raise RuntimeError('Worker exited without a completed campaign')
            completed=read_json(campaign/'COMPLETE.json')
            if completed.get('records')!=cfg.samples*len(cfg.arms):raise RuntimeError('Generated record count disagrees with request')
            for arm in cfg.arms:
                batches=sorted((campaign/arm).glob('batch_*'))
                if not batches:raise RuntimeError('Missing generated arm '+arm)
                for batch in batches:
                    record=read_json(batch/'trajectory.manifest.json')
                    if record['state']!='complete' or (batch/'trajectory.npz').exists() or (batch/'trajectory.stages').exists():
                        raise RuntimeError('Incomplete trajectory retirement: '+str(batch))
                    if digest(batch/'trajectory.h5')!=record['final']['package_sha256']:
                        raise RuntimeError('Final trajectory checksum changed')
            manifest.update(status='complete',exit_code=code,completion=completed)
            atomic_json(state,manifest)
            return manifest
        except BaseException as error:
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=15)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
            manifest.update(status='failed',error=str(error),exit_code=proc.returncode if proc is not None else None)
            atomic_json(state,manifest)
            raise


def main(argv=None):
    p=argparse.ArgumentParser(description='FLOWR.ROOT Steer sampling with per-step lossless storage')
    p.add_argument('--config',help='JSON containing FlowrRunConfig fields')
    p.add_argument('--flowr-root');p.add_argument('--python',dest='python_executable')
    p.add_argument('--checkpoint');p.add_argument('--input-dataset');p.add_argument('--output-dataset')
    p.add_argument('--campaign');p.add_argument('--samples',type=int);p.add_argument('--batch-size',type=int)
    p.add_argument('--steps',type=int);p.add_argument('--window-start',type=float);p.add_argument('--window-end',type=float)
    p.add_argument('--seed',type=int);p.add_argument('--arms',help='Comma-separated unguided,single,joint')
    p.add_argument('--verify-passive',action=argparse.BooleanOptionalAction,default=None)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args(argv);values=read_json(a.config) if a.config else {}
    values.update({k:v for k,v in vars(a).items() if k not in ('config','dry_run') and v is not None})
    if isinstance(values.get('arms'),str):values['arms']=tuple(values['arms'].split(','))
    import json
    result=launch(FlowrRunConfig(**values),dry_run=a.dry_run)
    print(json.dumps(result,ensure_ascii=False,indent=2))
