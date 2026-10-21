import csv
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(os.environ.get('PLIP_GUIDANCE_ROOT', '/opt/MolSteer/generated_datasets/plip_guidance_5nn4_5ost_s42_20261020'))
HIST = Path('/opt/MolSteer/generated_datasets/analysis_hiqbind_examples_20261020/examples')
FLOWR = Path('/opt/MolSteer/flowr_root')
REPO = Path(__file__).resolve().parents[4]
CHECKPOINT = FLOWR/'checkpoints/flowr_root_v2.ckpt'
EXPECTED_CHECKPOINT = 'f28e863b2b208718f3d3f85f09837c2f907a71123436db6f98a25d2f1858b6a0'
EXPECTED_SAMPLER = 'beaace3ab92a8377bc8eab95333c8f40ce370ca9224f7a78927c5a424bb38426'
SEEDS = [42,100045,200048,300051,400054]
TARGETS = ['5nn4_SC2_A_1017','5ost_AEZ_A_915']
TEACHER_IDS = {
 TARGETS[0]:['single_b002_final_005','single_b004_final_009','single_b002_final_007','single_b001_final_000','single_b001_final_014'],
 TARGETS[1]:['single_b000_final_016','single_b000_final_015','single_b000_final_000','single_b001_final_002','single_b001_final_006']}
BASELINE_DATASETS = {
 'steer':'/opt/MolSteer/generated_datasets/hiqbind_steer_learning_s100_b20_w050',
 'unguided':'/opt/MolSteer/generated_datasets/hiqbind_unguided_learning_s100_b20'}

def read(path):
    return json.loads(Path(path).read_text())

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    os.replace(temp,path)

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def json_digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def array_digest(a):
    import numpy as np
    a=np.asarray(a)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()

def package_version(name):
    import importlib,importlib.metadata
    try:return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        module=importlib.import_module(name)
        return str(getattr(module,'__version__','unversioned-source'))

def generation_fingerprint():
    return {name:digest(Path(__file__).parent/name) for name in ['common.py','reward.py','native.py']}

def campaign(arm,target):
    return HIST/arm/target/'results'/('single_w050' if arm=='steer' else 'unguided_w050')

def batch_folder(arm,target,batch):
    return campaign(arm,target)/('single' if arm=='steer' else 'unguided')/f'batch_{batch:03d}'

def protein(target):
    return HIST/'steer'/target/'inputs/target_protein.pdb'

def csv_write(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for row in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})

def parameters():
    return {'schema':'plip-local-guidance-1','targets':TARGETS,'teacher_ids':TEACHER_IDS,
        'baseline_datasets':BASELINE_DATASETS,'batch_seeds':SEEDS,'master_seed':42,'batches':5,
        'batch_size':20,'samples_per_group':100,'steps':100,'sde':True,'coord_noise_level':.2,
        'sampler_sha256':EXPECTED_SAMPLER,'checkpoint_sha256':EXPECTED_CHECKPOINT,
        'guidance_window':[0,1],'modes':['current','endpoint'],'schedules':['fixed_0.3','fixed_0.5','fixed_1.0','warmup'],
        'feature_sets':['C','A','B','Q','CA','CAB','CABQ'],'feature_weights':{'C':1,'A':1,'B':1,'Q':1},
        'max_atom_step_A':.025,'coordinate_scale_A':1.,'pseudo_huber_delta':1.,
        'warmup':'lambda[k] = k/99, k=0..99','gradient_support':'all ligand coordinates',
        'assignment':'geometry-only injective Hungarian, detached, core plus covalent one-hop context',
        'bond_loss':'core-incident unordered pairs; mean of nonempty bonded and nonbonded CE means',
        'quality_tolerance':.05,'min_recovery_gain':.05,'reuse_baselines':True,'zero_guidance_groups':0,
        'maximum_new_candidates':14000,'maximum_new_groups':140,'replication':'exploratory historical seed set only'}
