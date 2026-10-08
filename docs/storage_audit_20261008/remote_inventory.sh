set -eu
root=/root/private_data/MolSteer/flowr_root/experiments/ck2_clk3_lineage_20261003
PYTHONPATH=/root/private_data/MolSteer/flowr_root/experiments/evomolsteer_online_20261004/runtime_deps /opt/miniforge3/envs/molsteer-flowr-dtk/bin/python - <<'PY'
from pathlib import Path
from collections import defaultdict
import hashlib,json,os,shutil
import h5py
r=Path('/root/private_data/MolSteer/flowr_root/experiments/ck2_clk3_lineage_20261003')
files=[];groups=defaultdict(lambda:{'files':0,'bytes':0,'allocated_bytes':0})
for p in sorted(r.rglob('*')):
 if not p.is_file():continue
 st=p.stat();rel=p.relative_to(r).as_posix()
 row={'path':rel,'bytes':st.st_size,'allocated_bytes':st.st_blocks*512}
 files.append(row)
 key='/'.join(p.relative_to(r).parts[:3])
 groups[key]['files']+=1;groups[key]['bytes']+=st.st_size;groups[key]['allocated_bytes']+=st.st_blocks*512
single=r/'results/main1000_w050/single'
trajectories=[]
for p in sorted(single.glob('batch_*/trajectory*')):
 if not p.is_file() or p.suffix!='.h5':continue
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 fields=[]
 with h5py.File(p,'r') as f:
  attrs={k:str(v) for k,v in f.attrs.items()}
  for name,g in f['fields'].items():
   ds=[]
   def add(n,obj):
    if isinstance(obj,h5py.Dataset):ds.append({'name':n,'shape':list(obj.shape),'dtype':str(obj.dtype),'storage_bytes':obj.id.get_storage_size(),'compression':obj.compression})
   g.visititems(add)
   fields.append({'name':name,'attrs':{k:str(v) for k,v in g.attrs.items()},'datasets':ds,'payload_bytes':sum(x['storage_bytes'] for x in ds)})
 trajectories.append({'path':p.relative_to(r).as_posix(),'bytes':p.stat().st_size,'sha256':h.hexdigest(),'attrs':attrs,'fields':fields})
out={'server':'ksai.scnet.cn:10544','root':str(r),'logical_bytes':sum(x['bytes'] for x in files),'allocated_bytes':sum(x['allocated_bytes'] for x in files),'files_count':len(files),'groups':dict(groups),'files':files,'single_trajectories':trajectories,'disk_free_bytes':shutil.disk_usage(r).free,'units':'bytes; directory inode blocks excluded'}
dest=Path('/tmp/evomolsteer_storage_audit_20261008.json');dest.write_text(json.dumps(out,indent=2))
print(json.dumps({'logical_bytes':out['logical_bytes'],'allocated_bytes':out['allocated_bytes'],'files':len(files),'groups':dict(groups),'single_trajectories':len(trajectories),'report':str(dest),'disk_free_bytes':out['disk_free_bytes']},indent=2))
PY
