"""Read completed populations and pair by frozen logical seed/slot; no sampling."""
from pathlib import Path
import json, hashlib, gzip, tarfile, datetime, statistics
import numpy as np
import torch

ROOT=Path('/workspace/agenticgen_gradient_scope_20261101')
OUT=ROOT/'monitoring/20261101_0800'
def read(p):return json.loads(p.read_text())
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=True,indent=2)+'\n')
def measure(folder):
 a=read(folder/'aggregate.json');rows=read(folder/'records.json');c=read(folder/'CHEMICAL_CONDITION.json')
 with gzip.open(folder/'terminal.pt.gz','rb') as f:t=torch.load(f,map_location='cpu',weights_only=False)
 ass=np.array([r['condition_generated_slots'] for r in rows]);b=np.arange(len(rows))[:,None]
 at=t['atomics'].argmax(-1).numpy()[b,ass];ch=t['charges'].argmax(-1).numpy()[b,ass]
 bo=t['bonds'].argmax(-1).numpy()[np.arange(len(rows))[:,None,None],ass[:,:,None],ass[:,None,:]]
 feat=c['features'];ac=np.asarray(feat['atomics']).argmax(-1);cc=np.asarray(feat['charges']).argmax(-1);bc=np.asarray(feat['bonds']).argmax(-1)
 chemistry=dict(denominator=len(rows),atomic_exact=int((at==ac).all(-1).sum()),charge_exact=int((ch==cc).all(-1).sum()),induced_bonds_exact=int((bo==bc).all((1,2)).sum()),all_exact=int(((at==ac).all(-1)&(ch==cc).all(-1)&(bo==bc).all((1,2))).sum()),mapping='terminal geometric Hungarian; chemical identity is not guaranteed')
 q=read(folder/'QUALITY_COMPLETE.json') if (folder/'QUALITY_COMPLETE.json').exists() else None
 qr=read(folder/'quality.json') if q else []
 checks={}
 for row in qr:
  for k,v in row.get('quality',{}).get('posebusters_checks',{}).items():
   d=checks.setdefault(k,dict(evaluable=0,passed=0));d['evaluable']+=1;d['passed']+=int(bool(v))
 plip=[r['plip_recovery'] for r in qr if r.get('quality',{}).get('plip_evaluable') and isinstance(r.get('plip_recovery'),(float,int))]
 return dict(path=str(folder),aggregate=a,chemistry=chemistry,quality=q,posebusters_components=checks,plip_recovery_mean=statistics.mean(plip) if plip else None,plip_recovery_denominator=len(plip),records=rows)

plan=read(ROOT/'PLAN.json');assert read(ROOT/'SUITE_FINISHED.json')['all_quality_terminal']
jobs=[];baselines={};receipts={}
for j in plan['jobs']:
 cfg=read(Path(j['config']));root=Path(cfg['root']);acc=read(root/'acceptance/COMPLETE.json');scope=read(root/'GRADIENT_SCOPE_VERIFIED.json')
 assert scope['population']==50 and scope['steps']==100 and all(scope[k] for k in ['graph_radius_recomputed_exact','outside_extra_increment_exact_zero','native_motion_outside_scope_unchanged','selected_full_vjp_unchanged'])
 for checks in acc['checks'].values():
  b=checks['candidate_batch_equivalence'];assert b['exact_equivalence'] and b['full_steps']==100 and b['capacity_probe_success']==50
 data=measure(root/'fixed_population');assert data['quality']['checked_attempts']==10 and data['quality']['not_selected']==40
 key=j['teacher']
 if key not in baselines:
  br=Path(cfg['fixed_scale_reference']['baseline_root']);domain=read(br/'calibration/domains.json')[key];assert domain['evidence_revision']==0
  baselines[key]=measure(br/domain['initial_source'])
 base=baselines[key];bm={(r['logical_seed'],r['logical_slot']):r for r in base['records']}
 changes=[r['condition_coordinate_root_mean_square_deviation_angstrom']-bm[(r['logical_seed'],r['logical_slot'])]['condition_coordinate_root_mean_square_deviation_angstrom'] for r in data['records']]
 jobs.append(dict(id=j['id'],teacher=key,hops=j['hops'],scale=j['schedule'],scope_verification=scope,measurement=data,paired_delta_vs_baseline=dict(population=len(changes),mean=statistics.mean(changes),median=statistics.median(changes),improved=sum(x<0 for x in changes),worse=sum(x>0 for x in changes),equal=sum(x==0 for x in changes)),source_hashes={n:sha(root/n) for n in ['acceptance/COMPLETE.json','GRADIENT_SCOPE_VERIFIED.json','SAMPLING_FINISHED.json','attempts.json']}))
summary=dict(timestamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),science_commit=plan['commit'],formal_attempts=300,new_calibration_attempts=0,llm_calls=0,iterations=0,old_preserved_attempts=5170,cumulative_attempts=5470,baselines=baselines,jobs=jobs,paired_reproduction_not_independent_validation=True,quality_not_extrapolated=True)
write(OUT/'results.json',summary)
for p in sorted(ROOT.rglob('*')):
 if p.is_file() and not p.is_symlink() and (p.suffix in ['.json','.csv','.jsonl','.sdf'] or p.name.endswith('.log')) and 'launcher.log' not in p.name:
  if 'monitoring' in p.relative_to(ROOT).parts and p.name not in ['results.json','worker_cleanup_before.json','worker_cleanup_complete.json']:continue
  receipts[str(p.relative_to(ROOT))]=sha(p)
dest=Path('/root/private_data/uniguide_gradient_scope_20261101/audits/scope-47db1bf1/safe_results.tar.gz')
with tarfile.open(dest,'x:gz') as t:
 for rel in receipts:t.add(ROOT/rel,arcname=rel,recursive=False)
write(OUT/'safe_results_receipt.json',dict(path=str(dest),sha256=sha(dest),bytes=dest.stat().st_size,files=len(receipts),source_file_sha256=receipts))
print(json.dumps(dict(results=str(OUT/'results.json'),safe_results_archive=str(dest),sha256=sha(dest),bytes=dest.stat().st_size,files=len(receipts),jobs=[dict(id=j['id'],rmsd=j['measurement']['aggregate']['condition_coordinate_root_mean_square_deviation_angstrom'],built=j['measurement']['aggregate']['built'],paired=j['paired_delta_vs_baseline'],chemistry=j['measurement']['chemistry'],quality=j['measurement']['quality'],scope=j['scope_verification']) for j in jobs]),ensure_ascii=True))
