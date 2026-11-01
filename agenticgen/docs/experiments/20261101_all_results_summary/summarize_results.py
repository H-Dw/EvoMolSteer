"""Read-only CPU analysis of saved populations; never load a generation model."""
from pathlib import Path
import argparse, collections, datetime, gzip, hashlib, io, json, math, statistics
import numpy as np

SKIP={'sources','acceptance','sample_shards','initials','random_stream_reference','profiles','monitoring','inputs','context','teachers','engineering_audits','provenance'}
ROOT_NAMES={
 'agenticgen_absolute_coordinate_20261029':'绝对坐标初次验收',
 'agenticgen_absolute_coordinate_restart_20261029':'绝对坐标重启标定',
 'agenticgen_absolute_coordinate_uncapped_20261030':'取消增量限幅标定',
 'agenticgen_absolute_coordinate_adaptive_20261030':'自适应括区标定',
 'agenticgen_absolute_coordinate_batch50_20261030':'五十候选并行重测',
 'agenticgen_chem_inpainting_20261027':'化学状态硬注入',
 'agenticgen_joint_inpainting_20261027':'联合坐标化学覆盖',
 'agenticgen_lambda_preexperiment_20261028':'内部距离尺度标定',
 'agenticgen_chemical_interact_20261029':'化学互作初次验收',
 'agenticgen_chemical_interact_graphfix_20261029':'化学互作图修复',
 'agenticgen_chemical_interact_20261029_graphfix':'化学互作图修复',
 'agenticgen_chemical_interact_sum_20261029':'化学互作求和损失',
 'agenticgen_late_window_20261028':'晚期引导窗口',
 'agenticgen_time_window_20261028':'完整区间与内部窗口',
}
def read(p, default=None):return json.loads(p.read_text()) if p.exists() else default
def sha(p):
 with p.open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=True,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def finite(v):return isinstance(v,(int,float,np.number)) and not isinstance(v,bool) and math.isfinite(float(v))

def flags(values, metric):
 """Descriptive only; at least 10, at most 10%, no removal of nonfinite values."""
 x=np.array([float(v) for v in values if finite(v)],dtype=float)
 result=[False]*len(values)
 if len(x)<10:return result,True,{}
 q1,q3=np.quantile(x,[.25,.75],method='linear');span=3*(q3-q1)
 lo,hi=(-math.inf,q3+max(span,1.)) if metric=='rmsd' else (q1-span,q3+span)
 result=[bool(finite(v)and(float(v)<lo or float(v)>hi))for v in values]
 eligible=sum(result)/len(x)<=.1+1e-12
 return result,eligible,dict(lower=None if math.isinf(lo)else float(lo),upper=float(hi),excluded=sum(result),finite=len(x))

def stats(values, metric, excluded=None, eligible=True):
 good=[(i,float(v))for i,v in enumerate(values)if finite(v)]
 if not good:return dict(n=0,missing=len(values),mean=None,best=None,filtered_mean=None,top10_mean=None,top10_n=0,excluded=0)
 vals=[v for _,v in good];reverse=metric in ['pic50','rescore'];top=sorted(vals,reverse=reverse)[:max(1,math.ceil(.1*len(vals)))];excluded=excluded or [False]*len(values)
 kept=[v for i,v in good if not excluded[i]]
 return dict(n=len(vals),missing=len(values)-len(vals),mean=statistics.mean(vals),best=(max(vals)if reverse else min(vals)),filtered_mean=statistics.mean(kept)if kept and eligible else None,top10_mean=statistics.mean(top),top10_n=len(top),excluded=sum(excluded),filter_eligible=eligible)

def labels(root, folder):
 rel=folder.relative_to(root).parts;phase='标定预实验'if 'calibration'in rel else '范围取证'if 'range_evidence'in rel else '初始对照'if 'R0'in rel else '正式探索'if 'versions'in rel else '固定对照实验'
 name=ROOT_NAMES.get(root.name,root.name)
 if root.name=='agenticgen_uniguide_flowr_20261026':
  if rel[0]=='versions':
   names={'UG01':'原始梯度探索·首版','UG02':'原始梯度探索·修订二版','UG03':'原始梯度探索·修订三版','UG04':'原始梯度探索·冻结五轮','UG05':'原始梯度探索·异步质量版（未完成）'};name=names.get(rel[1],'原始梯度探索')
  else:name='原始梯度初始对照'if phase=='初始对照'else'原始梯度尺度与窗口标定'
 elif root.name in ['agenticgen_guidance_comparison5_20261031','agenticgen_guidance_selected_retest_20261031']:
  job=rel[1];name='动态匹配四窗口基线'if 'baseline'in job else '前十步联合注入'if 'prefix_joint'in job else '动态匹配五窗口'
  name+='·严格工具重测'if 'selected_retest'in root.name else '·五轮原测'
 elif root.name=='agenticgen_gradient_scope_20261101':
  h=int(rel[1][3]);name=['仅条件原子梯度','条件加一跳梯度','条件加二跳梯度'][h];phase='固定预标定尺度'
 elif root.name in ['agenticgen_joint_inpainting_20261027','agenticgen_late_window_20261028','agenticgen_time_window_20261028']:
  arm=rel[-1];names={'full_interior':'完整1至99步区间','late_only':'晚期10至99步区间','window':'原10至90步窗口','raw999':'千分位截断未覆盖终态','native1':'额外终态前向','raw1':'单位网格未覆盖终态','native':'原生终态覆盖','chemical':'化学注入','gradient':'纯梯度','joint':'坐标化学联合'}
  tokens=arm.split('__') if '__' in arm else arm.split('_')
  name+='·'+'·'.join(names.get(v,v)for v in tokens)
 return name,phase

def coverage(coords,mask,reference):
 from scipy.optimize import linear_sum_assignment
 out=[]
 for xyz,valid in zip(coords,mask):
  xyz=xyz[np.asarray(valid,dtype=bool)]
  if len(xyz)<len(reference) or not np.isfinite(xyz).all():out.append(None);continue
  cost=((reference[:,None]-xyz[None])**2).sum(-1);r,c=linear_sum_assignment(cost)
  out.append(float(np.sqrt(np.mean(cost[r,c]))))
 return out

def main():
 import torch,rdkit
 from rdkit import Chem,RDLogger
 from rdkit.Contrib.SA_Score import sascorer
 RDLogger.DisableLog('rdApp.warning');RDLogger.DisableLog('rdApp.error')
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--out',required=True);args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
 plan=read(Path(args.plan));roots=[Path(v)for v in plan['roots']]
 teachers=read(Path('/workspace/agenticgen_uniguide_flowr_20261026/teachers.json'));refs={t['target']:np.asarray(t['core_A'],dtype=np.float64)for t in teachers}
 assert all(v.shape==(9,3)for v in refs.values())
 sources=[];groups=[];skipped=[];root_coverage=[];cache={};allrows=[];quality_rows=[]
 for root in roots:
  paths=[p for p in root.rglob('records.json')if not any(v in SKIP for v in p.relative_to(root).parts)] if 'acceptance'not in root.name else []
  accepted=0
  for path in sorted(paths):
   folder=path.parent;rows=read(path)
   if not isinstance(rows,list)or not rows or not (folder/'COMPLETE.json').exists():skipped.append(dict(path=str(path),reason='no_complete_population'));continue
   if not (folder/'terminal.pt.gz').exists():skipped.append(dict(path=str(path),reason='no_saved_terminal'));continue
   ag=read(folder/'aggregate.json',{});man=read(folder/'manifest.json',{});ident=man.get('identity',{});comp=ident.get('compatibility',{})
   target=ag.get('teacher_id')or comp.get('teacher_id')or next((t for t in refs if t in str(folder)),None)
   if target not in refs:skipped.append(dict(path=str(path),reason='teacher_identity_unresolved'));continue
   experiment,phase=labels(root,folder)
   blob=(folder/'terminal.pt.gz').read_bytes()
   try:
    terminal=torch.load(io.BytesIO(gzip.decompress(blob)),map_location='cpu',weights_only=False)
    coords=terminal['coords'].double().numpy();mask=terminal['mask'].bool().numpy()
    if len(coords)!=len(rows):raise ValueError('Saved row population differs')
    heads=terminal.get('affinity',{}).get('pic50');pic=heads.detach().cpu().double().reshape(-1).tolist()if heads is not None else [None]*len(rows)
    if len(pic)!=len(rows):raise ValueError('Wrong pIC50 head shape')
   except Exception as e:skipped.append(dict(path=str(path),reason='terminal_read_error',error=repr(e)));continue
   rmsd=coverage(coords,mask,refs[target]);native=[r.get('condition_coordinate_root_mean_square_deviation_angstrom',r.get('condition_rmsd_A'))for r in rows];sas=[]
   for row in rows:
    smi=row.get('smiles','')
    if not row.get('build_success',False)or not smi:sas.append(None);continue
    if smi not in cache:
     try:
      mol=Chem.MolFromSmiles(smi);mol=Chem.RemoveHs(mol)if mol is not None else None
      cache[smi]=float(sascorer.calculateScore(mol))if mol is not None and mol.GetNumAtoms()else None
     except Exception:cache[smi]=None
    sas.append(cache[smi])
   resc=[r.get('pic50_on_rescore')for r in rows]
   metrics=dict(rmsd=rmsd,pic50=pic,sa=sas,rescore=resc,native_rmsd=native);flags_by={};elig={};fences={}
   for k,v in metrics.items():flags_by[k],elig[k],fences[k]=flags(v,'rmsd'if k=='native_rmsd'else k)
   jobroot=next((p for p in folder.parents if (p/'sample_shards').is_dir()),None)
   gid=str(folder);group_rows=[]
   for i,row in enumerate(rows):
    physical=str(jobroot/row['source_shard'])+':'+str(row['source_tensor_row'])if jobroot and row.get('source_shard')is not None else gid+':'+str(row.get('tensor_row',i))
    value=dict(experiment=experiment,phase=phase,teacher=target,group=gid,index=i,physical_id=physical,seed=row.get('logical_seed',row.get('seed')),slot=row.get('logical_slot',row.get('slot')),node_id=row.get('node_id'),built=bool(row.get('build_success')),connected=bool(row.get('connected')),**{k:float(v[i])if finite(v[i])else None for k,v in metrics.items()},excluded={k:f[i]for k,f in flags_by.items()},eligible=elig)
    group_rows.append(value);allrows.append(value)
   stats_by={k:stats(v,k,flags_by[k],elig[k])for k,v in metrics.items()}
   groups.append(dict(experiment=experiment,phase=phase,teacher=target,folder=gid,population=len(rows),built=sum(r['built']for r in group_rows),connected=sum(r['connected']for r in group_rows),schedule=ag.get('effective_guidance_scale_schedule',ag.get('schedule',man.get('schedule',ident.get('effective_guidance_scale_schedule')))),stats=stats_by,fences=fences,record_sha256=sha(path),terminal_sha256=hashlib.sha256(blob).hexdigest(),condition_sha256=ag.get('condition_sha256',comp.get('condition_sha256')),coordinate_policy=comp.get('policy'),native_vs_uniform_rmsd_max_difference=max([abs(a-b)for a,b in zip(native,rmsd)if finite(a)and finite(b)],default=None)))
   accepted+=1
   qfile=folder/'quality.json'
   if qfile.exists():
    byid={r['node_id']:r for r in group_rows}
    for q in read(qfile,[]):
     if q.get('node_id')in byid:
      r=byid[q['node_id']];quality_rows.append(dict(physical_id=r['physical_id'],experiment=experiment,phase=phase,teacher=target,quality=q.get('quality',{})))
  root_coverage.append(dict(root=str(root),completed_measurement_groups=accepted,classification='科学已完成组'if accepted else'工程验收或未保存完整科学终态',records_examined=len(paths)))
 # Largest complete view supplies the outlier flags of a reused physical sample.
 group_size={g['folder']:g['population']for g in groups};buckets={}
 for r in sorted(allrows,key=lambda r:-group_size[r['group']]):
  key=(r['experiment'],r['teacher'],r['phase']);buckets.setdefault(key,{}) .setdefault(r['physical_id'],r)
 pooled=[]
 for key,items in buckets.items():
  rows=list(items.values());item=dict(experiment=key[0],teacher=key[1],phase=key[2],unique_saved_candidates=len(rows),built=sum(r['built']for r in rows),connected=sum(r['connected']for r in rows),groups=len({r['group']for r in rows}),stats={})
  for metric in ['rmsd','pic50','sa','rescore','native_rmsd']:
   item['stats'][metric]=stats([r[metric]for r in rows],metric,[r['excluded'][metric]for r in rows],all(r['eligible'][metric]for r in rows))
  q={r['physical_id']:r for r in quality_rows if (r['experiment'],r['teacher'],r['phase'])==key};components={}
  for r in q.values():
   for n,v in (r['quality'].get('posebusters_checks') or {}).items():
    c=components.setdefault(n,dict(evaluable=0,passed=0));c['evaluable']+=1;c['passed']+=int(bool(v))
  item['posebusters']=dict(checked=len(q),evaluable=sum(bool(r['quality'].get('posebusters_evaluable'))for r in q.values()),passed=sum(bool(r['quality'].get('posebusters_pass'))for r in q.values()),components=components,not_extrapolated=True)
  pooled.append(item)
 result=dict(timestamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),rdkit=rdkit.__version__,sa_definition='RDKit Ertl-Schuffenhauer synthetic accessibility, 1 easy to 10 hard; not a 3D conformation metric',sascorer_sha256=sha(Path(sascorer.__file__)),sa_fragments_sha256=sha(Path(sascorer.__file__).with_name('fpscores.pkl.gz')),rmsd_definition='Post-hoc common original nine PLIP heavy atoms; injective terminal geometric Hungarian in original generation/protein frame; no ligand fitting and no changes to stored rankings',pic50_definition='Saved terminal output affinity.pic50 on-target prediction; no new forward or rescore',rescore_definition='Historical record pic50_on_rescore, reported separately',top10_definition='Best ceil(10% of finite observations); RMSD/SA ascending, pIC50 descending; independent selection by metric',filter_definition='Per complete configuration population, minimum 10 finite observations; RMSD upper Q3+max(3IQR,1A), pIC50/SA both outer 3IQR fences; if >10% flagged filtered mean is NA; missing/nonfinite never classified as outliers',physical_dedup_definition='Explicit source shard and source row within each experiment/teacher/stage; largest complete view first; do not sum reference views as independent attempts',source_roots=root_coverage,groups=groups,experiments=pooled,skipped=skipped,rows_written=len(allrows),unique_smiles_scored=len(cache),new_sampling=0,new_model_forwards=0,llm_calls=0)
 write(out/'all_results.json',result)
 with gzip.open(out/'candidate_rows.jsonl.gz','wt',encoding='utf-8')as f:
  for r in allrows:f.write(json.dumps(r,ensure_ascii=True,allow_nan=False)+'\n')
 print(json.dumps(dict(output=str(out/'all_results.json'),groups=len(groups),experiment_rows=len(pooled),candidate_views=len(allrows),deduplicated_bucket_candidates=sum(r['unique_saved_candidates']for r in pooled),scored_unique_smiles=len(cache),skipped=len(skipped)),ensure_ascii=True),flush=True)

if __name__=='__main__':main()
