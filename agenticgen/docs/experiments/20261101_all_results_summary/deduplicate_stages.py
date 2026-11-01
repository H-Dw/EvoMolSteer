"""Assign reused measurements to their original stage; keep all group views."""
from pathlib import Path
import json,gzip,collections,importlib.util,datetime
spec=importlib.util.spec_from_file_location('saved_statistics','/tmp/summarize_results_20261101_v3.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
base=Path('/workspace/uniguide_result_exports_20261101/summary');data=m.read(base/'final/all_results.json');rows=[json.loads(l)for l in gzip.open(base/'attempt_002/candidate_rows.jsonl.gz','rt',encoding='utf-8')]
roots=[Path(v)for v in m.read(Path('/workspace/uniguide_result_exports_20261101/REBUILD_PLAN.json'))['roots']]
for r in rows:
 root=next(p for p in roots if Path(r['group']).is_relative_to(p));r['experiment'],r['phase']=m.labels(root,Path(r['group']))
for g in data['groups']:
 root=next(p for p in roots if Path(g['folder']).is_relative_to(p));g['experiment'],g['phase']=m.labels(root,Path(g['folder']))
size={g['folder']:g['population']for g in data['groups']};priority={'标定预实验':0,'初始对照':0,'范围取证':1,'固定预标定尺度':1,'固定对照实验':1,'正式探索':2}
phases=collections.defaultdict(set)
for r in rows:phases[(r['experiment'],r['teacher'],r['physical_id'])].add(r['phase'])
owner={k:min(v,key=lambda p:priority[p])for k,v in phases.items()};chosen={}
for r in sorted(rows,key=lambda r:-size[r['group']]):
 key=(r['experiment'],r['teacher'],r['physical_id'])
 if key not in chosen:chosen[key]=dict(r,phase=owner[key])
buckets=collections.defaultdict(list)
for r in chosen.values():buckets[(r['experiment'],r['teacher'],r['phase'])].append(r)
qrows={};viewrows=collections.defaultdict(dict)
for r in rows:viewrows[r['group']][r['node_id']]=r
for group,mapping in viewrows.items():
 for q in m.read(Path(group)/'quality.json',[]):
  if q.get('node_id')not in mapping:continue
  r=mapping[q['node_id']];key=(r['experiment'],r['teacher'],r['physical_id']);qrows[key]=q.get('quality')or{}
experiments=[e for e in data['experiments']if e['phase']=='历史参考']
for key,rr in buckets.items():
 item=dict(experiment=key[0],teacher=key[1],phase=key[2],unique_saved_candidates=len(rr),built=sum(r['built']for r in rr),connected=sum(r['connected']for r in rr),groups=len({r['group']for r in rr}),stats={})
 for metric in ['rmsd','pic50','sa','rescore','native_rmsd']:item['stats'][metric]=m.stats([r[metric]for r in rr],metric,[r['excluded'][metric]for r in rr],all(r['eligible'][metric]for r in rr))
 qq=[qrows[k]for k in qrows if (k[0],k[1],owner[k])==key];components={}
 for q in qq:
  for n,v in (q.get('posebusters_checks')or{}).items():
   c=components.setdefault(n,dict(evaluable=0,passed=0));c['evaluable']+=1;c['passed']+=int(bool(v))
 item['posebusters']=dict(checked=len(qq),evaluable=sum(bool(q.get('posebusters_evaluable', bool(q.get('posebusters_checks'))))for q in qq),passed=sum(bool(q.get('posebusters_pass'))for q in qq),components=components,not_extrapolated=True);experiments.append(item)
data['experiments']=experiments;data['cross_stage_aliases_removed']=sum(len(v)-1 for v in phases.values());data['all_unique_saved_candidates_with_historical_reference']=sum(e['unique_saved_candidates']for e in experiments);data['physical_dedup_definition']+='; cross-stage calibration reuse belongs to calibration, then range evidence, then formal; no cross-stage double count';data['timestamp']=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
recent=sum(e['unique_saved_candidates']for e in experiments if '五轮原测'in e['experiment']);assert recent==3790,recent
(base/'final_v3').mkdir(exist_ok=True)
m.write(base/'final_v3/all_results_deduplicated.json',data)
with gzip.open(base/'final_v3/deduplicated_candidates.jsonl.gz','wt',encoding='utf-8')as f:
 for r in chosen.values():f.write(json.dumps(r,ensure_ascii=True)+'\n')
print(json.dumps(dict(experiment_rows=len(experiments),unique_saved_candidates_with_reference=data['all_unique_saved_candidates_with_historical_reference'],cross_stage_aliases_removed=data['cross_stage_aliases_removed'],original_five_round_attempts_verified=recent),ensure_ascii=True))
