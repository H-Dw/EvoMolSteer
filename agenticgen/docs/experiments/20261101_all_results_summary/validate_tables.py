from pathlib import Path
import csv,datetime,hashlib,importlib.util,json,math
B=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('saved_stats',B/'summarize_results.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
assert m.stats(list(range(1,11)),'rmsd')['top10_mean']==1
assert m.stats(list(range(1,11)),'pic50')['top10_mean']==10
assert m.stats([1,None,float('nan'),2],'sa')['n']==2
f,e,_=m.flags([1.]*19+[100.],'rmsd');assert sum(f)==1 and e and m.stats([1.]*19+[100.],'rmsd',f,e)['filtered_mean']==1
f,e,_=m.flags([1.]*8+[100.,200.],'rmsd');assert sum(f)==2 and not e and m.stats([1.]*8+[100.,200.],'rmsd',f,e)['filtered_mean'] is None
assert m.stats([None]*5,'sa')['mean'] is None
assert m.flags([1.]*8+[100.],'rmsd')[0]==[False]*9
x=json.loads((B/'raw/all_results_deduplicated.json').read_text(encoding='utf-8'));assert len(x['experiments'])==79 and len(x['groups'])==656
for r in x['experiments']:
 assert r['posebusters']['passed']<=r['posebusters']['evaluable']<=r['posebusters']['checked']
 for k in ['rmsd','pic50','sa']:
  s=r['stats'][k];assert s['n']+s['missing']==r['unique_saved_candidates']
  assert s['top10_n']==(max(1,math.ceil(.1*s['n']))if s['n']else 0)
  if s['n'] and k=='sa':assert 1<=s['best']<=s['mean']<=10
  if s['n'] and k=='pic50':assert s['best']>=s['mean']
assert sum(r['unique_saved_candidates']for r in x['experiments']if '五轮原测'in r['experiment'])==3790
assert sum(r['unique_saved_candidates']for r in x['experiments']if '严格工具重测'in r['experiment'])==1380
assert sum(r['unique_saved_candidates']for r in x['experiments']if r['phase']=='固定预标定尺度')==300
assert len(list(csv.DictReader((B/'全部实验指标表.csv').open(encoding='utf-8-sig'))))==79
assert len(list(csv.DictReader((B/'逐配置指标表.csv').open(encoding='utf-8-sig'))))==656
receipt=json.loads((B/'archive_receipt.json').read_text());assert receipt['all_member_hashes_verified'] and receipt['status']=='complete_workspace_only' and not receipt['moved_to_persistent']
result=dict(timestamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),synthetic_statistic_cases=7,all_experiment_rows=79,configuration_rows=656,denominator_checks_passed=True,budget_crosschecks=[3790,1380,300],archive_bytes=receipt['bytes'],archive_sha256=receipt['sha256'],raw_analysis_sha256=hashlib.sha256((B/'raw/all_results_deduplicated.json').read_bytes()).hexdigest(),source_experiments_modified=False,new_generation=0,new_model_forwards=0)
(B/'validation.json').write_text(json.dumps(result,ensure_ascii=True,indent=2)+'\n',encoding='utf-8',newline='\n');print(json.dumps(result,ensure_ascii=True))
