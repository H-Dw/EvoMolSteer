"""CPU-only description of sealed results. No generation/model/provider imports."""
import argparse,gzip,hashlib,json,math,statistics
from pathlib import Path
from datetime import datetime,timezone,timedelta
import numpy as np
import torch
import rdkit
from rdkit import Chem,RDLogger
from rdkit.Contrib.SA_Score import sascorer

RMSD='condition_coordinate_root_mean_square_deviation_angstrom'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def finite(v):return isinstance(v,(int,float,np.number)) and not isinstance(v,bool) and math.isfinite(float(v))
def stats(values,kind):
 good=[float(x) for x in values if finite(x)];n=len(good)
 if not n:return dict(n=0,missing=len(values),mean=None,best=None,filtered_mean=None,top10_mean=None)
 q1,q3=np.quantile(good,[.25,.75]);width=3*(q3-q1)
 lo,hi=(-math.inf,q3+max(width,1.)) if kind=='rmsd' else (q1-width,q3+width)
 kept=[x for x in good if lo<=x<=hi];excluded=n-len(kept);eligible=n>=10 and excluded/n<=.1
 top=sorted(good,reverse=kind=='pic50')[:math.ceil(.1*n)]
 return dict(n=n,missing=len(values)-n,mean=statistics.mean(good),best=(max(good) if kind=='pic50' else min(good)),
  filtered_mean=statistics.mean(kept) if eligible else None,excluded=excluded,filter_eligible=eligible,
  top10_mean=statistics.mean(top),top10_n=len(top),lower=None if math.isinf(lo) else float(lo),upper=float(hi))
def mean(v):return statistics.mean([float(x) for x in v if finite(x)]) if any(finite(x) for x in v) else None

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);parser.add_argument('--out',required=True);a=parser.parse_args()
 plan=read(a.plan);root=Path(plan['root']);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
 assert read(root/'SUITE_FINISHED.json')['jobs'] and all(j['status']=='completed' for j in read(root/'SUITE_FINISHED.json')['jobs'])
 RDLogger.DisableLog('rdApp.warning');RDLogger.DisableLog('rdApp.error')
 sas={};groups=[];teachers=[];rows_out=[];sources={}
 def group(r,gid,stage):
  p=r/gid;records=read(p/'records.json');ag=read(p/'aggregate.json');assert (p/'COMPLETE.json').exists() and len(records)==ag['attempts']
  with gzip.open(p/'terminal.pt.gz','rb') as f:terminal=torch.load(f,map_location='cpu',weights_only=False)
  heads=terminal.get('affinity',{}).get('pic50');pic=heads.detach().double().reshape(-1).tolist() if heads is not None else [None]*len(records)
  assert len(pic)==len(records)
  sa=[]
  for record in records:
   smi=record.get('smiles','')
   if not record['build_success'] or not smi:sa.append(None);continue
   if smi not in sas:
    mol=Chem.MolFromSmiles(smi);sas[smi]=float(sascorer.calculateScore(Chem.RemoveHs(mol))) if mol is not None else None
   sa.append(sas[smi])
  values=dict(rmsd=[r[RMSD] for r in records],pic50=pic,sa=sa)
  values={k:[float(x) if finite(x) else None for x in v] for k,v in values.items()}
  summary={k:stats(v,k) for k,v in values.items()}
  assert abs(summary['rmsd']['mean']-ag['coordinate_loss_statistics']['arithmetic_mean'])<1e-10
  if summary['rmsd']['filter_eligible']:assert abs(summary['rmsd']['filtered_mean']-ag['coordinate_loss_statistics']['upper_tail_filtered_mean'])<1e-10
  for i,record in enumerate(records):
   rows_out.append(dict(group=gid,teacher=ag['teacher_id'],stage=stage,seed=record['logical_seed'],slot=record['logical_slot'],
    source_shard=record.get('source_shard'),source_tensor_row=record.get('source_tensor_row'),built=record['build_success'],
    **{k:v[i] for k,v in values.items()}))
  item=dict(id=gid,teacher=ag['teacher_id'],stage=stage,population=len(records),built=ag['built'],connected=ag['connected'],
   unique_smiles=ag['unique_smiles'],schedule=ag['effective_guidance_scale_schedule'],metrics=summary,
   stored_coordinate_statistics=ag['coordinate_loss_statistics'],assignment_diagnostics=ag.get('assignment_diagnostics_not_used_for_ranking'),
   record_sha256=sha(p/'records.json'),terminal_sha256=sha(p/'terminal.pt.gz'))
  for n in ('records.json','terminal.pt.gz','aggregate.json','COMPLETE.json'):sources[str(p/n)]=sha(p/n)
  if len(records)==50:
   steps=read(p/'steps.json');assert [s['step'] for s in steps]==list(range(100))
   for s in steps:
    assert all(bool(x)==(1<=s['step']<=98) for x in s['chemical_inpainting_applied'])
    assert not s['affinity_vjp']
    if s['step'] in (0,99):assert all(x==0 for x in s['gradient_norm']+s['max_atom_increment_A'])
   item['chemical_after_injection']={name:mean([x for s in steps if 1<=s['step']<=98 for x in s['chemical_'+name+'_after_match_fraction']]) for name in ('atomics','charges','bonds')}
   assert all(abs(x-1.)<1e-7 for x in item['chemical_after_injection'].values())
   features=read(p/'CHEMICAL_CONDITION.json')['features'];ids=np.array([q['condition_generated_slots'] for q in records]);ii=np.arange(len(ids))[:,None]
   item['terminal_chemical_matches']={}
   for name in ('atomics','charges'):
    actual=terminal[name].argmax(-1).numpy()[ii,ids];expected=np.array(features[name]).argmax(-1)
    item['terminal_chemical_matches'][name+'_atom_fraction']=float((actual==expected).mean())
   bonds=terminal['bonds'].argmax(-1).numpy();actual=bonds[np.arange(len(ids))[:,None,None],ids[:,:,None],ids[:,None,:]];expected=np.array(features['bonds']).argmax(-1)
   upper=np.triu(np.ones(expected.shape,dtype=bool),1);positive=upper & (expected!=0)
   item['terminal_chemical_matches']['all_induced_pair_fraction']=float((actual[:,upper]==expected[upper]).mean())
   item['terminal_chemical_matches']['positive_bond_fraction']=float((actual[:,positive]==expected[positive]).mean()) if positive.any() else None
  if (p/'QUALITY_COMPLETE.json').exists():
   rc=read(p/'QUALITY_COMPLETE.json');qrows=read(p/'quality.json');assert (rc['attempts'],rc['checked_attempts'],rc['not_selected'],rc['coverage'])==(50,10,40,.2)
   item['sample_quality']=rc;components={}
   for q in qrows:
    quality=q['quality']
    for name,value in (quality.get('posebusters_checks') or {}).items():
     counts=components.setdefault(name,dict(evaluable=0,passed=0));counts['evaluable']+=1;counts['passed']+=int(bool(value))
   item['posebusters_components']=components
   item['posebusters_versions']=sorted({q['quality'].get('posebusters_version') for q in qrows if q['quality'].get('posebusters_version')})
   item['mean_plip_recovery']=mean([q.get('plip_recovery') for q in qrows])
  groups.append(item);return item
 for job in plan['jobs']:
  r=Path(job['root']);target=job['teacher'];v=r/'versions/GD01'/target;domain=read(r/'calibration/domains.json')[target]
  subjects=read(r/'SUBJECTS.json');assert all(subjects[0]['condition'][k]==z for k,z in plan['expected_conditions'][target].items())
  calibration=read(r/'calibration/frozen'/domain['calibration_key']/'all_results.json')
  ids=[q['id'] for q in calibration if q['complete']]
  assert len(ids)==len(set(ids))
  mapped={gid:group(r,gid,'calibration') for gid in ids}
  baseline=mapped[domain['initial_source']];best=read(v/'GLOBAL_BEST.json');rounds=[]
  for p in sorted(v.glob('advice_*/ROUND_COMPLETE.json')):
   q=read(p);actions=[]
   for g in q['groups']:
    if g['id'] not in mapped:mapped[g['id']]=group(r,g['id'],'formal')
    actions.append(mapped[g['id']])
   rounds.append(dict(iteration=q['iteration'],checkpoint=q['checkpoint'],groups=[g['id'] for g in actions],
    proposal=read(p.parent/'proposal.json'),synthesis=read(p.parent/'synthesis.json'),
    proposal_context=json.loads(read(p.parent/'proposal.prompt_source.json')['messages'][1]['content']),
    synthesis_context=json.loads(read(p.parent/'synthesis.prompt_source.json')['messages'][1]['content'])))
  assert len(rounds)==5
  winner=mapped[best['representative']]
  ledger=read(r/'attempts.json');totals={s:sum(x['count'] for x in ledger.values() if x['stage']==s) for s in ('calibration','formal')}
  assert totals['calibration']<=500 and totals['formal']<=750
  formal=[g for g in mapped.values() if g['stage']=='formal'];assert len(formal)==15
  assert all('sample_quality' in g for g in formal)
  teachers.append(dict(teacher=target,root=str(r),budget=totals,domain=domain,calibration_groups=ids,
   baseline=baseline['id'],best=winner['id'],best_state=best,rounds=rounds,
   quality_totals={k:sum(g['sample_quality'].get(k,0) for g in formal) for k in ('checked_attempts','not_selected','unbuilt_selected','built','posebusters_pass','posebusters_evaluable','posebusters_unevaluable_built','severe_clash','clash_evaluable','plip_evaluable','plip_unevaluable_built')},
   formal_built=sum(g['built'] for g in formal),formal_connected=sum(g['connected'] for g in formal)))
 report=dict(timestamp=datetime.now(timezone(timedelta(hours=8))).isoformat(),commit=plan['commit'],plan_sha256=sha(a.plan),
  rdkit=rdkit.__version__,torch=torch.__version__,sascorer_sha256=sha(sascorer.__file__),sa_fragments_sha256=sha(Path(sascorer.__file__).with_name('fpscores.pkl.gz')),
  rmsd_definition='Saved nine-atom terminal dynamic Hungarian geometry in the protein-aligned generation frame; no ligand fitting',
  pic50_definition='Saved terminal affinity.pic50 head prediction; no new model forward or rescore',
  sa_definition='Standard RDKit SA from saved successfully built SMILES; not a 3D conformation measure',
  top10_definition='Best ceil(10% of finite values), selected independently for each metric',
  filter_definition='Per configuration: at least 10 finite values, at most 10% excluded; RMSD high Q3+max(3IQR,1A); pIC50/SA two-sided 3IQR; missing and build failures never outliers',
  sources=sources,teachers=teachers,groups=groups,new_sampling=0,new_model_forwards=0,llm_calls=0)
 with (out/'final_results.json').open('x') as f:json.dump(report,f,ensure_ascii=True,indent=2,allow_nan=False)
 with gzip.open(out/'candidate_metrics.jsonl.gz','wt',encoding='utf8') as f:
  for row in rows_out:f.write(json.dumps(row,ensure_ascii=True,allow_nan=False)+'\n')
 print(json.dumps(dict(output=str(out),teachers=len(teachers),groups=len(groups),candidate_views=len(rows_out),unique_smiles=len(sas),rdkit=rdkit.__version__,new_sampling=0,new_model_forwards=0,llm_calls=0),ensure_ascii=True))
if __name__=='__main__':main()
