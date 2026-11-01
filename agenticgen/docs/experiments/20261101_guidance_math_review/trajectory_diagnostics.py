from pathlib import Path
import json,hashlib,datetime
import numpy as np
import h5py
ROOT=Path('/workspace/agenticgen_gradient_scope_20261101')
OUT=Path('/workspace/uniguide_result_exports_20261101/gradient_scope_diagnosis_20261101')
OUT.mkdir(exist_ok=False)
def sha(p):
 with p.open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
results=[];sources={}
for job in ['hop0_6pyy','hop0_3w54','hop1_6pyy','hop1_3w54','hop2_6pyy','hop2_3w54']:
 folder=ROOT/'jobs'/job/'fixed_population'; trace=folder/'trace.h5'; steps=folder/'steps.json'
 before={str(p):{'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,'sha256':sha(p)}for p in [trace,steps]}
 ss=json.loads(steps.read_text()); assert len(ss)==100
 with h5py.File(trace,'r') as f:
  g=f['unmasked_input_gradient'][10:99].astype(float)
  applied=f['input_gradient'][10:99].astype(float)
  masks=f['guidance_atom_mask'][10:99].astype(bool)
  valid=f['mask'][10:99].astype(bool)
  assignment=f['assignment'][9:99]
  native=f['native_drift'][10:99].astype(float)
  assert np.array_equal(np.where(masks[...,None],g,0),applied)
  raw2=(g*g).sum((2,3));used2=(applied*applied).sum((2,3))
  alpha=np.array([-np.asarray(s['coefficient']) for s in ss[10:99]])
  assert alpha.shape==raw2.shape and (alpha>0).all()
  nonzero=raw2>0
  ratio=np.divide(used2,raw2,out=np.zeros_like(used2),where=nonzero)
  selected=f['selected'][10:99].astype(bool)
  cond2=(g*g*selected[...,None]).sum((2,3))
  series=[]
  for lo,hi in [(10,25),(26,50),(51,75),(76,98)]:
   s=slice(lo-10,hi-9); full=raw2[s]; used=used2[s];a=alpha[s]
   series.append({'window':[lo,hi],'gradient_energy_retained':float(used.sum()/full.sum()),'coefficient_weighted_retained':float((a*used).sum()/(a*full).sum()),'selected_atoms_mean':float(masks[s].sum(-1).mean()),'condition_slot_gradient_energy_share':float(cond2[s].sum()/full.sum())})
  dt=np.array([s['dt'] for s in ss[10:99]])[:,None]
  native_loss_first_order=(g*native).sum((2,3))*dt
  switches=(assignment[1:]!=assignment[:-1]).mean(-1)
  toggle=(masks[1:]!=masks[:-1])&valid[1:]
  result={'job':job,'teacher':'6pyy'if job.endswith('6pyy')else'3w54','hops':int(job[3]),'evaluated_updates':'10..98 inclusive; step99 update does not enter returned prediction','states':int(nonzero.sum()),'gradient_energy_retained':float(used2.sum()/raw2.sum()),'coefficient_weighted_retained':float((alpha*used2).sum()/(alpha*raw2).sum()),'per_state_retained_median':float(np.median(ratio[nonzero])),'per_state_retained_min':float(ratio[nonzero].min()),'per_state_retained_max':float(ratio[nonzero].max()),'condition_slot_gradient_energy_share':float(cond2.sum()/raw2.sum()),'selected_atoms_mean':float(masks.sum(-1).mean()),'selected_atoms_min':int(masks.sum(-1).min()),'selected_atoms_max':int(masks.sum(-1).max()),'assignment_switch_fraction_mean':float(switches.mean()),'atom_mask_switch_fraction_mean':float(toggle.sum()/valid[1:].sum()),'native_drift_first_order_opposes_loss_descent_fraction':float((native_loss_first_order>0).mean()),'windows':series,'interpretation':'Same-state counterfactual full VJP vs masked VJP, not a comparison to the baseline trajectory; retained energy is first-order loss progress ratio under fixed correspondence and equal coefficient, not measured final condition improvement'}
  results.append(result)
 for p in [trace,steps]:
  now={'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,'sha256':sha(p)}
  assert now==before[str(p)]
 sources.update(before)
 print(json.dumps(result,ensure_ascii=True),flush=True)
receipt={'timestamp':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'source_preservation':sources,'results':results,'new_sampling':0,'new_model_forwards':0,'new_llm_calls':0,'gpu_used':False}
(OUT/'diagnostics.json').write_text(json.dumps(receipt,ensure_ascii=True,indent=2)+'\n')
