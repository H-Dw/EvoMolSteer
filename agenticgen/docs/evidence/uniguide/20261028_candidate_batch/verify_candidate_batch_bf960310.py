import copy, datetime, fcntl, json, subprocess, sys
from pathlib import Path
import h5py
import numpy as np
import torch
from agenticgen.uniguide.io import read,write,frozen,sha,value_sha
from agenticgen.uniguide.geometry_runtime import calibrated
from agenticgen.uniguide.geometry_protocol import source_binding
from agenticgen.uniguide.geometry_batch import execute_many
from agenticgen.uniguide.geometry_batch_acceptance import compare_trees
from agenticgen.uniguide.native_tools import load_pt
from agenticgen.uniguide.prepare import load_model

checkout=Path('/workspace/uniguide_audits/candidate-batch-bf960310')
config=read(str(checkout)+'-config.json')
config['initials_root']=config['generation_context_root']
root=Path(config['root']);old=Path('/workspace/agenticgen_geometry_acceptance_20261028_82dc89c1')
previous=read(old/'acceptance/COMPLETE.json');binding=source_binding()
differ=[k for k in set(binding)|set(previous['source_binding']) if binding.get(k)!=previous['source_binding'].get(k)]
assert len(differ)==2 and all(k.endswith(('geometry_batch.py','sampler.py')) for k in differ),differ
assert subprocess.check_output(['git','-C',str(checkout),'rev-parse','--short=8','HEAD'],text=True).strip()=='bf960310'
assert not subprocess.check_output(['git','-C',str(checkout),'status','--porcelain'],text=True).strip()
old_config=read(old/'GEOMETRY_CONFIG.json')
for key in ['generation_context_root','checkpoint_sha256','native_hashes','steps','noise_level','output_policy','windows','seeds']:
 assert config[key]==old_config[key],key
protected=read(str(checkout)+'-source-before.json')['source_hashes']
assert all(sha(p)==s for p,s in protected.items())
subjects=calibrated(config);checks=copy.deepcopy(previous['checks']);targets={}
with (root/'.executor.lock').open('a+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for subject in subjects:
  teacher,condition=subject['teacher'],subject['condition'];target=teacher['target']
  assert condition==read(old/'conditions'/(target+'.json'))
  model=load_model(config,teacher)
  destination=root/'acceptance'/target/'mixed_failure_validation'
  jobs=[dict(schedule=[.001,.002,.001,.004],folder=destination/'healthy',stage='calibration',count=10),
        dict(schedule=[1e30]*4,folder=destination/'nonfinite',stage='calibration',count=10)]
  results=execute_many(model,config,teacher,condition,jobs)
  assert isinstance(results[0],dict) and isinstance(results[1],FloatingPointError),[type(r).__name__ for r in results]
  reference=old/'acceptance'/target/'candidate_batch/separate_0'
  healthy=root/results[0]['id']
  differences=compare_trees(load_pt(reference/'terminal.pt.gz'),load_pt(healthy/'terminal.pt.gz'))
  assert not differences,differences
  with h5py.File(reference/'trace.h5') as left,h5py.File(healthy/'trace.h5') as right:
   for name in left:assert np.array_equal(left[name][:],right[name][:]),name
  failure=read(destination/'nonfinite/FAILED.json')
  for unit in failure['source_units']:
   assert (root/unit/'FAILED.json').exists() and not (root/unit/'COMPLETE.json').exists()
  charges=[r for name,r in read(root/'attempts.json').items() if (root/name/'manifest.json').exists() and read(root/name/'manifest.json')['identity']['compatibility']['teacher_id']==target]
  assert sum(r['count'] for r in charges)==20
  targets[target]=dict(healthy_full_100_steps_exact=True,terminal_exact=True,all_trace_fields_exact=True,
    nonfinite_units_preserved=True,nonfinite_units_not_marked_complete=True,no_scientific_retry=True,
    engineering_attempts=20,reference_terminal_sha256=sha(reference/'terminal.pt.gz'),
    new_terminal_sha256=sha(healthy/'terminal.pt.gz'),failure=failure,
    previous_complete_sha256=sha(old/'acceptance'/target/'candidate_batch/COMPLETE.json'))
  write(destination/'VERIFIED.json',targets[target])
  checks[target]['final_revision_mixed_failure_validation']=targets[target]
  del model
  import gc
  gc.collect();torch.cuda.empty_cache()
 assert all(sha(p)==s for p,s in protected.items())
 assert not (root/'calibration/domains.json').exists()
 receipt=dict(source_binding=binding,config_sha256=value_sha(config),checks=checks,
  time_asia_shanghai=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
  engineering_only=True,new_completed_candidates=0,engineering_attempts_this_revision=40,
  llm_calls=0,fitted_multipliers=False,original_attempts_sha256=protected[str(Path(config['generation_context_root'])/'attempts.json')],
  evidence_reuse=dict(previous_commit='82dc89c1',previous_complete_sha256=sha(old/'acceptance/COMPLETE.json'),
   previous_root=str(old),modified_source_keys=differ,
   reason='The source delta preserves failed-unit structural masks, releases completed GPU pockets, and adds diagnostic-only overflow repair hooks. Sampling tensors and integration formulas are unchanged; finite float32 diagnostics retain their original values. All other source hashes match. New real-model mixed-failure tests validate the changed path against the preserved independent 100-step trajectories; the unchanged VJP, disabled-guidance, resume, concurrency and OOM probes are reused with source receipts.',
   verifier_sha256=sha(__file__)),source_hashes_after={p:sha(p) for p in protected})
 frozen(root/'acceptance/COMPLETE.json',receipt)
 print(json.dumps(dict(complete=True,root=str(root),targets=targets),ensure_ascii=True))
