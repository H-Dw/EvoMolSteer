"""Verify released resources and preservation, then bind the actual pushed review."""
import hashlib
import json
import runpy
import shutil
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT=Path('/workspace/agenticgen_guidance_selected_retest_20261031')
OUT=ROOT/'monitoring/20261101_0221'
AUDIT=Path('/root/private_data/uniguide_selected_retest_20261031/audits/selected-78e6e180')
def read(p):return json.loads(p.read_text())
def sha(p):
    with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def stamp():return datetime.now(timezone(timedelta(hours=8))).isoformat()
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=True,indent=2)+'\n',encoding='utf-8')

plan=read(ROOT/'PLAN.json');repo=Path(plan['repo']);binding=read(AUDIT/'GPT_REVIEW_BINDING.json')
cleanup=read(OUT/'worker_cleanup_complete.json')
assert cleanup['all_owned_cpu_workers_stopped'] and len(cleanup['after'])==2
assert all(x['worker_lock']=='free' for x in cleanup['after'])
assert binding['review_commit_pushed'] and sha(AUDIT/'actual_gpt_final_review.zh-CN.md')==binding['review_sha256']
assert binding['science_commit']==plan['commit']=='78e6e1803758f7016a2568b00176eaaa27ef6dc4'
p=subprocess.run(['python',str(repo/'agenticgen/docs/experiments/20261031_selected_contract_retest/inspect_suite.py'),'--plan',str(ROOT/'PLAN.json')],capture_output=True,text=True)
assert p.returncode==0,p.stderr[-2000:]
d=json.loads(p.stdout)
assert d['actual_commit']==plan['commit'] and d['tracked_checkout_clean']
assert not d['runtime_source_changed'] and not d['protected_files_changed'] and d['preserved_files']==23175
assert d['suite_lock']==d['global_executor_lock']=='free'
for j in d['jobs']:
    assert j['configuration_hash_matches'] and j['completed_rounds']==5 and j['acceptance_complete'] and j['failed'] is None
    assert all(v in ['free','absent'] for v in j['locks'].values())
active=[]
for proc in Path('/proc').glob('[0-9]*'):
    try:
        argv=[x.decode() for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
        if len(argv)>1 and Path(argv[1]).name in ['run_suite.py','run_uniguide_geometry.py'] and (str(ROOT) in ' '.join(argv) or str(repo) in ' '.join(argv)):
            active.append(argv)
        if len(argv)>3 and argv[1:3]==['-m','agenticgen.uniguide.quality_queue'] and str(ROOT) in ' '.join(argv):
            active.append(argv)
    except OSError:pass
assert not active,active
shutil.copyfile(d['snapshot'],OUT/'final_snapshot.json')
mirror=runpy.run_path(str(repo/'agenticgen/docs/experiments/20261031_guidance_comparison5/durable_snapshot.py'))['mirror']
info=mirror(ROOT,plan['durable_snapshot_root'])
statepath=Path(plan['durable_snapshot_root'])/'MIRROR_STATE.json';state=read(statepath)
errors=[];advanced=[]
for name,row in state.items():
    copy=Path(plan['durable_snapshot_root'])/name;source=ROOT/name
    if not copy.exists() or sha(copy)!=row['sha256']:errors.append(name)
    if source.exists() and sha(source)!=row['sha256']:advanced.append(name)
assert not errors and not advanced,(errors,advanced)
verification={'timestamp':stamp(),'files':len(state),'bytes':sum(v['size'] for v in state.values()),
              'mirror_state_sha256':sha(statepath),'copy_hash_errors':errors,'source_advanced':advanced,
              'full_hdf5_included':False,'launcher_log_included':False,
              'disk_free_bytes':shutil.disk_usage(plan['durable_snapshot_root']).free}
write(OUT/'mirror_final_verification.json',verification)
archive=read(AUDIT/'archive_catalog.json');facts=read(AUDIT/'final_facts.json')
assert archive['all_member_hashes_verified'] and archive['all_job_hdf5_covered'] and archive['hdf5']==274
assert facts['total_new_attempts']==1380 and facts['total_old_and_new_attempts']==5170
assert facts['total_provider_responses']==58 and facts['accepted_role_submissions']==24
assert sha(Path('/workspace/agenticgen_guidance_comparison5_20261031/FINALIZED.json'))=='9c6fd372b65f1ee05bd9828e41cdb8e48c7b9ff6d8000ff1c2da58c52e83a781'
marker={'timestamp':stamp(),'status':'complete','science_commit':plan['commit'],
        'plan_sha256':sha(ROOT/'PLAN.json'),'source_preservation_sha256':sha(ROOT/'SOURCE_PRESERVATION.json'),
        'source_files_verified':23175,'protected_changes':[],
        'review_commit':binding['review_commit'],'review_sha256':binding['review_sha256'],
        'review_path':str(AUDIT/'actual_gpt_final_review.zh-CN.md'),'actual_main_gpt_review_completed':True,
        'completed_jobs':2,'completed_rounds_per_job':5,'quality_sources_complete':18,
        'new_scientific_attempts':1380,'old_scientific_attempts_preserved':3790,'total_attempts':5170,
        'final_facts_sha256':sha(AUDIT/'final_facts.json'),'archive_catalog_sha256':sha(AUDIT/'archive_catalog.json'),
        'raw_hdf5_downloaded_and_verified':274,'worker_cleanup_sha256':sha(OUT/'worker_cleanup_complete.json'),
        'mirror_verification_sha256':sha(OUT/'mirror_final_verification.json'),
        'global_executor_lock':'free','suite_lock':'free','all_owned_cpu_workers_stopped':True,
        'active_sampling_or_provider_processes':[],'next_version_started':False,'launcher_or_core_downloaded':False,
        'runtime_code_updated_by_documentation':False}
assert not (ROOT/'FINALIZED.json').exists()
write(ROOT/'FINALIZED.json',marker)
shutil.copyfile(ROOT/'FINALIZED.json',AUDIT/'FINALIZED.json')
shutil.copyfile(ROOT/'FINALIZED.json',OUT/'FINALIZED.json')
write(OUT/'closure_readout.json',{'timestamp':stamp(),'finalized_sha256':sha(ROOT/'FINALIZED.json'),
                                'mirror':verification,'active_processes':active,'science_commit':plan['commit']})
print(json.dumps({'finalized':str(ROOT/'FINALIZED.json'),'finalized_sha256':sha(ROOT/'FINALIZED.json'),
                  'preserved_files':d['preserved_files'],'mirror':verification,'active_sampling':active,
                  'worker_locks_released':True},ensure_ascii=True))
