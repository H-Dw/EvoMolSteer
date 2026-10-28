"""Read-only measurement and provenance verification for completed late-window study."""
from pathlib import Path
import json,hashlib,datetime,subprocess,fcntl
import numpy as np
import h5py
ROOT=Path('/workspace/agenticgen_late_window_20261028')
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def slot(r):return r['logical_seed'],r['logical_slot']
spec=read(ROOT/'STUDY.json');result=read(ROOT/'RESULTS.json')
audit={'time_shanghai':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'results_sha256':sha(ROOT/'RESULTS.json'),'groups':{},'protected_file_count':len(spec['protected_files'])}
audit['results_matches_finalized']=audit['results_sha256']==read(ROOT/'FINALIZED.json')['results_sha256']
audit['changed_protected']=[p for p,h in spec['protected_files'].items() if not Path(p).exists() or sha(p)!=h]
evidence={'groups':{}}
seeds=spec['seeds'];population={(s,i) for s in seeds for i in range(10)};selected={(s,i) for s in seeds for i in [0,1]}
for key,g in result['groups'].items():
    folder=Path(g['folder'])
    if g['status']!='complete':
        audit['groups'][key]={'status':g['status'],'failure':read(folder/'FAILED.json')};continue
    records=read(folder/'records.json');quality=read(folder/'quality.json');receipt=read(folder/'QUALITY_COMPLETE.json');complete=read(folder/'COMPLETE.json')
    identity={slot(r):r['node_id'] for r in records}
    checks=dict(records_complete_population=len(records)==50 and set(identity)==population,
        records_sha256_matches_complete=sha(folder/'records.json')==complete['records_sha256'],
        records_sha256_matches_quality=sha(folder/'records.json')==receipt['records_sha256'],
        quality_sha256_matches=sha(folder/'quality.json')==receipt['quality_sha256'],
        quality_coverage=all(receipt[k]==v for k,v in dict(attempts=50,checked_attempts=10,not_selected=40,coverage=.2).items()),
        selection_identity=len(receipt['selection'])==10 and {slot(r) for r in receipt['selection']}==selected and all(identity[slot(r)]==r['node_id'] for r in receipt['selection']),
        no_sampling_or_quality_failure=not (folder/'FAILED.json').exists() and not (folder/'QUALITY_FAILED.json').exists())
    evidence['groups'][key]={'records':records,'quality':quality,'records_sha256':sha(folder/'records.json'),'quality_sha256':sha(folder/'quality.json')}
    item={'status':'complete','new':not g['reused'],'record_and_quality_validation':checks}
    if not g['reused']:
        pairing=read(folder/'PAIRED_TRACE_ACCEPTANCE.json');item['pairing']=pairing
        checks['pairing_receipt_complete']=pairing['all_equal'] and pairing['compared_steps']==list(range(91))
        with h5py.File(folder/'trace.h5','r') as actual,h5py.File(pairing['reference'],'r') as ref:
            item['pairing_independent_recheck']={field:bool(np.array_equal(actual[field][:91],ref[field][:91])) for field in pairing['keys']}
        checks['pairing_arrays_exact']=all(item['pairing_independent_recheck'].values())
        times=read(folder/'GUIDANCE_TIMES.json');checks['guidance_steps']=times['active_steps']==list(range(10,100))
        item['raw_output_causality']=read(folder/'RAW_OUTPUT_CAUSALITY.json')
        item['last_loop']=read(folder/'LAST_LOOP.json');item['output_policy']=read(folder/'OUTPUT_POLICY.json')
        checks['raw_policy']=not item['output_policy']['extra_terminal_forward'] and not item['output_policy']['terminal_chemical_projection'] and not item['output_policy']['coordinate_projection'] and item['output_policy']['output_source']=='last_loop_uncovered'
        rows=read(folder/'steps.json');late=[r for r in rows if r['step']>=91]
        item['late_step_diagnostics']=[{'step':r['step'],'time':r['time'],'scale':r['scale'],'gamma':r['gamma'],'coefficient':r['coefficient'][0],'maximum_guidance_increment_A':max(r['max_atom_increment_A']),'mean_gradient_norm':float(np.mean(r['gradient_norm']))} for r in late]
        old=read(Path(pairing['reference']).parent/'records.json');old={slot(r):r for r in old}
        item['same_slot_molecular_identity']={field:sum(r.get(field)==old[slot(r)].get(field) for r in records) for field in ['smiles','build_success','connected']}
        if key=='6pyy_H7S_A_402/chemical__raw999__late_only':
            item['outlier']={'before':old[(42,2)],'after':next(r for r in records if slot(r)==(42,2))}
            item['outlier_late_steps']=[{k:(r[k][2] if isinstance(r[k],list) else r[k]) for k in ['step','time','scale','gamma','coefficient','loss','gradient_norm','max_atom_increment_A','delta_native_ratio','assignment_switch_fraction']} for r in late]
    audit['groups'][key]=item
queue=ROOT/'quality_queue';worker=read(queue/'worker.json')
audit['quality_worker']={'metadata':worker,'status':read(queue/'status.json'),'pid_still_present':Path('/proc',str(worker['pid'])).exists(),'stop_requested':read(queue/'STOP_REQUESTED.json') if (queue/'STOP_REQUESTED.json').exists() else None,'request_count':len(list((queue/'requests').glob('*.json'))),'current_module_hashes':{name:sha(Path(worker['module_directory'])/name) for name in worker['implementation_sha256']}}
audit['quality_modules_match']=audit['quality_worker']['current_module_hashes']==worker['implementation_sha256']
audit['code_commit']=subprocess.check_output(['git','-C','/workspace/uniguide_audits/late-window-585468d2','rev-parse','HEAD'],text=True).strip()
audit['code_worktree_status']=subprocess.check_output(['git','-C','/workspace/uniguide_audits/late-window-585468d2','status','--porcelain'],text=True).strip()
audit['all_group_checks_pass']=all(all(v.get('record_and_quality_validation',{}).values()) for v in audit['groups'].values())
assert audit['results_matches_finalized'] and not audit['changed_protected'] and audit['all_group_checks_pass'] and audit['quality_modules_match']
out=ROOT/'analysis';out.mkdir(exist_ok=True)
for name,value in [('review_slot_evidence.json',evidence),('review_audit.json',audit)]:
    (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'time':audit['time_shanghai'],'protected_files':audit['protected_file_count'],'changed':audit['changed_protected'],'complete_groups':len(evidence['groups']),'new_exact_prefix_groups':sum(v.get('new',False) for v in audit['groups'].values()),'all_checks_pass':audit['all_group_checks_pass'],'worker_pid_present':audit['quality_worker']['pid_still_present'],'requests':audit['quality_worker']['request_count'],'code_commit':audit['code_commit'],'code_clean':not audit['code_worktree_status']}))
