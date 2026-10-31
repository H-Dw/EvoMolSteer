"""Reconcile closed results and native receipts without importing model or sampler."""
from pathlib import Path
import hashlib

ROOT = Path('/workspace/agenticgen_guidance_selected_retest_20261031')
prior = ROOT / 'monitoring/20261101_0118/audit_and_archive.py'
assert hashlib.sha256(prior.read_bytes()).hexdigest() == 'e6ce73de325cf231af9068046e1a7ca7cbb24288555c98c3269e998898ce0c01'
source = prior.read_text(encoding='utf-8').split('mirror_root =', 1)[0]
source = source.replace("OUT = ROOT / 'monitoring/20261101_0118'", "OUT = ROOT / 'monitoring/20261101_0221'")
exec(compile(source, str(prior), 'exec'))

import csv
import math

METRIC = 'condition_coordinate_root_mean_square_deviation_angstrom'
facts, metric_rows = [], []
assert read(ROOT/'SUITE_FINISHED.json') and read(ROOT/'ALL_JOBS_ATTEMPTED.json')
assert not read(ROOT/'ALL_JOBS_ATTEMPTED.json')['failed_jobs']
for job, audited in zip(plan['jobs'], jobs):
    cfg = read(job['config'])
    root = Path(cfg['root'])
    assert cfg['rounds'] == 5 and len(audited['rounds']) == 5
    assert not read(root/'CAMPAIGN_FAILED.json')
    assert all(q['complete'] and not q['failed'] for q in audited['qualities'])
    assert not audited['quality_status']['active'] and audited['quality_status']['pending'] == 0
    groups = []
    for path in sorted(root.rglob('aggregate.json')):
        rel = path.relative_to(root)
        if any(s in rel.parts for s in ['acceptance','sample_shards','candidate_batches']):
            continue
        g = read(path)
        if not isinstance(g, dict) or METRIC not in g:
            continue
        folder = path.parent
        complete = read(folder/'COMPLETE.json')
        if not complete:
            continue
        for name, digest in complete['files'].items():
            assert sha(folder/name) == digest, str(folder/name)
        rows = read(folder/'records.json')
        values = [r[METRIC] for r in rows]
        assert values and all(isinstance(v,(int,float)) and math.isfinite(v) for v in values)
        n = len(values)
        st = g['coordinate_loss_statistics']
        ordered = sorted(values)
        def quantile(q):
            p = (n-1)*q
            k = int(p)
            return ordered[k]+(ordered[min(k+1,n-1)]-ordered[k])*(p-k)
        threshold = quantile(.75)+max(3*(quantile(.75)-quantile(.25)),1.)
        excluded = [i for i,v in enumerate(values) if n >= 10 and v > threshold]
        kept = [v for i,v in enumerate(values) if i not in excluded]
        assert n == st['expected'] == st['measured']
        assert math.isclose(sum(values)/n,g[METRIC],abs_tol=1e-10)
        assert math.isclose(threshold,st['threshold_angstrom'],abs_tol=1e-7)
        assert excluded == st['excluded_high_indices']
        assert math.isclose(sum(v*v/2 for v in values)/n,st['raw_mean_coordinate_loss_angstrom_squared'],abs_tol=1e-7)
        assert math.isclose(sum(kept)/len(kept),st['upper_tail_filtered_mean'],abs_tol=1e-10)
        row = {'job':job['id'],'source':str(folder.relative_to(root)),'population':n,
               'schedule':g.get('effective_guidance_scale_schedule',g.get('multipliers')),
               'raw_rmsd':g[METRIC],'ranking_rmsd':g.get('ranking_'+METRIC),
               'raw_loss':st['raw_mean_coordinate_loss_angstrom_squared'],
               'ranking_loss':st['upper_tail_filtered_mean_coordinate_loss_angstrom_squared'],
               'threshold':threshold,'excluded':len(excluded),'excluded_ids':st['excluded_high_samples'],
               'retained':len(kept),'ranking_eligible':st['ranking_eligible'],
               'stability_status':g.get('stability_status'),'built':sum(bool(r['build_success']) for r in rows),
               'unique_smiles':len(set(r.get('smiles') for r in rows if r['build_success'])),
               'initial_correspondence_rmsd':g.get('assignment_diagnostics_not_used_for_ranking',{}).get('initial_assignment_condition_coordinate_root_mean_square_deviation_angstrom'),
               'chemical_condition':read(folder/'CHEMICAL_CONDITION.json'),
               'complete_sha256':sha(folder/'COMPLETE.json'),'raw_records':rows}
        groups.append(row)
        metric_rows.append({k:v for k,v in row.items() if k not in ['raw_records','chemical_condition','excluded_ids']})
    eligible = [g for g in groups if g['population']==50 and g['ranking_eligible'] and g['stability_status']!='large_finite_displacement_growth']
    best = min(eligible,key=lambda g:g['ranking_rmsd'])
    role_details = []
    for path in sorted((root/'versions/GD01').glob('**/*.prompt_source.json')):
        role = path.name.split('.')[0]
        spec = read(path)
        contexts = []
        for msg in spec['messages']:
            if msg['role'] != 'user':
                continue
            try:
                value = json.loads(msg['content'])
            except (ValueError,TypeError):
                continue
            if isinstance(value,dict):
                contexts.append({k:v for k,v in value.items() if k in ['checkpoint_learning','incumbent_id','allowed_anchor_ids','confirmed_constant_levels','working_neighbourhoods','range_evidence_allowance','final_submission_contract','teacher','round','iteration']})
        role_details.append({'path':str(path.relative_to(root)), 'prompt_sha256':sha(path),
                             'role':role,'prompt_context':contexts,'answer':read(path.with_name(role+'.json')),
                             'receipt':read(path.with_name(role+'.receipt.json'))})
    quality = []
    for q in audited['qualities']:
        rows = read(root/q['source']/'quality.json')
        counts = Counter()
        for r in rows:
            counts.update(k for k,v in r['quality'].get('posebusters_checks',{}).items() if v)
        plip = [r['plip_recovery'] for r in rows if isinstance(r.get('plip_recovery'),(int,float))]
        quality.append({'source':q['source'],'receipt':q['receipt'],
                        'posebusters_check_pass_counts':dict(counts),
                        'plip_recovery_mean':sum(plip)/len(plip) if plip else None,
                        'plip_recovery_evaluable':len(plip),
                        'nonpass_or_unevaluable':[r for r in rows if not r['build_success'] or not r['quality'].get('posebusters_pass') or not r['quality'].get('plip_evaluable',True)]})
    shards = [read(p.parent/'records.json') for p in sorted((root/'sample_shards').glob('*/COMPLETE.json'))]
    assert sum(len(rows) for rows in shards) == sum(audited['attempts'].values())
    result = {'id':job['id'],'teacher':job['teacher'],'experiment':job['experiment'],
              'rounds':audited['rounds'],'roles':role_details,'groups':groups,'best':best,
              'domains':read(root/'calibration/domains.json'),'acceptance':read(root/'acceptance/COMPLETE.json'),
              'attempts':audited['attempts'],'saved_shard_rows':sum(len(rows) for rows in shards),
              'saved_shard_built':sum(bool(r['build_success']) for rows in shards for r in rows),
              'provider_responses':audited['actual_http_responses'],'budget_claims':audited['budget_claims'],
              'completion_tokens':audited['observed_completion_tokens'],'qualities':quality,
              'version_finalized':read(root/'versions/GD01/FINALIZED.json'),
              'formal_summary':read(root/'versions/GD01/REPORT.json'),
              'sampling_finished':read(root/'SAMPLING_FINISHED.json')}
    facts.append(result)
    write(job['id']+'_results.json',result)

summary = {'timestamp':stamp(),'science_commit':plan['commit'],'plan_sha256':sha(ROOT/'PLAN.json'),
           'suite_finished_sha256':sha(ROOT/'SUITE_FINISHED.json'),'old_finalized_sha256':sha(old_final),
           'jobs':[{'id':f['id'],'rounds_completed':len(f['rounds']),'attempts':f['attempts'],
                    'saved_shard_rows':f['saved_shard_rows'],'saved_shard_built':f['saved_shard_built'],
                    'responses':f['provider_responses'],'claims':f['budget_claims'],'completion_tokens':f['completion_tokens'],
                    'best':{k:v for k,v in f['best'].items() if k not in ['raw_records','chemical_condition']},
                    'quality_sources':len(f['qualities']),
                    'quality_totals':{k:sum(q['receipt'].get(k,0) for q in f['qualities']) for k in ['checked_attempts','built','unbuilt_selected','posebusters_pass','posebusters_evaluable','plip_evaluable','posebusters_unevaluable_built','plip_unevaluable_built']},
                    'format_repairs':sum(r['receipt']['interface_repairs'] for r in f['roles'])} for f in facts],
           'new_scientific_attempts_by_collector':0,'new_api_calls_by_collector':0,
           'all_group_statistics_recomputed':True,'all_native_receipts_and_claims_verified':True,'issues':issues}
write('result_summary.json',summary)
write('runtime_audit.json',{'timestamp':stamp(),'jobs':jobs,'issues':issues,'source':str(prior),'source_sha256':sha(prior)})
with (OUT/'all_population_metrics.csv').open('w',encoding='utf-8',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(metric_rows[0]));writer.writeheader();writer.writerows(metric_rows)

prior_manifest_path = ROOT/'monitoring/20261101_0118/archive_manifest.json'
assert sha(prior_manifest_path)=='3c9ff541a2d514ed37c20410fef20024d12b2a02abc1be74641a2c78c422d3bb'
prior_manifest = read(prior_manifest_path)['files']
reused = {}
for name,path in list(selected.items()):
    if name in prior_manifest:
        assert sha(path)==prior_manifest[name]['sha256'],name
        reused[name]=prior_manifest[name]
        del selected[name]
for path in [ROOT/'SUITE_FINISHED.json',ROOT/'ALL_JOBS_ATTEMPTED.json',ROOT/'PLAN.json',ROOT/'DURABLE_SNAPSHOT.json',ROOT/'HISTORICAL_ACCOUNTING.json']:
    if path.exists():selected[path.relative_to(ROOT).as_posix()]=path
for path in OUT.iterdir():
    if path.is_file() and path.suffix in {'.json','.csv','.py'}:selected[path.relative_to(ROOT).as_posix()]=path
manifest = {}
archive = OUT/'closed_artifacts.tar.gz'
with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
    for name,path in sorted(selected.items()):
        before=path.stat();digest=sha(path);tar.add(path,arcname=name,recursive=False);after=path.stat()
        assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns) and sha(path)==digest,name
        manifest[name]={'sha256':digest,'bytes':before.st_size}
write('archive_manifest.json',{'timestamp':stamp(),'files':manifest,'previous_files_reverified':len(reused),
                             'previous_manifest_sha256':sha(prior_manifest_path),'launcher_or_core_included':False})
complete={'timestamp':stamp(),'archive':str(archive),'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,
          'manifest_sha256':sha(OUT/'archive_manifest.json'),'files':len(manifest),'hdf5':sum(n.endswith('.h5') for n in manifest),
          'prior_files_reverified':len(reused),'all_jobs_hdf5':sum(len(list(Path(read(j['config'])['root']).rglob('*.h5'))) for j in plan['jobs'])}
write('backup_COMPLETE.json',complete)
print(json.dumps({'summary':summary,'backup':complete},ensure_ascii=True),flush=True)
