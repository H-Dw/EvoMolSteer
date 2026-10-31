"""Read-only final reconciliation. No model, sampler, provider or GPU imports."""
import csv
import hashlib
import json
import math
import re
import shutil
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path('/workspace/agenticgen_guidance_comparison5_20261031')
REPO = Path('/root/private_data/uniguide_comparison5_20261031/runs/guidance-comparison5-072e6f78')
OUT = ROOT / 'monitoring/20261031_2017'
METRIC = 'condition_coordinate_root_mean_square_deviation_angstrom'
SEEDS = [42, 100045, 200048, 300051, 400054]


def read(path, default=None):
    return json.loads(Path(path).read_text(encoding='utf-8')) if Path(path).exists() else default


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=True, indent=2) + '\n', encoding='utf-8')


def stamp():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def text_lengths(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {'reason', 'improvement_reason', 'uncertainty'} and isinstance(item, str):
                yield len(item)
            else:
                yield from text_lengths(item)
    elif isinstance(value, list):
        for item in value:
            yield from text_lengths(item)


def role_details(source, jobroot):
    folder, role = source.parent, source.name.split('.')[0]
    spec = read(source)
    session = folder / (role + '.session')
    binding = read(session / 'BINDING.json')
    answer = read(folder / (role + '.json'))
    context = json.loads(next(m['content'] for m in spec['messages'] if m['role'] == 'user'))
    requests, responses, diagnostics, tools = [], [], [], []
    for path in sorted(session.glob('step_*/REQUEST.json')):
        req = read(path)
        candidates = [m['content'].split('Final response schema: ', 1)[1]
                      for m in req['messages'] if m['role'] == 'user'
                      and 'Final response schema: ' in m.get('content', '')]
        assert candidates and json.loads(candidates[0]) == spec['response_schema']
        assert req['model'] == 'deepseek-flash' and req['max_tokens'] == 16384
        assert req['reasoning_effort'] == 'high' and req['thinking'] == {'type': 'enabled'}
        assert req['messages'][:len(spec['messages'])] == spec['messages']
        requests.append({'path':str(path.relative_to(jobroot)), 'sha256':sha(path),
                         'schema_transmitted_in_actual_request':True,
                         'model':req['model'], 'max_tokens':req['max_tokens']})
        for message in req['messages']:
            if message['role'] == 'user' and message.get('content','').startswith('Replace the entire final JSON;'):
                diag = json.loads(message['content'].split('Validation: ', 1)[1])
                if diag not in diagnostics:
                    diagnostics.append(diag)
    for path in sorted(session.glob('step_*/RESPONSE.json')):
        wire = read(path)
        assert wire['model'] == 'deepseek-flash'
        choice = wire['choices'][0]
        content = choice['message'].get('content')
        try:
            parsed = json.loads(content) if content else None
        except (ValueError, TypeError):
            parsed = None
        responses.append({'path':str(path.relative_to(jobroot)), 'sha256':sha(path),
                          'id':wire['id'], 'usage':wire['usage'],
                          'finish_reason':choice['finish_reason'],
                          'parsed_response':parsed,
                          'missing_required':sorted(set(spec['response_schema'].get('required', [])) - set(parsed)) if isinstance(parsed,dict) else None,
                          'text_lengths':list(text_lengths(parsed))})
    for path in sorted(session.glob('step_*/TOOL_*.json')):
        tool = read(path)
        tools.append({'path':str(path.relative_to(jobroot)), 'sha256':sha(path),
                      'tool_call_id':tool['tool_call_id'], 'call':tool['call'], 'result':tool['result']})
    if answer is not None:
        assert max(text_lengths(answer), default=0) <= 3000
    return {'role':role, 'folder':str(folder.relative_to(jobroot)),
            'prompt_source_path':str(source.relative_to(jobroot)), 'prompt_source_sha256':sha(source),
            'schema':spec['response_schema'], 'context_keys':list(context),
            'checkpoint_learning':context.get('checkpoint_learning'),
            'range_evidence_allowance':context.get('range_evidence_allowance'),
            'confirmed_constant_levels':context.get('confirmed_constant_levels'),
            'working_neighbourhoods':context.get('working_neighbourhoods'),
            'incumbent_id':context.get('incumbent_id'),
            'allowed_anchor_ids':context.get('allowed_anchor_ids'),
            'answer':answer, 'failed':read(session/'FAILED.json'),
            'binding_sha256':sha(session/'BINDING.json'),
            'binding_controls':{k:binding[k] for k in ['model','thinking','reasoning_effort','max_output_tokens','max_format_repairs','attempts','timeout']},
            'requests':requests, 'responses':responses, 'repair_diagnostics':diagnostics,
            'tool_receipts':tools, 'http_attempts':len(list(session.glob('step_*/HTTP_*.json'))),
            'receipt':read(folder/(role+'.receipt.json'))}


def group_details(path, jobroot):
    group = read(path / 'aggregate.json')
    if not isinstance(group, dict) or METRIC not in group or not (path/'COMPLETE.json').exists():
        return None
    rows = read(path/'records.json')
    values = [r[METRIC] for r in rows]
    assert values and all(math.isfinite(v) for v in values)
    n = len(values)
    stats = group['coordinate_loss_statistics']
    ordered = sorted(values)
    def quantile(q):
        pos = (n-1)*q
        k = int(pos)
        return ordered[k] + (ordered[min(k+1,n-1)]-ordered[k])*(pos-k)
    threshold = quantile(.75) + max(3*(quantile(.75)-quantile(.25)), 1.)
    excluded = [i for i,v in enumerate(values) if v > threshold]
    kept = [v for i,v in enumerate(values) if i not in excluded]
    assert n == stats['expected'] == stats['measured']
    assert math.isclose(sum(values)/n, group[METRIC], abs_tol=1e-10)
    assert math.isclose(threshold, stats['threshold_angstrom'], abs_tol=1e-7)
    assert len(excluded) == stats['excluded_high_count']
    assert math.isclose(sum(v*v/2 for v in values)/n, stats['raw_mean_coordinate_loss_angstrom_squared'], abs_tol=1e-7)
    if kept:
        assert math.isclose(sum(kept)/len(kept), stats['upper_tail_filtered_mean'], abs_tol=1e-10)
    complete = read(path/'COMPLETE.json')
    for name,digest in complete['files'].items():
        assert sha(path/name) == digest, str(path/name)
    return {'path':str(path.relative_to(jobroot)), 'aggregate':group,
            'complete_sha256':sha(path/'COMPLETE.json'), 'complete_files_verified':len(complete['files']),
            'record_count':n, 'records_sha256':sha(path/'records.json'),
            'raw_records':rows, 'statistics_recomputed':True,
            'chemical_condition':read(path/'CHEMICAL_CONDITION.json'),
            'output_policy':read(path/'OUTPUT_POLICY.json')}


plan = read(ROOT/'PLAN.json')
results, budget_rows, calibration_csv = [], [], []
for job in plan['jobs']:
    cfg = read(job['config'])
    root = Path(cfg['root'])
    versions = root/'versions/GD01'
    attempts = read(root/'attempts.json', {})
    budget = {s:sum(r['count'] for r in attempts.values() if r['stage']==s) for s in ['calibration','formal']}
    assert budget['calibration'] <= 350 and budget['formal'] <= 750
    rounds = [read(p) for p in sorted(versions.glob('*/advice_*/ROUND_COMPLETE.json'))]
    roles = [role_details(p,root) for p in sorted(versions.glob('**/*.prompt_source.json'))]
    groups = []
    for path in sorted(root.rglob('aggregate.json')):
        if 'acceptance' in path.relative_to(root).parts or 'sample_shards' in path.relative_to(root).parts:
            continue
        detail = group_details(path.parent,root)
        if detail is not None:
            groups.append(detail)
            g,st = detail['aggregate'],detail['aggregate']['coordinate_loss_statistics']
            calibration_csv.append({'job':job['id'],'source':detail['path'],'population':detail['record_count'],
                'scale':g.get('calibrated_guidance_reference_scale'),
                'schedule':json.dumps(g.get('effective_guidance_scale_schedule',g.get('multipliers'))),
                'raw_rmsd':g[METRIC],'ranking_rmsd':st['upper_tail_filtered_mean'],
                'raw_loss':st['raw_mean_coordinate_loss_angstrom_squared'],
                'filtered_loss':st['upper_tail_filtered_mean_coordinate_loss_angstrom_squared'],
                'threshold':st['threshold_angstrom'],'excluded':st['excluded_high_count'],
                'retained':st['retained_count'],'statistics_ranking_eligible':st['ranking_eligible']})
    quality = []
    for requestpath in sorted((root/'quality_queue/requests').glob('*.json')):
        req = read(requestpath)
        folder = Path(req['folder'])
        assert root in folder.parents and (folder/'COMPLETE.json').exists()
        assert sha(folder/'records.json') == req['binding']['records']
        assert sha(folder/'molecules_all_built.sdf') == req['binding']['sdf']
        assert sha(req['teacher']['protein']) == req['binding']['protein']
        q = read(folder/'QUALITY_COMPLETE.json')
        assert q is not None and (q['attempts'],q['checked_attempts'],q['not_selected'],q['coverage']) == (50,10,40,.2)
        assert [(r['logical_seed'],r['logical_slot']) for r in q['selection']] == [(s,k) for s in SEEDS for k in [0,1]]
        assert sha(folder/'quality.json') == q['quality_sha256']
        assert sha(folder/'records.json') == q['records_sha256']
        for name,digest in q['quality_implementation_sha256'].items():
            assert sha(REPO/'agenticgen/src/agenticgen/uniguide'/name) == digest
        qr = read(folder/'quality.json')
        recoveries = [r.get('plip_recovery') for r in qr if isinstance(r.get('plip_recovery'), (int,float))]
        counts = Counter()
        for r in qr:
            for key,passed in r['quality'].get('posebusters_checks',{}).items():
                counts[key] += bool(passed)
        quality.append({'source':str(folder.relative_to(root)), 'request_sha256':sha(requestpath),
                        'receipt_sha256':sha(folder/'QUALITY_COMPLETE.json'),'receipt':q,
                        'posebusters_check_pass_counts':dict(counts),
                        'plip_recovery_evaluable':len(recoveries),
                        'plip_recovery_mean':sum(recoveries)/len(recoveries) if recoveries else None,
                        'nonpass_or_unevaluable':[r for r in qr if not r['build_success'] or not r['quality'].get('posebusters_pass') or not r['quality'].get('plip_evaluable',r['build_success'])]})
    policies = []
    for path in sorted(versions.glob('**/POLICY.json')):
        policy = read(path)
        if 'max_model_calls_per_round' not in policy:
            continue
        rows = [read(p) for p in sorted(path.parent.glob('call_*.json'))]
        charged = sum(r.get('completion_tokens',r['reserved_output_tokens']) for r in rows)
        assert policy['max_model_calls_per_round']==30 and policy['max_output_tokens_per_round']==491520
        assert len(rows)<=30 and charged<=491520
        policies.append({'path':str(path.relative_to(root)),'policy':policy,'http_attempts':len(rows),'charged_output_tokens':charged,'claims':rows})
    shards = []
    for marker in sorted((root/'sample_shards').glob('*/COMPLETE.json')):
        rows = read(marker.parent/'records.json')
        if rows is not None:
            shards.append({'path':str(marker.parent.relative_to(root)),'rows':len(rows),
                           'built':sum(bool(r.get('build_success')) for r in rows),
                           'complete_sha256':sha(marker)})
    phases = {p.stem:read(p) for p in sorted((ROOT/'logs'/job['id']).glob('*_EXIT.json'))}
    logs = {}
    for path in sorted((ROOT/'logs'/job['id']).glob('*.log')):
        text = path.read_text(encoding='utf-8')
        logs[path.name] = {'sha256':sha(path),'bytes':path.stat().st_size,
            'gpu_kernel_error_tokens':[token for token in ['HSA_STATUS_ERROR_OUT_OF_RESOURCES','KERNEL VMFault','SIGKILL','SIGABRT'] if token in text]}
    finalized = read(versions/'FINALIZED.json')
    fifty = [g for g in groups if g['record_count']==50 and g['aggregate']['coordinate_loss_statistics']['ranking_eligible'] and g['aggregate'].get('stability_status')!='large_finite_displacement_growth']
    available = min(fifty,key=lambda g:g['aggregate']['ranking_'+METRIC]) if fifty else None
    result = {'id':job['id'],'experiment':job['experiment'],'teacher':job['teacher'],
              'root':str(root),'config_sha256':sha(job['config']),'configured_rounds':cfg['rounds'],
              'reserved_attempts':budget,'attempts_sha256':sha(root/'attempts.json'),
              'completed_rounds':len(rounds),'rounds':rounds,'roles':roles,'groups':groups,
              'finalized':finalized,'best_available_complete_population':available['path'] if available else None,
              'sampling_finished':read(root/'SAMPLING_FINISHED.json'),
              'campaign_failed':read(root/'CAMPAIGN_FAILED.json'),'job_result':read(ROOT/'job_results'/(job['id']+'.json')),
              'acceptance':read(root/'acceptance/COMPLETE.json'),'domains':read(root/'calibration/domains.json'),
              'condition_files':{p.name:{'sha256':sha(p),'value':read(p)} for p in sorted((root/'conditions').glob('*.json'))},
              'input_audit':read(root/'INPUT_AUDIT.json'),'subjects':read(root/'SUBJECTS.json'),
              'quality':quality,'worker':read(root/'quality_queue/worker.json'),
              'worker_binding':read(root/'QUALITY_MODULE_BINDING.json'),'quality_status':read(root/'quality_queue/status.json'),
              'phase_exits':phases,'safe_logs':logs,'model_budgets':policies,'sample_shards':shards,
              'actual_request_files':sum(len(r['requests']) for r in roles),
              'actual_response_files':sum(len(r['responses']) for r in roles),
              'actual_http_attempt_files':sum(r['http_attempts'] for r in roles),
              'model_budget_claims':sum(r['http_attempts'] for r in policies),
              'observed_completion_tokens':sum(s['usage']['completion_tokens'] for r in roles for s in r['responses']),
              'maximum_accepted_text_characters':max((n for r in roles for n in text_lengths(r['answer'])),default=0)}
    results.append(result)
    write(job['id']+'_final.json',result)
    print(json.dumps({'job':job['id'],'rounds':len(rounds),'budget':budget,'group_views':len(groups),
        'quality_sources':len(quality),'provider_responses':result['actual_response_files'],
        'http_attempts':result['actual_http_attempt_files'],'budget_claims':result['model_budget_claims'],
        'sample_shard_rows':sum(r['rows'] for r in shards),'best_available':available['path'] if available else None},ensure_ascii=True),flush=True)

summary = {'timestamp':stamp(),'commit':plan['commit'],'plan_sha256':sha(ROOT/'PLAN.json'),
           'source_preservation_sha256':sha(ROOT/'SOURCE_PRESERVATION.json'),
           'jobs':[{k:j[k] for k in ['id','teacher','experiment','configured_rounds','completed_rounds','reserved_attempts','actual_request_files','actual_response_files','actual_http_attempt_files','model_budget_claims','observed_completion_tokens','maximum_accepted_text_characters','best_available_complete_population']} for j in results],
           'total_reserved_attempts':{s:sum(j['reserved_attempts'][s] for j in results) for s in ['calibration','formal']},
           'total_provider_responses':sum(j['actual_response_files'] for j in results),
           'total_http_attempts':sum(j['actual_http_attempt_files'] for j in results),
           'total_model_budget_claims':sum(j['model_budget_claims'] for j in results),
           'total_observed_completion_tokens':sum(j['observed_completion_tokens'] for j in results),
           'quality_totals':{k:sum(q['receipt'][k] for j in results for q in j['quality']) for k in ['checked_attempts','built','unbuilt_selected','posebusters_pass','posebusters_evaluable','plip_evaluable','posebusters_unevaluable_built','plip_unevaluable_built']},
           'quality_sources':sum(len(j['quality']) for j in results),
           'all_jobs_attempted_sha256':sha(ROOT/'ALL_JOBS_ATTEMPTED.json'),
           'suite_finished_sha256':sha(ROOT/'SUITE_FINISHED.json'),
           'runtime_source_sha256':plan['runtime_source_sha256'],
           'provider_source_sha256':{n:sha(REPO/'agenticgen/src/agenticgen/providers'/n) for n in ['deepseek_tools.py','deepseek_api.py']},
           'collector_sha256':sha(__file__),'collector_imports_no_model_sampler_provider':True,
           'new_scientific_attempts':0,'new_api_requests':0,'all_group_statistics_recomputed':True}
write('suite_reconciliation.json',summary)
with (OUT/'all_population_metrics.csv').open('w',encoding='utf-8',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(calibration_csv[0]));writer.writeheader();writer.writerows(calibration_csv)
write('final_readout_manifest.json',{p.name:sha(p) for p in OUT.glob('*_final.json')} | {'suite_reconciliation.json':sha(OUT/'suite_reconciliation.json'),'all_population_metrics.csv':sha(OUT/'all_population_metrics.csv')})
print(json.dumps(summary,ensure_ascii=True)[:12000],flush=True)
