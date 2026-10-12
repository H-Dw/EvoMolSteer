"""Reconstruct measured ablation costs without changing the running study.

Coarse intervals are deliberately retained: historical logs do not separately
time forward kernels, backward kernels, matching, MMFF or PoseBusters.
"""
import csv
import gzip
import json
import re
import subprocess
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np

from ..io import digest, write_json


def json_events(text):
    rows = []
    for line in text.splitlines():
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def union_seconds(intervals):
    total = 0.0
    end = None
    for left, right in sorted(intervals):
        if right < left:
            raise ValueError('Negative measured interval')
        total += max(0, right - max(left, end if end is not None else left))
        end = max(right, end if end is not None else right)
    return total


def describe(values):
    a = np.asarray(values, dtype=float)
    return None if not len(a) else {'count': len(a), 'sum_seconds': float(a.sum()),
        'mean_seconds': float(a.mean()), 'median_seconds': float(np.median(a)),
        'min_seconds': float(a.min()), 'max_seconds': float(a.max())}


def table(path, rows):
    if not rows:
        return
    columns = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def clock(t):
    return datetime.fromtimestamp(t, timezone(timedelta(hours=8))).isoformat(timespec='seconds')


def analyze(repo, study, capture, output):
    repo, study, capture, output = map(Path, (repo, study, capture, output))
    output.mkdir(parents=True, exist_ok=True)
    remote_path = capture / 'remote_runtime.json.gz'
    if remote_path.exists():
        remote = json.loads(gzip.decompress(remote_path.read_bytes()))
    else:
        remote_path = capture / 'remote_runtime.json'
        remote = json.loads(remote_path.read_text(encoding='utf-8'))
    cutoff = remote['captured_unix']
    state_path = capture / 'local_state_snapshot.json.gz'
    if state_path.exists():
        state = json.loads(gzip.decompress(state_path.read_bytes()))
    else:
        state_path = capture / 'local_state_snapshot.json'
        state = json.loads(state_path.read_text(encoding='utf-8'))
    log_path = repo / 'test/m2_skill_api_transport' / study.name / 'background.stdout.log'
    events = [r for r in json_events(log_path.read_text(encoding='utf-8', errors='replace'))
        if 'updated_unix' in r and r['updated_unix'] <= cutoff]
    write_json(output / 'local_phase_events.json', events)
    commits = []
    for line in subprocess.check_output(['git', 'log', '-200', '--format=%ct|%h|%s'], cwd=repo, text=True).splitlines():
        timestamp, sha, subject = line.split('|', 2)
        if state['started_unix'] <= int(timestamp) <= cutoff:
            commits.append({'unix':int(timestamp), 'sha':sha, 'subject':subject})
    write_json(output / 'commit_events.json', commits)
    api = []
    for path in sorted(study.rglob('*.receipt.json')):
        r = json.loads(path.read_text(encoding='utf-8'))
        if not {'started_unix','completed_unix'} <= r.keys() or r['completed_unix'] > cutoff:
            continue
        rel = path.relative_to(study).as_posix()
        match = re.search(r'(?P<phase>D-[GU]|A)_(?P<module>[AD]\d+\.\d+)_', rel)
        role = 'Reviewer' if rel.startswith('reviews/') else (
            'Analyst' if '/Analyst.tools_v1/' in rel else 'Designer' if '/Designer.tools_v1/' in rel else 'Preflight')
        api.append({'path':rel, 'phase':match['phase'] if match else 'COMMON',
            'module':match['module'] if match else 'COMMON', 'role':role,
            'started_unix':r['started_unix'], 'completed_unix':r['completed_unix'],
            'seconds':r['completed_unix']-r['started_unix'], 'attempts':r.get('attempts'),
            'completion_tokens':r.get('usage',{}).get('completion_tokens'),
            'prompt_tokens':r.get('usage',{}).get('prompt_tokens')})
    table(output / 'api_timing.csv', api)
    batches = []
    for name, log in remote['logs'].items():
        if name.endswith('.driver.log'):
            continue
        for row in json_events(log.get('text',log.get('tail',''))):
            if {'batch','arm','seconds'} <= row.keys():
                batches.append({'cohort':name.removesuffix('.log'),
                    **{k:row[k] for k in ['batch','arm','n','seconds','bytes']}})
    if len({(r['cohort'],r['arm'],r['batch']) for r in batches}) != len(batches):
        raise ValueError('Duplicate batch timing')
    table(output / 'batch_timing.csv', batches)
    cohorts = []
    for name, r in remote['statuses'].items():
        complete = r['status']=='complete'
        stop = r['completed_unix'] if complete else cutoff
        rows = [b for b in batches if b['cohort']==name]
        row = {'cohort':name, 'status':r['status'], 'started_unix':r['started_unix'],
            'completed_unix':r.get('completed_unix'), 'elapsed_seconds':stop-r['started_unix'],
            'completed_batches':len(rows),'completed_attempts':sum(b['n'] for b in rows),
            'batch_measured_seconds':sum(b['seconds'] for b in rows)}
        if complete:
            if row['completed_attempts'] != r['complete']['records']:
                raise ValueError('Complete cohort log lacks full batch timing coverage: '+name)
            loop = r['complete']['seconds']
            row.update(generation_loop_seconds=loop,
                loop_outside_batch_seconds=loop-row['batch_measured_seconds'],
                startup_attestation_archive_seconds=row['elapsed_seconds']-loop,
                archive_bytes=r['transport']['bytes'])
            starts = [e['updated_unix'] for e in events if e.get('task')==name and e.get('status')=='evaluating_local']
            ends = [e['updated_unix'] for e in events if e.get('task')==name and e.get('status')=='running'
                and e['updated_unix']>r['completed_unix']]
            if starts and ends:
                start = min(starts); end = min(t for t in ends if t>=start)
                row.update(completion_poll_download_seconds=start-r['completed_unix'],
                    local_evaluation_and_diagnostics_seconds=end-start)
                terminal = repo/'results'/study.name/name/'terminal_report.json'
                if terminal.exists() and start<=terminal.stat().st_mtime<=end:
                    row.update(local_archive_audit_MMFF_PB_seconds=terminal.stat().st_mtime-start,
                        local_post_terminal_diagnostics_seconds=end-terminal.stat().st_mtime)
            binds=[c['unix'] for c in commits if c['subject'].startswith('experiment(m2-api): bind '+name+' ')]
            verifies=[c['unix'] for c in commits if c['subject'].startswith('report(m2-api): verify '+name+' ')]
            if binds:row['deploy_startup_seconds']=r['started_unix']-min(binds)
            if ends and verifies:row['cleanup_commit_seconds']=min(verifies)-min(ends)
        cohorts.append(row)
    table(output/'cohort_timing.csv',cohorts)
    registry=json.loads((study/'module_registry.json').read_text(encoding='utf-8'))['modules']
    labels={r['id']:r['label_zh'] for r in registry}
    modules=[]
    observed=sorted({(r['phase'],r['module']) for r in api if r['module']!='COMMON'},
        key=lambda key:min(r['started_unix'] for r in api if (r['phase'],r['module'])==key))
    for phase, module in observed:
        calls=[r for r in api if (r['phase'],r['module'])==(phase,module)]
        starts=[e['updated_unix'] for e in events if e.get('phase')==phase and e.get('task')==module]
        start=min(starts) if starts else min(r['started_unix'] for r in calls)
        decision=next((r for r in state['pruning'] if r['phase']==phase and r['module']==module),None)
        skipped=next((r for r in state['skipped_modules'] if r['phase']==phase and r['module']==module),None)
        done=[c['unix'] for c in commits if c['subject'].startswith('ablation(scientific): ')
            and c['subject'].endswith(' '+module) and c['unix']>=start]
        if skipped:
            done=[c['unix'] for c in commits if c['subject']=='ablation(scientific): skip failed comparison '+module+' without causal attribution']
        end=min(done) if done else cutoff
        owned=[r for r in cohorts if r['started_unix']>=start and (r.get('completed_unix') or cutoff)<=end]
        api_seconds=union_seconds((r['started_unix'],r['completed_unix']) for r in calls)
        remote_seconds=sum(r['elapsed_seconds'] for r in owned)
        local_seconds=sum(r.get('local_evaluation_and_diagnostics_seconds',0) for r in owned)
        total=end-start
        row={'phase':phase,'module':module,'label_zh':labels[module],
            'status':'complete' if decision else 'skipped' if skipped else 'running',
            'start_local':clock(start),'end_local':clock(end),'total_minutes':total/60,
            'API_minutes':api_seconds/60,'API_calls':len(calls),'remote_minutes':remote_seconds/60,
            'local_evaluation_and_diagnostics_minutes':local_seconds/60,
            'other_or_uninstrumented_minutes':(total-api_seconds-remote_seconds-local_seconds)/60,
            'fresh_cohorts':len(owned),'fresh_attempts_complete':sum(r['completed_attempts'] for r in owned)}
        if row['other_or_uninstrumented_minutes'] < -.1:
            raise ValueError('Overlapping intervals would invalidate a module decomposition: '+module)
        modules.append(row)
    table(output/'module_timing.csv',modules)
    api_seconds=union_seconds((r['started_unix'],r['completed_unix']) for r in api)
    remote_seconds=sum(r['elapsed_seconds'] for r in cohorts)
    local_seconds=sum(r.get('local_evaluation_and_diagnostics_seconds',0) for r in cohorts)
    total=cutoff-state['started_unix']
    samples=remote['samples'];a,b=samples[0],samples[-1];interval=b['unix']-a['unix']
    before={r['pid']:r for r in a['processes']}
    process_usage=[{'pid':r['pid'],'cpu_percent_one_core':100*(r['cpu_seconds']-before[r['pid']]['cpu_seconds'])/interval,
        'wchan_first':before[r['pid']]['wchan'],'wchan_last':r['wchan'],'rss_GiB':r['rss_bytes']/2**30,
        'drm_devices':r.get('drm_devices',[])} for r in b['processes'] if r['pid'] in before]
    grouped=defaultdict(list)
    for r in batches:grouped['M2' if '_controls_' in r['cohort'] and r['arm']=='gradient'
        else 'unguided' if r['arm']=='unguided' else 'candidate_gradient'].append(r['seconds'])
    summary={'observed_local':clock(cutoff),'elapsed_hours':total/3600,
        'local_state_at_capture':{k:state.get(k) for k in ['status','phase','task','api_calls']},
        'completed_modules':len(state['pruning']),'skipped_modules':len(state['skipped_modules']),
        'completed_cohorts':sum(r['status']=='complete' for r in cohorts),
        'completed_attempts':sum(r['completed_attempts'] for r in cohorts),
        'total_remote_hours_including_active':remote_seconds/3600,
        'API_seconds_union':api_seconds,'API_receipts_with_timing':len(api),
        'API_latency_seconds':describe([r['seconds'] for r in api]),
        'API_by_role':{role:describe([r['seconds'] for r in api if r['role']==role]) for role in sorted({r['role'] for r in api})},
        'local_evaluation_and_diagnostics_seconds':local_seconds,
        'wall_breakdown_percent':{'remote_inference_packaging':100*remote_seconds/total,'API':100*api_seconds/total,
            'local_evaluation_and_diagnostics':100*local_seconds/total,
            'other_or_uninstrumented':100*(total-remote_seconds-api_seconds-local_seconds)/total},
        'batch_seconds':{k:describe(v) for k,v in grouped.items()},
        'process_usage':process_usage,'sample_seconds':interval,'free_GiB':remote['free_bytes']/2**30,
        'measurement_limits':['Batch seconds include integration, recording, decoding and terminal affinity rescore.',
            'No separate forward/backward/matching/MMFF/PoseBusters timers exist in this running version.',
            'Completion-to-evaluating interval combines polling, connection and transfer; not pure network time.',
            'Residual includes Git, tools, statistics, cleanup, deployment and unrecorded validation errors.',
            'Live module timing and attempt count stop at the capture time; no completion forecast is a measurement.'],
        'sources':{'remote_capture_sha256':digest(remote_path),'local_state_sha256':digest(state_path),
            'study_path':str(study.resolve()),'inference_source':'src/evomolsteer/generation/controller.py'}}
    write_json(output/'runtime_summary.json',summary)
    return summary
