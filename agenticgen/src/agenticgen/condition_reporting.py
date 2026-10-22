"""Condition-first reports from all retained documentary trials, without resampling."""
from pathlib import Path
import statistics
from .adaptive_schedule.io import read, write, eligible, sha, value_sha
from .adaptive_schedule.objective import condition_summary, dominates
from .adaptive_schedule.group_tools import paired, score_summary, distribution
from .adaptive_schedule.selection import choose, rollback_advice, TOLERANCE_A, SENSITIVITY_A
from .adaptive_schedule.protocol_state import load_config


def representative(rows, tolerance=TOLERANCE_A):
    pool = [dict(id=f"r{x['round']:02d}:{x['collection']}",
        condition=x['condition_rmsd_A_all_attempts']['mean'],
        functional_gap=x['mean_affinity_absolute_gap'], best=x['affinity']['max'],
        ticks=x['ticks']) for x in rows if x['collection'].endswith('_unselected')]
    return choose(pool, tolerance)


def version_report(root, version):
    root=Path(root); c=load_config(root); base=root/'versions'/version
    n=c['population_size']; rows=[]; progress=[]; costs=[]; selections={}
    for teacher in read(root/'teachers.json'):
        identity=teacher['teacher_id']; reference=read(root/'shared_r0'/identity/'best.json')
        tested=[]; gpu_seconds=0
        def append(label,round_no,trials,shared):
            result=condition_summary(trials,teacher)
            records=[t['record'] for t in trials]
            result.update(teacher=identity,version=version,round=round_no,collection=label,
                shared_schedule=shared, ticks=trials[0]['ticks'] if shared else None,
                paired_affinity_vs_R0=paired([t['record'] for t in reference],records),
                paired_condition_delta_vs_R0=distribution([
                    item['rms_A']-baseline['rms_A'] for item,baseline in zip(result['individual_condition_errors'],
                    condition_summary(reference,teacher)['individual_condition_errors'])]),
                quality_evaluated=sum(r.get('quality_evaluated',False) for r in records),
                posebusters_pass_fraction_all_attempts=sum(bool(r.get('posebusters_pass')) for r in records)/len(records),
                severe_clash_fraction_all_attempts=sum(bool(r.get('severe_clash')) for r in records)/len(records),
                mean_plip_recovery=statistics.fmean([r['plip_recovery'] for r in records if r.get('plip_recovery') is not None])
                    if any(r.get('plip_recovery') is not None for r in records) else None)
            from .reporting import chemistry
            result['chemistry']=chemistry(records)
            rows.append(result)
            if shared: tested.append(result)
        append('R0',0,reference,True)
        for r in range(1,c['rounds']+1):
            wave=base/identity/f'round_{r:02d}'/'wave'
            if not (wave/'trials.json').exists() or not (wave/'QUALITY_COMPLETE.json').exists(): continue
            trials=read(wave/'trials.json')
            if len(trials)!=4*n: raise ValueError('A reported wave must contain all attempts')
            for g in range(4): append(f'G{g+1}_unselected',r,trials[g*n:(g+1)*n],True)
            best=base/identity/'incumbents'/f'r{r:02d}'/'best.json'
            if best.exists(): append('affinity_TOP1_search_archive',r,read(best),False)
            if (wave/'SAMPLING_COMPLETE.json').exists():
                cost=read(wave/'SAMPLING_COMPLETE.json'); gpu_seconds+=cost['seconds']
                costs.append(dict(cost,teacher=identity,version=version,round=r))
            frontier=[x for x in tested if not any(dominates(y,x) for y in tested if y is not x)]
            best_geometry=min(tested,key=lambda x:x['condition_rmsd_A_all_attempts']['mean'])
            progress.append({'version':version,'teacher':identity,'completed_round':r,'hypothesis_trials':4*r,
                'candidate_attempts':4*r*n,'gpu_seconds':gpu_seconds,
                'minimum_tested_condition_mean_rmsd_A':best_geometry['condition_rmsd_A_all_attempts']['mean'],
                'associated_affinity_absolute_gap':best_geometry['mean_affinity_absolute_gap'],
                'source':{'round':best_geometry['round'],'collection':best_geometry['collection']},
                'pareto_frontier':[{'round':x['round'],'collection':x['collection'],'ticks':x['ticks'],
                    'condition_mean_rmsd_A':x['condition_rmsd_A_all_attempts']['mean'],
                    'affinity_absolute_gap':x['mean_affinity_absolute_gap']} for x in frontier],
                'warning':'Minimum and frontier select previously tested shared schedules, not independent validation.'})
            progress[-1]['condition_band_selected'] = representative(tested)
        own = [x for x in rows if x['teacher'] == identity]
        selections[identity] = {'primary':representative(own),
            'sensitivity':{str(t): representative(own,t) for t in SENSITIVITY_A}}
    result={'version':version,'rows':rows,'progress':progress,'gpu_cost':costs,
        'representatives':selections,
        'counts':{'optimization_attempts':sum(x['attempts'] for x in rows if x['collection'].endswith('_unselected'))},
        'objective':c['agent_objective'],'limitations':['No hard acceptance thresholds',
            'Teacher-specific development search; no optimizer superiority claim',
            'Affinity TOP1 archives do not measure unselected condition reconstruction']}
    finalized=root/f'FINALIZED_{version}.json'
    if finalized.exists():
        if sha(base/'report.json') != read(finalized)['report_sha256']:
            raise ValueError('Finalized report checksum changed')
        if value_sha(read(base/'report.json')) != value_sha(result):
            raise ValueError('Reanalysis would replace a finalized report; publish a separately named analysis instead')
    else:
        write(base/'report.json',result)
    lines=['# Condition reconstruction and functional agreement','',
        '| Teacher | Round | Collection | Coordinate RMSD mean Å | Mean pIC50 | Mean minus teacher | Best minus teacher | Top10% mean |',
        '|---|---:|---|---:|---:|---:|---:|---:|']
    def fmt(x): return f'{x:.6f}' if x is not None else 'NA'
    for x in rows:
        lines.append('|'+ '|'.join(map(str,[x['teacher'],x['round'],x['collection'],
            fmt(x['condition_rmsd_A_all_attempts']['mean']),fmt(x['affinity']['mean']),
            fmt(x['mean_affinity_signed_gap']),fmt(x['best_minus_teacher']),fmt(x['top10_percent_mean'])]))+'|')
    (base/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result


def analyze(root):
    root=Path(root); c=load_config(root); versions=[]
    for version in c['versions']:
        if (root/'versions'/version).exists(): versions.append(version_report(root,version))
    baselines=[]
    for teacher in read(root/'teachers.json'):
        for arm in ['unguided','steer']:
            records=read(root/'baselines'/f"{teacher['target']}__{arm}.json")
            baselines.append({'teacher':teacher['teacher_id'],'arm':arm,**score_summary(records),
                'sample_count_note':'Historical 100 outcomes, reused once per target; descriptive comparison, not equal-budget Best or slot pairing.'})
    receipts={}; unknown=[]
    for file in (root/'versions').rglob('*.receipt.json'):
        item=read(file)
        if item.get('usage'):
            receipts[item.get('provider_request_id') or str(file.relative_to(root))]={
                'path':str(file.relative_to(root)),'usage':item['usage']}
    for pattern in ['*.invalid.txt','*.incomplete_*.txt']:
        for f in (root/'versions').rglob(pattern):
            if not f.with_suffix('.receipt.json').exists(): unknown.append(str(f.relative_to(root)))
    out=root/'reports';out.mkdir(exist_ok=True)
    write(out/'condition_comparisons.json',versions);write(out/'historical_baselines.json',baselines)
    write(out/'llm_usage.json',receipts);write(out/'llm_missing_receipts.json',unknown)
    completed=[]
    for report in versions:
        name=report['version']
        if not (root/f'FINALIZED_{name}.json').exists(): continue
        selected=[s['primary']['selected'] for s in report['representatives'].values()]
        expected=len(read(root/'teachers.json'))*c['rounds']*4*c['population_size']
        complete=bool(selected) and all(selected) and report['counts']['optimization_attempts']==expected
        selected=selected if complete else []
        approval_path=root/'versions'/name/'review_approval.json'
        approval=read(approval_path) if approval_path.exists() else {}
        completed.append({'id':name,
            'condition':statistics.fmean(s['condition'] for s in selected) if selected else None,
            'functional_gap':statistics.fmean(s['functional_gap'] for s in selected) if selected else None,
            'best':statistics.fmean(s['best'] for s in selected) if selected else None,
            'rollback_completed':bool(approval.get('rollback',{}).get('restored_version')),
            'teacher_selections':report['representatives']})
    write(out/'version_decisions.json',{'completed_versions':completed,'rollback':rollback_advice(completed),
        'ranking_tolerance_A':TOLERANCE_A,'not_an_automatic_scientific_approval':True})
    (out/'report.md').write_text('# Continuous condition reconstruction\n\n'+
        '\n'.join(f'- [{v["version"]}](../versions/{v["version"]}/report.md)' for v in versions)+
        '\n\nCondition error is primary; mean-affinity teacher agreement is the functional objective. No hard success thresholds. '+
        'Best and tail scores are bonuses. Historical 100-result baselines have different budgets.\n',encoding='utf-8')
    return {'versions':len(versions),'output':str(out),'objective':'condition_first_no_thresholds'}
