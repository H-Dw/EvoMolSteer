"""Render measurements and prompt readouts; consumes local immutable exports."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import collections, csv, gzip, hashlib, json, math, statistics

BASE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')


def csv_write(name, rows):
    with (BASE/name).open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def prompt_context(call):
    for message in call['first_actual_request_messages']:
        if message['role'] == 'user':
            try:
                value = json.loads(message['content'])
            except (ValueError, TypeError):
                continue
            if isinstance(value, dict) and 'calibration_baseline' in value:
                return value
    return None


def main():
    data = read(BASE/'final_results.json')
    evidence = read(BASE/'downloads/20261102_0826/actual_evidence.json')
    rows = [json.loads(line) for line in gzip.open(BASE/'downloads/20261102_0826/candidate_rows.jsonl.gz', 'rt')]
    groups = {g['folder']:g for g in data['groups']}
    bygroup = collections.defaultdict(dict)
    for row in rows:
        bygroup[row['folder']][(row['seed'], row['slot'])] = row
    controls = read(BASE/'monitoring/control_audit_2026-11-02T082628.905444_0800.json')
    assert not controls['errors']
    comparison, summaries, selected, actions, negative, calibration = [], [], [], [], [], []
    for teacher in evidence:
        target, best = teacher['teacher'], teacher['global_best']
        winner = next(g for g in groups.values() if g['teacher'] == target and g['folder'].endswith(best['representative']))
        baseline = next(g for g in groups.values() if g['teacher'] == target and g['folder'].endswith(best['calibration_baseline']['source_id']))
        references = [baseline] + [g for g in groups.values() if g['teacher'] == target and g['label'] in ('四窗口基线', '六窗口同步化学·先前最优')]
        for ref in references:
            paired = {}
            for metric in ('rmsd', 'pic50', 'sa'):
                common = sorted(set(bygroup[ref['folder']]) & set(bygroup[winner['folder']]))
                deltas = [bygroup[winner['folder']][key][metric]-bygroup[ref['folder']][key][metric]
                          for key in common if bygroup[winner['folder']][key][metric] is not None and bygroup[ref['folder']][key][metric] is not None]
                paired[metric] = dict(n=len(deltas), mean_change=statistics.mean(deltas),
                    improved=sum(d > 0 if metric == 'pic50' else d < 0 for d in deltas),
                    worse=sum(d < 0 if metric == 'pic50' else d > 0 for d in deltas), unchanged=sum(d == 0 for d in deltas))
            comparison.append(dict(teacher=target, reference=ref['label'], source=ref['folder'], winner=winner['folder'],
                rmsd_mean_percent_change=100*(winner['metrics']['rmsd']['mean']/ref['metrics']['rmsd']['mean']-1), paired=paired))
        selected.extend([g for g in groups.values() if g['teacher'] == target and g['label'] == '四窗口基线'] + [baseline, winner])
        control = next(t for t in controls['jobs'] if t['job_id'] == teacher['job'])
        round_summaries = []
        for rc in sorted(control['rounds'], key=lambda x:x['iteration']):
            iteration = rc['iteration']
            matches = sorted([g for g in groups.values() if g['teacher'] == target and '/iteration_%03d/' % iteration in g['folder'] and g['label'].startswith('多样组合')], key=lambda g:g['folder'])
            assert len(matches) == 3
            rb = min(matches, key=lambda g:g['ranking'])
            round_summaries.append(dict(iteration=iteration, round_best=rb['ranking'], best_source=rb['folder'],
                                        input_best=rc['input_best'], tiers=rc['tiers'], schedules=rc['target_profiles']))
            for i, group in enumerate(matches):
                actions.append(dict(teacher=target, round=iteration, action=i+1, tier=rc['tiers'][i],
                    actual_scale=json.dumps(group['schedule']), rank_rmsd=group['ranking'], raw_rmsd=group['metrics']['rmsd']['mean'],
                    pic50_mean=group['metrics']['pic50']['mean'], sa_mean=group['metrics']['sa']['mean'],
                    built=group['built'], connected=group['connected'], source=group['folder']))
        calls = []
        for call in teacher['actual_role_calls']:
            context = prompt_context(call)
            snapshot = None
            if context:
                assert context['calibration_baseline']['performance'] == best['calibration_baseline']['performance']
                learning = context['checkpoint_learning']
                snapshot = {k:context.get(k) for k in ('calibration_baseline', 'global_best_performance', 'incumbent_id',
                    'remaining_formal_attempts', 'no_future_execution', 'portfolio_diversity', 'decision_methodology', 'search_objective')}
                snapshot['checkpoint_learning'] = learning
            calls.append(dict(output=call['output'], output_sha256=call['output_sha256'], decision=call['decision'],
                first_actual_messages_sha256=hashlib.sha256(json.dumps(call['first_actual_request_messages'], sort_keys=True).encode()).hexdigest(),
                actual_prompt_readout=snapshot, receipt_mode=call['receipt']['final_response_mode'],
                repairs=call['receipt']['interface_repairs']))
        formal = [g for g in groups.values() if g['teacher'] == target and g['label'].startswith('多样组合')]
        full = [g for g in groups.values() if g['teacher'] == target and (g['label'].startswith('多样组合') or g['label'].startswith('负面'))]
        qkeys = ('checked_attempts','not_selected','unbuilt_selected','built','posebusters_pass','posebusters_evaluable',
                 'posebusters_unevaluable_built','severe_clash','clash_evaluable','plip_evaluable','plip_unevaluable_built')
        summaries.append(dict(teacher=target, best=best, winner_source=winner['folder'],
            actual_budget=dict(preexperiment=teacher['search']['consumed_attempts'], formal=750), rounds=round_summaries,
            actual_roles=calls, native_requests=control['native_requests'], accepted_receipts=len(control['accepted_native_receipts']),
            selected_tiers=dict(collections.Counter(tier for r in control['rounds'] for tier in r['tiers'])),
            all_full_source_quality={k:sum(g['quality']['receipt'].get(k,0) for g in full) for k in qkeys},
            formal_quality={k:sum(g['quality']['receipt'].get(k,0) for g in formal) for k in qkeys},
            formal_built=sum(g['built'] for g in formal), formal_connected=sum(g['connected'] for g in formal),
            eligible_boxes=teacher['negative_evidence']['eligible_box_ids'], frozen=teacher['frozen']))
        for box, values in teacher['negative_evidence']['coarse_contrasts'].items():
            for item in values:
                eff=item['effect']
                negative.append(dict(teacher=target, box=int(box), retained=item['retained'], native_off=item['native_off'],
                    retention_cost=eff['raw_retention_cost'], filtered_cost=eff['filtered_retention_cost'],
                    threshold=eff['material_threshold'], positive_seeds=eff['positive_seeds'],
                    interval_low=eff['descriptive_cluster_interval'][0], interval_high=eff['descriptive_cluster_interval'][1]))
        for measurement in teacher['search']['measurements']:
            g=measurement['row']
            calibration.append(dict(teacher=target, profile=measurement['name'], population=measurement['population'],
                raw_rmsd=g['condition_coordinate_root_mean_square_deviation_angstrom'],
                rank_rmsd=g['ranking_condition_coordinate_root_mean_square_deviation_angstrom'],
                built=g['built'], connected=g['connected'], schedule=json.dumps(g['effective_guidance_scale_schedule']), source=g['id']))
    write(BASE/'paired_comparison.json', comparison)
    write(BASE/'reasoning_readout.json', dict(timestamp=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        actual_evidence_sha256=sha(BASE/'downloads/20261102_0826/actual_evidence.json'), control_audit_errors=[], teachers=summaries,
        comparison_scope='Same fixed initial panel; adaptive configurations, paired reproduction, not independent validation.'))
    csv_write('formal_actions.csv', actions)
    csv_write('negative_contrasts.csv', negative)
    csv_write('preexperiment_readings.csv', calibration)
    longrows=[]
    table=['# 完整五轮任务的最优配置指标', '',
        '每个配置50候选；按RMSD选择配置，其他指标沿用同一组。每指标的最优10%单独排序。原始/缺测/异常分母均保留在机器结果中。', '']
    for metric, title in [('rmsd','条件几何RMSD Å↓'), ('pic50','保存终态预测pIC50↑'), ('sa','标准合成可及性SA↓')]:
        table += ['## '+title, '', '| 教师 | 实验关键词 | 有效数/50 | 均值 | 最优值 | 去异常均值 | 最优10%均值 |',
                  '| --- | --- | --- | --- | --- | --- | --- |']
        for group in selected:
            s=group['metrics'][metric]
            label='多样组合·最优第'+('4' if '6pyy' in group['teacher'] else '5')+'轮' if group['label'].startswith('多样组合') else group['label']
            values=['NA' if s.get(k) is None else f'{s[k]:.6f}' for k in ('mean','best','filtered_mean','top10_mean')]
            table.append('| '+' | '.join([group['teacher'].split('_')[0], label, str(s['n'])+'/50']+values)+' |')
            longrows.append(dict(teacher=group['teacher'], experiment=label, metric=metric, n=s['n'],
                mean=s['mean'], best=s['best'], filtered_mean=s['filtered_mean'], top10_mean=s['top10_mean'], source=group['folder']))
        table.append('')
    table += ['RMSD为蛋白配准后九条件点的终态动态Hungarian几何覆盖，未用配体自身拟合。pIC50为保存模型预测，未重评分。标准SA不是三维构象合理性；构象检查另见复盘。', '',
        '去异常规则：至少10有效观测，RMSD仅上侧Q3+max(3IQR,1Å)，pIC50/SA双侧3IQR；最多剔除10%，超限不报过滤均值，构建失败/非有限值不计异常。前10%按各指标独立选取。', '']
    (BASE/'final_metrics.zh-CN.md').write_text('\n'.join(table), encoding='utf-8', newline='\n')
    csv_write('comparison_metrics.csv', longrows)
    print(json.dumps(dict(teachers=[{k:t[k] for k in ('teacher','actual_budget','selected_tiers','formal_built','formal_quality','all_full_source_quality')} for t in summaries], comparisons=comparison), ensure_ascii=False))


if __name__ == '__main__':
    main()
