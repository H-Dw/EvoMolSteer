"""Compare saved five-round suites; no model, sampler, provider or remote mutation."""
from pathlib import Path
import collections
import csv
import datetime
import gzip
import hashlib
import json
import math
import statistics

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
SOURCE = HERE / 'raw/all_results_deduplicated.json'
CANDIDATES = HERE / 'raw/candidate_rows.jsonl.gz'
SUITES = [
    ('20261031_guidance_comparison5', '20261031_2017',
     '/workspace/agenticgen_guidance_comparison5_20261031', '五轮原测', '2.8.1'),
    ('20261031_selected_contract_retest', '20261101_0221',
     '/workspace/agenticgen_guidance_selected_retest_20261031', '严格工具重测', '2.8.2'),
]
METRICS = {'rmsd': '条件RMSD Å↓', 'pic50': '预测pIC50↑', 'sa': '标准SA↓'}
NAMES = {'baseline': '四窗口基线', 'prefix_joint': '前十步联合注入',
         'five_windows': '动态匹配五窗口'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, content):
    path.write_bytes(content.encode('utf-8'))


def numeric(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def aggregate(rows, metric):
    good = [(r, float(r[metric])) for r in rows if numeric(r[metric])]
    if not good:
        return dict(n=0, missing=len(rows), mean=None, best=None,
                    filtered_mean=None, top10_mean=None, top10_n=0, excluded=0)
    vals = [v for _, v in good]
    top = sorted(vals, reverse=metric == 'pic50')[:math.ceil(.1 * len(vals))]
    kept = [v for r, v in good if not r['excluded'][metric]]
    eligible = all(r['eligible'][metric] for r in rows)
    return dict(n=len(vals), missing=len(rows)-len(vals), mean=statistics.mean(vals),
                best=(max(vals) if metric == 'pic50' else min(vals)),
                filtered_mean=statistics.mean(kept) if kept and eligible else None,
                top10_mean=statistics.mean(top), top10_n=len(top),
                excluded=sum(r['excluded'][metric] for r in rows), filter_eligible=eligible)


def close(a, b):
    if a is None or b is None:
        return a is b
    return math.isclose(a, b, rel_tol=0, abs_tol=1e-10)


def f(value, digits=4):
    return '—' if value is None else f'{value:.{digits}f}'


def quartet(stats):
    return ' / '.join(f(stats.get(k)) for k in ['mean', 'best', 'filtered_mean', 'top10_mean'])


def task_name(task_id):
    return next(v for k, v in NAMES.items() if k in task_id) + '·' + ('6pyy' if '6pyy' in task_id else '3w54')


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def pair_rows(before, after):
    a = {(r['seed'], r['slot']): r for r in before}
    b = {(r['seed'], r['slot']): r for r in after}
    assert len(a) == len(before) and len(b) == len(after)
    assert a.keys() == b.keys()
    result = {}
    for metric in METRICS:
        pairs = [(a[k][metric], b[k][metric]) for k in a
                 if numeric(a[k][metric]) and numeric(b[k][metric])]
        deltas = [y-x for x, y in pairs]
        result[metric] = dict(
            paired_finite=len(pairs), missing_pair=len(a)-len(pairs),
            mean_paired_change=statistics.mean(deltas),
            improved=sum(d > 0 if metric == 'pic50' else d < 0 for d in deltas),
            worse=sum(d < 0 if metric == 'pic50' else d > 0 for d in deltas),
            unchanged=sum(d == 0 for d in deltas))
    return result


def main():
    data = read(SOURCE)
    groups = {g['folder']: g for g in data['groups']}
    selected = [e for e in data['experiments']
                if '五轮原测' in e['experiment'] or '严格工具重测' in e['experiment']]
    source_rows = []
    with gzip.open(CANDIDATES, 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if '五轮原测' in row['experiment'] or '严格工具重测' in row['experiment']:
                source_rows.append(row)
    # Use the largest complete population for flags and the original stage for ownership.
    priority = {'标定预实验': 0, '范围取证': 1, '正式探索': 2}
    physical = collections.defaultdict(list)
    views = collections.defaultdict(list)
    for row in source_rows:
        physical[(row['experiment'], row['teacher'], row['physical_id'])].append(row)
        views[row['group']].append(row)
    dedup = []
    for rows in physical.values():
        row = dict(max(rows, key=lambda r: groups[r['group']]['population']))
        row['phase'] = min((r['phase'] for r in rows), key=priority.__getitem__)
        dedup.append(row)
    validations = []
    for item in selected:
        rows = [r for r in dedup if all(r[k] == item[k] for k in ['experiment', 'teacher', 'phase'])]
        assert len(rows) == item['unique_saved_candidates']
        assert sum(r['built'] for r in rows) == item['built']
        for metric in METRICS:
            actual = aggregate(rows, metric)
            for key, value in actual.items():
                assert close(value, item['stats'][metric].get(key)), (item['experiment'], metric, key)
        validations.append(dict(experiment=item['experiment'], teacher=item['teacher'],
                                phase=item['phase'], unique_candidates=len(rows), passed=True))

    tasks = []
    inputs = [SOURCE, CANDIDATES]
    rounds = []
    for local, stamp, root, label, skills in SUITES:
        facts_path = EXPERIMENTS / local / 'monitoring' / stamp / 'final_facts.json'
        rounds_path = facts_path.with_name('rounds_and_range_events.json')
        inputs.extend([facts_path, rounds_path])
        facts = read(facts_path)
        rounds.extend(dict(suite=label, **r) for r in read(rounds_path))
        for job in facts['jobs']:
            task_id = job['id']
            teacher = '6pyy_H7S_A_402' if '6pyy' in task_id else '3w54_RNB_A_504'
            strategy = next(v for k, v in NAMES.items() if k in task_id)
            attempt = 'attempt_002' if label == '严格工具重测' else 'attempt_001'
            task_root = f'{root}/jobs/{task_id}/{attempt}'
            task_groups = [g for g in groups.values() if g['folder'].startswith(task_root + '/')]
            formal = next((e for e in selected if e['teacher'] == teacher and
                           e['phase'] == '正式探索' and e['experiment'].endswith('·'+label)
                           and any(g['experiment'] == e['experiment'] for g in task_groups)), None)
            best = job.get('best', {})
            best_source = job.get('best_population_source', best.get('source'))
            best_group = groups[f'{task_root}/{best_source}']
            assert best_group['population'] == 50
            old_rmsd = job.get('best_raw_rmsd', best.get('raw_rmsd'))
            assert close(best_group['stats']['native_rmsd']['mean'], old_rmsd)
            attempts = job.get('charged_attempts', job.get('attempts'))
            assert sum(attempts.values()) == sum(r['group'].startswith(task_root+'/') for r in dedup)
            assert attempts['formal'] == (formal['unique_saved_candidates'] if formal else 0)
            tasks.append(dict(suite=label, skills=skills, id=task_id, strategy=strategy,
                              teacher=teacher, rounds=job['rounds_completed'], attempts=attempts,
                              root=task_root, formal=formal, best_group=best_group))
    assert sum(sum(t['attempts'].values()) for t in tasks) == 5170

    reproduction = []
    pairs = []
    for task_id in ['03_prefix_joint_6pyy', '06_five_windows_3w54']:
        old, new = [next(t for t in tasks if t['suite'] == label and t['id'] == task_id)
                    for label in ['五轮原测', '严格工具重测']]
        maps = []
        for task in [old, new]:
            rows = [r for r in dedup if r['group'].startswith(task['root']+'/') and r['phase']=='标定预实验']
            mapping = {(tuple(groups[r['group']]['schedule']), r['seed'], r['slot']): r for r in rows}
            assert len(rows) == len(mapping) == 240
            maps.append(mapping)
        assert maps[0].keys() == maps[1].keys()
        checks = {}
        for metric in ['rmsd', 'native_rmsd', 'pic50', 'sa', 'built', 'connected']:
            values = [(maps[0][k][metric], maps[1][k][metric]) for k in maps[0]]
            checks[metric] = dict(compared=240, exactly_equal=sum(a==b for a,b in values),
                                  missing_both=sum(a is None and b is None for a,b in values))
            assert checks[metric]['exactly_equal'] == 240
        reproduction.append(dict(task=task_id, matching_key='schedule, seed, logical_slot',
                                 comparisons=checks, scope='saved scalar metrics and build flags; not full tensor bitwise audit'))
        old_schedules = {tuple(g['schedule']) for g in groups.values()
                         if g['folder'].startswith(old['root']+'/') and g['phase']=='正式探索'}
        new_schedules = {tuple(g['schedule']) for g in groups.values()
                         if g['folder'].startswith(new['root']+'/') and g['phase']=='正式探索'}
        assert not old_schedules.intersection(new_schedules)
        pair = pair_rows(views[old['best_group']['folder']], views[new['best_group']['folder']])
        pairs.append(dict(task=task_id, old_source=old['best_group']['folder'],
                          new_source=new['best_group']['folder'],
                          old_best_phase=old['best_group']['phase'],
                          common_formal_schedules=0, paired_by='seed, logical_slot',
                          different_guidance_schedules=True, metrics=pair))

    output = dict(timestamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
                  source_sha256={str(p.relative_to(EXPERIMENTS)): sha(p) for p in inputs},
                  tasks=tasks, same_configuration_calibration_reproduction=reproduction,
                  best_configuration_pairs=pairs, aggregate_recomputation=validations,
                  new_sampling=0, new_model_forwards=0, new_llm_calls=0,
                  total_charged=5170, matched_formal_configuration_pairs=0)
    write(HERE/'五轮原测与严格工具重测对比.json', json.dumps(output, ensure_ascii=True, indent=2)+'\n')

    # All saved scientific phases, plus separate RMSD-selected best populations.
    csv_path = HERE/'五轮原测与严格工具重测对比.csv'
    with csv_path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['套件','实验','教师','统计范围','指标','有效数','缺测','均值','最优值',
                         '剔除异常后均值','最优10%均值','最优10%数','异常数','来源'])
        for task in tasks:
            for item in selected:
                if item['teacher'] != task['teacher'] or not item['experiment'].endswith('·'+task['suite']):
                    continue
                if not any(g['experiment'] == item['experiment'] for g in groups.values()
                           if g['folder'].startswith(task['root']+'/')):
                    continue
                for metric in METRICS:
                    s=item['stats'][metric]
                    writer.writerow([task['suite'],task['strategy'],task['teacher'],item['phase'],METRICS[metric],
                                     s['n'],s['missing'],s['mean'],s['best'],s['filtered_mean'],s['top10_mean'],
                                     s['top10_n'],s['excluded'],task['root']])
            for metric in METRICS:
                s=task['best_group']['stats'][metric]
                writer.writerow([task['suite'],task['strategy'],task['teacher'],'RMSD选定最佳50人配置',METRICS[metric],
                                 s['n'],s['missing'],s['mean'],s['best'],s['filtered_mean'],s['top10_mean'],
                                 s['top10_n'],s['excluded'],task['best_group']['folder']])

    lines = ['# 五轮原测与严格工具重测：专项结果对比', '',
             '两项严格工具重测均完成五轮；旧原测的联合前缀6pyy仅完成一轮，五窗口3w54没有正式样本。'
             '严格工具修复解决了本次漏填selected导致的停止；重测后搜索参数和样本数也发生变化，科学分数差异不能全部归因于输出接口修复。', '',
             '统计时间：'+output['timestamp']+'。只读取已有记录，无新增采样、模型前向或LLM调用。', '',
             '## 完成度、实际消耗与正式人口质量', '',
             table(['实验','教师','套件/skills','完成轮次','标定含范围','正式尝试','正式构建','正式PoseB抽查通过/抽查'],
                   [[t['strategy'],t['teacher'].split('_')[0],t['suite']+'/'+t['skills'],f"{t['rounds']}/5",
                     t['attempts']['calibration'],t['attempts']['formal'],
                     f"{t['formal']['built']}/{t['attempts']['formal']}" if t['formal'] else '无正式人口',
                     f"{t['formal']['posebusters']['passed']}/{t['formal']['posebusters']['checked']}" if t['formal'] else '未抽查'] for t in tasks]), '',
             '原测3790次（1640标定含范围+2150正式），重测1380次（580+800），累计5170次。'
             '原失败两项的360/240次完整保留；重测是新根、新验收和独立标定，没有迁移旧预算或lambda。'
             '四项已成功原测未重跑。每个实际质量源组只检查固定10/50，其余40未补查。', '',
             '## 正式探索的全部独特已保存候选', '',
             '此处汇总所有正式配置的候选，不混入标定/范围取证，也不只挑最佳checkpoint。'
             '各指标的“最优值”是单候选极值，不能等同于最佳50人配置的平均值。', '']
    for metric, name in METRICS.items():
        rows=[]
        for t in tasks:
            s=t['formal']['stats'][metric] if t['formal'] else {}
            rows.append([t['strategy'],t['teacher'].split('_')[0],t['suite'],s.get('n',0),
                         *[f(s.get(k)) for k in ['mean','best','filtered_mean','top10_mean']],s.get('excluded','—')])
        lines.extend(['### '+name, '', table(['实验','教师','套件','有效数','均值','最优值','剔除异常后均值','最优10%均值','异常数'], rows),''])
    lines.extend(['## 按条件RMSD选出的最佳完整50人配置', '',
                  '每行固定同一个RMSD优胜配置，再报告其pIC50和SA；不会为不同指标各换一个配置。'
                  '旧五窗口3w54这一行仅是标定结果，绝不是已完成Agent五轮的结果。', '',
                  table(['实验','教师','套件','scale','来源阶段','RMSD均值 Å','pIC50均值','SA均值','构建/50'],
                        [[t['strategy'],t['teacher'].split('_')[0],t['suite'],str(t['best_group']['schedule']),
                          t['best_group']['phase'],f(t['best_group']['stats']['rmsd']['mean'],6),
                          f(t['best_group']['stats']['pic50']['mean'],6),f(t['best_group']['stats']['sa']['mean'],6),
                          f"{t['best_group']['built']}/50"] for t in tasks]), '',
                  '上述最佳配置的三个指标完整四项统计见CSV；原口径RMSD和源SHA见JSON。', '',
                  table(['直接前后比较','旧最佳来源','RMSD变化 Å（新−旧）','pIC50均值变化','SA均值变化','结论'],
                        [[task_name(p['task']),p['old_best_phase'],
                          f(p['metrics']['rmsd']['mean_paired_change'],6),
                          f(next(t for t in tasks if t['id']==p['task'] and t['suite']=='严格工具重测')['best_group']['stats']['pic50']['mean']-
                            next(t for t in tasks if t['id']==p['task'] and t['suite']=='五轮原测')['best_group']['stats']['pic50']['mean'],6),
                          f(next(t for t in tasks if t['id']==p['task'] and t['suite']=='严格工具重测')['best_group']['stats']['sa']['mean']-
                            next(t for t in tasks if t['id']==p['task'] and t['suite']=='五轮原测')['best_group']['stats']['sa']['mean'],6),
                          'RMSD降低约3.21%，均值pIC50略升、SA略降' if 'prefix' in p['task'] else
                          'RMSD较旧标定低约1.89%，均值pIC50下降、SA升高'] for p in pairs]), '',
                  '前后最佳配置按同seed/slot配对，联合前缀6pyy的RMSD为31改善/19变差，'
                  '五窗口3w54为30改善/20变差。SA均值使用各自实际构建成功分母；逐槽位SA配对仅使用两侧均可测样本，另存JSON，不能混淆两种分母。'
                  '两项原测/重测没有共同的正式scale配置，因此没有同配置正式复测的因果对照。', '',
                  '## 初始标定的同配置复现核对', '',
                  table(['任务','同scale/seed/slot对数','RMSD逐项相同','pIC50逐项相同','SA及缺测状态逐项相同','构建状态逐项相同'],
                        [[task_name(r['task']),240,r['comparisons']['rmsd']['exactly_equal'],r['comparisons']['pic50']['exactly_equal'],
                          r['comparisons']['sa']['exactly_equal'],r['comparisons']['built']['exactly_equal']] for r in reproduction]), '',
                  '核对对象是保存的标量和构建状态，不冒充本次重新验证所有张量逐位相同。'
                  '相同冻结初态/随机流的跨运行结果是配对复现，不是独立验证。', '',
                  '## 统计定义与可比性', '',
                  '- RMSD：原九个PLIP条件重原子的终态几何Hungarian覆盖误差，不做配体拟合；初始原子身份误差另见两套原始复盘。统一重算与原保存RMSD仅有浮点舍入差异。',
                  '- pIC50：读取终态张量保存的affinity.pic50头输出；不是实验测定，也不是新对接或重评分。构建失败仍可有该预测；其分母与SA不同。',
                  '- SA：已计算的标准RDKit合成可及性分数，RDKit 2026.03.6；只对成功构建的分子计数，不以0补缺，不当作三维合理性。',
                  '- “最优10%”：在原始有限值中各指标独立排序，RMSD/SA取最低、pIC50取最高，数量向上取整；不是同一批候选共同最优。',
                  '- 异常过滤沿用全部结果表：每完整配置至少10个有限值；RMSD只剔除高于Q3+max(3IQR,1Å)，pIC50/SA使用双侧3IQR。若某配置异常比例超过10%，其过滤汇总不提供；缺测/构建失败不作为统计异常删除。',
                  '- PoseB只报告已有抽样布尔检查。当前七个有正式人口的条目中，唯一不通过为四窗口基线3w54的minimum_distance_to_protein；该条目内部几何、内部能量检查均通过。其他正式抽查分项全部通过，详见JSON中components，不外推未查人口。',
                  '- 所有配置使用同样两个teacher/九原子条件、raw999、完整绝对坐标VJP和UG04串行调度；skills从2.8.1到2.8.2、原生strict工具控制和实际搜索路径不同，不宣称等成本或接口修复带来全部分数改善。', '',
                  '## 文件与证据', '',
                  '- [完整逐阶段和最佳配置四项统计CSV](五轮原测与严格工具重测对比.csv)',
                  '- [机器读数、源SHA、逐槽位配对与复算校验](五轮原测与严格工具重测对比.json)',
                  '- [原测主GPT复盘](../20261031_guidance_comparison5/final_review.zh-CN.md)',
                  '- [严格工具重测主GPT复盘](../20261031_selected_contract_retest/final_review.zh-CN.md)',
                  '- [全部实验指标表](全部实验指标表.md)', ''])
    write(HERE/'五轮原测与严格工具重测对比.md','\n'.join(lines))
    print(json.dumps(dict(tasks=len(tasks), aggregate_rows_recomputed=len(validations),
                          charged_attempts=5170, matched_calibration_pairs=480,
                          matched_formal_configuration_pairs=0,
                          output_files=['五轮原测与严格工具重测对比.md','五轮原测与严格工具重测对比.csv',
                                        '五轮原测与严格工具重测对比.json']),ensure_ascii=True))


if __name__ == '__main__':
    main()
