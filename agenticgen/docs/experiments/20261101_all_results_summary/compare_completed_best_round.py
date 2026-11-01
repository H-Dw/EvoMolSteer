"""Compare only completed five-round jobs, with one full 50-row winner per job."""
from pathlib import Path
import collections
import csv
import datetime
import gzip
import json
import math

import compare_comparison5_selected as common

HERE = Path(__file__).resolve().parent


def main():
    source = common.read(HERE / '五轮原测与严格工具重测对比.json')
    all_groups = {g['folder']: g for g in common.read(common.SOURCE)['groups']}
    inputs = [HERE / '五轮原测与严格工具重测对比.json', common.SOURCE, common.CANDIDATES]
    views = collections.defaultdict(list)
    with gzip.open(common.CANDIDATES, 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if '五轮原测' in row['experiment'] or '严格工具重测' in row['experiment']:
                views[row['group']].append(row)
    rounds = {}
    quality = {}
    for local, stamp, _, label, _ in common.SUITES:
        base = common.EXPERIMENTS / local / 'monitoring' / stamp
        round_path, quality_path = base / 'rounds_and_range_events.json', base / 'quality_source_summary.json'
        inputs.extend([round_path, quality_path])
        for r in common.read(round_path):
            rounds.setdefault((label, r['job']), []).append(r)
        for q in common.read(quality_path):
            quality[(label, q['job'], q['source'])] = q

    selected, excluded = [], []
    for task in source['tasks']:
        if task['rounds'] != 5:
            excluded.append(dict(suite=task['suite'], id=task['id'], strategy=task['strategy'],
                                 teacher=task['teacher'], rounds=task['rounds'],
                                 charged_attempts_preserved=task['attempts'],
                                 reason='未完成完整五轮，仅从比较主表排除，原始文件与账本不删除'))
            continue
        task_rounds = rounds[(task['suite'], task['id'])]
        assert len(task_rounds) == 5
        options = []
        for r in task_rounds:
            complete = [g for g in r['groups'] if g.get('complete') and g.get('attempts') == 50]
            if not complete:
                continue
            candidates = [g for g in complete
                          if common.numeric(g.get('ranking_condition_coordinate_root_mean_square_deviation_angstrom'))]
            assert len(candidates) == len(complete)
            winner = min(candidates,
                         key=lambda g: (g['ranking_condition_coordinate_root_mean_square_deviation_angstrom'], g['id']))
            performance = winner['ranking_condition_coordinate_root_mean_square_deviation_angstrom']
            assert common.close(performance, r['checkpoint']['performance'])
            options.append((performance, r['iteration'], r, winner))
        _, iteration, best_round, winner = min(options, key=lambda x: (x[0], x[1], x[3]['id']))
        best_path = task['root'] + '/' + winner['id']
        best_group = all_groups[best_path]
        assert best_group['population'] == len(views[best_path]) == 50
        assert len({r['physical_id'] for r in views[best_path]}) == 50
        for metric in common.METRICS:
            recomputed = common.aggregate(views[best_path], metric)
            for key, value in recomputed.items():
                assert common.close(value, best_group['stats'][metric].get(key))
        q = quality[(task['suite'], task['id'], winner['id'])]
        assert q['receipt']['checked_attempts'] == 10 and q['receipt']['coverage'] == .2
        assert q['receipt']['records_sha256'] == best_group['record_sha256']

        round_rows = {}
        round_groups = []
        for g in best_round['groups']:
            if not (g.get('complete') and g.get('attempts') == 50):
                continue
            path = task['root'] + '/' + g['id']
            assert len(views[path]) == 50
            round_groups.append(all_groups[path])
            for row in views[path]:
                round_rows.setdefault(row['physical_id'], row)
        whole_round = dict(groups=round_groups, unique_candidate_views=len(round_rows),
                           built=sum(r['built'] for r in round_rows.values()),
                           stats={metric: common.aggregate(list(round_rows.values()), metric)
                                  for metric in common.METRICS})
        requested = sum(r['requested_actions'] for r in task_rounds)
        events = [e for r in task_rounds for e in r.get('range_events', [])]
        range_full = sum(e['status'] == 'full_population_complete' for e in events)
        range_partial = sum(e['status'] == 'measured_not_promoted' for e in events)
        skipped = sum(e['status'] == 'budget_limited' for e in events)
        formal_groups = task['formal']['groups']
        # The actual six completed jobs have no extra/reused action outcomes to charge again.
        assert requested == formal_groups + range_full + range_partial + skipped
        accounting = dict(selected_actions=requested, formal_groups=formal_groups,
                          formal_attempts=task['attempts']['formal'],
                          range_complete50_actions=range_full,
                          range_partial20_actions=range_partial,
                          budget_skipped_actions=skipped,
                          hold_rounds=sum(r['decision'] == 'hold' for r in task_rounds))
        selected.append(dict(suite=task['suite'], skills=task['skills'], id=task['id'],
                             strategy=task['strategy'], teacher=task['teacher'], rounds_completed=5,
                             selected_round=iteration, best_configuration=best_group,
                             quality=q, best_round_all_configurations=whole_round,
                             full_run_action_accounting=accounting,
                             round_performance=[dict(iteration=r['iteration'],
                                                     best_new_complete50_ranking_rmsd=perf)
                                                for perf, _, r, _ in sorted(options, key=lambda x:x[1])]))
    selected.sort(key=lambda t: (0 if t['teacher'].startswith('6pyy') else 1,
                                 ['四窗口基线','前十步联合注入','动态匹配五窗口'].index(t['strategy'])))
    assert len(selected) == 6 and len(excluded) == 2
    assert sum(t['best_configuration']['population'] for t in selected) == 300
    assert sum(sum(e['charged_attempts_preserved'].values()) for e in excluded) == 600

    # Paired descriptive comparisons use the same frozen seed/slot, not new independent evidence.
    comparisons = []
    for teacher in ['6pyy_H7S_A_402', '3w54_RNB_A_504']:
        baseline = next(t for t in selected if t['teacher']==teacher and t['strategy']=='四窗口基线')
        for t in selected:
            if t['teacher'] != teacher or t is baseline:
                continue
            a, b = baseline['best_configuration'], t['best_configuration']
            delta = {m: b['stats'][m]['mean']-a['stats'][m]['mean'] for m in common.METRICS}
            comparisons.append(dict(teacher=teacher, strategy=t['strategy'],
                                    mean_difference_from_baseline=delta,
                                    rmsd_percent_change=100*delta['rmsd']/a['stats']['rmsd']['mean'],
                                    paired_same_seed_slot=common.pair_rows(views[a['folder']], views[b['folder']]),
                                    baseline_round=baseline['selected_round'], strategy_round=t['selected_round']))

    output = dict(timestamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
                  policy='completed-five-round-jobs; select lowest per-round best complete50 ranking RMSD; '
                         'take that round winner as primary N=50; also report all complete configurations in that round',
                  best_round_selection_uses_actual_saved_ranking=True,
                  calibration_only_best_excluded_from_iteration_selection=True,
                  source_sha256={str(p.relative_to(common.EXPERIMENTS)):common.sha(p) for p in inputs},
                  selected=selected, excluded=excluded, comparisons=comparisons,
                  primary_candidate_views=300, no_raw_files_deleted=True,
                  new_sampling=0, new_model_forwards=0, new_llm_calls=0)
    common.write(HERE/'完整任务最优轮对比.json', json.dumps(output, ensure_ascii=True, indent=2)+'\n')
    with (HERE/'完整任务最优轮对比.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer=csv.writer(stream, lineterminator='\n')
        writer.writerow(['实验','教师','套件','最优轮','统计范围','配置数','候选数','构建数','指标','有效数',
                         '均值','最优值','剔除异常后均值','最优10%均值','最优10%数','异常数','来源'])
        for t in selected:
            for label, item, count, population, built, paths in [
                ('最优轮最佳完整配置',t['best_configuration'],1,50,t['best_configuration']['built'],
                 t['best_configuration']['folder']),
                ('最优轮全部完整配置',t['best_round_all_configurations'],
                 len(t['best_round_all_configurations']['groups']),
                 t['best_round_all_configurations']['unique_candidate_views'],
                 t['best_round_all_configurations']['built'],
                 ';'.join(g['folder'] for g in t['best_round_all_configurations']['groups']))]:
                for metric in common.METRICS:
                    s=item['stats'][metric]
                    writer.writerow([t['strategy'],t['teacher'],t['suite'],t['selected_round'],label,count,population,
                                     built,common.METRICS[metric],s['n'],s['mean'],s['best'],s['filtered_mean'],
                                     s['top10_mean'],s['top10_n'],s['excluded'],paths])

    lines=['# 已完成五轮任务：最优轮结果对比','',
           '只保留完整完成五轮的六项任务；从主表排除原测中止的联合前缀6pyy（1/5）与五窗口3w54（0/5），'
           '由各自已完成五轮的严格工具重测结果进入比较。排除仅改变展示范围，原失败文件、600次历史消耗和完整账本保留。','',
           '选轮规则：以每轮新形成的完整50候选配置中最低的排名条件RMSD为轮成绩，再取成绩最低的一轮；'
           '主表只展示该轮最佳配置，统一每实验50候选。三个指标均在这同一配置上计算，不能分别选不同轮。'
           '最优轮全部分支另列，避免误称整个轮次总是50候选。','',
           '## 为什么先前各实验的总生成数量不同','',
           '五轮是五次决策，不是固定生成5×3×50。每轮最多三个动作，允许少选或hold；'
           '范围取证与正式采样分账，未晋级范围只形成20候选视图，预算不足的动作不采样。'
           '此前又汇总了所有轮次，并包含两项中止任务，所以总样本数不同。下表为六项完整任务的实际动作核对。','',
           common.table(['实验','教师','选中动作','正式50人组','正式总尝试','范围晋级50','范围止于20','预算跳过','hold轮次'],
                        [[t['strategy'],t['teacher'].split('_')[0],
                          *[t['full_run_action_accounting'][k] for k in ['selected_actions','formal_groups','formal_attempts',
                            'range_complete50_actions','range_partial20_actions','budget_skipped_actions','hold_rounds']]]
                         for t in selected]),'',
           '精确重复配置沿用原测量，不能将引用再次计为新生成。本表的六项动作数均已按正式/范围/跳过逐项对齐；'
           '构建失败继续占候选分母，不能靠补样拉齐数量。','',
           '## 主表：最优轮的最佳完整配置，各50候选','',
           common.table(['实验','教师','版本','最优轮','实际scale','比较候选','构建','PoseB抽查'],
                        [[t['strategy'],t['teacher'].split('_')[0],t['suite']+'/'+t['skills'],t['selected_round'],
                          str(t['best_configuration']['schedule']),50,f"{t['best_configuration']['built']}/50",
                          f"{t['quality']['receipt']['posebusters_pass']}/{t['quality']['receipt']['checked_attempts']}"]
                         for t in selected]),'']
    for metric,name in common.METRICS.items():
        lines.extend(['### '+name,'',
                      common.table(['实验','教师','有效数','均值','最优值','剔除异常后均值','最优10%均值','异常数'],
                                   [[t['strategy'],t['teacher'].split('_')[0],t['best_configuration']['stats'][metric]['n'],
                                     *[common.f(t['best_configuration']['stats'][metric][k],6)
                                       for k in ['mean','best','filtered_mean','top10_mean']],
                                     t['best_configuration']['stats'][metric]['excluded']] for t in selected]),''])
    lines.extend(['基线3w54主表采用第5轮新采样的[90,90,90,80]，RMSD为0.279807 Å；'
                  '先前0.279239 Å来自初始标定[90,90,90,90]，不是某一优化轮的新结果，本表没有将它充作最优轮。'
                  '基线6pyy第1轮的uniform90来自Agent选择的范围晋级50，其中10个槽位已有粗筛结果、40个为晋级新增；'
                  '它是完整50候选配置视图，不能记为该轮重新生成50个。','',
                  '## 相对同教师四窗口基线：主表均值变化','',
                  common.table(['实验','教师','RMSD变化 Å','RMSD相对变化','pIC50变化','SA变化'],
                               [[t['strategy'],t['teacher'].split('_')[0],common.f(t['mean_difference_from_baseline']['rmsd'],6),
                                 common.f(t['rmsd_percent_change'],2)+'%',common.f(t['mean_difference_from_baseline']['pic50'],6),
                                 common.f(t['mean_difference_from_baseline']['sa'],6)] for t in comparisons]),'',
                  '相同seed/slot用于描述性配对；不同scale、版本与探索次数仍然不同。选取最优轮是对已用样本的选择，'
                  '不是独立验证，也不代表等搜索成本。','',
                  '## 补充：最优轮全部完整分支汇总','',
                  '以下每格依次为均值／最优值／剔除异常后均值／最优10%均值；这张表合并所选轮的全部完整50人分支，'
                  '样本数仍可能不同，因此与统一50候选主表分开。','',
                  common.table(['实验','教师','最优轮','完整配置数','候选数（SA有效）','RMSD Å','pIC50','SA'],
                               [[t['strategy'],t['teacher'].split('_')[0],t['selected_round'],
                                 len(t['best_round_all_configurations']['groups']),
                                 f"{t['best_round_all_configurations']['unique_candidate_views']}（{t['best_round_all_configurations']['stats']['sa']['n']}）",
                                 *[common.quartet(t['best_round_all_configurations']['stats'][m]) for m in common.METRICS]]
                                for t in selected]),'',
                  '## 统计与文件','',
                  '- RMSD：原九个PLIP条件重原子的终态几何Hungarian覆盖误差；pIC50为保存的终态affinity.pic50预测，未重新推理或重评分。',
                  '- SA使用既有标准RDKit评分，低值更易合成；只对构建成功分子计数，不以0填缺，也不作为三维构象分数。',
                  '- 前10%在原始有限值中独立取最优，向上取整。主表RMSD/pIC50各50观测，SA可能49或48；各指标均取5个。',
                  '- 沿用原统计过滤：完整配置至少10观测；RMSD仅高侧Q3+max(3IQR,1Å)，pIC50/SA双侧3IQR；异常超过10%不输出过滤均值，不将构建失败或缺测算成统计异常。',
                  '- 六个最佳配置的现有PoseB均为10/10通过，内部几何、蛋白接触/冲突、内部能量分项也各10/10；只检查固定10/50，不外推其余40。',
                  '- 此次只做已有数据筛选、复算和展示，没有采样、模型前向、LLM调用或删除实验。', '',
                  '[逐指标CSV（含主表与整轮汇总）](完整任务最优轮对比.csv)｜'
                  '[完整来源、SHA、动作账本、质量回执与配对读数JSON](完整任务最优轮对比.json)｜'
                  '[先前全轮次统计，供追溯](五轮原测与严格工具重测对比.md)',''])
    common.write(HERE/'完整任务最优轮对比.md','\n'.join(lines))
    print(json.dumps(dict(completed_tasks=len(selected),excluded_tasks=len(excluded),primary_candidates=300,
                          all_primary_populations50=True, no_new_sampling=True,
                          winner_rounds=[dict(strategy=t['strategy'],teacher=t['teacher'],round=t['selected_round']) for t in selected]),
                     ensure_ascii=True))


if __name__ == '__main__':
    main()
