"""Audit and retain a scale experiment comparison, including incomplete controls."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from evomolsteer.io import digest, read_json, write_json, write_table
from evomolsteer.generation.path_evaluation import retain_round
from evomolsteer.generation.scale_comparison import compare_completed_panel


def table(rows: dict) -> list[str]:
    lines = ['|组别|N|全部尝试 pIC50 均值|有效/PB fast|有效分数 P95/最高|极高分有效数/独立图|应变中位/P90|',
             '|---|---:|---:|---|---|---|---|']
    for label, m in rows.items():
        lines.append(f'|{label}|{m["n"]}|{m["all_mean_pic50"]:.6f}|'
                     f'{m["valid_rate"]:.2%}/{m["pb_fast_rate"]:.2%}|'
                     f'{m["valid_p95_pic50"]:.6f}/{m["valid_max_pic50"]:.6f}|'
                     f'{m["elite_valid_n"]}/{m["elite_unique_graphs"]}|'
                     f'{m["strain_median_per_heavy"]:.6f}/{m["strain_p90_per_heavy"]:.6f}|')
    return lines


def plot_comparison(output: Path, frames: dict, full: dict) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'R11': '#2166ac', 'R26': '#67a9cf', 'unguided': '#999999',
              'historical_Steer': '#b2182b'}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), layout='constrained')
    for index, (label, frame) in enumerate(frames.items()):
        axes[0].bar(index, frame.pic50_on_rescore.mean(), color=colors[label])
        axes[0].scatter(np.full(frame.batch.nunique(), index),
                        frame.groupby('batch').pic50_on_rescore.mean(),
                        color='black', s=16, alpha=.6, zorder=3)
        energy = frame[frame.valid_connected & frame.energy_status.eq('converged')]
        x = np.sort(energy.mmff_relief_per_heavy.dropna().to_numpy())
        axes[1].step(x, np.arange(1, len(x)+1)/len(x), where='post',
                     color=colors[label], label=label)
    axes[0].set_xticks(range(len(frames)), frames.keys())
    axes[0].set_ylim(7.0, 7.8)
    axes[0].set_ylabel('Mean predicted pIC50 (all attempts)')
    axes[0].set_title(f'Paired completed panel, N={len(next(iter(frames.values())))} / arm')
    axes[1].set_xlim(0, 2.5)
    axes[1].set_xlabel('MMFF local relief / heavy atom (kcal/mol)')
    axes[1].set_ylabel('Cumulative fraction (converged valid poses)')
    axes[1].set_title('Paired panel: smaller strain is better')
    axes[1].legend()
    for label, frame in full.items():
        x = np.sort(frame[frame.valid_connected].pic50_on_rescore.to_numpy())
        axes[2].step(x, (len(x)-np.arange(len(x)))/len(x), where='post',
                     color=colors[label], label=f'{label} (N={len(frame)})')
    axes[2].set_xlim(7.7, 8.8)
    axes[2].set_ylim(0, .35)
    axes[2].set_xlabel('Predicted pIC50 threshold')
    axes[2].set_ylabel('Fraction of valid poses above threshold')
    axes[2].set_title('Full upper tails: unpaired historical reference')
    axes[2].legend()
    fig.savefig(output/'comparison.png', dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--r11-dataset', required=True)
    parser.add_argument('--r11-evaluated', required=True)
    parser.add_argument('--controls-dataset', required=True)
    parser.add_argument('--controls-evaluated', required=True)
    parser.add_argument('--steer-metrics', required=True)
    parser.add_argument('--steer-reference', required=True,
                        help='Frozen historical metadata with expected source SHA256')
    parser.add_argument('--protocol', required=True, help='Existing frozen elite threshold')
    parser.add_argument('--spec', required=True)
    parser.add_argument('--status-inspection', help='Later read-only status; does not expand evaluated snapshot')
    parser.add_argument('--output', required=True, help='New compact report directory')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    threshold = read_json(args.protocol)['tail_threshold_pic50']
    spec = read_json(args.spec)
    steer_reference = read_json(args.steer_reference)
    if digest(args.steer_metrics) != steer_reference['source_sha256']:
        raise ValueError('Frozen historical Steer metrics changed')
    historical_report = Path(args.steer_metrics).parent/'terminal_report.json'
    evaluation_definitions = ('reference_sha256', 'energy_definition',
                              'affinity_definition', 'posebusters_definition')
    for key in evaluation_definitions:
        if read_json(historical_report)[key] != read_json(Path(args.r11_evaluated)/'terminal_report.json')[key]:
            raise ValueError(f'Historical Steer evaluation differs: {key}')
    croot = Path(args.controls_dataset)/'results/R26_native'
    scope = read_json(croot/'SNAPSHOT.json')
    if scope['original_expected_n_per_arm'] != spec['n_per_arm']:
        raise ValueError('Snapshot planned N differs from registered experiment')
    input_digests = []
    for dataset, evaluated, campaign, cohort in [
        (args.r11_dataset, args.r11_evaluated, 'R11', spec['cohorts'][0]),
        (args.controls_dataset, args.controls_evaluated, 'R26_native', spec['cohorts'][1]),
    ]:
        cfg = read_json(Path(dataset)/f'results/{campaign}/config.json')
        report = read_json(Path(evaluated)/'terminal_report.json')
        if cfg['extension']['program_sha256'] != cohort['program_sha256']:
            raise ValueError('Reward program differs from frozen scale specification')
        if cfg['extension']['checkpoint_sha256'] != spec['checkpoint_sha256']:
            raise ValueError('FLOWR model checkpoint differs')
        if cfg['experiment']['batch'] != spec['batch'] or cfg['experiment']['steps'] != spec['steps']:
            raise ValueError('Integrator steps / batch size differ')
        input_digests.append({p.name: digest(p) for p in (Path(dataset)/'inputs').iterdir() if p.is_file()})
        if report['reference_sha256'] != read_json(Path(args.r11_evaluated)/'terminal_report.json')['reference_sha256']:
            raise ValueError('Structural evaluation reference differs between cohorts')
    if input_digests[0] != input_digests[1]:
        raise ValueError('Aligned ligand / receptor inputs differ')
    r11 = pd.read_csv(Path(args.r11_evaluated)/'candidate_metrics.csv')
    controls = pd.read_csv(Path(args.controls_evaluated)/'candidate_metrics.csv')
    steer = pd.read_csv(args.steer_metrics)
    if len(r11) != spec['n_per_arm'] or r11.arm.unique().tolist() != ['gradient']:
        raise ValueError('Full registered R11 cohort required')
    if set(controls.arm.unique()) != {'gradient', 'unguided'}:
        raise ValueError('Both control arms required')
    output.mkdir(parents=True)
    for dataset, campaign, evaluated, name in [
        (args.r11_dataset, 'R11', args.r11_evaluated, 'R11_full'),
        (args.controls_dataset, 'R26_native', args.controls_evaluated, 'controls_snapshot'),
    ]:
        retain_round(dataset, campaign, evaluated, output/name, threshold)
    result = compare_completed_panel(
        r11, controls[controls.arm.eq('gradient')], controls[controls.arm.eq('unguided')],
        steer, threshold, scope,
        read_json(output/'R11_full/execution_report.json'),
        read_json(output/'controls_snapshot/execution_report.json'),
    )
    sources = {str(path): digest(path) for path in [
        Path(args.r11_evaluated)/'candidate_metrics.csv',
        Path(args.controls_evaluated)/'candidate_metrics.csv', Path(args.steer_metrics),
        Path(args.steer_reference), historical_report, Path(args.protocol), Path(args.spec),
        croot/'SNAPSHOT.json', Path(__file__),
        Path(__file__).parents[1]/'src/evomolsteer/generation/scale_comparison.py']}
    if args.status_inspection:
        result['latest_status_inspection'] = read_json(args.status_inspection)
        sources[args.status_inspection] = digest(args.status_inspection)
    result['sources_sha256'] = sources
    result['integrity']['same_aligned_inputs'] = True
    result['integrity']['model_checkpoint_sha256'] = spec['checkpoint_sha256']
    execution_r11 = read_json(output/'R11_full/execution_report.json')
    execution_controls = read_json(output/'controls_snapshot/execution_report.json')
    result['guidance_feedback'] = {
        label: {'controlled_steps_per_batch': sorted({v['controlled_steps'] for v in rows}),
                'nonzero_steps_per_batch': sorted({v['nonzero_steps'] for v in rows}),
                'mean_cumulative_rms_A': float(np.mean([v['mean_cumulative_rms_A'] for v in rows]))}
        for label, rows in [
            ('R11', execution_r11['batch_results']),
            ('R26', [v for v in execution_controls['batch_results'] if v['arm'] == 'gradient']),
            ('unguided', [v for v in execution_controls['batch_results'] if v['arm'] == 'unguided'])]}
    scope_small = {key: value for key, value in scope.items() if key != 'source_files_sha256'}
    result['snapshot_scope'] = scope_small
    write_json(output/'comparison.json', result)
    write_json(output/'snapshot_scope.json', scope_small)
    write_table(output/'summary.csv', [dict(scope='paired_completed_panel', arm=k, **v)
                                     for k, v in result['paired_panel'].items()]+
                [dict(scope='full_unpaired', arm=k, **v) for k, v in result['full_unpaired'].items()])
    frames = {'R11': r11[r11.batch.isin(scope['included_batches'])],
              'R26': controls[controls.arm.eq('gradient')],
              'unguided': controls[controls.arm.eq('unguided')]}
    plot_comparison(output, frames, {'R11': r11, 'historical_Steer': steer})
    captured = datetime.fromtimestamp(scope['captured_unix'], timezone(timedelta(hours=8))).isoformat()
    m, s = result['full_unpaired']['R11'], result['full_unpaired']['historical_Steer']
    lines = ['# 当前生成与对照比较', '',
             f'固定快照：{captured}。R11 已完成注册的 {spec["n_per_arm"]} 次生成；'
             f'R26 与无引导各完成 {scope["snapshot_n_per_arm"]} 个可成对评估的样本，'
             f'共同批次 {scope["included_batches"]}。捕获时完整对照任务仍在运行。'
             '派生快照的 COMPLETE 只证明该快照完成，不代表 1,000 样本任务完成。', '',
             '主要结果：Gradient guidance 相对无引导的差异应以共同批次的配对比较判断；'
             'Steer 为历史非配对参照。当前记录不更换奖励函数、默认配置或正在运行的推理。', '',
             '## 相同初始随机状态的部分对照', '',
             '每组使用同一批次/slot，初始坐标、原子及键状态签名一致。master seed=42，'
             '批次种子为 42+100003×batch；推理共 100 步。']
    if args.status_inspection:
        latest = result['latest_status_inspection']
        latest_time = datetime.fromtimestamp(latest['inspected_unix'], timezone(timedelta(hours=8))).isoformat()
        state = latest['cohorts']['R26_native']['arms']
        lines[4:4] = [f'后续只读检查：{latest_time}，后台状态 {latest["status"]["status"]}；'
                      f'无引导已结束 {state["unguided"]["completed_attempts"]} 个样本，'
                      f'R26 已结束 {state["gradient"]["completed_attempts"]} 个。'
                      '这些新增批次没有混入下面已冻结并评估的统计。', '']
    lines += ['']+table(result['paired_panel'])
    lines += ['', f'极高分阈值为既定值 {threshold:.9f}，没有根据本次结果重新选择。', '']
    for label, effect in result['paired_effects'].items():
        lo, hi = effect['batch_bootstrap_CI95']
        lines += [f'{label}：全部尝试配对均值差 {effect["paired_mean_pic50"]:+.6f}，'
                  f'批次 bootstrap 95% 区间 [{lo:+.6f}, {hi:+.6f}]；'
                  f'{effect["positive_batch_n"]}/{effect["n_batches"]} 批均值为正。'
                  f'逐样本中位差 {effect["paired_median_pic50"]:+.6f}，'
                  f'至少 +0.01 的样本 {effect["slots_gain_at_least_0_01"]} 个，'
                  f'至多 −0.01 的样本 {effect["slots_loss_at_least_0_01"]} 个。', '']
    versus_native = result['paired_effects']['R11_vs_unguided']
    versus_r26 = result['paired_effects']['R11_vs_R26']
    interval_r26 = versus_r26['batch_bootstrap_CI95']
    r26_conclusion = ('当前部分对照没有证据证明 R11 优于 R26。' if interval_r26[0] <= 0 <= interval_r26[1]
                      else 'R11 与 R26 的差异需待完整注册批次确认。')
    lines += [f'R11 相对无引导的配对均值提升为 {versus_native["paired_mean_pic50"]:+.6f}；'
              f'{r26_conclusion}差异体现在少数生成路径上：R11 与 R26 的 '
              f'{len(frames["R11"])-versus_r26["slots_gain_at_least_0_01"]-versus_r26["slots_loss_at_least_0_01"]}'
              f'/{len(frames["R11"])} 个 slot 分数差绝对值小于 0.01。'
              '先前六个独立批次的 +0.021049 优势不应直接外推到这些新批次；'
              '需要把当前的批次依赖纳入最终结论。', '',
              f'{len(scope["included_batches"])} 个批次仍是部分结果。区间按批次重采样 2,000 次，随机种子 42；'
              f'没有将同一批次的 {spec["batch"]} 个分子视为 {spec["batch"]} 个独立重复。'
              f'完整 {len(spec["batch_indices"])} 批对照结束后才能给出完整判定。', '',
              '## 相同最终尝试数的历史参照', '']+table(result['full_unpaired'])
    lines += ['', f'R11 相对 Steer 的全部尝试均值差为 {m["all_mean_pic50"]-s["all_mean_pic50"]:+.6f}；'
              f'最高有效分数差为 {m["valid_max_pic50"]-s["valid_max_pic50"]:+.6f}。'
              f'极高分产率分别为 {m["elite_yield"]:.2%} 和 {s["elite_yield"]:.2%}，'
              f'达到阈值的独立化学图分别为 {m["elite_unique_graphs"]} 和 {s["elite_unique_graphs"]}。', '',
              f'R11 应变中位数比 Steer 低 '
              f'{1-m["strain_median_per_heavy"]/s["strain_median_per_heavy"]:.1%}，P90 低 '
              f'{1-m["strain_p90_per_heavy"]/s["strain_p90_per_heavy"]:.1%}。'
              '这反映本地同图松弛的能量变化，不直接证明结合自由能更好。', '',
              f'有效独立化学图数：R11={m["unique_valid_graphs"]}，Steer={s["unique_valid_graphs"]}；'
              '不能将某些坐标方向上的自由度增加等同于最终图多样性必然增加。', '',
              f'当前 R11 的前 {scope["snapshot_n_per_arm"]} 个样本已找到最高有效分数 '
              f'{result["paired_panel"]["R11"]["valid_max_pic50"]:.6f}，完整 {m["n"]} 个为 '
              f'{m["valid_max_pic50"]:.6f}；极高分有效计数由 '
              f'{result["paired_panel"]["R11"]["elite_valid_n"]} 增至 {m["elite_valid_n"]}。'
              '同一固定样本流内的累计结果表明，增加生成数带来更多命中，但本次没有缩小最高分与 Steer 的差距。'
              '先前 300 样本确认使用不同批次，因此不能将它与本次 1,000 样本的最高分差异完全归因于数量。', '',
              '## 如何解释当前结果', '',
              'R11 的质量结果须分别考察整体分布和极高亲和力尾部。扩大生成数量能够增加发现稀有路径的机会，'
              '但如果单位尝试的极高分产率没有提升，仅增加数量不会复现在线筛选的选择压力。'
              'Steer 根据在线分数复制较优父代，再继续生成；当前 guidance 依据历史空间奖励修正单条路径，'
              '没有在线亲和力筛选。因此，保留和扩增极高分路径仍可能是 Steer 的主要优势。'
              '这个机制解释与结果相符，但本比较没有进行新的因果消融，不能据此排除其他因素。', '',
              '历史 Steer 使用批次 0–19；当前使用 55–74，不能逐 slot 配对。'
              '历史 Steer 的 700 个设计供体参与了奖励构建，且在线评分/重采样预算更多。'
              '所以 N=1,000 的比较是同尝试数的描述性参照，而不是独立训练外、等计算量的胜负检验。', '',
              '## 执行与评估边界', '',
              'R11/R26 在输入声明的 0–0.5 学习窗口执行 50 次有效坐标更新，0.5–1.0 原生续推；'
              '无引导没有坐标注入。100 步均无粒子重采样，使用真实 FLOWR 端点 Jacobian 的反传，'
              '没有对 affinity head 求导，也没有额外的每步生产目标前向。'
              'preflight 额外前向属于一次数值验证，原生 untarget 诊断前向仍存在。', '',
              f'路径累计坐标注入 RMS（各步 RMS 求和，非终态位移）：R11 '
              f'{result["guidance_feedback"]["R11"]["mean_cumulative_rms_A"]:.6f} Å，R26 '
              f'{result["guidance_feedback"]["R26"]["mean_cumulative_rms_A"]:.6f} Å，无引导为 0。'
              '因此 R11/R26 结果接近不意味着 guidance 相对无引导没有执行。', '',
              '目标为 CK2α 的 FLOWR 预测 pIC50，越高越好，尚未经过实验亲和力验证。'
              '均值保留所有尝试，包括结构无效的 slot；P95/最高分和极高分计数只使用有效结构。'
              'PB fast 是 PoseBusters dock_fast 的结构检查子集，不是完整的蛋白复合物验证。'
              '应变为 MMFF94s：先固定重原子松弛氢，再进行同一化学图的本地松弛，'
              '取能量下降/重原子数（kcal/mol）；仅汇总双重收敛者，并保留缺失/失败计数。', '',
              f'R11 收敛能量记录 {m["strain_converged_n"]}/{m["valid_n"]}，'
              f'Steer {s["strain_converged_n"]}/{s["valid_n"]}。'
              '新旧结构采用同一已固定评估流程；源数据校验和、程序、执行记录与统计定义见 comparison.json 和各组保留目录。', '',
              f'推理代码固定为 `{result["integrity"]["inference_commit"]}`；'
              '远端后台 worker 使用独立固定源码快照，本次读取和分析没有改变它。', '',
              '![配对结果与完整尾部](comparison.png)', '',
              '复用入口：`scripts/report_guidance_scale.py --help`。新报告需要提供本次两个数据集、'
              '已评估结果、历史 Steer 源表/固定元数据、既定 protocol/spec 和全新的输出路径。']
    (output/'report.zh-CN.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    write_json(output/'manifest.json', {'files': [
        {'path': p.relative_to(output).as_posix(), 'sha256': digest(p), 'bytes': p.stat().st_size}
        for p in sorted(output.rglob('*')) if p.is_file()], 'status': 'complete'})
    print({key: value['all_mean_pic50'] for key, value in result['paired_panel'].items()})


if __name__ == '__main__':
    main()
