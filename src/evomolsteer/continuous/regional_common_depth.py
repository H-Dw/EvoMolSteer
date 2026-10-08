"""One shared ancestry lag for a frozen whole-window regional hypothesis.

The frozen regional formula, full XYZ fits and branch facts are reused. Only
the discovery-statistic choice of a single common depth is versioned here.
"""
from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .regional_reference import (
    METRIC, REGION_PATTERN, compact_json_bytes, learn_rules,
    prepare_inputs, synthesize_reference,
)
from ..io import digest, read_json

DEFAULT_DEPTHS = (2, 3, 5, 8, 13)
SELECTION_RULE = (
    "One common depth for every region: descending eligible region count, "
    "descending mean LOO positive fraction among eligible regions, then shallower depth"
)


def declared_depths(values):
    """Keep a literal positive, distinct declaration; do not infer a new lag."""
    values = tuple(values)
    if not values or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in values):
        raise ValueError("Declared positive integer ancestry depths required")
    if len(values) != len(set(values)):
        raise ValueError("Declared ancestry depths must be distinct")
    return values


def learn_common_depth_rules(effects, batches, fitted, catalog, window, *,
                            depths=DEFAULT_DEPTHS, q_threshold=.05,
                            minimum_batches=14, minimum_loo_support=.70,
                            absolute_floor=1e-12):
    """Select by discovered regional coverage, never generated quality labels.

    Every individual depth uses the exact frozen eligibility function. Existing
    shared-family p/q values and complete XYZ vectors are not recomputed or
    adjusted after depth selection. An empty discovery produces an empty field.
    """
    depths = declared_depths(depths)
    evaluations, summaries = [], []
    for depth in depths:
        subset = effects[effects.depth == depth]
        depth_batches = batches[batches.depth == depth]
        # With a single depth, frozen per-region depth selection is an identity.
        rules = learn_rules(subset, depth_batches, fitted, catalog, window,
                            q_threshold=q_threshold, minimum_batches=minimum_batches,
                            minimum_loo_support=minimum_loo_support,
                            absolute_floor=absolute_floor)
        regions = rules['selected_regions']
        if any(region['depth'] != depth for region in regions):
            raise ValueError("Frozen regional discovery returned a different depth")
        observed = sorted({int(match[1]) for feature in subset.feature
                           for match in [REGION_PATTERN.match(feature)] if match})
        chosen_ids = [region['region'] for region in regions]
        mean_support = float(np.mean([region['loo_positive_fraction'] for region in regions])) if regions else None
        summary = {
            'depth': depth, 'observed_regions': len(observed),
            'eligible_region_count': len(regions), 'eligible_region_ids': chosen_ids,
            'ineligible_region_count': len(set(observed) - set(chosen_ids)),
            'ineligible_region_ids': sorted(set(observed) - set(chosen_ids)),
            'mean_loo_positive_fraction': mean_support,
            'eligible_region_support': [
                {'region': region['region'], 'loo_positive_fraction': region['loo_positive_fraction'],
                 'loo_mean_cosine': region['loo_mean_cosine'],
                 'loo_observed_batches': region['loo_observed_batches'],
                 'minimum_component_q': min(region['component_q']),
                 'observed_time_start': region['observed_time_start'],
                 'observed_time_end': region['observed_time_end']}
                for region in regions],
            'explicit_eligibility_rejections': rules['rejected_candidates'],
            'status': 'no_observations' if not len(subset) else 'no_eligible_regions' if not regions else 'eligible_not_selected',
            'ineligible_semantics': 'No qualifying absolute XYZ covariance/CI/q or independent-batch LOO support under frozen discovery rules',
        }
        evaluations.append(rules)
        summaries.append(summary)
    eligible_indices = [i for i, value in enumerate(summaries) if value['eligible_region_count']]
    if eligible_indices:
        selected = min(eligible_indices, key=lambda i: (
            -summaries[i]['eligible_region_count'],
            -summaries[i]['mean_loo_positive_fraction'], summaries[i]['depth']))
        rules = copy.deepcopy(evaluations[selected])
        chosen_depth = summaries[selected]['depth']
        summaries[selected]['status'] = 'selected'
    else:
        rules = copy.deepcopy(evaluations[0])
        rules['selected_regions'] = []
        rules['eligible_region_depths'] = 0
        chosen_depth = None
    rules.update({
        'schema_version': 'whole-window-regional-common-depth-rules-1.0',
        'common_depth': chosen_depth, 'declared_depths': list(depths),
        'common_depth_selection_rule': SELECTION_RULE,
        'depth_selection': SELECTION_RULE, 'depth_evaluations': summaries,
        'new_significance_tests': False, 'selection_uses_final_generated_labels': False,
        'covariance_scale_policy': 'Every selected region has one identical ancestry lag; no lag division, variance rescaling or depth mixing',
        'spatial_transfer_assumption': 'Frozen ancestor-membership associations are transferred to current intact teacher density membership as an intervention hypothesis',
        'pose_policy': 'Subtract joint field mean (translation); rotational components are not projected out',
        'confidence_interpretation': 'Original lag-two local branch eligibility is retained; not calibrated confidence of the synthesized regional direction',
    })
    rules['limitations'] = [*rules['limitations'],
        'A common lag removes unequal-lag covariance mixing, not feature-density heterogeneity, nuisance bias or observational survivor conditioning.',
        'Global depth/region selection reuses discovery batches; shared frozen q-values are not post-selection confirmation.',
        'Unit-RMS normalization preserves combined direction but discards absolute effect/member density; natural branch RMS is a separate experimental scale.',
        'No rigid-rotation projection is applied; mean-zero COM does not imply absence of torque or pose changes.',
        'Missing initial common-depth observations remain zero; no synthetic smooth onset or temporal extrapolation is introduced.',
    ]
    return rules


def _report(rules, coverage):
    rows = [
        '# 共同谱系深度的区域联合参考', '',
        '此工具仅转换冻结的发现统计与联合教师参考，没有模型推理、亲和力模型拟合或终态标签访问。旧区域版本保持冻结。', '',
        f"声明深度：{rules['declared_depths']}；选中共同深度：{rules['common_depth']}。所有入选区域都使用该同一深度，不按幅度偏好混合不同时间跨度。", '',
        '固定选择规则：先比较符合原有 XYZ q/CI、非零绝对协方差及独立批次 LOO 支持标准的区域数；数量并列时比较入选区域的平均 LOO 正方向比例；仍并列时选择较浅深度。没有重算 p/q，也没有使用新生成结果选择深度。', '',
        '| 深度 | 观察区域 | 合格区域 | 平均 LOO 正方向比例 | 状态 |',
        '|---:|---:|---:|---:|---|',
    ]
    for item in rules['depth_evaluations']:
        support = '缺失' if item['mean_loo_positive_fraction'] is None else f"{item['mean_loo_positive_fraction']:.6f}"
        rows.append(f"| {item['depth']} | {item['observed_regions']} | {item['eligible_region_count']} | {support} | {item['status']} |")
    rows += ['', f"入选区域：{[value['region'] for value in rules['selected_regions']]}。完整 XYZ 函数、分量 q/CI、观察区间及所有未选深度计数保存在 regional_rules.json。", '',
             f"原合格教师 {coverage['original_eligible_teachers']}/{coverage['teachers']}；新有效场 {coverage['new_active_teachers']}/{coverage['teachers']}。共同深度之前的缺失节点没有补值，实际时间支持由每项冻结函数的交集决定。", '',
             '坐标、评分、原先验、批次、原谱系资格、自然变异 RMS 和原子权重保持原值。旧方向与 confidence 另存；只有新区域场为零时将其用于控制的 confidence 置零，避免制造资格。', '',
             '场为完整点云上全部入选区域的协方差加权软密度和，再减去整体平均并归一到 atom-RMS=1。整体平移在数值精度内消除；没有旋转投影，仍可能产生力矩或受体相对取向变化。RMS 位移上限不是逐原子位移上限。', '',
             '相同深度降低了跨时间跨度的幅度混合问题，但不把协方差变成可验证的亲和力空间梯度，也不解决区域重叠、绝对效应归一化、原方向 confidence 转用、存活条件偏差或深度/区域发现过拟合。必须继续进行正向、反向、空规则与配对剂量/终态验证。', '',
             '实际执行 receipt 在脚本运行前后记录 argv 和输入/源码/输出哈希；本文由本次工具生成并纳入输出哈希。']
    return '\n'.join(rows) + '\n'


def synthesize_common_depth(mining, reference, output, output_reference, *,
                            depths=DEFAULT_DEPTHS, q_threshold=.05,
                            minimum_batches=14, minimum_loo_support=.70,
                            absolute_floor=1e-12):
    mining, reference = Path(mining).resolve(), Path(reference).resolve()
    output, output_reference = Path(output).resolve(), Path(output_reference).resolve()
    if output.exists() or output_reference.exists():
        raise FileExistsError('Fresh common-depth summary/reference paths required')
    manifest = read_json(mining / 'manifest.json')
    original = json.loads(gzip.decompress(reference.read_bytes()))
    rules = learn_common_depth_rules(
        pd.read_parquet(mining / 'whole_window_effects.parquet'),
        pd.read_parquet(mining / 'batch_window_statistics.parquet'),
        read_json(mining / 'fitted_trends.json'), read_json(mining / 'feature_catalog.json'),
        manifest['window'], depths=depths, q_threshold=q_threshold,
        minimum_batches=minimum_batches, minimum_loo_support=minimum_loo_support,
        absolute_floor=absolute_floor)
    rules.update(source_manifest_sha256=digest(mining / 'manifest.json'),
                 source_reference_sha256=digest(reference),
                 source_evidence_sha256=digest(mining / 'evidence.json'))
    transformed, coverage = synthesize_reference(original, rules, source_reference_sha256=digest(reference))
    rules['teacher_coverage'] = coverage
    transformed['regional_reference'].update(
        common_depth=rules['common_depth'], declared_depths=rules['declared_depths'],
        common_depth_selection_rule=SELECTION_RULE,
        covariance_scale_policy=rules['covariance_scale_policy'],
        pose_policy=rules['pose_policy'], confidence_interpretation=rules['confidence_interpretation'])
    rule_bytes = compact_json_bytes(rules)
    if len(rule_bytes) >= 1024 * 1024:
        raise ValueError('Common-depth summary exceeds compact 1MiB budget')
    output.mkdir(parents=True)
    rule_path = output / 'regional_rules.json'
    rule_path.write_bytes(rule_bytes)
    transformed['regional_reference']['rules_sha256'] = digest(rule_path)
    output_reference.parent.mkdir(parents=True, exist_ok=True)
    output_reference.write_bytes(gzip.compress(compact_json_bytes(transformed), mtime=0))
    report_path = output / 'report.zh-CN.md'
    report_path.write_text(_report(rules, coverage), encoding='utf-8')
    result = {
        'schema_version': 'regional-common-depth-manifest-1.0', 'window': original['window'],
        'common_depth': rules['common_depth'], 'declared_depths': rules['declared_depths'],
        'depth_selection_rule': SELECTION_RULE, 'selected_regions': len(rules['selected_regions']),
        'selected_region_ids': [region['region'] for region in rules['selected_regions']],
        'source_reference_sha256': digest(reference), 'source_manifest_sha256': digest(mining / 'manifest.json'),
        'coverage': {key: value for key, value in coverage.items() if key != 'frames'},
        'reference_path': str(output_reference), 'reference_sha256': digest(output_reference),
        'rules_sha256': digest(rule_path), 'rules_bytes': len(rule_bytes),
        'report_sha256': digest(report_path), 'new_model_or_affinity_head': False,
        'new_significance_tests': False, 'generation_performed': False,
        'immutable_original_fields': 'All original fields except new direction/zero-field confidence; originals retained explicitly',
    }
    (output / 'manifest.json').write_bytes(compact_json_bytes(result))
    return result
