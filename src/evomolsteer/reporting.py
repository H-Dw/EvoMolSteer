"""Compact audit of the active selection-only analysis."""
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
from .io import read_json,write_json,digest


def summarize(output):
    out = Path(output)
    cfg = read_json(out/'config.json');scope = read_json(out/'selection_scope.json')
    manifest = read_json(out/'ingest_manifest.json');bundle = read_json(out/'agents/evidence_bundle.json')
    event = pd.read_parquet(out/'selection_events.parquet')
    nrows = (pq.ParquetFile(out/'features.parquet').metadata.num_rows if (out/'features.parquet').exists()
             else sum(a['B']*len(a['analyzed_steps']) for a in manifest['audits'])*len(cfg['analysis_representations']))
    summary = {'analysis_scope':scope['analysis_scope'],'feature_rows':nrows,'audited_arm_batches':len(manifest['audits']),
        'selection_events':int(event.resampled.sum()),'background_events':int((~event.resampled).sum()),
        'events_per_batch':len(scope['steps']),'window':[scope['window_start'],scope['window_end']],
        'stages':scope['stages'],'representations':cfg['analysis_representations'],
        'evidence_items':len(bundle['evidence']),'selection_prototypes':len(bundle['targets']),
        'manifest_sha256':digest(out/'ingest_manifest.json'),'final_outcomes_consumed':False}
    dest = out/'reports'
    write_json(dest/'run_summary.json',summary)
    lines = ['# EvoMolSteer 选择窗口分析 v2','',
        f'已分析 {len(manifest["audits"])} 个策略批次，窗口为评分时间 {scope["window_start"]}–{scope["window_end"]}，每批 {len(scope["steps"])} 个匹配事件。',
        f'特征表共 {nrows:,} 行，表示为预测终点与实际待复制的 proposal。',
        '终态、后代结局、窗口后轨迹和反事实续跑不进入当前统计或 LLM 证据。','',
        '| 阶段 | 评分范围 | 每批事件数 | 实际均值时间 |','|---|---|---:|---:|']
    for s in scope['stages']:
        close = ']' if s['right_inclusive'] else ')'
        lines.append(f'| {s["stage"]} | [{s["stage_start"]}, {s["stage_end"]}{close} | {s["n_events"]} | {s["mean_score_time"]:.6f} |')
    lines += ['', '## 方法','',
        '- 富集：逐评分事件的无重采样背景第 75 百分位；期望高特征质量、实际复制质量和选中/淘汰 log-OR。',
        '- 差异：同事件候选比较及同父同胞对照，不使用后代成功标签。',
        '- 趋势：期望选择、实际复制、随机偏差及 on/off 分量；保存事件和阶段导数、窗口内真实父子边变化率。',
        '- PCA：只在 discovery 无重采样组的匹配窗口拟合，保存选择质心变化与父子边 PC 变化率。',
        '- 推断：事件先配对、批次为独立单位；每个表示/策略/比较内跨特征和阶段做 BH 校正。',
        '',f'证据包含 {len(bundle["evidence"])} 项证据与 {len(bundle["targets"])} 个概率加权经验区间。',
        '经验区间描述选择偏好，不代表最终活性、结构有效性或最优几何值。',
        '', '## 入口','', '- `discovery/`：四个独立分析模块。',
        '- `selection_scope.json`：真实选择步、边界与阶段。',
        '- `agents/evidence_bundle.json` 与 `Analyst.request.json`：后续规则提出的输入。',
        '- `run_provenance.json`：源码、配置与结果哈希。']
    (dest/'RUN_SUMMARY.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return summary
