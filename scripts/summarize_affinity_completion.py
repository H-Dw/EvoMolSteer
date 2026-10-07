"""Failure-inclusive final readout from retained statistics, without inference."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from evomolsteer.io import read_json, write_json, digest
from evomolsteer.generation.affinity_reporting import arm_summary


def summarize(evidence, historical, output):
    evidence, historical, output = map(Path, (evidence, historical, output))
    report = read_json(evidence/'final_report/campaign_report.json')
    assert report['new_rounds_completed'] == report['maximum_rounds'] == 30
    assert read_json(evidence/'completion_audit.json')['passed'] is True
    original = pd.read_csv(historical)
    steer = original[original.arm.eq('single')]
    assert len(steer) == 100
    old = arm_summary(steer)
    assert np.isclose(old['all_head_mean'], report['historical_Steer100']['all_head_mean'])
    assert np.isclose(old['best_valid_head'], report['historical_Steer100']['best_valid_head'])
    frozen = report['frozen']['all_frozen']
    g, n = frozen['gradient'], frozen['native']
    diagnostics = pd.read_parquet(evidence/'response_diagnostics_full30/batch_diagnostics.parquet')
    checks = diagnostics[diagnostics['round'].ge(27)]
    assert len(checks) == 8 and checks.final_decoded_head_delta.gt(0).all()
    assert np.isclose(checks.final_decoded_head_delta.mean(), frozen['batch_contrast']['mean_head_delta'])
    value = {'schema_version': 'affinity-completion-readout-1.0', 'new_rounds': 30,
        'registered_discovery_rounds': 26, 'frozen_rounds': [27,28,29,30],
        'frozen_all': frozen, 'heldout_only': report['frozen']['heldout'],
        'historical_Steer100': old,
        'historical_scope': 'Unpaired historical100 subset, unequal inference/selection budget; not original global maximum',
        'response': {'positive_batches': 8, 'batches': 8,
            'mean_absolute_paired_head_change': float(checks.final_absolute_head_delta.mean()),
            'positive_slot_fraction': float(checks.final_positive_fraction.mean())},
        'remaining_gaps': {'frozen_best_head_above_native': g['best_valid_head'] > n['best_valid_head'],
            'frozen_best_head_above_historical_Steer': g['best_valid_head'] > old['best_valid_head'],
            'energy_p90_at_most_historical_Steer': g['MMFF_relief_per_heavy_p90'] <= old['MMFF_relief_per_heavy_p90'],
            'energy_median_at_most_historical_Steer': g['MMFF_relief_per_heavy_median'] <= old['MMFF_relief_per_heavy_median']},
        'sources': {str(p): digest(p) for p in [historical, evidence/'final_report/campaign_report.json',
            evidence/'completion_audit.json', evidence/'response_diagnostics_full30/batch_diagnostics.parquet']},
        'source_code_sha256': digest(__file__)}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'completion_readout.json', value)
    lo, hi = frozen['batch_contrast']['batch_bootstrap_CI95']
    lines = ['# 新30轮坐标引导：完成结果与剩余差距', '',
        f"已完成26轮注册搜索及4轮冻结验证。400个新批次配对样本的预测CK2亲和力均值Δ={frozen['batch_contrast']['mean_head_delta']:+.6f} pIC50；8个批次bootstrap95%区间[{lo:.6f},{hi:.6f}]，全部8批为正。",
        f"仅留出第28–30轮：300个配对样本，均值Δ={report['frozen']['heldout']['batch_contrast']['mean_head_delta']:+.6f}。主种子42；这些是同一口袋的新随机批次，不是新靶点或实测结合。", '',
        '|指标|冻结引导400|配对无引导400|历史Steer100|',
        '|---|---:|---:|---:|']
    for label, key in [('所有尝试预测亲和力均值','all_head_mean'), ('有效分子预测均值','valid_head_mean'),
            ('有效分子最高预测亲和力','best_valid_head'), ('去重Top5均值','unique_Top5_mean'),
            ('结构有效率','valid_rate'), ('MMFF应变中位数/重原子','MMFF_relief_per_heavy_median'),
            ('MMFF应变p90/重原子','MMFF_relief_per_heavy_p90'), ('周围区域松弛RMS/A','surround_relax_RMS_A_mean')]:
        lines.append(f"|{label}|{g[key]:.6f}|{n[key]:.6f}|{old[key]:.6f}|")
    lines += ['', '历史Steer与400个新样本不配对且预算不同。最高值与均值分开比较：平均亲和力有收益，最高值没有超过native或历史Steer。应变及周围松弛优于native，但仍明显高于历史Steer，尤其应变p90；不能声称已经实现Steer的能量稳定性。MMFF指标是孤立配体同化学图局部松弛能差，不是结合自由能。', '',
        '初始问题不能统一解释为剂量不足：归一化梯度控制会抵消整体奖励正数缩放；实际剂量、限幅、每步坐标更新才决定强度。已观察到个体大幅改变与均值抵消；在已生效条件下，负收益需要修订坐标目标或其输入。发现集较强剂量也未单调提高均值。', '',
        '最终奖励是逐实际节点的完整三维endpoint点云混合。匹配、teacher选择及条件先验固定，不按原子类型或化学图匹配。对最近4个teacher，以Hungarian均方距离C及已保存评分s构造log(pi)=logsoftmax(-C/4+2*(s-mean(s)))。令q为可微endpoint与匹配teacher的均方距离，rho=sqrt(1+q)-1，R=0.5*logsumexp(log(pi)-rho/0.5)。实际pointcloud曲率默认为1 A；遗留robust_delta=2不是该公式的参数。', '',
        '梯度为真实FLOWR endpoint对当前坐标的VJP。它复用原生forward，亲和力输出detach；对应50个学习节点均施加剂量比0.33，窗口结束后不再注入，并完成100步原生推理。窗口从输入参考及程序读取，不是硬编码；本次为0–0.5。累计路径长度不是最终净位移，日志的一阶奖励增益不是重新推理得到的奖励变化。', '',
        '完整点云目标、区域加权、稀疏坐标、谱系未来信用、teacher温度/邻居与物理剂量均已比较；表现下降时保留全局父方案继续探索。当前证据支持共同三维构象关系的价值，未证明单一优势区域因果性。终端信用400项检验无q<0.05，不能强行宣称显著motif。', '',
        '实际Skill输入、prompt及literal响应已冻结，行为验证含9类错误反例；独立复核截至第28轮，30轮运行合同和保留字节另经completion_audit验证。多数逐轮决定由注册搜索执行，不能当成30次独立LLM设计或LLM因果收益证明。', '',
        '第30轮本地评估曾MemoryError；以相同原始样本降低并发后成功，未重新推理、丢弃失败或更改统计定义。30轮生成结构与传输归档均在校验报告后退休；保留原始Steer、checkpoint、评分/失败表、实际配置、源commit与可复核设计摘要。', '',
        '后续研究应在保持已证实的平均亲和力收益前提下，检验窗口内几何不确定性与兼容性方向的剂量分配、软几何关系及终态能量信用。它们是待检验假设，不是本次已实施的函数；本次30轮预算已结束。']
    (output/'completion_readout.zh-CN.md').write_bytes(('\n'.join(lines)+'\n').encode('utf8'))
    return value


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--evidence', required=True)
    p.add_argument('--historical', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    print({'new_rounds': summarize(a.evidence, a.historical, a.output)['new_rounds']})
