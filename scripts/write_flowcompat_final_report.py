"""Write the final public scientific rationale from completed retained evidence.

Read-only with respect to frozen inputs. This is a report, not an Agent input,
new selection criterion, reward fit, or private reasoning transcript.
"""
import argparse
from pathlib import Path

from evomolsteer.io import digest, read_json, write_json


def write_report(input_dataset, configuration_dataset, output_dataset):
    source = Path(input_dataset).resolve()
    configuration = Path(configuration_dataset).resolve()
    output = Path(output_dataset).resolve()
    if output == source or output.is_relative_to(source):
        raise ValueError('Derived output must be separate from frozen evidence')
    state = read_json(configuration / 'campaign.json')
    if state['rounds_completed'] != 30 or [r['round'] for r in state['rounds']] != list(range(1, 31)):
        raise ValueError('Thirty completed actual rounds are required')
    confirmation = read_json(source / 'confirmation.json')
    integrity = read_json(source / 'confirmation_integrity.json')
    if not confirmation['complete']:
        raise ValueError('Independent confirmation must be evaluated first')
    if confirmation['adopt_candidate'] and not (integrity['integrity_passed'] and integrity['eligible_for_efficacy_decision']):
        raise ValueError('Adoption and independent integrity must agree')
    frozen = read_json(source / 'frozen_validation.json')
    selected = frozen['selected_round']
    active = read_json(configuration / 'active_workflow.json')
    output.mkdir(parents=True, exist_ok=True)
    inputs = {str(p): digest(p) for p in [configuration / 'campaign.json', source / 'confirmation.json',
              source / 'confirmation_integrity.json', source / 'frozen_validation.json', source / 'protocol.json']}
    for p in [configuration / f'round{selected:02d}.json', configuration / 'active_workflow.json',
              configuration / 'active_program.json', configuration / 'active_skills.json', source / 'checkpoint_metadata.json',
              source / 'runtime_architecture_addendum.zh-CN.md', source / 'scientific_design.zh-CN.md']:
        inputs[str(p)] = digest(p)
    if (source / 'activation.json').is_file():
        inputs[str(source / 'activation.json')] = digest(source / 'activation.json')
    for family in ['joint_contrast_v2', 'branch_mutation_v2', 'multi_depth_v1', 'regional_reference_v2']:
        for p in (source / 'mining' / family).rglob('*'):
            if p.is_file() and p.name != 'augmented_reference.json.gz':
                inputs[str(p)] = digest(p)
    attempts = sum(sum(m['n'] for m in r['results']['results'].values()) for r in state['rounds'])
    adopted = confirmation['adopt_candidate']
    lines = ['# 历史 Steer 路径挖掘与 FLOWR 梯度控制：30 轮最终报告', '',
        '本文件在全部实验结束后生成，汇总公开的证据、假设、公式和可复现参数；不作为已冻结 Agent 的输入，也不记录私有思维链。', '',
        f'完成 30/30 轮真实 FLOWR 推理，共 {attempts} 个生成尝试。轮数包含数值 null、反方向和新批次验证，不是 30 个独立 LLM 设计；重复控制也不代表不同分子。', '',
        '## 最终确认', '',
        f'冻结候选来自第 {selected} 轮，两批探索均值改善 {state["rounds"][selected-1]["mean_vs_R26"]:+.6f} pIC50；随后用六个新批次验证，验证期间未调奖励或强度。', '',
        '|组别|全部尝试平均预测 pIC50|有效分子平均预测 pIC50|有效/PB-fast|应变中位/P90，kcal/mol/重原子|应变覆盖|极高分产率|',
        '|---|---:|---:|---|---|---|---:|']
    for key, label in [('candidate', f'冻结 R{selected} 候选'), ('R26', '历史最优 R26'), ('native', '无引导')]:
        m = confirmation['metrics'][key]
        lines.append(f'|{label}，N={m["n"]}|{m["all_mean_pic50"]:.6f}|{m["valid_mean_pic50"]:.6f}|{m["valid_n"]/m["n"]:.1%}/{m["pb_fast_rate"]:.1%}|{m["strain_median_per_heavy"]:.4f}/{m["strain_p90_per_heavy"]:.4f}|{m["strain_converged_n"]}/{m["n"]}|{m["elite_yield"]:.1%}|')
    for key, label in [('versus_R26', 'R26'), ('versus_native', '无引导')]:
        effect = confirmation[key]
        low, high = effect['batch_bootstrap_CI95']
        lines += ['', f'候选相对{label}：配对均值差 {effect["paired_mean_pic50"]:+.6f}，六批 bootstrap 95% 区间 [{low:+.6f}, {high:+.6f}]。']
    lines += ['', f'独立完整性审核通过：{integrity["integrity_passed"]}；可用于效果决策：{integrity["eligible_for_efficacy_decision"]}。',
        f'预登记升级条件全部满足：{adopted}。'+('候选具备升级证据；实际默认切换需以代码中的活动配置为准。' if adopted else '候选未获确认，生产默认保留 R26；失败试验与代码记录仍保留。'), '',
        f'实际活动配置：{active["default"]}；candidate_enabled={active["candidate_enabled"]}；奖励 SHA={digest(configuration / "active_program.json")}。', '',
        '检验采用批次为重采样单位，不能把 300 个后代当作 300 次独立实验。应变只统计有效且 MMFF 优化收敛的分子，是生成构象到同图松弛构象的能量差代理，表中单列覆盖；不是全部尝试均有物理能量。指标来自同一预测亲和力模型及构象检查，尚无实验结合常数验证。', '',
        '## 与历史 Steer 的关系', '']
    steer = confirmation['historical_Steer_unpaired']
    lines += [f'原 Steer 共 {steer["n"]} 个尝试，平均预测 pIC50 {steer["all_mean_pic50"]:.6f}，最高有效分数 {steer["valid_max_pic50"]:.6f}，极高分产率 {steer["elite_yield"]:.1%}；应变中位/P90 {steer["strain_median_per_heavy"]:.4f}/{steer["strain_p90_per_heavy"]:.4f}。', '',
        '历史 Steer 的发现供体参与奖励构建，样本量、在线评分和分支探索预算也不同；这是非配对参照。本轮无引导和梯度组也保留原生 SDE 与离散随机采样。Steer 特有的是复制高分路径后获得多个随机后代并在线重新分配群体预算，历史结果具有更高的极高分尾部产率；不能用不等预算确定该优势的因果幅度。当前局部几何奖励尚未证明采样分布等价，也未证明更自由区域与邻域适配同步改善。', '',
        '## 从筛选机制到数据特征', '',
        'Steer 通过评分提高候选的期望复制数，再复制生成状态和条件并继续随机生成。一次实际存活包含抽样运气；立即复制的同胞没有几何变异，被淘汰分支的未来也没有观察。分析因此回溯真正共同祖先、折叠复制、比较不同即时父分支，并以祖先和批次处理相关性。', '',
        '区分当前状态 X、模型端点预测 Y、原生积分提议 Z、评分和复制索引。在线评分对应 Y；复制的是 Z 及条件。整窗使用输入声明的选择节点，不再人为分成固定时间箱。当前评分 0 至 .49，最后提议到 .5；终态仅用于效果评价。', '',
        '|新增工具|关键结果|对奖励设计的含义|',
        '|---|---|---|',
        '|selection innovation|内部形变约占坐标对比 78.65%；已观察子代再预测分数平均下降 .03633|存活不意味着绝对分数逐步上升；分别检验探索与稳定性|',
        '|共同祖先 branch mutation|调整后联合/内部再预测 RMS 与评分变化相关 −.14993/−.13733；q≈.000282/.000970；绝对协方差仅约 −3.60e−5/−3.42e−5 Å·pIC50|显著相关不能直接变成强坐标力；自然变异幅度与方向分开定标|',
        '|multi-depth mutation|深度 2/3/5/8/13，137 特征；部分区域方向关联存在，但逐节点跨批方向弱|整窗趋势拟合与留一批验证；保留缺失早期支持和反证|',
        '|common-depth regional synthesis|只用发现数据选共同深度 3、7 个区域，285/1400 教师具备方向支持|不混合观察跨度，完整联合形变保留空间相容关系|', '',
        '后续区域 XYZ 协方差函数采用整窗多项式候选、留一批验证和 1-SE 简化；本次合格区域方向选择常数函数，导数为零。没有观测支持人为制造区域多阶段方向切换。这不表示 R11 每节点的实际分支方向恒定。动态奖励来自每个时间对应的完整教师构象，而不是固定分箱。', '',
        '冻结 R11 使用 branch_mutation_v2 的两步共同祖先分支方向参考（SHA 50d723…）；共同深度 3 的七区域参考（SHA 434902…）仅用于第 22–24 轮，未替代 R11。', '',
        '统计输出保存全窗口矩、批次曲线、协方差/相关、置信区间、q 值、函数及导数和执行 receipt；不展开保存逐节点宽特征与所有原子对。多深度统计约 1.76 MB。历史原始轨迹保留，新增测试原始坐标在校验、发布报告后退休。', '',
        '## Analyst、Designer 与实施验证', '',
        '三套版本化 Skills/workflow 实际调用 sub-agent。Analyst 读取字面指令、工具 receipt、全窗统计、祖先证据和反证；Designer 读取完整已导入 Analyst 响应，在公式注册表内提出有界单轴参数。调用输入、响应、参考、源码和输出哈希绑定，编译器不执行任意 LLM 程序。', '',
        '真实 API 接口保留，与模拟后端共享消息、schema、证据及哈希验证。此次未调用外部 API。通用 Skills 不写死分子、区域编号或学习时间，也不包含 VPN/Git 环境指令。参数工程探索与新 LLM 设计明确分开记录。', '',
        '先检查空规则与 R26 标量/梯度/实际窗口/终态数值一致，再检查新奖励梯度、真实 VJP、窗口轨迹变化、动态覆盖与窗口外零注入；通过后才评价亲和力。已有梯度非零不能证明新指令生效。模块输出反传 RMS 是奖励敏感度，不能证明 attention 或 affinity 的因果重要性。', '',
        '## 实际可微奖励与控制', '',
        '保留完整教师 T_k(t)、坐标匹配和评分先验。四个邻近教师通过固定 Hungarian 对应；先验 log π ∝ −MSE/4 + 2(score−mean score)。', '',
        '`T′_k = T_k + sign·a_k·d_k`，`a_k=min(实际分支 RMS, .2 Å)`；只有观测到有效祖先对比与方向时 e_k=1。', '',
        '`p_original=π_k(1−αe_k)`，`p_virtual=π_kαe_k`。R11 候选 α=.025；R26 不使用新虚拟分支质量。', '',
        '`q_m=N⁻¹Σ_i w_mi||Y_i−T_mi||²`；`ρ(q)=δ²(√(1+q/δ²)−1)`；`R_t(Y)=τ logΣ_m p_m exp[−ρ(q_m)/τ]`，δ=1 Å，τ=.5 Å²，R11 默认 w=1。', '',
        '`∇_(Y_i)R=−Σ_m posterior_m·w_mi(Y_i−T_mi)/(N√(1+q_m/δ²))`。真实 FLOWR 条件 VJP 给出 `g_X=J_Fᵀ·scale·∇_(Y_world)R`；F 是 native 坐标端点预测，Y_world=scale·F+COM，自条件与匹配冻结，没有亲和力 head 梯度。', '',
        '当前引导按 η=.33 的预测 flow RMS 定标，逐原子 .15 Å、累计 RMS 6 Å，以及新增严重受体碰撞约束。全局奖励倍数可能被 RMS 归一化抵消，方向相对权重和实际剂量必须分别检查。主种子42、100步，学习窗口由输入绑定，当前0–.5，其余原生续推至1。', '',
        '## 与 FLOWR 底层原理的相容性', '',
        '实际 v=(F−X)/(1−t)，g=1/(t+.01)（t<.9），score=(tv−X)/(1−t+ε)，原生 Z=X+dt(v+g·score)+dt√(2g·noise)ξ。该实现噪声乘 dt，不能直接套用标准连续 SDE 分布保证。', '',
        '在原生 Euler 提议上加入位移可按离散等式写成 u=Δ_guidance/dt，并非天然不兼容。最终碰撞约束还依赖原生 Z/噪声，不能据此声称标准连续适应漂移或分布保证。当前 VJP 是条件端点几何代理的梯度，不是完整未来价值伴随、下一步 Jacobian 或精确 Doob/FK 控制。引导复用一次正常 target 前向并增加一次奖励反传；原有 untarget 诊断前向保留，没有额外生产 target 前向。每个梯度批次还做一次有限差分检查，两个步幅的中心差分共四次额外 target 前向，单列于 preflight；终态 target/untarget 重评分也属于评价开销。不做粒子重采样。', '',
        '实际 checkpoint 启用自条件，coord_scale=1.0。Steer 复制 current/prior/cond_batch，坐标控制保持原生条件更新；这是未完整迁移的联合状态信息，尚不能确定归因为性能差距原因。坐标进入 bond_refine，坐标引导造成化学图变化属于正常架构响应；无化学图相等 gate。', '',
        '## 逐轮公开假设与效果', '',
        '每轮只读公开 plan 与保留指标；负结果不改写假设。探索两批效应不等于独立确认。', '',
        '|轮|相对 R26 的均值差|实施/探索通过|改动与原因|实际推理 commit|',
        '|---|---:|---|---|---|']
    for entry in state['rounds']:
        n = entry['round']
        p = source / f'round{n:02d}_plan.json'
        plan = read_json(p)
        inputs[str(p)] = digest(p)
        retention_path = source / f'round{n:02d}/retention.json'
        retained = read_json(retention_path)
        inputs[str(retention_path)] = digest(retention_path)
        effect = entry['mean_vs_R26']
        formatted = '—' if effect is None else f'{effect:+.6f}'
        rationale = str(plan['reason']).replace('|', '/')
        changed = str(plan['changed_axis']).replace('|', '/')
        lines.append(f'|{n}|{formatted}|{entry["implementation_feedback"].get("implementation_passed")}/{entry["screening_admissible"]}|{changed}；{rationale}|{retained["inference_commit"][:12]}|')
    lines += ['', '## 结论与下一步可检验特征', '',
        '本轮证实新知识可以实际改变奖励梯度和生成轨迹；收益是否成立以六批冻结确认表为准。方向反证与剂量实验揭示了弱空间关联、归一化和个别样本的大效应问题。不能将“指令已实施”写成“指令有效”。', '',
        '两步分支场的正反方向对照在 α=.1（第9/10轮），不是冻结 R11 的 α=.025；七区域场的等预算正反对照在 α=.05（第22/23轮），不是 R11 的同一方向场。冻结确认即使通过，也检验的是整个候选方案，尚不能把收益唯一归因于方向知识而排除混合平滑等解释。α=.025 的等预算反方向/随机方向是后续独立检验，本轮未追加超过30轮的实验。', '',
        '尚值得独立注册并检验的特征：期望选择与复制噪声分离；局部刚体姿态与内部形变；条件记忆与预测一致性；教师模式切换/梯度冲突；跨区域距离与角度 motif；被观察终态高分路径的祖先信用；真实模块因果干预。不能填补淘汰分支未知未来，也不能把噪声点云伪称中间物理能量。', '',
        '这些建议未被追记为早先 Agent 输入。下一次探索应先补独立紧凑工具与来源 receipt，再实际调用两角色，检查指令响应，最后逐轴配对验证。', '',
        '## 代码、报告与依据', '',
        '数据工具、角色/API 和生成命令见同目录 `reproduction.zh-CN.md`；公式与限制见 `scientific_design.zh-CN.md`、`remaining_features.zh-CN.md`；独立校验见 `confirmation_integrity.json`；全部配置、哈希和结果见各 round 的 plan/retention。', '',
        '[FLOWR 作者代码](https://github.com/jule-c/flowr_root)、[FK steering](https://arxiv.org/abs/2501.06848)、[Flow guidance](https://proceedings.mlr.press/v267/feng25s.html)、[Optimal-control flow matching](https://arxiv.org/html/2410.18070v3)、[FK-Flow](https://arxiv.org/html/2509.01543v1)、[Price 分解](https://pmc.ncbi.nlm.nih.gov/articles/PMC4415573/)提供机制与分析依据；本项目不宣称完整实现这些方法或其分布保证。']
    target = output / 'final_report.zh-CN.md'
    target.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    write_json(output / 'report_manifest.json', {'input_dataset': str(source),
        'configuration_dataset': str(configuration), 'input_files': inputs,
        'script_sha256': digest(__file__), 'generation_attempts': attempts,
        'outputs': {target.name: digest(target)},
        'semantics': 'Post-completion descriptive synthesis, no new fit or selection rule'})
    return {'rounds': 30, 'generation_attempts': attempts, 'adopt_candidate': adopted}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dataset', required=True)
    parser.add_argument('--configuration-dataset', required=True)
    parser.add_argument('--output-dataset', required=True)
    args = parser.parse_args()
    print(write_report(args.input_dataset, args.configuration_dataset, args.output_dataset))
