# Analyst 标签、教师覆盖与后半程推理核对

2026-10-09。本次核对实际运行绑定的奖励程序、教师库、原始 Steer 谱系和已完成的 1,000 样本对照数据；没有进行新生成或改动奖励、Skills、推理策略。R11/R26 指奖励与控制代码的实验 checkpoint，三组使用同一个 FLOWR_ROOT v2 模型，模型 SHA256 为 `f28e863b2b208718f3d3f85f09837c2f907a71123436db6f98a25d2f1858b6a0`。

**结论：历史探索不全是只学习窗口内最高分，但当前保留的 R11/R26 教师仍按即时分数建立。** 用户指出的后期变优路径覆盖不足，有实际数据支持。同时，后半程的原生积分策略一致，控制结束的边界却相差一个选择事件：原始 Steer 在评分时间 0.50 仍执行重采样，作用于状态时间 0.51；R11/R26 的最后一次坐标引导作用于状态时间 0.50。因此严格的整个 0.5–1 区间并非完全相同，从状态 0.51 起才都进入无控制的原生续推。

**更正之前报告。** 之前将“曾探索终端信用”混同为“当前 R26 已使用终端信用”，不符合部署文件。已更正 [尾部报告](../steer_tail_20261009/report.zh-CN.md) 的机制解释，原有终态、谱系与尾部统计不受此更正影响。以下以绑定的库文件 SHA 和逐步执行记录为准。

**历史 Analyst 分析与实际保留配置。**

| 阶段或方案 | 分析/标签来源 | 是否属于当前 R11/R26 |
|---|---|---|
| 早期坐标挖掘 | 全群体的即时分数相关、期望选择偏移、实际复制偏移、同谱系对照和短期滞后 | 是分析基础；并非只看一个最高分样本 |
| 原始教师构建 | 每个实际学习节点、每个 donor 批次取最多两个即时高分且空间有区别的 endpoint 点云 | 是，两者使用完全相同的原始教师坐标与分数 |
| 历史 R17/R19/R24 终端信用探索 | 沿选择谱系回溯 step 99 在线联合潜变量 head 后代均值，灭绝支系记删失 | 曾测试，未替换保留的 R26 |
| elite_path20 探索 | 使用真实最终解码重评分回溯信用、保留路径 ID，探索路径潜势 | 曾测试；独立面板落后旧 R26 0.09479 pIC50，保留旧 R26 |
| 当前 R26 | `endpoint_pointcloud`，即时评分教师的局部多点云 mixture | 是，未绑定终态信用库 |
| 当前 R11 | `endpoint_branch_mixture`，原教师加 lag-two 同祖先自然分支坐标/评分协变方向 | 是，virtual mass 0.025；没有把教师标签换成终态分数 |

对应实现与历史记录：[即时教师](../../../src/evomolsteer/continuous/affinity_geometry.py)、[全群体挖掘](../../../src/evomolsteer/continuous/coordinate_mining.py)、[历史终端信用](../../../src/evomolsteer/continuous/terminal_lineage.py)、[分支坐标方向](../../../src/evomolsteer/continuous/branch_mutation.py)、[终态路径实验结论](../elite_path20_20261007/report.zh-CN.md)。

当前 [Analyst Skill](../../../skills/analyst/SKILL.md) 要求区分当前分数、选择、存活和最终质量，并允许终态信用回溯；它没有规定“只能分析当前最高分”。但 Skills 中的允许或要求不能证明运行时教师库已经使用了终态标签。当前 R11 的归档 Agent 输入与响应使用 selection innovation、lag-two 条件坐标协变；奖励实际读入的仍是以下库：

| 配置 | 实际绑定库 | SHA256 |
|---|---|---|
| R26 | `configs/experiments/skill_ablation_v1/endpoint_reference.json.gz` | `d706173b74a937c5becac08ccbcff187f48e50ca3303d9021667e466fa5fb77c` |
| R11 | `configs/experiments/flowcompat30_v1/branch_reference.json.gz` | `50d7232b5501152c87fe20329ebfa0e4b0fbfd2f519cdd0e1e6dad9b8f23f801` |

两者的 `label_semantics` 均为 recorded joint forward head label，明确不是 endpoint 坐标独立重评分；没有 terminal-credit 字段、显式 teacher slot 或路径 ID。50 个 frame 对应评分时刻 0.00–0.49，每个 frame 28 个老师（14 个 donor 批次各两个）。1,400 个老师全部匹配原始记录的坐标和即时 head 标签：坐标序列化误差最大 7.085×10⁻⁷ Å，标签误差最大 1.908×10⁻⁶。

**最终好分子的祖先，确实被当前教师筛选大量遗漏。** 为避免把验证批次作为学习源，覆盖统计只使用原始发现批次 0–13。该子集有 34 个有效高分终态、20 种唯一化学图；全 20 批是 37 个高分终态、23 种图。阈值沿用既定的预测 pIC50 ≥ 8.258901977539063，没有根据本次覆盖结果重新选择。

回溯每个高分终态在实际选择节点的祖先，再与部署教师点云匹配，得到：

| 评分时刻 | 被教师直接覆盖的最终高分槽位 / 34 | 被覆盖的不同高分祖先 / 当时祖先总数 |
|---|---:|---:|
| 0.00 | 0/34 | 0/8 |
| 0.10 | 0/34 | 0/8 |
| 0.20 | 11/34 | 2/8 |
| 0.30 | 1/34 | 1/11 |
| 0.40 | 6/34 | 3/16 |
| 0.49 | 10/34 | 7/27 |

全 50 个实际学习节点的终态槽位覆盖率平均为 18.82%，其中 8 个节点为零。每个时间点都独立对应原记录，不是把数据划为六个区间再分析。这里只列少量节点用于展示。

由于原库删除了 teacher slot ID，匹配保留所有在原子槽位顺序、受体固定坐标系下符合序列化误差的候选别名。因此该值是包含别名的直接收录覆盖，不能解释为其他坐标没有受到邻近教师的梯度吸引；34 个槽位也不是 34 条独立谱系，不能按 34 个独立样本做显著性检验。

“最终最优分子都没有早期高分”仍过于绝对。原始 batch 4、最终槽位 27 的祖先在初始批内只有约第 28 百分位，随后在 0.3/0.4 时已达到约第 86/84 百分位，在 0.5 是批内最高。它在 0、0.1、0.2、0.3、0.4 时均未进入该批两个教师，到 0.49 才被覆盖。batch 11、最终槽位 30 则初始已约第 94 百分位，到 0.5 降至约第 50 百分位。关键问题是即时排名不能稳定代表最终价值，且最多两个教师会遗漏非最高分但仍有未来潜力的路径。

原始 20 批中，0.50 在线祖先分数与最终有效分子重评分的批内 Spearman 相关中位数仅 0.083841。并且最后一次选择后，batch 4 的两个复制后代在 0.51 坐标、原子、键、电荷、mask 完全一致，最终 pIC50 分别为 7.822988、8.625889。早期同一状态可产生不同结果，不能把事后最优后代唯一归因于某个当时就可识别的优势区域。具体谱系与指标见 [尾部报告](../steer_tail_20261009/report.zh-CN.md)。

**后半程实现核对：积分器一致，选择边界有差异。** 核对原始 20 批谱系、已完成 R11/R26/无引导各 20 批的逐步日志和 selected indices，三组均使用 100 次线性 Euler 更新：

| 项目 | 原始 Steer | R26 | R11 |
|---|---|---|---|
| 连续坐标原生噪声 | SDE，coord noise level 0.2 | 相同 | 相同 |
| 原子/键等分类采样 | `uniform-sample`，noise level 1 | 相同 | 相同 |
| corrector / cosine scheduler / 对齐 | 0 / false / 无 rotation、permutation alignment | 相同 | 相同 |
| 窗口内操作 | 按即时分软重采样 | 坐标 reward VJP，无粒子重采样 | 坐标 reward VJP，无粒子重采样 |
| 最后受控更新的两个时钟 | score 0.50 → state 0.51，重采样 | score 0.49 → state 0.50，引导 | score 0.49 → state 0.50，引导 |
| state 0.50 → 0.51 | 原生更新后仍复制/淘汰 | 原生更新，零注入 | 原生更新，零注入 |
| state 0.51 → 1.00 | 原生续推，无选择 | 相同策略，无引导/选择 | 相同策略，无引导/选择 |

原始生成脚本使用包含右端点的评分窗口；先从当前状态进行积分，再复制积分后的 proposal。故评分截止 0.5 不等于状态截止 0.5。所有原始批次均有 51 次选择，最后影响 0.5099999904632568 的状态；R11/R26 均有 50 次受控更新，后 50 次奖励未求值、坐标注入逐元素恰为零，selected indices 全程恒等映射，未实施 Steer。

这里以实际 runtime integrator 为准。配置解析字段 `flowr_args.use_sde_simulation=false` 不能单独证明运行确定性 ODE：原始 `_generate_selective` 因 `apply_guidance` 启用 SDE，当前控制器保留同样的原生噪声；两组新配置明确记录 runtime SDE=true。上游 `fm_pocket.py`、`integrator.py` 文件 SHA 均与原始 provenance 完全匹配。控制器窗口结束后回到原生 forward、原生积分和原生 self-conditioning 更新，未重置为历史 Steer 的条件张量。实际 self-conditioning 值、初始状态与随机流当然会随前段路径不同而不同，这不属于后半程算法改变。

因此当前证据支持“0.51 后使用相同原生策略”，不支持“整个 0.5–1 控制操作严格一致”，更不支持“后半程轨迹应相同”。历史 Steer 批次 0–19 与新生成批次 55–74 不是同种子路径配对；重采样也改变随机抽样消耗和群体状态。没有测量该额外选择事件的贡献，不能将现有尾部差距归因于这一事件或假定其影响可忽略。

**对下一版 Analyst/Designer 的建议。** 特征测量与奖励控制窗口可以继续动态指定为 [0,0.5]；标签来自最终解码分子不意味着必须在 0.5 后加引导。应区分三个概念：即时选择优势、窗口内存活、原生续推后的最终质量。

可将最终有效高亲和力结局沿真实 selected indices 回溯到窗口内节点，构建条件尾部成功比例、上分位和均值的多种信用，并保留计数、删失和共同祖先关系。以相近时间、原子数、即时分数、同祖先自然分支为对照，重新比较空间区域占据、局部接触、坐标位移和速度变化；特别检查“当下普通但最终高分”与“当下高分但最终普通”的区别。未留下后代的分支没有已知自然续推终态，应记未知而非零分失败。

目标应接近 Qτ(s,t)=P(最终有效且预测分≥τ | 当前联合状态与原生续推)，其中状态还包含模型条件信息。现有单次谱系只能提供稀疏、选择偏倚下的事后证据，不能准确标定所有被淘汰分支的该概率。若做可微代理，可用带终态信用、保留多空间模式的坐标潜势，让 Designer 同时看到当前分与最终信用；不要把时间趋势导数当空间梯度，或仅更改 Skill 措辞却继续传入旧的即时教师库。没有必要引入 affinity head 求导，也不应限制自然化学图变化。

这一方向不是未经尝试的新方案：历史终态均值和真实 final 路径方案已有失败验证。下一轮须先确认 Agent 输入、输出和实际 reward reference 都响应了标签替换，再判断改善是否成立。过去失败可能来自删失、稀疏终态标签、共享祖先分化、点云邻域/剂量失配，而非证明终态信用毫无价值；这些是待检验原因，不能替代消融结果。

为严格比较同一后段，可保留原始 Steer，另建状态截止 0.5 的 Steer 对照；或把两个方法的共同状态边界设为实际最后选择后的 0.51，并重新定义一致的学习/控制范围。不能只在说明文字中把原始 0.50 选择事件忽略。本轮没有自动更改该边界或开展实验。

**可复核文件与复现。**

- [审计 JSON](audit.json)：模型/教师/代码指纹、20 批原始边界、60 次续推执行核对与数据指纹。
- [逐批次教师覆盖](teacher_coverage_by_batch.csv)、[逐实际节点覆盖](teacher_coverage_by_event.csv)：700 条批次事件、50 个实际节点，无结构副本。
- [只读审计脚本](../../../scripts/audit_analyst_credit_and_native_suffix.py)、[8 项边界/动态窗口/零注入测试](../../../tests/test_analyst_suffix_audit.py)。

```powershell
.venv/Scripts/python.exe scripts/audit_analyst_credit_and_native_suffix.py --completed-report docs/experiments/flowcompat_scale1000_20261009/completed_20261009_1836 --original-root data/raw/ck2_clk3_lineage_20261003 --original-campaign main1000_w050 --steer-metrics docs/experiments/guidance_vs_steer_20261007/steer_full1000/candidate_metrics.csv --r11-dataset test/data/flowcompat_scale1000/R11_restored --r26-dataset test/data/flowcompat_scale1000/R26_native_restored --r11-reference configs/experiments/flowcompat30_v1/branch_reference.json.gz --r26-reference configs/experiments/skill_ablation_v1/endpoint_reference.json.gz --threshold 8.258901977539063 --output docs/experiments/analyst_label_suffix_audit_20261009
.venv/Scripts/python.exe -m pytest tests/test_analyst_suffix_audit.py -q
```
