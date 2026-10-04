# Designer 执行审阅：CK2 多阶段 live gradient guidance

日期：2026-10-05。身份：**Designer subagent simulation**，延续用户授权的 LLM 接口模拟。本次只读复核生成和评价实现，仅写本文；未改奖励、原型、控制参数、冻结决策或工程源码，未执行新的 GPU 生成。未读取 validation/heldout 来选择规则，也未读取正在运行的独立比较结果或按亲和力选择设计。

审阅对象为 `multistage_reward.py`、`multistage_controller.py`、`multistage_evaluation.py`，以及支撑它们的 instrumentation、gradient_controller、controller 和 comparison 中相关代码。工程证据来自 `results/multistage_v1/pilot_evaluation/` 下的 calibration、live_gradient_preflight、zero_native_equivalence、generation_audit 和 control_summary。本文记录的评价问题基于下方哈希对应的审阅版本；主执行 Agent 已接收问题并计划修正评价，本文不把待修正事项写成已经验证通过。

## 结论与边界

没有发现足以否定本轮 live 梯度执行或要求改变已冻结科学假设的实现错误。当前实现确实对**当前生成坐标经过实时 endpoint 预测的链式梯度**求导，缺失 N/O/S 时跳过该粒子，保留 native 类别更新，并在完整 100 个积分步中执行多阶段组控制。η=.30 按事前规则成为三个候选中最大的可行值；这不是最优权重，也不是生物学优化成功的证据。

独立结构评价存在需要修正或明确解释的口径：形状距离的重原子定义、梯度可用性阈值，以及未构建候选的碰撞缺测。它们不需要重写或重跑已冻结的生成算法，但在独立比较形成科学结论之前，应使用修正后的评价代码重算并记录版本。原型哈希还存在可证明的换行表示差异，应补元数据勘误，不回溯改动已经运行的程序。

## Live 梯度和时域执行

令当前 native 坐标为 x，实时模型预测为 hθ(x,t)，世界坐标 endpoint 为 y=s·hθ(x,t)+COM。代码在 `torch.enable_grad()` 内重新创建当前 x 的可求导副本，计算实时 y、两项 N/O/S 特征 z 和混合奖励 R，随后执行 `grad(R.sum(), x)`。模型参数被冻结，当前求导中的原子类别来自 detached endpoint argmax，先前自条件和先前积分步不构成跨步反向传播。其梯度为

`g_native = (∂y/∂x)ᵀ ∇y R`。

这与在保存终态上计算一个空间方向再直接相加不同，也没有把目标曲线的时间导数当成空间梯度。每个粒子的奖励求和，不取 batch mean。类别变化仍由 native 采样完成，坐标控制可以间接改变以后预测的类别，但没有对 argmax 求导。

GPU preflight 的解析方向导数为 **0.3967717886**；native L2 扰动 ε=.001、.003、.01 的中心差分相对误差分别为 **1.0925%、0.3114%、0.0710%**，三个数值方向导数均为正，参数梯度记录为 false。这支持本轮实际模型上的 live 梯度链路。它只验证首次 preflight 的一个方向/时刻，不是对所有粒子、所有 100 个 Jacobian 或类别跳变点的穷尽验证。求导与差分均固定当次 endpoint 类别 mask，不能据此宣称离散类别优化已验证。

`prepare()` 强制 `steps=100`；回调被插入每次 native 积分后的 proposal 记录之前。generation_audit 对五个 pilot 组均记录 **100 步、每步 12 个槽位、1188 条相邻步坐标父子链接**；所有组的重采样事件均为空。多阶段控制组的 reward_evaluated_fraction=1，每批实际非零注入步数=100，日志中 t≥.5 的粒子步全部非零。该批汇总的“某步非零”表示至少有一个粒子非零；对逐粒子全程非零的结论应读取原始 trace，而不能只用这个计数推断。

实际控制时刻为 **t=0,.01,…,.99**，最后保存 t=1 的 native 终态，不在 t=1 再添加第 101 次奖励注入。观察证据只覆盖 [0,.5]；之后固定 μ(.5)、Σ(.5)，继续计算同一个核心奖励。这是明确的 continuation 假设。`multistage_window` 的 `t<=.5` 包括 51 个步点，full 组包括 100 个；两组总注入剂量不同，比较时应同时呈现累计实际位移。

零强度检查对 coords、atomics、bonds、charges、mask 均为精确相等；generation_audit 中 unguided 与 gradient_zero 的完整轨迹 SHA256 也相同。这支持新增梯度路径在零强度时保持该 pilot 的 native 输出和记录，不应扩展为所有未测试模型配置都已验证。

## 类型缺失、尺度与奖励构造

两个特征都使用当前 endpoint 预测的 N/O/S 槽位和 active mask。没有 N/O/S 时，内部临时使用 all-active 数值 mask 以避免 logsumexp 的未定义计算，但该粒子的最终奖励通过 validity mask 置为计算图中的零，坐标梯度为零；遥测把奖励、特征及责任度写为 null，并记录 `no_endpoint_N_O_S`。因此这个内部零值不是“获得完美奖励”，也不应该加入已观测奖励均值。现有缺类型测试确认缺失粒子的梯度为零且不影响另一有效粒子。

奖励由两个相关距离组成一个 joint patch。14 个 discovery batch 的 retained 祖先特征均值作为 14 个等权代表，26 个 PCHIP 节点形成 25 个局部段；协方差保留全部 51 个时刻并作 SPD 凸插值。after-.5 冻结与 static_full 的 .5 全程参考均按声明执行。混合的代表是 batch 均值，不是真实姿态 mode；两个距离也不足以恢复三维构象、原子对应、氢键或化学身份。

Mahalanobis pseudo-Huber 的各 mode 代价和 log-sum-exp 实现与冻结公式相符。Huber 限制的是特征空间中的远场梯度增长，**奖励数值并不有界**；live endpoint Jacobian 仍可能放大拉回梯度。PCHIP 局部插值与协方差缩放是冻结的目标构造，η 是额外位移强度，两者未混成同一个可调权重。没有使用亲和力头作为梯度奖励。

控制器先运行完整 native 积分得到位移 u，再按

`δ_native = η·RMS(u_world)/(s·RMS(g_native)) · g_native`

形成额外修正，随后乘每粒子一个正标量来满足上限。由于统一坐标尺度 s，归一化方向与 world 基底一致；native 位移已经含实际积分步长和随机项，没有再次额外乘 dt。RMS 梯度≤1e−8 或 native 位移为零时不强迫运动。对超过门槛但很小的梯度仍会归一化为请求强度，这会弱化梯度幅度本身的信息，必须依靠实际轨迹和终态评价检验是否造成局部振荡或过度控制，不能把它称为已证明最优的步长策略。

## η=.30 的工程含义

各候选均有 1200 个记为可用的 active 粒子步，已建且连通的 unguided 基线为 11/12。校准只使用工程条件，`uses_affinity_for_selection=false`；不是从亲和力成绩中挑选 η。

| 指标 | η=.05 | η=.15 | η=.30 |
|---|---:|---:|---:|
| 裁剪粒子步比例 | 1.00% | 13.67% | **90.50%** |
| median(actual/requested RMS) | 1.0000 | 1.0000 | **.7158** |
| median(actual/native RMS) | .0500 | .1500 | **.2147401** |
| 与 native 位移负向点积的比例 | 59.25% | 60.42% | **63.25%** |
| 发生几何回溯／完全拒绝 | 0 / 0 | 0 / 0 | 0 / 0 |
| 平均累计已接受 RMS 路径（Å） | .2743 | .7769 | **1.1346** |
| 已建且连通候选数 | 11/12 | 12/12 | **12/12** |
| 已测候选出现 <1.2 Å 碰撞的个数 | 0 | 0 | **0** |

**90.5% clip** 是 1200 个可用粒子步中 1086 次 `cap_factor<.999999`，不是 90.5% 的原子、不代表 90.5% 的分子失败，也不是回溯拒绝率。当前 100 步、每原子每步 .025 Å 上限已蕴含每步 RMS≤.025 Å，2.5 Å 的累计预算主要是最坏加法路径上界，因此此高裁剪率主要反映逐步原子位移上限经常生效。最大记录值 .0250000041 Å 与 .025 Å 的差约 4.1×10⁻⁹ Å，属于 float32 舍入量级，不应当成科学上有意义的越界。

**中位 actual/native=.2147** 表示典型粒子步实际额外 RMS 位移约为该步 native RMS 的 21.47%；名义请求是 30%，典型实际达到请求的 71.58%。它通过事前 `.5` 的实现比例门槛，但说明“η=.30”不能被报告为所有时间/粒子实际都得到 30% 的控制。应将 η、请求位移、裁剪后的位移和累计路径并列报告。

**63.25% native 冲突** 是 759/1200 次梯度与本次 native 位移的余弦小于零。它只说明投影方向经常相反；native 位移含随机部分，也承载模型本身的结构构造，不是某个已知最优奖励梯度。负余弦既不是亲和力改善证据，也不能单独判定控制失败。应查看负余弦时的实际剂量、geometry guard、独立结构指标及最终有效性，不能以“反向 native”作为新的奖励目标。

几何回溯约束的是额外修正相对**同一个 native proposal**是否新引入严重受体原子对接近，以及配体成对距离变化。已有 native 缺陷不因此消失；被检查的是当前加载的局部 pocket 原子集合，不能宣称全蛋白或真实能量安全。配体 .05 Å 成对距离变化上限已由两个原子各自≤.025 Å 的修正蕴含，主要是显式一致性检查。累计 2.5 Å 限制累计额外路径，不限制以后 flow 放大的最终位置偏移。

回溯没有检查非线性奖励实际增量。因此即使额外修正沿当前解析梯度，完整 native+guided 更新后的 R 也不保证逐步上升；目标本身随 t 变化，更不能把奖励曲线跨时刻增加等同于同一个势函数的单调上升。

## 独立评价审阅与待办

**E1：重原子形状距离口径，需修正评价。** 审阅版本的 `nearest_pose_chamfer()` 宣称 nearest-heavy-atom，但接收的只是坐标和 active mask，没有类别参数；H 与 PAD 都存在于词表，函数自身不能保证排除它们。预测为 H/PAD 的 active 槽位可能影响距离。应明确 heavy mask=active 且非 H/PAD，缺少重原子时保留 missing 与覆盖率；同时保留两距离的 N/O/S coverage，避免将二者分母混为一谈。主执行 Agent 已表示将按此修正评价，不改变生成。本文未把该修正宣称为完成验证。

**E2：可用梯度的汇总阈值应与执行一致。** 控制器使用 `gradient_rms_native>1e−8`，审阅版本的汇总使用 L2 `gradient_norm>1e−8`。对 n 个原子，L2=√n·RMS，故极小梯度区间可能被汇总算为可用，实际却跳过。应直接使用已记录的 RMS 字段，并严格沿用 `>`，而非将等于阈值也算可用。pilot 汇总与实际活动没有显示足以推翻 η=.30 的证据；仍应重算工程汇总，确认 eligibility 和 selected_ratio 不变。该修正是计数口径修复，不是重选奖励或读取亲和力后调参。

**E3：终态严重碰撞门槛有缺测范围。** 上游 `final_metrics()` 主要对成功构建的分子记录 `pairs_below_1_2A`；校准用 `fillna(0)`，把缺测也算作没有碰撞。现有 `final_clash_candidates=0` 应解释为已测候选中没有出现严重碰撞，不能证明所有未构建 raw 终态都没有碰撞。η=.30 的 12 个候选全部已建且连通，本次没有可见迹象利用缺测获得资格，但 unguided 的失败候选不能被证明无碰撞。建议独立评价另列原始终态重原子/active 槽位的全候选碰撞及覆盖率，保留原冻结门槛值，不回溯重选。这里的 clash count 实际是“至少一个碰撞的候选数”，不是原子对总数。

**E4：通用校准入口对输入完整性的断言可加强。** `calibrate()` 明确断言 unguided 有 12 个终态，却未对每个 guided arm 显式断言同样 12 个且槽位唯一；完整版独立 `summarize_records()` 有重复/配对人群检查，pilot 入口没有等价检查。本次外部 generation_audit 已记录每组 12 个、100 步和失败保留，未发现当前 pilot 缺组；后续复用应将这些前置条件纳入统一审计。preflight 和 zero-equivalence 也依赖执行脚本与外部记录，不能仅凭 calibration.json 的 selected 状态声称所有独立前置验证已通过。

**E5：late 的汇总边界包含 .5。** 现有 `late_nonzero_particle_fraction` 使用 `time>=.5`，因而 window-only 在 .5 的最后一次注入也计入 late。对 continuation 的文字应明确是 `.5<t<1`；评价可以分别呈现 .5 边界与严格后半程，避免把 window-only 的边界活动称为后半持续控制。full pilot 此项为 1，边界选择不改变其后半实际活动的结论。

二维 Mahalanobis 白化与经验 energy statistic 计算方向正确，可度量 reward-observable 分布相似性；它是目标同空间的 imitation 指标，不能作为完全独立的结构成功标准。固定 receptor frame 的形状 Chamfer、接触加权 centroid、Rg、all-atom 距离、化学图指纹及终态有效性提供互补信息。nearest-SMC Chamfer/Tanimoto 都可能在模式塌缩时仍较好，需要同时报告 unique structures、覆盖率和参考样本数。现有终态 unique-SMILES 统计有价值，但不能替代三维构象多样性。

独立比较以 **batch** 为统计单位正确；4 个新批次只支持探索性结论。四批精确双侧 sign-flip 的最小可得 p 通常为 2/16=.125，不能把很多粒子或 100 个时刻当作独立重复来获得更小 p。对 undefined 距离、无可用 NOS 或无可用重原子的批次应保留缺失；不能用零填充制造“完全匹配”。评价的新 SMC 组是每批 16 粒子，与历史发现集的 50 粒子条件不同，应按实验记录声明。

## 原型元数据勘误

冻结 IR 的 nested `retained_target_origin.prototype_sha256` 仍为旧值 `33ef28e79fde101b967541ebc1e3c22ad8f6166eb595ed27c2803d15da7e781a`。当前 canonical LF 文件的 SHA 为 `5c3bf2fa2d2b5cd4397a4529a70c68f133f699aba0aa037b2ecce56a55b8025e`，compiled program 已绑定后者。

本审阅直接读取当前文件，将 LF 字节替换为 CRLF 后重新计算 SHA，**精确得到前述旧值**。这确认是换行编码差异，不是数值原型变更。应新增 metadata 勘误描述这层表示对应关系；不改已执行的 program、Designer 文件及其哈希来“清理”历史。

reference packet 保存 regularized Σ，没有把原始 covariance 与全部 retained spread 都嵌入；builder 保留的发现集候选表可以重建这些诊断。后续独立导出 raw covariance、within-retained spread 可以补齐审计，但不应将补充审计冒充本轮已改变了混合目标或在 pilot 后重新拟合了模式。

## 本次验证与可追溯性

只读运行 `.venv/Scripts/python.exe -B -m pytest tests/test_multistage_guidance.py -q -p no:cacheprovider`，**10 passed（12.52 s）**。覆盖 typed 特征一致性、合成 live 链式差分、多时刻/逐粒子独立性、缺类型零梯度、刚体/置换性质、局部插值、协方差与祖先计数、强度与 cap、几何回溯和评价基本性质。这些本地测试不能替代真实 GPU 多时刻差分或独立终态验证。

pilot generation_audit 的执行提交为 `3322d21ca67c4475f0879c2bfaa70d7c1fef1049`。审阅时本地文件的 SHA256 为：

| 文件 | SHA256 |
|---|---|
| multistage_reward.py | `1d416a4206f9061d1a50c56f546caf6e0a4ad6ad35dcfaf9776bbb7ad5c4d9e6` |
| multistage_controller.py | `fb941d4237cef9a1337b52571045ae5cc536ba367558a5700937cae0ead25009` |
| multistage_evaluation.py（上述问题的审阅版） | `048027a5f6982cf484c6896bb3f09344fbf9b7a6bdf319a23457094c38ad8b32` |
| reward_program.json | `036f03d27a50b2f8a66f170e16bbaf4a3463f1505b62723bb2c0e682be3f50b3` |
| reward_catalog.json | `af3d15470fcfbaf6c5469b07b2a2bdce5409ca3cc48780a8c9e140cfa06dba01` |
| reference_packet.json（canonical LF） | `5c3bf2fa2d2b5cd4397a4529a70c68f133f699aba0aa037b2ecce56a55b8025e` |

reward_program 的本地字节哈希与 calibration.json 的 source_program_sha256 一致。本文没有读取新比较的终态分数，故没有对结构模仿、真实结合能力、物理能量或亲和力提升作结果判断。后续报告应在保留 η=.30 和当前目标不变的前提下，单独给出修正评价的版本、缺失覆盖率、实际控制剂量及独立终态比较。
