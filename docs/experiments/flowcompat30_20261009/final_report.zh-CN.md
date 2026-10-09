# 历史 Steer 路径挖掘与 FLOWR 梯度控制：30 轮最终报告

本文件在全部实验结束后生成，汇总公开的证据、假设、公式和可复现参数；不作为已冻结 Agent 的输入，也不记录私有思维链。

完成 30/30 轮真实 FLOWR 推理，共 3500 个生成尝试。轮数包含数值 null、反方向和新批次验证，不是 30 个独立 LLM 设计；重复控制也不代表不同分子。

## 最终确认

冻结候选来自第 11 轮，两批探索均值改善 +0.011002 pIC50；随后用六个新批次验证，验证期间未调奖励或强度。

|组别|全部尝试平均预测 pIC50|有效分子平均预测 pIC50|有效/PB-fast|应变中位/P90，kcal/mol/重原子|应变覆盖|极高分产率|
|---|---:|---:|---|---|---|---:|
|冻结 R11 候选，N=300|7.595054|7.598953|97.0%/97.0%|0.5144/1.0287|291/300|0.7%|
|历史最优 R26，N=300|7.574005|7.577157|97.7%/97.7%|0.5183/1.0657|293/300|0.3%|
|无引导，N=300|7.453901|7.462849|93.7%/93.3%|0.5161/1.3613|281/300|0.3%|

候选相对R26：配对均值差 +0.021049，六批 bootstrap 95% 区间 [+0.010423, +0.037746]。

候选相对无引导：配对均值差 +0.141153，六批 bootstrap 95% 区间 [+0.096771, +0.185625]。

独立完整性审核通过：True；可用于效果决策：True。
预登记升级条件全部满足：True。候选具备升级证据；实际默认切换需以代码中的活动配置为准。

实际活动配置：confirmed_flowcompatibility；candidate_enabled=True；奖励 SHA=65e21d4b3cd2f6472fbe4f6c4710c5fc9b4974e15a24de876a997ac63296834c。

检验采用批次为重采样单位，不能把 300 个后代当作 300 次独立实验。应变只统计有效且 MMFF 优化收敛的分子，是生成构象到同图松弛构象的能量差代理，表中单列覆盖；不是全部尝试均有物理能量。指标来自同一预测亲和力模型及构象检查，尚无实验结合常数验证。

事后配对分布显示：第26/28/30轮各100个尝试中，绝对差异最大的5个尝试分别占总绝对差异的99.35%/85.03%/68.24%；配对差异中位数分别为-7.15e-06/+1.48e-05/+5.01e-06 pIC50。六批均值改善并不意味着大多数分子明显改善；收益集中于少数路径。此描述未新增选择阈值或改变冻结确认。

## 与历史 Steer 的关系

原 Steer 共 1000 个尝试，平均预测 pIC50 7.651480，最高有效分数 8.704806，极高分产率 3.7%；应变中位/P90 0.5303/1.3809。

历史 Steer 的发现供体参与奖励构建，样本量、在线评分和分支探索预算也不同；这是非配对参照。本轮无引导和梯度组也保留原生 SDE 与离散随机采样。Steer 特有的是复制高分路径后获得多个随机后代并在线重新分配群体预算，历史结果具有更高的极高分尾部产率；不能用不等预算确定该优势的因果幅度。当前局部几何奖励尚未证明采样分布等价，也未证明更自由区域与邻域适配同步改善。

## 从筛选机制到数据特征

Steer 通过评分提高候选的期望复制数，再复制生成状态和条件并继续随机生成。一次实际存活包含抽样运气；立即复制的同胞没有几何变异，被淘汰分支的未来也没有观察。分析因此回溯真正共同祖先、折叠复制、比较不同即时父分支，并以祖先和批次处理相关性。

区分当前状态 X、模型端点预测 Y、原生积分提议 Z、评分和复制索引。在线评分对应 Y；复制的是 Z 及条件。整窗使用输入声明的选择节点，不再人为分成固定时间箱。当前评分 0 至 .49，最后提议到 .5；终态仅用于效果评价。

|新增工具|关键结果|对奖励设计的含义|
|---|---|---|
|selection innovation|内部形变约占坐标对比 78.65%；已观察子代再预测分数平均下降 .03633|存活不意味着绝对分数逐步上升；分别检验探索与稳定性|
|共同祖先 branch mutation|调整后联合/内部再预测 RMS 与评分变化相关 −.14993/−.13733；q≈.000282/.000970；绝对协方差仅约 −3.60e−5/−3.42e−5 Å·pIC50|显著相关不能直接变成强坐标力；自然变异幅度与方向分开定标|
|multi-depth mutation|深度 2/3/5/8/13，137 特征；部分区域方向关联存在，但逐节点跨批方向弱|整窗趋势拟合与留一批验证；保留缺失早期支持和反证|
|common-depth regional synthesis|只用发现数据选共同深度 3、7 个区域，285/1400 教师具备方向支持|不混合观察跨度，完整联合形变保留空间相容关系|

后续区域 XYZ 协方差函数采用整窗多项式候选、留一批验证和 1-SE 简化；本次合格区域方向选择常数函数，导数为零。没有观测支持人为制造区域多阶段方向切换。这不表示 R11 每节点的实际分支方向恒定。动态奖励来自每个时间对应的完整教师构象，而不是固定分箱。

冻结 R11 使用 branch_mutation_v2 的两步共同祖先分支方向参考（SHA 50d723…）；共同深度 3 的七区域参考（SHA 434902…）仅用于第 22–24 轮，未替代 R11。

统计输出保存全窗口矩、批次曲线、协方差/相关、置信区间、q 值、函数及导数和执行 receipt；不展开保存逐节点宽特征与所有原子对。多深度统计约 1.76 MB。历史原始轨迹保留，新增测试原始坐标在校验、发布报告后退休。

## Analyst、Designer 与实施验证

三套版本化 Skills/workflow 实际调用 sub-agent。Analyst 读取字面指令、工具 receipt、全窗统计、祖先证据和反证；Designer 读取完整已导入 Analyst 响应，在公式注册表内提出有界单轴参数。调用输入、响应、参考、源码和输出哈希绑定，编译器不执行任意 LLM 程序。

真实 API 接口保留，与模拟后端共享消息、schema、证据及哈希验证。此次未调用外部 API。通用 Skills 不写死分子、区域编号或学习时间，也不包含 VPN/Git 环境指令。参数工程探索与新 LLM 设计明确分开记录。

先检查空规则与 R26 标量/梯度/实际窗口/终态数值一致，再检查新奖励梯度、真实 VJP、窗口轨迹变化、动态覆盖与窗口外零注入；通过后才评价亲和力。已有梯度非零不能证明新指令生效。模块输出反传 RMS 是奖励敏感度，不能证明 attention 或 affinity 的因果重要性。

## 实际可微奖励与控制

保留完整教师 T_k(t)、坐标匹配和评分先验。四个邻近教师通过固定 Hungarian 对应；先验 log π ∝ −MSE/4 + 2(score−mean score)。

`T′_k = T_k + sign·a_k·d_k`，`a_k=min(实际分支 RMS, .2 Å)`；只有观测到有效祖先对比与方向时 e_k=1。

`p_original=π_k(1−αe_k)`，`p_virtual=π_kαe_k`。R11 候选 α=.025；R26 不使用新虚拟分支质量。

`q_m=N⁻¹Σ_i w_mi||Y_i−T_mi||²`；`ρ(q)=δ²(√(1+q/δ²)−1)`；`R_t(Y)=τ logΣ_m p_m exp[−ρ(q_m)/τ]`，δ=1 Å，τ=.5 Å²，R11 默认 w=1。

`∇_(Y_i)R=−Σ_m posterior_m·w_mi(Y_i−T_mi)/(N√(1+q_m/δ²))`。真实 FLOWR 条件 VJP 给出 `g_X=J_Fᵀ·scale·∇_(Y_world)R`；F 是 native 坐标端点预测，Y_world=scale·F+COM，自条件与匹配冻结，没有亲和力 head 梯度。

当前引导按 η=.33 的预测 flow RMS 定标，逐原子 .15 Å、累计 RMS 6 Å，以及新增严重受体碰撞约束。全局奖励倍数可能被 RMS 归一化抵消，方向相对权重和实际剂量必须分别检查。主种子42、100步，学习窗口由输入绑定，当前0–.5，其余原生续推至1。

## 与 FLOWR 底层原理的相容性

实际 v=(F−X)/(1−t)，g=1/(t+.01)（t<.9），score=(tv−X)/(1−t+ε)，原生 Z=X+dt(v+g·score)+dt√(2g·noise)ξ。该实现噪声乘 dt，不能直接套用标准连续 SDE 分布保证。

在原生 Euler 提议上加入位移可按离散等式写成 u=Δ_guidance/dt，并非天然不兼容。最终碰撞约束还依赖原生 Z/噪声，不能据此声称标准连续适应漂移或分布保证。当前 VJP 是条件端点几何代理的梯度，不是完整未来价值伴随、下一步 Jacobian 或精确 Doob/FK 控制。引导复用一次正常 target 前向并增加一次奖励反传；原有 untarget 诊断前向保留，没有额外生产 target 前向。每个梯度批次还做一次有限差分检查，两个步幅的中心差分共四次额外 target 前向，单列于 preflight；终态 target/untarget 重评分也属于评价开销。不做粒子重采样。

实际 checkpoint 启用自条件，coord_scale=1.0。Steer 复制 current/prior/cond_batch，坐标控制保持原生条件更新；这是未完整迁移的联合状态信息，尚不能确定归因为性能差距原因。坐标进入 bond_refine，坐标引导造成化学图变化属于正常架构响应；无化学图相等 gate。

## 逐轮公开假设与效果

每轮只读公开 plan 与保留指标；负结果不改写假设。探索两批效应不等于独立确认。

|轮|相对 R26 的均值差|实施/探索通过|改动与原因|实际推理 commit|
|---|---:|---|---|---|
|1|—|None/False|{}；Original matched native/R26 baseline and frozen initial random streams.|9cc4ae3dc655|
|2|+0.000000|None/False|{}；Paired native/R26 control; new-interface empty rule and measured Jacobian diagnostics.|512a1d1df5f2|
|3|+0.009990|True/False|{'innovation.field_strength_A': 0.25}；Actual Designer-compiled first pilot; literal input, tools, formula and code bound.|84b0f9c66103|
|4|-0.000161|True/False|{'innovation': {'field_strength_A': 0.15}}；Bounded extrapolation from higher endpoint toward its coordinate contrast against local lower-scoring candidates; conditional empirical direction, not physical force or causal affinity gradient.|14a799947114|
|5|+0.003990|True/False|{'innovation': {'field_strength_A': 0.3}}；Bounded extrapolation from higher endpoint toward its coordinate contrast against local lower-scoring candidates; conditional empirical direction, not physical force or causal affinity gradient.|cb57e987cdb3|
|6|-0.005854|True/False|{'innovation': {'field_strength_A': 0.6}}；Bounded extrapolation from higher endpoint toward its coordinate contrast against local lower-scoring candidates; conditional empirical direction, not physical force or causal affinity gradient.|34e655ce1559|
|7|+0.007860|True/True|{'innovation': {'field_strength_A': 1.0}}；Bounded extrapolation from higher endpoint toward its coordinate contrast against local lower-scoring candidates; conditional empirical direction, not physical force or causal affinity gradient.|deeb62f71cd7|
|8|+0.000000|None/False|{}；Paired native/R26 control; new-interface empty rule and measured Jacobian diagnostics.|8e4ed2086d0d|
|9|-0.018806|True/False|{'branch_mixture.virtual_mass': 0.1}；Actual Designer conservative natural-branch joint-mode pilot; original modes retain at least ninety percent mass.|4944142593ad|
|10|-0.000195|True/False|{'branch_mixture.direction_sign': -1.0}；Matched reversed natural-branch direction at frozen pilot budget; counterevidence only.|d4120935d69e|
|11|+0.011002|True/True|{'branch_mixture.virtual_mass': 0.025}；Pilot9 actual field changed gradient/trajectory but lowered affinity in both paired batches; reduce only virtual-mode mass to test an over-strong/noisy observational coordinate direction without changing teachers or controller.|f9ba00461497|
|12|+0.001291|True/False|{'branch_mixture.virtual_mass': 0.05}；Pilot9 actual field changed gradient/trajectory but lowered affinity in both paired batches; reduce only virtual-mode mass to test an over-strong/noisy observational coordinate direction without changing teachers or controller.|30515fa05907|
|13|-0.000534|True/False|{'branch_mixture.region_weight_mix': 0.25}；Isolate confidence-qualified regional coordinate cost weights without teacher extrapolation.|01e986a2b967|
|14|+0.007130|True/False|{'flow_control.parallel_component_scale': 0.5}；Isolate gradient component parallel to predictive native flow; raw scalar derivative remains separately measured.|4470c39474b9|
|15|-0.025112|True/False|{'flow_control.parallel_component_scale': 0.0}；Isolate gradient component parallel to predictive native flow; raw scalar derivative remains separately measured.|8c932c013e61|
|16|-0.022918|True/False|{'flow_control.parallel_component_scale': 1.5}；Isolate gradient component parallel to predictive native flow; raw scalar derivative remains separately measured.|07f785b1a191|
|17|-0.016523|True/False|{'flow_control.parallel_component_scale': 2.0}；Isolate gradient component parallel to predictive native flow; raw scalar derivative remains separately measured.|e55a048938f2|
|18|-0.077750|True/False|{'flow_control.jacobian_gain_saturation': 0.45}；Restore measured real endpoint-to-state sensitivity to bounded coordinate dose, without extra model calls.|668e0feac096|
|19|-0.068722|True/False|{'flow_control.jacobian_gain_saturation': 0.15}；Restore measured real endpoint-to-state sensitivity to bounded coordinate dose, without extra model calls.|a6e2b8aed2e8|
|20|-0.129670|True/False|{'flow_control.jacobian_gain_saturation': 1.0}；Restore measured real endpoint-to-state sensitivity to bounded coordinate dose, without extra model calls.|bf827d72b5b7|
|21|-0.151796|True/False|{'native_rms_ratio': 0.14676547233500647}；Uniform lower dose control for gain-gate temporal allocation; no affinity-driven budget calibration. Frozen batch/time baseline proxy budget factor=0.4447438555606257; actual dose is separately compared.|05fcaef562a1|
|22|+0.009268|True/False|{'branch_mixture.virtual_mass': 0.05}；Actual Designer-compiled common-depth joint-regional field; small virtual mass with original teacher background retained.|920e746e2148|
|23|-0.012183|True/False|{'branch_mixture.direction_sign': -1.0}；Matched reversed common-depth regional direction at identical virtual mass, teacher support and native dose; counterevidence only.|f80eff812830|
|24|-0.001935|True/False|{'branch_mixture.virtual_mass': 0.15}；Change only virtual-mode mass of the same frozen common-depth field; test controlled amplitude sensitivity while retaining at least85percent original teacher weight.|36ec38b9b5de|
|25|—|None/False|{}；Original matched native/R26 baseline and frozen initial random streams.|079bb30348a5|
|26|+0.012175|True/True|{}；Independent confirmation of unchanged frozen reward and controller; no retuning on confirmation labels.|306b6b53022a|
|27|—|None/False|{}；Original matched native/R26 baseline and frozen initial random streams.|f4273eb250b4|
|28|+0.014306|True/True|{}；Independent confirmation of unchanged frozen reward and controller; no retuning on confirmation labels.|222bd369f028|
|29|—|None/False|{}；Original matched native/R26 baseline and frozen initial random streams.|d8914a89abb6|
|30|+0.036667|True/True|{}；Independent confirmation of unchanged frozen reward and controller; no retuning on confirmation labels.|a667cdb15096|

## 结论与下一步可检验特征

本轮证实新知识可以实际改变奖励梯度和生成轨迹；收益是否成立以六批冻结确认表为准。方向反证与剂量实验揭示了弱空间关联、归一化和个别样本的大效应问题。不能将“指令已实施”写成“指令有效”。

两步分支场的正反方向对照在 α=.1（第9/10轮），不是冻结 R11 的 α=.025；七区域场的等预算正反对照在 α=.05（第22/23轮），不是 R11 的同一方向场。冻结确认即使通过，也检验的是整个候选方案，尚不能把收益唯一归因于方向知识而排除混合平滑等解释。α=.025 的等预算反方向/随机方向是后续独立检验，本轮未追加超过30轮的实验。

尚值得独立注册并检验的特征：期望选择与复制噪声分离；局部刚体姿态与内部形变；条件记忆与预测一致性；教师模式切换/梯度冲突；跨区域距离与角度 motif；被观察终态高分路径的祖先信用；真实模块因果干预。不能填补淘汰分支未知未来，也不能把噪声点云伪称中间物理能量。

这些建议未被追记为早先 Agent 输入。下一次探索应先补独立紧凑工具与来源 receipt，再实际调用两角色，检查指令响应，最后逐轴配对验证。

## 代码、报告与依据

数据工具、角色/API 和生成命令见同目录 `reproduction.zh-CN.md`；公式与限制见 `scientific_design.zh-CN.md`、`remaining_features.zh-CN.md`；独立校验见 `confirmation_integrity.json`；全部配置、哈希和结果见各 round 的 plan/retention。

[FLOWR 作者代码](https://github.com/jule-c/flowr_root)、[FK steering](https://arxiv.org/abs/2501.06848)、[Flow guidance](https://proceedings.mlr.press/v267/feng25s.html)、[Optimal-control flow matching](https://arxiv.org/html/2410.18070v3)、[FK-Flow](https://arxiv.org/html/2509.01543v1)、[Price 分解](https://pmc.ncbi.nlm.nih.gov/articles/PMC4415573/)提供机制与分析依据；本项目不宣称完整实现这些方法或其分布保证。
