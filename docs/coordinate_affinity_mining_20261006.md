# 坐标优势挖掘与重新计数的 seed42 实验

2026-10-06 开始新 campaign，编号从 1 开始，上限 **30 个已启动轮次**。
历史八轮和五轮的编号、配置及结果不改写。30 是上限，不是必须完成的数量。
每轮必须先保存上轮报告并完成远端清理审计；原始 Steer 和 checkpoint 受保护。

## 本次补齐的分析

入口 `scripts/mine_coordinate_advantages.py` 接收 dataset、campaign、已有 analysis
元数据和独立 output。实际 `resampled` 决定时间窗口；本数据为 0–0.5、51 节点。
读取 current、predicted、proposal 三种坐标，但不展开保存逐候选特征缓存。
40 个受体残基 × all/NOS 两通道 × 11 个观测，共 **880 个特征**。
三维坐标固定在对齐受体坐标系；单位 Å，速度单位 Å/生成时间。

对区域 r，令 d_ir 为原子到残基重原子的最短距离，
`w_ir ∝ exp(-d_ir²/(2σ²))`，σ=4 Å，在符合条件的原子槽位中归一化。
观测为加权质心相对残基中心的 xyz、加权散布、三轴终点漂移、三轴实测速度和实测 RMS 速度。
终点漂移为 `(predicted-current)/(1-t)`；原始积分器连续模式使用此项，SDE
还加入 score/noise，corrector 也可能改变实测位移，二者不能混同。
实测速度为 `(proposal-current)/dt`，只匹配同一候选的原子槽位。
NOS 使用预测终点类别给当前槽位建立条件掩码，不解释成早期噪声的真实化学。

分别保存选择均值偏移、上下尾质量富集、窗口末端后代加权偏移、与匹配 native
的差异、同期 affinity 相关及调整全局质心/散布/NOS 数后的相关。
新增下一步 gain 分析：同一个被保留父节点的子代评分先取均值，再减该父节点当前评分；
每个父节点只计一次，并进一步调整其起始评分。该分析仍受存活条件和 oracle 共享影响，
不能证明因果。0.5 没有窗口内后继，lag 最后一个节点为 0.49；不读取 0.51 来补齐。

所有效应在完整窗口积分，以批次为独立单位计算 t 区间和按 split/metric 的 BH q；
lag 数值缺失时明确给出可用时间覆盖，不填零或外推。
绝对选中均值以及选择/留存/调整后同期效应另有全窗口 Legendre 函数，
0–5 阶采用 leave-one-batch-out 和 one-SE 规则。导数是时间导数，不是奖励空间梯度。
高阶拟合可能违背速度非负等约束，所以推理使用实测同刻参考，不直接编译这些多项式。

```powershell
$env:PYTHONPATH='src'
python scripts/mine_coordinate_advantages.py --dataset data/optimized/main1000_w050/analysis_inputs_v2 --campaign main1000_w050 --analysis results/continuous_single_v3 --output results/coordinate_affinity_v1
```

输出：`batch_coordinate_statistics.parquet` 为无损 float64/zstd 的批次统计；
`whole_window_evidence.csv`、`continuous_functions.json`、`effect_functions.json`、
`lineage_diagnostics.csv`、catalog 和带源文件 SHA256 的 manifest。
不保存逐候选特征、逐边明细、worker 分片或坐标复制件。
当前 20 批共 897,600 条批次/节点/特征统计，主体约 70 MB；相比此前约 730 MB
特征缓存更小，但不能把两种不同统计范围的大小当作严格等价压缩率。
保留原始无损轨迹、输入配置和 commit，才能重建这些统计。

## 审阅结果与可证伪设计

使用现有 subagent 模拟 Analyst/Designer，完整记录位于
`docs/experiments/ck2_coordinate_seed42_20261006/`，设计不访问 14–19 的结果。
all-spread 负选择偏移在 39/40 区域显著，上尾抑制在 40/40 区域显著。
调整后 lag 的 880 项均未通过 q<0.05。ASN117/VAL116 暂无独有的坐标优势证据；
它们是预先给定的代表性位置，不能称为两个已确认的亲和力优化机制。
0.5 discovery 平均仅剩 1.214 个根谱系，11/14 批只有一个根。
物理区域能量尚不能在噪声坐标和不确定键图上可靠计算，不伪造能量证据。

`scripts/analyze_regional_specificity.py --mining <分析目录> --output <独立目录>`
进一步将每个区域的选择效应减去同刻跨区域共同效应；检验完整窗口的区域残差，
同时保存批次残差积分、端点变化率、共同时间曲线及其导数、PCA 基底和批次时间投影/导数。
这比较的是选择效应，不能通过相减 Pearson 系数冒充个体坐标残差的相关性。
此对照显示 all-spread 选择效应变化能量约 80.10% 由共同方向解释。
NOS 实测 y 速度在 ASN117/VAL116 仍有额外负偏移，区域残差 q≈0.0153/0.0133；
这是局部运动的观察性筛选证据，尚不是后续 affinity 改善证据。

第一轮采用 ASN117 all 通道的散布上尾约束；它限制局部加权散布，
不指定质心方向，不要求原子向残基靠近，也不奖励无限压缩。
`U_r(t)` 为原始 discovery 候选等批次、批内选择概率加权的 75% 分位；
`s_r(t)` 为同刻未加权候选的批内散布标准差，实验性 SD 下限 0.15 Å。

`v=max((spread_r(x)-U_r(t))/s_r(t),0)`；`R=-(sqrt(1+v²)-1)`。
直接求实际 proposal 坐标的梯度。FLOWR 负责原生预测和更新；此坐标奖励本身不需要
FLOWR endpoint Jacobian。NOS、键型和电荷均不使用伪造的连续导数。
引导状态时间 s 与实测参考 s 对齐；范围从 reference 动态读入。
本配置控制 0→0.5 的 50 次更新，0.5→1 保持原生推理，无新增粒子选择。

另一架构在代码中保留为显式模仿候选：对当前 regional xyz/spread 使用时间变化混合，
每个 discovery 批次的原始加权候选给出均值/协方差，协方差 0.1 对角收缩与 0.15 Å
特征 SD 下限。它不是已恢复的真实分子模态，也不是 selection/background 密度比。
`R=τ log mean_m exp[-δ²(sqrt(1+q_m/(Dδ²))-1)/τ]`，τ=0.25、δ=1 为实验假设。
不能因为此代理改善就宣布整体 Steer 或 affinity 改善。

两架构均采用独立的外部剂量：η=0.2 倍原生 RMS，再乘有界残差 gate；
每原子每步最多 0.025 Å，累计 RMS 最多 1.25 Å，成对距离变化上限 0.06 Å，
新严重受体碰撞阈值 0.8 Å，最多 7 次回退。实际剂量、拒绝和投影必须报告。
代码还提供全局 uniform-weight spread 对照，排查区域权重是否有额外价值。
可选 `preserve_native_rigid_pose` 将梯度正交投影到全分子平移/瞬时转动的补空间，
用以排查通过移动整分子改变 Gaussian 权重、而没有修复形状的奖励捷径。
该投影是单独的执行假设，不能追溯性地写成第一轮已采用的条件。
数值梯度检查及零剂量与 native 的逐字节等价是工程检查。
实际窗口末端双向形状匹配、终态化学有效率、PB 子集、MMFF 松弛代理、
同态 affinity head、多样性为分开的科学指标。共享 head 不是独立实测亲和力。

生成接口：`scripts/generate_coordinate_flowr.py --flowr-root <FLOWR目录>`；
可复用现有生成 CLI 的 program/reference/checkpoint、固定 seed42、完整100步及指定批次参数。
源代码必须本地修改、commit、VPN push，远端 pull 同一 commit 后运行。
推理源只导出/无损记录；科学分析在本地完成。每轮结论以报告保存，随后删除生成坐标、
结构、临时分析和传输归档，保留冻结配置、奖励依据、失败行和 SHA256。
