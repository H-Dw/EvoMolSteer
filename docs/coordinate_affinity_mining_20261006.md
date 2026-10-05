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
python scripts/mine_coordinate_advantages.py --dataset data/optimized/main1000_w050/analysis_inputs_v2 --campaign main1000_w050 --analysis results/continuous_single_v3 --output results/coordinate_affinity_v2
```

输出：`batch_coordinate_statistics.parquet` 为无损 float64/zstd 的批次统计；
`whole_window_evidence.csv`、`continuous_functions.json`、`effect_functions.json`、
`lineage_diagnostics.csv`、catalog 和带源文件 SHA256 的 manifest。
不保存逐候选特征、逐边明细、worker 分片或坐标复制件。
当前 20 批共 897,600 条批次/节点/特征统计，主体约 70 MB；相比此前约 730 MB
特征缓存更小，但不能把两种不同统计范围的大小当作严格等价压缩率。
保留原始无损轨迹、输入配置和 commit，才能重建这些统计。
v2 修正 feature 特定缺失掩码：缺少 NOS 不会排除无关 all-atom 回归的候选。
897,600 行重算确认，选择、富集和未调整相关等 13 个字段逐值一致。

## Analyst / Designer 接口

`scripts/coordinate_agents.py` 提供 export/import/api/compile 四种操作。
与旧的 stage/endpoint 契约分离，使用严格 `coordinate-1.0` schema。
导出只包含 discovery 的窗口效应、反证、区域观测、函数和谱系诊断；
典型请求约 389 KB，不向 LLM 输入庞大的逐原子/逐种子坐标缓存。
导入核查 bundle SHA、证据 ID、feature、region、窗口及有限的控制参数。
Designer 只能选择白名单的 spread_upper 或 coordinate_mixture，不能执行任意生成代码。
compile 从原始 discovery 坐标提取参考，输出可执行 program/reference，固定 seed42 与约束。
API 复用现有 OpenAI-compatible HTTP 传输；需要显式设置 `EVOMOLSTEER_BASE_URL`、
`EVOMOLSTEER_MODEL`、`EVOMOLSTEER_API_KEY`。本轮使用现有 subagent 模拟，不声称调用外部 API。

```powershell
python scripts/coordinate_agents.py --mining results/coordinate_affinity_v2 --action export --role Analyst
python scripts/coordinate_agents.py --mining results/coordinate_affinity_v2 --action import --request results/coordinate_affinity_v2/agents/Analyst.coordinate.request.json --response <Analyst响应.json>
python scripts/coordinate_agents.py --mining results/coordinate_affinity_v2 --action export --role Designer
python scripts/coordinate_agents.py --mining results/coordinate_affinity_v2 --action compile --dataset data/optimized/main1000_w050/analysis_inputs_v2 --campaign main1000_w050 --output <新的冻结配置目录> --round 1
```

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

## 预测终点归属与实际运动的分离

第一轮按当前坐标划区域时，全部平方注入落在所定义的5 Å核心之外。
早期噪声的空间位置不足以代表未来结合区域。因此新增显式
`--spatial-anchor endpoint --control-representation proposal --regions <残基列表>`。
区域权重由同一步预测终点确定，标签也是预测终点的条件NOS标签；观测仍是实际坐标，
不能把它称作早期物理接触。预测终点只确定槽位权重，求导时固定；每步重新预测。

```powershell
.venv/Scripts/python.exe scripts/mine_coordinate_advantages.py --dataset data/optimized/main1000_w050/analysis_inputs_v2 --campaign main1000_w050 --analysis results/selection_window_v2 --output results/coordinate_endpoint_anchor_v2 --spatial-anchor endpoint --control-representation proposal --regions ck2:A:ASN117,ck2:A:VAL116,ck2:A:HIS115,ck2:A:ASN118,ck2:A:ILE95,ck2:A:LYS68,ck2:A:GLY46,ck2:A:ASP175
```

这一组预先指定8区域×2通道×15观测=240特征，包含 current/proposal 的xyz与spread、
endpoint residual drift、实测native步速度/RMS。统一分析全部51次选择事件，绝不切0.1区间。
保留float64批次统计、q/CI、覆盖度、谱系诊断、整体Legendre拟合及解析时间导数。
约26 MB（含LLM证据缓存），无逐节点/逐边明细缓存。当前40区域分析为约79 MB。

较小的多重检验集合与原40区域分析不是同一假设族，q不能直接比较。
新discovery adjusted lag有16/240项q<.05，但全部64项proposal选择偏移无q<.05。
ASN117/VAL116 NOS proposal_x的偏相关约.0330/.0318；ASN117 NOS proposal_spread约-.0452。
这些提示可测试条件区域轨迹模仿；它们不支持直接写成固定+x或无限压缩奖励。
仍有11/14批在窗口末端单根，不能把后代当独立样本。

第三轮将两个NOS区域的proposal xyz/spread拼为8维，联合协方差与混合参考处理重复观测；
η=.05为保守实验剂量。参考在score t上建立，其对应实际proposal状态为t+dt。
所以最后控制score .49产生x_.5；score .5的proposal x_.51仅属记录，不能再控制。
官方形状评价独立读取actual x_.5，不使用reward值替代形状改善。
Analyst/Designer均由已授权subagent模拟，严格schema、引用和范围验证后编译配置；
保存证据/决策/公式摘要与校验值，不记录内部思维链。

新增 `--feature-family transport` 可单独提取5项诊断，避免复制240项几何统计：
剩余位移RMS、native位移与剩余位移的加权余弦、有效槽位数、endpoint核心权重、
当前坐标核心权重。8区域共80项，输出约7.4 MB。核心半径5 Å用于诊断，未从q值优化。
ASN117的endpoint核心权重始终为0；VAL116 NOS的等批次窗口均值为0.7242，
effective slots约1.90。它表征预计参与该区域的少量槽位，不代表早期已经成键或接触。

`scripts/analyze_coordinate_influence.py --mining <目录> --output <目录>`
将原完整窗口统计分解为节点的求积贡献，只保存每个特征的首节点/主导节点贡献与有效节点数，
不切子区间、不增加逐粒子缓存。ASN117/VAL116 NOS native_y首节点贡献分别
-.05133/-.04693 A/time，完整窗口效应-.05052/-.04677；其余节点有正负抵消。
首节点占绝对贡献总和约75.3%/74.2%。由此不能把全窗显著性解释成全程持续方向优势。

原生线性endpoint参数化漂移v=(y-x)/(1-t)，实测SDE步还包含g_t score与噪声，
g_t=1/(t+.01)。在t=0、dt=.01时，g_t score几乎抵消初态；native RMS可能比纯flow位移大两个数量级。
新增可选 `dose_reference=predictive_flow` 使用dt*v的RMS标定外部剂量，保持原生SDE不变。
默认仍为observed_native以保留历史实验；线性日程不匹配时该新选项拒绝执行。
记录实际native、预测flow和校准RMS，并保存原生积分器参数，避免把标签中的native一词误读为pure flow。

参考构建还补充紧凑的background联合均值/协方差、selected原始特征值、概率ESS/KL及
标准化均值差。这些是未来选择/背景对照所需的可观测性，不自动变成新奖励，也不回写已冻结第三轮参考。
