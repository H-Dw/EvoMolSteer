# EvoMolSteer 分析范围、时间语义与分支数据核查

核查日期：2026-10-04。依据：当前分析源码、主实验 60 批的 ingest_manifest、实际 NPZ/PT/JSON、已生成统计表和 LLM 证据包。本文记录审计结果和建议，没有修改分析代码、重算结果或启动生成。

## 1. 结论及论文边界

针对“从重采样中学习局部选择规则”的目标，应把实际重采样事件作为主要发现集；窗口之后的轨迹和 final 用于结局回溯、优势保持/消失及失败分析。后期状态具有分析价值，但不能直接解释成后期发生了选择。

当前主实验 `main1000_w050` 的 joint、single 各 20 批，均在评分时刻 0、0.01、…、0.50 重采样，共 51 次；unguided 的 20 批没有重采样。各批 50 个粒子，100 个积分步。这里的 [0,0.5] 是当前实验配置，不能据此声称 FLOWR.ROOT 算法固定只使用前半程，或 CK2α/CLK3 原论文明确采用这一窗口。

之前对正文、四份补充材料和归档代码的核查没有恢复出 CK2α/CLK3 的精确运行窗口。详见 [原时间表核查](../../output/FLOWR_ROOT_CK2_CLK3_Resampling_Schedule_Audit_2026-10-03.md)（工作区路径实际为 `D:/llm_controlled_generation/output/FLOWR_ROOT_CK2_CLK3_Resampling_Schedule_Audit_2026-10-03.md`）。论文原文：[Nature Communications](https://www.nature.com/articles/s41467-026-74130-9)。重采样是基于评分的有放回随机复制，不使用奖励梯度；论文称其为 importance-sampling guidance。

## 2. 当前工程实际做了什么

数据流为：原始完整轨迹 → 谱系一致性校验与坐标恢复 → 三种表示的几何特征 → 分批统计 → discovery 证据包 → Analyst → Designer。

- 三种表示为 `current_state`、`predicted_endpoint`、`proposal_state`。总特征表有 900,000 行；默认统计只读 `predicted_endpoint`。
- 发现集为批次 0–13，验证集 14–16，留出集 17–19。默认 LLM 证据包包含发现集三组、全部 100 个评分节点，共 210,000 个节点表示。这里不是 210,000 个独立样本。
- 在两个对齐口袋的晶体配体 8 Å 邻域定义 84 个残基区域；每区计算归一化 soft-min 距离、N/O/S soft-min 距离、平滑接触比例；另有回转半径、最小原子间距、过近原子对比例，共 255 个特征。
- 统计推断以独立批次为单位，采用批次符号翻转检验、bootstrap 和每个结果文件内部 BH 校正。粒子、克隆和连续帧不作为独立实验重复。
- 成功标签是本工程的探索性定义：最终分子可构建、PoseBusters 通过、on-rescore ≥ 7 且 off-rescore ≤ 6。它不是论文公开的实验活性判定标准。灭绝分支记为结局删失，不能记作最终失败。

| 方法 | 当前计算 | 当前时间范围 |
|---|---|---|
| 富集 | 以 discovery unguided 同阶段第 75 百分位定义“高特征”，比较高特征的 log-odds ratio | 选中/淘汰比较只读 resampled；成功后代/无成功后代比较读全程 |
| 差异 | 同时刻同父节点的选中/淘汰同胞；成功后代关联；匹配初始批次的策略差异 | 同胞选择比较只读 resampled，其他读全程 |
| PCA | 在 discovery unguided 全程拟合标准化和轴，再固定投影；计算质心及父子边变化率 | 全程；后期变异也会影响 PCA 轴 |
| 趋势 | 逐步分离期望选择、实际选择、后续生成及总体变化；另算阶段变化率 | 全程；未重采样步骤实际选择项为 0 |
| 终态 | 成功与化学失败、几何失败、目标未达标分别比较；最后预测至 final 的几何变化 | 终态；用于解释早期分支的结局 |
| LLM 证据 | 多方法候选证据、弱同胞对照、经验目标区间 | 当前没有强制把候选规则限制在重采样窗口 |

源码入口：`configs/default.json`；`src/evomolsteer/{ingest,geometry,enrichment,differential,pca,trends,terminal,evidence,statistics}.py`。

“富集”是几何特征分布的富集，不是 GO/通路富集；PCA 的高载荷表示变异来源，不直接表示有利方向。预测终点随评分时刻变化的导数，也不是实际原子的物理运动速度。

## 3. 七个区间的来源与问题

`configs/default.json` 显式给出 `[0,.1,.2,.3,.4,.5,.75,1.00001]`。这些是分析端人工指定的汇总分箱，不是论文节点、保存节点或算法必需参数。其布局可用于前期较细、后期较粗的轨迹概览，但没有证据说明它是发现局部控制规则的最优分箱。

`stage_for` 在分箱时把评分时刻舍入到六位小数，采用左闭右开；原始浮点时间仍用于变化率。实际结果如下，每批每个时刻有 50 个粒子：

| 分箱 | 实际评分时刻 | 保存帧数 | joint/single 重采样次数 |
|---|---|---:|---:|
| [0,.1) | 0.00–0.09 | 10 | 10 |
| [.1,.2) | 0.10–0.19 | 10 | 10 |
| [.2,.3) | 0.20–0.29 | 10 | 10 |
| [.3,.4) | 0.30–0.39 | 10 | 10 |
| [.4,.5) | 0.40–0.49 | 10 | 10 |
| [.5,.75) | 0.50–0.74 | 25 | 1，仅 0.50 |
| [.75,1] | 0.75–0.99 | 25 | 0 |

`final` 单独保存和分析；上表最后一个分箱不会自动把 final 混入中间帧统计。NPZ 的最后 proposal 到达 t=1，最后评分时刻仍为 t=0.99。

因此，不能通过保留前五箱或 `t < 0.5` 来实现选择窗口分析，否则会丢掉 t=0.50。对 joint/single 应以保存的 `resampled` 为准。unguided 作为匹配背景时应选择对应评分步，不能用它自己的 `resampled=True` 过滤，否则整个对照组都会被删除。

本次确认了四项需要调整的边界：

1. **选择对比的阶段时间有具体偏差。** `enrichment.py:33`、`differential.py:20` 对选择子集算效应，但时间使用整个阶段的 `g.geometry_time.mean()`。已生成表中 stage 5 的选择效应只来自 t=0.50，却记为 time≈0.62。与上一箱均值 0.445 相比，当前变化率分母是 0.175；按实际参与选择的时刻应为 0.055。该项“进入最后选择阶段的变化率”不能直接当作事件变化率，应修正时间口径后重算。这不是说全部统计结果都失效。
2. **阶段选择速率和单次事件强度被混在同一分箱。** `trends.py:41` 在 stage 5 平均 25 步，其中只有一次重采样，其余选择项为 0。结果可以解释为整个阶段的平均选择作用，不能解释为 t=0.50 那一次选择的强度。
3. **后期证据能够成为规则候选。** 当前证据包 134 条；stage 5 有 14 条、stage 6 有 4 条；60 个经验目标中 stage 5 有 7 个、stage 6 有 2 个。stage 5 同时包含真实的 t=0.50 选择证据和后续关联证据，不能把整箱都标为后期或选择期。应按事件/证据类型拆开。
4. **奖励时间窗不是硬截断。** 当前 Designer 只选择了 CLK3 LEU162、标称 t=0.4–0.5 的规则，但 `reward.py` 使用 sigmoid 时间门，窗口外仍有尾部。若后续要求严格在选择窗口内施加梯度，需要额外的明确时间约束。现有程序尚未用于实时无重采样生成。

## 4. 建议的分析范围

保留原始全程数据，把证据用途分开：

- `selection_evidence`：joint/single 的 51 个真实重采样事件及同时间 unguided 背景。逐事件分析几何特征、双目标权重、繁殖次数和选择位移。粗分箱只用于展示与稳健性检验，最后一次事件必须保留。
- `continuation_evidence`：最后选择后的路径，用于追踪早期形成的特征能否维持、何时丢失、何时出现冲突或化学失败。不能把后期关联自动解释为一个新的施加梯度时间窗。
- `terminal_outcome`：用 final 的多目标评分和质量结果回溯早期节点，区分“高分但无效”“几何失败”“有效但不够选择性”“达到探索性目标”。

若要产生“选择窗口专用”的规则包，应在该窗口重新定义背景、拟合 PCA、计算比较和变化率、执行多重检验校正与证据排序。不能只删掉最终 CSV 的后两箱，再沿用原先全程分析的 PCA 轴和 q 值。全程 PCA 仍可作为独立的描述性视图保存。

时间筛选必须区分评分时间与积分后的状态时间：t=0.50 的评分用于选择已推进到约 0.51 的 proposal。选择事件属于评分窗口 [0,0.5]，并不要求被复制状态的时间也 ≤0.5。

## 5. 原始分支是怎样保存的

以 `data/raw/ck2_clk3_lineage_20261003/results/main1000_w050/joint/batch_000/` 为例，一个批次保存整个人口的时间张量和父子索引，不是每个 seed 一份独立文件。

令 T=100、B=50、N=25。实际 `trajectory.npz` 有 28 个数组：

| 字段 | 形状 / dtype | 含义 |
|---|---|---|
| current_coords / predicted_coords / proposal_coords | [T,B,N,3] / float32 | 当前输入 x_t；从 x_t 预测的终点；积分后尚未选择的 proposal |
| current_atomics / predicted_atomics / proposal_atomics | [T,B,N] / uint8 | 原子类别词表索引 |
| current_bonds / predicted_bonds / proposal_bonds | [T,B,N,N] / uint8 | 硬键类型索引 |
| current_charges / predicted_charges / proposal_charges | [T,B,N] / uint8 | 电荷词表索引，不能当作实际电荷数值 |
| predicted_atomics_probs_f16 | [T,B,N,15] / float16 | 预测原子类别概率 |
| predicted_charges_probs_f16 | [T,B,N,8] / float16 | 预测电荷类别概率 |
| mask | [T,B,N] / uint8 | 有效原子槽位 |
| pic50_on / pic50_off | [T,B] / float32 | 本次评分时刻的 CK2α / CLK3 预测分数 |
| weight_on / weight_off / selection_probability | [T,B] / float32 | 单目标 softmax 和组合权重；窗口外只是记录的候选权重，没有执行选择 |
| selected_indices | [T,B] / int64 | 新槽位 j 来自旧候选 i 的映射 |
| offspring_count | [T,B] / int64 | 各候选被复制次数，等于 selected_indices 的 bincount |
| root_slot / parent_slot | [T,B] / int64 | 初始根槽位、上一评分步的父槽位 |
| score_time | [T,3] / float32 | 模型三个时间通道；本实验同步 |
| step_size / state_time | [T] / float32 | 积分步长、积分后的 proposal 时间 |
| resampled | [T] / uint8 | 是否真正执行了重采样 |

坐标换算为 `x_world = x_model * coord_scale + target_COM`，映射保存在 campaign 的 `config.json`、`frame_batch_000.json` 和初态文件。final_prediction 的坐标已经恢复到世界坐标。噪声态的 25 个槽位不能解释为已经确定的 25 个化学原子。

一个积分事件的语义是：

```text
current x_t
  -> 模型预测 endpoint 与 on/off 分数
  -> 积分得到 proposal x_(t+dt)
  -> 窗口内按权重有放回复制 proposal
  -> 下一评分步的 current
```

`selected_indices[t,j]=i` 表示：

```text
current_coords[t+1,j] == proposal_coords[t,i]
parent_slot[t+1,j] == i
```

该关系已在坐标、原子、键和电荷通道校验。因此 slot 不是跨时间永久身份；路径必须通过索引连接。特征表使用 `campaign:arm_bBBB_tTTT_sSSS` 作为节点 ID，representation 另列保存。

真实例子：joint/batch_000、step=30，评分时间约 0.30，proposal 时间约 0.31。

| 候选槽位 | root | CK2α pIC50 | CLK3 pIC50 | 概率 | 子代数 | 下一步槽位 |
|---|---:|---:|---:|---:|---:|---|
| 4 | 6 | 8.9061 | 6.4498 | 0.029013 | 3 | 8、31、41 |
| 2 | 6 | 7.6470 | 6.5407 | 0.015295 | 0 | 无 |

两个候选共享初始 root，但父槽位分别为 40、29，不能称作同一直接父节点的同胞。未获子代的候选仍有完整当步三种状态及评分；其未被执行的未来路径不存在于主轨迹中。

其他文件：

| 文件 | 保存的信息 |
|---|---|
| initial_state.pt.gz | 初始先验、两个口袋、COM、随机状态 |
| restart_step_*.pt.gz | 当前状态、self-conditioning、先验、时间、root、CPU/GPU/NumPy/Python RNG |
| endpoint_step_*.pt.gz | 锚点的完整 float32 终点预测，包括 soft-bond 概率 |
| final_prediction.pt.gz | 每个终态槽位的完整预测张量，保留无法构建分子的槽位 |
| final_records.json | seed、slot、root、最终父节点、两套评分、构建成功与失败原因、可用时的 SMILES |
| quality_records.json / posebusters_checks.csv | 可评估终态的质量、具体失败项、QED、SA、MMFF 应变与收敛信息 |
| events.json | 是否重采样、ESS、淘汰数量、剩余根数量 |
| molecules_*.sdf | 能构建或解码的分子；不能替代失败槽位张量 |

完整 restart/endpoint 锚点为 step 0、10、20、30、40、50、75、99。每一步都有 NPZ 的三种坐标和硬类别，不是只保存这些锚点；但每一步没有保存完整 float32 soft-bond 概率，不能声称任意时刻都可恢复完整模型软状态。

campaign 的 `analysis/` 另有 `*_nodes.csv.gz`、`*_edges.csv.gz`、`*_lineage.graphml.gz` 和离线谱系查看器。EvoMolSteer 的分析表另存为 features/edges/terminals Parquet；新增的 HDF5 特征包是无损索引/去重存储，数据语义不变，见 [存储格式](storage_format.md)。完整原始 NPZ 仍保留，HDF5 轨迹转换只做了试验。

## 6. 值得学习的内容与不能推出的结论

优先学习以下对应关系：

1. **区域—阶段—方向。** 在同一评分步/批次中，哪些残基附近的距离、接触或局部形状与高权重稳定关联；方向是靠近、远离还是保持某个区间。全局 PCA 可辅助发现协同变化，不直接提供奖励方向。
2. **选择预期与随机实现分开。** 当前 joint 权重为 `0.5*(softmax(on)+softmax(-off))`，不是 `softmax(on-off)`。相同绝对分数在不同群体中可能有不同概率。分析应同时使用双目标分数、权重和子代数；被淘汰不等于低质量。
3. **选择与继续生成分开。** 对几何特征 φ，期望选择位移为 `sum_i (p_i-1/B)*φ_i`，实际位移为 `sum_i (n_i/B-1/B)*φ_i`。除以事件间隔可得到约定的每单位生成时间变化率。用 proposal 测量复制造成的实际几何替换；用 current/endpoint 测量选择所依据状态的关联，两者不可混称。
4. **优势形成速度和稳定性。** 沿真实父子边计算 `(φ_child-φ_parent)/Δt`，观察何时出现、何时被选择放大、停止选择后是否保持。不能把第 i 行直接连到下一帧第 i 行。
5. **终点质量回溯。** 同一早期区域特征是否最终带来 CK2α 增强、CLK3 降低且结构有效；把化学失败、几何失败、目标未达标、未知结局分开。
6. **可编辑性及副作用。** 原子类型/电荷概率、键类型、局部距离与碰撞可用于定义解析奖励的适用条件和约束；仅从距离不能证明氢键或 π 相互作用。

需要保留三种不同的“负例”：当步未选中但未来未知；存活到终态却失败；额外反事实续跑中观察到失败。当前少量反事实续跑从保存的评分前状态配以新的未来噪声出发，不等同于所有被丢弃 proposal 的真实终局。

同一父节点复制出的候选在 current/endpoint 上可能完全一致，之后 proposal 才发生分化，因此近零同胞差异本身是信息，不能强求 LLM 从中解释一个几何优势。早期 endpoint 是模型预测，不是已经形成的真实分子；LLM 应读取带单位、时间、表示、比较方式和不确定性的特征证据，再提出受约束规则。

这些数据能够支持解析奖励假说及对已观测选择偏好的近似；不能仅凭幸存者相关性证明某个坐标梯度必然重现最佳路径。尤其是后代成功标签受存活者偏差影响，灭绝路径结局未知。下一阶段应冻结早期规则，在无重采样、匹配初始先验的验证生成中检查最终收益及结构质量。
