# 选择窗口分析的输出集成与结构输入优化

本次已将独立副本中的输出策略接入共享源码，并扩展了可以直接供分析使用的结构输入格式。基准是 **selection_window_v2** 的同一批数据、同一统计定义；不使用旧全程分析的节省比例。

## 固定不变的科学范围

- joint/single 使用实际 `resampled` 标记：评分时间 0、0.01、…、0.50，共 51 次选择；unguided 取同一时间表作为背景。
- 两种表示、255 个特征、306,000 行候选特征均不减少。t=0.50 的 proposal 约在 0.51，仍归入最后一次选择事件。
- 20 个独立先验批次 × 3 个策略；discovery 为 0–13，validation 为 14–16，heldout 为 17–19。LLM 仅接收 discovery 证据。
- 事件内比较、批次/阶段均匀权重、缺失值覆盖阈值、bootstrap、符号翻转、BH 检验族、PCA 拟合背景和原型分位数的定义不变。
- 不引入窗口后的分析、终态成功标签、后代关联或反事实续跑；不删除非显著、负向或同胞零效应。当前保留的 5,100 条同胞阶段效应中，2,578 条绝对值不超过 1e-10，仍是对照证据。

## 1. 输出策略已完成集成

此前共享源码虽然声明 `profile=compact`，四个统计模块仍固定写出 CSV、逐事件表和逐边明细。现在这些开关真正生效：

| 产物 | compact 默认行为 |
|---|---|
| 全部批次阶段效应、批次阶段变化率 | 始终保存，8 张无压缩 Parquet，浮点字段为 float64 |
| 跨批次效应、CI、p/q、支持数与方向一致率 | 始终保存；原 CSV 内容不变 |
| 富集阈值、PCA 基底/载荷/解释方差 | 始终保存 |
| 逐事件效应与相邻事件导数 | 全部计算；`write_step_details=true` 时保存明细 |
| PCA 逐节点投影、逐父子边变化率 | 全部计算；`write_node_details=true` 时保存明细 |
| 计算一致性 | 始终保存事件诊断及父子数组的分区校验值、行数、有限值计数 |
| LLM 目标特征的事件曲线 | 始终保存；无事件明细时从候选特征按需重算 |
| 特征缓存、工作分片 | 合并验证后清理本轮分片；归档态回收缓存，工作态保留一份 |

父子变化率按批计算与校验，compact 不再把所有批次的巨大明细表拼接后保存。浮点特征保持双精度；存储策略不再隐式增加 `current_state` 或改变节点范围。输入读取也只加载实际消费的 24 个字段，其余字段仍保存在结构包中。

关闭逐事件/逐边明细意味着这些明细不再常驻磁盘；诊断校验值和阶段统计不能反向恢复它们。恢复需使用结构轨迹、谱系、配置、PCA 基底和对应版本代码。完整覆盖率、概率质量和理论抽样标准差仍参与计算，逐事件数值可通过诊断模式重新输出。

以十进制 MB（1 MB = 1,000,000 字节）计算：

| 相同 51 事件范围 | 集成前 | 集成后 | 减少 |
|---|---:|---:|---:|
| `discovery/` 正式结果 | 624.34 MB | 55.45 MB | **91.12%** |

这项比较不包含特征缓存、原始结构或测试目录，也不混入从 100 步缩小到 51 个选择事件的收益。可重建的单一特征缓存仍为 675.83 MB；不是通过删特征或改浮点精度缩小它。结果目录归档态约 58 MB，工作态在其上增加这一份缓存。精确字节数见 `results/selection_window_v2_compact_v3/verification/storage_measurements.json`。

## 2. 结构输入采用先去重、后无损压缩

新输入目录：`data/optimized/main1000_w050/analysis_inputs_v2/`。

这里没有裁去任何 trajectory 字段、生成步骤或未被选中的候选。所有 60 个轨迹仍包含原来的 **100 步、28 个字段**，分析器只消费其中的实际选择事件。

结构归一化分别验证以下关系：

1. 对每个状态字段逐位检验 `current[t+1] == proposal[t, selected_indices[t]]`。成立时只存初始 current，通过 proposal 和索引恢复；不成立时保留独立数组。
2. 仅对逐位对称的硬键矩阵保存上三角和对角线，不强制对称化。
3. 完全相同的状态行可共享字典；常量和正零稀疏数组使用精确表示。
4. 整数索引仅缩小物理存储类型，读取时恢复原 dtype。浮点数组不降精度，正负零、NaN 有效载荷和非零小概率均保留。
5. v2 显式使用 gzip level 6 + byte shuffle；拒绝其他滤镜及 scale-offset。它包含熵压缩收益，不能描述成单纯结构去重收益。旧 v1 无滤镜文件和默认 `--codec none` 行为继续支持。

| 全量轨迹，信息相同 | 原压缩 NPZ | 归一化 + 无损压缩 HDF5 |
|---|---:|---:|
| 60 个文件 | 437,546,852 字节 | 291,733,819 字节 |
| MB | 437.55 | 291.73 |
| 减少 | — | **33.33%** |

另复制约 1.65 MB 的共享结构、对齐坐标框架、配置和完成标记，并提供来源清单。这个分析输入包不包含 RNG、restart/endpoint 检查点、最终导出或其他实验，因此不是完整实验归档的替代品。原始完整归档和 NPZ 均保留。

三个真实 batch_000（joint/single/unguided）的对照试验说明，不能仅换容器就假定节省空间：

| 三个文件合计 | 大小 |
|---|---:|
| 原压缩 NPZ | 21.93 MB |
| 旧归一化、无压缩 HDF5 | 27.10 MB |
| 不归一化、gzip + shuffle HDF5 | 22.85 MB |
| 归一化、gzip + shuffle HDF5 | **14.63 MB** |

只有组合方案在这里明显更小。三个批次的 24 字段热读取中位时间，NPZ 约 0.038–0.040 秒，归一化 HDF5 约 0.065–0.074 秒。这是局部读取测量，不是端到端性能基准；新格式以少量解码开销换取空间，未声称加速读取。原 NPZ 仍可直接使用。

## 3. 验证内容

- **1,680 个原始数组**逐位核对：全部字段的 dtype、shape、数据位一致，包括淘汰候选及其评分、权重和谱系。
- 从新输入完整提取 60 个策略批次，得到的 `features.parquet`、`edges.parquet`、`selection_events.parquet`、scope 和特征目录，均与原 NPZ 提取结果有相同 SHA-256。特征缓存 SHA-256 为 `9ff3ec335279d37fed7882c76853bd30d6e014ff58d2eca744208d29d24da148`。
- **12 个核心正式结果文件逐字节一致**；8 张批次 Parquet 重新按原 12 位有效数字 CSV 格式序列化后，与基准 CSV 逐字节一致。Parquet 本身保留 float64，不沿用 CSV 的输出舍入。
- trends 和 PCA 各 **140,000 条父子边**的计算数组分区校验值与旧明细相同，即计算未因省略明细而跳过。
- LLM 的 **205 条证据、4 个概率加权候选原型、8 组事件曲线**保持相同。仅来源路径和数据集 provenance ID 更新；事件曲线文件本身逐字节相同。它们仍是选择偏好描述，不能解释为最终效果已验证的最优区间。
- **27 项测试通过**，涵盖四种方法的一致性、浮点精度、真实选择窗口、随机复制偏差、谱系、NaN/负零、损坏输入拒绝和旧明细混用拒绝。
- 最终用紧凑输入配置复跑，**26 个统计文件逐字节复现**，运行记录中的 63 个源码/配置哈希与实际文件一致。

机器可读证明位于新结果的 `verification/`；结构来源与各字段验证计数在输入包的 `input_bundle_manifest.json`。存储试验和原代码快照位于 `test/selection_compact_v3/`。

## 4. 后续复用

在 `EvoMolSteer` 下执行；用新的输出目录运行新实验。

```powershell
# 直接使用已验证的紧凑结构输入，默认输出归档态结果
.venv/Scripts/python.exe scripts/run_pipeline.py --root data/optimized/main1000_w050/analysis_inputs_v2 --output results/NEW_RUN

# 反复分析时保留一份特征缓存
.venv/Scripts/python.exe scripts/run_pipeline.py --root data/optimized/main1000_w050/analysis_inputs_v2 --output results/NEW_WORK_RUN --config configs/working.json

# 核验结构输入包；无需重新生成结构
.venv/Scripts/python.exe scripts/pack_analysis_inputs.py --output data/optimized/main1000_w050/analysis_inputs_v2 --verify-only

# 重建当前已回收的特征缓存并核对原 SHA-256
.venv/Scripts/python.exe scripts/materialize_feature_cache.py --analysis results/selection_window_v2_compact_v3

# 复核与集成前同范围结果的一致性
.venv/Scripts/python.exe scripts/verify_compact_selection.py --baseline results/selection_window_v2 --compact results/selection_window_v2_compact_v3
```

开启诊断明细时复制配置，将 `write_step_details` 和/或 `write_node_details` 设为 true，在新目录运行。只改存储开关，不改 `analysis_representations`、scope 或任何统计参数。旧目录与 compact 策略冲突时明确报错，不自动删除旧文件，也不把旧明细混入新证据。

Analyst 的输入仍为结构化证据包，接口契约未改变；Designer 不需要理解 HDF5 编码。需要具体三维候选时可使用 `TrajectoryPackage.node(step, slot)` 按节点恢复精确坐标，而不把全部轨迹文本塞进提示词。存储优化没有触发新的 LLM 推断或新的生成实验。

原始数据和旧结果目前保留作比较，因此上述数值表示新产物所需空间的减少，不表示已通过删除旧目录实际释放了对应磁盘空间。
