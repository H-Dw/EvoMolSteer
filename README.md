# EvoMolSteer · 完整选择窗口连续分析 v3

**当前坐标优势挖掘与实验**：重新从第1轮计数，上限30轮。新增880项当前区域坐标/漂移观测，分开筛选富集、同期 affinity 关联和去重父节点的后续 gain。使用实测连续窗口的坐标奖励，固定 seed42，窗口后原生完成 t=1；每轮先保留报告再清理旧生成数据。见 [方法、接口与证据限制](docs/coordinate_affinity_mining_20261006.md)。

**历史五轮 seed42 终态实验**：`scripts/generate_local_flowr.py` 使用真实 FLOWR endpoint Jacobian，对学习窗口内的相关局部可接受集合施加有界梯度，窗口后原生继续至 t=1。无新增 SMC。见 [历史协议](docs/terminal_seed42_protocol_20261005.md) 和 [五轮结果与清理报告](docs/ck2_terminal_seed42_report_20261005.md)。

**实际中间态控制入口**：`scripts/generate_window_flowr.py` 将梯度范围绑定到学习参考的动态窗口，关闭 FLOWR 内置粒子重采样，主目标是严格窗口末端的 actual current 状态。接口、数学定义、数据格式和复用命令见 [window_gradient_generation.md](docs/window_gradient_generation.md)。旧 full-horizon multistage 配置保留为历史实验，不是此入口的默认行为。

**历史八轮实验**：共 416 个候选，当时78项测试通过。冻结奖励在新种子上的形状距离平均只改善 0.89%，四个配对批次的区间跨零，尚未确认稳定拟合 Steer 中间态。完整结果、各轮决策、奖励公式、范围和局限见 [八轮窗口拟合报告](docs/ck2_window_iter8_report_20261005.md)。生成原始输出已按用户要求清理，报告与冻结配置保留。

主分析现已改为对 **0–0.5 的全部实际选择节点**统一分析，不再汇总为 0.1 宽子区间。连续趋势、双侧富集、窗口末端谱系留存、选择/生成变化分解、全局函数拟合及解析导数分别保存；新增区域坐标、原子、形式电荷和化学键观测。物理能量缺失时不以几何代理冒充。

```powershell
.venv/Scripts/python.exe scripts/run_pipeline.py --root data/optimized/main1000_w050/analysis_inputs_v2 --output results/continuous_single_v3 --config configs/continuous_single_target.json
```

[v3 方法、公式和输出接口](docs/continuous_analysis_v3.md) · [完整运行、优势 seed 特征和函数拟合结果](docs/continuous_single_v3_report_20261005.md)。本地与远端已完成复算，104 张主要结果表数值一致；58 项测试通过。默认配置已切换 v3；下面的 v2 方法与既有梯度实验作为历史记录保留，不能与新统计混用。

## 历史实验：多阶段奖励与全程梯度推理

`scripts/build_multistage_reward.py` 从 discovery 的逐时刻留存祖先提取相关区域目标；Designer 选择的稳健混合奖励使用局部PCHIP拟合，在全部100个积分步上通过 live endpoint Jacobian引导当前坐标。0.5后的末端参考维持是显式实验假设。内部协方差尺度和外部native-step位移比例分开；强度先在独立pilot校准。

生成入口为 `scripts/generate_multistage_flowr.py --flowr-root <FLOWR目录> ...` 或 `evomolsteer-multistage`；分析入口为 `scripts/evaluate_multistage_runs.py`。见[Designer决策与公式](docs/ck2_multistage_design.md)、[执行接口和冻结比较方案](docs/ck2_multistage_execution.md)。旧版早期奖励作为独立对照保留。

已完成 **60个pilot + 384个独立对照候选**。25段奖励在全程100步实际生效；选择窗内到SMC留存祖先的二维距离降低11.13%，但终态结构模仿未改善，多阶段全程未整体优于恒定目标。完整结果和限制见[实验报告](docs/ck2_multistage_report_20261005.md)、[Designer结果复核](docs/experiments/ck2_multistage_20261005/Designer.outcome_review.md)。74项本地测试通过，18张远端/本地评价表在1e−10容限内一致；两个生成归档的854个文件已下载并逐一校验。

## 历史 v2 分析与已完成实验

从 FLOWR.ROOT **实际执行重采样的事件**中提取几何选择特征，为后续局部规则提出提供依据。主分析不使用终态、后代成功标签、窗口后的轨迹或反事实续跑结果。

当前有效结果：[`results/selection_window_v2_compact_v3/reports/RUN_SUMMARY.md`](results/selection_window_v2_compact_v3/reports/RUN_SUMMARY.md)。方法、公式和解释边界见 [`docs/selection_analysis_v2.md`](docs/selection_analysis_v2.md)。存储集成与实测见 [`docs/selection_storage_integration_20261004.md`](docs/selection_storage_integration_20261004.md)。

## 生成与即时存储

`scripts/generate_flowr.py` 通过指定的 FLOWR.ROOT 路径、Python 环境和 checkpoint 启动 Steer sampling。每个积分步产生结构、评分和选择结果后自动转为无损容器并清空原始数组；批次结束后合并为 `trajectory.h5`，逐字节核验成功后自动删除阶段文件。新过程不创建原始 `trajectory.npz`，保留淘汰候选和完整谱系。

可直接修改 [`configs/generation_ck2_clk3.example.json`](configs/generation_ck2_clk3.example.json)，运行 `python scripts/generate_flowr.py --config configs/generation_ck2_clk3.example.json`。已有完整数据集使用 `scripts/optimize_trajectories.py --input <输入数据集> --output <输出数据集> --delete-source` 转换并回收已验证替代的 NPZ。当前模块适配已有 CK2α/CLK3 控制流程。

输入输出契约、Python API、远端环境、删除范围及异常恢复见 [`docs/online_generation_storage.md`](docs/online_generation_storage.md)。

## 范围

实际 `resampled` 标记决定分析范围；无重采样组取匹配时间节点作背景。本实验 joint/single 每批为 t=0,0.01,…,0.50，共 51 次选择。t=0.50 的 proposal 位于约 0.51，仍属于最后一次选择事件。

基本统计单位是**同一批次内的同一次选择事件**。为了阅读方便，再汇总为 [0,.1)、[.1,.2)、[.2,.3)、[.3,.4)、[.4,.5]，每批分别包含 10、10、10、10、11 个事件。原始时间用于变化率，舍入只用于分箱。

仅提取 `predicted_endpoint` 与 `proposal_state` 两种表示，共 306,000 行、255 个几何特征。预测终点描述评分时的模型预测，proposal 描述被实际复制的候选；二者不得混称实际原子轨迹。

## 模块与产物

| 模块 | 计算 |
|---|---|
| `scope.py` / `ingest.py` | 从真实选择标记恢复范围，校验父子关系、匹配先验和背景；不读 final 文件 |
| `enrichment.py` | 同一评分步的无重采样背景阈值；期望与实际高特征质量变化、选中/淘汰 log-OR |
| `differential.py` | 逐事件选中/淘汰差异、同一直接父节点的同胞对照 |
| `trends.py` | 期望选择、实际复制、随机偏差、on/off 分量；窗口内父子边变化率 |
| `pca.py` | 仅在 discovery 匹配窗口的背景上拟合；选择质心变化与父子边 PC 变化率 |
| `statistics.py` | 事件先比较、阶段内均匀汇总；批次检验、向量化 bootstrap、事件及阶段导数 |
| `evidence.py` | 仅选择证据；概率加权候选原型，不使用成功后代 IQR |
| `agents.py` / `contracts.py` | Analyst/Designer 接口，拒绝旧证据包、窗口外规则和表示错配 |

每种方法单独保存：

- `batch_effects.parquet`、`effects.csv`：完整批次阶段效应和跨批次推断。
- `batch_effect_rates.parquet`、`effect_rates.csv`：以实际参与时间计算的阶段变化率。批次 Parquet 无压缩，浮点数保留 float64。
- `event_diagnostics.json`：事件数、导数有效值数和选择分解一致性；trends/PCA 另存父子变化率的逐批计算校验值。
- PCA 的轴、载荷和解释方差始终保留。LLM 目标特征的逐事件曲线保存在 `agents/selection_feature_profiles.json`。
- `events.parquet`、`event_rates.parquet`、PCA 逐节点投影及父子边明细是可选诊断。将 `write_step_details` 或 `write_node_details` 设为 true，并使用新输出目录即可保存。计算本身不因关闭明细而省略。

所有独立粒子批次 0–13 为发现集；14–16 为验证集；17–19 为留出集。LLM 只收到发现集。克隆和连续时间帧不是独立样本；同一父节点的零差异是对照信息。被淘汰表示当步零子代，不能解释为最终失败。

## 复用

在项目目录执行；当前 `.venv` 与锁定依赖已安装。

```powershell
.venv/Scripts/python.exe scripts/run_pipeline.py --root data/optimized/main1000_w050/analysis_inputs_v2 --output results/selection_window_v2_compact_v3 --skip-ingest
```

重新提取到新目录时去掉 `--skip-ingest`。已有提取仅允许更新不影响范围和几何的分析参数；配置和源码哈希写入 provenance。

默认 compact 策略在流水线完成后回收可重建的几何缓存。反复工作时可加 `--config configs/working.json`，保留一份缓存；后续使用默认配置运行可恢复归档态。再次分析时，读取器会从登记的 NPZ 或 HDF5 轨迹重建缓存并核对 SHA-256。`selection_events.parquet` 和 scope 文件可直接用于查看节点范围。已有目录若残留与策略冲突的旧明细会报错，避免混用历史结果；请使用新目录。

独立入口保留：`extract_geometry.py`、`analyze_enrichment.py`、`analyze_differential.py`、`analyze_trends.py`、`analyze_pca.py`、`build_evidence.py`。旧终态、反事实分析模块及命令已从主源码移除。

```powershell
.venv/Scripts/python.exe -m pytest tests -q
.venv/Scripts/python.exe scripts/check_reproducibility.py --output results/selection_window_v2_compact_v3
.venv/Scripts/python.exe -m evomolsteer.cli agent-export --role Analyst --output results/selection_window_v2_compact_v3
```

`agents/Analyst.request.json` 包含选择窗口专用 skill、prompt 和严格 JSON 契约。API 使用 `EVOMOLSTEER_BASE_URL`、`EVOMOLSTEER_MODEL`、`EVOMOLSTEER_API_KEY`，或把 request 中的 messages 交给 sub-agent。分析流水线只导出请求，不自动调用 LLM。

Designer v2 只能编译有来源的预测终点规则；奖励必须限定于观测到的评分窗口。每个阶段门有明确的区间截断，窗口外坐标梯度严格为零。原 v2 程序只验证 saved-endpoint 梯度；新增的 live-regional 程序有独立执行契约，见下文。

## CK2 单目标 live gradient 实验

已完成新的单目标分析、Analyst/Designer subagent 模拟、24 个 pilot 候选和 800 个正式比较候选。所有生成代码先在本地提交并经本机 VPN 推送 GitHub，再由远端拉取运行。主实验使用历史数据同版本的 v2 checkpoint，生成 commit 为 `9eae1cc`。

早期 VAL116 区域距离与随后回转半径的选择关联可复现；单侧 Huber 奖励通过真实 FLOWR forward 反传到当前坐标，梯度组没有重采样。零权重与原生输出逐位一致，live 有限差分通过。正式比较中，区域梯度/组合梯度相对基线的 CK2 模型均分差为 −0.0014/+0.0041，四个独立批次的区间均跨零；目前不能宣称规则已经带来明确亲和力提升或替代 SMC。

- [完整分析、奖励公式与比较报告](docs/ck2_single_guidance_report_20261004.md)
- [生成接口与复用命令](docs/gradient_generation.md)
- [冻结设计、来源及验证计划](docs/ck2_single_guidance_design.md)
- 入口：`scripts/generate_gradient_flowr.py`；比较：`scripts/compare_gradient_runs.py`；谱系审计与打包：`scripts/audit_generation.py`、`scripts/archive_generation.py`。

此前本地数据包 `data/single_guidance_comparison_v1.tar.gz`（约212 MB）的449个文件曾通过校验；该生成结构包已清理，仅保留比较报告和配置。原始 Steer 输入与 checkpoints 保留；模型中间状态和大表不进入 Git。

## 数据与历史

`data/raw/` 保留完整原始实验。新 `data/optimized/main1000_w050/analysis_inputs_v2/` 是可直接运行分析的无损轨迹输入包，保留所有 100 步和 28 个字段；范围仍由 `resampled` 限制为 51 个事件。它不包含恢复生成所需的 RNG/检查点，不能替代完整实验归档。旧 `features_v1/` 是历史全程特征缓存，不能直接当作 v2 证据输入。存储接口说明见 [`docs/storage_format.md`](docs/storage_format.md)。

旧结果 `results/main1000_w050`、旧 Analyst/Designer 文档和原型奖励均为历史记录，已由 v2 主分析取代。修改前源码存于 `delivery/EvoMolSteer_pre_selection_window_v2_20261004.zip`；旧全程分析不与当前选择证据混用。
