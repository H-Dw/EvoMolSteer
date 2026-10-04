# EvoMolSteer · 选择窗口分析 v2

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

Designer v2 只能编译有来源的预测终点规则；奖励必须限定于观测到的评分窗口。每个阶段门有明确的区间截断，窗口外坐标梯度严格为零。新增生成接口已完成小规模 SMC 生成验证；尚未执行 Designer 奖励的梯度引导优化实验。

## 数据与历史

`data/raw/` 保留完整原始实验。新 `data/optimized/main1000_w050/analysis_inputs_v2/` 是可直接运行分析的无损轨迹输入包，保留所有 100 步和 28 个字段；范围仍由 `resampled` 限制为 51 个事件。它不包含恢复生成所需的 RNG/检查点，不能替代完整实验归档。旧 `features_v1/` 是历史全程特征缓存，不能直接当作 v2 证据输入。存储接口说明见 [`docs/storage_format.md`](docs/storage_format.md)。

旧结果 `results/main1000_w050`、旧 Analyst/Designer 文档和原型奖励均为历史记录，已由 v2 主分析取代。修改前源码存于 `delivery/EvoMolSteer_pre_selection_window_v2_20261004.zip`；旧全程分析不与当前选择证据混用。
