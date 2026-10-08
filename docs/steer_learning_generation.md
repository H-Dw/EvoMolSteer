# 指定输入的 Steer Sampling 与可读归档

此入口调用指定位置的 FLOWR.ROOT，在配置的评分窗口内进行原生 SMC 分支选择，继续推理到 t=1，再发布可直接读取的数据目录。默认窗口为 `[0, 0.5]`、100 个积分步、主随机种子 42。没有坐标奖励梯度。`single` 优化目标口袋预测亲和力；`joint` 使用目标/非目标口袋的联合选择权重；可追加 `unguided` 获得匹配背景。

本版本按“窗口＋完整谱系＋最终结构、评分、失败结果及输入”保留数据，默认 **完全不做熵压缩**：`storage_codec=none`，HDF5 无压缩过滤器，评分表 Parquet 使用 `compression=None`，也不产生 tar.gz/zip。重复数组引用、对称键矩阵及整数索引的可逆存储仍启用。读取器恢复原始数组的形状、dtype 和数值位模式；这些表示不影响评分事件读取。需要压缩时可以显式选择 `gzip_shuffle`，它也是无损且透明读取，但不是当前默认配置。

## 输入与执行

源码入口为 `scripts/generate_steer_learning.py`，模块接口为 `evomolsteer.generation.steer_launcher`。入口不要求把 FLOWR 复制到本项目中；配置 `flowr_root`、`checkpoint` 和 `python_executable` 指向其程序、模型和已有运行环境。

复制 `configs/generation_steer_learning.example.json`，填写输入和输出目录。在输入目录保存 `pocket_inputs.json`，格式参见 `configs/pocket_inputs.example.json`：

```json
{
  "format": "evomolsteer.pocket_inputs.v1",
  "files": {
    "target_protein": "my_protein.pdb",
    "target_ligand": "my_reference_ligand.sdf"
  }
}
```

目标蛋白 PDB 与参考配体 SDF 应在同一坐标系。它们用于构建口袋和参考配体原子数条件；参考配体不是需要被继续优化的初始分子。需要 `joint` 时，再提供 `off_target_protein` 和 `off_target_ligand`，两口袋须预先结构对齐。单目标未提供非目标时，在运行接口中将非目标角色指向目标文件，不重复保存文件；这时 off-target 分数不能当作真实选择性证据。清单路径相对于清单所在目录；也可以用 `--input-manifest` 指定外部清单。

```bash
python scripts/generate_steer_learning.py --config /path/to/generation.json --dry-run
python scripts/generate_steer_learning.py --config /path/to/generation.json
```

`--dry-run` 校验路径、输入角色、采样范围和输出占用，只输出执行计划。正式执行必须使用新的空输出目录，输入文件按 SHA-256 校验后复制，原输入不修改。`samples` 是每个 arm 的候选总数，必须是 `batch_size` 的整数倍；不在输出端提前剔除失败或重复分子。

已有 CK2/CLK3 输入可直接使用 `configs/generation_ck2_clk3_steer_learning.json`：指定历史实验输入、v2 checkpoint、1000 候选/arm、100 候选/批次（10批）、100 步、窗口 `[0,0.5]`、seed 42，保存到新的实验目录。配置包含 `single,unguided`，以便之后进行匹配背景的富集分析。在远端 EvoMolSteer 仓库目录执行：

```bash
python scripts/generate_steer_learning.py --config configs/generation_ck2_clk3_steer_learning.json
```

只需要 Steer 数据时加 `--arms single`。更换输入、路径或窗口可直接修改配置，或用 `--input-dataset`、`--output-dataset`、`--flowr-root`、`--checkpoint`、`--python`、`--window-start`、`--window-end` 等覆盖。窗口和积分步数独立配置，没有在录制代码中硬编码半程。

## 评分时间与保存范围

选择范围按原生 **score_time** 的闭区间定义，而不是按后续 proposal 时间截断。100 步的默认线性网格会记录 `t=0,0.01,…,0.50` 共 51 次评分/选择事件；最后一次评分所选择的 proposal 时间约为 0.51，仍须完整保存，否则丢失最后一次实际选择的数据。具体时间直接取模型原生浮点网格，并在运行记录中保存。

窗口内每个事件保留完整候选群体：

| 数据 | 含义 |
|---|---|
| `current_*` | 评分前的实际状态 |
| `predicted_*` | 此时模型对终点的预测，不是实际 t=1 结果 |
| `proposal_*` | 此次积分产生、随后被复制或淘汰的状态 |
| `pic50_on/off`、`weight_on/off`、`selection_probability` | 原始评分与选择概率，未重新定义选择算法 |
| `selected_indices`、`offspring_count`、`root_slot`、`parent_slot` | 下一代槽位对应的父候选、复制次数和谱系标签 |
| `score_time`、`state_time`、`step_size`、`source_step` | 原生时间、积分步长与原始步编号 |

三种结构视图均含坐标、原子标签、键标签、电荷、mask；模型存在 hybridization 时也保存。坐标保持原始浮点精度；预测原子/电荷/hybridization 概率按模型原始 dtype 保存，不再新增 float16 转换。窗口内的完整预测软键概率张量可用 `--capture-bond-probabilities` 开启；默认只存键标签，避免额外的 `O(B N² K)` 存储。t=1 的原生输出始终保存所有数值张量，包括模型实际返回的键概率和 affinity 字段。

“无损”指上述声明保留的数组可以逐比特恢复。它不是整个运行内存的快照：不保存窗口外的中间几何、不保存 RNG/restart/重复 endpoint pickle，不能用作完整随机过程断点恢复。窗口外仍保留全部积分步的谱系与时钟，继续使用原生积分直到最终结构产生；窗口之后没有粒子重采样或坐标奖励梯度。

## 输出数据集

```text
<output_dataset>/
  inputs/
    pocket_inputs.json
    target_protein.pdb, target_ligand.sdf, [off_target_*.pdb/sdf]
  learning_dataset_manifest.json
  results/<campaign>/
    generation_request.json, config.json, runtime.json, generation.log
    frame_batch_000.json, ...
    provenance/upstream_generate_selective.py
    provenance/instrumented_generate_selective.py
    final_records.json, COMPLETE.json
    <arm>/batch_000/
      trajectory.h5, trajectory.manifest.json
      lineage/trajectory.h5, lineage/trajectory.manifest.json
      terminal.h5
      selection_events.parquet
      terminal_ancestry.parquet
      events.json
      final_records.json
      molecules_all_built.sdf, molecules_raw_decodable.sdf
      quality_records.json, QUALITY_COMPLETE.json
      learning_manifest.json, COMPLETE.json
```

`trajectory.h5` 只含窗口内全部候选；`lineage/trajectory.h5` 含完整 100 步的轻量谱系；`terminal.h5` 是模型实际返回的最终输出。最终坐标在 world Å 坐标系；窗口坐标在 model 单位，利用 `coord_scale` 和每批 `target_com` 还原，事件读取接口会同时返回坐标参照及词表。

`selection_events.parquet` 是小型在线评分索引，保留所有候选和 dtype，不重复展开坐标；`terminal_ancestry.parquet` 单独存储回溯到最终槽位的后代数量。终末标签不会混入在线选择表。窗口内的零子代只表示当步被淘汰，不能被解释为化学构建失败；已淘汰分支没有未经运行的反事实最终评分。

最终每个槽位均有记录，包含构建状态、失败原因、原生终末评分、另列的终末再评分及可获得的基础化学/碰撞描述符。能够解码但不能通过 sanitization 的结构另存 raw SDF；即使 SDF 无法生成，`terminal.h5` 和失败记录仍保留原始预测。此入口不进行 MMFF 或 PoseBusters 能量评估，质量记录明确标注 `energy_status=not_evaluated`。

生成时每步同步写入和校验，自有临时分片在批次合并、逐比特验证并发布后回收。不会先积累整个原始轨迹 NPZ。中断时保留已提交数据并标记失败，不会把尚未到 t=1 的运行标记为完整。文件 SHA-256、数组哈希、完整谱系/窗口一致性、输入、坐标参照、模型及源码来源均有记录。

## LLM 评分事件接口与分析复用

从指定事件读取单个候选，包括被淘汰候选，无须加载全部轨迹到 prompt：

```bash
python scripts/read_steer_event.py \
  --batch-directory /path/to/dataset/results/single_w050/single/batch_000 \
  --step 50 --slot 1 --verify --output /path/to/event.json
```

`--step` 使用原始积分编号，`--slot` 可省略以返回整次候选群体。`--scores-only` 输出轻量评分/谱系，`--terminal-outcome` 单独附加后代终末槽位映射及回溯标签说明。默认不把最终结局作为选择时已知的证据。坐标上下文和轨迹哈希不匹配时拒绝读取。

Python 接口：

```python
from evomolsteer.generation.steer_launcher import SteerLearningConfig, launch, verify_dataset
from evomolsteer.storage.selection_dataset import read_scoring_event, verify_learning_batch

# cfg = SteerLearningConfig(flowr_root=..., checkpoint=..., input_dataset=..., output_dataset=...)
# launch(cfg, dry_run=True)
# manifest = launch(cfg)
event = read_scoring_event(batch_directory, source_step=50, slot=1)
```

现有 HDF5 分析读取器已接入此数据格式，通用口袋角色可用于区域目录和坐标参考；历史 CK2/CLK3 文件名仍可读取。使用当前完整窗口富集/差异/趋势分析时，需要配置对应 campaign、匹配背景 `unguided` 和输入口袋名称（建议 `feature_pockets=["target"]`）。单独的 `single` 归档可提取选择事件，但不满足要求独立背景的统计入口。归档程序不更改已有的分析统计定义、奖励函数或 Skills。

## 验证与边界

验证记录见 `docs/steer_learning_validation_20261008/`。测试覆盖：无压缩/无损压缩两种存储、被淘汰同胞、全程谱系与窗口边界、原始概率精度、通用输入、坐标参照校验、旧分析入口兼容、失败和完整状态区分。原生 FLOWR 方法的完整源码 fixture 还验证去掉五个录制回调后逐行恢复原算法。

真实历史 50 候选批次回放验证了 51 个窗口事件、100 步谱系、28 个轨迹数组和 13 个终末数组逐比特一致，原始文件未修改。回放保留历史 f16 原貌；它不能将历史 f16 概率逆向恢复为 float32。新生成直接保存模型原始精度，因此旧包的容量不能直接承诺给新包。

当前验证为 CPU 存储回放与接口测试，尚未用新入口执行 GPU 生成。示例中的 0.01 是本配置 100 步线性积分的事件间隔；不据此声称论文 CK2/CLK3 原始实验的间隔、粒子数和窗口完全相同。
