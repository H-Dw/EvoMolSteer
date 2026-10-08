# 远端 Steer sampling 原始数据与选择窗口存储审计

审计日期：2026-10-08。服务器：`ksai.scnet.cn:10544`。原始实验目录：`/root/private_data/MolSteer/flowr_root/experiments/ck2_clk3_lineage_20261003/`。算法来自 EvoMolSteer 基线 commit `d0fcf8c577a2ccbaa4e1a9618720768c36b7c639`。

本次只读取远端原始数据，使用经 SHA256 确认相同的本地副本测量压缩。没有替换或删除远端原始 Steer 数据，也没有运行新的 FLOWR 推理。

## 实际占用与统计口径

整个原始实验目录的文件长度合计 **1,076,104,210 B = 1,076.10 MB**，包含单目标、联合目标、无 Steer 对照及其他实验/分析，不能作为一个分子的占用。

本次主要测量 `results/main1000_w050/single`：20 个批次，每批 50 个生成槽位，100 个积分步，共 1000 个终末槽位。每个候选含 25 个原子。该目录实际仍保存 **`trajectory.npz`**，不是已经转换后的 HDF5。

- 单目标目录：540 个文件，文件长度合计 **311,527,970 B = 311.53 MB**。
- Linux `du -s -B1` 实际分配：**311,761,920 B**，包含目录块。整个原始实验实际分配为 **1,077,060,608 B**。以下压缩比较统一使用文件长度，1 MB = 1,000,000 B；不是 MiB。
- 按 1000 个生成槽位摊算，当前全目录约 **311.53 KB/槽位**；按一个 50 候选批次约 **15.58 MB/批次**。包含构建失败与重复后代，不是 1000 个独立、有效、不同的小分子。

| 当前单目标目录组成 | 文件长度 / MB |
|---|---:|
| 20 份压缩 NPZ 轨迹，包含 100 步 | 136.401061 |
| restart 快照 | 69.873680 |
| endpoint 快照 | 69.667559 |
| initial_state | 19.735414 |
| final_prediction | 9.056879 |
| 评分、事件、SDF、质量结果等 | 6.793377 |
| 合计 | **311.527970** |

超过一半空间由轨迹之外的附件占用。因此仅优化轨迹，不能对整个目录宣称 80% 或 95% 的无损缩减。

## 0–0.5 窗口究竟包含哪些数据

直接读取当前轨迹确认：实际 `resampled` 节点为 step 0–50，`score_time` 从 0 到 0.5，间隔 0.01，共 **51 个评分/选择事件**，20 批共 **51,000 个候选观测**。这些是当前复现数据的实测设置，本报告不据此推断论文的设置。

时间必须区分：评分 t=0.5 的当前状态在 t=0.5，但该事件筛选的 proposal 位于 **t≈0.51**。本次按“实际选择事件的评分时间 ≤0.5”保存，因此保留这个必要的选择结果。若严格要求所有实际几何时刻 ≤0.5，则需另设范围，只分析 step 0–49 的更新并保留 t=0.5 边界；不能悄悄丢掉最后一个选择事件。预测终点坐标是当时的 t=1 预测，也不是实际已经积分到 t=1 的状态。

原文件未单独拆出窗口，不能把它的大小直接除以二。我们对全部 20 批实际提取这 51 个事件，重新使用同样的 `np.savez_compressed` 保存，窗口轨迹总量为 **66,081,074 B = 66.08 MB**。这是新提取副本的实测大小，并非远端目前存在一个 66.08 MB 的窗口文件。

## 已有算法与实测压缩结果

现有实现位于：

- [arrays.py](../../src/evomolsteer/storage/arrays.py)：常量折叠、逐比特相同的行字典、精确正零稀疏编码、整数存储位宽收缩后恢复原 dtype、gzip + byte shuffle。
- [trajectory.py](../../src/evomolsteer/storage/trajectory.py)：当前状态引用上一 proposal 和 `selected_indices`；对确实对称的键矩阵保存上三角与对角；重新打开文件并核验每个字段的形状、dtype 和全部数值比特。
- [streaming.py](../../src/evomolsteer/storage/streaming.py)：生成阶段即时提交、单阶段 HDF5 外层 gzip 压缩重复元数据；完成批次后归并，再清理已验证阶段副本。已有生成控制器使用这个 writer。
- [lifecycle.py](../../src/evomolsteer/storage/lifecycle.py) / [optimize_trajectories.py](../../scripts/optimize_trajectories.py)：完成后的 NPZ→HDF5 数据集转换，默认保留源文件；删除源文件需要显式选项且通过验证。

没有新增坐标量化、float32→float16 转换或损失性压缩。原始概率数组已有的 float16 精度保持原样。

| 方案与保留范围 | 原大小 / MB | 优化后 / MB | 减少 |
|---|---:|---:|---:|
| 100 步轨迹的全部 28 个数组，无损转换 | 136.401061 | **91.162413** | **33.17%** |
| 全单目标目录：仅替换轨迹，其他附件全部保留 | 311.527970 | **266.289322** | **14.52%** |
| 同一 51 事件窗口：全部 28 字段，NPZ→HDF5 | 66.081074 | **49.093736** | **25.71%** |
| 选择学习工作包，含全部候选、最终结果和谱系 | 311.527970 | **61.930929** | **80.12%** |
| 上一工作包再归档为 tar.gz | 61.930929 | **48.918016** | **21.01%** |
| 学习归档相对原全目录 | 311.527970 | **48.918016** | **84.30%** |

完整目录 266.29 MB 是“20 个实测 HDF5 的总大小 + 保留附件的实测原大小”之和，未在远端进行替换。其余窗口和学习归档均由真实副本生成、测量，不是从单批推算。

**前两行是原始轨迹信息的无损重编码；后几行包含明确的数据范围裁剪，不能还原整个原始实验。** 此前分析输出目录的 95%/99% 节省比例也不能套用到本次原始生成数据。

学习工作包 61,930,929 B 的组成：

| 内容 | B |
|---|---:|
| 51 事件窗口全部原始字段 | 49,093,736 |
| 原轨迹最后一行全部字段 | 3,503,670 |
| 全 100 步父子关系、重采样标记、时钟等 | 806,488 |
| 完整终末 SDF、评分、失败原因、质量结果、事件、输入与坐标变换、配置等 | 8,420,334 |
| 可追溯清单 | 106,701 |

保留 current/proposal/predicted 坐标、原子/键/电荷及概率、两靶点评分、选择概率、权重、offspring_count、root/parent/selected_indices、mask、时间与步长。窗口内所有未被选中候选的几何和评分仍在，不能只保留幸存者。最后一行和原始终末结果保留；全程谱系可以将窗口分支与终末槽位关联。逐项验证了选择窗口末端的后代坐标，以及终末祖先映射。

裁剪内容：选择窗口以外的连续几何/预测轨迹（最后一行除外）、PyTorch 的 restart/endpoint 快照和 RNG 状态。这些对当前选择窗口特征学习不是必要输入，但对任意时刻严格重启生成可能必要。归档无法替代完整重启档案。窗口学习包为**测量用视图**，有独立清单，尚未接入所有正式读取流程，不应冒充现有完整 input_bundle 直接使用。原来的 COMPLETE.json 作为历史来源文件保留，包自身以 manifest 定义范围和容量。

## 如果“单一 molecular”指一条分子的路径

重采样产生共享祖先和分支，一条最终分子并不拥有一个独立原始轨迹文件。按批次摊算和实际抽取一条谱系是两种口径：

- 全窗口轨迹，按 1000 个终末槽位摊算：原 NPZ **66.08 KB/槽位**，HDF5 **49.09 KB/槽位**。
- 包含失败对照、终末结果、完整谱系和共享输入的学习工作包：**61.93 KB/槽位**；归档后 **48.92 KB/槽位**。这些是共享数据的摊算，不能认为每个分子可以独立取走这个大小的文件。
- 真实抽取 `batch_000` 的 final slot 0 的 51 事件祖先链，25 原子，保留三种状态表示、概率、评分和路径信息：压缩 NPZ **88,507 B**；独立 HDF5 **165,835 B**；HDF5 外层 gzip 后 **67,530 B**，较该路径 NPZ 减少 **23.70%**。这是一个实测示例，不是所有分子的固定容量。
- 单独这一条路径的 current XYZ 数组本身只有 **15,300 B**；评分、其他状态、概率、键和元数据使完整路径更大。单粒子 HDF5 元数据开销较大，建议按 50 个候选批次组织，再进行无损归档。

单路径仅含该终末槽位祖先，不含未幸存同胞；原始 population 指针映射为本地单路径指针，并显式保存原 candidate/child slot。这是有损范围抽取视图，不能单独用于无偏富集分析。

## 验证、复用与建议

540 个单目标原始文件全部与远端新计算的 SHA256 一致。20 批完整 HDF5 与源 NPZ 共 **560 个数组**逐比特一致；窗口、终末行和谱系编码各自重新打开核验。学习归档 **251 个成员**全部逐文件 SHA256 一致，归档 SHA256 为 `789424c62316e85e7b5009d346fa30eb8e7c752cef0cc9e17e6dc861d32d6f9e`。

建议：当前选择特征学习使用 **约 61.93 MB 的工作视图 / 48.92 MB 的归档**，输入几何和全部失败对照保留。完整原始档案可放在独立归档位置；若远端需严格保留完整重启能力，则采用约 **266.29 MB** 的轨迹无损替换方案，其整体节省为 14.52%。本次没有删除原始 Steer 或 checkpoints。

本地已保留验证后的学习归档：`test/storage_audit_20261008/path_and_archive/selection_learning.tar.gz`；另有 67,530 B 的单谱系归档 `ancestral_path.h5.gz`。

计划在验证后清理仅由本次审计产生的工作副本和重复 NPZ baseline，但自动审批拒绝了删除操作，返回原因仅为 `blocked by policy`。因此本地临时副本仍保留，合计 **128,266,345 B**，不应计入学习包的必要存储容量。远端原始档案及本地原始数据均未删除。清理状态见 [retention.json](retention.json)。

复现脚本：

```powershell
.venv/Scripts/python.exe docs/storage_audit_20261008/benchmark.py `
  --input data/raw/ck2_clk3_lineage_20261003 `
  --normalized data/optimized/main1000_w050/analysis_inputs_v2 `
  --remote-inventory docs/storage_audit_20261008/remote_inventory.json `
  --remote-hashes docs/storage_audit_20261008/remote_single_hashes.json `
  --output test/storage_audit_20261008/window_trial `
  --report docs/storage_audit_20261008/benchmark.json

.venv/Scripts/python.exe docs/storage_audit_20261008/single_path_measure.py `
  --source data/raw/ck2_clk3_lineage_20261003/results/main1000_w050/single/batch_000/trajectory.npz `
  --output test/storage_audit_20261008/path_and_archive `
  --window-package test/storage_audit_20261008/window_trial `
  --report docs/storage_audit_20261008/single_path_and_archive.json
```

脚本要求输出目录不存在，支持动态指定评分范围 `--start/--end`，不会自动删除原始数据。重新运行时应指定新的试验输出路径和报告路径，以保留此前证据。

机器可读数据：[远端清单](remote_inventory.json)、[所有单目标文件新鲜 SHA256](remote_single_all_hashes.json)、[20 批测量](benchmark.json)、[单路径与归档测量](single_path_and_archive.json)、[最终核验](verification.json)。
