# HiQBind / CrossDocked100 比较与 100 候选、20 粒子默认配置

日期：2026-10-09。根据用户本轮要求，将新 Steer 数据生成的默认配置改为每 target 100 个候选槽位、5 个独立群体、每群体共同竞争 20 粒子，steering duration 0.5。

## HiQBind 测试集的组成

HiQBind 是通过结构清理流程整理的实验蛋白–配体复合物及结合标注集合。它以 PDB 共晶结构为基础，将 BioLiP、Binding MOAD、BindingDB 等来源的结合测量与结构关联，处理配体连接/化学结构、原子缺失、质子化、蛋白结构补全和空间冲突等问题。它不是单一蛋白家族或 CK2/CLK3 专用集合。

FLOWR.ROOT 的 Supplementary Information 报告使用 31,571 个带结合标注的 HiQBind 复合物，整个资源超过 18,000 个不同 PDB 条目。此处 31,571 是 FLOWR 使用的全数据规模，不是测试集规模，也不应与其他 HiQBind 发布版本的全量数字混同。

FLOWR 采用与 Plinder 一致的训练/验证/测试划分。正文 Figure 2 的测试亲和力评测报告：pIC50 116 个样本、pKi 54 个、pKd 108 个，综合为 278；Figure 3 的 Steer 评测图注也报告 278 个测试 target、每个采样 100 个配体。Steer 的共同目标是预测 pIC50，即使参考输入带有其他实验测量类型，也不能将其当作该参考配体都有已测 IC50。

target 指口袋/参考配体结构系统，不等于唯一蛋白种类。进一步计数出版社 Figure 3a Source Data，实际有 297 个不同 target_id 和 148 个 PDB 前缀，与图注 278 不一致，原因尚未确定。完整标识保存在 `flowr_figure3_source_target_ids_20261009.csv`；没有从该表擅自构造一个“已经核验的 278 蛋白清单”。

来源：[FLOWR.ROOT 论文](https://www.nature.com/articles/s41467-026-74130-9.pdf)、[FLOWR 补充材料](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41467-026-74130-9/MediaObjects/41467_2026_74130_MOESM1_ESM.pdf)、[HiQBind 作者代码及数据入口](https://github.com/THGLab/HiQBind)、[出版社 Source Data](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41467-026-74130-9/MediaObjects/41467_2026_74130_MOESM4_ESM.zip)。

## 与当前 CrossDocked100 的区别

| 比较项 | FLOWR 使用的 HiQBind 测试数据 | 当前 CrossDocked100 输入 |
| --- | --- | --- |
| 基础结构 | 清理过的实验共晶蛋白–配体结构 | CrossDocked 衍生的口袋/配体配对，包含对接或最小化结构；不能认为每个输入都是原位共晶姿态 |
| 参考亲和力 | 核心数据带实验结合标注，可用于模型亲和力预测评测 | 常用生成子集主要提供结构；实验 affinity 不保证逐条齐全，不能把 docking score 当作实验 pIC50 |
| 输入规模 | 图注报告 278；Source Data 含 297 个结构标识 | 已有输入审计为 100 个复合物配对，分布于 93 个蛋白区域目录，部分目录有多个配对 |
| 划分依据 | FLOWR 统一遵循 Plinder 划分 | 常用公开资源使用 AR/Pocket2Mol/TargetDiff 对应的固定生成测试划分；当前脚本冻结已准备目录的 100 个输入及哈希 |
| 主要用途 | 实验亲和力预测评测，以及预测亲和力 Steer | 口袋条件生成、几何/能量/对接评价；本项目另以 FLOWR predicted affinity 实施 Steer |
| 口袋准备 | FLOWR 标准预处理通常取参考配体周围 7 Å | 公开 pocket10 文件通常由 10 Å 截取；本项目控制器推理时仍传 `--cut_pocket --pocket_cutoff 7` |

CrossDocked 基于相关受体构象与配体之间的 cross-docking 扩充结构数据。常用公开预处理保留 RMSD < 1 Å 的姿态，并发布 pocket10 和固定 split 文件。TargetDiff 文档明确沿用 AR/Pocket2Mol 的 split，其推理测试索引是 0–99。当前目录的 100 个配对和文件哈希已经核对，但不能仅凭目录名和数量证明其逐项对应某个论文版本的固定 split；若进行严格复现，应与该版本 split 清单逐项比对。

来源：[TargetDiff 官方数据处理与划分说明](https://github.com/guanjq/targetdiff#data)、[AR 官方 pocket10 数据入口](https://github.com/luost26/3D-Generative-SBDD/blob/main/data/README.md)、[当前已审计输入](../crossdocked_steer_campaign_20261008/remote_inputs.json)。

两者的测试划分不是同一种协议，跨来源训练数据的重叠也需要按 PDB/口袋/配体核对。因此不能直接比较两个集合的平均 predicted affinity，并把差异完全归因于采样算法。HiQBind 的实验标注用于参考配体和预测头验证；新生成配体在两个集合上的亲和力仍是模型预测。

## 本轮配置与代码修改

- `steer_launcher.py` 定义共享默认值：`DEFAULT_STEER_SAMPLES=100`、`DEFAULT_STEER_BATCH_SIZE=20`；单任务与多 target 控制器共用。
- `TargetCampaignConfig` 与 `SteerLearningConfig` 默认变为 100/20；显式配置及 CLI 覆盖优先级保留。
- `generation_crossdocked100_steer.json`、多 target 示例、单任务示例同步为 100/20。
- `window_start=0.0`、`window_end=0.5`、`steps=100`、`seed=42`；100 个积分步的评分网格有 51 个窗口内事件。末次选择对应的 proposal 仍完整保存，之后无重采样完成到 t=1。
- 模式仍为单目标 SMC，不添加坐标梯度；100 候选包含构建失败和重复后代，不进行过滤后补采。
- 每完成 10 个 target 归档，共 1000 个候选槽位；归档验证后回收对应原始工作目录。并行 `.running/.finished/.error` 领取逻辑保留。
- 新服务器默认输出为 `/opt/MolSteer/generated_datasets/crossdocked100_steer_learning_s100_b20_w050`，与旧集合分开；历史 CK2/CLK3 显式配置不被重写。已经启动的任务按冻结配置运行，新默认不会回溯修改该任务。

100/20 参考官方 `scripts/generate_pdb.sl` 的数量与 batch；Steer 时长 0.5 按本轮要求显式应用。官方该模板自身未启用 Steer，论文尚未披露实际 B，不能将本轮配置宣称为作者完整实参的恢复。

## 验证

本地相关回归 45 项通过，覆盖多 target 控制器、并行领取、归档恢复、Steer 窗口/原始精度/失败对照、健康检查和生成批次。使用 CPU 记录夹具，没有用其冒充真实模型推理。CLI 帮助已显示100个槽位和20个共同竞争粒子。

远端已拉取包含本轮生成修改 `1051f37` 的提交，实际检查的 HEAD 为 `5df58d5c09e0fd5e816c73a61f7f57a66df81d2c`，并核验该 HEAD 包含 `1051f37`。真实 100 个 target 的输入哈希全部与冻结清单一致，父控制器及子配置均为 100 槽位/20 粒子/5 群体、窗口 `[0,0.5]`、100 步、seed42，尾程至1.0，归档仍每10个target。新输出目录在检查前后都不存在，未创建或启动新的生成任务。

机器可读记录：[本地检查](default100_local_validation_20261009.json)、[实际远端配置检查](default100_remote_validation_20261009.json)。配置检查验证参数传播、输入哈希和归档粒度，不代表新的 B=20 全流程 GPU 推理已经完成。
