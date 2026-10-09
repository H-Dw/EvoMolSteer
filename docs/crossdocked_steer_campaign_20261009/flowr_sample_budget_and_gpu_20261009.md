# FLOWR.ROOT 每目标 100 个配体、CrossDocked100 预算及 GPU batch 的解释

日期：2026-10-09。沿用源码版本 `739a33c2aab74d074dd24605b9a2e34dc809ae9c`。本次为文献、源码和本地既有失败证据分析，没有启动 GPU 推理或调整后台实验。

## 1. “每目标 100 个”具体指哪些目标

正文 Figure 3 是 HiQBind 测试集的亲和力 steering 实验。图注报告每个 target 100 个配体、共 278 个测试 target。比较条件包括无 Steer 和 steering duration 0.3、0.4、0.5；正文明确将其解释为基于 importance sampling 的 inference-time steering。因此，100 不是只有无 Steer 时才采用的输出规模。

这些 target 不是当前 CrossDocked 100-target 输入清单，也不是 CK2α/CLK3 双目标案例。这里的单位是测试集中的蛋白口袋/参考配体结构系统，不应直接等同于 278 种互不重复的蛋白。本文采用与 Plinder 一致的数据划分；HiQBind 是有实验结合信息的整理共晶数据集。

正文 Table 1 另外也有“每 target 100 个配体”，但对应 SPINDR 的 225 个测试 target，主要用于结构质量基准。不能把该表的输出数或硬件与 Figure 3 的 Steer 协议混为一体。CK2α/CLK3 Figure 6 报告的是每种优化策略每口袋 1000 个配体。

来源：[论文正文及图注](https://www.nature.com/articles/s41467-026-74130-9.pdf)、[补充材料中的数据集说明](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41467-026-74130-9/MediaObjects/41467_2026_74130_MOESM1_ESM.pdf)。

### 新发现：正文图注与 Source Data 的输入计数不一致

本次逐项计数出版社 Source Data 的 `final/Figure_3.csv`，只用 `subfigure=3a` 的数据核验：

| 条件 | 不同 target_id | 不同 target_id/分子索引对 | 每 target 评分行数范围 |
| --- | ---: | ---: | --- |
| no-guid | 297 | 28,947 | 86–100 |
| 0.3 | 297 | 26,516 | 45–100 |
| 0.4 | 297 | 26,509 | 44–100 |
| 0.5 | 297 | 27,987 | 57–100 |

这些 target_id 包含 148 个不同 PDB 前缀。例子有 `gen_ligs_1bcj_NGA_1_1`、`gen_ligs_1bcj_NGA_2_1`、`gen_ligs_1bcj_NGA_3_1`，以及 `gen_ligs_1fj4_TLM_A_500`、`gen_ligs_1fj4_TLM_B_501`。它们展示同一个 PDB 条目可以对应多个结构输入标识。PDB 前缀数不是已验证的独特蛋白数量。

图注的 278 与 CSV 的 297 不一致，原因尚未解释；不能擅自断言哪个是作者实际使用的最终 target 清单，也不能猜测差异一定由过滤产生。评分行数也不是每个条件每个 target 都恰好 100。无论是有效性过滤、评分导出还是其他处理导致减少，都需要原始日志才能确定。已保存完整 CSV target_id 清单及各条件评分计数，便于后续核对。

Source Data 在 3b/3c/3d 还重复展开同一分子的属性和投影，不能按整个 CSV 的行数计数分子。公开材料仍不提供 batch/群体划分信息，100 个配体不能证明共同竞争粒子数 B=100。

来源：[出版社 Source Data](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41467-026-74130-9/MediaObjects/41467_2026_74130_MOESM4_ESM.zip)。本地原件 SHA-256 为 `5f17ba1c6e6e30b031d908b94669aab062f71d38abed5bd8997712809065d2e3`。

## 2. CrossDocked100 是否应该从 1000 改成 100

CrossDocked100 中的“100”是输入任务数；每任务生成多少配体是另一个预算轴。1000 个候选是本项目为获取多群体 Steer 谱系、优势路径和失败对照采用的预算，不是论文给 CrossDocked 规定的数字。

应分别设置两种实验用途：

| 用途 | 建议数量策略 | 理由 |
| --- | --- | --- |
| 跨 target 的均值/中位数、有效率、应变等基准评价 | 每 target 100 个作为初始统一预算 | 更接近论文常用输出规模，降低总耗时和存储；仍需完整披露群体划分与过滤，不能称为严格复现 HiQBind |
| 学习 Steer 的区域富集与路径变化、搜索高分上尾 | 保留 1000 预算或按稳定性继续增加独立群体 | 更多独立选择历史可支持群体级重复、罕见路径和失败对照；最终分子不等于相同数量的独立谱系证据 |
| 资源受限的多 target 探索 | 首先每 target 100；对谱系/效应不稳定或值得继续探索的 target 追加完整群体 | 把计算放在信息不足的 target；统一基准评价与追加学习集分开保存，避免自适应预算偏置比较 |

数量 S 与群体 B 必须同时说明。例如 S=100、B=100 只有一个共同竞争群体；S=100、B=20 是五个群体；S=1000、B=50 是二十个群体。对于学习跨群体重复富集，一个 100 粒子群体提供的独立选择历史较少。

**减少 S 并不会在 B 不变时降低首批显存。** S=1000/B=100 改成 S=100/B=100，只是十批变为一批，首次口袋编码仍需要编码 100 个副本。本次已有 OOM 在首批编码阶段发生，因此这种总量修改不能解决其失败。降低 B，或将 GPU 计算分块与共同竞争 B 解耦，才直接针对这个峰值。

本项目记录固定候选槽位预算，包括未成功构建最终分子的失败结果；论文所说的 sampled ligands 和公开评分行数则不能自动等同于这个预算定义。比较需同时报告尝试槽位、成功构建数量、有效独特分子数和每群体实际 B。

## 3. 官方 GPU / 内存证据能说明什么

| 信息来源 | 明确内容 | 推断边界 |
| --- | --- | --- |
| 正文 Methods “Training and Inference” | 模型约 33M 参数；使用一个含八张 NVIDIA H100 的节点训练 | 未披露 Figure 3/6 推理单卡/多卡分配、具体精度或峰值显存 |
| 官方 `scripts/train.sh` | 注释一般使用八张 H100；`batch_cost=5`，注明适合 80GB GPU；`val_batch_cost=20` | 训练设置不是 Steer 粒子数；训练反向传播保留的激活与推理不同。是否启用 bucket sampler 也会改变 training cost 的含义 |
| 官方 `scripts/generate_pdb.sl` | 单 GPU、输出目标 100、推理 `batch_cost=20` | 最直接的公开推理批次案例，但命令未启用 Steer，不能证明论文 Steer 的 B |
| 官方 README | 单 H100 大约 15 秒生成 100 个配体，速度依系统和 batch 而变 | 没有同时给出该计时的 B、输入大小、Steer 和存储设置，不能反推出 B=100 |
| NVIDIA 官方 H100 规格 | H100 SXM 为 80GB；H100 NVL 为每 GPU 94GB | 论文只写 H100，没有完整声明具体变体 |

八张 GPU 的总显存不能直接视为一个普通 batch 可用的显存池。原生推理代码没有自动把八张卡的粒子纳入一个跨卡统一 SMC 群体的实现。80GB 可以用作结合训练脚本注释的估算场景，不能标记为论文已经披露的 Steer 推理峰值。

生成 SLURM 模板中的 `--mem-per-cpu=12G` 是主机内存资源申请，不是 GPU 显存。训练的 gradient accumulation 也不会自动扩大推理中一次共同竞争的粒子数。

来源：[训练脚本](https://github.com/jule-c/flowr_root/blob/739a33c2aab74d074dd24605b9a2e34dc809ae9c/scripts/train.sh)、[生成脚本](https://github.com/jule-c/flowr_root/blob/739a33c2aab74d074dd24605b9a2e34dc809ae9c/scripts/generate_pdb.sl)、[README](https://github.com/jule-c/flowr_root/blob/739a33c2aab74d074dd24605b9a2e34dc809ae9c/README.md)、[NVIDIA H100 规格](https://www.nvidia.com/en-us/data-center/h100/)。

## 4. 从坐标特征张量约束可能的 batch 大小

论文令 pocket/ligand 的 equivariant feature dimension 为 128，补充材料的标准数据准备通常使用参考配体周围 7 Å、10–800 个 pocket atoms。实际推理需要记录裁剪和去氢后原子数，不能直接用 PDB 文件行数或数据目录名称代替。

当成对距离特征开启且完整物化时，`pocket_util.py:_PairwiseMessages` 形成坐标差分张量：

\[
D\in\mathbb{R}^{B\times n_p\times n_p\times3\times d_{equi}}.
\]

假设 d_equi=128、每元素四字节（FP32），**仅这一个张量**需要：

\[
M_D = B n_p^2\times3\times128\times4\ \text{bytes}.
\]

| pocket atoms n_p | B=20 | B=50 | B=100 |
| ---: | ---: | ---: | ---: |
| 400 | 4.58 GiB | 11.44 GiB | 22.89 GiB |
| 600 | 10.30 GiB | 25.75 GiB | 51.50 GiB |
| 800 | 18.31 GiB | 45.78 GiB | 91.55 GiB |

这是条件算术，不是完整模型峰值。模型还会同时保存节点、边、拼接后的 pairwise features、MLP 输出、编码状态和其他张量。半精度使这个张量减半，部分运算若仍为 FP32 则不能照此估算全模型。其他特征开关、checkpoint 架构和代码优化也会改变内存。

在上述 FP32/完整物化场景下，800 原子口袋的 B=100 连这一个张量就超过一张 80GB H100；B=50 也已占据显存的大部分，不能认定必定可运行。B=20 的公开推理模板因此是比“论文生成 100，所以 B=100”更有依据的复现起点。较小口袋可能容纳 50 或 100，但需要测量全流程峰值；没有依据从 H100 型号唯一反推出作者 B。

若把论文的实际 B=20 暂作为工作假设，则 HiQBind 100 输出对应首轮五个群体，CK2/CLK3 1000 输出对应首轮五十个群体。这只能标注为基于模板的假设，不能标为论文真实配置。

当前服务器既有 ABL2 失败日志显示总显存 63.98 GiB；PyTorch 已分配 49.85 GiB，还要新申请 20.40 GiB，在口袋编码的特征拼接处失败。这与 B 和口袋成对特征相关，而非累计生成了第 1000 个分子造成。该日志不是本次重新检测服务器状态所得。

来源：[成对特征实现](https://github.com/jule-c/flowr_root/blob/739a33c2aab74d074dd24605b9a2e34dc809ae9c/flowr/models/pocket_util.py)、[既有 OOM 分析](oom_recovery_and_population_20261009.md)。

## 5. 对当前任务的建议

1. 为广覆盖基准设置独立的 100/target 配置，第一轮试验可从 B=20、五个群体起步。不是已测得的安全上限，也不是声称复现作者 Steer 实参。
2. 对当前面向 LLM 谱系学习的用途，不因图注的 100 就整体缩减到一个 B=100 群体。保留多群体预算；1000/B=50 可作为未分块实现下的候选方案，须先完成大口袋真实显存测试。
3. 如果希望保持 B=100 的共同竞争，优先实现计算 microbatch（如 20/25）并在完整 100 分数上统一归一化和重采样。该接口目前尚未接入，不能先当作已可用能力。
4. 如采用分阶段追加，统一 benchmark 使用每 target 相同预算；追加结果进入学习/搜索集合并标记原群体大小。保留现有结果与失败对照。
5. 要严格复现作者实验，仍需要作者补充 Figure 3/6 实际命令、群体大小、补采策略、输入 target 清单、精度和推理显存记录。

配套证据：[计数与张量算术 JSON](flowr_sample_budget_memory_evidence_20261009.json)、[Source Data target_id 列表](flowr_figure3_source_target_ids_20261009.csv)。
