# FLOWR.ROOT 独立群体与共同竞争粒子数：论文及源码核查

核查日期：2026-10-09。GitHub 固定版本：`739a33c2aab74d074dd24605b9a2e34dc809ae9c`。本次只阅读文献和源码，没有启动推理或更改已有生成任务。

## 1. 结论与证据边界

FLOWR.ROOT 原生 PDB 推理入口没有一个独立的 `independent_groups` 参数。每次 DataLoader 批次启动一次完整生成轨迹；启用 SMC 后，该批次内的粒子共同评分、归一化和重采样。批次之间没有 SMC 粒子交换。因此，共同竞争粒子数由实际推理 batch 大小决定，独立群体数由目标采样数、batch 容量、GPU 数据划分和补采次数共同决定。

论文用符号 **B** 表示共同竞争粒子数，但在已核查的正文、Supplementary Information、Peer Review、Reporting Summary、Source Data 和公开代码中，没有找到足以确定主要 Steer 案例数值 B 和独立群体数 M 的完整运行配置。**每目标生成 100 个、每策略生成 1000 个配体，不能直接解释成 B=100 或 B=1000。**

这里的“独立”指批次之间没有 SMC 交互。全局去重或多样性过滤可能影响最终保留的分子集合，因此不主张最终结果一定统计独立同分布。本次没有逐一检查 Zenodo/Google Drive 上所有原始生成数据包，缺失实参的结论限定于上述已检查材料。

## 2. 两类群体在代码中的位置

### 2.1 参数映射

以下映射针对单目标 `flowr.gen.generate_from_pdb` 及双口袋 `flowr.gen.generate_from_pdb_selective` 原生入口。

| 参数 | 作用 | 与群体数的关系 |
| --- | --- | --- |
| `--sample_n_molecules_per_target S` | 输出数量目标；标准 PDB 数据集首先复制该 target S 次 | 决定首轮待采样槽位数量；过滤补采时不等于固定总计算预算 |
| `--batch_cost C` | 推理 DataLoader 的 batch 容量 | 实际 batch 长度 B 是一次 SMC 共同竞争群体大小，末批可能小于 C |
| `--max_sample_iter K` | 输出不足时允许继续进行外层采样 | 控制补采上限，不是独立群体数 M |
| `--guidance_config` | 提供 SMC 是否开启、目标及时间设置 | 不单独规定群体大小；默认未提供时不启用 Steer |
| `--gpus` | GPU 数据处理/分配数量 | 不等同于 M，也不会自动把所有 GPU 粒子合成一个 SMC 群体 |
| `--num_workers` | CPU 数据加载工作进程数 | 不决定 SMC 群体数量 |

双目标入口的 argparse 默认值是 `sample_n_molecules_per_target=1`、`max_sample_iter=20`、`guidance_config=None`；`batch_cost` 没有显式数值默认值，省略时为 `None`。因此并不存在一个原生 CLI 默认的“20 群体”或“100 竞争粒子”设定。默认 `seed=42`、`gpus=8`、`num_workers=24` 分别是随机数、设备和加载设置，不是 B/M，也不能据此还原论文实验环境。

源码调用链为：

1. `flowr/data/dataset.py` 的 `GeometricDataset.sample_n_molecules_per_target` 复制 target。
2. `flowr/gen/utils.py:get_dataloader` 把 `args.batch_cost` 传给 `val_batch_size`。
3. `flowr/data/datamodules.py:test_dataloader` 使用普通 `DataLoader(batch_size=self.val_batch_cost)`，没有设置 `drop_last=True`。
4. 推理入口的 `evaluate` 遍历 DataLoader，每个 batch 调用一次完整的生成过程。
5. `flowr/models/fm_pocket.py:apply_smc_guidance` 在本 batch 的评分向量上计算 `softmax(dim=0)`，再执行 `torch.multinomial(..., num_samples=len(weights_combined), replacement=True)`。

实际单目标亲和力最大化的重采样可写成：

\[
w_i(t)=\frac{\exp(s_i(t))}{\sum_{j=1}^{B}\exp(s_j(t))},\quad
I_1,\ldots,I_B\sim\mathrm{Categorical}(w(t)),\quad
x'_i(t)=x_{I_i}(t).
\]

重采样一直保持 B 个槽位；B 不等于重采样后互不相同的分子或祖先数量。归一化分母只包含本 batch，不包含之前或之后的 batch。B=1 时即使开启这种重采样也无法形成粒子之间的选择。

双目标 `apply_selective_smc_guidance` 对同一批粒子计算两组评分，当前实现相加 `softmax(s_target)` 和 `softmax(-s_offtarget)` 后重采样。`torch.multinomial` 对这组非负权重按总和归一化。同一组 `selected_ids` 同步应用到两套口袋相关状态。**CK2α 与 CLK3 是同一群体的两种评价条件，并非两个独立群体。**

可复核源码：[PDB 单目标入口](https://github.com/jule-c/flowr_root/blob/739a33c2aab74d074dd24605b9a2e34dc809ae9c/flowr/gen/generate_from_pdb.py)、[双目标入口](https://github.com/jule-c/flowr_root/blob/739a33c2aab74d074dd24605b9a2e34dc809ae9c/flowr/gen/generate_from_pdb_selective.py)、[加载器及 guidance 配置](https://github.com/jule-c/flowr_root/blob/739a33c2aab74d074dd24605b9a2e34dc809ae9c/flowr/gen/utils.py)、[推理 DataLoader](https://github.com/jule-c/flowr_root/blob/739a33c2aab74d074dd24605b9a2e34dc809ae9c/flowr/data/datamodules.py)、[SMC 实现](https://github.com/jule-c/flowr_root/blob/739a33c2aab74d074dd24605b9a2e34dc809ae9c/flowr/models/fm_pocket.py)。

### 2.2 独立群体数的推导及其适用条件

标准 PDB 入口、单 target、单 GPU、第一轮完整遍历时：

\[
M_{\mathrm{first}}=\lceil S/C\rceil,\qquad
B_i=\min(C,S-iC),\quad i=0,\ldots,M_{\mathrm{first}}-1.
\]

| S | C | 第一轮群体构成 | 证据性质 |
| --- | --- | --- | --- |
| 1000 | 50 | 20 × 50 | 依据源码推导，非论文披露配置 |
| 1000 | 100 | 10 × 100 | 依据源码推导，非论文披露配置 |
| 1000 | 64 | 15 × 64 + 1 × 40 | 原生入口允许不足整批，非 GPU 运行结果 |
| 1000 | 80 | 12 × 80 + 1 × 40 | 同上 |
| 1000 | 125 | 8 × 125 | 同上 |

这只是第一轮，不是全部实验的固定 M。过滤使输出不足时，外层重新生成，新增群体。`while ... and k <= max_sample_iter` 从 k=0 开始，K 在这份实现中允许至多 K+1 个外层采样 pass；一个 pass 又包含多个 batch。必须记录实际 `(pass_id,batch_id)` 才能得到真实尝试的群体数。

多个 GPU 的数据分配、UI 的数据集补齐和不同生成入口会改变上述首轮算术的适用条件。不能把该式直接套到任何入口上。

## 3. 文献中可确认的使用案例

| 案例 | 文献明确披露 | 共同竞争 B | 独立群体 M |
| --- | --- | --- | --- |
| HiQBind，正文 Figure 3 | 每目标 100 个配体，共 278 个测试 target；比较 steering duration 0.3、0.4、0.5 | 已检查材料未披露数值 | 已检查材料未披露数值 |
| CK2α/CLK3，正文 Figure 6 | 单目标 CK2α 最大化、CK2α 最大化/CLK3 最小化；每种策略每口袋 1000 个配体 | 已检查材料未披露数值 | 已检查材料未披露数值 |
| Methods “Steering via Importance Sampling”，第 16 页 | 维护 B 个粒子，按评分权重从本批粒子中重采样 B 个 | 给出符号 B 和算法，无实验数值 | 没有独立群体 M 的实参说明 |

Figure 6 的“每口袋”输出描述不能证明双目标实验使用两组独立粒子。实现中两套口袋评价共享粒子及其被选择的索引。

来源：[论文 PDF](https://www.nature.com/articles/s41467-026-74130-9.pdf)、[Supplementary Information](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41467-026-74130-9/MediaObjects/41467_2026_74130_MOESM1_ESM.pdf)。同行评议文件、Reporting Summary 和 Source Data 使用之前下载的出版社原件核查，哈希存于配套证据 JSON。

Source Data 的 Figure 3 表包含 target、分子索引、评分和性质；Figure 6 主要为两目标 pIC50 和方法名；Figure 7 为 CK2α/CLK3 值。它们没有 population/batch/lineage ID，不能通过最终分子行数反推出竞争群体划分。

论文另有亲和力预测的多随机种子重复和补充材料中的多次生成重复；这些是其他实验的重复轴，不能直接当作 Steer 的独立粒子群体数。已检查材料也没有给出“固定 1000 预算下 20×50 对比 10×100”的作者消融结果。

## 4. GitHub 及发表代码归档中的具体配置

下列例子能证明参数如何控制批次。**它们默认没有启用 Steer，不能当作论文实际 Steer 实参。**

| 位置 | 明确设置 | 第一轮实际批次 | Steer 状态 |
| --- | --- | --- | --- |
| `scripts/generate_pdb.sl` | 单 GPU；S=100，C=20；`max_sample_iter=30` | 5 个 batch，每个 20；有效性/多样性过滤可能引发补采 | 命令未提供 `guidance_config`，默认关闭 |
| `examples/examples.ipynb` 的生成命令示例 | `N_MOLECULES=5`，`BATCH_COST=20`，`NUM_GPUS=1` | 1 个 batch，实际 5 个粒子，20 为容量上限 | 默认关闭 |
| `flowr_vis/worker.py` 的 SBDD UI | 默认 `n_samples=10`，`batch_size=25`；数据集补齐到 `max(n_samples,batch_size)` | 首个 dataset 为 25 个副本，首 batch 25，输出目标仍为 10 | `guidance_config=None` |

发表代码归档 `jule-c-flowr_root-9ff49e8` 中同样存在 PDB 模板 S=100、C=20、单 GPU，以及 notebook S=5、C=20。归档 MD5 为 `bf50284ef0ed38f4c64d2761ad3bcb1e`。这支持“这些是公开代码的配置案例”，仍不能证明论文 Figures 3/6 采用相同配置。

当前 UI 允许 `batch_size` 范围 1–200，是界面参数范围，不能当作 B=200 在任意口袋或 GPU 上都能运行的证据。Notebook 后面的 `AFFINITY_BATCH_COST` 是评分任务的 batch，不能与生成 SMC 的竞争群体混用。训练配置中的 batch cost 也不能直接用于解释推理粒子数。

来源：[PDB 生成模板](https://github.com/jule-c/flowr_root/blob/739a33c2aab74d074dd24605b9a2e34dc809ae9c/scripts/generate_pdb.sl)、[Notebook](https://github.com/jule-c/flowr_root/blob/739a33c2aab74d074dd24605b9a2e34dc809ae9c/examples/examples.ipynb)、[UI worker](https://github.com/jule-c/flowr_root/blob/739a33c2aab74d074dd24605b9a2e34dc809ae9c/flowr_vis/worker.py)、[发表代码归档记录](https://zenodo.org/records/20068203)。

## 5. 对 EvoMolSteer 设置的解释和更正

EvoMolSteer 的归档入口用 `samples` 和 `batch_size` 控制固定候选预算及群体大小：S=1000、B=100 是 10×100，B=50 是 20×50。这些是本项目的实验设置，不应标记为作者披露的设置。

`src/evomolsteer/generation/steer_launcher.py` 和 `steer_campaign.py` 目前要求 `samples % batch_size == 0`，用于整批候选与归档核验。**该整除限制不是 FLOWR.ROOT 原生要求。** 原生 S=1000、C=64 可得到末批 40；当前 EvoMolSteer 入口会拒绝这种配置。

已检查的原生推理路径没有把“SMC 共同竞争粒子数 B”与“GPU 计算 microbatch m”分成两个参数。把 B=100 保持为一个共同竞争群体、同时分成多个较小 GPU microbatch 计算，需要增加分块推理并汇总全体权重的代码，不能仅修改一个现有 CLI 参数而假设该能力已存在。

实验元数据应分别保存目标输出数/候选预算、batch 容量、每批实际 B、实际 pass/batch 数、GPU 分配、过滤配置、SMC 是否启用，以及祖先索引。只有这些信息齐全，才能判断“1000 个输出”实际来自多少个彼此独立的 Steer 群体，并复现与比较其搜索行为。

配套核查结果：[flowr_population_source_audit_20261009.json](flowr_population_source_audit_20261009.json)。其中保留源码版本、SHA-256、CLI 默认值、AST 函数位置、Notebook 常量、Source Data 表头及证据限制。
