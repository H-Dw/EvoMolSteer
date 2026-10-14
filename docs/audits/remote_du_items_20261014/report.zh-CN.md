# `experiments/du -sh *` 逐项来源与删除资格核查

审计日期：2026-10-14。远端路径：`/root/private_data/MolSteer/flowr_root/experiments/`。本次只读核查，没有执行删除。当前实际发现 **26 个目录、6 个顶层交付 ZIP**；比用户给出的清单多了一份 `CK2_Steer_Selection_Mechanism_20261014.zip`（约 78.13 MiB）。

**第一批建议清理的是重复压缩包，不是原始轨迹或正式统计。** 已逐一验证六份顶层 ZIP 的远端 CRC，并计算远端/本地 SHA-256，全部与 `D:/llm_controlled_generation/outputs/` 内的本地 ZIP 一致。再加上目录内部的已备份重复交付包和转移包，共有 **14 个明确目标，逻辑大小 476.65 MiB**。目标路径、源 SHA 和本地备份路径逐项记录在 [verified_duplicate_candidates.json](verified_duplicate_candidates.json)。

本报告的 MiB 是实际字节除以 1,048,576；用户 `du -sh` 显示的是文件系统占用并进行粗略取整。库存计算排除了 `.git` 与可重建的 `__pycache__`，因此目录数字可能略小于 `du`。释放空间还会受文件系统计量影响。

## 一、原始数据及七个 CK2 分析目录

这些是对历史 Steer 的重新测量和统计，**不是七次独立的新分子生成，也不是模型 checkpoint**。Quality67、质量组内 Top10、Top10 独特图/独特根等有各自冻结成员规则，不能与最近 guidance 实验“发现集终态 Top10 教师”混为一套选择。

| 用户列出的目录 | 对应工作 | 删除建议 |
|---|---|---|
| `ck2_clk3_lineage_20261003`，1.1G | 原始 CK2/CLK3 实验：single、joint、unguided 的轨迹、评分、结构、真实父子索引和失败记录 | **保留整目录**。是后续教师、谱系和规则挖掘的来源；`.pt.gz` 中还有预测状态，不应误当失败模型权重删除 |
| `ck2_original_steer_analysis_20261013`，325M | 原 single-target Steer 的全套再分析：候选质量、67 质量样本、Top10、几何/接触/化学富集、PCA、51 事件谱系和图表 | **不删整目录**；保留权威 `features/`、`tables/`、`candidates/`、脚本和报告。内部约 **178.50 MiB** 是可清理的重复 ZIP/转移包，详见下一节 |
| `ck2_candidate_birth_mechanisms_20261014`，168M | 961 个有效终态中，Quality67/Top10 的诞生与成熟过程、分支差异和同胞对照 | 可先删除 `history_transfer.zip`，**34.62 MiB**。正式包已备份；60 个逐批提取分片没有逐文件本地副本，暂不建议整目录删除 |
| `ck2_final_spatial_analysis_20261014`，22M | Quality67/Top10 终态的三维结构、残基邻域、键角及组成调整对照 | 可先删除根部 `extracted.zip`，**2.68 MiB**；它与 `inputs/extracted.zip` 内容完全相同。100 个逐批提取分片未逐文件备份，暂不整目录删除 |
| `ck2_quality_advantage_audit_20261014`，2.5M | 质量组的真实终态优势与选择评分、重采样权重及反事实评分核对 | **可迁出并删除远端工作副本**：35 个正式文件全部有 SHA 一致的本地备份。删除前需确认后续远端任务不再引用；收益约 **2.43 MiB** |
| `ck2_quality_process_features_20261014`，26M | 67 质量样本与其他样本/同根对照的过程坐标、运动、接触、变化率及 PCA | **可迁出远端副本**：79 个正式文件全部备份，约 **25.61 MiB**；继续在远端重跑时要恢复数据和脚本 |
| `ck2_steer_selection_mechanism_20261014`，164M | single/joint/unguided 各 1,000 次尝试的实际选择偏好、复制权重、未来后代信用及留出系统统计 | 可先删除 `pool_transfer.zip`，**54.15 MiB**。80 个逐批提取分片及一份小日志未逐文件备份；保留合并池表、信用表、脚本，不建议整目录删除 |
| `ck2_top10_process_analysis_20261014`，37M | 质量组内 Top10 独特图、普通 Top10、Top10 独特根的真实路径分析，含同 seed/同根/其他 seed 对照 | **可迁出远端副本**：63 个正式文件全部备份，约 **36.61 MiB**；这不是重新生成一批 Top10 |

三个完整备份的分析工作副本合计 **64.65 MiB**。如果未来分析改在本地执行，它们可以从远端移除；如果仍要在远端复用这些统计脚本，应保留或先修改数据接口再恢复需要的输入。

本地质量优势审计目录实际名为 `outputs/ck2_quality_advantage_audit_20261013`，而远端工作副本带 `20261014`。本次按内容 SHA 验证了全部文件，不仅按目录名称判断。

原 Steer 分析目录的正式科学文件都在本地有匹配副本，仅八个运行日志/退出码/PID 小文件未匹配。但是当前工程确实读取该分析的 `tables/terminal_all_features.csv`、`tables/high_quality_candidates.csv`、`features/pocket.npz` 和逐批 node 特征，因此默认仍保留其权威展开数据，只清理内部重复包。

三个目录中尚未逐文件备份的提取分片都属于 `extracted/` 下的逐批输出；合并表和正式交付物已有备份。**不能据此直接删除整个 `extracted/`**，它还包含分析入口使用的合并文件。若要清理分片，应另做合并行数、字段和重建一致性核验；本次没有将这些分片写入低风险删除清单。

## 二、顶层 ZIP：全部是本地已有的交付副本

| 远端文件 | 对应分析 | 实际 MiB | 判断 |
|---|---|---:|---|
| `CK2_Candidate_Birth_Mechanisms_20261014.zip` | 候选诞生机制 | 84.83 | 可删除远端重复 ZIP |
| `CK2_Final_Spatial_Analysis_20261014.zip` | 终态空间分析 | 10.61 | 同上 |
| `CK2_Quality_Advantage_Audit_20261014.zip` | 质量优势核对 | 1.72 | 同上 |
| `CK2_Quality_Process_Features_20261014.zip` | 质量组过程特征 | 16.43 | 同上 |
| `CK2_Top10_Process_Analysis_20261014.zip` | Top10 路径分析 | 14.99 | 同上 |
| **新增** `CK2_Steer_Selection_Mechanism_20261014.zip` | Steer 选择机制 | 78.13 | 同上，本次现场新增于所贴清单之外 |
| **合计** | | **206.70** | 全部远端 CRC 通过，全部与本地 ZIP SHA-256 相同 |

这里的“可删除”只指远端重复副本；本地 `outputs/` 的交付 ZIP 和正式结果保留。

## 三、目录内部可以优先删除的重复包

以下也是逐字节找到本地备份的文件，不依赖重新运行分子生成：

| 相对 `experiments/` 的路径 | MiB | 保留的权威内容/副本 |
|---|---:|---|
| `ck2_original_steer_analysis_20261013/CK2_Original_Steer_Analysis_20261013.zip` | 129.30 | 本地同名完整 ZIP；远端正式展开结果继续保留 |
| `ck2_original_steer_analysis_20261013/features_bundle.tar.gz` | 48.07 | 与 `transfer_archives/initial_extraction_before_SMARTS_QA.tar.gz` 相同，也有本地匹配文件 |
| `ck2_original_steer_analysis_20261013/binding_bundle.tar.gz` | 0.88 | `transfer_archives/binding_transfer_snapshot.tar.gz` 及本地文件 |
| `ck2_original_steer_analysis_20261013/candidates_bundle.tar.gz` | 0.03 | `transfer_archives/candidates_transfer_snapshot.tar.gz` 及本地文件 |
| `ck2_original_steer_analysis_20261013/corrected_chemistry_bundle.tar.gz` | 0.22 | `transfer_archives/imidazole_SMARTS_correction_overlay.tar.gz` 及本地文件 |
| `ck2_candidate_birth_mechanisms_20261014/history_transfer.zip` | 34.62 | 本地 `inputs/history_transfer.zip`；正式合并输入/输出保留 |
| `ck2_final_spatial_analysis_20261014/extracted.zip` | 2.68 | 同目录 `inputs/extracted.zip`，以及本地匹配副本 |
| `ck2_steer_selection_mechanism_20261014/pool_transfer.zip` | 54.15 | 本地 `inputs/pool_transfer.zip`；正式合并池表继续保留 |
| **合计** | **269.95** | 与顶层 ZIP 清理不重叠 |

早期 `features_bundle` 是 SMARTS 修正之前的提取快照；保留历史快照用于审计，但**它不能替代最终已修正的 `features/`**。删除根部重复包不会把科学版本回退。

## 四、18 个 EvoMolSteer 目录逐项对应

| 用户列出的目录 | 对应实验/功能 | 删除建议 |
|---|---|---|
| `evomolsteer_affinity_geometry30_20261007`，500K | 历史 30 轮亲和力/几何奖励探索，R26 的前置来源 | 大载荷已清理。保留小报告；备份后可退休，但收益很小 |
| `evomolsteer_agentic_steer_20261014_v1`，330K | 真实 Agent 的中间状态选择与原生续生成 C0–C6/B_diversity | 保留首轮与失败证据；不是可删模型 |
| `evomolsteer_dependency_20261009`，59M | R26 信息依赖消融、I3/I4/M2、早期 API 基线；约 52 MiB 是生成归档 | **暂不整目录删除**。仍有旧实验结构归档，本次没有核验它们全部具备本地归档副本；删除前应先备份，不因某轮评分低就删唯一结构 |
| `evomolsteer_dependency_compact_skills_20261012`，494K | 精简 Skills 后 M2 的 400 样本部署复验 | 已只剩小记录。保留部署、接口和重放证据 |
| `evomolsteer_dependency_full_drop_20261012`，545K | 仅完整/删除的早期消融执行器 | 已被工具版取代，可在备份日志后退休；空间收益很小 |
| `evomolsteer_dependency_scientific_20261011`，675K | 早期细化模块完整/删除/替代及接口失败记录 | 同上；保留作为接口失败证据，不将报错当模块科学负效应 |
| `evomolsteer_dependency_temporal_20261013`，729K | W/S/C 九种数值程序、十个真实 API 条件及证据重试 | 大数据已清理。保留 S3/W1/S1 等比较和 Agent 选择记录 |
| `evomolsteer_dependency_tools_full_drop_20261012`，3.3M | 工具式真实 API、17 子模块消融、固定对照复用、最终 Skills 选择 | 正式消融证据，建议保留；不是训练权重 |
| `evomolsteer_dynamic_contrast10_20261008`，98K | 10 轮动态正负样本/对比特征探索 | 仅小报告，保留或在备份后退休 |
| `evomolsteer_elite_path20_20261007`，215K | 20 轮优质路径搜索探索 | 同上 |
| `evomolsteer_flowcompat30_20261009`，354K | 30 轮真实 FLOWR Jacobian/flow compatibility 探索，R11 来源 | 保留最佳版本来源和失败回溯；大数组已清理 |
| `evomolsteer_flowcompat_scale1000_20261009`，122M | R26/R11/native 各 1,000 的匹配验证；约 92 MiB 归档，其余含固定源码 snapshot | **暂不整目录删除**。是重要基准和零引导对照；须先核验本地完整归档。过期源码 checkout 可单独退休，不动基准记录 |
| `evomolsteer_motif_seed42_20261006`，231K | 早期 motif/几何奖励探索 | 小型历史报告，保留或备份后退休 |
| `evomolsteer_online_20261004`，23M | 早期在线引导工作区，**同时保存当前推理用 `runtime_deps`** | **不能删除整个目录**。当前 generation/temporal/Agentic 等入口的 PYTHONPATH 指向其中 h5py 等依赖。根部安装 wheel 可另行审计；本次不列为删除目标 |
| `evomolsteer_selection_path20_20261008`，211K | 20 轮选择压力/路径特征探索 | 小报告，保留或备份后退休 |
| `evomolsteer_skill_ablation_20261007`，29K | 最早的 Skills 指引消融 | 小报告，不值得优先清理 |
| `evomolsteer_terminal_outcome15_20261010`，260K | 15 轮终态信用/家族/尾部标签探索及 R26/R11 对照 | 大数组已清理。保留 TO07/10/11/13 回退和验证报告 |
| `evomolsteer_terminal_regions_20261014`，57M | 终态 Top10 全局/局部 × 两窗口，每组 400，另有桥接 | 本地已收集结构与执行证据。此前核验的三个退步方案远端重复归档可单独清理 **30.55 MiB**；保留状态和报告。其余归档可按相同 SHA 策略再核验，而非直接整目录删除 |

本次没有把这些旧目录的小日志逐文件都做本地备份验证，所以“小报告目录可以退休”是**备份后**的条件判断，不是已经验证可以直接无损删除的清单。按占用优先级，先处理重复 ZIP 的收益远大于删除失败报告。

## 五、建议的清理顺序

1. **先清理 14 个已核验重复包：约 476.65 MiB。** 正式统计、失败对照、原始轨迹和运行依赖保留。
2. 如果远端以后不再承担这三个分析任务，可迁出质量优势审计、质量过程特征、Top10 过程分析目录，额外约 **64.65 MiB**；后续复跑要从本地备份恢复。
3. 最新 guidance 的三个失败/退步方案远端归档按上一轮已核验清单清理，额外约 **30.55 MiB**。保留本地完整归档及远端结果报告。
4. 再研究逐批提取分片和旧源码 checkout 的退休。不要一次删除整个 `extracted/` 或 `runtime_deps/`。

前三项若全部实施，逻辑载荷合计约 **571.85 MiB**，没有重复计算。同一原始 trajectory、模型权重、当前依赖、正式结果表均不在这三个清理项中。**本次只给出了审核结果，没有执行删除。**
