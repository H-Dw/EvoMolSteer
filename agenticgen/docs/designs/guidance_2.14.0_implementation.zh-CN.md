# Guidance 2.14.0：早期路径探索与 condition 对齐评价

本次交付为代码、英文 skills 和独立实验配置。没有启动远程正式实验，也没有调用真实提案模型。旧 release 文件不修改；新协议必须使用新的结果目录和重新测量的标定数据。

## 运行流程

确定性标定 → 单个 Guidance 提案角色 → 程序提交验证 → 按原顺序执行全部合法提案 → 以完整 50 候选的 condition-aligned RMSD 算术均值更新最佳。

`2.14.0` 不加载 box Selection 或 synthesis 技能，也不调用这两个模型角色。`DECISION.json.selected` 仅为下游兼容保留，由程序从验证后的提案 ID 生成，并记录 `execution_source=deterministic_all_proposals`。不存在伪造的 synthesis 回执。提案不合格时沿用一次有界修正，修正消息明确要求 `proposals`；再次失败保存会话证据，不删项、不改写数组。

执行前再次校验父节点、完整数组、优化目标、真实证据 ID、路径焦点、重复配置、方向差异及累计预算。新协议拒绝手工替换初始化根和导入旧标定成绩。只运行 `GD01` 的五轮搜索。

## 配置与标定

配置覆盖文件：[`uniguide_early_path_search.json`](../../configs/uniguide_early_path_search.json)。将其顶层字段覆盖到完整的 absolute-coordinate 基础配置上，并为 `root` 指定新的独立目录。模型、condition 输入和生成上下文路径使用实际部署环境的值。此覆盖文件不包含 SSH 凭据，也不会自行提交实验。

| 项目 | 2.14.0 设置 |
|---|---|
| 存储窗口 | `[0,10]、[11,20]、[21,30]、[31,40]、[41,50]、[51,99]` |
| 实际额外引导 | 步骤 1–98；51–98 共用末窗 scale；0/99 沿用原生端点处理 |
| 首窗网格 | `0,1,2,5,10,20,…,400`，首窗形状固定为 constant |
| 其余窗口网格 | `0,10,20,…,400` |
| 距离 | 逐窗网格索引距离；单窗最大 43，总距离最大 243 |
| 标定 | 每教师最多 10 个完整配置 × 50 = 500 候选 |
| 正式搜索 | 每教师 5 轮 × 最多 5 配置 × 50，上限 1250 |
| 失败 | 消耗预算，不补样、不重试该配置；数值失败保留可检索的配置身份 |

每位教师先测量两个背景：

| 教师 | 背景一 | 背景二 |
|---|---|---|
| 6pyy | `[50,70,70,50,50,70]` | `[50,100,40,40,30,50]` |
| 3w54 | `[30,40,40,40,60,80]` | `[30,40,110,110,200,200]` |

选出对齐均值较低的有效背景后，将其固定为参照，分别改变 B1、B2、B3、B4 的相邻低、高合法剂量，形成八个配置。边界处使用另一侧最近的两个未测剂量。补充测量不会逐项更新参照。程序从全部有效完整测量中选出标定根；两个背景均不可排名时，该教师停止，其他教师仍可标定。未用完的标定预算不转入正式搜索。

## 英文技能与路径状态

release 目录：[`2.14.0`](../../src/agenticgen/resources/skill_releases/guidance/2.14.0)。三个技能具有 YAML 名称、触发描述和 `metadata.version`：

| 模块 | 技能入口 | 实际 runtime 文件 |
|---|---|---|
| History and Evidence | `jump-guidance-history/SKILL.md` | `analysis.md` |
| Path Exploration and Proposal | `jump-guidance-proposal/SKILL.md` | `proposal.md` |
| Evidence Computation | `jump-guidance-evidence-computation/SKILL.md` | `computation.md` |

正文明确收益目标、前移与回溯、独立窗口及证据边界；方法参考和扩展示例保存在 supporting references。采用 [Tree of Thoughts](https://arxiv.org/abs/2305.10601) 的分支与回溯概念，以及 [LATS](https://arxiv.org/abs/2310.04406) 的外部反馈思路。这里的节点是实测完整参数配置，不是中间分子状态；没有增加模型自评分，也没有实现 MCTS。

`search_focus` 包含零起点的 `window_id`、`operation` 和实测 `source_id`。首轮关注 B1/B2，使用 `start`；之后用 `advance`、`refine`、`backtrack` 表达前移、停留和回溯，程序检查操作与窗口变化一致。焦点不会锁定其他参数。

每轮优先提出五个配置，至少两个主动改变各自真实父节点的 B1–B4。程序把前四窗的网格变化编码为四维增减符号：例如 B1 上调一格与上调两格属于同一方向；B1 上调与下调、B3 单独改变与 B4 单独改变属于不同方向。继承自父节点的前缀差异和纯后缀改动不能计为第二个早期方向。预算不足或合法候选不足时，审计回执记录剩余预算、精确候选数量、方向数量和豁免原因；有限建议列表的缺失不能证明搜索空间耗尽。

上下文按前四窗前缀分组，给出已测后缀数、最好均值和来源，突出三个不同前缀的代表，同时保留完整历史。连续两轮无改善（包括全失败轮）触发重新审查最佳路径和支线；兼容的历史非赢家仍可作为父节点。技能要求支线有具体续测机会，并区分首次借用旧后缀和已有后缀适配证据，避免一次失败永久封禁，也避免无依据的无限延续。

[`build_guidance_214.py`](../../scripts/build_guidance_214.py) 从技能入口同步 runtime 副本并生成哈希，默认仅校验。三个模块逐字节一致，参考文件也打包并复制到实验目录。发布后不再使用 `--sync` 修改此 release，后续修改使用新版本。

## 评价定义与结果字段

沿用现有终态 Hungarian 匹配确定 condition 对应点，然后固定对应关系。仅使用这些 condition 点做中心化、proper Kabsch 旋转拟合和 RMSD 残差计算，使用 float64；不允许反射、缩放或对齐后重新匹配。

对每个候选计算拟合后的 RMSD，再对全部 50 个值取算术均值。非 condition 坐标不进入拟合和误差求和；但原始终态匹配本身仍从现有有效生成原子中寻找对应点，因此不能将“固定匹配后的拟合不受非 condition 坐标影响”扩大为“整个匹配过程完全不受其影响”。

| 字段 | 含义 |
|---|---|
| `condition_evaluation.policy` | `condition-kabsch-mean-v1`，绑定配置、domain、采样兼容性、测量、checkpoint 和报告 |
| `condition_aligned_rmsd_angstrom` | 候选级拟合 RMSD；aggregate 中保留其总体均值 |
| `mean_condition_aligned_rmsd_angstrom` | 完整 50 候选的排名均值 |
| `protein_frame_pose_rmsd_angstrom` | 原 protein-frame 位姿误差，仅作诊断 |
| `coordinate_loss_statistics` | 完整数目、均值、无截尾标识和排名资格 |
| `ranking_measurement` | 最终报告中最佳配置的有效测量摘要，与搜索是否包含失败分开记录 |

旧通用 `ranking_condition_coordinate_root_mean_square_deviation_angstrom` 字段在新协议内同步为对齐均值，必须与新指标及策略标识一致。新协议缺少有效对齐指标时不可排名，不回退旧位姿指标。混用评价策略、七维数组、关闭窗口或不匹配的 domain 都会被拒绝。

标定、工具比较、checkpoint、回溯判断和最终最佳选择使用相同均值；配对比较也使用全部配对候选的均值差。分子构建失败但终态坐标合法的候选仍计入。缺失、非有限测量和不足 50 个候选不能通过删除样本成为可排名总体。新配置移除旧截尾控制，不引入恢复率阈值排名；有限的更新幅度诊断不替代终态收益。guidance loss、梯度、逐步匹配、化学处理和原生积分不变。

## 验证记录

新增测试：[`test_early_path_search.py`](../../tests/unit/test_early_path_search.py)。覆盖固定对应的旋转/平移不变性、非 condition 扰动、反射、退化点、缺失/非有限测量、未截尾均值、与旧位姿指标相反的排序、原生分片合并、双背景与八邻居、预算和失败记账、早期方向、历史父节点、前移/停留/回溯、严格提案修正、单角色调用和五轮全部执行。

同时运行原线性/开放域搜索、六窗/七窗协议、批处理、化学状态、端点、工具及严格提交的相关回归。技能通过 `skill-creator` 格式校验，runtime 副本与 release 清单校验；旧 2.12.0、2.13.0 清单保持可验证。

2026-11-03 本地组合回归：**239 passed**（其中新协议 19 项）。使用 Python 3.12、CPU PyTorch 和本地冻结 FLOWR 源码；测试采用短临时路径以避开 Windows 路径长度限制。wheel 构建成功，包内 2.14.0 的全部 9 个清单文件哈希一致，未包含本版本的 Selection/synthesis 模块；仓库内 17 个采用 `files_sha256` 的 release 清单全部通过核验。

测试为本地 CPU、模拟 provider 与合成坐标验收，未证明新的科学收益或真实模型的探索质量。远端 GPU 验收和两位教师的新正式实验另行启动。
