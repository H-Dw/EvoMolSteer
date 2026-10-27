# UniGuide–FLOWR 实施与启动记录

记录时点：2026-10-26 03:21（Asia/Shanghai）。这是启动快照，后续进度以远端产物为准。

实现位于独立分支 `codex/uniguide-flowr-agent`。本地工作树为 `D:\llm_controlled_generation\EvoMolSteer-uniguide`，原工作树已有的修改未改动。科学协议见 [实施协议](20261026_uniguide_flowr_implementation.zh-CN.md)，机器可读验收及启动证据见 [远端快照](../evidence/uniguide/20261026_remote_acceptance_and_launch.json)。

## 已实现

- 按准确 `node_id` 提取最终确认的两位教师；核查 SDF 成员哈希、已有分数及 native 槽位映射。两位教师分别含 17、22 个重原子，各有 9 个 PLIP 直接互作重原子，未扩展一跳邻居。
- 使用指定 `flowr_root_v2.ckpt`，已上传到指定路径并验证 SHA256。未使用服务器原有的其他版本权重。
- 逐步构造 detached Hungarian condition map，在干净终态预测上计算坐标损失，通过当前前向对带噪输入坐标求梯度，再按原生 FLOWR score/velocity 关系注入坐标增量。模型参数冻结，类别采样与 self-conditioning 保留原生路径。
- 实现四阶段共同尺度标定、三私有 Guidance Agent 与汇总选择、UG01–UG20 版本快照、当前 GPT 主 Agent 复盘接口、停滞研究触发和连续两次下降后的最优 UG skills 回退。
- 记录尝试预算、输入和输出哈希、逐步梯度及增量向量、随机流、质量检查和结构保留清单。数值失败保留证据并计入预算，没有自动减弱尺度或补样。

## 远端验证

执行环境为用户指定服务器的 `/workspace/EvoMolSteer`。仅在远端运行测试，使用现有 Torch 2.9.0 / DTK 环境。补充依赖 h5py 3.11.0 隔离安装于 `/workspace/uniguide_deps`，未升级 Torch 或 DTK。

初次记录时，14 项测试已通过：验收入口运行 11 项；随后新增的 3 项版本冻结、UG20 终止和 Agent 提案选择契约测试单独在同一远端环境通过。三项新增测试使用替身 provider 验证接口约束。后续通过了真实 DeepSeek API 检查，并补充了部分批次的回归测试，见后续记录。

两位教师的真实模型验收均通过：

| 检查 | 6pyy | 3w54 |
|---|---|---|
| 无引导增量的原生短轨迹等价 | 状态、self-conditioning、时间、随机流完全一致 | 完全一致 |
| step10 前恢复并继续到 step12 | 上述四项完全一致 | 完全一致 |
| 真实输入梯度方向有限差分相对误差 | 0.27%、1.09% | 0.55%、0.91% |
| 模型参数冻结 | 通过 | 通过 |
| 固定物理批内分组一致性 | 通过 | 通过 |

3w54 的原生 50 行网络调用曾在工程验收中超出显存。因此保持逻辑人口与原生随机数协议，改为固定 10 行网络前向、pocket 编码和 VJP；不足 10 行时填充并丢弃填充输出。比较固定物理批与拆分逻辑批后结果一致，没有放宽数值容差。原始失败日志均保留。

初次科学引擎提交为 `24b863c671a4e1f9862754d23cd91cb3e046a599`。新增契约测试提交为 `78eb5f64636ebf7a6f39a6d2776dba95b25fc81f`，未改变科学引擎。后续窗口阶段因工程错误停止；经审计、修复和重新验收后，从固定提交 `ebd0ee83e5701e759c3a0551acc680241bae1cd8` 继续，详见末节。

## 实验状态与继续入口

结果根目录为 `/workspace/agenticgen_uniguide_flowr_20261026`。标定进程已启动；记录时点已完成 17 个粗标定配置（170 个候选），下一个配置运行中，已预留 180 次尝试。R0 与正式 UG 版本尚未运行。这些是粗标定结果，尚不能据此确定共同初值或报告优化收益。

查看状态与日志：

```bash
cd /workspace/EvoMolSteer
bash agenticgen/scripts/run_uniguide.sh status
tail -n 30 "$(cat /workspace/agenticgen_uniguide_flowr_20261026/calibration_pipeline.logpath)"
```

标定完成后，满足 API 条件才启动正式版本：

```bash
bash agenticgen/scripts/run_uniguide.sh run-version UG01
```

每版完成后写入 `AWAITING_GPT_REVIEW.json`。当前 Codex 主 Agent 应实际读取全部指定证据，必要时研究文献和代码，在本地更新 skills 后提交并部署，再用 `accept-review` 绑定修改和证据。此接口不会伪造 GPT 复盘或自行声称完成文献检索。

启动快照时，非交互 SSH 进程未读取到 `DEEPSEEK_API_KEY`，无凭据的 API 连通性检查超时。后续核查已确认：服务器早已配置密钥和代理，原先将进程未继承变量表述为服务器缺少配置不准确。具体原因与修复见下节。没有创建新的本地 API 代理，标定始终独立运行。

GitHub 直连及四个指定 mirror 均依序超时。代码先在本地成功提交和推送 GitHub，再通过 SSH 传输已推送提交的 Git bundle，远端执行 `git bundle verify` 并检出固定提交。凭据未交给镜像，也未写入仓库。

## 证据路径

- 当前 GPU 验收：`acceptance/runs/1792955229428978455/`
- 新增契约测试：`acceptance/additional_contract_tests.log`
- 已验证教师与初始状态：`teachers.json`、`TEACHERS_PREPARED.json`、`INITIALS_VERIFIED.json`
- 标定完整结果及数值失败：`calibration/`
- 尝试预算：`attempts.json`
- 静态启动快照：`implementation_status_snapshot.json`

上述路径均相对于远端结果根目录。保留所有现有产物，没有执行结构清理。

## SSH 环境加载问题已定位

实际连接端点为 `root@ssh.zzai.scnet.cn:11741`，服务器主机名为 `crdnotebook-2114384248860155906-daweihuang-76949`。外部端口经服务端转发，服务器内部 `SSH_CONNECTION` 的目标端口为 22，不能据此判断连接了另一台服务器。

`/root/.bashrc` 第 6 行为 `[ -z "$PS1" ] && return`；密钥的 export 位于第 120、129 行。Paramiko `exec_command` 创建非交互 shell，普通 `bash -c` 与 `bash -lc` 都没有绕过该判断。因此执行进程既没有继承 `DEEPSEEK_API_KEY`，也没有继承已有的小写 `http_proxy`、`https_proxy`。仅改为 login shell 不足以解决问题。

实测 `bash -ic` 与 `bash -lic` 均能读取密钥；在交互式 shell 中再加载 DTK 环境后，密钥和小写代理变量仍然存在。使用该环境对 `https://api.deepseek.com/models` 发起带鉴权请求，返回 HTTP 200，模型列表包含 `deepseek-flash`。检查只输出是否存在、HTTP 状态及模型可用性，没有输出密钥或代理凭据。

启动脚本现已在 `run-version` 缺少密钥时，通过一次交互式 shell 加载现有配置并重新执行自身；有显式传入的密钥时保留调用方环境。如果加载后仍缺少密钥则明确失败，避免递归重试。此变更只涉及启动环境，不改科学引擎或正在运行的标定。

真实 provider 进一步使用实验配置调用 `deepseek-flash`，成功返回符合 schema 的 `{"ok": true}`，回执为 `backend=real_api`、`finish_reason=stop`，消耗 346 tokens。诊断证据见 [环境检查与真实调用](../evidence/uniguide/20261026_environment_diagnosis.json)。原先关于重新配置密钥和新增代理的询问已无必要。GitHub fetch 在加载现有代理后仍超时，因此部署仍通过已经校验的 Git bundle 完成。

## 25 样本窗口阶段的工程修复

检查启动环境时发现，粗标定和细标定已经结束，共 58 个配置产生有限结果、18 个配置发生已记录的数值失败。随后窗口标定的第一组 `calibration/windows/6pyy_H7S_A_402/p000` 因 `IndexError` 停止。该组已预留 25 次尝试，尚未完成首个积分步。

原因是 native 输出的 `fragment_mode` 按样本存为列表，而拼接器把它当成各批长度固定的结构列表。25 个样本分为 10、10、5 行，最后一批触发越界。修复明确区分逐行元数据与结构列表，同时使末尾批次的输入 VJP 也使用固定 10 行物理形状。

补充 GPU 检查还发现：不同逻辑人口会将同一随机种子的样本放到物理批中的不同位置。6pyy 在首个有效引导步之前的坐标、endpoint 和噪声完全一致，首个 VJP 的最大绝对差为约 `7.63e-6`，随后干净预测的差异可累积到约 `4.0e-4 Å`。因此跨人口的 VJP 使用原有 `atol=2e-5, rtol=2e-5` 数值比较；随机流、引导前轨迹以及同一人口的 checkpoint 恢复仍要求逐位一致。新增检查最初错误地要求不同人口在后续所有状态上逐位相同，两次失败验收的原始证据均已保留，没有把它们标记为成功。

新增远端回归测试覆盖 25 行的逐行元数据、两项时间张量和梯度。完整验收同时覆盖两位教师的 25 行短轨迹、首次 VJP 数值一致性及同人口精确恢复。失败科学组须经工程审计后标记失败并继续下一组，其 25 次预算保留，不重新执行或补采该组。既有粗、细标定使用完整物理批，采样张量计算不受此修复影响；已有结果继续保留。

2026-10-26 11:38（Asia/Shanghai）完成重新验收并启动续跑。15 项测试及两位教师的全部 GPU 检查通过，25 行检查首次 VJP 的最大绝对差分别为 `7.6294e-6` 和 `1.5251e-5`，均在既有数值容差内；引导前轨迹、随机流及同人口 checkpoint 恢复均逐位一致。证据目录为 `acceptance/runs/1792985580497988164/`，审计为 `engineering_audits/partial_batch_window_p000.json`。

续跑命令顺序为 `calibrate && run-version UG01`，仅在标定成功后启动 UG01。该版本结束后停在 `AWAITING_GPT_REVIEW`，等待实际主 Agent 复盘。当前日志为 `calibration_to_UG01_20261026_113842.log`；最初失败日志仍完整保留。详见 [验收、失败预算审计及续跑回执](../evidence/uniguide/20261026_partial_batch_recovery.json)。

## 全人口质量检查的显式氢校验修复与定时巡检

2026-10-26 15:30（Asia/Shanghai）检查发现进程已经退出，PID 文件已失效。粗标定 52 组、细标定 24 组、窗口标定 48 组已结束；有限结果分别为 38、20、45 组，其余均按已记录失败计入预算。全人口标定前两组共 100 次采样及重评分已经完成，第二组的质量检查在 `ug_b001_final_004` 处停止。此时累计标定尝试 2300 次，R0 与正式版本均为零。

失败原因是 OpenBabel 在 PDB 回读中为亚胺双键添加了 E/Z 表示，RDKit 默认保留定义该立体信息的显式氢。原始分子为 17 个重原子，回读表示为相同重原子图加一个显式氢；原代码直接比较非立体 SMILES，误判为图变化。该行为与 [RDKit RemoveHsParameters 文档](https://www.rdkit.org/docs/cppapi/structRDKit_1_1MolOps_1_1RemoveHsParameters.html) 的 `removeDefiningBondStereo=false` 默认值一致（访问日期 2026-10-26）。

修复提交 `ad1510b97a3af3be56773840bca373d3237c59d2` 仅在校验副本中统一显式氢表示，同时按坐标映射核查形式电荷、同位素、自由基数、连接氢数和键。实际传入 PLIP 的含氢 OpenBabel 分子、坐标、互作规则和源 SDF 均不变，已有成功 profile 的缓存继续保留。真实失败结构被保存为回归 fixture；远端 16 项测试通过，包括真实案例、原子重排，以及对错误坐标、电荷、同位素、键的拒绝检查。代码先本地提交并推送 GitHub，再通过此前已验证的 SSH Git bundle 路径部署固定提交。工程诊断与补后处理不生成新候选、不额外预留科学采样预算。

本次未发现本项目已有的活动自动化，因此补建当前对话的 heartbeat：`uniguide-flowr`（UniGuide FLOWR 巡检与主轮复盘），状态 ACTIVE，间隔 35 分钟，沿用旧计划的巡检间隔。每次检查真实进程、执行锁、日志与产物；异常时保存证据并按既定流程修复续跑。每个完整 UG 版本仍必须由当前 GPT 主 Agent 实际完成证据复盘、按触发条件研究文献和代码并更新 skills，接受复盘后才运行下一版。连续两次下降回到最优兼容 UG skills，不重算最优版。UG20 最终交付后停用本巡检。其他实验的自动化未改动。

15:47 已补完该组 50 个分子的质量检查：50 个均可进行 PLIP 评估，49 个通过 PoseBusters，49 个为有效连通图，1 个存在严重碰撞。这些结果完整计入记录。补后处理前后的 106 个既有文件哈希全部一致，新增采样为零，尝试预算仍为 2300。见 [补后处理审计](../evidence/uniguide/20261026_explicit_h_recovery.json)。远端随后启动固定修复提交的完整验收，日志为 `explicit_h_acceptance_20261026_154750.log`；只在该验收成功后恢复标定队列。

15:53 完整验收已通过，证据目录为 `acceptance/runs/1793000874512566622/`：16 项测试通过；两位教师原生零增量短轨迹、真实 VJP 有限差分、50/25 行布局、随机流及同人口精确恢复均通过，梯度数值和既有容差未变。随后从固定提交 `ad1510b97a3af3be56773840bca373d3237c59d2` 启动 `calibrate && run-version UG01`，启动 PID 为 53009，日志为 `calibration_to_UG01_20261026_155323.log`。PID 仅为启动回执，后续必须结合进程状态、执行锁和产物更新时间确认运行。见 [完整验收与续跑回执](../evidence/uniguide/20261026_explicit_h_relaunch.json)。本次全人口标定配置为每位教师 20 组，已完成的前两组仅复用结果；本次修复不补采旧失败组。

## 历史修订：保留 50/50 检查，与推理并行（已被下文10/50修订取代）

用户要求避免在推理之外单设全量质量检查时段，并在询问覆盖范围后明确选择「仍检查50/50，但并行且不阻塞推理」。原来每组生成后同步调用 PLIP/PoseBusters，确实使 GPU 等待了 CPU 后处理。它们并不参与现行 C–gap 标定初值、尺度域和排序，因而没有必要阻塞下一组生成。

提交 `13abe0889a22b6181e397361801238825c4e6f29` 将标定、R0、正式每组中的慢检查统一改为持久化队列。两路独立 CPU 进程运行原有全人口检查，数值库各限1线程，屏蔽 GPU。主进程只计算基础构建、连通性和碰撞统计，写入请求后继续推理；慢检查完成、待完成和工程失败分别记录，结果延迟不改变已冻结的同步 Agent 输入。队列不在每轮、教师切换、标定结束或版本归档处等待清空。主复盘读取当时可用的诊断并明确缺失项。最终仍保留全部尝试及结构，不能把异步状态误报为通过。

切换前等到当前组的采样与重评分全部完成，再停止旧串行执行进程。此时累计尝试为2400，未中断任何科学采样；29个既有科学产物的 SHA256 在切换前后全部一致。旧检查缓存继续复用。17项远端测试通过，覆盖队列不调用慢检查、持久化与幂等请求、全部50次尝试的分母、未构建候选、错误隔离和源文件变更拒绝。部署仍遵循本地修改、提交、推送 GitHub 后远端固定提交执行。

16:11 实测四组已有50候选的基础统计与入队分别用时约0.54、0.39、0.39、0.39秒；前三组复用完整检查，第四组由后台补完，49个成功构建分子被实际检查、1个未构建候选计入50次总分母。CPU环境验证为 OMP/MKL/OpenBLAS 各1线程，CUDA/HIP/ROCR 可见设备均为空。定时巡检 `uniguide-flowr` 保持 ACTIVE、35分钟间隔，已改为同时监控推理与质量队列，不再等待全量检查才能启动R0、正式版本或主复盘。

进一步选取两个已有分子做并行一致性检查，包含此前显式亚胺氢的案例。两个独立CPU进程各耗时约6.68秒、时间区间重叠；PLIP互作、core、指纹和质量指标与原顺序检查完全一致。检查复用已有结构，新增科学候选和尝试均为零，工程比较用的临时工作进程已在完成后关闭。正式质量worker持续保留。

16:17 完整验收通过并恢复 `calibrate && run-version UG01`。验收目录为 `acceptance/runs/1793002308561800480/`，17项测试及两教师真实GPU检查全部通过，VJP数值、原生轨迹和恢复结果与先前一致。新日志为 `calibration_to_UG01_20261026_161731.log`，启动PID为70056。详见 [队列启动与耗时](../evidence/uniguide/20261026_async_quality_start.json) 和 [验收、并行一致性与续跑回执](../evidence/uniguide/20261026_async_quality_relaunch.json)。这一节及实施协议中的用户修订取代此前“完整质量检查后才开始下一阶段”的执行方式。

## 16:56 巡检：极端坐标的诊断边界

巡检时 GPU 推理健康，正在6pyy第20个全人口配置，累计预留3200次尝试。后台两组 `p008`、`p009` 在PDB回读坐标核查中失败。诊断确认，前者一个候选的z坐标达到约−1109 Å，后者坐标达到约±2900万 Å，且均含大量断开的片段。PDB的坐标字段为 [Real(8.3)](https://www.wwpdb.org/documentation/file-format-content/format33/sect9.html)（访问2026-10-26），这些值超出字段范围，回读会降精度或错列。这是生成结果的几何发散触发了序列化限制，不能通过平移、截断坐标或放宽一致性阈值掩盖。

CPU修复在写PDB前检查真实格式化宽度，包括四舍五入边界；不能表示时保存 `plip_evaluable=false` 和具体原因，互作恢复为null，不把缺失结果记为零互作。两个真实案例进一步触发PoseBusters空间网格的IndexError，故评价器异常单独记录 `posebusters_evaluable=false`、`posebusters_pass=null` 及错误；基础连通性、碰撞统计和源坐标继续保留。所有50次尝试仍被记账，完整检查回执另列PLIP和PoseBusters可评估数量。工程错误与已测量的不合格状态保持区分。

提交 `13785f848a9c847e5002331a99e3f6121b9506f1` 已推送GitHub并在指定远端通过20项CPU测试，两个真实失败结构也完成验证，未生成新候选。CPU代码单独检出于 `/workspace/EvoMolSteer-quality-2b51f97f`；目录名沿用首次检出标识，实际HEAD以该提交为准。GPU主仓库仍为 `13abe0889a22b6181e397361801238825c4e6f29`，PID70058持续运行，源码和科学配置未改动，未重跑GPU验收或中断推理。两种运行代码分别可由主仓库HEAD及 `quality_queue/worker.json` 的目录、commit和模块哈希追溯。

17:09 在CPU在途任务完成后，仅替换队列调度器及其空闲子进程，新worker PID180837。保留两组旧 `QUALITY_FAILED` 证据，重新入队补后处理，原始SDF、records、aggregate、terminal和trace均绑定哈希；已有成功profile缓存继续使用。CPU worker保持存活以避免旧主仓库重新启动旧检查程序，35分钟巡检已加入独立CPU目录的恢复规则。证据见 [CPU worker更换审计](../evidence/uniguide/20261026_pdb_range_worker_recovery.json) 和 [真实极端坐标案例](../evidence/uniguide/20261026_pdb_range_cases.json)。

17:16 核验两组补后处理已经完成，10个原始科学文件的SHA256全部不变。`p008`的50次尝试中49个成功构建，45个可被PLIP和PoseBusters评估，43个通过PoseBusters，4个因坐标范围问题无法评估；`p009`的50次尝试中40个成功构建，22个可被两项工具评估并通过PoseBusters，18个无法评估。未构建候选分别为1个和10个，仍计入50次分母。这里的 `QUALITY_COMPLETE` 表示检查任务完成，并不表示全部候选合格。见 [完成回执与源文件一致性](../evidence/uniguide/20261026_pdb_range_recovery_verified.json)。

同次核验发现，旧CPU worker在切换前还记录了 `p018`、`p019` 的相同坐标回读错误。17:17 检查两组源SDF后分别确认4个超出PDB坐标字段范围的候选，保存旧错误和源文件哈希后，交由新CPU worker补后处理。见 [后续同类错误审计](../evidence/uniguide/20261026_pdb_range_late_failures_requeue.json)。此时全人口标定已完成26/40组，第27组正在推理，累计标定预留3550次；R0和正式版本尚未开始，执行锁与CPU队列锁均正常持有。两组补后处理不阻塞下一组推理。

17:31（Asia/Shanghai）巡检确认 `p018`、`p019` 均已完成补后处理，10个绑定的源文件哈希全部不变。两组均为50次尝试、49个成功构建、45个可供PLIP评估、4个PLIP无法评估；PoseBusters分别可评估45/46个、通过42/43个、无法评估4/3个。完整检查回执与无法评估结果分别记录，原失败证据保留。至此，全人口标定已完成31/40组，31组的50次尝试均已完成检查记账，队列无积压、无当前工程失败；这不代表所有分子通过质量检查。正在推理第32组（3w54 `p011`），累计标定预留3800次，R0和正式版本尚未开始。真实进程、执行锁、CPU锁、模块哈希和持续更新的轨迹均正常，保持原执行队列及35分钟巡检，不重复启动或部署。见 [17:31巡检与补后处理完成证据](../evidence/uniguide/20261026_quality_recovery_1731.json)。

## 用户最新修订：取消全量慢检查，固定抽样10/50

用户明确取消全量质量检查，并确认每组50个候选中检查10个，即五个固定种子各取前两个槽位。此修订取代本文较早的50/50慢检查要求。全部50个仍保留生成时的基础统计；PLIP/PoseBusters只检查种子和logical_slot=0、1对应的候选，构建失败不换位、不补采。抽样结果不外推成全人口通过率；抽样后继续Agent思考与迭代推理，不追加其余40个的全量补查。

17:48（Asia/Shanghai）旧CPU队列在无在途任务时停止；随后保持调度器暂停，避免运行中的旧GPU代码重新启动全量worker。GPU推理及源码均未中断或覆盖，原有请求和profile缓存保留。17:54固定提交 `3f07d1fa660cc46afb8f3c256cd45e6ab54024c7` 在独立CPU目录完成部署，21项远端测试通过（5.61秒），覆盖确定性种子/槽位选取、失败不补位、记录顺序无关、10/50分母、旧回执缓存和旧请求封装兼容。新worker PID491876，CPU数值库各1线程、GPU屏蔽。见 [切换前证据](../evidence/uniguide/20261026_sampled_quality_transition.json) 与 [抽样部署回执](../evidence/uniguide/20261026_sampled_quality_deployment.json)。

17:55实测3w54 `p017`、`p018`各仅产生10份profile及10行慢检查结果，均明确记录attempts=50、checked_attempts=10、not_selected=40和coverage=0.2；两个组的10个绑定源文件哈希均未改变。前组10个通过PoseBusters，后组8个通过、1个测量不通过、1个无法评估；未知结果未当作通过。检查队列无积压。40组全人口标定采样此时已经完成，标定总预算4200次，原执行队列已自动进入 `run-version UG01`（该入口先建立共享R0）。没有重复生成候选或新增工程采样。详见 [真实组抽样与源文件核验](../evidence/uniguide/20261026_sampled_quality_validation.json)。

定时巡检保持ACTIVE、35分钟间隔，已将旧50/50要求替换为最新10/50抽样规则及新CPU提交。新CPU worker将旧GPU封装中的all_attempts请求按最新政策执行并记录实际scope；旧GPU版本归档中的旧mode标签不能用于判断覆盖率，应读取每组回执。后续有审计的统一部署会使用新快照格式。在此之前，禁止从旧GPU仓库恢复全量质量worker。

## 18:06巡检：R0完成、UG01开始及巡检文本修正

18:06（Asia/Shanghai）两教师的共享R0均完成，共100次尝试，两个组均完成10/50抽样检查。UG01已实际完成第一轮的increase组，decrease组正在推理；正式尝试预留100次，其中50次已完成。进程、执行锁和CPU队列正常，抽样worker的三个模块哈希均与部署回执相符，当前无QUALITY_FAILED或AWAITING_GPT_REVIEW。没有重跑R0或中断健康进程。见 [UG01启动与远端健康证据](../evidence/uniguide/20261026_UG01_started_1806.json)。

本次发现上述17:55记录中“定时巡检规则已替换”的结论缺少读回核验：此前读取自动化配置经终端返回时，中文变成Unicode替换字符，随后字符串替换未命中，导致工具虽返回更新成功，保存的提示仍含损坏文字及旧50/50规则。远端抽样代码和实际覆盖未受影响。本次已根据最新协议完整重写本任务的提示与名称，保留ACTIVE状态、35分钟间隔和原目标对话；通过JSON ASCII转义读回，确认保存的提示与提交文本逐字相同、无Unicode替换字符、含checked_attempts=10及固定CPU提交3f07d1fa。今后更新自动化说明必须进行同样读回验证。其他自动化未修改。

## 20:16巡检：UG01实际复盘、更新三份skills并启动UG02

2026-10-26 19:50（Asia/Shanghai）UG01完成两教师各5轮、共30组及1500次尝试。19:57核对真实进程已正常退出、执行锁空闲、FINALIZED与AWAITING_GPT_REVIEW齐备，30组基础统计和30组10/50抽样均完成，没有当前工程失败。主Agent实际读取全部30组聚合指标、四份绑定的代表策略上下文以及共享R0；并核验源SDF/records哈希、抽样种子/槽位/node_id、上下文哈希和两份结构保留回执。参见[完整性审计](../evidence/uniguide/20261026_UG01_integrity_audit.json)。

6pyy代表配置为R03/redistribute，C从R0的0.597012降至0.585163 Å，gap从2.253831升至2.258944，变化很小；3w54为R04/redistribute，C从0.428588降至0.337682 Å，gap从1.020806降至0.747575。上一UG版本不存在，当前最佳UG为UG01，没有连续下降回退。虽然不触发相邻版本停滞规则，主Agent因6pyy接近R0、早期极大增量及窗口loss归因问题，主动查阅UniGuide原文、官方补充代码、On the Guidance of Flow Matching及PyTorch 2.9 autograd文档，保存精确版本、代码压缩包哈希和适用性。

提交`f649d9a32b59089efc44f7704c119bf58ffcd4fa`更新analysis、proposal、synthesis三个skills：明确FLOWR实际时间系数与guidance/native比值；从当前版本已测量的较优配置考虑合法替代锚点；以新测量结果检验上一轮预测，避免仅依据窗口内proxy loss走向继续加减尺度。三项分别记录证据链、预期和UG02可证伪条件。科学loss、匹配、VJP、积分、窗口、随机流与预算未改变。详见[中文复盘](../reviews/20261026_UG01_review.zh-CN.md)及[机器可校验复盘](../reviews/20261026_UG01_review.json)。

300个抽样槽位中，10个未构建；PoseBusters通过276个、已测量不通过13个、已构建但无法评估1个；PLIP可评估287个、已构建但无法评估3个。其余1200个仅保留基础统计，不追加慢检查；检查完成不等于全部合格，也不把抽样结果外推成全人口通过率。

20:08，在代码已本地提交并推送GitHub后，远端直连及四个镜像按批准顺序均8秒超时，改用仅含已推送提交且通过verify的Git bundle。GPU主目录从13abe088统一到f649d9a3；变更只包括此前已审计的三个CPU质量模块及本次三份skills，独立CPU目录仍为3f07d1fa，worker PID491876未中断。远端稀疏检出不包含docs/reviews，首次按工作树路径读取复盘失败，尚未执行accept-review或预留UG02预算；随后从固定提交的Git对象提取同一文件到`reviews/UG01_review_f649d9a3.json`，不改变复盘内容或仓库源码。

远端验收`acceptance/runs/1793016536820714424`通过21项测试和两教师全部GPU检查。真实VJP有限差分相对误差为0.002665/0.010906及0.005510/0.009064；原生零增量、随机流、50/25行布局及同人口checkpoint恢复通过，既有容差未改变。UG01/REVIEW_ACCEPTED.json随后绑定复盘SHA256 `7ec04e3af9ea2892a1ca9f3323d235a6deeb7cb65d7a5af8d85b6f97a358dfa6`。

20:15:34启动UG02，固定执行提交f649d9a3，启动PID562572，日志`UG02_20261026_2015.log`，定位文件`UG02.logpath`。20:16核对真实run-version UG02进程、执行锁、三份skills哈希、manifest父版本及首轮真实API请求均正常；CPU队列无积压，三个质量模块哈希与新主目录一致。预算为标定4200、R0 100、正式1500，UG02尚在首轮策略思考，没有重算R0或UG01。见[部署、验收、接受复盘与启动证据](../evidence/uniguide/20261026_UG02_started.json)。

35分钟heartbeat继续ACTIVE，提示已更新至本次部署与UG02启动状态，并按UTF-8/JSON ASCII转义读回，确认无替换字符、checked_attempts=10、独立CPU提交及间隔均正确。后续健康推理期间保留固定源码，UG02结束后仍需实际GPT复盘，不能自动跳过复盘启动UG03。

## 21:03巡检：UG02输出截断与恢复入口修复

UG02已完成6pyy的R01–R04共12组、600次尝试及12组10/50抽样，在R05/redistribute调用停止。原始回执确认finish_reason=length，16384个输出token全部用于reasoning，正文为空；输入104977个token。没有启动R05采样或预留其预算。总预算仍为标定4200、R0 100、正式2100。未完成整个UG02，不能提前复盘或启动UG03。

21:05保存失败回执、费用和185个既有产物哈希后，原配置恢复在R01/synthesis缓存校验处停止，尚未发起任何新API调用。逐一重建至R05的19个请求证实：15个完全相同，4个synthesis请求仅JSON对象字段顺序不同，科学内容及数组顺序均相同。原因是provider首次返回模型原始对象顺序，却将成功输出按键排序保存；恢复读取排序后的结果，再嵌入综合Agent提示时改变了字符串。原请求校验拒绝该字符串差异，行为应保留，不能删除缓存或跳过校验。

新增独立恢复入口`agenticgen/scripts/resume_uniguide_cached_requests.py`：只对已存在请求严格比较用户消息内JSON对象，拒绝数组顺序、数值、类型、文字、角色和重复键变化；确认仅序列化差异后，将原始消息原样传入未修改的provider，使原有完整请求、schema、模型参数、响应哈希和应用校验继续执行。恢复入口不自动重试输出截断，不改输出额度、thinking、skills或任何采样算法。所有原始请求、回答、失败回执、结构和预算账本均保留；新增审计记录恢复入口提交及哈希、冻结manifest、每次原始序列化复用和既有文件校验。

恢复入口从已推送固定提交单独部署，导入GPU主仓库f649d9a3的冻结模块；主仓库HEAD、manifest及source_binding均保持不变。它在独占执行锁内验证实际模块目录、固定提交、源码哈希与185个源文件，再调用原run_version。CPU抽样worker不变。这是请求重建的工程修复，不是UG策略更新；测试和真实缓存验证只在指定远端进行，通过后才实际续跑。

21:19固定恢复提交`ce0ec57c50a2130021cfd0cc390da65e23ce6467`通过12项远端测试（1.53秒），以及18个真实成功缓存、1个中断请求的无网络预演；4个综合提示复用原始序列化，其余15个请求逐字一致。185个既有文件全部保持原哈希。GitHub直连及四镜像按批准顺序均8秒超时后，通过已推送提交的Git bundle验证和部署；仅向主仓库Git对象库导入提交，未移动其HEAD或修改冻结源码。

21:20:01实际启动续跑，恢复入口位于`/workspace/uniguide_recovery/ce0ec57c50a2130021cfd0cc390da65e23ce6467/agenticgen/scripts/resume_uniguide_cached_requests.py`，启动PID574642，日志`UG02_request_recovery_20261026_212001.log`。21:20:49核对真实进程、独占执行锁和恢复审计，已到达原R05/redistribute并保存wire_2；其字节与wire_1完全一致，原失败记录保留。此时请求仍在途，不以进程存在宣称该API已成功。见[失败原因、缓存复核、远端测试与启动证据](../evidence/uniguide/20261026_UG02_request_recovery.json)。普通run-version不含这个显式恢复适配，不能在同一错误上重复启动。

21:22巡检提示更新至上述恢复入口、固定提交、日志与审计路径，ACTIVE和35分钟间隔保持不变；UTF-8/JSON ASCII读回逐字核验通过，无替换字符，checked_attempts=10、固定CPU提交和恢复提交均存在。

21:24确认原R05/redistribute真实API成功：finish_reason=stop，12488个输出token，其中10578个reasoning token；模型、16384上限及原request SHA256均未变，单次HTTP成功，无格式修复。随后R05/synthesis也成功。21:25:38已完成第13组（R05/increase），第14组decrease推进至第20步；未FINALIZED、未到GPT主复盘节点。GPU/CPU执行锁均持有，GPU主仓库仍为干净f649d9a3，185个原始文件哈希一致，CPU模块哈希匹配且无诊断积压或工程失败。详见[恢复成功与实际采样进度](../evidence/uniguide/20261026_UG02_recovery_confirmed.json)。原输出截断和第一次恢复受阻的费用、回执与日志保留，不重生成前600个候选；新增尝试只对应此前未执行的R05组。

## 22:14巡检：3w54第二轮Agent输出截断与原配置恢复

22:14（Asia/Shanghai）核对真实进程退出、GPU执行锁空闲：UG02在21:47:07停于3w54/advice_R02/redistribute。6pyy五轮15组和3w54第一轮3组均已完成，共18组900次；18组基础统计和18组10/50抽样完成，无采样FAILED或QUALITY_FAILED。3w54第二轮尚未开始采样，预算仍为标定4200、R0 100、正式2400。此次是不同调用的输出截断，不是采样失败或主版本成绩下降。

回执记录finish_reason=length、输入96577 token、输出16384 token，其中15586为reasoning，正文余量798 token，未产生完整合法答案。原失败回执与usage保留。原恢复入口于退出时核验此前185个文件仍一致；本次将保护范围扩大为311个文件，并备份6pyy已完成的结构保留结果。已在远端用原provider验证30个成功调用缓存和第31个中断请求，全部通过；预演没有API调用或新增候选。

22:16:56沿用已验证恢复入口ce0ec57c50a2130021cfd0cc390da65e23ce6467启动一次显式续跑，PID582777，日志`UG02_3w54_recovery_20261026_221656.log`，`UG02.logpath`已更新。保护账本为`engineering_audits/UG02_3w54_length_20261026_2216.json`。GPU主仓库保持f649d9a3，CPU仍为3f07d1fa；未改源码、skills、模型、thinking或16384额度。22:17:52核对真实进程及新wire_2，续发请求字节与原wire_1完全相同；当时API仍在途，不宣称已经成功。

两次不同的redistribute调用触及相同输出上限，将作为UG02完整结束后的GPT复盘证据，研究如何通过skills减少无效复述和重复推演。当前版本内不改提示、配置或预算，不以模板替代真实Agent输出。详见[3w54截断及恢复审计](../evidence/uniguide/20261026_UG02_3w54_recovery.json)。

22:20:36确认本次恢复成功：redistribute在48.65秒内返回完整答案（finish_reason=stop，输出7662 token，其中5964 reasoning），随后synthesis也单次HTTP成功，均无格式修复、无额度变更。第19组3w54/R02/increase已经推进至第60步，正式累计预留2450次，其中此前18组900次全部复用。311个既有文件哈希一致，6pyy的三份retention文件与恢复前备份完全一致；18组抽样完成，无QUALITY_FAILED。详见[真实API成功及恢复后的采样进度](../evidence/uniguide/20261026_UG02_3w54_recovery_confirmed.json)。35分钟巡检继续ACTIVE，最新PID、日志、311文件保护账本及成功回执已纳入提示并按UTF-8/JSON ASCII方式核验。

## 22:48巡检：3w54第三轮输出截断，保留21组后续跑

22:48（Asia/Shanghai）确认恢复进程582777已于22:30:19退出，执行锁空闲。UG02完成6pyy全部15组、3w54前两轮6组，共21组1050次尝试；21组基础统计和21组10/50抽样检查完成，没有采样FAILED或QUALITY_FAILED。停机点为3w54/advice_R03/redistribute，第三轮尚未采样。正式累计2550次，标定4200次、R0 100次不变。

此次原始回执记录输入100043 token、输出16384 token，输出全部用于reasoning，正文为空，finish_reason=length。这是UG02第三个不同调用触及上限；不代表第三次主版本成绩下降。所有失败usage与原始请求保留，完整UG02结束后必须结合三次截断分析skills中的推理效率。当前版本保持原模型、thinking、输出额度与skills冻结，不自动循环重试。

22:55完成无网络预演：34个成功调用缓存均通过原provider及应用校验，第35个中断请求与原请求相同；7个综合提示仅复用原始JSON序列化。此次保护账本扩大至363个既有文件，包含21组科学产物、请求、回答、失败回执和manifest/skills；原311个文件哈希也全部一致。6pyy结构保留结果已另存备份。预演没有新增API调用或候选。

22:55:27从已推送并验证的ce0ec57c恢复入口启动一次显式续跑，启动PID589254，日志`UG02_3w54_R03_recovery_20261026_225527.log`，`UG02.logpath`已更新。保护账本为`engineering_audits/UG02_3w54_R03_length_20261026_2252.json`，预演回执为`UG02_3w54_R03_recovery_preflight_2252.json`。GPU主仓库仍为干净f649d9a3，CPU worker仍为独立3f07d1fa；未覆盖科学源码、未重采已有1050个候选。启动后先验证真实入口进程和执行锁，再以API回执与实际采样进度确认恢复效果。

22:59:36确认续跑成功，3w54/R03/increase已推进至第30步。redistribute续发的wire_2与原wire_1字节完全相同，首次完整回答因违反“一至两个窗口、一个声明的对数步长”约束而被拒绝，随后原有json-interface-v5应用约束修正流程返回合法答案；这次恢复包含两次HTTP调用，不能记为无接口修正。两次输出分别13384和4485 token，后者reasoning为908；最终finish_reason=stop。综合Agent随后单次成功，输出11212 token，其中9057为reasoning。模型、thinking和16384上限保持原值；失败答案及修正请求均保留。

363个保护文件及6pyy retention备份哈希仍完全一致。CPU队列无积压或当前工程失败，执行锁和worker锁正常，最新正式预留2600次仅新增此前未执行的R03/increase组。详见[第三次截断、无网络预演与实际采样恢复证据](../evidence/uniguide/20261026_UG02_3w54_R03_recovery.json)。巡检保持ACTIVE、35分钟间隔，并纳入最新恢复入口、PID、日志及363文件保护账本；更新后按UTF-8/JSON ASCII读回核验。

## 10月27日：UG02完成、实际复盘与UG03准备

23:58巡检确认UG02恢复进程已正常退出、执行锁空闲，两教师30组1500次及30组10/50抽样全部完成，FINALIZED与AWAITING_GPT_REVIEW齐备。正式累计3000次，标定4200、R0 100保持不变。363个受保护文件无变化，CPU抽样队列无积压。主Agent实际读取30组指标、四份绑定上下文以及当前/UG01/最佳UG/R0；00:11补充核验两教师50个保留槽位的选择、node_id与源结构绑定及代表trace/terminal哈希。

UG02平均C较UG01降低0.002786 Å、平均gap增加0.077928，按冻结规则记录一次整体下降，最佳保持UG01；未触发连续两版下降，UG03从UG02的skills修订。6pyy改善，3w54退步，不能用平均C微降掩盖后者。300个抽样槽位有291个PoseBusters通过、6个实测不通过、3个未构建，297个PLIP可评估；其余1200个不追加慢检查。

本轮主动重查UniGuide原文/官方代码、Flow Matching guidance原文、PyTorch反传文档及DeepSeek thinking官方说明。三项skills修订聚焦原始scale/变化窗口核对、整数格点合法邻居构造和紧凑且可核验的综合决策，回应代表文本事实错误、三次输出截断及接口修正。科学代码、采样及LLM参数均未更改；下一版仍需实际验收复盘与启动检查。详见[中文复盘](../reviews/20261027_UG02_review.zh-CN.md)、[机器复盘](../reviews/20261027_UG02_review.json)、[完整原始证据](../evidence/uniguide/20261027_UG02_review_evidence.json)和[保留审计](../evidence/uniguide/20261027_UG02_retention_review_audit.json)。

00:20固定提交`9dd21edea35fb7fd901d8763f511f425f7cda9a0`在GitHub推送成功后部署。远端直连与四镜像均8秒超时，使用通过verify且仅含已推送提交的Git bundle。部署前的分支名断言发现远端采用既有detached固定提交检出，尚未修改文件；核实后保持这种固定提交方式，在空闲执行锁内从f649d9a3快进至9dd21ede。本地仍仅操作`codex/uniguide-flowr-agent`。部署后21项远端测试通过（5.38秒），科学source binding逐项等于已有双教师验收`acceptance/runs/1793016536820714424`，因此没有重复GPU数值采样。独立CPU worker与提交3f07d1fa保持不变。

UG02的accept-review成功，绑定机器复盘SHA256 `6c17336732488032c3247abc31d7b4748dbe044a37895f9a7eba7972bbc4f906`、三份新skills及下一版本UG03。00:21:35启动UG03，定位PID599601，日志`UG03_20261027_002135.log`、定位文件`UG03.logpath`。00:23:54核对真实`run-version UG03`进程、执行锁、manifest、源码和skills哈希均正常；首个真实API increase调用成功，finish_reason=stop、没有接口修正，decrease在途。此时未开始UG03采样，预算仍为4200/100/3000；不能据首个合法回答宣称新skills已经改善推理效率。CPU锁持有、模块哈希一致、抽样队列无积压。详见[部署、接受复盘和启动证据](../evidence/uniguide/20261027_UG03_started.json)。

35分钟巡检已更新为UG03固定部署和一次主版本下降的状态，保持ACTIVE；UTF-8/JSON ASCII转义读回与提交提示逐字相同，无Unicode替换字符，含checked_attempts=10、最新CPU提交及35分钟间隔。后续若发生停止，应先保存该版本证据并做无网络预演；已验证的ce0ec57c恢复入口按实际manifest验证冻结提交，可继续用于严格的原始序列化复用，不自动循环重试、不删除缓存。当前UG03健康运行期间不部署后续文档提交或改写源码。

## 01:09巡检：UG03第二轮输出截断，保留150个候选后恢复

2026-10-27 01:09（Asia/Shanghai）确认原进程599601退出、执行锁空闲。UG03停在6pyy/advice_R02/redistribute，第一轮3组150次尝试及3组10/50抽样均已完成，无采样FAILED或QUALITY_FAILED，CPU队列无积压。预算为标定4200、R0 100、正式3150；第二轮尚未采样。冻结GPU提交9dd21ede、三份skills及独立CPU模块哈希均一致。

原始回执记录输入94791 token、输出16384 token，全部输出用于reasoning，正文为空、finish_reason=length。此次事实已反证“本次skills足以消除截断”的预期，须纳入UG03完整结束后的GPT复盘；不提前改写本版skills、模型、thinking或额度。第一轮synthesis曾出现JSON缺少分隔符，由原有一次接口修正成功，回执保留；这不是采样失败或新的主版本下降。

01:10完成远端无网络预演：6个成功调用经过原provider完整缓存与应用校验，第7个中断请求与原请求精确一致；一个synthesis提示仅复用原始JSON对象序列化。保存230个既有文件的SHA256，包含已生成结构、轨迹、抽样记录、请求/回答和冻结配置；会继续追加的history.json另存内容快照，不将其误当作不可变化文件。保护账本为`engineering_audits/UG03_6pyy_R02_length_20261027_0111.json`，预演回执为`UG03_6pyy_R02_recovery_preflight_20261027_0111.json`。预演没有API调用或新增候选。

01:11:20使用已推送、通过12项远端测试的ce0ec57c独立恢复入口进行一次显式原配置续跑，定位PID601512，日志`UG03_6pyy_R02_recovery_20261027_011120.log`，`UG03.logpath`已更新。该入口校验UG03自身manifest所绑定的9dd21ede，不改科学源码、不删除缓存、不重新生成既有150个候选、不自动循环重试。应以新wire/receipt及后续采样产物确认恢复效果，不能以PID存在宣称调用成功。

01:14:53确认恢复成功：redistribute续发wire_2与原wire_1字节完全一致，单次HTTP返回合法答案，finish_reason=stop，无接口修正，输出14930 token，其中reasoning为13197。随后synthesis首次完整回答缺少必填selected字段，被原校验拒绝；原有一次格式修正后成功，两次输出分别7994和1876 token，不能记为单次成功或无修正。所有原始失败回答、修正请求及费用均保留，模型、thinking和16384上限未变。

第二轮increase已推进至第40步，正式累计预留3200次，仅新增此前未执行的一组；230个保护文件哈希全部一致，GPU仍为干净9dd21ede，执行锁持有，CPU抽样队列无积压。完整停止证据、保护账本、无网络预演、恢复绑定和真实API成功记录见[UG03第二轮恢复审计](../evidence/uniguide/20261027_UG03_6pyy_R02_recovery.json)。35分钟巡检保持ACTIVE，并更新至本次恢复入口、日志、PID和已确认进度；UG03完成前仍保持skills冻结。

## 01:43巡检：UG03第四轮综合决策截断

2026-10-27 01:43（Asia/Shanghai）确认恢复进程601512已经退出、执行锁空闲。UG03已完成6pyy前三轮9组450次尝试和9组10/50抽样，停止在第四轮synthesis，第四轮尚未采样。没有采样FAILED或QUALITY_FAILED，CPU无积压；预算为标定4200、R0 100、正式3450。GPU冻结提交9dd21ede、manifest、skills及独立CPU提交3f07d1fa均核验一致，上一轮230个保护文件无变化。

失败发生于01:36:53，输入108235 token，输出16384，其中reasoning16295，正文仅89 token且JSON不完整，finish_reason=length。第四轮三个私有建议均已成功。截断已从redistribute延伸到synthesis，须纳入完整版本复盘，不能据原配置恢复偶然成功宣称skills消除了问题；本版继续冻结科学代码、skills与API额度。

01:44保存662个既有文件哈希，history另存内容快照。保护账本为`engineering_audits/UG03_6pyy_R04_length_20261027_0144.json`。无网络预演校验15个成功缓存及第16个中断请求，四个synthesis提示只在严格证明JSON对象内容相同后复用原序列化；API调用和新候选均为0。预演记录为`UG03_6pyy_R04_recovery_preflight_20261027_0144.json`。

01:48:53使用同一已推送ce0ec57c独立恢复入口进行一次显式原配置续跑，定位PID603743，日志`UG03_6pyy_R04_recovery_20261027_014853.log`，`UG03.logpath`同步更新。启动前曾因将完成标记误写为无扩展名的COMPLETE而被只读断言拦截，核实实际文件为COMPLETE.json后修正启动检查；该次检查没有启动进程、调用API或改写产物。实际续跑仅启动一次，保留已有450个候选及所有失败记录，后续仍须以真实回执和采样进度确认恢复。

01:51:45确认恢复成功：synthesis新wire_2与原wire_1字节完全一致，单次HTTP返回合法答案，finish_reason=stop，无接口修正，输出9424 token，其中reasoning7846。6pyy/R04/increase已推进第70步，正式累计预留3500次，仅新增此前未执行的一组；9组既有产物及全部662个保护文件哈希不变。GPU保持干净9dd21ede、科学代码和skills哈希一致，独立CPU worker保持3f07d1fa、队列无积压；本次未重采候选或增加检查范围。恢复运行绑定为`UG03_request_recovery_20261027_014858.jsonl`，完整停止、保护、预演、启动和恢复证据见[UG03第四轮恢复审计](../evidence/uniguide/20261027_UG03_6pyy_R04_recovery.json)。35分钟巡检继续保持ACTIVE，并以本次恢复入口、日志和真实进度作为下一次核查基线。

## 02:53巡检：UG03第二位教师的减弱建议截断

2026-10-27 02:53（Asia/Shanghai）确认恢复进程603743退出、执行锁空闲。6pyy五轮15组和3w54第一轮3组共900次尝试完成，18组基础统计及18组10/50抽样完成，无采样FAILED或QUALITY_FAILED，CPU无积压。全部18组抽样种子、槽位、node_id及源文件绑定核验通过。预算为标定4200、R0 100、正式3900，3w54第二轮尚未采样。GPU保持干净9dd21ede，科学代码、skills、CPU模块和662个既有保护文件均无变化。

停止位置为`3w54_RNB_A_504/advice_R02/decrease`；输入96679 token，输出16384，其中reasoning15068，正文因截断未形成完整JSON，finish_reason=length。该轮increase已经成功。至此UG03的redistribute、synthesis、decrease三个不同调用曾发生长度截断；不能把本轮skills改进评价为已经解决该问题。新增6pyy R05/synthesis的JSON解析修正、3w54 R01/redistribute缺少falsifier的一次修正也保留原始回执，待完整版本统一复盘。

02:54:56完成无网络预演，29个成功缓存通过原provider和应用验证，第30个中断请求精确匹配。保存1318个源文件哈希、两教师history内容快照，并单独备份6pyy结构保留目录；无API调用、无新增候选。保护账本`engineering_audits/UG03_3w54_R02_length_20261027_0255.json`，预演`UG03_3w54_R02_recovery_preflight_20261027_0255.json`。文件名中的0255是审计标签，实际完成时间以JSON内Asia/Shanghai时间为准。

02:55:21从固定ce0ec57c独立恢复入口显式原配置续跑一次，定位PID606835，日志`UG03_3w54_R02_recovery_20261027_025521.log`，`UG03.logpath`同步更新。该次恢复不改科学代码、skills、thinking或16384上限，不自动循环重试，不重采已有900个候选；成功仍须由原请求对应回执和后续采样进度确认。

02:59:28确认恢复成功：decrease的wire_2与原wire_1字节相同，单次HTTP返回合法结果，finish_reason=stop，无接口修正，输出11558 token，其中reasoning10029；后续redistribute单次成功，输出7668。synthesis首次完整回答出现JSON Extra data，经原有一次格式修正后成功，两次输出分别5255和1397，不能记为无修正。全部失败和修正回执保留。

3w54/R02/increase已推进第40步，正式累计预留3950次，仅新增此前未执行的一组；1318个保护文件及6pyy retention备份哈希均一致。GPU保持干净9dd21ede、冻结科学代码及skills无变化，独立CPU worker正常且无积压。恢复运行绑定`UG03_request_recovery_20261027_025527.jsonl`；完整证据见[UG03第二位教师第二轮恢复审计](../evidence/uniguide/20261027_UG03_3w54_R02_recovery.json)。35分钟巡检继续保持ACTIVE，以新日志和实际进程为核查入口，抽样仍为10/50。后续再次停止时重新保存证据、定位并审计，不自动循环重试。

## 04:03巡检：UG03最后一轮重分配建议截断

2026-10-27 04:03（Asia/Shanghai）确认进程606835退出、执行锁空闲。UG03完成27组1350次尝试及27组10/50抽样，停在3w54/advice_R05/redistribute，第五轮尚未采样。预算标定4200、R0 100、正式4350；CPU无积压、无采样FAILED或QUALITY_FAILED，27组抽样身份和请求源文件绑定均核验通过。GPU仍为干净9dd21ede，冻结科学代码、skills、CPU模块及1318个保护文件和6pyy retention备份均一致。

失败输入107040 token，输出16384全部用于reasoning，正文为空、finish_reason=length。本版本现有四个不同调用发生截断，均保留失败回执，不能将恢复成功等同于消除截断。第五轮increase和decrease已成功，后续只继续未完成的建议和最后3组，不改科学配置或额度。

04:04:33完成无网络预演，42个成功缓存经原provider和应用验证，第43个原中断请求精确一致。保存1964个文件哈希、两教师history快照及6pyy retention独立备份。保护账本`engineering_audits/UG03_3w54_R05_length_20261027_0405.json`，预演`UG03_3w54_R05_recovery_preflight_20261027_0405.json`；0405为文件标签，实际操作时间以JSON内Asia/Shanghai时间为准。预演没有API调用或新增候选。

04:04:56从固定ce0ec57c独立恢复入口显式原配置续跑一次，定位PID612741，日志`UG03_3w54_R05_recovery_20261027_040456.log`，同步更新UG03.logpath；旧PID606835已退出。保留1350个已生成候选，不重新采样，不改skills、thinking或16384上限，不自动循环重试；继续以真实回执和采样产物核实恢复效果。

04:08:35确认恢复成功：redistribute续发wire_2与原wire_1字节相同，单次HTTP、无接口修正、finish_reason=stop，输出14302 token，其中reasoning12847；随后synthesis单次成功、无接口修正，输出12515，其中reasoning11068。3w54/R05/increase已到第20步，正式累计预留4400次，仅新增此前未执行的一组。

1964个保护文件和6pyy retention备份哈希全部一致，GPU保持干净9dd21ede，源码、skills、CPU模块均无变化，CPU抽样队列无积压。恢复运行绑定为`UG03_request_recovery_20261027_040502.jsonl`。完整证据见[UG03第五轮恢复审计](../evidence/uniguide/20261027_UG03_3w54_R05_recovery.json)。35分钟巡检继续ACTIVE；等待最后3组及最终分析、结构保留完成后，主Agent实际读取30组和4份上下文复盘，不能因接近结束跳过该步骤或提前启动UG04。

## 2026-10-27 UG03完成与实际主复盘

04:38（Asia/Shanghai）核实UG03已结束，执行锁空闲。30组1500次尝试、30组基础统计、30组10/50抽样、两教师结构保留和FINALIZED/AWAITING_GPT_REVIEW均齐全，无采样或质量工程失败。预算4200/100/4500，CPU抽样队列无积压。1964个既有保护文件及6pyy保留备份不变。04:44进一步核对两教师各50条按槽保留与源node_id、选择结果和SDF身份，代表组终态/轨迹哈希均通过；6pyy代表49个构建成功但C分母为50。

当前GPT主Agent实际读取30组指标、四份绑定策略上下文、48份成功及11份初始失败/无效HTTP回执，并比较UG03、UG02、UG01和R0。UG03平均C=0.4556837230、gap=1.4918186045；相对UG02分别变化−0.0029526364、−0.0893685269，成为冻结规则下最佳版本。UG02→UG03未下降，连续下降计数归零；UG04从UG03 skills更新，不重跑最佳版或R0。6pyy的C/gap变化均小于0.02，触发实际论文和代码检索；其余两种触发条件未满足。

重新阅读UniGuide条件映射/全状态梯度原文及官方源码、Flow Matching路径相关引导公式、PyTorch反传和DeepSeek thinking官方说明。三项skills更新针对窗口与格点步长核对、坐标/构建/抽样分母、可比人口、C带内gap排序、字段完整和单一JSON对象。没有更改采样代码、科学参数、LLM参数或重试政策。UG03的四次输出截断和七次初始JSON/字段错误表明上一版指引未解决此问题；新改动仍需UG04真实结果检验。

300个抽样槽位全部可评价：PoseBusters通过290、测量不通过10；PLIP可评价300，不代表互作全部恢复。其余1200个不补查。详见[中文复盘](../reviews/20261027_UG03_review.zh-CN.md)、[机器复盘](../reviews/20261027_UG03_review.json)、[完整原始证据](../evidence/uniguide/20261027_UG03_review_evidence.json)及[保留核验](../evidence/uniguide/20261027_UG03_retention_review_audit.json)。完成GitHub推送、空闲锁内固定提交部署、远端验证和accept-review后，方可启动UG04。

04:50:47已完成固定提交`930db03280c66c3f8a39f131ae01df3202544af4`部署及验证。提交先成功推送GitHub；远端直连及四个镜像按顺序各8秒超时，随后通过已验证、仅含已推送提交的Git bundle，在空闲执行锁内从9dd21ede快进。三份skills及文档之外无科学源码变更。21项远端测试全部通过（5.38秒），source_binding与已有双教师验收`acceptance/runs/1793016536820714424`逐项相同，没有新增数值采样。accept-review绑定UG03机器复盘SHA256 `0ab45956db5fb4353b59b7c752938a34f703ed84ae59c96b60f59b32d4eb2689`，基础checkpoint为UG03、recovery_derived=false、下一版UG04。

04:51:04启动UG04，定位PID616820，日志`UG04_20261027_045104.log`及`UG04.logpath`。04:51:53核对真实run-version进程、执行锁、manifest、skills和源码绑定全部匹配，独立CPU worker仍为3f07d1fa且无积压。04:52:51确认首轮increase真实API调用成功，finish_reason=stop、单次HTTP且无接口修正，输出10299（reasoning9276）；decrease请求在途，尚无UG04采样，预算4200/100/4500。首个成功调用不证明新skills消除了截断或改善了生成质量。详见[UG04部署、接受复盘与启动审计](../evidence/uniguide/20261027_UG04_started.json)。巡检保持ACTIVE、35分钟、checked_attempts=10；UTF-8及JSON ASCII读回确认中文无替换字符、GPU/CPU提交和抽样政策正确。健康运行期间不再覆盖远端源码。

## 05:48巡检：UG04第三轮重分配建议截断

2026-10-27 05:48（Asia/Shanghai）确认原进程616820退出、执行锁空闲。6pyy前两轮6组300次尝试、6组基础统计和6组10/50抽样已完成，第三轮尚未采样。没有采样FAILED或QUALITY_FAILED，独立CPU队列无积压；抽样种子、槽位、node_id及源文件绑定核验通过。预算为标定4200、R0 100、正式4800。GPU仍为干净930db032，manifest、skills及独立CPU代码绑定匹配。

失败发生于05:16:12，位置为`6pyy_H7S_A_402/advice_R03/redistribute`。输入98607 token，输出16384全部用于reasoning，正文为空、finish_reason=length；同轮increase和decrease已成功。这是UG04首个长度失败，表明新skills未消除截断，须纳入完整UG04复盘，本版不修改科学代码、skills或API额度。

05:52:44完成无网络预演，10个成功缓存经原provider及应用验证，第11个中断请求与原请求、原wire完整对象精确一致。两个synthesis提示仅在严格验证JSON对象内容相同后复用原序列化。保存438个文件哈希和可变history内容快照，API调用和新候选均为0。保护账本`engineering_audits/UG04_6pyy_R03_length_20261027_0554.json`，预演`UG04_6pyy_R03_recovery_preflight_20261027_0554.json`；0554为文件标签，实际时间以JSON字段为准。最初只读预检把manifest的规范化配置摘要误当作原文件摘要而被断言拦截；改用原pipeline.value_sha核验后通过，此前未启动任务或写入保护账本。

05:53:04从固定ce0ec57c独立恢复入口显式原配置续跑一次，定位PID619336，日志`UG04_6pyy_R03_recovery_20261027_055304.log`，同步更新UG04.logpath。保留已有300个候选和全部失败回执，不重发成功调用、不自动循环重试，继续以真实回执及采样进度核实恢复结果。

05:55:04该次原配置续跑仍在同一redistribute请求处失败：输入98607 token，输出16384全部用于reasoning，正文为空、finish_reason=length。新wire_2与原wire_1字节完全一致，没有接口修正、成功建议或新候选。05:56:11确认进程619336退出、执行锁空闲，438个保护文件哈希均一致；预算保持4200/100/4800，6组抽样完成且CPU无积压，GPU源码和skills未变。停止事实、两次失败usage、保护账本、预演及运行绑定见[UG04第三轮恢复审计](../evidence/uniguide/20261027_UG04_6pyy_R03_recovery.json)。

同一请求已连续两次失败，按“不自动循环重试”和本版冻结要求，不再自行发起第三次调用或改写提示/API额度。已向用户询问下一步：额外一次原配置尝试、授权单请求额度偏离，或继续保持停止。等待明确选择期间，35分钟巡检保持ACTIVE，仅核查真实状态、保护源产物与CPU队列；禁止把下一次heartbeat当作自动再次重试的授权。UG04未完成，不复盘为完成版，也不启动UG05。

## 2026-10-27 用户授权单请求输出额度增加50%

用户明确回复“允许增加输出额度，将输出额度提升50%”。此授权解除UG04第三轮重分配请求的等待状态：对`6pyy_H7S_A_402/advice_R03/redistribute`进行一次16384→24576的显式额度变更恢复。其余调用仍为16384；版本manifest、科学配置、skills、已完成300个候选和两次失败请求保持原状。11:30:57（Asia/Shanghai）只读核查执行锁空闲、无推理进程、438个保护文件哈希一致、6组10/50抽样完成，预算4200/100/4800，CPU无积压。

恢复入口增加hash绑定的单请求授权文件参数。原请求和失败记录不覆盖；增加额度的真实请求、原始回答和provider回执使用独立目录。成功后为原上下文路径创建明确标注有效请求路径的响应别名，不伪装成原16384额度的成功。重放必须重新验证有效缓存、原请求、schema和应用约束。持久化独占claim限制本次额度变更只发起一次provider调用序列，保留原网络重试及一次格式修正；长度失败不自动循环重试。后续恢复UG04必须携带该授权文件并使用新版入口，普通run-version及旧入口不能识别此次协议偏离。

官方Chat Completions文档确认max_tokens为输出token上限，当前支持范围包含24576；本次只改该字段，不调整thinking/reasoning_effort或提示。来源：https://api-docs.deepseek.com/api/create-chat-completion/ （访问2026-10-27）。新增测试覆盖仅目标额度变化、成功缓存复用、原产物保留、实质输入拒绝、缓存/别名篡改拒绝及新额度长度失败后不自动重试。实际远端测试、部署与恢复结果后续追加审计。

11:37:30真实缓存预演通过：10个成功调用及第11个中断请求、history、schema和原wire均核验一致；保护文件扩展至441个。恢复入口固定提交`e9598cb0092783f41b19740de8a2e21a5fafb82d`已推送，直连和四镜像依序超时后以验证Git bundle部署到独立恢复目录，25项远端测试通过（1.90秒）。GPU仍冻结于930db032。

11:38:10启动一次授权恢复，定位PID620111，日志`UG04_token24576_20261027_113810.log`。授权文件为`engineering_audits/UG04_token24576_authorization_20261027.json`，保护账本为`UG04_token24576_preservation_20261027.json`，预演为`UG04_token24576_preflight_20261027.json`。11:39:30确认真实恢复进程和新wire：有效请求max_tokens=24576，与原wire逐字段比较，仅max_tokens变化；当前请求等待结果，不能提前宣称恢复成功。35分钟巡检已同步此次授权和新入口，UTF-8/ASCII JSON读回确认一致。

11:40:05提高额度后的redistribute成功：finish_reason=stop，单次HTTP、无接口修正，输出16490（reasoning15303），实际超过原16384上限106；新wire与原wire仅max_tokens不同。11:41:07后续synthesis以原16384额度单次成功，输出8365，无接口修正。11:41:59确认6pyy/R03/increase已到第30步，执行锁持有、真实进程推进，正式累计预留4850次（其中新50次在执行），既有6组300次产物不变，CPU抽样无积压。441个保护文件及科学/skills/CPU绑定均无变化。此为本次请求恢复成功，不能宣称整个版本完成或已消除所有截断。

审计见[UG04提高输出额度后的恢复证据](../evidence/uniguide/20261027_UG04_token24576_recovery.json)。原两次失败保留，增加额度的有效产物位于advice_R03/redistribute.token24576；原上下文response/receipt是明确标记protocol_deviation的别名，统计实际HTTP时不能重复计数。后续恢复必须使用e9598cb0新版入口并携带原授权文件；已成功的扩大额度调用仅做缓存复用，不重新消耗额度。UG04继续完成剩余轮次后进行实际GPT复盘，35分钟巡检保持ACTIVE。

## 12:12核查：UG04第四轮新截断及一次原配置恢复失败

2026-10-27收到此前第三轮问题的延迟答复“再按原配置显式尝试一次，保持版本可比性”。第三轮已按中间另行明确授权的24576上限成功，该延迟答复不撤销已发生的协议偏离，也不触发重复调用。核查发现新的停止点是`6pyy_H7S_A_402/advice_R04/redistribute`：进程620111于11:51:46退出，前三轮9组450次尝试、9组基础统计及9组10/50抽样完成，第四轮尚未采样，预算4200/100/4950。第四轮首次失败输入102056、输出16384（reasoning16367、正文17），finish_reason=length。

按已批准的“新停止先审计，再显式原配置恢复一次”流程，12:13:25无网络预演验证14个成功缓存和第15个中断请求。原请求、schema、wire与history精确匹配；第三轮提高额度的有效结果与别名也通过验证，仅复用成功缓存。保存659个既有文件哈希及history内容快照，原441个保护文件不变。预演没有HTTP调用或新候选。保护账本为`engineering_audits/UG04_6pyy_R04_preservation_20261027_1214.json`，预演为`UG04_6pyy_R04_preflight_20261027_1214.json`；文件标签1214不代表操作时间，实际时间见JSON字段。

12:13:52从已验证e9598cb0新版独立入口启动一次原配置恢复，定位PID623468，日志`UG04_6pyy_R04_recovery_20261027_121352.log`，运行绑定`UG04_request_recovery_20261027_121357.jsonl`。继续携带第三轮授权文件以验证其成功缓存；第四轮目标请求仍为16384，不改提示、科学代码、skills、manifest或其他API参数。第三轮有效目录仍只有一个wire，没有重复发送成功调用。

12:15:44第四轮原配置恢复再次截断：输入102056、输出16384（reasoning15571、正文813），finish_reason=length，未返回完整JSON，无接口修正。新wire_2与原wire_1字节完全相同。12:18:30核查PID623468退出、执行锁空闲，没有其他UG04推理进程；9组450次产物不变，没有新候选或采样失败，659个保护文件全部一致。GPU仍为干净930db032，科学代码、skills和独立CPU模块绑定匹配；CPU worker保持3f07d1fa且无积压。全部9组抽样种子、槽位、node_id与源绑定通过核验。

第四轮同一请求现已两次长度失败，本次恢复结果为失败，不能记为健康推理。保持停止，不自行发第三次请求或提高此新请求额度，不启动UG05；需要用户明确第四轮下一步方向。35分钟巡检继续ACTIVE，等待期间只读核验，状态未变化时保持安静。此次延迟答复的指代、两次原始失败、保护账本、预演、启动、运行绑定和最终状态见[UG04第四轮恢复证据](../evidence/uniguide/20261027_UG04_6pyy_R04_recovery.json)。

## 2026-10-27 用户授权检查冗余并修正提示与输出规范

用户针对第四轮明确要求“检查prompt内容，分析是否存在大量冗余。针对易于产生冗余信息的指引进行修正，并约束LLM的输出规范”。该新指令授权一次提示策略修订，取代第四轮保持原提示等待选择的状态；不授权增加额度、改变采样科学参数或重复已保存候选。UG04将标注为含中途提示修订的版本，不能宣称与先前三个版本拥有完全相同的提示协议，也不提前完成主复盘。

实际请求为102056输入token：system 11891字符，user 255066字符。其中calibration_knowledge为163273字符、domain为56660字符（constant_comparison占56290）、history为28599，comparison_best及default_anchor各重复2967字符。三个不同角色的全部指引被拼入每次请求，私有角色也收到综合决策和主Agent复盘要求；数值记录与相同窗口诊断反复展开。失败输出绝大部分用于reasoning，故只能把输入冗余和角色混杂作为已观察的问题及可检验原因，不能声称已证明其必然导致截断。

修订采用可逆的共享对象引用和窗口列式表示，保留全部数值精度、配置、失败记录、顺序、类型和教师数据，不筛选标定赢家、不删失败行、不修改源产物。每次转换后执行反向还原并与原上下文规范化JSON严格比较。第四轮预览将user压缩至122792字符，共107个共享对象；含新角色指引的总消息由266957降至128269字符（约减少52%）。这是字符测量，实际token变化须从真实provider回执读取。

三份指引改为按角色发送analysis+proposal或analysis+synthesis；限制决定性比较数量、两案构造范围、字段职责与重复解释，移除对私有角色无关的主版本复盘任务。输出保留全部原有字段与行动约束，在新schema增加解释字段maxLength，并用严格JSON解析拒绝重复键、NaN、Markdown和追加对象。私有analysis上限1600字符、uncertainty 500，anchor_reason 240、evidence 500、prediction/falsifier各280；综合analysis 1400、choice_evidence 1200。网络重试、原有一次格式修正、thinking/high及16384输出额度保持不变。

新策略通过独立恢复入口作用于UG04尚未成功的调用；已有成功调用继续走原provider/schema及原应用校验，第三轮24576成功仅缓存复用。新调用的原始重建请求、修订有效请求、原始响应和有效回执分别保留，原上下文路径仅建立标注prompt_revision的别名；每调用独占claim阻止失败后自动再次发送。GPU冻结仓库、manifest、原skills快照和科学source_binding不覆盖，独立入口、策略模块及新skills须先本地提交推送、固定提交部署、远端测试和真实缓存无网络预演。随后只从第四轮未完成处继续；如新提示仍失败，保存证据并停止，不自动循环。

12:34固定提交`3b7ebbc1341a99593bd43f7ee614f90237a1ffe4`已部署到独立目录。GitHub直连及四镜像依序各8秒超时后，使用仅含已推送提交的验证Git bundle；GPU仍干净930db032，未部署本地新skills到冻结主仓库。远端48项测试通过（1.94秒）。12:35:50无网络预演验证14份真实成功缓存和原中断请求，history一致，提示内容可逆转换通过；662份文件受保护，无HTTP或新候选。授权`engineering_audits/UG04_prompt_revision_authorization_20261027.json`，保护`UG04_prompt_revision_preservation_20261027.json`，预演`UG04_prompt_revision_preflight_20261027.json`。

12:36:28以新入口启动一次用户授权的提示修订续跑，定位PID623965，日志`UG04_prompt_revision_20261027_123628.log`，运行绑定`UG04_request_recovery_20261027_123634.jsonl`。必须同时传入原第三轮`--token-override`、新`--prompt-revision`与独立目录`--prompt-skills`，不能使用普通run-version或旧入口。已有成功调用不会重新发送；所有新的UG04未完成调用遵循此次授权的提示与输出规范。

12:38:03第四轮redistribute单次成功、无接口修正、finish_reason=stop。实际输入53697 token，相对原102056减少47.38%；输出12373（reasoning11549），上限仍16384。12:38:27综合决策单次成功、无接口修正，输入56415、输出4373（reasoning3846），原额度不变。主Agent读取两份实际回答，字段长度全部合规；原行动约束已由应用验证。12:39:34真实进程和执行锁持有，第四轮increase第40步，预算4200/100/5000（新50次正在执行），662份保护文件、科学/原skills/CPU绑定均一致，CPU抽样无积压。不能据两个成功回答推断所有后续长度问题已消除。

详见[冗余分析及修订报告](../reports/20261027_UG04_prompt_redundancy.zh-CN.md)和[完整恢复证据](../evidence/uniguide/20261027_UG04_prompt_revision_recovery.json)。UG04完成后的实际GPT复盘须记录第三轮单请求额度偏离及本次中途提示修订，并读取独立策略与新skills的绑定信息。主版本后续部署应显式延续此输入表示、角色路由和输出约束，不能仅复制skills后恢复原先三角色全拼的提示构建方式；须在下一次空闲版本边界审计集成。35分钟巡检继续ACTIVE，健康推理期间不覆盖源码、不重复启动。

## 12:47按用户要求终止旧进程、更新远端主目录并重启

用户明确要求指引优化提交GitHub后，在远端拉取最新代码、终止先前任务并重启UG04第四轮。12:46:33（Asia/Shanghai）实际检查时，第四轮三组已全部完成，UG04累计12组600次尝试；旧进程正在执行第五轮建议。为保留已完成候选和成功调用，在无未完成采样组、无未完成有效provider调用的安全边界终止PID623965。12:47:42执行终止；随后再次核查进程不存在、执行锁释放。独立CPU抽样worker继续运行。第五轮三个已成功的私有建议同样保留，不重新发送。

本地及GitHub分支`codex/uniguide-flowr-agent`均核对为`8eccc068d1696e0ab73cd270b2859d4543a4382a`，其中指引和恢复实现来自`3b7ebbc1341a99593bd43f7ee614f90237a1ffe4`。远端GitHub直连及四镜像按指定顺序各8秒超时，改用已核验、只包含已推送提交的Git bundle；SHA256为`949d5dbd08c2c7cf9ce845a20e4f1cb50f729fa255393e2de3ae55e1cc1a465a`。12:50:05主目录`/workspace/EvoMolSteer`已在指定分支快进到8eccc068，工作区干净。远端科学协议及恢复测试共69项通过（19.78秒）。

新恢复入口、提示策略和三份修订skills均从更新后的主目录读取。为了继续严格校验既有UG04 manifest、原始请求和缓存，另保留`/workspace/EvoMolSteer-UG04-frozen-930db032`固定科学快照。其科学模块与最新主目录、原数值验收的source_binding逐项相同；只有历史提示构建与原skills用于重建并验证旧请求，新调用仍由主目录的修订策略转换。此分离保留原manifest的真实历史含义，没有把新Git提交冒充成原提交，也没有覆盖原skills快照。

12:51:10无网络预演验证19份成功缓存、前三轮和第四轮全部12组proposal/aggregate以及history一致；第三轮24576缓存和第四、五轮修订提示的有效缓存均完整复用。API调用和新候选为0，877个保护文件哈希一致。停止、部署、保护和预演分别保存于`engineering_audits/UG04_user_restart_{stop,deployed,preservation,preflight}_20261027.json`。

12:51:48重启，定位PID627185，日志`UG04_latest_code_restart_20261027_125148.log`及UG04.logpath。入口`/workspace/EvoMolSteer/agenticgen/scripts/resume_uniguide_cached_requests.py`必须同时使用`--frozen-repo /workspace/EvoMolSteer-UG04-frozen-930db032`、`--adapter-commit 3b7ebbc1341a99593bd43f7ee614f90237a1ffe4`、`--prompt-skills /workspace/EvoMolSteer/agenticgen/src/agenticgen/uniguide/skills`及原第三轮token授权、提示修订授权；保护账本改用上述877文件版本。PYTHONPATH的agenticgen/src与src指向固定科学快照，提示入口和新skills来自最新主目录。不能直接普通run-version，不删除claim或缓存。重启复用已完成第四轮，实际下一个未完成调用为6pyy/advice_R05/synthesis，不重采第四轮。

12:53:03核验真实新进程与GPU/CPU锁持有；12组基础统计与12组10/50抽样完成，全部选中seed、slot、node_id及源绑定正确，CPU无积压，未发现采样或质量工程失败。预算4200/100/5100，877文件及新旧科学/提示/CPU绑定全部一致。UG04仍须完成剩余18组及最终分析、结构保留，再做真实GPT复盘；不提前启动UG05。已部署代码在健康运行期间保持不变。

12:53:23重启后的首个新请求（第五轮synthesis）成功，finish_reason=stop，单次HTTP、无接口修正；输入57705，输出11197，其中reasoning10681，上限仍16384。12:53:54第五轮increase推进到第20步，正式累计预留5150次；877个保护文件仍全部一致。主Agent实际读取综合回答，三分支selected完整，原有行动约束通过；回答将未测试提案标为预测，没有将抽样未检查项判为通过。本次部署与重启的完整证据为[UG04最新代码部署及重启审计](../evidence/uniguide/20261027_UG04_latest_deploy_restart.json)。后续文档审计提交不在健康推理中覆盖部署目录。

## 13:32巡检：第二教师把压缩引用标签误作执行锚点

2026-10-27 13:32:21（Asia/Shanghai）确认PID627185退出、执行锁空闲。6pyy五轮15组750次尝试、15组基础统计和15组10/50抽样完成，四份最终分析均成功；第二教师3w54尚未采样。预算4200/100/5250，CPU worker及固定代码绑定正确且无积压，采样FAILED和QUALITY_FAILED均0，UG04尚无FINALIZED或主复盘标记。原877个保护文件均一致。6pyy保留清单50个逻辑槽位、49个可构建SDF身份与源数据匹配，代表组终态和轨迹哈希通过；一个保留槽位未构建，没有补样。

停止点为`3w54_RNB_A_504/advice_R01/increase.promptcompact_v1`，原有一次格式修正用尽后于13:01:49失败。两个HTTP均finish_reason=stop，分别输入49207/50088、输出5791/1389（reasoning5294/776），没有触及16384上限。失败原因都是`Action anchor must have measured results`，不能描述为长度失败或分子采样失败。

实际读取两份错误回答与完整请求后定位：第一份把压缩注册表引用`r80`直接写入两案anchor_id；修正回答改为`calibration/full/3w54_RNB_A_504/p000`。当前history唯一合法锚点是`R0/3w54_RNB_A_504`。压缩数据可无损还原，r80所指记录内部id也是该真实路径，但现有输出schema的anchor_id只是非空字符串，错误消息也只给通用应用约束；因此无损性没有防止模型混淆表示层引用与执行身份。不得把r80当作合法ID或自动改写回答为成功。

13:33:06扩展保护账本至1123个文件并保存history快照，保留两个失败回答、有效请求、修正请求、回执和authorization_claim。13:35:16完成一个待执行请求预览：仅在anchor_id schema加入由原history提取的精确enum，并显式说明共享引用标签和history之外的标定ID不能作为执行锚点。原数值上下文反向还原严格相同；两个真实错误回答均被enum拒绝。另在内存中进行“只替换anchor_id”的反事实诊断，原行动格点/窗口/分支/去重校验通过；这不是LLM成功结果，没有写入正式回答或候选。输出上限16384、一次格式修正和所有科学约束保持原值。

预览为`engineering_audits/UG04_anchor_contract_preview_20261027.json`，状态`preview_only_not_authorized_for_network_retry`，新请求规范化摘要`a72a3406834971611881f54b19b3ee0b9855a38ab9478ed5e9d1b7f15065129b`。当前只是离线修订预览，尚未改生产入口/策略/skills或部署，不宣称已修复运行。新提示调用的独占claim已消耗，按照已批准“不自动循环重试”约束保持停止，等待用户明确是否按该锚点契约修订显式续跑一次。若获准，必须先实现带历史v1缓存兼容的新固定入口/授权与独立有效目录，提交推送、远端验证后再运行；不得覆盖旧请求、删除claim、弱化锚点校验或重复生成6pyy候选。

本轮巡检API调用、新候选及生产代码修改均为0。完整原始证据、1123文件账本、真实失败和离线预览见[UG04锚点身份混淆审计](../evidence/uniguide/20261027_UG04_prompt_anchor_failure.json)。35分钟巡检保持ACTIVE；等待期间只读核查和保持CPU抽样服务，不启动UG05，状态不变时不重复通知。
