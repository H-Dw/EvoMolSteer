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
