# PLIP局部梯度引导：LLM幅度调度与逐链 TOP1 保留

目标服务器固定为 `ssh -p 10104 root@ksai.scnet.cn`。密钥从服务器工作进程环境变量
`DEEPSEEK_API_KEY`读取；SSH密码不写入文件。10399端口不用于本实验。

## 冻结的科学协议

- 教师为已有5nn4_SC2_A_1017、5ost_AEZ_A_915的各Top5，读取原PLIP实验teachers.json并核对node ID。
- 每教师100条独立起点，原五个20槽位批次，种子42、100045、200048、300051、400054。
- 固定FLOWR_ROOT原checkpoint、设备/精度、100步原生SDE、离散转移、蛋白及原子数。
- 固定Endpoint坐标奖励：PLIP核心的pseudo-Huber，delta=1，1 Angstrom无量纲化；
  每步几何Hungarian映射，停止匹配梯度，不进行配体刚体对齐。
- 梯度仅作用于配体坐标，所有模型参数冻结；affinity只用于外层决策。
- 原生更新后注入lambda*dt*grad(R)，正确处理coord_scale；
  每分子最大原子位移0.025 Angstrom，整分子同比裁剪，无RMS归一化。
- R0在步骤10..89使用0.5，其余0；LLM只能调整八个相邻十步区间的幅度。
- 幅度0.0..1.0，分辨率0.1。通常每轮±0.1，至多两个区间；
  同区间同方向连续两次改善后才允许±0.2。
- 每个候选独立优化，R0后10轮；达到教师分数后继续。每条链保留历史最优结构和全部轮次的分析记录。
- 历史Unguided/Steer不重采样，不设零引导实验组；不使用旧“互作恢复增加5个百分点”门槛。

## 教师数量核验（2026-10-21）

设计和程序均为每个 target 的 Top5，共10位教师，每位100条链。远端旧实验
`plip_guidance_5nn4_5ost_s42_20261020/teachers.json` 中也包含全部10位教师。
当前恢复的旧快照中，5nn4有Top1–Top5的运行目录，5ost仅有Top1–Top3的运行目录；
这是旧快照的完成进度，并非将教师范围改为Top3。

| Target | Top1 → Top5 的原 Steer node ID |
|---|---|
| 5nn4_SC2_A_1017 | single_b002_final_005、single_b004_final_009、single_b002_final_007、single_b001_final_000、single_b001_final_014 |
| 5ost_AEZ_A_915 | single_b000_final_016、single_b000_final_015、single_b000_final_000、single_b001_final_002、single_b001_final_006 |

## 完整checkpoint及逐链 TOP1

保存0、0.1、…、0.9、1的完整恢复状态、最终输出和每步噪声/漂移/引导记录。
checkpoint包含current、self-conditioning、prior引用、蛋白编码引用、times、积分器状态、
完整Python/NumPy/Torch CPU/GPU RNG、坐标变换、来源哈希。附属head文件保存原始预测与endpoint。
每轮计算期间保存完整数据，轮末按下述规则整理。每条链在可构建、连通且具有有限
`pic50_on_rescore` 的结果中保留最高分；并列保留较早结果。尚无合格分子时保留R0恢复状态，
报告失败，不将其算作有效候选。TOP1是逐链选择：每位教师最终仍有100个槽位，而非仅留1个。

1. 提交本轮全部20条链的历史；完成评分、PLIP和PoseBusters，保留成功或失败的分析文档。
2. 将每条链的最佳状态物化为自包含快照，保存全部11个恢复节点、预测头、终态、
   实际噪声/速度轨迹、最佳分子及互作结构。检查每份载荷的SHA256和状态往返一致性。
3. 原子更新最佳快照索引后，再删除各旧轮次和被替换最佳快照的张量、HDF5、SDF/PDB等结构文件。
   删除前保存文件名、大小和SHA256；保留所有JSON/XML、prompt、响应、失败日志及逐步分析记录。
4. 后续Agent读取全部历史分析文档、最新轮与TOP1的诊断，只允许从本链TOP1恢复。
   失败轮次参与反思，但不能再从已删除的失败分支直接重启。此约束写入技能及动作校验。

原始R0作为V1/V2共用的不可变参考单独保留一次，以保证V2复用同一初始实验。
除此之外，每个版本只保留每条链当前TOP1；当前轮的临时结果在分析和校验结束后清理。
清理期间可能同时存在旧TOP1、新TOP1和当前轮数据，因此仍需预留临时空间。
所有轮次的分数与分析仍参与统计，删除结构不等于删除失败尝试或缩减100条链的分母。

新调度从最早修改区间入口恢复，保留全局时间并重新推理全部后缀。类别重新按当前概率和原生
RNG抽样，不复制旧离散输出。连续Gaussian噪声直接记录，不能将总位移当成确定性漂移。

只有原生重启逐位相同、非目标槽位隔离及混合父状态恢复验收通过，才启用批量合并。
否则采用完整20槽位上下文逐候选分支，不降为batch1。
最佳快照中的任一节点若无法安全合并，也保留每条链对应的完整原生批上下文；
这些上下文服务于精确恢复，不作为额外候选计数。因此该回退模式节省空间的幅度较小。

## DeepSeek与技能

使用deepseek-flash、thinking enabled、reasoning_effort high、结构化JSON；
并发4，输入软预算32 KiB、硬预算128 KiB（本地UTF-8字节保护），输出16K tokens；网络重试3次，格式修复1次。
实际API usage作为token计数依据，本地使用保守UTF-8字节上界进行输入保护。

候选skill优先诊断0.5..1.0，优先控制0.1..0.5；比较最新、历史最优、全部历史方向。
诊断覆盖实际位移/裁剪、局部误差、匹配变化、类型不确定性、原生漂移与引导夹角，
区分中间affinity head与最终pic50_on_rescore。输出假设、证据、动作、预期与反证条件。
禁止重复调度、奖励修改或未提供的证据。每候选有独立记忆。

## 两个完整技能版本

R0共1000轨迹，V1最多10000后缀，统一GPT复盘后V2最多10000后缀，总计最多21000终态结果。
独立起点仍每版本1000；迭代分支不是额外独立样本。
V1全部完成后生成每50条决策一个审阅分片，保留全部prompt、证据、结果及引用。
GPT按分片审阅并记录覆盖，修订skill后本地测试/提交/push，远端拉取指定commit再运行V2。
V2复用相同R0并清空候选记忆，不直接移植V1的teacher/slot最优调度表。
当前任务是同一批开发样本上的技能改进，不宣称跨target泛化。

## 运行与交付

统一入口 `python -m evomolsteer.adaptive_schedule.cli`：
prepare、validate、run --skill-version v1|v2 --resume、profile、compact、status、review-pack、analyze。
远端shell入口 `scripts/run_plip_adaptive.sh`，传入background启动监督进程。
新协议结果根目录 `/opt/MolSteer/generated_datasets/plip_schedule_agent_top1_5nn4_5ost_s42_20261021/`。
旧V1原结果目录和部署目录在当前容器中缺失；不能将新协议运行冒充旧V1续跑。
必须显式prepare并对新代码重新validate。background在缺少PREPARED时直接拒绝启动。

每轮代码/技能修改必须本地提交，推送GitHub后远端运行固定commit，不能在线改远端科学代码。
成功决策和分析不可覆盖；终态先落盘，经评价与最佳快照校验后才删除旧结构；三次恢复失败后报告异常。
不修改已有批次大小或精度以绕过OOM。

报告每教师100条链的Mean、Best、Top10、分位数、教师恢复/超越率、首次达到轮次、
原始/SMILES去重/PoseBusters通过分层、配对R0及V2-V1差异、五批探索性bootstrap、
PLIP恢复/新互作、局部几何和类型/电荷、唯一性、耗时和LLM用量。
不足10个有效结果的Top10记为空，不减少Top10定义。共享历史基线不重复计数。

远端作业可脱离SSH独立运行。2026-10-21依用户要求删除了本线程定时任务`plip`，
不自动检查状态或唤醒GPT复盘；后续由显式调用推进。V2仍须完成真实全量复盘后才能启动。
存储优化不能替代持久化备份；当前`/opt`结果曾随容器回退而缺失，重新开展长任务前应确定持久化保存位置。
完整checkpoint重启、RNG隔离、调度边界、原生路径、位移上限、失败恢复和统计口径必须验收。

## 方法参考

- 区间引导：https://arxiv.org/abs/2404.07724
- APG：https://arxiv.org/abs/2410.02416
- Reflexion：https://arxiv.org/abs/2303.11366
- GEPA：https://arxiv.org/abs/2507.19457
- ACE：https://arxiv.org/abs/2510.04618

借鉴以上研究的诊断和经验更新方法，不改变本实验的梯度方向、奖励或模型权重。
