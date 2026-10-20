# PLIP局部梯度引导：LLM幅度调度与技能迭代

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
- 每个候选独立优化，R0后最多10轮；达到教师分数后继续。保留历史最优及全部失败。
- 历史Unguided/Steer不重采样，不设零引导实验组；不使用旧“互作恢复增加5个百分点”门槛。

## 完整checkpoint及分支

保存0、0.1、…、0.9、1的完整恢复状态、最终输出和每步噪声/漂移/引导记录。
checkpoint包含current、self-conditioning、prior引用、蛋白编码引用、times、积分器状态、
完整Python/NumPy/Torch CPU/GPU RNG、坐标变换、来源哈希。附属head文件保存原始预测与endpoint。
共享context和未改变前缀以校验过的引用去重。

新调度从最早修改区间入口恢复，保留全局时间并重新推理全部后缀。类别重新按当前概率和原生
RNG抽样，不复制旧离散输出。连续Gaussian噪声直接记录，不能将总位移当成确定性漂移。

只有原生重启逐位相同、非目标槽位隔离及混合父状态恢复验收通过，才启用批量合并。
否则采用完整20槽位上下文逐候选分支，不降为batch1。

## DeepSeek与技能

使用deepseek-flash、thinking enabled、reasoning_effort high、结构化JSON；
并发4，输入软预算32K、硬预算64K，输出16K；网络重试3次，格式修复1次。
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
prepare、validate、run --skill-version v1|v2 --resume、profile、status、review-pack、analyze。
远端shell入口 `scripts/run_plip_adaptive.sh`，传入background启动监督进程。
结果根目录 `/opt/MolSteer/generated_datasets/plip_schedule_agent_5nn4_5ost_s42_20261021/`。

每轮代码/技能修改必须本地提交，推送GitHub后远端运行固定commit，不能在线改远端科学代码。
成功分支不可覆盖；终态先落盘，评价失败只重跑评价；三次恢复失败后报告异常。
不修改已有批次大小或精度以绕过OOM。

报告每教师100条链的Mean、Best、Top10、分位数、教师恢复/超越率、首次达到轮次、
原始/SMILES去重/PoseBusters通过分层、配对R0及V2-V1差异、五批探索性bootstrap、
PLIP恢复/新互作、局部几何和类型/电荷、唯一性、耗时和LLM用量。
不足10个有效结果的Top10记为空，不减少Top10定义。共享历史基线不重复计数。

远端作业脱离SSH独立运行；本线程heartbeat初始2小时一次，根据耗时调整1..6小时，
状态无变化时安静，V1结束执行复盘、V2结束报告并停止heartbeat。
完整checkpoint重启、RNG隔离、调度边界、原生路径、位移上限、失败恢复和统计口径必须验收。

## 方法参考

- 区间引导：https://arxiv.org/abs/2404.07724
- APG：https://arxiv.org/abs/2410.02416
- Reflexion：https://arxiv.org/abs/2303.11366
- GEPA：https://arxiv.org/abs/2507.19457
- ACE：https://arxiv.org/abs/2510.04618

借鉴以上研究的诊断和经验更新方法，不改变本实验的梯度方向、奖励或模型权重。
