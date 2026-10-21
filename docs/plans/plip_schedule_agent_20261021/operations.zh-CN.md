# 后台运行与主 Agent 接续

目标服务器固定为 `root@ksai.scnet.cn:10104`。凭据仅使用本线程已提供的信息；
DeepSeek 密钥由远端环境变量 `DEEPSEEK_API_KEY` 读取，不写入仓库。

结果根目录：
`/opt/MolSteer/generated_datasets/plip_schedule_agent_top1_5nn4_r1345_s42_20261021`

这是5nn4原Top1/3/4/5四教师、逐链TOP1协议的新目录，不是原V1结果的恢复。2026-10-21核验时原V1结果目录及
原部署目录在当前容器中均不存在；当前`/opt/MolSteer`为较早快照，不能据此推断实验已完整完成。
不要重新prepare旧目录或用旧VALIDATED标记启动新代码。

Git 分支：`codex/plip-adaptive-schedule`。
远端每次使用 `/opt/MolSteer/runtimes/` 下的固定 commit checkout，保留运行中的旧 checkout。
实际运行版本查 `launch.json`、`versions/v1/deployment.json` 和 `workers.json`。

## 入口与状态

固定 checkout 下的 `scripts/run_plip_adaptive.sh` 提供：

```bash
bash scripts/run_plip_adaptive.sh prepare
bash scripts/run_plip_adaptive.sh validate
bash scripts/run_plip_adaptive.sh check-gradients
bash scripts/run_plip_adaptive.sh background --skill-version v1
bash scripts/run_plip_adaptive.sh status
bash scripts/run_plip_adaptive.sh analyze --skill-version v1
bash scripts/run_plip_adaptive.sh review-pack --skill-version v1
bash scripts/run_plip_adaptive.sh report --skill-version v1
bash scripts/run_plip_adaptive.sh compact --skill-version v1
```

`background`是监督进程入口；远端后台启动时使用下面的命令脱离SSH并保留日志（在固定checkout目录执行）：

```bash
ROOT=/opt/MolSteer/generated_datasets/plip_schedule_agent_top1_5nn4_r1345_s42_20261021
nohup setsid bash scripts/run_plip_adaptive.sh background --skill-version v1 > "$ROOT/supervisor_v1.log" 2>&1 < /dev/null &
```

先完成prepare；可手动validate及check-gradients，或由background在首次启动时依次执行。两项均通过后才允许开始Agent轮次。密钥须由启动进程环境继承，禁止放入命令行或日志。
它要求已有PREPARED，不自动创建新实验。变更采样代码后必须显式准备新协议并重新验证。
不要同时启动两个 supervisor 或 GPU worker。锁文件存在不等于进程仍在运行；
检查 `workers.json` 中的进程和最近进度后再恢复。

`VALIDATED.json` 确认所选5nn4 target的逐节点重启和批内隔离检查。
`validation/gradient_checks.json` 确认预测头有限差分、坐标单位和 RNG 隔离。
`status.json` 表示采样状态，`quality_status.json` 表示 PLIP/PoseBusters 队列状态。
`versions/v1/COMPLETE.json` 只表示采样完成；`FINALIZED_v1.json` 才表示最终评价扫描与报告结束。
评价失败保留失败记录，不补抽分子。

每个 teacher/batch 的 `histories.json` 为20条候选链；每条完整链有 R0 加10轮记录。
完整V1应有20个teacher/batch、400条教师条件链、4000次独立LLM决策。
每个批次必须有20条完整R0–R10历史及轮末清理提交标记；陌生教师目录或仅有COMPLETE文件不能凑足完成数。
完整前缀按哈希引用共享，不重复保存；随机流、self-conditioning 和积分器状态均在恢复快照内。

## 每轮清理与恢复

- 当前仅使用5nn4原Top1/3/4/5，保留原排名及teacher ID。prepare从十教师历史归档选择这四位，
  严格核对node ID、原排名和PLIP核心，不需要为被排除教师重新采样或准备模板。
- `PREPARED.json`绑定config和teachers哈希；新选择不能覆盖旧实验。`teacher_selection.json`保存来源、纳入与排除清单及工作量。
- 新配置`retention.mode=top1_per_chain`。20个槽位各自选择历史最高有效pIC50，
  每教师仍为100条链。R0作为跨版本共用参考单独保留；V1/V2分别保留自身各链TOP1。
- 每轮提交历史后，GPU worker同步等待本轮质量评价，随后建立并校验自包含最佳快照，
  原子发布`versions/<version>/<teacher>/batch_###/retention.json`后清理旧结构。
- `retained/<version>/<teacher>/batch_###/best_<hash>/`保存当前TOP1的结构与恢复状态。
  `COMPLETE.json`列出全部载荷哈希；任一文件缺失/损坏时停止，不继续删除原始数据。
- `PRUNED.json`保留每份被删载荷的名称、大小和哈希。各轮的历史、节点摘要、逐步诊断、
  全部prompt/响应、PLIP XML、互作JSON、质量失败和错误日志保留。报告从最佳快照定位分子文件。
- `compact`用于在独占worker锁下补完已提交轮次的清理；它不会生成新分子。
  不允许独立`profile --watch`与TOP1 worker并行，以免质量评价读取正在删除的结构。
- 断电发生在快照发布前时原数据仍在；发生在删除期间时按索引和删除日志幂等继续。
  Agent只能以TOP1为恢复父节点，已删失败分支通过分析文档参与回溯。
- 合并快照要求所有原生上下文与RNG相容，否则保留逐槽完整20分子上下文。
  保留上下文不是增加独立样本。空间峰值包含当前轮、旧最佳、新最佳及共享R0。
- 独立profile和报告只读取尚未完成的评价；已清理分支的分析应已完成或有明确失败记录。
  若质量依赖缺失则停止清理，避免丢失待评价输入。

## V1 完成后的统一复盘

1. 读取 review skill、冻结的 optimizer skill、全量指标与 `review/v1/manifest.json`。
2. 按 manifest 分片逐一分析，记录已审阅分片及 SHA256。每条决策均关联实际 prompt 证据。
   先对重复固定输入去重，再审阅每轮差异；异常案例回读原始请求、校验错误及节点分析。
   已清理分支仅有分析文档与删除清单，不得声称检查了其原始张量。
3. 比较几何响应与亲和力响应、裁剪和实际引导量、不同区间/方向、有效性及失败假设。
   不将10轮分支视为额外独立候选，不只挑选成功案例。
4. 生成通用技能修改及适用边界，不建立 teacher/slot 对应的最优调度表。
5. 本地修改、测试、提交、推送；远端拉取固定新 commit。
   技能文件在采样代码指纹之外，因此只改技能可复用 R0；修改采样代码须重新审计与验证。
6. 完成全量覆盖后写入 `versions/v2/review_approval.json`：
   包含 V1 manifest 哈希、全部已审阅分片及哈希、决策覆盖数、复盘文档、
   新技能 SHA256、Git commit 和修改理由。文件必须对应真实完成的审阅。
7. 使用新技能启动 `background --skill-version v2`，复用原 R0，清空 V2 候选记忆。

V2结束后比较TOP1系统输出及固定R10生成分布的Mean、Best、Top10、恢复率、质量和成本；
`fixed_round_distributions.csv`保存全部轮次的未跨轮择优统计，部分完成状态不填充未执行轮次。
分析教师PLIP数量、核心覆盖率、教师pIC50与改善难度的关系时明确仅有四位教师、两个来源谱系。
结果限定于这一个target、四个经筛选教师和同一组开发种子，历史100个Unguided/Steer不重复计数。

## 定时任务已删除

2026-10-21按用户要求删除ID为`plip`的定时任务。不自动巡检，不自动唤醒主Agent；
统一复盘和V2启动需要后续显式执行，但覆盖台账和review approval要求保持不变。
本次用户已指定开展5nn4四教师V1探索；通过原生重启与梯度验收后由后台流程进入迭代。V2仍等待真实全量复盘，不自动跳过审阅。

代码异常在本地修复、提交并推送后再部署。成功分支和成功 API 调用复用；
请求、响应、失败记录不得被不同输入覆盖。预检分析保留，验证成功后清理其临时结构载荷。
`/opt`不是持久化备份。本次优化降低增量存储，不能恢复缺失的旧V1结果，
也不能防止容器回退丢失数据；长任务应另行落实结果持久化和可恢复备份。

## 输入容量调整

首轮巡检发现部分候选在第9–10轮超过64 KiB的本地输入字节上界。
输入此前已经按区间汇总逐步诊断，保留完整历史；本次将上界调整为128 KiB。
这不会修改发给模型的消息、模型、输出预算、奖励、技能或采样配置。
待 supervisor 和 worker 退出后，运行 `adjust-input-budget`，再续跑 background。
该命令同时持有两个运行锁，归档原配置、版本绑定、预算与错误记录，并以事务日志
更新输入上限及配置哈希。成功请求继续复用，不因调整容量重新采样。
运行中的 config.json 是调整后的权威配置；不要重新调用 prepare 覆盖它。

## 本次四教师部署记录

- 科学代码提交：`1d178b7b87101223cecbdc0f82749b113a688113`，已本地提交并推送GitHub。
- 固定运行目录：`/opt/MolSteer/runtimes/plip_schedule_1d178b7b`。
- 本地相关测试34通过、3因环境依赖跳过；远端相同测试37通过，包含真实Torch/RDKit恢复与清理。
- 已在新结果根目录完成prepare：4教师、5套共享初始批次、20个teacher/batch、每版本400条链与4000次Agent决策。
- 已提交独立session后台V1启动；启动请求在`launch_request.json`，实际进程与验收进度以`launch.json`、`status.json`和`validation/progress.json`为准。
- 启动后台进程不代表验收或科学实验已完成。监督进程先做完整原生重启验收和保存状态的梯度检查，全部通过才开始Agent决策；不重建定时任务。
