# 后台运行与主 Agent 接续

当前唯一实验服务器为 `root@ssh.zzai.scnet.cn:11762`。凭据仅使用本线程已提供的信息；
DeepSeek 密钥由远端环境变量 `DEEPSEEK_API_KEY` 读取，不写入仓库。
旧10104服务器仅用于原权重取得与安全停止旧任务；其结果不得计入新机实验。

结果根目录：
`/opt/MolSteer/generated_datasets/plip_schedule_agent_top1_5nn4_r123_i4_s42_20261021`

这是11762新机的完整重跑，使用5nn4原Top1/2/3三教师、逐链TOP1协议，保持相同目录名称。
新机核验时该目录及旧部署不存在；历史基线、教师及初态存在。不要将同名目录与旧机进度混淆，
也不要复用旧机VALIDATED或优化历史。FLOWR位于`/opt/MolSteer/flowr_root`。

Git 分支：`codex/plip-adaptive-schedule`。
远端每次使用 `/opt/MolSteer/runtimes/` 下的固定 commit checkout，保留运行中的旧 checkout。
实际运行版本查 `launch.json`、`versions/v1/deployment.json` 和 `workers.json`。

## 入口与状态

固定 checkout 下的 `scripts/run_plip_adaptive.sh` 提供：

```bash
bash scripts/run_plip_adaptive.sh prepare
bash scripts/run_plip_adaptive.sh validate
bash scripts/run_plip_adaptive.sh check-gradients
bash scripts/run_plip_adaptive.sh background --skill-version v1 --preflight-only
bash scripts/run_plip_adaptive.sh background --skill-version v1
bash scripts/run_plip_adaptive.sh status
bash scripts/run_plip_adaptive.sh analyze --skill-version v1
bash scripts/run_plip_adaptive.sh review-pack --skill-version v1
bash scripts/run_plip_adaptive.sh review-check --skill-version v2
bash scripts/run_plip_adaptive.sh report --skill-version v1
bash scripts/run_plip_adaptive.sh compact --skill-version v1
```

`background`是监督进程入口；远端后台启动时使用下面的命令脱离SSH并保留日志（在固定checkout目录执行）：

```bash
ROOT=/opt/MolSteer/generated_datasets/plip_schedule_agent_top1_5nn4_r123_i4_s42_20261021
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
完整V1应有15个teacher/batch、300条教师条件链、3000次独立LLM决策。
每个批次必须有20条完整R0–R10历史及轮末清理提交标记；陌生教师目录或仅有COMPLETE文件不能凑足完成数。
完整前缀按哈希引用共享，不重复保存；随机流、self-conditioning 和积分器状态均在恢复快照内。

## 每轮清理与恢复

- 当前仅使用5nn4原Top1/2/3，保留原排名及teacher ID。prepare从十教师历史归档选择这三位，
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

严格批准文件使用schema=1。字段为`manifest_sha256`、`finalized_sha256`、`v1_binding_sha256`、
`review_commit`（部署的40位commit）、`skill_path`（仓库相对路径）、`skill_sha256`、
`review_document`、`review_document_sha256`及`coverage`。
coverage每项包含原manifest的`path`与`sha256`，以及已提交的仓库相对`review_file`和`review_sha256`。
每份review_file是JSON，包含`shard_sha256`和`decisions`；每条decision必须有原`teacher/batch/slot/id`、
非空实质性`assessment`及实际prompt中的`evidence_ids`，不得以自动填充套话冒充主Agent审阅。
审阅文档、逐片review文件和改进skill先在本地提交并推送；远端固定到该commit后，才写入包含该commit的批准文件。
批准文件自身保存在结果目录，不写入凭据。`review-check`验证Git中与磁盘上的内容一致，
每次V2运行也强制执行同一校验；缺少全量完成状态、重复/遗漏决策或哈希变化均拒绝启动。

新增统计文件`teacher_equal_summary.json`和`shared_batch_bootstrap.json`按教师等权并同步重采样共享五批；
配对表记录有效性转换及固定R10差异。`case_selection.json`保留最高分、改善幅度中位数及失败/质量冲突案例。
已清理失败轨迹仅展示其分析文档和保存的局部特征，不声称保留了原始完整结构。

V2结束后比较TOP1系统输出及固定R10生成分布的Mean、Best、Top10、恢复率、质量和成本；
`fixed_round_distributions.csv`保存全部轮次的未跨轮择优统计，部分完成状态不填充未执行轮次。
分析教师PLIP数量、核心覆盖率、教师pIC50与改善难度的关系时明确仅有三位教师、两个来源谱系。
结果限定于这一个target、三个经筛选教师和同一组开发种子，历史100个Unguided/Steer不重复计数。

## 恢复定时推进

用户最新指令授权恢复附着本线程的定时任务，默认每2小时，按耗时调整至1–6小时。
巡检只连接11762，读取全部批次完成数、错误是否已恢复、进程身份、质量状态和磁盘增长。
健康无实质变化时安静；V1复盘就绪后按完整覆盖台账推进技能修订及V2启动。
V2报告交付后删除任务。不能跳过真实全量审阅或在轮次中更改冻结技能。

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

## 已停止的旧四教师部署记录

- 科学代码提交：`1d178b7b87101223cecbdc0f82749b113a688113`，已本地提交并推送GitHub。
- 固定运行目录：`/opt/MolSteer/runtimes/plip_schedule_1d178b7b`。
- 本地相关测试34通过、3因环境依赖跳过；远端相同测试37通过，包含真实Torch/RDKit恢复与清理。
- 旧结果根目录为`/opt/MolSteer/generated_datasets/plip_schedule_agent_top1_5nn4_r1345_s42_20261021`；原4教师、400链的配置已经被最新要求替代。
- 变更时旧Top1 batch0已提交R3；已终止经过PID与启动参数核验的旧进程组1187，已有结果和请求未删除。旧目录`operations/superseded_top123_four_intervals.json`保存停止原因和此前状态，状态标记为superseded。
- 启动后台进程不代表验收或科学实验已完成。监督进程先做完整原生重启验收和保存状态的梯度检查，全部通过才开始Agent决策；不重建定时任务。

## 当前三教师四区间协议

- 只运行5nn4原Top1/2/3，每教师100条链。四幅度依次对应10..25、26..50、51..75、76..90步，含两端，共81次有指导更新；总积分100步不变。
- 完整恢复节点为0、10、25、26、50、51、75、76、90、91、100。最早修改区间从10/26/51/76入口恢复，后续全部重新推理。
- Agent优先提升教师100条链的Mean，Best其次。仍逐链独立设计；先生成全部五批R0，再将五批按轮推进。
- 每轮`round_contexts/round_XX.json`保存完整上一轮的总体统计及来源哈希；同轮所有批次用同一份证据。恢复时只截取此前历史验证上下文，避免混入本轮已完成批次。
- 禁止只运行单批次来代替100条链的Mean目标；可用`--teacher`选择完整教师，`--batch`会被拒绝。
- 本协议不得复用旧80步R0或八维调度历史；V1/V2仅在新协议内部共享R0。失败、结构清理、全部文档保留及本地提交后部署规则不变。

## 三教师四区间部署记录

- 科学代码提交：`f7868656bc68bedf7a0adb1cae62063b52cb263b`，已本地提交、推送GitHub并在远端拉取固定提交。
- 固定运行目录：`/opt/MolSteer/runtimes/plip_schedule_f7868656`。
- 结果目录：`/opt/MolSteer/generated_datasets/plip_schedule_agent_top1_5nn4_r123_i4_s42_20261021`。
- 本地相关测试40通过、3因缺少科学依赖跳过；远端相同测试43通过、无跳过，包含真实Torch/RDKit恢复与清理。
- prepare已核对三位教师原排名1/2/3、15个teacher/batch、300条链及每版本3000次决策；原始五批种子不变。
- V1监督进程PID为8901，启动参数与固定提交见结果目录的`launch_request.json`和`launch.json`。
  首次检查进程存活、状态为`validating`；此状态仅代表已启动原生恢复验收，不代表验收通过或生成完成。
- 全部恢复验收及梯度检查通过后，由同一后台进程自动进入迭代生成；轮末先评价、提交最佳快照再清理结构。
  不新增定时任务；V2仍需真实全量审阅和批准文件。
