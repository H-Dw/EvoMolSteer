# 后台运行与主 Agent 接续

## 2026-10-22最新状态：群体分析、默认4假设，正式实验暂停

最新要求已改为4个私有组内Agent并行，每组100候选、提2个假设，再由第5个汇总Agent从8个中选4个排序。
新入口`scripts/plip_group_council.py --manifest <四组证据清单> --output <独立分析目录>`只执行分析，不启动GPU。
保存独立reflection.md/json、供后轮读取的published_memory.json及全量MASTER_REVIEW_INPUTS.json；后者不是审阅批准。
`teacher_pipeline.run_teacher_pipeline`提供单GPU所有者、两教师就绪队列和LLM/CPU重叠调度，质量与TOP1发布完成后才允许下一轮。
它要求显式注入经过验收的原生推理和清理适配器；当前尚未接入正式100H混合父状态执行器，不可用旧20槽位runner代替。
当前正式实验和定时任务仍暂停。早先单会话4假设实测只作历史证据，不代表已经运行4+1新流程。
最新真实4+1验证现已完成：固定代码`1bb51aba`，目录`operations/council_validation_1bb51aba`。
其中四组实际API输入归属核验通过，汇总八选四成功；12次成功请求及另1次失败请求均保留。
汇总32K输出预算独立于组内16K，输入仍限制128 KiB。诊断反思存在已标注的科学措辞误读，不能导入正式历史。
完整记录见[council_validation_20261022/README.zh-CN.md](council_validation_20261022/README.zh-CN.md)。

当前教师仅原Top1/Top2。下文r123三教师部署段落属于历史记录，不能据此恢复旧三教师任务。
最新决策/预算见[group_hypotheses.zh-CN.md](group_hypotheses.zh-CN.md)：每教师100候选一起进行Agent分析，每假设共用四区间幅度。
容量按8→4→3→2测量；8×100已OOM，4×100两教师均完成100步，默认4，不再测3/2。
固定容量代码`dd942431`，结果`operations/hypothesis_wave_8432_dd942431`：179.0/180.2秒，峰值45.69/45.78 GiB。
早先未包含3档位的探测114078已按新指令停止，保留失败及部分结果；修订探测114995完成后退出。

群体规划器代码`fa5a5779`已本地测试提交推送，并经GitHub API校验与Git bundle部署；远端81项测试通过。
`operations/group_agent_smoke_fa5a5779`完成真实Top1全100候选、R0–R4历史的Agent分析，2次API请求和4个主动计算工具调用，输出4个假设。
未启动这些假设的正式分子生成。暂停的旧状态、已完成数据和请求均保留，`plip-11762`仍PAUSED。
新群体规划功能已可执行；正式100H混合父状态生成/保留流程尚未接入，不能调用旧逐链runner替代。

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

`background`是监督进程入口；正式后台运行使用经过连接检查的启动器（在已配置API和代理环境的交互式shell、固定checkout目录执行）：

```bash
RUN_ROOT=/opt/MolSteer/generated_datasets/plip_schedule_agent_top1_5nn4_r123_i4_s42_20261021
/opt/miniforge3/envs/molsteer-flowr-dtk/bin/python scripts/launch_plip_adaptive.py --root "$RUN_ROOT" --skill-version v1
```

启动器先验证带认证的DeepSeek连接，检查运行锁，再脱离SSH；完整继承父进程环境（包括大小写代理变量），不序列化环境值。
连接不通时不启动worker；连接检查成功仅代表当时API可达，仍须核对实际决策及轮次提交。
每次启动在operations下保存独立请求、PID和此前控制/错误记录的压缩归档，日志追加而非覆盖。
非交互式shell可能跳过用户环境配置；不能只传递API密钥而漏掉代理。不得输出shell配置文件或代理凭据。

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

用户2026-10-22最新指令将首轮巡检调整为每20分钟；确认连续稳定推进后，按完整轮次耗时逐步延长，可至1–6小时。
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

## 旧10104三教师四区间部署记录

- 科学代码提交：`f7868656bc68bedf7a0adb1cae62063b52cb263b`，已本地提交、推送GitHub并在远端拉取固定提交。
- 固定运行目录：`/opt/MolSteer/runtimes/plip_schedule_f7868656`。
- 结果目录：`/opt/MolSteer/generated_datasets/plip_schedule_agent_top1_5nn4_r123_i4_s42_20261021`。
- 本地相关测试40通过、3因缺少科学依赖跳过；远端相同测试43通过、无跳过，包含真实Torch/RDKit恢复与清理。
- prepare已核对三位教师原排名1/2/3、15个teacher/batch、300条链及每版本3000次决策；原始五批种子不变。
- V1监督进程PID为8901，启动参数与固定提交见结果目录的`launch_request.json`和`launch.json`。
  首次检查进程存活、状态为`validating`；此状态仅代表已启动原生恢复验收，不代表验收通过或生成完成。
- 全部恢复验收及梯度检查通过后，由同一后台进程自动进入迭代生成；轮末先评价、提交最佳快照再清理结构。
  不新增定时任务；V2仍需真实全量审阅和批准文件。

## 11762新机部署记录

- 科学代码提交：`403460e54b6ef497360f2df3be3a1bb071b83e7f`，已本地提交并推送GitHub，新机从GitHub拉取固定提交。
- 部署目录：`/opt/MolSteer/runtimes/plip_schedule_403460e5`；使用稀疏检出保留src/scripts/tests/skills及本计划目录，不覆盖`/opt/MolSteer`原主仓库。
- 原v2权重从本地已核验缓存上传，连接中断后使用SFTP断点续传；新机校验全部677580035字节的SHA256符合原始记录后，原子发布至FLOWR checkpoints目录。现有v2.2权重保持不变。
- 新机运行环境：Torch2.9.0、HIP6.3.26093、BW设备；完整软件和硬件清单见`deployment_environment.json`，不宣称与历史机器逐位一致。
- 本地相关测试55通过、3因依赖跳过；新机相同测试58通过、无跳过。
- prepare确认3教师、15个teacher/batch、300条链、每版本3000次决策；未导入旧机R0或优化历史。
- 独立后台验收PID614，日志`preflight.log`，请求记录`preflight_launch_request.json`。
  全部11个恢复入口的终态/RNG/重评分一致，槽位隔离和混合父状态逐位一致，允许批量执行。
  有限差分、坐标单位、RNG隔离及伪Huber检查通过；affinity VJP调用数为0，`PREFLIGHT_COMPLETE.json`已生成。
- 已核对旧10104监督进程8901的精确启动路径和进程组，发送SIGTERM后确认其与worker10585均退出。
  旧状态标记superseded，历史5批文件和全部已有数据保留，旧目录`operations/superseded_by_11762.json`记录此前状态。
  新目录`HANDOFF_COMPLETE.json`绑定旧交接记录及新验收哈希，不再巡检旧服务器。
- 正式V1已在新机后台启动，初始监督PID2629，记录`formal_launch_request.json`；实际后续进程以`launch.json`和`workers.json`为准。
  R0及10轮优化使用新机原生过程，V2等待真实全量复盘与严格审阅校验。
- 定时任务已恢复，ID为`plip-11762`，每2小时检查并按实际耗时调整至1–6小时；只在可操作变化时通知，V2报告交付后删除。

## 2026-10-22首次巡检：连接环境修复

- 实际检查15个预期batch：Top1五批共100条R0已提交；Top2/Top3尚未开始，已提交优化轮次为0。
  Top1的R1多批次均出现DeepSeek网络重试耗尽；`status.json`虽然显示running，不代表有效推进。
- `/proc`核验原监督2629与worker2664都有API密钥但均无代理变量。交互式环境继承既有代理后，
  带认证的模型列表请求返回200；相同环境禁用代理则连接失败。诊断不输出密钥、代理值或异常正文。
- 已核对进程组2629并停止监督与worker，确认两者退出。已有R0、请求及错误保持原位，
  诊断保存在`operations/network_failure_20261022/inspection.json`。
- 新增`scripts/launch_plip_adaptive.py`，在启动前测试认证连接并完整继承环境；失败时拒绝启动。
  续跑前归档旧控制文件与错误摘要，避免原重试编号再次使用时丢失旧失败记录。
- 本次仅修改启动器、测试和运行文档，采样源码指纹、冻结配置、教师、技能、初态与R0不变。
  新提交需经本地测试和GitHub推送后在远端固定检出，核对源码指纹后通过启动器续跑。
- 已部署`fc95e314544b447f6d21cf9ed5117deafb1f827b`至`/opt/MolSteer/runtimes/plip_schedule_fc95e314`。
  新增启动器8项测试本地通过，原相关测试55通过、3因依赖跳过；远端合计66通过。
  `assert_prepared`通过，科学源码指纹与PREPARED一致，未修改冻结配置。
- 通过启动器重启后的监督PID24402、worker24438已核验均继承API与代理变量。
  启动收据位于`operations/launch_20261021T162035Z_e7653243/`，时间为UTC。
  状态恢复至Top1 batch0 R1；启动成功不等于该轮已提交，下一步须确认真实API收据和历史推进。
  已有100条R0的质量评价全部完成，零评价执行失败；结果目录当时约213 MiB，剩余空间约1.17 TiB。

## 2026-10-22 最新指令：暂停、两教师、批量验收

- 用户将范围改为原Top1、Top2。已核验监督24402与worker24438的启动参数和进程组并停止，确认退出。
  Top1五批均完成R4，保留100条R0、400次优化结果及尚未消费的R5请求/响应。Top2未开始，Top3排除。
- 暂停前快照位于旧r123结果根目录的`operations/user_pause_top12_batch_probe_20261022/pause_snapshot.json`；
  `status.json`标记`paused_by_user`，定时任务`plip-11762`已设为PAUSED。
- 新默认两教师协议为200条条件链，每版本2000次决策；默认新目录为`plip_schedule_agent_top1_5nn4_r12_i4_s42_20261022`。
  不重新prepare旧目录，不改写旧冻结config/binding。恢复前须明确引用/迁移已有Top1工作并重新核验部署。
- `scripts/benchmark_plip_batches.py`只生成容量诊断，不调用Agent，不进入科学统计；测试完整100步、真实梯度、
  checkpoint及重评分。原20槽位与更大物理批量分别报告随机流约束，不能仅以不OOM决定切换。
- 新增单GPU所有者与两教师交替的`interleaved_prefetch`执行器。全局API并发仍为4，最多当前和下一批请求，
  预取结果必须匹配冻结历史/总体证据哈希；各教师完整上一轮提交后才开放下一轮。
  不并发修改模型或清理同一候选数据。该代码部署和验收不代表正式任务已经恢复。
- 本次任务结束仍保持正式实验及自动巡检暂停。具体批量结果见`batch_capacity_20261022.zh-CN.md`。
- 用户随后要求从100逆序、首次两教师共同成功即停止，允许终态差异。已停止升序诊断进程组110702及尚未执行的预测头等价审计111985，
  保留原诊断及失败记录；修订记录为`operations/batch_probe_user_revised.json`。新的下降搜索使用固定提交`2fbb0890`。
- 下降搜索完成标记确认两教师在100各通过两次，选择100并立即停止；实际耗时51.2642/51.5017秒，设备峰值12.4624 GiB。
  诊断进程112225正常退出；复核正式24402/24438、取消的110702/111985均已退出。`status.json`仍为`paused_by_user`。
  本地相关测试68通过5跳过，远端完整73通过（24.73秒），日志`operations/batch_capacity_remote_tests_2fbb0890.log`。
- 所有代码本地提交推送后部署。GitHub Git传输持续超时，因此改用SSH传送Git bundle，校验bundle SHA256，
  再通过GitHub API核验已发布commit，与远端`git fetch <bundle> HEAD`和固定worktree一致。没有直接在远端改写科学源码。
- 100已作为容量选择；正式Agent仍保留20槽位记录布局和恢复保护，后续100槽位适配尚未启用。
  不将诊断候选纳入正式统计，不自动续跑。定时任务仍PAUSED，提示已写明100通过和取消跨批量终态一致要求。

## 2026-10-22 00:58（北京时间）进展与巡检间隔

- 11762监督24402与worker24438身份及进程组核验正常。Top1全部100条链已提交R1、R2，
  第三轮batch0另提交20条：累计220/3000次V1优化结果；Top2/Top3按原教师顺序等待运行。
- 先前11条批次网络错误均已有对应轮次COMPLETE，重启后无新增批次错误。
  当时235个成功API收据，13个格式校验失败均已通过既定格式修复取得有效响应，失败证据保留。
- 11个完成的轮次批次均有RETENTION_COMPLETE；320条R0/优化结果完成质量评价，零评价执行失败。
  结果目录约416 MiB，剩余磁盘约1197.6 GiB。
- Top1固定R0/R1/R2的有效数均为100，Mean分别4.1476/4.1295/4.1547；
  截至R2逐链TOP1集合Mean为4.1561。这只是进行中的开发样本结果，不是最终效益结论。
- R1全部提交时间为UTC16:38:28，R2为16:53:23，完整第二轮约14分55秒。
  已将现有`plip-11762`改为每20分钟，后续依据实际轮次耗时及异常情况调整，不新增重复任务。
