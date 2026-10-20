# 后台运行与主 Agent 接续

目标服务器固定为 `root@ksai.scnet.cn:10104`。凭据仅使用本线程已提供的信息；
DeepSeek 密钥由远端环境变量 `DEEPSEEK_API_KEY` 读取，不写入仓库。

结果根目录：
`/opt/MolSteer/generated_datasets/plip_schedule_agent_5nn4_5ost_s42_20261021`

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
```

`background` 由独立 session 启动并将输出重定向至结果目录日志，SSH 断开后继续。
不要同时启动两个 supervisor 或 GPU worker。锁文件存在不等于进程仍在运行；
检查 `workers.json` 中的进程和最近进度后再恢复。

`VALIDATED.json` 确认两个 target 的逐节点重启和批内隔离检查。
`validation/gradient_checks.json` 确认预测头有限差分、坐标单位和 RNG 隔离。
`status.json` 表示采样状态，`quality_status.json` 表示 PLIP/PoseBusters 队列状态。
`versions/v1/COMPLETE.json` 只表示采样完成；`FINALIZED_v1.json` 才表示最终评价扫描与报告结束。
评价失败保留失败记录，不补抽分子。

每个 teacher/batch 的 `histories.json` 为20条候选链；每条完整链有 R0 加10轮记录。
完整 V1 应有50个批次、1000条链、10000次独立 LLM 决策。
完整前缀按哈希引用共享，不重复保存；随机流、self-conditioning 和积分器状态均在恢复快照内。

## V1 完成后的统一复盘

1. 读取 review skill、冻结的 optimizer skill、全量指标与 `review/v1/manifest.json`。
2. 按 manifest 分片逐一分析，记录已审阅分片及 SHA256。每条决策均关联实际 prompt 证据。
   先对重复固定输入去重，再审阅每轮差异；异常案例回读原始请求、校验错误及完整节点数据。
3. 比较几何响应与亲和力响应、裁剪和实际引导量、不同区间/方向、有效性及失败假设。
   不将10轮分支视为额外独立候选，不只挑选成功案例。
4. 生成通用技能修改及适用边界，不建立 teacher/slot 对应的最优调度表。
5. 本地修改、测试、提交、推送；远端拉取固定新 commit。
   技能文件在采样代码指纹之外，因此只改技能可复用 R0；修改采样代码须重新审计与验证。
6. 完成全量覆盖后写入 `versions/v2/review_approval.json`：
   包含 V1 manifest 哈希、全部已审阅分片及哈希、决策覆盖数、复盘文档、
   新技能 SHA256、Git commit 和修改理由。文件必须对应真实完成的审阅。
7. 使用新技能启动 `background --skill-version v2`，复用原 R0，清空 V2 候选记忆。

V2 结束后比较 Mean、Best、Top10、恢复率、质量和成本，分析教师 PLIP 数量、核心覆盖率、
教师 pIC50 与改善难度的关系。结果限定于这两个 target、十个教师和同一组开发种子。

## 自动接续

本线程 heartbeat 名称为“PLIP 调度实验与技能迭代”，ID `plip`，初始间隔2小时。
健康且无可操作变化时保持安静；根据实测耗时调整至1–6小时。
统一复盘可分多次唤醒推进，覆盖台账持续保存。V2 报告交付后暂停 heartbeat。

代码异常在本地修复、提交并推送后再部署。成功分支和成功 API 调用复用；
请求、响应、失败记录不得被不同输入覆盖。已失败的早期预检目录保留用于审计。

## 输入容量调整

首轮巡检发现部分候选在第9–10轮超过64 KiB的本地输入字节上界。
输入此前已经按区间汇总逐步诊断，保留完整历史；本次将上界调整为128 KiB。
这不会修改发给模型的消息、模型、输出预算、奖励、技能或采样配置。
待 supervisor 和 worker 退出后，运行 `adjust-input-budget`，再续跑 background。
该命令同时持有两个运行锁，归档原配置、版本绑定、预算与错误记录，并以事务日志
更新输入上限及配置哈希。成功请求继续复用，不因调整容量重新采样。
运行中的 config.json 是调整后的权威配置；不要重新调用 prepare 覆盖它。
