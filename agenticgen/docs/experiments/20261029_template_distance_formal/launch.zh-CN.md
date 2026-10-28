# 正式后台运行启动记录

2026-10-29 02:20:37（Asia/Shanghai）已启动独立后台任务。科学实现为 `777bfa363d349dc03311b5e396f133b0c3e91ac1`；增加可审计启动脚本后的执行提交为 `11258033b9aecbb5c99958a856d139da9b8950ac`，均已推送 GitHub。

- 执行目录：`/workspace/uniguide_runs/template-distance-777bfa36`，分支 `codex/uniguide-flowr-agent`。
- 结果目录：`/workspace/agenticgen_template_distance_20261029_777bfa36`。
- 冻结配置：`/workspace/uniguide_runs/template-distance-777bfa36-config.json`。
- 控制进程初始 PID：114755；验收子进程初始 PID：114757。PID 仅用于定位，实际状态同时核对命令、锁、心跳及产物。
- `campaign.log`、`STATUS.json`、`HEARTBEAT.json` 记录当前阶段；阶段日志分别为 `acceptance.log`、`calibrate.log`、`formal.log`。

远端 108 项测试通过，耗时 43.05 秒。DTK / PyTorch 2.9.0 检测到 BW GPU，API 与代理仅核验是否加载，不记录密钥。GitHub 直连和 ghfast.top 超时后，gh-proxy.com 拉取成功，无需 Git bundle。三个质量模块与已验证 CPU 提交 `3f07d1fa660cc46afb8f3c256cd45e6ab54024c7` 字节一致。

两份输入直接来自原教师 SDF 的三维构象。6pyy 17 个重原子、3w54 22 个重原子，各保留原九个 PLIP 条件原子身份。不存在构象重嵌入、旧最佳得分复制或跨教师 scale 平均。`interact_atoms=[]`，起始三项权重 1/0/0。源文件保护清单独立保存。

执行顺序为真实模型验收 → 两教师独立新标定 → GD01 每教师 5 轮真实 LLM 引导。标定最多 700 次新尝试，正式最多 1500 次；小人口晋级复用原随机槽位，重复配置不视为独立验证。固定四模块 skills 2.2.0，每教师每轮最多 30 次 HTTP 调用，失败同样计数。采样结束后等待本专项 10/50 诊断，停止本专项空闲 worker；不启动 GD02，也不恢复 UG05。

02:21:56 实查：真实验收进程存活，两锁持有；6pyy 已产生禁用指引、原生前缀、连续及恢复轨迹，采样推进。此时尚未完成全部验收，也尚无新标定、正式分子结果或真实 LLM 调用；不能将已配置的后续阶段当作已完成。后台控制器仅在前置阶段成功后继续；失败保留日志及账本并停止，不自动重试。

部署原始证据见同目录 `deployment_audit.json`。后续状态以远端实际进程和最新阶段产物为准，本文为启动时快照，不是最终结果报告。
