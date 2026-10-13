# 首个完整条件的实际运行确认

`W00_canonical` 的六条空白上下文 Analyst→Designer 链全部完成，条件状态为 complete。后续条件已由独立后台协调器继续派发。本记录只确认首个条件，不宣称全部条件已经完成。

| 链 | 最终执行选择 | 真实 API action 调用 | 科学 finish 拒绝后修复次数 | Analyst 实际曲线数 |
|---|---|---:|---:|---:|
| 0 | W1_early_soft | 3 | 0 | 1 |
| 1 | retain_M2 | 3 | 0 | 1 |
| 2 | retain_M2 | 3 | 0 | 1 |
| 3 | retain_M2 | 10 | 3 | 3 |
| 4 | retain_M2 | 11 | 4 | 4 |
| 5 | W2_middle_peak | 5 | 1 | 1 |

共 35 次真实 DeepSeek Flash action API 调用、12 份有效角色输出。所有六链均读取实际曲线并为时序假设绑定对应曲线；edit 的 Designer 也引用了相关 Analyst 假设及其曲线。累计 8 次 valid-JSON finish 因科学交付缺失被退回，之后通过实际工具读取与引用修复，而非放松校验或自动补写证据。曲线、完整 API 请求/原始回复/提取回复的校验值、指引来源、角色完成凭证、实际注册奖励编译签名均通过只读审计。

最终众数为 M2（4/6），其余分别 W1、W2（各 1/6）。原数学候选及 M2 具有已完成的同随机输入生成结果，故本任务没有重复生成分子；明确记录“本次 Agent 输出未改变执行函数”。固定 M2 400、无引导 400 与历史 Steer 1,000 的结构、评分、应变、失败及覆盖继续复用。

这说明“实际指引交付→工具读取→假设→Designer 引用→注册程序编译→匹配已生成结果”已走通，不等于时序扩展已改善亲和力。保留 M2 是允许的研究决策，不能为追求 edit 强迫 Agent 选择更差的奖励。

复查入口：`scripts/check_temporal_role_condition.py --config configs/experiments/temporal_guidance_evidence_fix_20261013/runner.json --condition W00_canonical --output docs/experiments/temporal_guidance_evidence_fix_20261013/first_completed_task.json`。该入口只读已有数据，不调用 API、SSH 或 FLOWR。

修复代码提交 `0c1e3ebf` 已 push，远端已从 GitHub 拉取并通过编译检查。原失败目录及原 Steer 均保留。后台采用每条件隔离失败、其余继续的策略；本对话在首个完整条件确认后停止持续监控。
