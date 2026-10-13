# 时序探索科学证据交付修复

## 执行日志追溯

原研究 `temporal_guidance_20261013` 的远端 12 个 cohort 均已 complete，驱动日志显示结构生成、传输与报告后的清理正常。远端当前没有原始测试轨迹；原始 Steer 目录保留。Agent 的真实 API 请求在本地运行，科学证据拒绝并非远端 FLOWR 报错。

原 10 个条件只有 chain_00 完成了角色 JSON 输出，均没有 `get_curve` / `get_stage_profile` 观察。W11 的 Analyst 引用 `continuous:2` 等摘要，并将 `temporal_profiles_available=false` 解读成缺少逐节点数据；Designer 随后提出时序修改。`run_roles` 在两个角色结束后才检查曲线，抛出 `Temporal edit needs an actual Analyst hypothesis and observed temporal evidence`，没有将缺失反馈给 Agent。

原来使用摘要或任意曲线的宽泛 `curve_support` 判断也不足以证明 Designer 采用的假设读到了对应特征的曲线。

## 修复

1. 通用 ReAct 引擎新增可选、版本绑定的 scientific finish validator。JSON 校验与科学证据校验分开；科学 finish 被拒绝时保存反馈并允许下一轮工具读取，仍受统一 action/context 预算限制。旧工作流不启用该参数时保持原行为。
2. Analyst 必须读取实际曲线；每个时序假设引用实际观察的曲线或 profile。摘要行、metadata 和曲线审计行不能代替曲线。证据不足时允许无假设，不强迫提出改动。
3. Designer 的 edit 必须关联真实 Analyst 假设及该假设对应的特征曲线。逐层处理曲线字段页身份，拒绝“读过另一条曲线即可过关”。retain/defer 保持可用。
4. 独立标明 `curves_available` 与 `stage_profiles_available`。新增通用交付协议 Skill 作为所有条件的共同接口约束，不规定优先特征、目标分子、时间窗口或奖励方向。
5. LLM 只接收注册选项的参数层级、描述和数值签名，完整积分表留在确定性编译器，减少重复输入。温度、奖励公式、剂量、种子与积分策略均未改变。
6. 在新输出目录启动完整 10 条件 × 6 角色链队列，保留原失败记录。读取原收集审计的固定 SHA、初始随机状态和执行审计，再复用 12 个数值 cohort；不复用旧角色回复。原 M2、无引导与 Steer 也复用。

## 验证与运行

测试覆盖：缺失曲线的反馈恢复、真实 HTTP 封装的缓存重放、有限预算耗尽无虚假成功、摘要不能替代曲线、无阶段 profiles 时曲线可用、不相关曲线拒绝、字段页来源追溯、允许保留基准，以及跨版本复用时拒绝篡改指标文件。

相关测试共 67 项通过；原 12 个数值 cohort 的审计 SHA、全部尝试质量字段与相同初始随机输入已通过本地启动检查，未导入任何旧角色失败或回复。

运行配置：`configs/experiments/temporal_guidance_evidence_fix_20261013/runner.json`。
使用真实 DeepSeek Flash；原数值候选及冻结对照不重复生成。条件按原顺序运行，出错条件隔离并保留 traceback，其余继续。各条件完成六条链后按众数程序选择，不向 Agent 暴露本轮生成质量结果。

首个完整条件的实测确认将在 `first_completed_task.json` 与 `first_completed_task.zh-CN.md` 记录；在确认前本文件不宣称所有角色链已运行成功。曲线读取和引用只证明科学证据交付，不等于假设科学正确或生成质量提升。
