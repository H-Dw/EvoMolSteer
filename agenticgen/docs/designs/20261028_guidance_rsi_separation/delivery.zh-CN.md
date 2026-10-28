# 交付与接入边界

独立新版本位于 agenticgen/src/agenticgen/resources/skill_releases/guidance/2.0.0/。原 1.6.1 标准技能、渲染提示、实验快照和所有历史审计均原样保留。

Guidance Agent 使用 skills/guidance-experiment-optimizer 或对应的 distance-guidance-analysis/proposal/synthesis；前者的 legacy action contract 与后者的整数 updates 接口分开打包，不能混用。内部距离实验候选绑定目录为 runtime/internal_distance；legacy 目录仅为既有旧动作接口的版本化产物，不代表恢复旧实验。

外部 Agent 使用 skills/guidance-postexperiment-rsi/SKILL.md。只在原实验声明的全部 target 及规定工作完成后运行，针对一个主要流程问题创建新版本，不在实验中修改指引。该外部技能本身的未来修订也必须另建版本。

复盘正文使用抽象特征和完整指标名称。具体对象、测量数值、来源链保存在更新日志/证据中，不随角色提示装载。当前任务经验不能进入固定技能；对后续新任务的具体迁移思路须作为未验证先验独立提供，不能伪装成本教师实测。

实际代码检查：geometry_runtime.py 的 skill_assets（约第 18 行）硬编码旧 resources/skills 路径，accept_review（约第 233 行）仍要求一至三个实际 skill 变更。它们与新用户决策不兼容。本交付只构建技能版本和防覆盖渲染工具，不修改现有实验流程。不能直接用旧 accept-review 强迫更新 skills，也不能宣称新版本已经接入或在服务器运行。下一次接入须显式绑定新版本、全实验冻结哈希，并将实验内结果验收与全 target 结束后的 RSI 分开；保留旧协议复现实例。

本次没有启动 LLM 或分子实验，没有部署，没有修改自动化，没有对实际分子质量或行为改善做新声明。
