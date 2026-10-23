# V14固定部署与启动

固定提交`962a2cacdba94e1390bd3ff1005163894367f02a`，runtime `/opt/MolSteer/runtimes/agenticgen_962a2cac`。
原GitHub官方API逐commit/tree/blob及获取器SHA校验完成，固定安装成功；没有SSH源码上传。
安装前后68项旧协议/R0/历史结果关键hash不变，科学Python指纹兼容，复用原生验收。

V13自身最优轮评估真实通过：Top1 R5G1、Top2 R5G2，2教师4上下文，full_review_completed=false。
本次仅skills/文档变化，两端没有运行自动化测试；tests_receipt是not_run_skills_documents_only，不能写成测试通过。
V13已完成精确回滚并成为当前历史最好，下一版不再次回滚。V14在历史基底上选择性适配用户经验框架，
以可比较响应选择动作，并及时检验有效背景上的维持/释放配合；不是字节一致恢复，也未证明性能提升。
基底、候选及新skill哈希见[适配记录](../../experiments/v13_strategy_review/selective_adaptation.json)。

U1、coarse0.2、q0.1、max_range_probes0、max_changed_intervals2以及模型/奖励/采样/RNG/位移/预算不变。
双方使用96K私有/bootstrap、64K汇总、短引用/私有最多两次中性修正，从空动态历史开始。
worker 2501226，启动1792778682.9171023；实际初始核验见startup_health.json，不代表首轮完成。
不重复安装/批准/启动，不为后续文档同步切换科学源。35分钟只对活动实验正常巡检。

五份远端JSON逐字节SHA核验见source_manifest.json。
远端收据`/opt/MolSteer/operations/agenticgen_deploy_962a2cac`。
V13结果、有限评估和成本见[评估](../../experiments/v13_strategy_review/review.zh-CN.md)。
