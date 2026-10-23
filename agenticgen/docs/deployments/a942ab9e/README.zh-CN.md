# V15固定部署与启动

固定提交`a942ab9e2afca41d47415d4c84bffd2f59a300ee`，runtime `/opt/MolSteer/runtimes/agenticgen_a942ab9e`。
原GitHub官方API逐commit/tree/blob及获取器SHA校验完成，固定安装成功；没有SSH源码上传。
安装前后72项旧协议/R0/历史结果关键hash保持，科学Python兼容，复用原生验收。

V14自身最优轮评估真实通过：两教师R5G1私有/同轮汇总，2教师4上下文，full_review_completed=false。
本次只有三份skills和文档变化，两端没有自动化测试；tests_receipt=not_run_skills_documents_only，不是测试通过。
V14整体为回滚后第一次退步，历史最好仍V13，未触发再次回滚。V15按用户经验模块授权改写有界反转分支和背景依赖的方向结论；
只用已有聚合对照支持下一合法动作，不增加全量审阅、诊断或采样。新策略未有性能结果。
基底与新skill hash见[适配记录](../../experiments/v14_strategy_review/selective_adaptation.json)。

U1、coarse0.2、q0.1、max_range_probes0、max_changed_intervals2及模型/奖励/采样/RNG/位移/预算不变。
双方96K私有/bootstrap、64K汇总、短引用/私有最多两次中性修正，从空动态历史开始。
worker 2587449，启动1792782804.6200216；初始核验见startup_health.json，不代表首轮完成。
不重复部署、批准或启动，不为后续文档同步切换科学源。35分钟仅活动实验正常巡检。

五份远端JSON逐字节SHA核验见source_manifest.json。
远端收据`/opt/MolSteer/operations/agenticgen_deploy_a942ab9e`。
V14结果及费用见[有限评估](../../experiments/v14_strategy_review/review.zh-CN.md)。
