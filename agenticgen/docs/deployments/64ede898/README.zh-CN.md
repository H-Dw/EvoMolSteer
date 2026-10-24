# V16固定部署与启动

固定提交`64ede898387316fcbd2ee6022016e4a66234dc52`，runtime `/opt/MolSteer/runtimes/agenticgen_64ede898`。
原GitHub官方API逐commit/tree/blob及获取器SHA校验完成，固定安装成功；没有SSH源码上传。
安装前后76项旧协议/R0/历史结果关键hash不变，科学Python指纹兼容，复用既有原生验收。

此前403限流等待已结束。本次先inspect-only确认没有后续部署收据，再经正常入口核对目标不存在、前版进程退出，
对同一固定提交作一次有界恢复成功：复用Git缓存、取得21个缺失blob及7次限定树请求。
原transport_rate_limit.json、quota_diagnosis.json、resume_plan.json作为历史证据保留，
已由[恢复收据](recovery.json)结束等待，不可再据旧resume_plan重复部署。
本次成功不代表未认证API额度限制永久解除；日常健康巡检不访问GitHub API。

V15自身最优轮评估真实通过：Top1 R4G4、Top2 R5G3，2教师4上下文，full_review_completed=false。
本次仅三份通用skills及文档变化，两端没有自动化测试；tests_receipt为not_run_skills_documents_only，不能写成测试通过。
V15相对V14等权gap减少0.000018497，退步计数归零，历史最好仍V13；本次无需回滚。
V16改写相对缓存最佳的预算价值，以及预测反证与实际排序的区分。未测内侧仍为未知，
保留背景依赖和有效释放；不是教师数组注入、冻结奖励改变或提前启动补充方向。
能力假设、反证标准及skill哈希见[本版适配](../../experiments/v15_strategy_review/selective_adaptation.json)。

U1、coarse0.2、q0.1、max_range_probes0、max_changed_intervals2、模型/奖励/采样/RNG/位移/预算不变。
双方96K私有/bootstrap、64K汇总、精确短引用/私有最多两次中性修正，从空动态历史开始。
worker 2672754，启动1792787457.6023173；真实初始核验见startup_health.json，不代表首轮完成。
不重复安装/批准/启动，不为后续文档同步切换科学源。35分钟只对活动实验正常巡检。

五份远端JSON逐字节SHA核验见source_manifest.json。远端收据`/opt/MolSteer/operations/agenticgen_deploy_64ede898`。
V15结果、有限评估和成本见[评估](../../experiments/v15_strategy_review/review.zh-CN.md)。
