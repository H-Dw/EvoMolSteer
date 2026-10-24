# V25固定部署与启动

科学提交`5401fb410856f14d9c19416b1eb0cd6e154d79b3`，runtime `/opt/MolSteer/runtimes/agenticgen_5401fb41`。官方GitHub API固定对象/限定子树及获取器SHA核验、缓存复用成功，无SSH源码上传。安装前后115项旧协议/R0/历史关键hash保持。科学Python兼容，显式复用既有原生验收，无prepare/native迁移或新跑原生采样。

仅三份skills/文档变化，两端未跑单元测试，无LLM离线回放；tests_receipt=not_run_skills_documents_only、passed=null，不称新测试通过。
真实V24→V25门槛为Top1 R5G1与Top2 R5G2私有/同轮汇总，2教师4上下文，full_review_completed=false。
V24较V23等权gap增加0.000350633，连续退步2、历史最好V17，已按原门槛恢复V17兼容策略，目标未达成。
V25精确恢复V17三份skills及完整搜索政策，保留当前已验证资格上下文，本次不叠加新模块，见[有限适配](../../experiments/v24_strategy_review/selective_adaptation.json)。
U1/coarse0.2/q0.1/max_range_probes0/max_changed_intervals2及科学设置保持；双方沿用prior_selected_schedules_v1，私有96K/汇总64K、精确引用、最多两次私有中性修正和版间空动态历史保持。

worker3396400，启动1792829845.5593977；实际核验1792829853.7254612存活、命令匹配、无新failure。初始bootstrap不代表首轮完成，不重复启动或为文档同步更改科学源。
五份远端JSON逐字节SHA核验见source_manifest，远端`/opt/MolSteer/operations/agenticgen_deploy_5401fb41`。
V24质量/表现/成本见[有限评估](../../experiments/v24_strategy_review/review.zh-CN.md)。历史失败成本继续单列，最终报告须合并operations并按provider ID去重。
35分钟巡检保持；V30后PB/inpainting计划继续休眠，原方向完整交付和commit/push保存后再重新读取交接；I20完整交付后才删除自动任务。
