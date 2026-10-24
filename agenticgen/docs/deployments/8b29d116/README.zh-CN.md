# V26固定部署与启动

科学提交`8b29d116e421daad8ea134795119ef86a3212abd`，runtime `/opt/MolSteer/runtimes/agenticgen_8b29d116`。官方GitHub API固定对象/限定子树及获取器SHA核验、缓存复用成功，无SSH源码上传。安装前后119项旧协议/R0/历史关键hash保持。科学Python兼容，显式复用既有原生验收，无prepare/native迁移或新跑原生采样。

仅三份skills/文档变化，两端未跑单元测试，无LLM离线回放；tests_receipt=not_run_skills_documents_only、passed=null，不称新测试通过。
真实V25→V26门槛为Top1 R5G2、Top2 R4G3私有/同轮汇总，2教师4上下文，full_review_completed=false。
V25较V24等权gap减少0.000692873，连续退步0、历史最好V17，V25恢复已完整完成，无需再次回滚，目标未达成。
V26在V17恢复基底上新增弱响应重新比较、同波反证更新选择预测两个通用模块，见[有限适配](../../experiments/v25_strategy_review/selective_adaptation.json)。
U1/coarse0.2/q0.1/max_range_probes0/max_changed_intervals2及科学设置保持；双方沿用prior_selected_schedules_v1，私有96K/汇总64K、精确引用、最多两次私有中性修正和版间空动态历史保持。

worker3466528，启动1792833881.5355833；实际核验1792833890.9096732存活、命令匹配、无新failure。初始bootstrap不代表首轮完成，不重复启动或为文档同步更改科学源。
五份远端JSON逐字节SHA核验见source_manifest，远端`/opt/MolSteer/operations/agenticgen_deploy_8b29d116`。
V25质量/表现/成本见[有限评估](../../experiments/v25_strategy_review/review.zh-CN.md)。历史失败成本继续单列，最终报告须合并operations并按provider ID去重。
35分钟巡检保持。
