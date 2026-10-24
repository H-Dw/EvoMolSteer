# V28部署：一次Git拉取被代理拒绝

2026-10-24按用户要求，正常部署已改为一次有界Git fetch（pull拉取阶段）和固定commit检出。传输提交088456aac447e17b28fc4823c4a81cb66daaaa3b已推送；科学目标仍d7d35b132a04c5b58964aad3cfcc552a792a9909。

实际收据时间1792843413.149713：一次fetch，REST API请求0；原代理返回HTTP 503 after CONNECT，Git退出128。流程在传输失败处停止，没有回退或再次拉取。V28尚未安装、批准或启动，当前无活动科学实验。传输相关远端测试尚未执行；本地未运行自动化测试，不宣称验证通过。

完整本地控制收据原样保存为git_deployment_failure.json；远端git_fetch_receipt.json逐字节SHA核验见source_manifest.json。原403失败、一次额度诊断和原等待状态分别保留deployment_failure.json、quota_and_inspection.json、api_attempt_summary.json，均仅为历史；旧API reset+60及API重试安排已失效。

下一步等待实际Git网络恢复证据。35分钟自动任务保留，但不盲重试拉取、不查询API额度、不轮询已完成V27，不换代理/出口/凭据。结果不确定先inspect-only，已有安装/绑定/launch不可重复启动。科学目标、V27有限评估和连续退步1保持。

当前只关注LLM协助gradient guidance，V30完整收尾、报告交付并commit/push保存后删除定时任务。详见[一次拉取策略](../../operations/git-single-fetch.zh-CN.md)。
