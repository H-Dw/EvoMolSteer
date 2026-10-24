# V28部署：一次Git拉取被代理拒绝

2026-10-24按用户要求，正常部署已改为一次有界Git fetch（pull拉取阶段）和固定commit检出。传输提交088456aac447e17b28fc4823c4a81cb66daaaa3b已推送；科学目标仍d7d35b132a04c5b58964aad3cfcc552a792a9909。

实际收据时间1792843413.149713：一次fetch，REST API请求0；原代理返回HTTP 503 after CONNECT，Git退出128。流程在传输失败处停止，没有回退或再次拉取。V28尚未安装、批准或启动，当前无活动科学实验。传输相关远端测试尚未执行；本地未运行自动化测试，不宣称验证通过。

完整本地控制收据原样保存为git_deployment_failure.json；远端git_fetch_receipt.json逐字节SHA核验见source_manifest.json。原403失败、一次额度诊断和原等待状态分别保留deployment_failure.json、quota_and_inspection.json、api_attempt_summary.json，均仅为历史；旧API reset+60及API重试安排已失效。

下一步等待实际Git网络恢复证据。35分钟自动任务保留，但不盲重试拉取、不查询API额度、不轮询已完成V27，不换代理/出口/凭据。结果不确定先inspect-only，已有安装/绑定/launch不可重复启动。科学目标、V27有限评估和连续退步1保持。

当前只关注LLM协助gradient guidance，V30完整收尾、报告交付并commit/push保存后删除定时任务。详见[一次拉取策略](../../operations/git-single-fetch.zh-CN.md)。

2026-10-24用户要求连接诊断：1792844356.4990513实际只进行一次github.com HTTPS HEAD探测（无REST API、无fetch）。代理CONNECT已返回200，但随后TLS握手超时，curl退出28、目标HTTP状态000，连接仍未恢复；tls_verify=1不是已经证明证书无效。完整收据connectivity_diagnostic.json。

只读Git元数据：Git2.34.1，origin匹配固定仓库，http.proxy/目标URL proxy/remote.origin.proxy均未单独设置，沿用小写http_proxy/https_proxy；不记录代理值。remote.origin.promisor=true、partialclonefilter=blob:none为现有部分克隆设置，不足以解释本次TLS超时。因HTTPS探测未完成，未发起ls-remote、fetch，也未进入条件性的对象/锁/工作树检查；不能称仓库完整健康或已定位仓库故障。诊断收据not_run中的“tunnel failed”为脚本泛化跳过原因；精确阶段以connect=200及SSL connection timeout为准，隧道本身已建立。
