# V28部署：ghfast后备克隆超时

2026-10-24最新用户授权ghfast镜像后备，实施提交72e8a6d21161ae2b8805c409a8dc3e0204a7ef58。实际一次120秒有界bare clone超时退出124，收据1792846560.466603。网络fetch0、clone1、本地导入0、REST API0，无重试。日志只有开始克隆，不足以判断连接/传输具体阻塞阶段，也不证明仓库损坏。

V28尚未安装/批准/启动，当前无活动科学实验。未抵达远端相关验证，本地未跑自动化测试，不称验证通过。科学目标d7d35b132a04c5b58964aad3cfcc552a792a9909保持；等待处理网络/传输障碍。缓存/opt/MolSteer/source_cache/ghfast_72e8a6d2.git超时后是否残留尚未检查，不自动删除或覆盖，后续须先inspect。完整控制收据ghfast_deployment_failure.json，远端快照ghfast_git_fetch_receipt.json逐字节SHA核验；原Git/API失败收据均保留。

后备策略已生效，GitHub异常时使用固定ghfast URL，不再要求仅等直连恢复；失败不循环重试、不换其他镜像、不用REST API。正常科学门槛、35分钟频率和V30结束删除任务保持。下文为历史。

2026-10-24按用户要求，正常部署已改为一次有界Git fetch（pull拉取阶段）和固定commit检出。传输提交088456aac447e17b28fc4823c4a81cb66daaaa3b已推送；科学目标仍d7d35b132a04c5b58964aad3cfcc552a792a9909。

实际收据时间1792843413.149713：一次fetch，REST API请求0；原代理返回HTTP 503 after CONNECT，Git退出128。流程在传输失败处停止，没有回退或再次拉取。V28尚未安装、批准或启动，当前无活动科学实验。传输相关远端测试尚未执行；本地未运行自动化测试，不宣称验证通过。

完整本地控制收据原样保存为git_deployment_failure.json；远端git_fetch_receipt.json逐字节SHA核验见source_manifest.json。原403失败、一次额度诊断和原等待状态分别保留deployment_failure.json、quota_and_inspection.json、api_attempt_summary.json，均仅为历史；旧API reset+60及API重试安排已失效。

下一步等待实际Git网络恢复证据。35分钟自动任务保留，但不盲重试拉取、不查询API额度、不轮询已完成V27，不换代理/出口/凭据。结果不确定先inspect-only，已有安装/绑定/launch不可重复启动。科学目标、V27有限评估和连续退步1保持。

当前只关注LLM协助gradient guidance，V30完整收尾、报告交付并commit/push保存后删除定时任务。详见[一次拉取策略](../../operations/git-single-fetch.zh-CN.md)。

2026-10-24用户要求连接诊断：1792844356.4990513实际只进行一次github.com HTTPS HEAD探测（无REST API、无fetch）。代理CONNECT已返回200，但随后TLS握手超时，curl退出28、目标HTTP状态000，连接仍未恢复；tls_verify=1不是已经证明证书无效。完整收据connectivity_diagnostic.json。

只读Git元数据：Git2.34.1，origin匹配固定仓库，http.proxy/目标URL proxy/remote.origin.proxy均未单独设置，沿用小写http_proxy/https_proxy；不记录代理值。remote.origin.promisor=true、partialclonefilter=blob:none为现有部分克隆设置，不足以解释本次TLS超时。因HTTPS探测未完成，未发起ls-remote、fetch，也未进入条件性的对象/锁/工作树检查；不能称仓库完整健康或已定位仓库故障。诊断收据not_run中的“tunnel failed”为脚本泛化跳过原因；精确阶段以connect=200及SSL connection timeout为准，隧道本身已建立。
