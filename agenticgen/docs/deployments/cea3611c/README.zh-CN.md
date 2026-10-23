# V11 已固定部署并启动

固定提交 `cea3611c7c07b47868038a20a71877706e845d97`，运行目录 `/opt/MolSteer/runtimes/agenticgen_cea3611c`。
2026-10-23正常认证后一次GET /rate_limit返回HTTP200，未认证额度60、已用9、剩余51；不是文件上传超限。定点SFTP检查确认V11未创建、前worker已退出，随后同一固定提交的一次有界恢复成功。

官方GitHub固定对象获取、安装、三个直接相关远端测试文件全部通过，无本地自动化测试或无关全套。缓存复用后仅下载20个blob，目录树请求7次。安装前后53项旧协议/R0/历史完成关键hash保持。源包科学Python未改变，复用已有原生兼容性，未执行迁移/删除/重跑V10。

真实V10→V11有限review-check通过：2教师4上下文、full_review_completed=false。V11保持U=1、96K短引用和已写两项通用能力假设。普通启动入口已正确验证当前teacher_restart协议及历史保护来源。

worker **2175969**，启动时间1792762086.6861448；核验1792762099.729195存活、命令身份匹配、无新failure。pipeline尚未创建属于bootstrap阶段，不能称首轮完成。完整API/代理环境继承，值不记录。

本目录五份远端JSON均逐字节验证SHA256，见source_manifest.json。quota_diagnosis.json和startup_health.json是客户端限定读取收据。旧transport_rate_limit/resume_plan为历史失败与等待记录，已被本次成功收据取代，不再据其等待、重装或重启。后续只做35分钟活动实验巡检。

[限流原因和后续策略](github_rate_limit.zh-CN.md)。V10结果、来源差异及失败成本见[有限策略评估](../../experiments/v10_strategy_review/review.zh-CN.md)。
