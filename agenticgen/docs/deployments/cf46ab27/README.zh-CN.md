# cf46ab27：已安装，启动前置兼容性错误，已由 cea3611c 修复

最新真实收据：限定部署子树的官方 GitHub API 获取成功，获取器固定2304badc且9项远端检查通过；cf46ab27安装成功，53项旧协议/R0/完成结果关键哈希一致。三份远端来源逐字节校验见installed_source_manifest.json。

随后普通启动入口读取已被V10修复协议归档的strategy_review_operation字段，报KeyError；未执行版本工程检查、批准或启动，没有V11结果。详见[startup_protocol_fix](startup_protocol_fix.zh-CN.md)。不要再部署cf46ab27。

新固定目标为[cea3611c](../cea3611c/README.zh-CN.md)，仅修复普通启动门槛的协议分派；科学Python和V11策略不变。其首次获取遇GitHub API 403限流，尚未安装。以下是历史超时记录，不作为当前待执行命令。

---

# V11准备完成，源码获取超时

预定固定提交`cf46ab27bf953c05ecd2e604539596b052348d2a`已本地提交并推送原GitHub。
V10已FINALIZED，有限最优轮评估完成，两项skills改进及正常API部署入口均在该提交。
没有本地自动化测试。

2026-10-23本次部署请求已被工具放行，正常SSH认证及固定用途入口执行成功。
官方GitHub API固定对象获取阶段触及240秒总时限，退出码124；本次未获得
github_object_receipt、installation、tests、approval或launch收据。
入口使用set -e，后续安装/测试/批准/启动没有执行。V11未启动，V10未重启。
这不是工具审批拒绝，也没有证据将本次具体原因写成HTTP503；获取阶段超时的
更底层原因尚未确认。见[控制收据](transport_timeout.json)。

按已有计划，本次停止部署、不循环重试、不因网络失败制造新的科学commit；
保留远端已写入的Git对象缓存。下一次正常续接先使用同一入口的`--inspect-only`
核对该commit是否有后续/部分安装收据，再按既有有界规则继续同一固定提交。
不调用V10修复/删除事务，不恢复已完成版本；若发现已有V11绑定或launch，停止
重复启动并按实际状态处理。此次文档提交只记录超时，不改变待部署科学提交。

待部署命令：
`python -X utf8 agenticgen/scripts/deploy_reviewed_version.py --version v11 --commit cf46ab27bf953c05ecd2e604539596b052348d2a --output .cache/v11_deploy_cf46ab27_retry.json`

只有实际远端相关工程检查、V10有限review-check和launch成功后才能称V11启动。
V11仍计划U=1，V10两教师各R5G1四上下文评估不能被误写为52会话覆盖。
