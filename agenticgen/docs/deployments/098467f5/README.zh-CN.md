# V10 Top1已按96K重新启动

固定提交`098467f5bc0e4f4860eecd12a68c42773a82b5d8`，运行目录
`/opt/MolSteer/runtimes/agenticgen_098467f5`。2026-10-23 17:35北京时间启动
worker2045658；17:36:19启动核验确认进程存活、命令身份一致、无新failure。
当时仍在bootstrap，pipeline尚未创建，不能称首轮完成。Top2原五轮全部标记完整。

- 私有/bootstrap上限98304；汇总65536，thinking/high、输入512KiB、timeout600、HTTP4保持。
- 本地63项相关测试通过，原生随机流一项留远端；远端64项全部通过，无跳过。
- 两教师原生验收通过：同调度坐标/评分最大差0，类别/RNG一致，组间不同调度及共同随机流、质量和梯度验证通过。最大单步指导位移0.010813/0.003432Å，低于0.025Å。
- 31729个保护文件哈希不变，包括原协议、R0、V6–V9、完成Top2和历史失败来源。
- 先归档后删除失败Top1四轮400候选对应2731文件，完整重跑Top1五轮；Top2保留原d1163b7e及64K来源。不能声称两教师分析预算一致。
- 原V9→V10有限复盘重新绑定新提交与验收，2教师4上下文门槛通过，full_review_completed=false；没有V11批准，也没有补做全量审阅。

活动协议`protocols/analysis_output_repair_v10_098467f5`，事务
`operations/failed_teacher_rerun_v10_098467f5/journal.json`已完成归档删除，随后批准并启动。
不重复stage/finish/delete/launch。新Top1动态历史从R1重建，不导入失败搜索答案；
V6–V9和完成Top2不重跑。科学奖励/模型/采样/RNG/起点/幅度/位移上限和skills保持。

本次失败尝试26份provider ID去重API收据，1202309输入/332311输出tokens，
无已知缺失；4个失败GPU波次215.489767秒。不是货币账单，也不含R0/验收。
原始prompt/响应/usage、删除哈希清单和完整费用在上述事务目录保留。最终V10成本
须追加这次作废成本，并与正式versions收据按provider ID去重，不并入失败候选分数。

两次Git HTTPS安装前失败，已确认代理CONNECT HTTP503；没有工具审批拒绝。
之后从GitHub官方API取得固定commit/tree/blob并逐对象校验Git SHA，补齐25个blob，
正常固定安装成功。API/代理环境完整继承，无SSH源码上传。此事实不保证后续所有工具调用免审批。

来源见[source_manifest.json](source_manifest.json)、[部署概要](deployment_summary.json)和
[启动核验](startup_health.json)。17个来源的原始字节哈希已验证；大体积保护/删除/费用明细
保留远端及本地.cache证据包，清单明确哪些文件有本地逐字节副本。
后续本地控制客户端/文档提交不更换098467f5科学运行来源。恢复35分钟正常巡检。
