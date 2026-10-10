# 重复调用队列的引用契约修复

2026-10-10 本地审计。主屏幕的十个条件和 M3 已保存并完成本地评估；重复调用及六批次独立确认没有完成。此前启动的本地收集、确认、报告进程均已退出，不应将它们描述为仍在运行。

本地 `test/luna_dependency_live/collect.err` 在 2026-10-09 22:21 记录 `Luna_I0_full__replicate_1` 生成失败。该任务规范指向 `structure.json.gz`，而对应 Designer 编译结果的 family 是 pocket，`compiled.json` 及奖励程序内的 reference SHA256 指向 `pocket.json.gz`。两份文件各自的哈希都正确，但互相不匹配；这足以触发现有生成入口的 `Reference hash mismatch`。远端完整错误日志尚未重新取得，因此这是已确认的配置缺陷，不能用来推断其他远端故障不存在。

修复增加提交前校验：任务 reference 哈希必须同时等于文件实际哈希和奖励程序嵌入的哈希。保留原失败规范，新建 `Luna_I0_full__replicate_1_retryref_v1.json`，由 `prepare_dependency_retry.py` 读取已保存的编译元数据构建；数值奖励和生成种子没有变化。重试使用独立名称，禁止覆盖原失败状态。

本次两次远端连接均在 SSH 握手 banner 阶段失败，本次连接未重新读取远端状态，尚未部署或提交重试。仓库另已保存今日 20:23 的 `docs/experiments/outcome_label_reassessment_20261010/remote_status.json` 快照：确认控制队列 failed、completed 为空、没有确认输出，与本地中断证据一致。不得声称重复/确认实验继续运行。恢复连接后，应先核对队列、GPU 与现有状态，再拉取已推送的修复提交，以新的任务名称执行，并将新名称明确映射回原重复条件。独立确认协议仍只确认已预注册的零 Steer 信息候选，不会自动为 I3、I4 或 M2 提供独立验证。
