# V6修复、清理与五轮重跑

已在11762服务器部署固定提交 `0513b30ef804c3b772c74539de73b32429126bcd`，运行目录 `/opt/MolSteer/runtimes/agenticgen_0513b30e`。代码先本地测试、提交并推送，再由远端从GitHub取得。Git代理两次503后改用GitHub官方API取得固定commit/tree/blob，每个Git对象均核对原始对象哈希；没有使用未提交代码替代部署。

- 本地78项通过、12项因本地依赖缺失跳过；远端90项全部通过。
- 两教师原生恢复、共同随机流、不同组调度、真实质量评价及梯度验收全部通过。原科学采样、评分、模型、R0和版内skills未修改。
- 按最新人工授权，旧失败V6的200条候选结果已从正式版本目录删除，共移除1396个文件；失败prompt/响应/usage另行归档。383个受保护文件的哈希再次一致，包含共享R0和原协议文件。两教师重新从R1执行完整五轮，正式优化预算仍为1000条。
- 新进程PID1346453，启动时间戳1792694593.3598337。实际命令、API与代理继承通过核验。`summary.json`为1792694825.1786253的快照：两教师bootstrap均合法，Top1 R1生成和质量完成、正在分析；Top2 R1正在生成，没有新error。这不是完成五轮或性能改善的证明。
- 输入硬上限512KiB；输出仍为私有32K/汇总64K。当前组动作参照、固定R0物理重启和历史因果比较明确分开。实际失败证据的共享块表示为187903字节，原297012字节，34块，逐项还原验证通过。
- 每40分钟巡检保持ACTIVE。V5分析继续终止，V6及以后完成版本仍需全量52会话复盘。运行保持固定0513b30e，后续文档提交不触发重启。

远端操作目录为 `operations/failed_version_rerun_v6_0513b30e/`，`journal.json`记录新验收后的删除事务；`failure_audit/`保存原失败会话，`deletion_manifest.json`保存原文件哈希，`aborted_resource_usage.json`保存被排除尝试的资源支出。最后一项成本须与新正式运行费用合并、按provider请求ID去重；旧候选分数不合并。原生验收的 `native_validation_receipt.json` 创建于删除前，因此其中 `candidate_outputs_deleted=false` 是当时状态；后续 `summary.json.cleanup` 才是删除结果。

收据： [运行与清理快照](summary.json)、[原生验收](native_validation_receipt.json)、[远端测试](tests.txt)、[无损证据验证](payload_validation.json)、[失败成本](aborted_resource_usage.json)、[GitHub对象来源](github_commit_tree_receipt.json)。
