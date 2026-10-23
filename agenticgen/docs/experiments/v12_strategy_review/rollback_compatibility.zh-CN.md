# V13恢复V10策略及当前代码兼容性

V11和V12是连续两个完整退步版本；冻结选择器的历史最好为V10。本次新提交恢复V10 binding所绑定的两份运行时Guidance skills（analyst/synthesizer），逐字节SHA相同，控制/范围恢复为U=1、粗尺度0.2、q=0.1、range_probe配额0、最多改变两窗口。具体来源与hash见rollback_source_inventory.json和rollback_plan.json。

恢复来源是c86706b565c81949192f0147d93298dde55d9355。不传V10教师最优数组、历史动态记忆或分子答案。主Agent的plip-schedule-review继续使用用户要求的有限最优轮/机制框架；它不在运行时SKILLS binding中，不进入Guidance请求，不替代这次V12自身四上下文评估。

保留V12已接受的兼容代码。与V10 binding的科学Python来源相比，只有group_agent.py、deepseek_api.py和restart_failed_teacher.py三项变化：私有/bootstrap由最多一次中性格式修正变为两次，json-interface-v5要求完整JSON对象/字段，以及支持已完成V11教师修复事务的兼容校验。无模糊引用匹配或手工补JSON。双方仍96K私有/64K汇总、精确短引用、输入512KiB、thinking/high、timeout600、HTTP4。

原生模型、采样、随机流、固定R0、奖励、坐标匹配、评分、位移上限和范围执行代码的hash保持。V11接口修复已通过相关检查及两教师原生验收：同调度坐标/评分差0、类别/RNG一致，不同调度/共同随机流及质量/梯度通过；来源在docs/deployments/07fa7423。V13相对V12不改任何包内科学Python，复用已接受原生兼容性，不重复协议迁移或原生采样。

V10保留Top2的64K/旧引用来源，Top1为96K短引用且一次私有修正；V13两教师统一当前96K/短引用/两次修正。因此这是历史最好策略与控制的兼容恢复，不是完全复刻V10的LLM预算或保证性能重现。来源差异须在后续报告披露。

唯一新的可执行改动位于正常启动脚本：从完整历史自动计算强制回滚，在写新版本前验证已提交rollback_plan、V10 binding、两份技能字节hash、范围和本兼容说明，再把rollback字段写入真实review_approval。现有check_control_gate保持原样，仍重新校验这些绑定；无条件豁免、手改pipeline、删除或重跑历史。审批收据记录真实恢复来源。

本地不运行自动化测试。固定远端仅运行test_reviewed_start.py和既有test_control_gates.py，验证错误来源/政策/技能不能通过、无回滚条件不能重置历史、正常边界仍可执行。完成/锁/源hash/有限review-check照常保留。当前文档提交前这些远端检查尚未执行，真实结果另存部署收据。
