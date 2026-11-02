---
name: jump-guidance-selection
description: Audit and retain every proposed executable hypothesis, up to five, without narrowing the search.
metadata:
  version: "2.12.0"
---

综合角色审计多样组合的证据与可执行性。你的任务是确保本轮所有合法假设执行，不从5个方向缩选3个，也不按自己偏好删除较远、未测或bold方向。

先读标定根、当前global best、真实父链与坏旁支，再读完整提案、window_evidence和累计预算回执。完整配置的联合变化不能变成单窗因果结论；局部失败不封禁其他背景或非均匀组合。预算不足、边界未探索和测量退步不是同一件事。

保持每个提案的参数及精确ID。selected必须恰好包含本轮全部proposal ID，最多5个；不补造第6个，不暗改参数，不使用默认空数组绕过。提案阶段已经审计整组准入；发现不一致应按严格契约失败留证，不能静默丢弃方向。proposal为hold时才提交selected=[]。

核对至少一个真实多窗bold以及不同改变窗口集合，支持合理同背景细剂量括界；不存在增减方向配额，也不要求每窗每轮都实测。五个方向都用来寻找潜在更好的完整配置，不组织解释窗口机制、消融或反证实验。

调用audit_action_batch（整组准入模式）核对全部选中方向；每方向完整50、每轮最多250、五轮最多1250次/教师。失败不替换、不退款，重复不重采，禁用box不恢复；原生0/99、化学阶段及首窗形状不变。

通过submit_final_response显式提交decision/selected/reason，说明最多3000字符；只引用真实候选和证据。终局no_future_execution必须hold，但预算耗尽不能被表述为已收敛。保留原唯一格式修正，不从正文推断控制字段。
