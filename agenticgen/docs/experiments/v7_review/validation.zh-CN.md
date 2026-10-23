# V7 首批复盘核验

本次核验通过，但仅支持6/52正式会话的实质覆盖，不构成V8批准。

- FINALIZED SHA256：`03bf0922b06d679d32031e258477bb984631d929f705fce9a1dc903ca40d4759`。
- manifest SHA256：`d62f299da5ab0399572b35827a00281b29be9563c1aeaddbc59515d136c5d15e`。
- report SHA256：`ca64222f65ea1c4714b91554c669f05f78fe90b9730632c077dac71ad7052bc1`。
- 两教师R1–R5全部完成/质量/汇总/保留/清理标记齐全，1000正式记录，worker1589286退出，无启动后新failure。旧failure属于已处理尝试。
- 6条意见逐一绑定manifest evidence_sha256、18份实际请求、23个去重消息及工具/响应来源。全文已读的analyst/synthesizer冻结skills与实际system消息一致。
- 初次分析/汇总的完整接口schema已读，后续9个格式前缀逐字段差异核验；这6个会话没有额外修复诊断。12项消息引用验证包括初始工具复制、前轮记忆和汇总共享证据解引用/精度投影/提案字段。
- 本地review门槛回归6项通过：`python -X utf8 -m pytest agenticgen/tests/unit/test_group_review_gate.py -q`。没有修改门槛或科学代码；测试不证明亲和力收益。
- 全证据包1429文件，其中1428来源文件逐hash核验。下载不等于剩余46正式会话及15作废上下文的实质阅读。
- 62正式+19作废provider ID唯一收据成本核对，无重复记账、无已知usage缺失。GPU只计已知优化波次，不把R0/验收成本假设为零。

阅读器曾输出过长的重复初始工具schema及报告，随后用逐字段差异和限定R1/R2指标重读完整新增字段；没有将截断输出作为已读证据。所有实质意见先明确撰写在 `.cache/review_agenticgen_v7_notes.json`，打包器只验证和整理，未自动生成判断。

检查点详情见[checkpoint_validation.json](checkpoint_validation.json)、[prefix_validation.json](prefix_validation.json)、[prompt_copy_validation.json](prompt_copy_validation.json)。原始完成核验中的reviewed_sessions=0是导出时快照，后续实质进度以[progress.json](progress.json)为准。保留Top1旧source/32K与Top2新source/64K差异。未创建V8批准、未改skills、未启动V8。
