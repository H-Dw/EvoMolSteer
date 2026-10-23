# V7 32会话复盘核验

本次核验通过，但仅支持32/52正式会话的实质覆盖，不构成V8批准。

- FINALIZED SHA256：`03bf0922b06d679d32031e258477bb984631d929f705fce9a1dc903ca40d4759`。
- manifest SHA256：`d62f299da5ab0399572b35827a00281b29be9563c1aeaddbc59515d136c5d15e`。
- report SHA256：`ca64222f65ea1c4714b91554c669f05f78fe90b9730632c077dac71ad7052bc1`。
- 两教师R1–R5全部完成/质量/汇总/保留/清理标记齐全，1000正式记录，worker1589286退出，无启动后新failure。旧failure属于已处理尝试。
- 32条意见逐一绑定manifest evidence_sha256、80份实际请求、88个去重消息及工具/响应来源。原26条意见与来源绑定逐字段比较保持。全文已读的analyst/synthesizer冻结skills与实际system消息一致。
- 初次分析/汇总的完整接口schema已读，38个格式前缀逐字段差异核验；这32个会话没有额外修复诊断。64项消息引用验证包括初始工具复制、前轮记忆和六个汇总会话共享证据完整解引用/精度投影/提案字段。新增R4–R5初始上下文额外字段也逐项比较；executed_schedules与已经完整读过的execution_history工具逐结构一致，才跳过相同副本。
- 本地review门槛回归6项通过：`python -X utf8 -m pytest agenticgen/tests/unit/test_group_review_gate.py -q`。没有修改门槛或科学代码；测试不证明亲和力收益。
- 全证据包1429文件，其中1428来源文件逐hash核验。下载不等于剩余20正式会话及15作废上下文的实质阅读。
- 62正式+19作废provider ID唯一收据成本核对，无重复记账、无已知usage缺失。GPU只计已知优化波次，不把R0/验收成本假设为零。

阅读器曾输出过长的重复初始工具schema及报告，随后用逐字段差异和限定轮次指标重读完整新增字段；本次额外字段中的executed_schedules长副本也在与已读工具逐结构相等后排除，再完整读取余下变化，没有将截断输出作为已读证据。R4/R5已选方案的条件、功能、Best、分位数和质量均从原report核对，并分开当前锚与历史匹配参考计算差值。新增十会话的意见逐条撰写，原16条及其全部来源绑定与HEAD逐字段比较不变。R4/R5新增格式前缀和额外输入字段已重读，五轮汇总均完整解引用并校验精度投影及提案映射；没有把被截断的总览输出视为已读证据。所有实质意见先明确撰写在 `.cache/review_agenticgen_v7_notes.json`，打包器只验证和整理，未自动生成判断。

检查点详情见[checkpoint_validation.json](checkpoint_validation.json)、[prefix_validation.json](prefix_validation.json)、[prompt_copy_validation.json](prompt_copy_validation.json)。原始完成核验中的reviewed_sessions=0是导出时快照，后续实质进度以[progress.json](progress.json)为准。保留Top1旧source/32K与Top2新source/64K差异。未创建V8批准、未改skills、未启动V8。

本次26会话审阅已提交推送4cc0f795。向既有远端会话发送进度文件时，工具审批策略拒绝操作，原因是需要批准而当前AskForApproval=Never；没有绕过拒绝，也没有确认远端元数据更新。远端master_review_progress_v7.json最后已知仍为16会话，本地和GitHub的26条意见可正常用于继续审阅。该同步限制不是科学worker故障，不触发恢复、重跑或新批准。

新增Top2索引26..31的实际输入额外字段、全部不同消息、工具、提案、汇总解引用和R2实测已经阅读，六条意见已明确撰写。原26条及其绑定与提交7bbdf595逐字段比较不变，新增来源hash核验通过；38项前缀差异与64项消息引用核验通过。当前本地覆盖32，未再次尝试被拒绝的远端上传，远端最后已知仍为16。旧段落中的26条已提交与此前读取器处理过程保留为历史记录。
