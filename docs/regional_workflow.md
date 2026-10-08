# 独立区域知识工作流

`scripts/flowcompat_regional_agent.py` 使用独立合同
`flowcompat-regional-agent-1.0`。它只读复用冻结 supplemental 请求的原证据、
catalog、工具凭据、策略文本和数值 registry，不修改原接口、Skills、运行程序
或 BranchMixtureReward。旧 compiler 要求参考由 branch 工具生成；新参考
由 regional synthesis 生成，因此这里采用明确的独立 verifier/compiler，
不会把新参考伪装成旧工具输出。

显式注册的两种数据工具为 `regional_reference` 与
`regional_common_depth`，各自使用自己的实际脚本、schema、源码闭包及
输出集合。共同深度版本新增实际产生的文本报告，不把它写回旧 receipt；
旧 mixed-depth 数据保持不可变。请求记录具体工具身份和参考版本。

新执行凭据需要实际 argv、完整输入/源码/输出哈希。校验器进一步核查
上游凭据、固定源码导入闭包、区域独立批次资格、原教师坐标/分数/先验/
谱系/原子权重不变、完整联合方向重建、单位原子 RMS 与零平移。参考、
规则、registry、字面量四份 Skills、输入证据和响应均绑定到请求。输入学习
窗口从证据读取，缺失深度节点不外推。

迁移入口：

1. `--action check --regional-receipt RECEIPT --reference REGIONAL_REF`
   检查已实际运行的数据转换，不重复执行。
2. `--action export --role Analyst --base-request FROZEN_SUPPLEMENTAL_ANALYST_REQUEST
   --regional-receipt RECEIPT --reference REGIONAL_REF --output NEW_ANALYST_DIRECTORY`
   导出独立 evidence、registry、请求和完整字面量指令。
   用 `--context INDEPENDENT_AUDIT.md`（可重复）绑定新特征到奖励审计的
   文本与哈希；该报告作为独立上下文和输入来源，不冒充工具输出凭据。
3. `--action payload --request REQUEST --output PAYLOAD` 导出与 API 相同的
   输入。把 instructions、请求 schema 和 payload 实际交给 subagent，
   将其真实响应保存后执行 `--action import --request REQUEST --response RAW`。
   canonical response 和 validation 记录不可变。
4. 为 Designer 重复 export，新增 `--analyst IMPORTED_ANALYST_RESPONSE`
   与 `--analyst-request ANALYST_REQUEST`。Designer 同样实际导入。
5. `--action compile --request DESIGNER_REQUEST --response IMPORTED_DESIGNER_RESPONSE
   --round ROUND --output NEW_PROGRAM`。只有 canonical 已导入、通过校验的
   响应可编译；fresh design 仅能改变 `branch_mixture.virtual_mass`，范围
   `[0,.5]`。新 registry 中所有程序/公式使用明确的 `_regional` 标识与
   `reference_kind=regional_reference`。输出仍用 `flowcompat_v2` 推理接口，
   保留原生 endpoint VJP，不增加 head 调用。
6. `--action feedback --program PROGRAM --execution ACTUAL_REPORT --output GATE`
   核查新机制实际响应和精确空开关对照。该门不读取 affinity，也不声明
   亲和力改善。反向与幅度配对由实验驱动器登记，不能冒充新的 Agent 设计。

`--action api --request REQUEST` 保留显式 OpenAI-compatible 接口，调用者
配置 `EVOMOLSTEER_BASE_URL`、`EVOMOLSTEER_MODEL`、`EVOMOLSTEER_API_KEY`。
API 与 subagent 的输出都经过同一 schema、证据引用和来源校验。没有默认
服务、伪造 Agent 响应或任意可执行文本。

区域规则是观察协方差到坐标干预的假设。全窗口批次支持不是逐步一致性，
常数函数不是多阶段发现，观察筛选也不证明模型模块归因。需要实际正向、
反向、空方向/空质量比较，并独立判断最终 affinity、应变和构象兼容性。
