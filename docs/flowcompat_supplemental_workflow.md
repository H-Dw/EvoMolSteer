# 条件分支变异的独立接口

新版入口为 `scripts/flowcompat_supplemental_agent.py`，合同版本为
`flowcompat-supplemental-agent-2.0`。它不会修改原 `flowcompat_agent.py`、原
Skill 或旧请求；正在进行的旧实验继续使用原合同。新模块按校验值固定这三个
原文件，并保留现有两张真实工具 receipt。

可选的已完成多深度工具使用 `--extra-evidence` 与 `--extra-receipt` 成对
接入。仅接受固定的 `multi_depth_mutation` 实时 selfreceipt，核查实际 argv、
已加载源码与 eager 导入闭包、全部消费输入和输出/manifest；不会修改 receipt
或重跑数据。该 receipt 单独保存于 `extra_receipts`，旧三张 receipt 保持
原结构。响应的 `extra_tool_receipt_sha256` 绑定新增 receipt，没有额外工具
时为空数组；`source_evidence_sha256` 始终覆盖所有证据输入。

可选数据会补齐所引区域的 XYZ、同深度原始/调整后绝对矩、相关性及选中的
全窗趋势和方向覆盖。没有把批次对当成独立统计样本，也没有把全窗平均
方向的一致性等同于逐节点向量一致性。

迁移顺序：

1. 为 `branch_mutation` 创建新的工具计划，写入 `tool_id`、字面量
   `arguments`、`input_files` 和 `output_files`。参数包括数据集、campaign、
   冻结端点参考、新摘要目录；完整增强参考是可选独立输出。必须声明
   `evidence.json`、`manifest.json` 和所请求的增强参考。
2. 运行 `--action tool --plan PLAN.json --output RECEIPT.json`。第三工具实际
   执行后，receipt 自动记录全部消费的参考、配置、frame、轨迹，以及本地
   Python 源码导入闭包和全部输出。已有目录不能补造 receipt。
3. 使用 `--action export --role Analyst --evidence OLD_EVIDENCE.json
   --branch-evidence NEW_BRANCH_EVIDENCE.json --registry REGISTRY.json`
   并分别提供三次 `--receipt` 和新的 `--output` 目录。
4. 将保存的字面量指令、请求和 `payload(request)` 输入 subagent；用
   `--action import --request REQUEST.json --response RESPONSE.json` 校验。
   `--action api --request REQUEST.json` 使用同一 schema、证据和导入校验，
   API 地址、模型和密钥由调用者显式配置，没有默认服务。
5. 为 Designer 重复导出，另提供 `--analyst` 指向已校验的新 Analyst 响应。
   Designer 需要引用新分支证据。使用 `--action compile --request REQUEST
   --response RESPONSE --output PROGRAM --round NUMBER` 生成声明式程序。
6. 实际推理后使用 `--action feedback --program PROGRAM --execution REPORT
   --output GATE`。实施反馈门与亲和力效果判断分开；没有实际新机制响应或
   精确零开关对照不能通过。

数值 registry 延续 `programs` 与 `formulas` 两部分。程序记录原程序和所选
参考的路径、SHA256；公式记录 `reward_view`、`allowed_updates`、`code_files`
及显式 `reference_kind`（`incumbent` 或 `branch_mutation`）。新版的
`endpoint_innovation` 必须选择 `branch_mutation`：参考需带有原工具写入的
分支元数据、相同动态学习窗口，并且必须是第三张 receipt 的真实输出。
数值更新沿用现有有界字段，每次只允许一个轴，不执行 LLM 生成的代码。

独立分支混合公式为 `endpoint_branch_mixture`，仍然要求真实第三工具参考。
它可注册 `branch_mixture.virtual_mass`、`branch_mixture.direction_sign` 和
`branch_mixture.region_weight_mix`；数值范围由接口 schema 限定。`context_files`
可以绑定 `numerical_strategy_v2.md` 等文本来源，实际内容同时送入两种后端。
混合预算、原教师保留、虚拟教师变异尺度和资格的具体定义由该冻结策略及数值
模块决定，接口不擅自修补 LLM 方案。新增字段不会进入旧合同。

该新公式的实施报告使用 `branch_virtual_mass_mean`、
`branch_virtual_teacher_shift_rms_A`、`branch_gradient_relative_change` 和
`paired_window_coordinate_rms_A`。独立区域权重轴使用
`branch_region_weight_rms_change`，可在零虚拟质量下生效。活跃机制必须响应；
虚拟场和区域权重均为空时需要精确原基线对照。该标量奖励没有要求原生方向
预处理器改变。正/反方向是配对探索，门通过不代表已经改善亲和力。

LLM 输入会合并旧选择证据与新分支证据；对新证据中的显著特征，还提供其
原始/调整后协方差和相关性对照及对应全窗口函数。这样可以比较绝对效应与
标准化相关性，而不展开逐节点宽表。前两个缺失两步祖先的节点不补为统计零。

本次真实迁移工具产物位于
`docs/experiments/flowcompat30_20261009/mining/agent_branch_v3`，receipt 位于
`agents/supplemental_v2/branch.receipt.json`：20 个本地源码文件、30 个输入、
9 个输出已绑定；新 evidence 和增强参考与先前正式统计逐字节相同。这里只
完成了工具执行与接口验证，没有运行新的生成试验或宣称亲和力改善。
