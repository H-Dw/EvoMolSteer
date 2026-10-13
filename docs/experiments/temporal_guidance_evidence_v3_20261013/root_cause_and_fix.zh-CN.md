# Designer JSON 闭合错误与 C00 曲线引用循环：原因与修复

## 结论与证据边界

这次 Designer 错误不是已观察到的 token 上限截断，也不是本地提取程序剪掉了数组结尾。W10 的两份非法回答与 W11 的两份非法回答，原始 `message.content` 都在 `/result/failure_modes` 数组尚未闭合时出现 `}`，需要的闭合符号是 `]`。四份输出为 451–741 completion tokens，请求上限为 8,192；原接口只有 `finish_reason=stop` 才会进入语法校验，没有相应 incomplete 记录。这支持“正常结束标记下返回了非法 JSON”，不能进一步确定服务端内部解码、网关或其他提供方内部原因。

实际发送内容已经包含 JSON Schema，API 请求也包含 `response_format={"type":"json_object"}`。因此不能简单归因于“没有格式提示词”。但这不是本地 Schema 的服务端严格解码：角色分支、字段、引用的约束仍由本地程序验证。旧修复只返回 JSONDecodeError 的行、列和“Expecting ',' delimiter”，没有指出具体的未闭合数组；一次中性修复又重复了同类缺口。失败必须保留，不能自动插入 `]` 后当作 LLM 的成功回答。

[DeepSeek Chat Completions](https://api-docs.deepseek.com/api/create-chat-completion/) 将 JSON object 模式描述为合法 JSON 输出，并以 `length` 标记长度耗尽；本次原始异常与该正常语法承诺不符。代码现在分别记录结束标记、token 上限、用量和容器诊断，便于区分截断与非法完整回答及提交提供方排查。不能用提高 token 上限解释或修复本次错误。

## 先前修复做了什么，为什么 C00 仍失败

`0c1e3ebf` 的 v2 修复把科学校验放回 action 循环：不合格 finish 会收到反馈，并可继续调用工具；区分“曲线可用”和“可选阶段 profile 可用”；Analyst 要实际读曲线，Designer 要沿用相应假设的来源。该修复解决了“调用结束后才拒绝、无法补读”的问题。它使七个条件完成，并恢复了其中 31 次被拒绝的 finish。

但是 C00 chain_00 最终有九个假设，只读了三条曲线；前两个假设有相应曲线，其余七个仅引用 summary。20 次 action 中有 16 次 finish 被拒绝，后期不断重复同样的缺口。旧合法引用 enum 允许任何已交付句柄，摘要句柄在格式上是合法的；科学反馈没有显式的特征声明、具体补读参数或针对重复 finish 的进展约束。旧无进展检测只拦重复工具请求，没有拦重复 finish。

之前只监控完整 W00 成功，足以确认反馈路径可用，却不足以验证 C00 的九假设、多特征情形。完整队列已经结束为 `complete_with_skipped_errors`，并非 C00 正常完成。本轮保留三项失败和原先七项完整结果，采用新协议重试失败条件；不会将旧失败改写为成功。

## 合理的引用与确定性提取

一条可操作假设应声明 `curve_features`：所讨论的坐标 observable 的准确名称。通过 `get_curve` / `get_curves` 获取该特征实际曲线后，引用程序返回的句柄。身份包括 source ID、文件、文件 SHA256 和 JSON pointer。摘要可提供效应、稳定性等背景，但不替代所讨论特征的曲线。

例如声明特征 A 和 B，就需要分别读 A、B，并在该假设的 `evidence_refs` 中提供相应句柄。只读 A 不能支持 B 的变化；另一条假设读 B 也不能让当前假设自动获得 B 引用。Designer 采用某假设后，需要对其每个特征提供该假设实际使用的曲线/profile 来源。分页字段引用保留原始特征身份，不能把不相关字段伪装成该特征。

角色 Schema、原始回答、科学校验、提取后 H 编号与来源映射、奖励编译和签名分别保存。绑定成立仅证明证据确实交付和可追溯，不自动证明自然语言中的数值、因果或剂量结论正确。曲线时间变化率不等于奖励的坐标梯度，更不能直接确定最优 guidance 剂量。

## 本轮代码与 Skills 改动

1. 新增只读 JSON 容器诊断：识别字符串转义、字段路径、未闭合/错配容器、期待符号与字符位置。诊断从不修改原始回答。
2. v3 增加数组闭合示例和针对性反馈，最多两次独立、有记录的中性格式修复；通用接口默认仍为一次。`length` 等非 stop 输出单独标为 incomplete，不通过语法补救伪装完成。
3. 新增 opt-in `temporal-feature-curves-v3`，保留原 v2 gate 与旧 Skill，避免历史源哈希被改写。每个时序假设必须有 `curve_features`，按特征逐项核对真实来源。
4. 显式反馈缺失特征、已读相关句柄和可执行 `next_reads`。新增最多八条曲线的 bounded `get_curves`，依然由 Agent 决定并调用，主机不伪造工具行为。
5. 还没有实际时序观察时，允许的 action 只有工具。连续两次相同缺口后，动态 finish Schema 只允许已经观察且带对应引用的特征；工具分支保持开放，可补读后再完成。允许删除不支持的假设并保留 findings/limitations，不强迫修改 M2。
6. Designer 的 edit 逐个核对采用的假设及各个特征；一个相关曲线不能替代多特征假设的其他来源。
7. 新 Skill 明确以上交付规则，与原研究 Skills 分开配置，没有目标、残基、时间段、温度或环境硬编码。

动态 Schema 在提示词与本地 jsonschema 中生效，**没有声称启用了提供方的严格 schema 解码**。[DeepSeek strict tool calls](https://api-docs.deepseek.com/guides/tool_calls/) 是另一套 beta 协议且只支持部分 JSON Schema。若未来切换，应分别编写兼容 schema 和传输测试，不能把现有 oneOf/if/then/contains 合同原样传入后假定生效。本轮不混入接口迁移或奖励优化。

## 验证与重试范围

确定性测试重放了四份真实非法输出和 C00 最终九假设回答：准确识别七个缺失曲线，连续重复后补读七条曲线，再完成逐项引用。另验证错误特征、摘要替代、每个 Designer 采用特征、分页、截断、修复次数耗尽、原始回答保留及旧接口兼容。

真实 API 验证和后台状态另存 `live_validation.json`；没有该记录或完整凭证时，不能声称线上全部修复。此次恢复队列只包含 W10、W11、C00，研究配方、教师温度、输入窗口、随机数、样本与对照不变。七项旧完整条件不重跑，也不把接口版本不同的新旧调用合并为严格提示词消融。

本地冻结对照记录检查发现，其他任务已经更新了七个生成源码文件。已完成 cohort 的原始生成绑定、结构/评分、执行审计和哈希仍按原先冻结记录验证。本轮启用 `reuse_only`：只准引用已核验的原数值 cohort，未缓存签名或批次会被拒绝，禁止启动新 FLOWR；生成源码差异保存于 `reuse_manifest.json`，不修改旧对照 bank 的哈希。本轮角色重试不会占用远端生成空间。

复现入口：`scripts/prepare_temporal_evidence_recovery.py` 读取已结束的旧 runner，自动选择其失败角色条件，为新的版本目录准备计划、独立 Skill 和可选小规模真实 API 冒烟配置。它不调用 API、SSH 或 FLOWR。
