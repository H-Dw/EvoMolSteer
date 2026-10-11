# 为什么原 M2 没有 Analyst 字段错误，以及接口修复

日期：2026-10-11。本次进行了代码修复、离线回归测试和一次独立真实 API 格式修复探针，没有启动新的消融生成。

## 原 M2 与当前流程并不是同一种接口运行

`configs/experiments/steer_dependency_v1/jobs/M2_static_teacher.json` 直接指定冻结的 `D2_without_trajectory_static.json` 奖励及 `static.json.gz` 教师库。该 M2 任务运行时不要求 Analyst 再生成文档。奖励程序 SHA 为 `91fd96a203e1d4138378fecb01e11d433cd6d9e7b8455da036a70c713c4999b6`，因此 M2 推理成功不能证明新的 LLM JSON 接口不会报错。

旧的 Agent 实验是另一条流程：导出请求，接收 subagent 产出的 `Analyst.response.json`，再验证、编译。Luna 子研究的请求显式记录 `gpt-6-luna`；更早的对应请求没有模型字段，不据此补造模型归属。旧 Analyst schema 也设置了 `additionalProperties: false`，其顶层字段为 `role/condition/findings/directions/limitations`。当前 schema 同样严格，顶层为 `role/findings/hypotheses/limitations`。`design_recommendation` 在这两份 schema 中都不合法。

所以问题不是“旧程序允许额外字段、新程序突然收紧”。原 M2 的冻结推理、旧 subagent 已验证响应和当前重新调用真实 DeepSeek 的任务，需要分别理解。

当前错误发生在完整 `P_struct` 基线第 5 链，尚未执行某个模块的删除或替换。前四条相同完整 Skills 的链都成功，不能把这一格式错误因果归于某个被删除的模块。

## 代码中可以直接确认的原因

1. API 请求使用 `response_format: {type: json_object}`。这保证的请求目标是 JSON 对象；实际代码没有通过该参数发送服务端 JSON Schema 约束。字段契约在提示词中给出，并由本地 `jsonschema.validate` 检查。因此“合法 JSON”仍可能具有不合法的顶层字段。
2. 原格式修复只发送 `Error type: ValidationError`，没有告知模型多出了哪个字段、出错路径及可用字段。模型第二次保留了相同的 `design_recommendation`。
3. 共享合同把“Choose only a registered bounded declarative design, retain the incumbent or defer”同时发给 Analyst 与 Designer；A4 还要求可检验的奖励编辑假设。这可能促使 Analyst 使用设计建议组织答案。实际职责应是 Analyst 提供证据和假设，Designer 决定最终编辑；不能要求 Analyst 同时输出 Designer 的决策对象。
4. 新证据包为 423819 字节，旧 D2 包为 8254 字节；本次失败调用约有 10 万 prompt tokens。旧导出流程把具体 schema 紧接在 system 指令后，当前主请求主要通过任务 JSON 提供 schema。上下文、模型、契约位置和任务表述都发生过变化；这些是合理的风险因素，但单次错误不足以证明哪一个因素具有独立因果作用。
5. 原客户端在缓存检查前重写请求文件；重新调用失败案例也会覆盖同名 `.attempt_1/2.invalid.txt` 和修复请求。原队列只在六条链全完成后报告，所以单次异常还会使部分结果在常规报告中不可见。

## 已完成的修复

接口代码：`src/evomolsteer/deepseek_api.py`。

- 保留原严格 schema，不将所有额外字段放行，不自动把错误响应解释为“保留 M2”。
- 新请求末尾附加统一的输出格式合同：完整 schema、允许/必需顶层字段、额外属性规则和角色字段边界。它不要求任何研究模块必须在响应中出现，也不指定奖励选择。
- 格式修复收到字段级诊断。对本次样本，诊断明确为 `unexpected_properties=[design_recommendation]`、根对象路径、`schema_pointer=/additionalProperties`，以及四个合法顶层字段。
- 在格式修复中再次发送完整 schema，要求将原内容保存在兼容字段中，不添加新的科研建议。没有把 `design_recommendation` 设为新的必需字段，也没有放宽 Designer 可执行编辑的校验。
- 一次格式修复与网络重试使用独立预算，429/503 等重试不会提前用尽格式修复机会。
- 科学请求保持原记录和绑定。实际新增的传输格式提示记录在独立 `wire_N.request.json` 中；回执标记 `json-interface-v2`、实际 wire 请求 SHA 与修复记录。
- 成功缓存需要核对请求与响应校验值；不同请求直接拒绝并保留旧文件。旧 v1 的已验证成功响应仍可读取，不因升级重新调用 API。
- 后续失败尝试、wire 请求和修复请求使用递增文件名，保留原失败样本；持续失败抛出精简的 `LLMInterfaceError`，携带字段诊断而非完整实例转储。

队列代码：`scripts/run_m2_skill_api_study.py`。

- 保存正在调用的条件、链号与角色。
- 记录接口失败，不把其计入成功链，不生成静默 incumbent fallback。
- 每条成功链后及失败时保存部分结果；每条件报告预期链数和是否完整。
- 报告实际格式策略及是否混用 v1/v2；未完成的筛查不允许模块必要性结论。
- 六份有效 Analyst 基线是 Designer 消融的依赖，因此保留明确暂停和可恢复行为，不用缺失或伪造的基线继续比较。

本次没有修改数学奖励、FLOWR 引导强度或窗口。改动属于调用接口与审计层。

## 验证结果

`tests/test_m2_skill_api.py` 与 `tests/test_deepseek_interface.py` 共 **21 项测试通过**。覆盖实际失败样本的精确字段诊断、一次中性修复、网络重试与格式修复预算分离、旧失败记录不覆盖、旧成功缓存兼容、篡改缓存拒绝、嵌套 JSON 路径和队列部分报告。原编译器及奖励梯度测试也通过。

随后用原 `chain_04` 的请求和第一次失败响应，通过 `scripts/probe_llm_schema_repair.py` 进行了独立真实 API 探针。服务端返回 `deepseek-flash`，一次请求成功，没有二次修复；原严格 Analyst schema 验证通过，顶层仅保留合法四字段。原 `findings` 数组及 `limitations` 完全一致，额外设计建议移入 `hypotheses`。

格式修复不是逐字节复制：五项原假设中四项完全相同，一项将字符串中的 `pair-kernel_7.0` 改为 `pair_kernel_7.0`；设计建议被重组为另一项假设。故不能自动声称全部科学内容具有严格等价性。这份输出只作为接口诊断，没有替换原调用、传给 Designer 或记作新增消融链。原两次失败响应及修复请求的 SHA 与 Git 中原记录一致。

真实探针回执、原/新输出保留情况见 `repair_probe/probe_summary.json`。一次成功证明本次故障可通过改进后的格式提示解决，不证明所有未来调用都不会失败；持续失败仍被明确记录和停止。

## 后续消融的使用要求

恢复或新建实验时应统一标记格式策略。严格比较 Skills 模块时，建议在同一 v2 接口下重新取得完整的基线 Agent 响应，再执行模块删除/修改；旧 v1 响应仍作为历史对照保留，不能把接口版本变化归因于模块变化。已有 M2/无引导的数值生成结果可以在输入、批次和程序绑定一致时复用，避免无意义的重复生成。

本次保留原规划和科学 Skills 文件，使用统一的传输格式规则解决接口边界。几何对象混淆、负效应方向一致性误读仍属于下一层的语义分析问题，不能仅凭 schema 合法就宣布这些研究解释正确。
