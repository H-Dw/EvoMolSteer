# v3 证据、角色输出和程序绑定契约（拟实现）

本文件是下次开发的语义契约，不代表当前 API schema 已支持这些字段。先实现并通过真实 API 的固定案例验证，再在 runner 中启用。现有 `local_motif` schema 不应通过忽略未知字段来假装兼容。

## 1. Analyst 输入

固定包包含：目标/口袋身份和 hash、词表、学习/控制范围、终态目录及评分/失败定义、实际祖先/对照索引、选择分解、局部区域与模式 registry、启用知识模块、parent 程序及响应、历史失败家族、可执行 actions 和工具 schema。

ReAct 工具循环使用 JSON function/tool calling；工具参数限定为注册 ID、时刻域和允许的统计请求。返回小摘要与大数据的不可变引用，不把全轨迹嵌入提示词。工具未知或证据不足返回 `need_tool`，不让模型生成可被直接 exec 的任意 shell/Python。

拟工具组：`inspect_ancestor`、`compare_regions`、`selection_shift`、`window_trend`、`compare_modes`、`chemical_context`、`controller_response`、`failure_family`。若沿用旧工具名，必须由能力清单映射到真实实现；计划名不能自动当作可调用函数。

## 2. Analyst 输出的必需语义

| 字段 | 含义与确定性校验 |
|---|---|
| `schema_version / decision` | 固定版本；`propose / need_tool / defer` 互斥分支，分别校验必填项 |
| `hypothesis_id / parent_program_sha256 / evidence_bundle_sha256` | 调用者提供或注册的身份，不能引用另一个 parent |
| `evidence_refs / counterevidence_refs` | 逐项必须真实存在；findings、hypotheses、设计建议等所有嵌套引用统一校验 |
| `terminal_basis` | 终态候选/对照 ID、最终评分定义、质量与未知记录，禁止中间分数冒充成功真值 |
| `mode_ids / compatibility_observables` | 已注册模式、当前 seed 可计算的相容量；不能使用当前新样本的未来真值 |
| `support_id / support_sha256 / support_reason` | 真实局部点和必要邻居、对称性、非连续关系；所有模式/项并集严格小于有效生成点数 |
| `observable_spec` | 名称、单位、方向或目标带、状态表示、实际可用算子 |
| `clock_spec` | forward/状态时刻、教师目标时刻规则、预测类别来源、学习/控制域、插值域 |
| `chemical_assumptions` | 元素/角色/电荷/键/微态的来源、可靠性、缺失及适用条件；无化学信息时显式几何模式 |
| `allowed_design_changes` | 本条件可改模块、不可改协议、允许的支持/函数范围 |
| `formation_retention_value_tests` | 形成、保持、终态收益分别怎么观察，含反例与否证 |

输出使用 JSON Schema 的闭合对象（`additionalProperties:false`）、枚举与条件分支。数值范围由运行配置注入，不在 Skills 或 schema 内硬编码目标残基/学习窗口。

## 3. Designer 输入/输出

输入必须包含 Analyst 完整已验证结果、其工具证据、同一 parent 程序、实际 registry 与控制响应。不得仅给一个摘要段落再让 Designer 自选目标。

输出包括 `analyst_output_sha256`、`hypothesis_id`、`compiled_terms`、`support_union`、`clock_policy`、`mode_policy`、`dose_policy`、`categorical_sensing`、`derivative_sources`、`parameter_changes`、`control_falsifiers`、`terminal_tests`。

每个 term 精确绑定 evidence/observable/support。`derivative_sources` 允许当前坐标、生成坐标预测、生成元素/电荷及已验证键 head；affinity 分支不在允许集合。E0/E1/E2 必须明确 `stop_gradient` 位置。仅感知匹配不能声称执行了可微角色奖励。

`parameter_changes` 用完整路径记录原值、新值、单位、原因和实际消费者。没有消费者的字段拒绝进入活动程序；继承但本算子不使用的字段可保留在说明中，但不当作有效项。未实现函数返回能力请求，不输出“看起来能运行”的伪代码作为已编译程序。

## 4. 编译与请求—执行一致性

确定性编译器完成：

1. 所有嵌套证据引用、候选/模式/节点身份、哈希和范围检查；
2. 表示/时刻兼容、窗口内节点、禁止外推；对 current/proposal 时钟分别校验；
3. 支持、邻居、词表、对称性、化学前提和每 seed 所有有效项并集检查；
4. 数值函数签名：实际目标数据、原生协议、算子版本、路由/记忆/调度与导数开关；
5. 编译清单与文字声称的实际正权重项逐一一致；
6. 有限差分、平移/旋转一致性、模式置换/原子对称性、零梯度/达到后减力、总预算匹配等小型数学检查；
7. 真实 forward 记录 affinity VJP=0、额外 affinity 调用=0，以及实际活跃窗口/步数。

Hungarian、离散角色识别、top-k 支持选择或模式路由有不连续处时，有限差分应在固定同一分支下核验条件导数，再另测切换稳定性。不能要求跨分支的普通导数相等，也不能用冻结分支测试掩盖实际切换抖动。

## 5. 渐进模块功能验证

固定少量真实历史案例，分别改变知识模块的启用/删除，保持其余输入、API 模型/默认温度及 Designer 固定。检查“交付→工具→可操作假设→reward→响应”，不要按是否出现某个词验收。

| 改进模块 | 期望响应 | 失败判定 |
|---|---|---|
| 祖先时刻 | 不同节点或表示被实际绑定，目标时刻随配置正确改变 | 文案称同刻而仍读固定节点 |
| 剂量 | 归一化后每步/累计真实位移随调度变化 | 只乘共同 reward 系数、轨迹无变化 |
| seed 模式 | 不同兼容状态产生不同路由；相同输入可重现 | 所有 seed 使用同一全局教师顺序却称自适配 |
| 化学角色 | 支持/站点/方向/条件至少一项改变且有证据 | 只把 N 概率重命名成受体 |
| 电荷 | 字典解码正确、分通道、有效梯度源可测 | 只读字段未使用，或类型/图不合法还称成功 |
| 局部选择 | 机制区域与同大小对照可区分，Designer 未加点 | 新算子的支持并集覆盖整个分子 |

本轮只验证文档与静态引用；这些功能验证仍为待执行，不声称 Skills 已产生性能改善。

## 6. 保存策略

保存一次 evidence bundle、每调用原始请求/回复/工具结果与哈希、已提取的 JSON、编译清单和错误诊断。推理 trace 以逐 seed 的少量量为主：时刻/模式、请求与实际剂量、目标误差、类别/角色摘要、裁剪/切换；事件稀疏保存，不重复整个教师库。

保存终态结构、统一评分、所有失败及能源/结构报告；中间结构沿用无损共享轨迹包。可选诊断不默认落全 Jacobian/逐原子逐步大表。工作缓存可重建，原始 Steer 和完整恢复状态永久保留。
