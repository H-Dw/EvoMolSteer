# UniGuide 搜索复用与 Prompt/Skills V3 改造方案

2026-10-27，基于暂停后的 UG01–UG05 汇总及 UG05 第一教师第四轮 increase 的真实请求。**状态：设计与离线原型；未修改生产入口、既有 skills、manifest、额度或采样算法，未部署、未调用 LLM、未恢复实验。**

## 一、建议的方向

把工作拆成四层：**代码维护已测实验与合法动作，决策视图提供必要证据，skills 规定判断任务，schema 与语义校验保证输出可执行。** LLM 负责比较有限的合法选择并给出简短依据，不再同时承担跨版本查重、格点计算、完整历史整理和输出结构设计。

优先实现计算结果复用与角色输出契约；全局新颖性过滤、候选列表裁剪、较低 reasoning effort 和大位移中止策略应分别登记为协议变化。它们解决不同问题，不能混在一次提示修订中后再归因效果。

## 二、问题发生在何处

下列行号对应本方案编写时已推送的父提交 `10bbbfcb`，用于定位，后续重构可能改变行号。

| 位置 | 现有行为 | 影响 |
|---|---|---|
| `uniguide/pipeline.py:96` | 每版、每教师从 `history=[r0]` 开始 | 跨版本重复符合当前动态记忆重置协议，并非去重函数失效 |
| `uniguide/council.py:24–28` | seen 只包含本版 history；task_memory 明确限定本版 | 没有全局实验索引可供执行层避免相同科学任务重复计算 |
| `uniguide/sampler.py:205–218` | 缓存命中按输出目录 COMPLETE；预算键含版本路径 | 相同配置换一个版本目录会再次生成；当前防重启保护只覆盖同目录 |
| `council.py:25–34` | 每次带完整 domain、history、四阶段 calibration | domain.constant_comparison 又包含双教师比较与原记录，信息重复 |
| `run_uniguide_prompt_version.py:53–64` | 为每条已完成本版 history 追加完整 proposal | 已测数值、自然语言证据、prediction/falsifier 在每轮重复增长 |
| `uniguide_prompt_policy.py:84–110` | 无损共享引用和字符上限 | 减少序列化体积，但没有降低决策任务数量；字符上限只约束最终答案 |
| `run_uniguide_prompt_version.py:46–49` | 固定 thinking/high、16,384 | 不能通过“简短回答”保证 reasoning 留出正文空间 |
| `providers/deepseek_api.py:213、242` | schema 作为额外提示发送；非 stop 响应保存并拒绝 | 实际使用 json_object，schema 是本地校验契约，不是服务端完整字段强约束 |
| `condition.py:7、49`；`sampler.py:Control.after` | finite 检查与原生增量，没有有限值绝对幅度门槛 | 能拒绝 NaN/Inf，不能识别巨大但有限的坐标增量 |

### 真实失败请求的组成

UG05/6pyy/advice_R04/increase 的原始字段紧凑序列化字符数：

| 字段 | 字符数 |
|---|---:|
| calibration_knowledge（82 条，含 12 条失败） | 154,602 |
| domain（含 constant_comparison） | 53,699 |
| history（10 条） | 27,368 |
| completed_action_evidence | 10,181 |
| comparison_best + default_anchor | 5,664 |

原始上下文总长 252,107 字符；v2 共享引用压缩后 134,848 字符。连同角色指引与实际追加的 schema 提示，v2 请求文本约 143,850 字符；真实 API 回执输入为 58,826 token。字符数与 token 数不是同一单位。

## 三、先解决重复计算，再决定是否改变搜索

### 3.1 建立全局科学评价索引

建议新增 `evaluation_registry.py`，键为可验证的 `evaluation_fingerprint`，不要以 UG 版本目录或 prompt hash 作为科学评价身份，也不要仅用 teacher 名、seed 或四个整数索引作为键。

指纹至少包含：

- 教师身份、蛋白/SDF/核心坐标与原子映射、固定模型 checkpoint 的内容哈希；
- R0 checkpoint、全部初始状态和完整 RNG 状态、五种子十槽布局、物理批形状；
- 四窗口定义、合法整数索引，以及实际执行 scale 的精确浮点表示；
- 数值计算依赖：loss、VJP、原生适配、积分器、重评分、随机数、精度、执行环境及确定性配置。

**保留整数格点和原始 scale 双重校验。** 不能只把近似浮点数四舍五入为同一个 j 就宣布缓存等价。缓存所需的数值依赖清单应单独受审计管理；当前 source_binding 包含更多模块，不能随意删掉不方便的绑定以制造命中。

索引状态至少为 `reserved/running/complete/failed`。原子占用同一指纹；complete 必须核验 COMPLETE、records、SDF、终态、轨迹、源 manifest 及评分绑定，失败或缺失不能当作可用结果。相同指纹正在运行时只登记等待，相同指纹已失败时进入审计状态，不能换目录自动补跑。

缓存命中生成 `REUSED_RESULT.json`：明确 source_group、源文件哈希、evaluation_fingerprint、来源 node_id 与种子/槽位、new_sampling_attempts=0。版本中可引用该结果，但不能伪造新的采样 COMPLETE 或把文件复制后宣称新增样本。node_id 在不同组会复用，应使用 `(source_evaluation, node_id, seed, slot)` 的完整身份。

慢检查使用独立 `quality_fingerprint`，绑定源结构、蛋白、PLIP/PoseBusters 实现和抽样政策。科学结果可复用不代表旧质量回执一定可复用；必须分别判断，并保留 50/10/40/0.2 口径及源身份。已有全量旧回执不删除，新视图仅按授权槽位解释。

### 3.2 预算与报告不得因缓存变得含糊

同时记录三个量：`budget_charge`、`new_sampling_attempts`、`reused_attempts`。为先保持原研究的计算额度上限，建议一次复用仍保守占用原计划 50 个决策槽位额度，但 new_sampling_attempts=0、reused_attempts=50，不释放额外采样机会。历史 6,450 次账本一条也不回减。未来如改成只按新采样扣预算，必须另立规则，不能用省下的额度静默扩大搜索。

版本报告应写“30 组决策，X 组新采样，Y 组复用”，而不是一律声称产生 1,500 个新候选；代表组排名可沿既定规则，但不把来源相同的结果作为独立重复证据。

### 3.3 全局新颖性是另一个选择

计算缓存可以保持“每版从 R0 重新思考”的原协议。若目标还包括更多不同配置，则在动作提交前查全局索引，屏蔽已完成/失败/预留的同指纹配置，返回结构化 `already_evaluated`，从合法未测集合选择。

**推荐分两阶段：先做精确复用；再把全局新颖性作为新协议试验。** 新颖性过滤即便不向 LLM 暴露旧分数，也已经改变可选集合与记忆边界；不能声称完全维持原实验可比性。若后续允许读取跨版本旧分数，应显式设置 `memory_scope` 并绑定输入快照。

## 四、输入规范：完整审计原件与 LLM 决策视图分开

保留原始 context/request/receipt，不再要求每个请求反复读取所有原件。新增 `decision_view.py`，以固定字段和确定性规则生成下表内容；保存 view_sha256、源文件哈希、保留/省略字段清单和规则版本。相同数据和协议应产生相同文本。

| 输入模块 | 保留内容 | 删除或移入审计附件的内容 |
|---|---|---|
| task | 教师、版本、轮次、角色、proposal/final 阶段 | 角色混杂与重复任务说明 |
| domain | min/max j、步长、初始 scale、四窗口 | 重复的双教师 constant_comparison 全对象 |
| measured_history | 全部本版已测组的 ID、scale、C、gap、分母、构建数 | 嵌套源路径、回执和重复快照 |
| selector | 程序计算的最低 C、上界、带内成员、代表 ID | 让三私有角色与综合角色各重算一次 |
| comparisons | 有来源的精确差分、变化窗口、不可作单窗归因标记 | 重复长段对比叙述 |
| calibration_summary | 各阶段原始 scalar 指标、分母、全部失败；最大绝对增量及所在窗口 | 非相关窗口的重复详细诊断、元数据；原件保留 |
| window_detail | 默认锚点、当前代表及近期相关组的完整窗口证据 | 所有轮次重复展开同一诊断 |
| previous_actions | 最近已测行动的 anchor、prediction、falsifier 和结果引用 | 每个历史 proposal 的全部解释、传输路径 |
| action_catalog | 已校验合法动作及精确 ID | 要求模型自己计算 2^(j/2)、去重和找合法邻居 |
| quality_state | 明确同步可用范围与未知状态 | 将异步完成时点偷带入冻结同步输入 |

失败、反证和极端值必须保留。压缩后的未知值使用 null/明确状态；不把“字段未提供”改成零。比较标定与正式组时必须带阶段与 attempts，不能删掉分母。

新视图是**有记录的证据投影，不是无损压缩**：原始数值字段在保留时不舍入，但部分细节不送入每次调用。若 LLM 需要未提供的证据，应报告 insufficient_evidence 或由受限检索补充，不能让它假装读过附件。进一步检索 top-K 标定记录必须经过反证/极端值保留验证，再作为独立输入政策测试。

### 离线原型的实际结果

本目录的 `offline_audit.json` 来自上述一个真实失败请求：

- 保留 10 条 history、82 条标定概要和其中 12 条失败，保留 scalar 原值和各条绝对增量极值；
- 原动作规则枚举出 184 种合法推导，对应 136 个不同未在本版测过的 schedule；同 schedule 的不同锚点推导保留在审计附件；
- 用于展示的列表为 12 个动作，按窗口及 fine/coarse 分层选取；它不是已批准的搜索默认值；
- 投影上下文为 39,896 字符，含 skills 与输出 schema 的请求为 **49,689 字符**，相对旧请求文本减少 **65.46%**；
- 若发送全部 136 个动作，投影上下文为 57,812 字符，保留更大选择范围但增加比较负担；
- 7 类错误被离线 schema/语义校验拒绝，包括 r80 误作动作、未知 ID、额外字段、超长字段、重复动作、反证方向矛盾和缺少第二个选项。

这些结果只证明该请求的文本及结构可精简，不能证明 token 减少相同比例、reasoning 收敛、错误率降低或分子指标改善。没有进行任何线上调用。

## 五、输出规范：固定模块，而不是几个长文本框

当前 analysis、evidence、anchor_reason、prediction、falsifier 经常讲同一件事。建议把可计算部分交给控制器，每种角色用独立 schema；原科学提案需要的完整字段在接收合法 action_id 后由控制器展开，注明哪些内容来自 LLM、哪些是程序派生。

| 角色 | 固定内容模块 | 内容边界 |
|---|---|---|
| 私有 proposal | diagnosis、options[2]、uncertainty | 一句观察；两项已有 action_id；各自短理由、证据 ID、C/gap 预测与反证；一个不确定性 |
| synthesis | choices[3]、retrospective、uncertainty | 每分支选一个已有 candidate_id；一句理由及替代项弱点；回看已测预测 |
| final | representative_id、retrospective、limitation | 引用程序已选代表，回看末轮，说明限制；没有 options/selected/R6 |

所有对象 `additionalProperties=false`，要求全部键；ID 为本次输入生成的 enum，文本有 maxLength，列表有限长。schema 自身用 `$defs` 共享定义，避免在同一提示内多次复制长枚举。输出示例仅用于说明结构，不提供可冒充实测结果的分数。

建议初始字符界限：diagnosis.conclusion 180，option.reason 120，uncertainty 160，综合 alternative_weakness 100。不要求输出完整推理过程、重复数组或方程。schema 验证之后仍要做原 validate_action、证据存在、两选项不同 schedule、预测/反证相容和三分支组合校验。

例如一个私有选项可以只包含：

```json
{
  "action_id": "act_<本次目录中的真实ID>",
  "reason": "一个简短且可由证据支持的选择理由",
  "evidence_ids": ["<实际测量ID>"],
  "prediction": {"C": "lower", "gap": "unknown"},
  "falsifier": {"C": "delta_gt_zero", "gap": "unresolved"}
}
```

上面是带占位符的说明；本目录另有通过实际 schema 校验的 `proposal.example.json`。action_id 绑定实际 anchor、mode、整数差和精确 schedule，模型不再重复生成这些可计算字段。lower/higher 的反证分别对应 Δ>0/Δ<0，零变化为 unresolved；这只是方向回看，不是显著性判断。旧自由文本预测无法确定解析时维持 unresolved，不能强行机器评分。

当合法候选不足两项，或三分支没有互不重复的可行组合，应返回结构化 blocked，终止当前决策步骤。不能生成假提案、丢掉分支、放宽去重或悄悄替换种子。综合器最多只有 2×2×2=8 种组合，组合合法性可以完全由代码预先列出。

已准备三份草案：[analysis.md](skills/analysis.md)、[proposal.md](skills/proposal.md)、[synthesis.md](skills/synthesis.md)。它们位于设计目录，未替换线上 skills。

## 六、reasoning 截断需要单独处理

DeepSeek 当前官方说明把 thinking 与 reasoning_effort 分开控制，支持 low/high/max；medium 会映射到 high。仅把 high 改成 medium 不会降低实际 effort。正式试验的可选对照应明确写 low，而不是靠“简短一点”或 temperature 解决；thinking 模式下 temperature 不生效。[官方 thinking 文档](https://api-docs.deepseek.com/guides/thinking_mode/)

当前 Chat Completions 的 response_format 是 text/json_object；代码把 JSON Schema 放在提示中再本地验证，不能把它当成已启用服务端严格 schema。官方 JSON 文档也说明可能出现空内容，length 截断需要正确处理。[API 参数](https://api-docs.deepseek.com/api/create-chat-completion/)、[JSON Output](https://api-docs.deepseek.com/guides/json_mode/)

建议把改造拆成三个离线/影子对照臂，而不是恢复 UG05 后边跑边改：

1. 冻结 v2/high 作为已有日志参照，不为参照重复调用已成功请求。
2. 新决策视图与结构化输出、保持 high 和现有上限，观察表示/任务拆分本身的作用。
3. 相同新视图、显式 low 与相同总输出上限，观察 reasoning effort 的作用。

私有与综合角色以有限动作选择为主要任务，low 是值得验证的候选；复杂论文检索与主版本审计仍由当前 GPT 主 Agent 执行。不能先假设 low 保持决策质量，也不应把非思考模式或其他模型作为静默 fallback。

`max_tokens` 限制总生成量，现有接口没有在本代码中实现“硬保留若干 token 给最终 JSON”的机制。字符上限不是内部 reasoning 上限，也不应使用 `stop='}'` 这类会截断嵌套 JSON 的手段。降低总 max_tokens 可能让正文更早消失；单纯增大总上限也无法修复任务冗余。

每次调用记录 input、reasoning、content token、首个正文时间（若采用流式）、finish_reason、schema/语义错误和修正次数。一次格式修正继续有硬上限；length 或空正文不能冒充格式已修复。失败仍消耗独占 claim，不自动无限重发、不自动换模型/额度/effort。任何新调用均属于未来授权的独立试验，本次为零调用。

## 七、大位移：独立的数值诊断通路

输出规范只能减少错误行动表达，无法保证合法 scale 不产生巨大梯度增量。UG04/R05/increase 的有限绝对位移证明“finite=True”不等于轨迹健康；尚不能仅凭汇总认定具体根因是某一窗口、VJP 或时间系数。

建议新增 `numerical_risk.py`，首先只做诊断：

- 在已有 trace 中定位首次异常 step、seed、slot、atom，而不只看窗口平均；
- 同时记录有效原子的增量 RMS、最大值/分位数、相对原生步长、状态绝对幅度、endpoint loss、梯度范数、时间系数及 assignment 变化；
- 区分实际位移、guidance delta、native delta 和最终核心 C；每项标明单位及有效原子分母；
- 已知极端记录进入风险证据表，LLM 必须看到具体 ID 和数值；不要只给一个含糊“高风险”标签；
- 对整组已完成的有限异常写 `COMPLETE_WITH_NUMERICAL_ALERT` 式独立告警字段，原 COMPLETE/分数/源结构不改，不把告警等同于构建失败。

若进一步引入逐步中止，应单独冻结经审计的绝对阈值和相对阈值，不临时挑一个 Å 数值宣称已标定。中止前保存状态、增量及 RNG；保留整组预留预算，不替换或弱引导重试。硬中止会改变部分终点的可得性和缺失数据口径，必须重新定义完整版本验收，不能在原协议中悄悄加入。

本轮建议先做观测与“完成当前组后暂停下一组等待审计”的控制器门槛，再讨论硬中止。两者都不做梯度裁剪、归一化、改 loss 或隐藏异常样本。

## 八、代码落地与验收顺序

| 顺序 | 改动位置 | 验收重点 |
|---|---|---|
| P0 | evaluation_registry + execute 调用边界 | 相同指纹只采样一次；prompt 不入科学键；任一科学输入变化必 miss；并发只占一次；失败不自动重跑 |
| P0 | proposal compiler + protocol.validate_action | 全部可编译动作通过原规则；rNN/标定 ID 无法执行；重复 schedule 和不可行三分支组合被拒绝 |
| P0 | decision_view + schemas + 三份 skills | 保留分母、失败、极端值、来源；数值不舍入；每角色仅相应模块；普通 R05 与 final 不混淆 |
| P1 | provider/launcher 的版本化配置 | high/low 对照分离；旧 manifest 和缓存原样校验；length 不循环；协议哈希可追溯 |
| P1 | risk diagnostics | 有效原子分母正确；定位首个异常；不改变采样 RNG/数值路线；不自动裁剪或补样 |
| P2 | 全局 novelty + shortlist | 明确记忆与可选集合变化；记录全集/候选集/排除原因；稀缺候选可解释地 blocked |

当前 `pipeline.source_binding()` 覆盖整个 uniguide 包及 provider，直接加文件或改 council/protocol 会使数值验收绑定变化。因此不能在生产目录随意补代码再用旧 acceptance 冒充通过。实施时应建立新的 protocol revision 和完整依赖清单，执行规定的远端验收；或者使用明确审计的控制器适配层，保持数值核心字节不变并验证接入点。不要覆盖现有冻结文件。

建议先用全部历史源请求做离线回放，验证上下文选择、source ID、所有成功旧提案及已知坏例；只以真实高风险/失败例作为回归证据，不增加科学采样。之后才做有独立预算的 LLM 影子试验，覆盖两教师、各角色、普通 R05/final、长历史和候选不足。影子输出不进入正式候选或预算成绩。

验收指标至少包括：新增配置数/实际采样、复用数、重复采样数、非法行动数、first-pass schema 与应用通过率、空正文/length 次数、input/reasoning/content token、机器回看一致性、反证保留率。比较应控制同一输入集合、模型与上限，不能用不同教师/轮次混合均值归因。

UG05 当前停在第四轮，任何未来中途接入都将成为新的协议段，必须保留前三轮及失败账本，并记录 `UG05/R04+` 的修订身份；不是普通 run-version 原样恢复。暂停指令目前继续有效，本方案不修改自动化状态。

## 九、交付范围

- [离线审计数字](offline_audit.json)、[真实请求压缩原件](UG05_R04_increase.prompt_source.json.gz)、[决策视图示例](input_projection.example.json)、[全部合法动作及推导](action_catalog.audit.json)。
- [私有 schema](proposal.schema.json)、[综合 schema](synthesis.schema.json)、[最终分析 schema](final.schema.json)，以及对应 example 文件。示例不是模型回答或正式提案。
- 三份 skills 草案及只读审计脚本 `agenticgen/scripts/audit_uniguide_prompt_redesign.py`。
- 原始请求 SHA256：`2bd4d64a82f70094ca0c80572073fc7f81c87c2b80622638ea0a271ef9c28ee1`。
- 外部来源仅用于核对 API 控制能力，访问日期 2026-10-27；DeepSeek 文档为滚动网页，无固定 commit，未据此改动部署参数。

本次已完成代码定位、真实输入审计、固定输出/skills 草案和离线结构验证。全局缓存、生产 prompt 集成、在线 reasoning 对照和数值中止尚未实现或运行。
