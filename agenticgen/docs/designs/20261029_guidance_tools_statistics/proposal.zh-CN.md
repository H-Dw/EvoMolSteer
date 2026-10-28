# Guidance Agent 计算工具与稳健群体分析设计

日期：2026-10-29，Asia/Shanghai。状态：代码核查与设计建议；没有注册运行时工具、修改生产代码、切换 skills、修改实验排名或执行新分子实验。计算 skill 是独立 0.1.0 设计草案，不覆盖 Guidance v2.0.0。

## 结论与适用范围

值得配置计算工具，但应把重复、确定的统计计算交给执行器，把证据选择、假设、权重分配交给 LLM。常规统计预计算并缓存，专项问题通过少量只读 tools 查询；不提供任意 Python/shell，也不把分子生成包装为分析工具。该设计延续实验内技能冻结、运行时不问用户、经验只留在任务经验库的决定。

对当前已冻结实验，新增统计只作为旁路诊断，不修改其 winner、标定倍率或失败账本。将稳健中心用于后续正式排名属于新统计协议，需要在下一实验开始前绑定；不能把旧排名改名为新排名。本文建议参数是工程起点，不是已经验证的科学最优设定。

## 当前代码核查

| 位置 | 实际行为 | 直接照搬 function tools 的问题 |
|---|---|---|
| providers/deepseek_api.py:130–151 | call_json 没有 tools 参数；固定最终 json_object 响应 | 注册说明不会改变实际请求体 |
| 同文件:159–168 | 缓存只保存 role 和 content，并调用 content.encode | assistant 的 content=null 会失败；tool_calls、tool_call_id 和 reasoning_content 不在完整记录中 |
| 同文件:242–250 | finish_reason 不是 stop 即失败 | 合法 tool_calls 被拒绝；content=null 还会导致 write_text 的 TypeError |
| 同文件:282–284 | 修正上下文只追加 assistant.content | 带 tools 的 thinking 对话需要完整协议字段，不能继续用该拼接器 |
| uniguide/geometry_council.py:72–85 | 每个角色直接请求最终 JSON；声明一次 execution_claim | 不区分中间工具响应与角色最终结果，恢复逻辑需扩展 |
| 同文件:35–47 | history_view 主要传组均值、计数和窗口摘要 | LLM 缺少逐样本尾部、配对变化、异常影响分析 |
| uniguide/geometry_sampling.py:111–141 | records 已有 seed/slot、几何、构建状态 | 可以只读复算，无须重新生成或额外慢质量检查 |
| 同文件:206–210、264–266 | 组几何主要使用普通均值，最大误差取最大 | 少数有限大值会支配均值；仅聚合字段无法重建稳健统计 |
| uniguide/geometry_protocol.py:47–50、87 | choose 使用总内部距离均方根偏差；标定也调用 choose | 不能仅改提示就宣称实际排名已改成条件原子优先/稳健中心 |

probe.py 使用当前 provider 和 httpx.MockTransport 做无网络核查，保留源文件哈希。它不是新工具实现或真正的端到端工具能力验收。probe.json 是实际运行记录。模拟统计案例与实验对象无关。

## 推荐计算接口

由启动配置提供真实 tool registry、JSON Schema、版本、数据快照和 policy_id。LLM 只传注册 ID/少量枚举；服务器解析原始文件、校验哈希，禁止任意路径或任意代码。工具只读源数据，可在独立审计目录写确定性缓存。

| 工具建议名 | 输入要点 | 输出与作用 |
|---|---|---|
| summarize_population | run_ids、statistics_policy_id | 每个教师分别报告逐分子指标的均值、稳健中心、离散程度、尾部、缺失/失败与分母；附明确异常样本 ID |
| compare_paired_runs | candidate_id、reference_id、statistics_policy_id | seed/slot 对齐，比较同一群体的中心变化、逐槽变化、尾部、失效转移和种子敏感性；协议不匹配时返回 unavailable |
| inspect_interval_evidence | history_snapshot_id、anchor_id、候选窗口、diagnostic_kind | 提取匹配背景下的实测局部对照、成功/反例、绝对增量与异常样本轨迹；没有可辨识对照时不宣称因果 |
| audit_action_batch | anchor_id、合法整数 updates、history_snapshot_id | 原动作校验、展开后实际配置身份、跨分支重复/历史复用、预计新增槽；不生成分子、不预测未测得分 |

统一工具回执至少包含 status、evidence_id、input/source hashes、tool/version、statistics_policy_id、teacher/condition/population、数据覆盖、结果与限制。缺失、数据不兼容和工具异常不能返回成功的零值。重采样使用独立分析随机数；绝不改采样 RNG。外层根据冻结策略产生 eligibility；工具不能以隐含阈值自作科学判定。

## 一次 Agent 决策与一次模型请求不同

应用层的一次决策可以包含多个 API 子调用，无需用户参与。客户端工具至少需要：模型请求工具 → 客户端执行 → 第二次模型请求读取结果并回答。不能要求一次普通 HTTP 请求同时获得尚未执行工具的结果。

最稳妥的两种模式：

1. **预计算模式**：执行器自动提供常用证据包，LLM 一次请求给出最终 JSON。适合现有 call_json 的低改动路径，不能冒称 LLM 直接调用工具。
2. **有界工具模式**：独立 provider 状态机处理 tools；通常一次批量计算后终答，有数据依赖时允许额外一轮计算。轮数、工具次数、总 token、超时由外部配置绑定，不写死在 skills。旧 JSON-only provider 保留给旧协议。

状态转换：PREPARED → MODEL_REQUEST → TOOL_CALLS → EXECUTE_READ_ONLY → APPEND_TOOL_RESULTS → MODEL_REQUEST → FINAL_JSON → VALIDATED。length/HTTP失败/预算耗尽分别进入明确终态；不能当成分子失败或自动填入 incumbent。已无工具需求时可以直接 FINAL_JSON，但必须已经具备所需证据回执。

关键实现约束：

- 有 tools 的响应先检查 tool_calls，再按协议检查终止原因；允许合法 content=null。纯文本“调用某工具”不算工具调用。不要在工具阶段强迫原 proposals/selected 最终 schema。
- 把模型原始 assistant 工具消息完整保留，再为每个 tool_call_id 回传一条 tool 消息。不要人为捏造模型没请求的工具调用；预计算数据应作为普通证据上下文传入。
- 按 DeepSeek 当前 thinking+tools 文档保存并原样回传必要的 reasoning_content；应用不读取或展示其推理作为证据。持久化格式须支持 null、工具参数、call ID、顺序和完整消息；不能复用仅缓存 role/content 的结构。[1][2]
- 最终阶段要求原角色 JSON，并做本地严格 schema 与原科学动作校验。tool_choice/response_format/thinking 的实际组合必须能力探测；例如支持时用 tool_choice=none 结束工具阶段，不应凭假设切换接口。Beta strict 并不替代本地验证；官方列出 maxLength/minLength/minItems/maxItems 等不支持项，现有 schema 不能原样搬过去。[1]
- 一个角色一次父 execution_claim；其下按子调用序号追加记录，不在每次工具结果后创建新角色 claim。缓存键绑定上下文、统计策略、源数据和工具代码版本；恢复时验证所有已完成回执，从未完成子步继续。工具正常续调不是旧请求失败后的自动重试。
- 并列独立调用可批量执行，但所有 call ID 都需结果，包括结构化错误；在收到全部对应结果前不发下一次模型请求。相同确定性工具请求复用结果。参数纠错、网络重试、工具回合和最终格式修正分别计数，防止乘法式无限重试。
- 默认不让 LLM 选择统计裁剪比例/失效阈值来影响排名，不允许借工具扩大预算、访问其他教师不兼容实测或变更冻结 skills。没有注册 capability 时，技能要求使用已有证据或报告缺口，不能编造调用。
- 工具调用不能保证消除 reasoning 耗尽。减少手算/完整历史重复，降低每次决策的输入负担；thinking 模式/额度变更属于单独实验变量，不能静默修改。总输出预算必须累计各子请求，不能以每轮16384掩盖总体增加。

## 群体分析：稳健中心与风险分开

NIST 区分错误数据和真实极端观测；无法证明错误时，不能仅因数值大就删除。[3] rliable 的研究支持同时观察稳健聚合、性能分布和不确定性，但研究对象是基准任务/运行，不直接证明分子候选应采用某个裁剪比例。[4]

不修改单分子的指标：每个分子仍在固定条件原子对上计算距离误差的均方根。不要先跨所有分子平方再汇总，也不要删除分子内部误差最大的原子对。群体稳健聚合只作用于已经定义的逐分子指标；条件原子间距离与条件—邻域距离分别计算，避免更多邻域原子对稀释主目标。教师之间不平均配置或混合分布。

| 方法 | 优点 | 不适合作为唯一依据的原因 |
|---|---|---|
| 普通均值 | 保留每个观测的贡献，兼容旧协议 | 少数极大有限值可支配排序 |
| 中位数 | 对少量极端值稳健、易解释 | 对尾部和非中心的改善不敏感 |
| 双侧各10%截尾均值 | 保留中间80%，比纯中位数利用更多样本 | 估计的是中心而非全体期望；必须同时保留尾部 |
| 中间四分位均值 | rliable 中使用的稳健聚合方式 | 对50个分子仅使用中部约一半；不宜未经比较照搬 |
| Huber 等稳健估计 | 平滑降低极端值影响 | 额外尺度/调节参数，容易出现实现不透明或事后调参 |

建议首版旁路比较普通均值、中位数、双侧各10%截尾均值；后者作为未来“多数样本表现”的候选口径，而不是当前已采用排名。50个均有有效几何测量的样本，排序后去掉两端各5个，只计算中间40个的中心；原50个仍全部保留在账本、失败统计、风险分析中。不是删除异常，也不是挑40个好分子。[5]

每个 run 同时输出四类证据：

1. **典型几何**：条件原子间距离均方根偏差的均值/中位数/截尾均值，条件—邻域对应指标，四分位距或中位绝对偏差。
2. **尾部风险**：第90/95百分位、最大误差、最差5个均值、最大单原子增量及原始样本身份。百分位可能看不见1/50严重异常，不能替代最大值和失败计数。50样本不足以稳定估计极稀有尾概率。
3. **全人口覆盖**：attempts、可测几何数、构建失败、断连、未知评价各自数量。几何可测而化学构建失败的样本仍参与几何统计，并单独计失败；缺失/非有限几何不能 nanmean 后冒充全人口。没有预设缺失处理策略时，完整群体胜负不可用，子集结果仅描述。
4. **达标曲线**：固定、事先指定的条件几何阈值下，达标数/全部尝试数；再单列构建成功且达标的比例、缺失数和未知状态。阈值按接收教师自己的标定/设计要求确定，不从当前候选极端值反推，不把10/50慢质量抽样当成50/50。

不能把稳健中心、所有尾部指标和失败率随意加权为一个新分数。推荐未来策略：先按冻结风险约束区分可接受/受限探索/未知；在可比且可接受的候选中优先条件原子内部几何的稳健中心；邻域仅在既定主目标保护和平台期规则下取舍。未过风险界限的配置可记录为有信息价值的探索，不能替代 incumbent；严重来源不明异常时优先定位、稳健分配。这与允许探索暂时退化、连续退化后回到最佳配置兼容。稳健中心改善不等于风险改善。

## 共同随机数、阶段筛选与不确定性

先校验协议与原始初始化/随机流哈希，再按 logical_seed、logical_slot 对齐。node_id 同名不代表相同随机来源；重复配置相同槽位只能算一次。10→20→50晋级必须复用既有槽，不能当三次独立验证。

报告候选与锚点的逐槽变化、改善/退化/持平数、5个种子分组结果与风险变化。若正式估计量是“候选截尾均值减参考截尾均值”，应在每次配对重采样后分别计算两侧截尾均值再相减；“先算逐槽差再截尾”是另一个估计目标，不能混称。只比较完整、匹配人群；交集子集须明确是描述，不能隐瞒失败造成的选择偏差。[6]

当前相关专项采用5个种子簇的配对分析。建议保守保留种子为簇，配对抽取整簇并同步作用于两侧；不要把50槽直接当50个独立重复。只有来源/相关性证据支持时才采用分层槽内重采样。仅5簇时区间可能粗糙不稳定，配对bootstrap不是精确保证；保留五簇原值、leave-one-seed-out敏感性和“探索性/固定面板”限定。不得为了更窄区间改抽样单位。重采样不消耗生成预算，也不增加真实独立证据。

固定面板被多轮自适应搜索反复使用，普通区间不能校正最优候选选择偏差或证明新样本泛化；沿用用户固定面板要求，不擅加留出采样。小人口筛选只用于排序晋级；中位数/截尾均值在10样本上也不稳定，尤其不能以小人口最优替换完整50人口最佳。不同阶段的截尾规则和配对规则应事先固定，而非事后挑稳定结论。

## 计算 skills 与验证顺序

skills/guidance-evidence-computation/SKILL.md 将“什么时候计算、如何读取证据、什么时候不能下结论”写为标准技能草案。具体工具名和真实 Schema 来自 controller capability，数值策略来自冻结 policy；不把研究对象数值、历史赢家或特定窗口经验写入技能。外部 RSI 以后只能在所有 target 工作完成后审视工具选择/证据解释流程，并另建版本。

实施顺序建议：先构建同一纯计算库与预计算证据回执 → 使用已有 records 做稳健统计旁路比较 → 新增独立 tool-session provider 和注册表 → 基于模拟传输做协议与恢复测试 → 在另行授权的小规模真实 API 验收中检查模型是否实际调用并引用正确回执 → 再绑定新实验。静态 schema 测试不能证明模型会用工具。

至少需要覆盖的验收：工具后终答；合法 null 正文；两个并列工具；依赖工具；完整 thinking 字段回传；工具不可用/未知名字/参数错/超时；length 不伪造成功；总预算耗尽；运行中断恢复且不重复计算/生成；缓存版本漂移；最终非法动作仍被原校验拒绝；工具回执不触发分子预算；来源不匹配/缺失样本/构建失败/同配置复用；整体中等改善/单个极端值/整簇退化；截尾不删除源数据；10/50质量覆盖不扩张。

## 来源

[1] DeepSeek Tool Calls，官方接口文档，2026-10-29查阅：https://api-docs.deepseek.com/guides/tool_calls/

[2] DeepSeek Thinking Mode，官方接口文档，2026-10-29查阅：https://api-docs.deepseek.com/guides/thinking_mode/

[3] NIST/SEMATECH e-Handbook, Detection of Outliers：https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm

[4] Agarwal et al., Deep Reinforcement Learning at the Edge of the Statistical Precipice, NeurIPS 2021：https://papers.nips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html；作者代码：https://github.com/google-research/rliable

[5] SciPy trim_mean 官方文档（裁剪定义；不作为分子协议最优性的证据）：https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.trim_mean.html

[6] SciPy bootstrap 官方文档（paired=True 的索引同步语义；种子簇设计需由本项目另行保证）：https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html
