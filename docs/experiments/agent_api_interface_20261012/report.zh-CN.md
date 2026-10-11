# 真实 LLM API 契约与工具工作流验证

日期：2026-10-12。模型：真实 DeepSeek `deepseek-flash`。沿用温度 0，
thinking disabled；未使用 subagent 替代 API，未运行新的 FLOWR 分子生成。

本次完成了分步请求、来源引用、固定字段提取、Designer 交接、确定性奖励编译和
Reviewer 引文提取。格式测试通过不代表科学假设正确，也不代表亲和力提高。

## 改动

| 旧风险 | 新机制 |
|---|---|
| 根部混入 `design_recommendation` | Analyst 与 Designer 分别使用严格白名单 schema；公共格式 Skill 明确字段归属 |
| `hypotheses` 内引用未经约束，或拼错表名/行号 | 程序分配来源 handle；每轮 schema 只允许引用已交付 handle；所有嵌套字段均校验 |
| 合法发现超过 12 条便报错 | 新接口取消 findings 科学数量上限，保留独立的 token、上下文及轮次预算；不自动截断 |
| Reviewer 改写原文后无法通过 exact quote 校验 | Reviewer 选真实回答片段编号；程序按 chain、JSON pointer 和字符位置还原原文 |
| 大包输入与全量编号枚举反复出现 | 初始工具目录，后续按模型请求分页；引用枚举仅含已交付记录；请求文本内容地址缓存 |
| 松散 JSON 提取或缓存绕过校验 | 单对象严格解析；拒绝重复键、非有限数、额外文本与截断；重新验证原始回答和缓存校验值 |

新增 `agent_workflow` 模块分别管理 contracts、sources、engine、reviewer 与
ablation_adapter。格式 Skill 不包含目标残基、固定学习时段、科学模块的强制功能、
VPN 或 Git 配置。实际活动研究指引仍由原模块注册表决定。

## 实际测试

| 测试 | 地点 | 真实 API 请求数 | 结果 |
|---|---|---:|---|
| 14 条 findings 格式探针 | 本地 | 1 | 14 条完整保存，未触发格式修复 |
| 完整 Analyst 指引 → Designer → 奖励编译 | 远端 | 6 | 6 条 findings、2 个 hypotheses、19 次合法来源引用；Designer retain，编译签名与冻结 M2 相同 |
| 14 条 findings 格式探针 | 远端 | 1 | 14 条完整保存 |
| 删除 A1.1 的 Analyst 指引 | 远端 | 4 | 4 条 findings、2 个 hypotheses、15 次合法引用；不强制补回被删除模块 |
| Reviewer 真实引文提取 | 远端 | 1 | 2 项评分的片段编号合法，引文由程序逐字还原 |
| 曲线、定义、嵌套字段工具路由 | 远端 | 5 | 实际执行 get_curve、read_definition、get_evidence、read_field，再 finish；5 条 findings、9 次合法引用 |
| 合计 | 本地与远端 | **18** | **0 次格式修复** |

18 是实际模型请求数，含工具循环各轮；不是 18 个独立科学消融或分子实验。
远端占 17 次。总计 prompt tokens 103,299，completion tokens 6,075；所有成功
receipt 的 requested/returned model 均为 `deepseek-flash`。

另有 **88 项**单元与相关兼容性测试通过，覆盖不存在的来源、未知工具、非法参数、
缺失/错误字段、14 条发现、分页保留弱/null 证据、完整曲线读取、跨链引用、重复
工具死循环、null 回答的有限修复、原始回答缓存损坏、固定温度和原消融接口兼容性。

本地归档重新检查了全部 18 份供应商原始回答、解析后的 JSON、wire 请求及内容
地址文件的 SHA256；解析结果与原始文本严格解析所得对象相等。
归档约 0.68 MiB，不重复下载可由原始训练证据重建的 compiler evidence 副本。

## 测试证明与未证明的内容

固定格式、实际工具参数、来源存在、回答提取、交接和编译链条均经过真实 API
验证。Designer 本次选择保留 M2，没有将新温度或未注册参数写入奖励。

Reviewer 对完整指引和删除 A1.1 的两份小样本回答均评为 A1.1 的 1 分，即泛泛提及。
完整条件没有充分呈现明确的即时选择正负群体对比。因此，本次不能证明该研究
模块的目标功能已经充分实现，更不能由这两个评分判定 A1.1 必需或冗余。
这是科学功能评估结果，与 JSON/来源契约异常分开记录。

本次没有评估新的预测 pIC50、应变或生成质量。需要另用接口一致的科学 full/drop
批次检验这些目标。少量格式案例成功也不保证所有未来回答均无异常；超预算或
非法返回仍会明确失败，不会伪装为有效奖励。

## 运行与版本

主要代码提交：`6502a50`；补充工具路由和空回答测试：`85c74cc`。
远端先拉取源码再测试。测试归档各自记录实际 commit、输入指令和来源 SHA256。

现有消融派发器曾为源码提交短暂停止，随后已恢复，保持 `direct-v4`。
新的独立 API 测试不会混入该冻结队列的 full/drop 分子统计。
现有入口已支持 `interface_mode="evidence-tools-v1"`，科学 runner 支持
`agent_interface: "evidence-tools-v1"`；活跃 study 不允许中途切换接口。

标准独立调用使用 `scripts/run_agent_workflow.py`；小规模自测使用
`scripts/run_agent_interface_checks.py`。详细输入输出与工具说明见
[接口架构](../../agent_api_workflow.zh-CN.md)。

可核对 [总验证记录](verification.json)、
[远端主测试](remote_contract_probe/summary.json)、
[远端工具测试](remote_tools_probe/summary.json)、
[Analyst 实际输出](remote_contract_probe/citation/Analyst/Analyst.json)、
[Designer 实际输出](remote_contract_probe/citation/Designer/Designer.json)、
[Reviewer 提取结果](remote_contract_probe/Reviewer.extracted.json)。
