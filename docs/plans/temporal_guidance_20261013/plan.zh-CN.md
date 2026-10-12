# 时序剂量与多阶段坐标奖励的独立探索计划

本次为 **2026-10-13 的前瞻规划与本地实现**。不调用真实 LLM API，不连接远端，不运行 FLOWR，也不重新分析大型结构轨迹。当前精简 Analyst/Designer、M2、原始 Steer 和固定对照库保持原样。新方向保存为可选扩展 Skills，启用与否由条件配置决定。

## 1. 假设合理，但“单一方向”需要重新定义

当前 M2 的平均预测 pIC50 为 7.70475，无引导为 7.49508，历史 Steer 为 7.65148；M2 的 p95 为 8.12716，Steer 为 8.23844。在固定阈值 8.258901977539063 下，有效高亲和力分子分别为 3/400 和 37/1000。M2 的应变中位数和 p90 并不差于这组 Steer。因此，问题主要是**极高亲和力尾部与路径覆盖不足**，目前没有证明“大量早期异常位移”就是原因。数字来自冻结的 `compact_skills_verification_20261012/verification_quality.json`，均为预测指标，非实测亲和力。

M2 并非沿同一个固定向量推进：每个 seed、每一步都会基于本次 FLOWR 端点预测重新做几何匹配、选择近邻教师、计算后验响应，再通过实际 FLOWR 的端点 Jacobian 回传。固定的是奖励形式和教师来源策略。它可能存在以下问题：

- 早期端点预测尚不可靠，当前状态与教师几何的对应关系、近邻选择较不稳定。
- 固定教师库压缩了历史路径；接近同一终端构象的早期路径可能很多，当前库不一定保留这些路径的可达性。
- 以近邻几何为主的教师先验不直接优先极高亲和力教师；改变时序不能自动补齐缺失的尾部标签或教师。
- 控制器对梯度做 RMS 归一化，几何不适配时仍可能注入较大位移；原子位移上限、累积位移上限和碰撞回退会改变实际剂量。
- Steer 的克隆与淘汰会逐渐富集适应路径的粒子；纯 guidance 对所有初始粒子施加控制。二者的在线搜索预算、初始样本数及依赖结构也不同。

这些是竞争解释。新实验应同时保留“时序没有帮助、需要改善教师覆盖或奖励目标”的否证结果，不预先断言早期必须减弱。

## 2. 文献能提供哪些借鉴

| 原始来源 | 可借鉴的结论 | 本项目的边界 |
|---|---|---|
| [Applying Guidance in a Limited Interval，NeurIPS 2024](https://arxiv.org/abs/2404.07724) | 图像扩散中，高噪声时的 CFG 可能压缩分布；指导强度的作用随阶段不同。论文中平滑权重也并非总能超过区间开关。 | 不把图像的噪声区间直接转换成 FLOWR 的时间节点，不承诺平滑调度有效。 |
| [Stage-wise Dynamics of CFG，ICLR 2026](https://arxiv.org/abs/2509.22007) | 多模态分析区分早期方向偏置、中期模式分离和后期收缩，提示弱势模式可能在早期失去机会。 | CFG 与本项目的坐标奖励 VJP 不同；三阶段只是机制假设，不能当作已发现的分子阶段。 |
| [On the Guidance of Flow Matching，ICML 2025](https://arxiv.org/abs/2502.02150) | Flow Matching 的引导需要依据其条件路径推导；端点梯度近似有适用条件。 | 当前端点 VJP 是实际模型的导数，但其注入不等于精确的目标分布采样；不宣称理论无偏。 |
| [FK Steering，2025](https://arxiv.org/abs/2501.06848) | 中间势函数可以表达未来高奖励倾向；不同势函数影响路径保留、效率和多样性。差分势可能低估已经提前达到高奖励的路径。 | 借鉴“当前增益与未来价值分开”的分析思路；新推理继续禁止粒子重采样。 |
| [How to Guide Your Flow，2026](https://arxiv.org/abs/2604.27147) | Flow-map 奖励引导从确定性控制角度研究路径与终点收益。 | FLOWR 不是现成的 flow-map 模型。本轮不引入新模型、训练、额外生产前向或 affinity 梯度。 |

相应公开代码包括 [flow_guidance](https://github.com/AI4Science-WestlakeU/flow_guidance)、[tvcfg](https://github.com/sqt24/tvcfg)、[FK-Diffusion-Steering](https://github.com/zacharyhorvitz/Fk-Diffusion-Steering)。本次仅借鉴机制与实验对照，不直接复制其采样算法。文献中的扩散时钟可能从高噪声向零反向计数；FLOWR 本项目时钟为从先验走向结构，不能混用。

## 3. 两条独立路线及实际执行公式

设真实 FLOWR 的端点预测为 \(y_t=\hat X_{1,\theta}(X_t,t,P)\)。当前静态教师奖励为：

\[
 R_{\rm static}(y)=\tau\log\sum_k\exp\{\log\pi_k-\rho_\delta(q_k(y))/\tau\},\qquad
 \rho_\delta(q)=\delta^2(\sqrt{1+q/\delta^2}-1).
\]

\(q_k\) 是在冻结 Hungarian 对应下的端点均方距离，\(\pi_k\) 是分离梯度的几何近邻先验。\(\tau\)、先验温度、近邻数与 score beta 保持 M2 默认值。教师具有不同构象时保留 logsumexp 混合，不对坐标求平均。

### W：只改实际剂量，保持奖励目标

\[
 g_t=J_{\hat X_1}(X_t)^\top\nabla_yR_{\rm static}(y_t),\quad
 \Delta X_t^{\rm control}\propto\eta\,w(t)\,c_t\,g_t/\operatorname{RMS}(g_t).
\]

\(c_t\) 沿用已验证的 native/predictive-flow RMS 标定和原子、路径、碰撞控制。\(w(t)\) 写入控制器实际读取的 `dose_gate`。若仅用 \(w(t)R\)，归一化会抵消幅度变化，可能造成“改了函数却没有改注入”的假象。

所有 W 候选满足 \(\sum_i\Delta t_i w_i=b-a\)，初始注册范围为 0.25–2，学习窗口内每步保持正剂量。这个条件匹配的是**请求比例的积分**；由于 \(c_t\) 及回退依赖 seed，它不保证实际位移积分相同。必须额外比较每个节点和 seed 的 requested/delivered RMS、累积位移、cap 与回退。不能把总实际剂量改变误判为时序效应。

### S：固定剂量，改变教师在各阶段的相关性

\[
 R_t(y)=(1-b(t))R_{\rm static}(y)+b(t)R_{\rm observed\ forecast,t}(y),\qquad 0\le b(t)\le1.
\]

动态部分读取原始 Steer **同一评分时刻的端点预测教师**，不是将早期含噪声的 \(X_t\) 当成 \(X_1\)。两个解析标量使用同一个本次 FLOWR 端点；对合并后的标量做一次 VJP。无需新增模型前向或对 affinity head 求导。

权重用 smoothstep 连续连接；**没有插值教师坐标**。现有逐节点教师库可能发生成员替换，平滑权重不保证教师集合或梯度本身连续。若没有 ancestor/node ID，不能宣称跟踪了同一优秀祖先路径。实际角度跳变、模式覆盖和身份替换需要检查；不自动添加图相等限制或筛掉不适配 seeds。

### C：固定教师和剂量，只改几何曲率

探索 \(\delta(t)\) 从较小的稳健响应尺度过渡至 M2 默认尺度。\(\delta\) 改变 pseudo-Huber 代价与混合响应，**不是温度或容忍半径**。在小残差时代价约为 \(q/2\)，大残差时约为 \(\delta\sqrt q-\delta^2\)；较小的 \(\delta\) 降低大残差的代价，增大 \(\delta\) 可能加强对远教师的惩罚及混合锐度。当前注册 0.5→1 与其反向及恒定 0.75 对照；不把它称为已经更适合早期 seed 的函数。该路线与 S 分开；RMS 归一化后应验证方向与教师责任是否实际改变。

W、S、C 均从 M2 独立出发。只有单独试验通过后，才在新的配置版本里注册组合路线；初始配置没有启用组合条件。

## 4. 全窗口挖掘与阶段证据

此前冻结证据中有 200 条区域坐标曲线，全部未通过低阶多项式保真检查，实际方法为观测节点插值。不能从现有报告直接宣布最优分段边界。`mine_temporal_guidance_profiles.py` 新工具读取已保存的批次节点矩，而非再次展开全部结构特征。

分析始终覆盖整个学习窗口的实际评分节点，不做固定的 0–0.1 等分层统计。针对每个区域的 soft mass、局部位移、形状协方差等：

1. 按独立批次形成优势–对照标准化差异曲线，保留反方向、弱效应和不确定性。
2. 报告全窗口积分效应、批次置信区间与 BH q 值；计算节点变化率及其批次区间。节点区间是描述性区间，不伪称同时显著。
3. 在相同全窗口支持上比较常量、线性、一个及两个转折的连续样条。每个留出批次内部，仅用其他批次选择节点；使用 one-SE 规则偏向简单函数，并避免数值舍入诱导虚假复杂度。
4. 检查转折在各留出批次中的稳定性。最多两处转折，最少节点间距，稳定比例不足时不注册“数据支持的阶段”。这些是探索性证据，不能代替独立验证。
5. 将连续变化和人群替换分开读：沿用既有 `turnover_accounting`、parent-adjusted effects、survival effects。选择压力、局部效应幅度和变化率均不直接给出最优控制权重或空间梯度。

输出只有一个紧凑 `stage_profiles.json`：均值曲线、变化率、区间、全窗口统计、CV 损失、候选节点及稳定性、输入和代码 SHA。保存双精度，不缓存新逐 seed 明细，不复制坐标或原始特征表。原始失败与谱系继续留在原数据中。

`get_curve` 与新增的可选 `get_stage_profile` 为真实 API 的只读 ReAct 工具。新增工具不开放 shell、任意路径或任意模型生成代码。证据引用沿用绑定文件 SHA 和精确来源的 handles，Designer 继承真实 Analyst 已读来源。

阶段节点分两种：初始规范化 0、0.5、1 是**干预超参数**，映射到实际已观测的评分时间支持后使用，不是新发现的生物阶段。当前是评分时钟 0–0.49，最后一次更新到状态时钟 0.5，不能将未观测的 0.5 当成已保存的 forecast。这样正向、反向及常量混合/曲率的离散平均参数相同；控制窗口仍来自原 metadata。数据支持的节点由 `register-profile` 读取经批次稳定性检验的 profile 注册。LLM 选择注册程序，不能自由捏造节点。

当前证据明确没有 decoded terminal learning labels。其生存关联不能自动解释为最终 pIC50 改善。若要研究极高亲和力祖先，需要先用已有终态/谱系工具验证 decoded 最终评分与祖先的连接，再将高尾部、普通生存和灭绝对照分开；本轮不虚构这项数据已经齐备。早期选择的环境适配类比有启发性，但几何关联本身不能揭示模型内部模块的因果贡献。

## 5. 新 Skills、Prompt 与条件矩阵

当前 `skills/analyst` 与 `skills/designer` 保留。新增四份扩展：

| 扩展 | 核心新增功能 | 不改变的约束 |
|---|---|---|
| temporal-dose-analyst | 实际全窗口曲线、时序证据、剂量假设及否证；不默认早期有害 | 优势/失败对照、关联/因果区分、可追溯交接 |
| temporal-dose-designer | 把假设映射到实际 dose factor；分离比率预算和真实位移 | 奖励几何、教师、温度与 eta |
| staged-reward-analyst | 时变区域几何、forecast/current 区别、教师覆盖与 seed 适配 | 不发明绝对坐标目标、不将生存当最终亲和力 |
| staged-reward-designer | 同端点多标量混合、一次 VJP、曲率或教师变化分开测试 | 真实模型、固定剂量、无 Steer、允许化学图变化 |

运行时的顺序为：**当前角色 Skill → 配置选择的扩展 → 本研究独立合同 → JSON 接口**。不附加旧 G 合同中的固定控制器限制。新合同仅为显式选择的 lane 开放对应时序参数；温度等旧约束没有放松。Skills 不包含 VPN、GitHub、服务器环境或当前 target 残基。

每条路线先采用扩展开关的 2×2 对照：`00` 均关闭、`10` 仅 Analyst、`01` 仅 Designer、`11` 两者。每条件六条空白上下文 DeepSeek Flash 角色链，固定先前 API 温度，两种等义 task 表述各三次。无需 subagent 模拟。

Analyst 输出仍为 findings/hypotheses/limitations；时段含义放在 hypothesis 的 observable、direction、scope、falsifier 中。Designer 仍输出 decision/edit_choice/evidence_refs/hypothesis_refs/rationale/failure_modes；确定性程序从注册选择中提取数组。**不新增容易漂移的自由 JSON 字段或任意 Python 奖励源码**。

| 顺序 | 候选数值程序 | 科学对照 |
|---|---|---|
| W，剂量 | 早期较轻→较重；中间较强；早期较强→较轻 | 原 M2；比例积分一致；时间反向对照 |
| S，教师 | observed→static；常量 0.5 混合；static→observed | 原 M2；固定剂量；常量与时间反向对照 |
| C，曲率 | 0.5→1；恒定 0.75；1→0.5 | 原 M2；固定教师、剂量与温度 |
| 数据节点复核 | 已支持的 profile 转折替换预设节点 | 相同参数水平，区分边界来源 |
| 最后组合 | 仅组合单路线已确认有效的程序 | 各单路线与组合的比较；新版本预登记 |

初始配置共 **10 个 Prompt 条件、9 个数值候选**，不是十个已经派发的任务。C 先进行 `00/11`；只有实际输出响应且存在质量信号，再补齐该路线的角色删除条件。所有阶段逐个推进，不并行探索或边做 S 边修改 W。

Algorithm control 与 Skill 效果必须分开：即使 Agent 全部 retain，也可用 `compile-control` 验证已登记的数值策略；这只能证明策略实验，不能记为 Agent 或 Skill 的效果。若 Agent 的输出仍为 M2，直接复用 400 个 M2，记录“本次 Agent 输出未改变执行函数”。

先验证指引是否实施：实际指令 SHA → 曲线/profile 读取 → Analyst 假设与真实引用 → Designer 参数选择 → 执行表 SHA → trace 中剂量/方向变化。没有实际响应时，先修复交接或程序瓶颈，不解释质量差异。静态函数审计只证明交付/提取，科学解释还需人工审查及未来实际 trace。

六条链按**数值程序众数**选择，平票优先 M2；不看新质量挑选幸运回复。数值等价输出共享生成结果，六次调用不等于六组独立分子证据。

## 6. 样本量、判据与停止规则

初筛每个新数值程序 200 个（批次 6376–6379），确认 400 个（6376–6383），master seed 42、原先相同的 indexed-batch prior 与采样噪声。确认阶段仅补齐余下四批，不重复已经生成的 200；缓存按执行表、checkpoint、生成源码与 RNG 协议绑定。相同程序跨 Prompt 条件复用。已有 400 M2、400 无引导及历史 Steer 不重新生成。

初筛用于识别失败方向，不能宣称改善尾部。固定四批筛选后的八批确认也不是完全独立验证；如通过，应在预登记的新批次上做独立确认。400 个仅有约三个原 M2 高尾部事件，对稀有路径推断较弱。可在独立预算批准后扩展至少 1600 个候选及匹配的**新增**对照输入；旧 400 始终复用。初始配置不派发此扩展，也不自动生成新对照。

- 主指标：所有生成尝试的最终预测 pIC50，同时报告评分覆盖和全部失败。评分缺失不插补为低分或删除后称作全尝试结果；覆盖改变时给出缺失敏感性分析，暂缓均值优势结论。
- 时序目标：相对 M2 的 p95/p99、固定阈值下的有效高亲和力比例及区域/教师覆盖。最大值不能在 400/1000 不同样本量下直接比较。对 Steer 的克隆与唯一结构/谱系另做描述。
- 亲和力保护：候选–M2 的批次配对区间下界须高于 −0.02 pIC50；声称提升则区间下界须大于 0。与无引导要求有明确优势。确认尾部改善应报告批次区间，并对预登记候选的多重比较校正。
- 历史 Steer 为独立批次、不同在线预算，单独 bootstrap，不伪称配对或显著超越。M2 之前约 +0.053 的均值差不能当作已证明优势。
- 次指标：有效率、PB、收敛 MMFF relief/heavy 中位数和 p90，明确覆盖及收敛失败。暂定确认时应变相对 M2 不恶化超过 10%、有效率不降低超过 2 个百分点；区间不清楚则标记待验证。
- 机制诊断：全窗口 requested/delivered RMS、累积量、上限触发、碰撞回退、梯度范数/邻步角度变化、0.5 对应窗口结束态适配。时序表的 smoothstep 不保证真实梯度平滑；教师成员和近邻切换需结合保存的轨迹解释。
- 若方向变更但亲和力下降，检查教师表示/时钟、错误目标、稀有模式遗漏、标量分量的梯度抵消和 VJP 敏感性，不能简单增加强度。若方向/位移没有改变，先定位实际生效字段和 cap，而非直接判定奖励有效。

低于保护线的候选不进入组合，保留报告与 commit，回到 M2/当前已确认单路线。无显著效果不自动等同无效：先看功效、输出是否变化、实际剂量与目标是否改变。原始 Steer 数据与固定对照结构、评分、应变、失败记录继续保留。

## 7. 可执行接口与后续用法

计划配置：`configs/experiments/temporal_guidance_20261013/study.json`。默认 `execution_enabled=false`，`enabled_conditions=[]`。本轮仅生成请求文件，以下真正运行命令均供以后获准实验时使用。

```powershell
# 本地准备：无 API、无结构计算、无 FLOWR
python scripts/temporal_guidance_study.py --config configs/experiments/temporal_guidance_20261013/study.json --mode prepare
```

全窗口额外分析有单独输入/输出接口：

```text
python scripts/mine_temporal_guidance_profiles.py --moments <batch_node_moments.parquet> --prior <region_prior.json> --packet <packet.json> --output <stage_profiles.json>
```

将 profile 中某个已稳定的转折注册为新条件配置，不启动任何实验：

```text
python scripts/temporal_guidance_study.py --config <plan.json> --mode register-profile --profile <repo-relative-stage_profiles.json> --feature <actual-feature> --recipe W1_early_soft --new-recipe W1_data_knots --new-config <new-plan.json>
```

后续调用时，另存 run 配置和新输出目录，显式激活当前一个条件并设置 execution_enabled；不要直接改变已绑定的 `prepared_final` 目录。该目录是本次最终请求文本；此前本地草稿没有发布为运行配置。API key 使用现有环境接口，调用仍为 `deepseek-flash`，不在 Skill 中写环境要求。

```text
python scripts/temporal_guidance_study.py --config <activated-run.json> --mode agents --condition W11_both_extensions --execute
python scripts/temporal_guidance_study.py --config <activated-run.json> --mode infer --condition W11_both_extensions --runtime-config <runtime.json> --execute
```

数值时间反向/常量对照可显式 `--mode compile-control --recipe <declared-recipe>`，然后 infer 时增加 `--recipe`；记录为预注册数学对照，不假称 LLM 新设计。部署配置在 `runtime.template.json` 中；checkpoint 路径需填原冻结文件并通过 SHA 验证。远端只运行生成，代码仍按既有本地 commit/push→远端 pull 流程部署。

新 FLOWR 入口是 `scripts/generate_scheduled_dependency_flowr.py`，额外要求 `--allow-execution`；继承已审计的无 Steer 端点控制器，继续全程 native SDE，只有元数据窗口内添加坐标 VJP。

结构导出沿用生成入口的 `--export-terminal`，归档用现有 `archive_generation.py`，本地最终评分/应变使用 `evaluate_dependency_archive.py`；结果分析沿用固定 bank 的批次配对统计。新 `analyze_temporal_guidance_delivery.py` 单独审计控制 trace，不额外调用模型。完整数据归档仍使用已有可校验紧凑格式；本轮不删除任何已有结果。

本地单元验证覆盖纯调度、窗口、剂量归一化、实际剂量因子、来源引用、编译约束与合成预测器的一次前向/VJP。它不构成真实 FLOWR/Jacobian 验证或生成质量结果。后续真实实验仍须先通过实际模型的数值 preflight 和每步前向计数检查。
