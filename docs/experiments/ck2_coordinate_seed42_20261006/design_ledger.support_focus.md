# VAL116/NOS 支持度驱动的第五轮候选

日期：2026-10-06。身份：subagent simulation，模拟 Analyst / Designer API 的可审阅响应。

本记录仅提出条件性候选；没有读取 Round4 结果，没有运行推理、编译、清理或修改源码与配置。Round4 已启动，其结果未知。先完成 Round4 评价、归档及既定清理，再由主流程冻结第五轮；当前文档不构成自动执行指令。

## 决策

第五轮只将 Round4 的 joint ASN117+VAL116/NOS 区域集合缩减为 **VAL116/NOS**。保留 `coordinate_mixture`、`native_rms_ratio=0.05`、`mixture_temperature=0.25`、`robust_delta=1`、`dose_reference=predictive_flow`、`preserve_native_rigid_pose=false`。沿用原生 100 步、master seed42、现有种子/初态配对、参考提取、正则化、caps 与评价口径。动态窗口由请求读取为 [0, 0.5]，之后 native 续推至 1；不增加 SMC。

这是**去除缺乏核心支持的区域约束**的探索，不是已识别 VAL116 的亲和力因果机制。η=.05 是沿用的工程剂量，不是从相关系数估计的最优强度。由两区域八维变为单区域四维会相应改变 q 的维度归一化与梯度，故仍需实测实际注入量，不能声称数值奖励或实际剂量完全相同。

## 证据与反证

|问题|本次 discovery 0–13 证据|允许的解释|
|---|---|---|
|ASN117 是否具有本定义下的核心支持|all/NOS 的 endpoint-anchor 与 current 5 Å 核心权重均为 0|本轮可移除其局部约束；不推断残基在生物学上无功能|
|VAL116/NOS 支持度|selected anchor 核心质量 0.724177476116，selected current 核心质量 0.00016943977033；有效槽位约 1.90|主要是预测终点定义的未来区域槽位，不是当前真实接触；约两槽位限制协方差与结构可辨识性|
|native_y 的选择偏移|全窗口 −0.046773033934，q=.0024620473；首节点 −0.0469313428864，其余合计 +.000158308952435|首步主导，不构造持续 −y 力；首节点占绝对贡献和 74.19%，不是 100%|
|proposal_x 的调整后下一步 gain 关联|r=.0318474115707，CI [.0169770235855,.0467177995559]，q=.0189623642788；首节点占绝对贡献和 .3663%，其余贡献 +.0320666309025，有效时间节点 32.659|小效应分布于多个节点，不能用 native_y 的启动解释将全部 lag 否定；也不能直接构造固定 +x 力|
|proposal_x 的其他统计|选择偏移 q=.99809940935；调整后同期评分关联 r=−.0087037313958、q=.757789504611|选择富集、同期关联和 lag 关联必须分开，没有完整因果链|
|proposal spread|选择偏移 q=.73963616438；调整后 lag q=.499449798087|可作为联合形状参考，不支持单方向持续收缩|
|独立性|窗口末平均根数 1.2143；14 批中 11 批仅一根；lag 最后时刻 .49|批次是统计单位，不能把克隆、原子或时间节点当独立重复|

关键引用均为复合请求已有 ID：

- `coordinate:discovery:ck2:A:ASN117::NOS::anchor_core_weight:population_mean`
- `coordinate:discovery:ck2:A:ASN117::NOS::current_core_weight:population_mean`
- `coordinate:discovery:ck2:A:VAL116::NOS::anchor_core_weight:selected_mean`
- `coordinate:discovery:ck2:A:VAL116::NOS::current_core_weight:selected_mean`
- `coordinate:discovery:ck2:A:VAL116::NOS::effective_slots:selected_mean`
- `coordinate:discovery:ck2:A:VAL116::NOS::native_velocity_y:selection_shift`
- `influence:discovery:ck2:A:VAL116::NOS::native_velocity_y:selection_shift`
- `coordinate:discovery:ck2:A:VAL116::NOS::proposal_centroid_x:lag_partial_gain_correlation`
- `influence:discovery:ck2:A:VAL116::NOS::proposal_centroid_x:lag_partial_gain_correlation`
- `coordinate:discovery:ck2:A:VAL116::NOS::proposal_centroid_x:selection_shift`
- `coordinate:discovery:ck2:A:VAL116::NOS::proposal_centroid_x:partial_affinity_correlation`
- `coordinate:discovery:ck2:A:VAL116::NOS::proposal_spread:selection_shift`
- `coordinate:discovery:ck2:A:VAL116::NOS::proposal_spread:lag_partial_gain_correlation`
- `D_COUNTS`、`D_LINEAGE`。

主 240 特征的 q 与新增 transport/support 80 特征的 q 不能伪装为统一的 320 特征校正结果；本次区域集合的 q 也不能直接与旧 40 区域检验相比较。均值不为零的显著性仅描述坐标或尺度，不构成选择优势证据。

## 原生积分与剂量来源

`native_integrator_source_audit.json` 记录实际原生更新（以下为模型归一化坐标）：

\[
 x_{new}=x+\left[v_{flow}+g_t\frac{t v_{flow}-x}{1-t+10^{-6}}\right]\Delta t
 +\epsilon\sqrt{2g_t\,\mathrm{coord\_noise\_level}}\,\Delta t,
 \qquad v_{flow}=\frac{\widehat y_t-x_t}{1-t}.
\]

低于 .9 时 \(g_t=1/(t+.01)\)；本例 \(g_0=100\)、\(\Delta t=.01\)，启动 score drift 几乎抵消初态先验。该源码中的噪声乘子为 **dt**，不能擅自改写为 sqrt(dt)。因此 observed-native displacement 包含 drift 和随机增量，不能全当成预测 flow。首步机制与 native_y 节点贡献相容，但不证明其他所有 lag 都是伪影。

第五轮沿用 Round4 的 predictive-flow **剂量标尺**：

\[
 d_{flow}=\Delta t\,(\widehat y_t-x_t)/(1-t),\qquad
 a_t=0.05\,\mathrm{RMS}(d_{flow}).
\]

单位应与实际注入坐标一致，原子 mask 与 RMS 定义沿用实现；归一化方向来自奖励梯度，幅度仍受既有 caps、阈值和可用性约束。该变化不改原生 SDE、不删除 t=0，不让奖励复制 flow 方向，也不使 `native_rms_ratio` 字段重新表示 observed-native 比率。需分别报告 predictive-flow RMS、observed-native RMS、desired/applied RMS 与裁剪频率。

## 可执行的奖励定义

设 \(u=x^{native}_{t+\Delta t}\)。使用同一次预测终点 \(\widehat y_t\) 定义 VAL116 区域的 Gaussian 槽位权重与硬 NOS 掩码，并在本次求导中冻结；核参数沿用已有特征/编译契约，5 Å 只是核心支持诊断半径，不新加为奖励截断。

从 **u** 计算四维特征 \(z(u)=(c_x,c_y,c_z,s)\)：加权质心及加权 spread；其实现和特征定义必须一致。梯度是 \(\nabla_uR\)，不经过 endpoint 的模型 Jacobian，不假装对硬类型或键求导。下一步可以重新预测槽位和类别，但单次有限差分与反传必须固定同一组权重、类别和模型状态。

\[
 q_b(z,t)=\frac{(z-\mu_b(t))^T\Sigma_b(t)^{-1}(z-\mu_b(t))}{D},\quad D=4,
\]
\[
 \rho(q)=\delta^2\left(\sqrt{1+q/\delta^2}-1\right),\qquad
 R(u,t)=\tau\log\sum_{b=0}^{13}\frac1{14}\exp[-\rho(q_b(z(u),t))/\tau],
\]

其中 \(\tau=.25,\delta=1\)。\(\mu_b,\Sigma_b\) 必须取自真实 discovery 批次参考，沿用 Round4 的提取、时间插值及协方差正则化，仅保留 VAL116/NOS 对应子空间，不能由报告均值臆造数组。14 个组件代表批次经验分布，不是已证明存在的 14 个真实姿态模式；克隆数量不决定混合权重。执行前由主流程记录实际 reference、program 与源码 hash；本次未编译，故没有可声称的第五轮程序 hash。

**时间对齐**：评分 t 的参考对应 proposal 时间 t+dt。最后控制 .49→.50；score=.50 的 proposal .51 不能用于 .50 的末端目标或继续注入。分析保留完整 51 节点及其贡献，执行参考按既有正确时间映射截取；不切固定 .1 bins。缺少可用 NOS、数值异常或投影后极小梯度时，应沿用已有跳过/记录策略，不能补零后正常化成大剂量。

## 对既有 Round3 的有限解释

以下由主流程提供，作为开发反馈，未伪造为请求中的 discovery evidence ID：Round3 真实 shape 指标改善 2.587%；valid/PB-fast 为 100/99，对照 native 为 98/98；有效分子 head 为 7.45576 对 7.43704，unique 均为 69。MMFF 每重原子应变释放指标为 .570958 对 .548773，周围松弛 RMS 为 .366832 对 .327214；原始 SMC 对应 head=7.501144、MMFF=.398679、surround=.202921。

这些结果支持几何及可生成性出现有限正向信号，尚不支持改善周围适配或物理应变，也不支持超过原始 SMC。head 是同一模型的预测分数，不是真实结合亲和力；更大的 MMFF 松弛释放及周围松弛不能称为更低生成态能量。不同结构之间的单一 MMFF 数值也不是结合自由能。

Round4 只改 predictive-flow 标尺，其结果在此未知。第五轮与它比较才适合解释区域删减；与 Round3 直接比较同时包含剂量标尺差别。

## 条件性执行与可证伪标准

1. 先评价并归档 Round4，按用户既定政策清理其生成结构/轨迹/缓存/压缩包，保护原始 SMC 与 checkpoints；主流程在读取结果后确认是否仍执行本候选。当前无新的清理动作。
2. 冻结其余代码与参数并绑定参考哈希；核验真实 proposal 上的 FD、原生/zero 全轨迹一致性、硬标签冻结、mask、100 步与窗口后零注入。若这些契约失败，不作科学效应解释。
3. 使用相同 seed42 初态与原生随机流进行配对；保留 Round4 报告、配置和实际剂量统计。比较真实 t=.5 坐标双向结构指标和局部四维偏差，不能只汇报被奖励特征或 predicted endpoint。
4. 同时报告 t=1 的 valid/PB-fast、全部失败分母、unique/clone、多样性、同状态 head 重评分、有效图上可计算的 MMFF 及收敛覆盖、周围松弛和冲突；保留 native 与历史 SMC 参照。均值按批次比较，不将样本数当作独立批次数。
5. 若局部四维偏差下降而实际 shape/反向覆盖退化，则反证“该低维区域表征足以恢复结构”；若实际 shape 改善但周围松弛/MMFF 不改善，则仅支持几何模仿，不支持兼容性假说；若只有 head 上升，也不能宣布物理或真实亲和力改善。
6. 若 VAL-only 在匹配剂量下不优于 joint，记录删减假说未获支持；若改进主要伴随注入强度变化，只能联合报告，不把变化纯归因于区域名称。继续沿用既定质量门槛，不在观察结果后另造数值晋升标准。

## 未执行的刚体保持候选

`preserve_native_rigid_pose=true` 保留为**另一项独立消融**，不加入当前第五轮主候选。若后续获准检验，应匹配 VAL-only/false 与 VAL-only/true，避免同时改区域和投影却声称单因素结果。

在同一 active-slot 度量下，用当前原生 proposal 构造平移与无穷小转动基 \(B\)，取 \(P=I-BB^+\)，\(g_{int}=P\nabla_uR\)。对固定槽位、可微奖励及正交 P，有

\[
 \nabla_uR^Tg_{int}=\|P\nabla_uR\|^2\ge0.
\]

这仅保证一阶 ascent：有限步、重新预测的掩码、数值不稳定、逐原子非均匀 clipping 或后续约束都可能破坏该性质；需对真正注入方向作 FD 并审计净平移/转动残差。正标量整体缩放保留方向，不能将任意后处理都称为同一个正交投影。退化几何需稳定伪逆，极小剩余梯度不能因归一化而获得满额剂量。

刚体分量抵消会在周围原子上产生平移/转动补偿，因此它们**不是完全未受控区域**。零净平移与线性化转动也不自动等于有限步精确保持原生刚体姿态；实现若有刚体对齐/回缩须另行说明。该投影只检验内部形状控制的可能性，不是 MMFF、键角或受体兼容性的能量最小化。

## 溯源与合同验证

- 输入：`results/coordinate_endpoint_anchor_v2/agents/Analyst.coordinate.request.json`。
- bundle SHA256：`338525148de6bad39a8818cf1dd4045c8fb1aa9768a22f206ca5c247d1ab37c4`。
- main manifest SHA256：`08aeea01350806452a042c9fbbc7ef6607b8e9349742f55993d003fca3b189b2`。
- transport/support manifest SHA256：`010ba742338ee1a0a772be219165f9fe16093d8464d17d4a5a1dccc7d238d85e`。
- node-influence manifest SHA256：`3adf44201008c850bab215fbd4a445a5e9b8aaefa67675843dcfc4a8cdefd6ea`。
- 原生 integrator 源 SHA256：`e9abd83167f708da4712db2ac49ab24ac44a39c59c46a54c20789c7873c58000`。
- selective-loop 源 SHA256：`beaace3ab92a8377bc8eab95333c8f40ce370ca9224f7a78927c5a424bb38426`。
- `Analyst.support_review.response.json`：严格 request schema 检验通过。
- `Designer.support_focus.response.json`：严格 coordinate-1.0 schema（包括两个可选字段）检验通过。
- 26 个不同引用 ID 全部存在于复合请求；规则特征与引用相符，VAL116/NOS 四项 proposal 特征均已供应。
- 只做本地合同检查，没有调用会改写 agents 结果目录的 import，也没有编译或验证 GPU 执行。
