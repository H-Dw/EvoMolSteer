# Designer：自然分支混合奖励的小预算反证试验

本次为新的实际 Designer sub-agent 模拟调用。已通过 `flowcompat_supplemental.payload` 校验当前 request、三份字面 Skill、全部绑定证据、四个实际工具 receipt、完整 Analyst 响应、registry 与报告上下文，并读取其内容。响应经现有 CLI 导入和编译通过，输出 `branch_pilot.json`，轮次为 9。本文给出可复现的证据、设计与检验依据，不记录私有思维链。未执行模型、远端生成或提交代码。

## 决策与唯一改动

选择 `base_program_id=R26_branch`、`formula_id=branch_mixture`，只设置 **`branch_mixture.virtual_mass=0.1`**。原始 R26 继续作为已验证默认方案；这是检验方向信息是否有用的探索试验，不是性能升级声明。

0.1 是公开的实验预算选择，不是从相关系数换算的概率，也不是新的拟合结果。现有联合 teacher 的跨批次即时方向一致性很弱，confidence 最大仅 0.00107922，故本次每个原模式至少保留 90% 先验质量。方向默认正向，区域混合默认 0；不改变 R26 的 `native_rms_ratio=0.33`、稳健代价参数、教师数量、原生随机流、积分器或路径/单原子步长上限。之后若探索反方向、预算或区域权重，必须顺序改变一个参数，分别保存角色响应与编译程序。

## 使用的观测与反证

紧邻复制子代的端点坐标严格相同，不能充当已经出现自然变异的独立对照。两步共同祖先下的不同立即父节点，则提供 672 个已观察条件事件；14 个生成批次是推断单位。joint innovation 的原始/调整后相关性为 −0.141814/−0.149932，原始/调整后协方差为 −5.15017e−5/−3.59935e−5 Å·pIC50。调整后关联支持同背景下分析变异，但不支持普遍禁止坐标或化学图变化。

方向参考由实际 `branch_mutation` 工具生成：1400 个原始联合 teacher 坐标、评分与批次保持不变；前两个缺失祖先节点为零场。非零 confidence 比例为 20.6429%，均值 1.98260e−5。已有审计中的自然变异 RMS 均值 0.01117 Å，最大 0.18920 Å；可用跨批次方向的平均余弦约 −0.002824。评分并列和相同点云对照各 1344 次均为精确零。因此只把高评分 teacher 相对同祖先低评分自然分支的联合坐标差，视为待反证的局部外推方向。

多深度证据另外提出局部方向：depth5/region17 的调整后协方差 XYZ 为 `(7.98668e−7, 3.85376e−6, 5.20463e−7)` Å·pIC50，其中 y 的 q=0.012369，z 的 q=0.772123；region18 的 +y 在 depth3/5 也存在绝对效应。全窗口协方差函数选为常数，时间导数为零；region18/depth5 的调整后**相关性**拟合另选为一阶，不能把所有统计函数统称为零阶。这些不同统计量均不能直接转为每步空间力。region17 的逐事件完整 XYZ 余弦仅 0.045882；报告中窗口积分余弦 0.420743 是另一个描述尺度，91 个依赖批次对不构成新增重复。

本次注册公式使用的是冻结 lag-two teacher 联合方向，**没有实施多深度 ROI 合成**。先前 round02 的新机制梯度差与实际轨迹差均为零，证明那次添加没有响应，不能据其旧 R26 非零梯度宣称本次方向生效。当前证据也未测得这些方向对最终亲和力、能量或模型模块的因果效应。

## 可微奖励与真实导数

在输入学习窗口内，使用未移动的原始 teacher 完成 Hungarian 匹配和最近教师选择，冻结对应关系 `σ_k` 与原有评分/距离先验 `π_k`。当前注册规则将具有非零 confidence、非零原始方向幅度、真实两步共同祖先、至少两种观测变异且存在不同立即父节点低分对照的 teacher 标为 `e_k=1`。前两个节点及无资格 teacher 不加入虚拟模式。这是观测资格，不代表已校准的可靠性概率。

令完整 teacher 点云为 `T_k`，其 RMS 归一化联合差为 `d_k`，自然幅度为 `a_k=min(raw_direction_RMS_A,0.2 Å)`，方向符号为 `s`。固定两种完整中心及其质量：

\[
T_{k0}=T_k,\quad T_{k1}=T_k+s a_kd_k,\qquad
p_{k0}=\pi_k(1-\alpha e_k),\quad p_{k1}=\pi_k\alpha e_k.
\]

这里 `α=0.1`、`s=+1`。原始中心与虚拟中心共用原匹配及评分先验；既不拼接互不兼容的局部片段，也不对原子类型求优化。confidence 不乘到 Å 位移上。独立区域权重机制为 `w_ki=1+μ c_k e_k(m_ki−1)`，本轮 `μ=0`，所以全部权重为 1；未来 `c=0` 区域仍必须为单位权重。

定义 `r_kbi=Y_i−T_kb,σ_k(i)`、`q_kb=N⁻¹Σ_i w_ki‖r_kbi‖²`，以及原 R26 代价 `ρ_δ(q)=δ²(√(1+q/δ²)−1)`。奖励为：

\[
R(Y,t)=\tau\log\sum_{k,b}p_{kb}\exp[-\rho_\delta(q_{kb})/\tau].
\]

`τ` 与 `δ` 均沿用实际注册程序/基类默认，不把其他 reward 的 `robust_delta` 字段误当 pointcloud 参数。模式 posterior 为 `A_kb=softmax(log p_kb−ρ_δ(q_kb)/τ)`，冻结上述条件量时：

\[
\nabla_{Y_i}R=-\sum_{k,b}\frac{A_{kb}w_{ki}r_{kbi}}{N\sqrt{1+q_{kb}/\delta^2}}.
\]

实际 FLOWR 在旧状态 `X_t` 上给出端点 `Y=scale·F_θ(X_t,C_detached,P,t)+COM`。单次现有反向图计算 `g_X=J_Fᵀ·scale·∇_Y R`，保持 self-conditioning、离散标签与几何对应关系的条件约定。不将 `dR/dt` 当空间梯度，不求 affinity head 导数，不增加生产 model/head forward，不做 Steer 重采样，也不限制原生化学图变化。控制仍在原生更新之后注入，是滞后的条件 VJP 控制，不声称精确条件采样分布。

窗口从 request/reference 动态读取；当前绑定值为 `[0,0.5]`，评分观测为 0.00–0.49。前两个节点只有原 R26 引导，没有本次新增分支场；窗口结束后继续无 guidance 的原生推理。

## 顺序检验与回退

1. 先验证 α=0、零方向和无资格数据的精确标量/梯度对照，并核对空接口的配对轨迹一致性。实施门读取真实 virtual mass、teacher shift、相对新增梯度和 `t=window_end` 坐标变化；旧奖励的非零梯度不能替代这些证据。
2. 运行本次正方向小预算；保持同初态与随机流，随后只反转 `direction_sign`，并与 null/R26 比较。保留实际累计 RMS、动作成本、碰撞回退与受控节点覆盖。相同 η 仅代表请求比率相同，不保证实际剂量完全相等。
3. 通过响应检查后，再比较最终预测亲和力与配对 R26、无引导及历史 Steer；区分可评分比例、平均/上尾和最佳样本，避免分母变化冒充提升。应变是次级结果。正反方向共同改善时，优先考虑模式平滑/剂量替代解释；正向稳定优于反向、null 和 R26，才进一步支持方向信息有用。独立冻结确认前保留 R26。
4. 后续预算或 confidence 条件化 region mix 均为独立单轴试验。native/gain/uniform-dose 控制在原 R26 标量上另外顺序比较；记录 `raw_gradient·actual_injection`，预处理后的控制内积另存，避免把预处理向量称为新势函数梯度。

即时方向噪声、模式切换、滞后原生更新和生存者条件偏差都可能导致失败。无响应先检查有效剂量；响应后亲和力变差则说明该外推/适用背景有问题，不能用放大 confidence 修补成“可靠”方向。

## 尚缺的全窗口 ROI 工具

下一种真正不同的空间规则需要正式工具将 ancestor Gaussian 区域的全 XYZ 协方差函数与每个完整 teacher 的软成员联系起来，联合解重叠区域约束、处理整体平移、保留周围结构，并按观测自然变异幅度约束位移。还需输出批次方向覆盖、不确定性、反向/null 对照及新的完整参考哈希，然后注册可执行 formula。本次 registry 没有该工具或参考，不能把 region17/18 的协方差向量直接替代已有 lag-two teacher 方向；也没有编造已运行的 regional 规则。

## 调用绑定

- Request：`7f062205b36fa37bd1956d9851b04b569cbcf97324e5d3903f45abd00faa9bb7`。
- Evidence：`2c5aeeb35d79b61be184eed8a7d5a3dc8dd965a49f2da75cb9616b38058cb5d5`。
- Analyst：`da0848b71b60c772ce65eb014b41da9b5e869046fd6143489b5d74dca8d1043f`。
- 冻结 branch reference：`50d7232b5501152c87fe20329ebfa0e4b0fbfd2f519cdd0e1e6dad9b8f23f801`。
- Literal input 保存为 `Designer.simulated.payload.json`；声明响应为 `Designer.simulated.raw.json`，规范响应为 `Designer.flowcompat-supplemental.response.json`，导入校验记录为 `Designer.flowcompat-supplemental.validation.json`。
- 注册源码、三个常规 receipt、多深度 receipt、Skill/工作流与 context 哈希全部保存于编译程序的 `flowcompat_provenance`。

导入与编译证明响应被规范工作流接受；不证明 FLOWR 中新增机制已经响应，也不证明最终性能提升。
