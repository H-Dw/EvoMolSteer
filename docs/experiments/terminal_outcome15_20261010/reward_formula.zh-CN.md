# TO12/TO14–TO15 实验候选的可执行奖励公式（未采纳为当前最佳）

本文记录 TO12 配置实际调用的坐标奖励和控制路径。它描述程序如何计算和注入力，不表示奖励已提升最终亲和力。

十五轮已完成：该候选虽然优于无引导，独立验证的均值低于历史 R11/R26，且应变尾部偏高，因此未采纳。本轮 `active_program.json` 已恢复为历史 R11 的时钟匹配版本，采用即时 on-target 教师分数，不包含本文的最终家族 utility 与尾部收缩。其端点点云混合、真实 FLOWR VJP、0.025 虚拟质量及 0.33 原生流 RMS 标定机制相同；教师点云及 score 的来源不同。参考 `active_workflow.json` 与 `final_decision.json` 区分实验候选和实际保留配置。

## 奖励作用在 FLOWR 端点预测上

在步 (t)，FLOWR 用当前坐标 (X_t) 做一次原生前向，得到预测端点 (Y_t=f_{\mathrm{FLOWR}}(X_t))。奖励是在 (Y_t) 上计算的端点点云相似度；它不是把最终解码构象直接当作当前态坐标力，也不是 affinity head 的梯度。当前态的梯度为

\[
g_t=\nabla_{X_t}R(Y_t)=J_{\mathrm{FLOWR}}(X_t)^\top\nabla_{Y_t}R.
\]

实现中先从预测端点复制一个停止梯度的 anchor，用它确定教师对应、排序和 prior；这些离散/先验量在反传时冻结。奖励对 live 端点坐标求导，再经同一次 FLOWR 计算图的 Jacobian VJP 回到 (X_t)。原子类型和 affinity 输出均停止梯度；不调用 affinity 梯度，也不增加每步 production forward。端点坐标以复合物世界坐标 Å 表示。

控制器先完成原生随机推进，再向 proposal 注入由步前 (X_t) 的 VJP 构造的有界位移：\(X_{t+\Delta t}=X^{\mathrm{native}}_{t+\Delta t}+\operatorname{bounded}(g_t)\)。这是滞后一个积分步的坐标注入，并未在新 proposal 上额外重算端点梯度；self-conditioning、原子类别、assignment 和 prior 都视作本步固定条件。日志的首阶响应是该步前梯度与实际注入的内积，不能解释为重新前向测得的奖励或亲和力提升。

## 家族标签、尾部项与批次权重

教师的最终质量来自 decoded-final 标签。对 exact-copy family (f)，先在每个化学图 (g) 内求有效 decoded 分子的最终亲和力均值，再让每个图等权：

\[
\mu_f=\frac{1}{|G_f|}\sum_{g\in G_f}
\left(\frac{1}{|V_{fg}|}\sum_{k\in V_{fg}}y_{fgk}\right),
\]

其中 (V_{fg}) 是该图中有效 decoded 分子。最终标签时钟是 1.0；TO6/TO9/TO12 的教师排序使用最终家族均值。TO6 的主先验使用均值，TO9 使用均值加原始尾部比例，TO12 则使用下文的均值加收缩尾部比例 utility；TO12 公式中的 (s_m) 明确是 (u_f)。中间 head 分数仅保留在继承的即时分支方向构建及条件匹配诊断中，不是这些教师的先验质量分。

尾部定义使用 final pIC50 阈值 (8.258901977539063)。原始 family tail fraction (p_f) 的分母是观测到的全部 terminal offspring slots，**包括无效 decoded offspring**；分子是其中达到阈值的有效 offspring 数。批次基准 (p_b) 使用同一分母。TO12 的 count shrinkage 是

\[
\widetilde p_f=\frac{n_f p_f+2p_b}{n_f+2},\qquad
u_f=\mu_f+0.25\widetilde p_f.
\]

(u_f) 是构造 prior 的启发式 utility；0.25 项不与 pIC50 同单位，也不是经校准的 native success probability。收缩强度 2 是计数启发式，不是 Bayesian posterior。family 中后代有共同祖先且受选择影响；灭绝分支的未来标签仍未知，不用批次基准填补。

TO12 计算器在 1,203 条 family 记录中报告 134 条 singleton、510 条 prior utility 改变；batch-equal raw tail fraction 为 0.0690860，regularized fraction 为 0.0608179，family-mean 分量变化为零，batch-equal utility 变化为 -0.00206703。这些是参考库计算结果，不是 TO12 的生成效能。参考的 batch base log weight 对每个事件内的 donor batch 等权，再在 batch 内对教师等权；程序校验的字段为每条教师记录
\(b_m=-\log N_{b,e}\)，其中 (N_{b,e}) 是该事件该批次的教师数。

## 冻结 anchor 的教师选择与多模态点云奖励

令 (A_t=\operatorname{stopgrad}(Y_t))，第 (m) 个教师端点点云为 (T_m)。每个教师都通过 Hungarian 最小平方距离 assignment 与 anchor 匹配：

\[
\sigma_m=\arg\min_\sigma\sum_i\|A_{t,i}-T_{m,\sigma(i)}\|^2,
\qquad
C_m=\frac1N\sum_i\|A_{t,i}-T_{m,\sigma_m(i)}\|^2.
\]

按 (C_m) 选最近的 (K=4) 个教师点云。TO12 的 prior logits 为

\[
\ell_m=-C_m/T+\beta(s_m-\bar s)+b_m,
\quad \pi_m=\operatorname{softmax}_{m\in S_4}(\ell_m),
\]

其中 (T=4.0\ \text{Å}^2)，\(\beta=2.0\)，(s_m) 是参考库中绑定的教师 utility/score，(b_m) 是上面的批次等权 base log weight。批次等权只描述完整教师库的基础权重；最近 K 个教师的截取、几何邻近项和质量项会改变实际归一化先验，并不保证每个生成分子的 prior 在批次间等权。选择、assignment 和 \(\pi_m\) 均冻结。当前端点 (Y_t) 到匹配教师点云的平方距离均值为

\[
q_m(Y_t)=\frac1N\sum_i\|Y_{t,i}-T_{m,\sigma_m(i)}\|^2,
\quad
\rho_\delta(q)=\delta^2\left(\sqrt{1+q/\delta^2}-1\right),
\]

\[
R_0(Y_t)=\tau\log\sum_{m\in S_4}\pi_m
\exp\left(-\rho_\delta(q_m(Y_t))/\tau\right),
\qquad \tau=0.5.
\]

该 endpoint point-cloud 实现读取 `pointcloud_delta_A`，TO6/TO9/TO12 配置未覆盖它，因此使用代码默认值 1 Å；配置中的 `robust_delta=2` 不被这条 endpoint point-cloud 公式读取。

## 即时分支的虚拟模式

TO12 从 TO9 的 family-tail-0.25 参考继承 instantaneous 分支字段；它不使用 TO7 的 final-family-matched 分支方向。分支资格要求已观察到前两个选择事件、共同祖父与不同直接父节点的谱系证明、至少两个观测变异、正置信度和正幅度；前两个节点没有合格方向。低分直接父节点列表不得包含当前父节点，也不得重复。这里的资格检查用于确保方向有观测谱系支持，不限制化学图变化。对合格教师 (m)，参考中给出的单位方向 (d_m) 和观测幅度 (r_m) 生成虚拟点云

\[
T'_m=T_m+\min(r_m,0.2\ \text{Å})d_m.
\]

(d_m) 的点云 RMS 范数为 1；无合格方向的情况精确退回原奖励。每个合格教师的虚拟质量为 \(\alpha=0.025\)，原点云仍保留 (1-\alpha)。若 (e_m\in\{0,1\}) 表示该教师当前是否有合格方向，则 TO12 的分支奖励可写作

\[
R(Y_t)=\tau\log\sum_{m\in S_4}\pi_m\left[
(1-\alpha e_m)e^{-\rho_\delta(q_m(Y_t))/\tau}
+\alpha e_m e^{-\rho_\delta(q'_m(Y_t))/\tau}
\right],
\]

其中 (q'_m) 使用 (T'_m) 与相同冻结 assignment。branch `region_weight_mix` 未设置，故默认为 0，原子残差权重为 1。虚拟分支不是化学图限制，也不改变自由的原生图转变。

## 注入大小与保护

配置保持 \(\eta=\texttt{native\_rms\_ratio}=0.33\)、`dose_reference=predictive_flow`。线性 FLOWR schedule 的预测位移为

\[
F_t=(Y_t-X_t)\frac{\Delta t}{1-t}.
\]

代码先按 reward 梯度方向构造注入，再将每个分子的 RMS 注入目标限制为 \(0.33\times\operatorname{RMS}(F_t)\)（本配置 (\texttt{time\_ramp\_power}=0)，时间因子为 1），随后施加单原子最大步长 0.15 Å 和整个控制窗口路径预算 6 Å。路径预算累加每步每个分子的 RMS 注入，即 \(\sum_t\operatorname{RMS}(\Delta X_t)\)，并非把所有位移向量相加后再取 RMS。新产生的严重受体碰撞由 `reject_new_severe_clashes` 保护；阈值为 `severe_receptor_clash_A=0.8` Å，并使用配置的 7 次 backtrack 限额。这是几何安全保护，不是化学图 gate。

在梯度非零且数值正常时，将整个奖励乘一个正的常数主要会被该方向归一化抵消，不能把奖励的整体尺度直接当作注入强度。教师 utility、混合 prior 和各项相对权重会改变梯度方向；`native_rms_ratio` 则设定目标剂量，最终实现剂量还受单步/路径预算与碰撞回溯影响。应从轨迹中的实际注入、累计路径 RMS 和对照的终态指标判断是否有效，而不是只看原始梯度范数。

## 控制与标签时钟

坐标奖励只在 score window ([0,0.5]) 内求值；选择发生在 score time (t) 的事件后，proposal/state 控制延伸至实际窗口末端 0.51。实际 100 步轨迹因此是前 51 步带控制、后 49 步纯 native continuation。最终标签在 clock 1.0 产生，晚于坐标奖励支持。TO6、TO9、TO12 均保留这两个窗口和相同 VJP/剂量参数；TO6 是 family-mean 教师参考，TO9 加入 0.25 observed-tail utility，TO12 则把该 tail fraction 按上式做 count shrinkage。TO7 的 matched-direction 试验是单独的失败分支实验，不属于当前 TO12 奖励，也不能作为当前奖励成功的证据。

## 对应实现与配置

- `src/evomolsteer/generation/endpoint_reward.py`、`affinity_geometry_reward.py`：endpoint 视图、匹配、排序 prior 与点云 log-sum-exp 奖励。
- `src/evomolsteer/generation/branch_mixture_reward.py`：保留原教师质量并加入有界虚拟分支模式。
- `src/evomolsteer/generation/endpoint_controller.py`：真实 FLOWR endpoint VJP、剂量、cap、碰撞保护与逐步轨迹记录。
- `src/evomolsteer/generation/flowcompat_controller.py`：FLOWR forward/VJP 包装与接口审计。
- `src/evomolsteer/generation/flowcompat_v2_controller.py`：当前 `flowcompat_v2` 接口适配器，选择分支奖励并调用上述控制器。
- `src/evomolsteer/generation/local_reward.py`：本配置使用其中 `bounded_local_step` 控制 helper；`LocalIntervalReward` 不是 TO12 的奖励函数。
- 参数核对文件：`configs/experiments/terminal_outcome15_v1/round06_R11_family_rank.json`、`round09_R11_family_tail025.json`、`round12_R11_tail_shrink2.json`。
