选择实测全局父程序 **R7**：终点 all-head mean=7.663567433357239，较 native +0.2292925739288325；比 R3 高 0.1674862337112426。R6 的 best-valid=8.35207748413086 虽最高，其全体均值却回落 −0.006338667869568049；R8 为 −0.009526290893554723、有效率变化 −0.03，均不能替代 R7。R7 的负 MMFF 相对变化 +0.11347669622126166 仅作次级证据。历史 Steer 子集均值 7.510350561141967、最佳值 8.347940444946289 是不同指标，既非配对等预算验证，也未核实原实验完整最大值。输入缺少 absolute valid/unique 计数及均值；score coverage=1 不代表化学有效率为 100%。

R6 输入报告实际 FLOWR FD 和完整 zero 检查通过，平均注入 RMS=0.013322108623199165 Å；窗口末节点均值增益 +0.11198570728302001 到终点 −0.006338667869568049，损失 0.11832437515258806。这是明确控制与暂时增益后的终点消失，应选 persistent_coordinate_target_revision，不能仅说权重太小或没有扰动。案例动作分别为 R1 dose_escalation（先核实实际剂量）、R2 secondary_energy_assessment/parent_refinement、R4 parent_refinement、假设负值 geometry_target_revision；假设 −0.1 不写成已执行结果。

首个新设计为 endpoint_supported_attractor，**η=0.3，history_strength=0**；块权重 [0,1,2,1]，time_ramp_power=0，τ=0.5，teacher_score_beta=2，prototype_robust_delta=2，observed_abs_effect salience。response.json 精确复制 74 位权重及 24 条×50 节点 values_z，按 linear_node_effect 保留真实节点；24 条多项式全未过 fidelity（例如 shape_yy 末节点 +0.16329622239089556，旧拟合 −0.5128231820840609）。74 项证据均保留在所绑定 prior，14 项 scale-floor 不纳入；q 显著也不自动入选。

以现有几何函数 Φ 及固定尺度 s 定义 z=Φ(Y)/s；各原始 elite prototype 单独保留 T_p=Φ(Y_p)/s。支持权重先按块归一化：shape 每项 1/2，pair 每项 2/5，landmark 每项 1/17；随后乘 clip(|真实节点效应|,0.1,2)，实现没有再次归一化。令 v 为对应帧 variance_scaled 的行均值，q_p=Σ_k W_k(z_k−T_pk)²/v_k，C(q)=4(√(1+q/4)−1)，log π_p=log_softmax[2(score_p−mean score)]，R=0.5 logsumexp_p(log π_p−C(q_p)/0.5)。R≤0 有有限上界，但并非下界有界；匹配一个 prototype 也不必达到 0。score prior 是固定模型标签，不是实测结合或亲和力网络梯度。

V3 历史统计排除首个未观测零占位，用真实父子边高减低增量总和/0.49 汇总，不作占位梯形积分。仅 landmark_00_softmin 通过：rate=−4.914369782450085，CI [−6.854885532781242,−2.6157352943386276]，13/14 批次负方向，q=0.04608273058076118。landmark_00_occupancy3 的历史 q=0.08105526993766458 不通过。landmark 00 的受体定位为 A 链 VAL53 CG1，[26.351,−30.381,16.189] Å；关联不能证明特定氢键、原子类型条件或因果结合机制。

后续可在相同 η 下单独比较 λ=0.25、1、4。仅该历史字段获一单位质量，目标是逐节点高分绝对平均增量 μ_n：d_n=z(Y_n)−stopgrad(z(Y_(n−1)))−μ_n，H=Σ h_k d_nk²/v_k，P=λ·phase²·C(H)，总奖励 R−P；首节点历史永不激活，内部节点需紧邻的 detached forecast。筛选依据高减低差异，奖励却匹配高分绝对增量：49 边 high 总和 -1.4667331286332943、low +0.9413080647672474，high 有 22 正/27 负；第一边 high=−1.1789173480595225，而 high−low=+0.04861708018195943，末边 high=+0.11909325047405649。不得把全窗负差异改写成每步负向力、物理原子速度或已证实的亲和力持久机制。

只在原本必需的 target forward 上取预测终点，真实 FLOWR VJP 为 g=J_Fᵀ∇_Y R，历史、self-conditioning、assignment、原型和 prior detached；生产额外 forward/head 调用均为 0。窗口 [0,0.5] 覆盖 0…0.49 的 50 个支持节点，正连续调度后继续 native 到 1；不扩到无证据阶段，不加入 graph/atom-type veto、选粒子或克隆。原生步后使用旧 x_t 的梯度注入，g·Δx 只是固定节点的一阶估计，未重新测得 post-native endpoint reward。

新损失的 GPU 核验 **未执行**，不能继承 R6 的 zero 证明。history0 的实际模型 FD 一次额外 4 次 joint forward（ε=0.003/0.01，各 ±）；未来非零历史分别审计 startup 与内部激活状态，合计 8 次一次性 forward。另需新程序完整 eta0/native 对照。精简表示可能遗漏 R7 的有效点云细节；任何终点回归都保留较强实测父程序，并将窗口增益、all/valid/unique/best/Top5 和失败分母分开报告。

Skill、完整 prompt、冻结 input、prior V2、kinematics V3、velocity 表、landmark context 与实际奖励源码的 SHA256 均在 JSON 绑定。input=2718831be541b30c2255bb9f90bfbe034df114daafdf82f5d6366c2449345270；kinematics=4f20693cdadfd5a53c4c4981e5971082733cd7c8b05de732d77ca6d9361b2426。本次只生成字面响应和数据一致性检查，不声称新实验通过。
