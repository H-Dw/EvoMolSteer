# CK2 多阶段、全程坐标引导：冻结实验方案

本实验由 continuous-3.0 的 discovery 证据及 Designer 决策产生，不改动分析的全窗统计口径。分段仅用于拟合逐时刻原型；没有恢复按 .1 区间发现规则的旧方法。

## 奖励与权重

令 z 为预测终点相对 ASN117、VAL116 的两个 N/O/S 条件 soft-min 距离，单位 Å。μ_b(t) 是发现集第 b 批窗口留存祖先的拷贝加权特征均值；14个批次在混合中各占1/14，不按克隆数再加权。Σ(t) 是同一时刻、各匹配无引导批次内部协方差的等权均值，经0.1向对角收缩及0.1 Å最小特征标准差约束。它处理相关性和尺度，不是两个可任意叠加的区域权重。

定义 q_b=(z−μ_b)ᵀΣ⁻¹(z−μ_b)，ρ(q)=√(1+q)−1，R=log[Σ_b exp(−ρ(q_b))/14]。对坐标做奖励上升。此式为新构造的经验几何奖励；不宣称来自物理能量或已证明的最优路径。奖励不是全局有界，但白化特征空间的远场导数有界，实际位移另受控制约束。

目标曲线在全部51节点上按最大白化残差≤0.1选择局部PCHIP节点，得到26节点、25段。协方差按原51节点凸插值保持正定。t>.5冻结末端参考，不外推高次多项式；这是显式待检验的后期维持假设。核心目标不在段边界消失。原子类别由同次live预测argmax得到，每步更新、求导时固定；缺少N/O/S则跳过并记录null。

## 执行

g=∇x R(hθ(x,t))，通过真实终点网络Jacobian回到当前坐标。冻结模型参数及历史self-conditioning。先运行原生积分，再加入与g同向的修正：单分子RMS修正目标为 η×本次原生坐标步长RMS。η是外部控制强度，和上述目标尺度分开。

每原子单步附加位移≤.025 Å；累计逐步RMS位移≤2.5 Å。新修正相对同一步原生proposal不得新增<1.2 Å的受体原子对，也不得令配体原子间距额外改变>.05 Å；依次尝试1、1/2、1/4、1/8、1/16修正，全部失败则还原原生proposal。后两项位移上界部分由三角不等式保证，是审计不变量。它们不控制原生流放大的最终偏移，也不能认证化学有效性。

梯度分支完全不重采样。实际奖励评估次数、可用原子掩码、raw/masked梯度、分量冲突、混合责任度、native冲突、clipping/backtracking和后半程实际位移均落盘。单步原始结构按既有验证过的HDF5流程即时转换。

## 预声明的测试

1. 本地验证特征实现、有限差分、batch独立尺度、刚体／排列不变性、缺原子情况、全时间支持、正定协方差、谱系及控制上界。
2. 远端pilot：每组12个候选，同一初始先验和随机种子20261015；无引导、零权重、η=.05/.15/.30三个全程组。校验零权重与原生完全一致、live有限差分符号及误差。
3. 不按亲和力挑选η。选最高可行比例：可用且非零期望修正的中位 applied/desired≥.5；完全拒绝率≤5%；有效连通数不低于无引导超过1个；末态严重碰撞候选数不增加。全不合格则不自动进入比较。
4. 冻结后用新种子20261105，4个独立批次×16候选＝每组64个，比较无引导、同批次大小单目标SMC、旧版早期奖励、恒定末端目标全程奖励、仅窗口多阶段奖励、全程多阶段奖励。SMC粒子数16与历史50不同，不能把二者当同一规模复现；本轮6组内部保持一致。
5. 评价完整曲线、相对同批SMC的二维白化energy统计、独立的接触质心／全原子距离／Rg变化、末态构建与连通／多样性／碰撞、既有CK2头重评分。另计算受体固定坐标系中的对称Chamfer形状距离（不旋转对齐、不假定原子对应）以及末态有效分子的Morgan指纹最近SMC Tanimoto（radius=2、2048 bits），分别检查三维外形与化学图相似度，避免仅凭奖励自身指标评价“拟合结构”。早期观测窗口、后期延续和最终输出分别评价。以独立批次而非克隆或帧作统计单位；4批不足以作强显著性结论。

## 入口

```powershell
.venv/Scripts/python scripts/build_multistage_reward.py --analysis results/continuous_single_v3 --dataset data/optimized/main1000_w050/analysis_inputs_v2 --references results/multistage_v1/references
.venv/Scripts/python scripts/build_multistage_reward.py --references results/multistage_v1/references --designer configs/experiments/ck2_multistage_v1/Designer.decision.json --output configs/experiments/ck2_multistage_v1
```

生成：`scripts/generate_multistage_flowr.py --flowr-root <FLOWR目录> ...`；远端环境封装：`scripts/scnet_multistage_experiment.sh pilot|comparison`；评价：`scripts/evaluate_multistage_runs.py --campaign <生成目录> --output <结果目录> [--calibrate]`。代码在本地commit/push后远端pull，不在远端改代码。默认实验工作目录放在`/opt/evomolsteer-multistage-20261005`，避免占满仅剩少量空间的private_data；归档和可定位记录再放回MolSteer实验目录。
