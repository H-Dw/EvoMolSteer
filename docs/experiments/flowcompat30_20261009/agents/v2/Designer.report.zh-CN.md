# Designer：联合选择方向的首个单轴试验

本次真实sub-agent模拟读取了当前Designer与flow-compatibility Skills、冻结registry、两份已执行tool receipt、`agent_joint_v3/evidence.json`及已验证Analyst响应。实际API的同一个payload函数产出并保存为 `Designer.simulated.payload.json`，沿同一import validator校验，再由受约束编译器生成 `configs/experiments/flowcompat30_v1/designer_pilot.json`。没有调用外部API，没有编写模型生成的可执行代码，没有运行远端推理。

选择：从 `R26_augmented` 出发，仅改 `innovation.field_strength_A=0.25`，注册公式 `joint_innovation`。这是一项有反证、待实施验证的探索假设，不是已经证实的亲和力梯度。其他奖励、匹配、teacher评分先验、η=0.33、窗口及native续推均保持R26合同。圆形/原子图变化不设置接受gate。

## 证据到方向

同事件高分teacher与较低分几何近邻的完整点云差异主要是内部形状：78.65%的平方幅度，整体平移13.22%、小转角旋转8.12%。真实保留边的端点预测创新与该联合方向平均余弦0.22609，批次区间[0.19691,0.25161]，BH q=2.219×10⁻⁹。它支持检验完整构象中的局部联合方向，不能把每个显著区域独立相加。

反证同样进入设计：方向RMS0.06567Å与噪声0.06668Å同量级；独立控制根ESS平均1.29738；已观测下一节点评分均值下降0.03633 pIC50。在线评分本来就是selector，不构成独立未来亲和力因果标签。因此只采用工具提供的保守支持度，未观察后续保持删失，未以克隆数放大置信度。

工具的联合方向为 `d_k=(T_k−matched lower-score local clouds)/RMS`；原子排列由Hungarian确定，不按原子类型筛选。这里d的符号来自当步高分减低分；时间趋势导数没有用于力。支持度c是局部根权重ESS/方向相干性收缩，再乘跨独立批次匹配方向支持，它不是因果概率。

## 可执行标量及空间导数

每个实际观察时刻t，从原始完整teacher库选K=4近邻。邻居及匹配σ、先验均基于未移位teacher与冻结当前endpoint anchor：

`ℓ_k=−HungarianMSE(Yanchor,T_k)/4 + 2·(score_k−mean score)`，`π=softmax(ℓ)`。

本试验的虚拟中心为：

`T'_k=T_k+λ·c_k^γ·d_k`，`λ=0.25Å`，默认`γ=1`。

全部区域背景保留，`region_weight_mix=0`，所以每原子权重w=1。令 `r_ki=Y_i−T'_(k,σ(i))`、`q_k=mean_i ||r_ki||²`、`ρδ(q)=δ²(√(1+q/δ²)−1)`，当前点云δ=1Å，τ=0.5。

`Rλ(Y,t)=τ log Σ_k π_k exp[−ρδ(q_k)/τ]`。

令 `a_k=softmax(logπ_k−ρδ(q_k)/τ)`，固定匹配、支持度、方向和先验时：

`∂Rλ/∂Y_i=−Σ_k a_k·r_ki/[N√(1+q_k/δ²)]`。

使用真实FLOWR endpoint，`Yworld=scale·Fθ(X_t; frozen old self-conditioning,pocket,t)+COM`，条件VJP为：

`gX=J_Fθ(X_t)^T·scale·∇Yworld Rλ`。

没有对affinity head求导、没有额外生产模型forward、没有新拟合预测器或粒子重采样。原生积分完成后注入旧状态条件VJP，按原R26 predictive-flow RMS校准、逐原子上限及严重新口袋碰撞回退。这个滞后控制不等于全程ODE adjoint或新状态reward保证单调增加。

## 强度实际有多大

本输入1400个teacher中，仅11.5%支持度非零，最大c=0.007693297774520592。方向RMS规范为1，所以λ=0.25对应：

- teacher移位最大RMS：0.001923324443630148Å；
- 全部teacher平均移位RMS：0.00004544551664555Å。

这些由绑定的增强reference直接计算；0.25Å是参数单位，**不是**每一步实际注入RMS。不把置信度不足当成允许移除家族支持的理由。c=0时不新增方向，R26原有reward仍作用于全部实际学习节点；后期大多数节点c=0，但实际0.41、0.47仍有少量非零支持，不能概括成所有后期严格归零。输入学习窗口当前0–0.5，50个评分节点0–0.49，状态终点0.5，随后native到1；窗口不写入Skill常数。

选择范围较保守，是因为信号与噪声同量级且独立家族极少。新增位移可能小到不足以形成有效响应，不能靠原有R26梯度非零掩盖这一问题。若实施门槛失败，先核查支持覆盖、参考单位、实际移位及梯度，再决定补充特征或注册另一控制轴；不未经检验扩大强度。

## 先实施，再评价

本试验启动前的新接口空机制控制必须与旧R26逐位一致。field_strength=0应直接委托R26原算术，区分新知识与微小算术差异造成的随机分支变化。

实际生成后，必须检查：teacher shift RMS、相对新reward梯度变化及配对窗口坐标RMS均确实响应；已执行程序/请求/源码SHA一致；实际FLOWR VJP有限差分通过；同初始状态和RNG；50步支持覆盖、窗口外零注入；生产每步无新增forward、无head梯度、无重采样。然后才比较配对R26与无引导的最终平均预测pIC50，另报告有效分子平均、PB与应变。历史Steer是不同预算的非配对参考。失败留下结果与版本，恢复R26。

当前registry不允许负field_strength；反方向控制不能临时修改冻结reference实现。本首试验以精确零机制、等信息算术控制为必需。如后续需要反方向，应独立注册、绑定、提交再测试。

## 后续注册轴与真实Agent调用范围

以下是允许顺序检验的设计家族，**不代表已批准无条件运行全部参数组合**：

1. `joint_innovation`中的field strength；只有观察到机制响应后才考虑confidence exponent。γ=0会让零支持方向也获得幅度，不能把这种schema许可值当成有科学依据的推荐。保留零置信度抑制，独立测试region weighting，并记录仍保持完整构象背景。
2. `native_control`中的有界Jacobian传递增益剂量：G为同次反传的`||gX||/||gY_native||`，门函数G/(G+Gref)，Gref由先行冻结baseline诊断定义；设置总剂量匹配的固定-dose控制。它是控制调度，不是新scalar reward或真实条件协方差。
3. `native_control`的parallel-component scale与有限time-envelope，分别单轴探索。允许调整与native flow平行分量，但不能把负余弦硬判为坏方向；time envelope不等于搬用t=0奇异的1/t扩散公式。梯度范数饱和也需单位和剂量匹配，不与其他控制轴一起改变。

本次是一份真实Designer响应和一次受约束编译。后续由工程工作流执行的参数探索应标成已注册的单轴选择，不能称为每轮一次新的LLM发明或因果Skill消融。新增证据、公式、源或Skill变化需重新导出请求与真实角色响应，不复用不匹配的旧hash。

## 绑定与验证

请求SHA：`b3ec908488f479f81e6aa4af9ba2d463770802dbe2bf2919ba34b4910194507f`。

实际payload SHA：`2864837a4d931e5c68d9a2c284a0e148ac1dd6843ccabe854ee2cffef77fe71e`。

导入Designer响应SHA：`e151412227a31b3789d8a67e878884cebce5496ae83804b9f59cdaefabf6d536`。

编译pilot SHA：`00ae324746447fd3a2427bb89cf41c609fb4d0d62207841e688247502438201b`。

Schema、来源、证据ID、tool执行receipt、Analyst响应、registered numerical sources与单轴编译已验证；`efficacy_assessed=false`。实际模型实施与性能仍待主工作流完成。本文件记录可复现的科学依据、公式和约束，不包含私有推理链。
