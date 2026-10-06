实际选择 **R7**，其全体 predicted-affinity 增益 +0.2292925739288325、unique 增益 +0.2288584818143642、η=.3，均优于本轮新增候选。R9 在相同 η 下全体 −0.04142868518829346、unique −0.03796944725950535，平均实际步 RMS=.013420811693184077 Å，故选择 geometry_target_revision，不能归结为没有控制。R10 将 η 提至 .51（1.7 倍），全体增益 +0.20678879261016814，比 R7 少 .022503781318664373，unique 少 .05816664288186679；更强剂量不必更好。R9 数值窗口增益和 FD/zero 输出未包含在当前 input，不补写为已核实结果。

已数值遍历全部 200 个局部字段和 63 条受支持节点函数。每个 Gaussian 区域含 soft mass、dx/dy/dz 和六个 covariance 项，20 区域共 200 项；推断应先去除重复父代的后代权重，再按 14 个独立批次统计。Skill 指定三个 measure 的全部 600 检验联合 BH；输入只给 200 enrichment 汇总和 63 节点函数，无法凭这些独立重算另 400 项。全窗富集、富集随时间变化、真实父子边 forecast 增量差是不同统计量；d_dt_z 不是空间力或物理原子速度。

选择 **landmark 00**：按正全窗 soft-mass 效应、q<.05、至少 12/14 正批次，再取最大效应。符合的区域 0、7、10 分别为 +.6392349943936658、+.48060148482325576、+.395255271050672；区域 0 的 q=.00018794547124610335、14/14 为正，CI [.4924599163269142,.7906507287766374]。但其首节点 −.36595078451489843、末节点 −.2753379093708555；50 节点有 43 正、7 负，真实节点梯形均值复算为 +.6392349943936657。因此 constant_signed_attraction_allowed=false，不能把正全窗结果改写成恒定增加区域质量或趋近受体的力。

全 20 个重叠区域各有受支持字段，不能证明 20 个独立因果结合模体。41 个 q 显著 covariance 富集为负，也存在相反证据：区域 4 czz=+.37909097686204707、q=.00722186117701822、13/14 为正，不支持统一压缩。

信息审计的 excess fixed RMS=.08097363029151962 Å，CI [.06969549178775168,.0921417865342447]；相对几何最近邻的非负 excess 是构造性质，不是亲和力结果。压缩匹配的 teacher-score correlation=.10182576506541118，反而高于完整匹配 .06407848917780258；两个独立报告的 CI 重叠，不能据此宣称显著差异。Top4 overlap=.46974125364431485。审计没有测完整 score-weighted reward 或 FLOWR Jacobian，proper rigid RMS 还固定 Hungarian 对应，不能因果解释 R7。

首个设计 endpoint_regional_pointcloud：η=.3，K4，beta2，teacher temperature=4 Å²，radius=5 Å，background=.25，τ=.5，δ=1 Å，time_ramp_power=0，恒定区域幅度，精确 20 权重 [1,0,…,0]。在已需 forward 的预测终点上冻结 anchor A、attention、Hungarian 对应和离线 priors。a_i=.25+Σ_r w_r exp(−||A_i−p_r||²/(2·5²))；q_k=Σ_i a_i||Y_i−T_ki||²/Σ_i a_i；ρ_k=δ²(√(1+q_k/δ²)−1)；R=τ logsumexp(log π_k−ρ_k/τ)。独立 elite 点云保留在混合中，不平均坐标、不选粒子或克隆。正 background 保留区域外作用，R≤0 有有限上界但没有有限下界。

只求固定 attention/assignment/prior 的条件坐标梯度，再通过实际 FLOWR endpoint VJP 回到 x_t；不声称包含重新计算 membership 的总导数，也不加入 atom/graph gate 或 affinity-head loss。窗口从本轮输入读取 [0,.5]，覆盖其 50 个实际节点，之后 native 继续到 1；该终点不是通用硬编码。原生步后的响应只是旧 x_t 的一阶估计，生产额外 forward/head 调用为 0。

新 GPU 验证未执行。激活前需四次一次性实际 FLOWR FD forward，以及独立的新家族完整 100-step eta0/native equality；CPU/fixture 不能替代。若终点亲和力回归，保留 R7；后续单独改变区域范围、半径或经验绝对效应幅度。对 all/valid/unique/best/Top5 和失败分母分别报告；当前缺少绝对均值、best、Top5、失败数和 strain 数据，coverage=1 也不代表化学有效率 100%。历史 Steer100 仅为子集标尺，未提供完整最大值或配对 heldout 证明。

完整 Skill/input/prompt 均已读取并自行 SHA256；reference/region-prior 和审计来源哈希精确采用 input 的绑定值，并明确未越过三文件读取范围重算外部字节。完整绑定和字面参数见 response.json。input=1294f21c9883357aa4530b0bb2fc74f2e977b62f2d59c6b337b31d80ab91dbc6；region prior=5a6b23af25e4b322ae8d49a77741a08954b6360aaacf906c4862414aaeca6b3e。本次仅写两个响应文件，不修改源码或启动 GPU。
