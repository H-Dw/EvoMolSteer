# 最多八轮：真实窗口末端状态的梯度控制设计

2026-10-05 · Designer subagent simulation。本文是声明式研究方案，不是已执行结果；只写本文，不改旧奖励、配置或工程源码，不启动远端任务。用户的新目标覆盖旧 multistage skill/prompt 的全程控制假设：**从学习数据动态确定窗口，以严格窗口末端的真实 X_t 接近原始 Steer sampling 为主目标，窗口之后不再引导。** 禁止新 SMC、候选淘汰与复制重采样。主执行 Agent 正在确认相关措辞，依赖该确认的推理仍应保持未启动状态。

## 1. 先冻结新的数据和时间契约

当前 continuous_single_v3 的 evidence scope 声明 window_start=0、window_end=.5，评分网格含 51 个节点；这些边界应从输入数据读取并绑定哈希，不能在通用代码里硬编码。若将来数据的选择时刻不连续，不能仅取最小/最大值就假定中间均有证据。本例网格连续，记动态窗口为 [a,b]=[0,.5]。

必须区分以下对象：

| 对象 | 历史记录字段 | 含义及新用途 |
|---|---|---|
| 实际 current 状态 X_t | current_coords、current_atomics、current_bonds、mask | 主参考和主验收对象；类别是当时真实 one-hot 状态的紧凑标签 |
| native proposal U_s，s=t+Δt | proposal_coords 等 | 积分后、历史选择前的实际状态；不能用 score_time=t 给它命名 |
| 历史选择后的 X_s | proposal_*[selected_indices] | 与下一事件 current_* 应逐位一致；用于重建实际群体和谱系 |
| 预测 endpoint Ŷ_1(X_t,t) | predicted_coords、predicted_* | 模型对终态的预测；只作辅助观测，不再充当主结构目标 |
| native t=1 终态 | final_prediction | 若顺带生成可保留诊断，不用于本轮主目标或规则选优 |

现有 Trace 在 score_time=t 记录 current，先积分，再对 state_time=s 的 proposal 进行选择。因此原 score=.5 的重采样作用于 **X_.51**。严格 X_.5 应取 current_coords[score_time=.5]，并用前一事件 state_time=.5 的 selected proposal 交叉校验；不能把 score=.5 的 proposal、offspring 或预测 endpoint 当作 X_.5。

新控制以**被修正状态的时间 s≤b**作为门槛。本例最后一次修正是 .49→.5；.5→.51 不再引导。实际注入最多 50 次，参考 current 状态有 51 个时点，两者数量不同是正常的。浮点时刻按已记录积分网格及固定容差定位，同时保存原始浮点值；不得以“最近的一步”偷换严格 b。若运行时必须完成到 t=1，保持 b 之后 native 演化，主验收仍读取独立保存的 X_b。不得把 integration_steps 从 100 改为 50 来提前结束，否则 Δt 会变化；可以在原 100 步网格到达 b 时导出或停止。

现有 continuous_v3 正式特征只包含 predicted_endpoint 和 proposal_state，**没有 current_state**。需在本地从原 HDF5 重建 current 参考和特征，旧 reference_packet 的数值不能直接改名复用。严格 b 的主参考是那一刻实际存在的群体。若研究其留存祖先，终止事件必须按 state_time≤b 重新计算；原包含 score=.5→state=.51 的祖先标签只能作为单独注明的历史分析。

## 2. 参考分布、形状和独立性

所有坐标转到固定 target pocket 的世界坐标系：x_world=coord_scale·x_model+target_COM，并验证原输入蛋白、配体、口袋和变换哈希。只允许对蛋白和配体共同施加坐标变换；不能单独对生成配体做 Kabsch 对齐来抹去位姿错误。

优先保留真实参考分子点云，不把 14 个 batch 均值当作 14 个分子或姿态。参考批次等权，批次内用该时刻实际群体质量；完全重复的克隆可合并存储，但复制质量必须保留。另报去重及 root-balanced 敏感性结果，不能把两种权重混作同一个目标。旧分析显示根谱系强烈塌缩；精确 X_.5 的根数要重新计算，不能无条件沿用旧 score-window 的数字。节点、克隆和 51 个时点都不是独立重复。

几何默认使用固定 active 生成槽位，称为 active-slot 点云。重原子指标另外明确排除当时的 H/PAD，并记录支持数量和缺失；不能把 active mask 自动称为 heavy mask。窗口中间的类别与键可能尚未化学有效，RDKit 可构建率只能辅助报告，不能把 t=.5 的所有构建失败解释成生成失败，或直接套用终态能量/价态假设。

同一模型槽位在谱系内可追踪，但跨独立分子不存在天然原子对应。形状损失必须对原子置换不变。精确网格上使用同刻参考；若确需不同时间插值，可插值核密度/经验测度并标注假设，不能在无对应的两个分子的原子坐标之间直接线性插值。

## 3. 候选函数族：先对实际坐标求梯度

建议先运行原 native 更新得到 U_s，再把其坐标 q_s 作为可求导叶子，计算同刻参考损失，施加有界局部修正，得到实际 X_s。主形状梯度是对 q_s 直接求导，不再经过 endpoint 坐标 Jacobian。这是一种明确的新控制器语义，不应包装成旧方案的简单权重调整。

**F0：实际状态区域基线。** 将 ASN117/VAL116 两距离的相关奖励改为作用于真实 current/proposal 坐标，并从同表示的历史 X_t 提取参考。该项用于检验表示修正本身，不假定旧 endpoint 选择效应在 current 上仍成立；若实际选择对照无支持，仅作为几何诊断基线。模型均值不能直接用作新 actual-state target。

**F1：多尺度点云核距离，推荐首个主形状族。** 对两个实际点云 X、Y，定义

    kσ(x,y) = exp(−||x−y||²/(2σ²))
    Dσ(X,Y) = Σ_i,l a_i a_l kσ(x_i,x_l)
              + Σ_j,m b_j b_m kσ(y_j,y_m)
              − 2Σ_i,j a_i b_j kσ(x_i,y_j)
    Lshape(X,Y) = meanσ Dσ(X,Y)

active 原子质量归一化为和 1；自项必须保留，以免只把所有原子吸到同一参考热点。该 MMD 对槽位置换不变，固定受体帧下保留位置与形状。核导数为 −(x−y)kσ/σ²，可由普通 autograd 精确组合，不需训练额外神经模型。

σ 从同刻发现集真实点云的距离尺度确定，并保留多个尺度；可预声明采用中位近邻尺度的 .5、1、2 倍及 .1 Å 数值下限。这些倍数/下限是工程先验，不是观察到的生物最优值；应先冻结再看候选结果。对多分子参考使用

    Lmix(X,t) = −T log Σ_r π_r exp(−Lshape(X,Y_r(t))/T)
    R = −Lmix

π 遵循批次等权及批次内质量，T 由发现集损失分布尺度确定。它允许不同参考构型，但逐粒子 soft-min 仍可能让整个生成群体集中在少数参考附近，需用分布覆盖验收，不能只看最近参考距离。

该族的主要失败模式也要记录：σ 太小会使远距离核梯度接近零，太大则只分辨粗位置而忽略局部形状；soft-min 温度过小容易只追逐一个参考，过大可能平均不兼容模式。多尺度不是自动解决方案，应以同刻实际参考的距离/梯度覆盖决定尺度，并记录数值零梯度，不用归一化强迫不存在的方向。

**F2：质量守恒的形状匹配备选。** 若核距离改善却仍存在原子重叠、局部缺口或错误覆盖，比较 debiased entropic Sinkhorn：Sε=OTε(X,Y)−OTε(X,X)/2−OTε(Y,Y)/2，代价用固定帧中的平方距离。记录 ε、收敛残差与梯度差分。它比最近邻 Chamfer 更明确地约束原子质量匹配；Chamfer 可作便宜诊断，但单独作为目标可能容许多对一覆盖。不要用错误的“能量”命名这些统计/几何距离。

**F3：软原子类型的空间观测。** 当前 one-hot 类别 A_s 对坐标没有可微梯度。可计算 live 模型概率 pθ(q_s,s)，构造 typed kernel，例如

    ktyped(i,j) = kσ(x_i,y_j) · Σ_c p_i,c p_ref_j,c

或元素通道的受体锚定空间占据。梯度包含显式坐标项和概率头项：

    ∇q R = ∂q R + (∂pθ/∂q)ᵀ ∂p R + (∂Qθ/∂q)ᵀ ∂Q R .

模型输出的 pθ 若预测终态类别，就必须标为“同刻模型 endpoint 概率观测”，不能改称真实 A_s 的概率；若使用 native 类别转移概率，必须按真实积分器推导并记录它预测的下一状态时间。不能对 argmax 假装求导，不能把 hard reference one-hot 冒称为历史保存的概率。

**F4：键概率与几何的联合观测。** 对 live 对称键概率 Qθ,i j,k，可用归一化软键长核矩、键中点在 pocket 帧中的占据、软度数分布等置换不变量；例如

    h_kℓ = Σ_i<j Q_i j,k · κℓ(||x_i−x_j||) / (Σ_i<j Q_i j,k + ε) .

它同时含距离梯度和 Qθ 的坐标 Jacobian。若只有实际 hard bond 标签，仍可将其作为 detached 权重求几何梯度，但不存在对硬键本身的梯度。键长目标和尺度取自同刻实际参考，不直接套终态标准键长。历史完整 bond 概率主要在 anchor snapshot 保存，不能伪造其他时点的连续概率证据；缺概率时使用有定义的 hard-bond 几何族，或明确暂不启用该项。原子类型概率的 float16 历史精度也需记录。

F3/F4 必须与 stop-gradient(p,Q) 对照，分别记录概率 Jacobian 和直接几何项，避免只改善模型软信念而没有改善真实类别/图。最后一次 post-native 坐标修正不能改变已经抽出的 A_b/B_b；若期望影响真实类别，辅助梯度必须在窗口内仍有后续 native 类别更新时作用。最后一步可只保留形状项。若将来改成 pre-native 修正，需要另立相位契约并验证一次 native 类别抽样，不能悄悄增加抽样或刷新离散标签。

**F5：参考覆盖与强度调度。** 若出现群体 mode collapse，可比较预先按 seed/slot 哈希确定、批次质量均衡的参考分配；这只是固定目标条件，不复制/淘汰生成粒子，也不改参考质量。若加入群体核排斥，需明确粒子间耦合，统计单位仍是完整 batch。两者都应与无覆盖控制对照，不把“均值匹配”当作恢复原分布。

外部 η 与内部各项 λ 分开。主项先按发现集自然波动归一化；λ 是相对目标权重，η 决定额外位移。用实际 native 位移 RMS 设控制幅度、每原子 cap 和累计路径预算，不再次乘 dt。η 可从 .05/.15 的小范围重新做数值检查，不能照搬旧 η=.30 已可行的结论，因为目标、梯度位置和控制窗口都变了。.025 Å cap 若保留，应声明为继承的工程上限；50 个合格步的最大加法 RMS 路径为 1.25 Å，不是终态偏移上界。

可用 u=(s−a)/(b−a) 定义 η(s)=η0·g(u)·e(L)，候选包括恒定、前强后弱、前弱后强及目标误差门控。函数连续，无任意 .1 分箱。旧实验的 t>.5 继续控制损害 SMC 终态模仿，只支持删除本轮窗口外控制，**不能直接推出新 [0,.5] 内也应早强晚弱**；方向必须由新 X_t 的误差与实际剂量反馈决定。

## 4. 最多八轮的反馈决策表

推荐把每轮定义为“一份基于上一轮本地报告的新决策 + 至多两个明确变体 + 冻结 incumbent 的配对比较”，不是预先全自动扫完八轮。若用户将八轮限定为八个单独配置，则每轮仅保留表中的首个候选，备选不执行。下表均可由确定性统计、解析核和现有 FLOWR 梯度实现，不需要额外拟合模型。

| 轮次 | 本轮候选及唯一主要变化 | 反馈后如何选择下一轮 | 本轮最低验收 |
|---|---|---|---|
| 1 | 真实状态基线 F0；同时冻结 current/proposal/endpoint 相位、b 快照与直接 proposal 梯度语义 | 若相位、比例或梯度不正确，先修数据/适配器，不用科学分数选方向；通过后看实际空间缺口 | current_b 与历史 selected proposal 的契约核对；无窗外注入/重采样；有限差分、mask、零强度等价与所有失败保留 |
| 2 | F1 实际点云多尺度 MMD，保留最小区域项作可关闭对照 | 若全局形状改善但局部质量错配明显，下一轮选 F2；若形状已好而类型空间分布偏差大，进入 F3 | b 时刻独立 assignment/OT 形状主指标改善，双向覆盖不退化；不能只验收奖励本身下降 |
| 3 | 优先检验 F2 与 F1 的形状度量差别；若无需 F2，则提前做固定参考分配的覆盖控制 | 依据真实形状、局部占据和参考质量覆盖选择 incumbent；失败保留原方案 | 改善须跨适配批次方向稳定；无以原子聚集、遗漏参考模式换取最近邻好看 |
| 4 | 增加 F3 软类型空间项，并对照概率 stop-gradient；无匹配概率证据时只用 hard-type 几何观测 | 如果软概率变好但实际 A_b/typed 形状未改进，不升级；若还有图/键几何缺口，考虑 F4 | 同时呈现软概率与真实类别指标、Jacobian 有限差分、缺类型覆盖；不把 head 变化当作 A_b 已优化 |
| 5 | 依据局部报告二选一：F4 键几何/概率项，或 F5 参考覆盖控制；不得二者一起加 | 只解决最主要且可测的剩余缺口；若无证据，不增复杂度，转控制调度 | 对应 hard graph/几何或覆盖指标改善，同时主形状指标不退化；无伪能量与无新增离散抽样 |
| 6 | 固定目标项，仅改窗口内 η(s)/误差门控；比较一个反馈支持的连续调度与恒定强度 | 根据实际位移、饱和率、目标残差及 b 指标选择；含实际总剂量可比的解释 | 记录请求/实际剂量及截断，消除“调度获胜其实只是加量”的混淆；b 指标改善而非亲和力单项获胜 |
| 7 | 最小化消融：移除一个证据最弱项，或比较同一目标的最小反事实 | 接受更简单且未损失主要效果的方案；本轮结束后冻结公式、参考、参数、代码、评价与新 seed 列表 | 明确各项的可识别贡献和冲突；没有支持就删项，不为用满预算添加项 |
| 8 | 冻结方案的新 seed 确认，无参数修改；与同 seed 的已锁定基线作配对 | 成功则按限定范围报告；失败则如实报告，不能把第八轮反馈再用于本轮调参 | 主终点、覆盖与类型/图副终点按预声明报告；不把适配成绩当新 seed 结果，不启动第九轮 |

最多八轮不是必须八轮。任何时点达到预声明目标可提前冻结确认；连续两轮没有可信改善可停止，或将剩余预算用于确认。不得只更换 seed、事后换指标或扩大边界来把失败写成改善。若进入下一轮需要源码修复，应区分“工程修复重跑”和“新的科学候选”，并在预算账中记录所有实际推理，不能隐藏试错。

## 5. 本地验收与数据拆分

**主终点固定为 b 的真实 X_b。** 建议用与奖励不同的确定性形状度量：同一受体帧下，等原子数点云使用 Hungarian 最小均方匹配距离；不同支持数量使用声明好的质量匹配 OT。再在分子群体层做批次等权的质量匹配，防止“每个生成分子都只接近同一个参考”。另外报告生成→参考与参考→生成的最近距离分布、覆盖、重复率，以及 typed 占据/实际类别分布、hard bond 几何与图统计。不能把非欧氏形状距离随意代入 energy statistic 后截断负值，冒充已经证明有效的分布度量。

参考可以有克隆且分子形状不唯一；原子、节点和参考原型的数量不能用作 n。对发现参考批次和生成批次的敏感性分别报告，bootstrap 必须整批或整根谱系，不能对全部原子/克隆独立重采样制造置信度。这里的统计 bootstrap 仅是本地误差估计，不是生成时的 SMC 筛选；若用户措辞要求连这种统计抽样也禁用，可采用确定性 leave-one-batch-out 敏感性。

发现参考沿用原 discovery 批次 0–13；此前已被分析过的数据不能重新命名为未见验证集。第 1–7 轮使用固定适配 seed 池和相同初始状态配对，所有 round 结果均视为开发资料。第 8 轮使用预先封存、未参与规则选择的新 seed；若原 validation/heldout 参考批次仍未被用于选规则，可在冻结后额外检验对它们的覆盖，但不得反过来调整当轮目标。新的生成 seed 只验证对固定目标的生成稳定性，不代表跨靶点泛化。

适配轮可沿用每变体四批×16=64 的量级；确认轮建议增加独立批次数而非只增加同批粒子，例如八批×16。它们是建议规模，不是已获资源承诺。每轮比较的所有候选数、批次数与失败分母一致。若不允许额外零强度/native 对照，优先复用初态和 seed 可严格配对的既有无引导轨迹；不能拿不配对的旧均值冒充控制。此事项由主执行 Agent 统一按用户确认解释。

验收顺序固定：

1. 工程门槛：正确 b、正确状态表示/坐标系、无窗外控制、无生成重采样、有限梯度/坐标、逐粒子缺失处理和完整输出。
2. 主形状与分布覆盖：候选对 incumbent 的批次配对改善，同时报告分布宽度、参考覆盖和失败。若需要自动晋级，可在第一轮之前冻结工程性阈值，例如平均主损失至少降低 2%、四个适配批次至少三个方向一致、覆盖下降不超过预声明容忍度；这些阈值不等于统计显著或生物最优，实际值应写入本地配置。
3. 当前类别/键及阶段几何：不容许只靠软概率、删掉缺失候选或牺牲主要类别群体来获得漂亮几何指标。中间态化学有效性与终态有效性分开。
4. pIC50/QED 等只作标记清楚的副指标，不能主导晋级。物理能量目前缺少力场类型、部分电荷和一致化学状态，区域重叠、软键损失、模型评分均不能命名为 kcal/mol、结合能或氢键能。

第八轮无论是否达到常见显著性门槛，都应报告全部预声明指标和批次差异；没有显著性并不自动等于等价。不得挑选一个后验最优指标作为唯一结论。

## 6. 每轮记录和执行边界

本地保存输入数据/参考/源码/配置哈希、数据时间语义、split、seed 与初态签名、证据摘要、公式、参数来源、备选方案、简短选择理由、验收阈值、运行输出哈希、全部指标及保留/淘汰决定。记录可审阅的证据和决定，不要求或保存私有思维链。LLM 只提出声明式函数族和参数来源；确定性程序生成参考、计算损失和校验梯度，不允许 LLM 任意修改评价定义。

远端只加载提交并同步的 FLOWR 推理/梯度适配器与冻结参考包，输出原始轨迹和遥测；参考提取、统计、指标评价、Designer 决策与报告全部在本地进行。未确认的执行范围不启动。保留全部生成候选，生成谱系在无重采样时为每粒子的线性状态链；不可再用“零 offspring”定义新实验失败。

至少保留实际 current、未修正 native proposal、修正后状态、score_time/state_time、真实类别和键、mask、世界坐标变换、控制项分解、请求/实际位移、缺失原因及严格 b 的独立快照。b 快照应命名为 window_state 并标 actual_current，不能复用会让人误解为 endpoint/t=1 的文件名。float32 坐标用无损压缩；新加压缩不能暗中把真实状态量化为 float16。类型/键概率可按明示精度单独保存，不能影响主坐标判据。

简洁声明式建议：

    {
      "schema_version": "iterative-window-design-1.0",
      "agent": "Designer",
      "identity": "subagent simulation",
      "status": "design_only_pending_execution_scope_confirmation",
      "max_optimization_rounds": 8,
      "window_source": "continuous_evidence.scope + raw event/grid audit",
      "primary_target": "actual_current_state_at_exact_window_end",
      "primary_reference_population": "original_SMC_actual_current_population_at_same_state_time",
      "reference_split": "discovery_only_for_design",
      "coordinate_frame": "fixed_target_receptor_world_A",
      "guidance_injection_state": "native_proposal_at_state_time",
      "inject_only_if": "window_start <= state_time <= window_end",
      "post_window_guidance": false,
      "new_SMC_or_particle_resampling": false,
      "preferred_shape_family": "multiscale_pointcloud_MMD_with_reference_mixture",
      "shape_alternative": "debiased_entropic_Sinkhorn",
      "soft_chemistry_role": "explicitly_typed_auxiliary_model_observables_with_live_coordinate_Jacobian",
      "physical_energy_status": "not_available",
      "external_strength_separate_from_objective_weights": true,
      "analysis_and_decisions_location": "local",
      "remote_role": "frozen_FLOWR_inference_and_trajectory_output_only",
      "round_8": "frozen_new_seed_confirmation_no_retuning"
    }

依据：[continuous Analyst](../results/continuous_single_v3/agents/Analyst.continuous.response.json)、[原 continuous scope](../results/continuous_single_v3/agents/continuous_evidence.json)、[上一轮独立比较复核](experiments/ck2_multistage_20261005/Designer.outcome_review.md)、[Trace 的时间相位](../src/evomolsteer/generation/controller.py)。本文没有将旧 endpoint 数值或旧 t=1 成绩重新标记为新 X_.5 结果。
