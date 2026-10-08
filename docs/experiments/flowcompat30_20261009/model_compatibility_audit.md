# FLOWR梯度控制与历史选择数据的兼容性审计

日期：2026-10-09。本报告是只读源码与文献审计；没有新增推理，没有测得新的亲和力收益，没有执行模型模块消融。它供本轮Analyst/Designer作为来源绑定的证据使用。

## 1. 首先区分五个对象

历史Steer每个选择事件同时保存当前噪声状态 `X_t`、当步FLOWR预测端点 `Y_t=Fθ(X_t,C_t,P,t)`、原生积分提议 `Z_(t+dt)`、在线评分与重采样索引。评分来自 `Y_t` 的共享模型输出；被复制的是提议状态及条件信息。当前控制器的主要奖励也作用在 `Y_t`，不能将它解释为实际 `X_t` 的势能或t=1真实解码状态的亲和力。

Steer有效的直接机制是：按在线评分把计算预算重新分配给有利候选，保留其状态和self-conditioning，再通过后续随机生成分支探索周边；它不需要给出可微的有利坐标方向。高分候选拥有更高的**期望**复制数，实际存活仍受多项式抽样随机性影响。单次未被选中不能等同于低质量；缺失后续不能等同于终态失败。

源码：`generation/controller.py:63`（current/时钟/谱系）、`:82`（score与endpoint）、`:108`（choose）、`:126`（复制事件与持久化）；`tests/fixtures/upstream_generate_selective.py.txt`（完整原生循环）。应结合实际文件版本核验行号。

## 2. 当前R26的真实梯度是什么

`generation/endpoint_controller.py:20–32` 在当前归一化坐标X上重新启用自动微分，用唯一的原生target forward计算Y及新条件C。亲和力输出detach，条件与Hungarian对应冻结，求得：

`gX = J_F(X; frozen C,P,t)^T · scale · ∇_Yworld R(Yworld,t)`。

这里J是真实FLOWR坐标预测器的Jacobian，并非额外拟合模型。它是固定当前离散标签和旧self-conditioning的条件导数；不是从t到1整个随机、离散、递归生成过程的完整Jacobian。`endpoint_controller.py:54–96` 只在实际学习窗口计算；`:123–173` 等原生积分完成后才注入该方向。因此g是在旧X_t上计算的、施加在新提议Z上的**滞后控制方向**。已有代码准确把 `g·ΔX` 命名为旧状态奖励的一阶估计，不能宣称这是新状态的新鲜奖励增加。

R26的完整端点点云混合奖励见 `generation/affinity_geometry_reward.py:45–76` 与 `endpoint_reward.py:7–35`。先验含在线评分，坐标匹配与teacher邻居离散冻结。原子与化学图没有接受gate；坐标变化可通过下一次模型前向改变原生图分布，这本身不是质量失败。

## 3. 原生动力学不是只有线性FLOW ODE

`generation/coordinate_reward.py:15–18` 的predictive_flow_increment准确计算线性端点参数化的确定性flow分量：

`v_t=(Y_t−X_t)/(1−t)`，`ΔX_flow=dt·v_t`。

但是 `generation/controller.py:271` 显式启用原生SDE模式且坐标噪声为0.2；`window_controller.py:68–82` 在关闭粒子选择后仍保持此设定。不要把关闭Steer重采样理解为关闭原生随机生成。

本轮下载并SHA验证的实际远端源码 `test/flowcompat30_driver/integrator.py::coord_step`（126–176行）在非cosine、非velocity-sample情况下实际执行：

`g_t=1/(t+0.01)`（t<0.9，否则0），

`s_t=(t·v_t−X_t)/(1−t+ε)`，

`Z=X_t+dt·(v_t+g_t·s_t)+dt·sqrt(2·g_t·coord_noise_level)·ξ`。

最后一项源码乘的是dt而非sqrt(dt)。本项目必须保留这个已匹配的实现；不能直接套用连续时间标准SDE的分布、KL或Doob变换保证。观测native位移混合flow、score漂移、噪声；特别早期的收缩不能被解释成有利区域的稳定优化方向。predictive-flow剂量仅选定校准分量，并没有删除score或噪声。

在固定C、离散标签、同一个ξ时，令 `d=1−t+ε`、`A=1+g_t·t/d`、`B=g_t/d`，则原生坐标提议为：

`Z=[1−dt·B−dt·A/(1−t)]X + [dt·A/(1−t)]F(X)+noise`。

其条件Jacobian为 `J_Z=[1−dt·B−dt·A/(1−t)]I+[dt·A/(1−t)]J_F`。这由原生源码代数推导，区别于J_F；不是已实测的模块敏感度。若Designer设计proposal坐标奖励、pre-native控制或endpoint输出修正，必须声明使用哪个状态与Jacobian，并使用实际dt/schedule。奖励对时间的导数dR/dt不等于空间控制梯度。

## 4. 容易被忽略的控制效应

`generation/local_reward.py:55–67` 令控制RMS等于 `η·predictive_flow_RMS·dose_gate`，再执行逐原子与累计上限。因此把整个R乘任意正标量，在非零梯度阈值与数值误差以外会被归一化抵消；修改reward coefficient不必然改变真实剂量。多奖励的相对权重主要改变方向，η和gate改变剂量。

当前上限是防过大位移与新严重受体碰撞；它们不能证明能量合理。`path_rms`是逐步RMS之和，不是最终净位移或控制能量。固定累计RMS也不等同于固定 `Σ||ΔX||²/dt`。现有R26 dose有效参数η=0.33；实际点云曲率由 `pointcloud_delta_A` 默认1Å控制，继承的 `robust_delta=2` 不作用于该公式。

匹配冻结让一次数值VJP核验有明确含义，但跨步Hungarian对应/K近邻切换仍会造成目标不连续。无需禁止切换，应先记录teacher posterior与对应变化、梯度方向一致性，分析其与后续预测和终态的关系。

## 5. 尚可新增且不增加生产前向的测量

下面是待实现的工具输出，不是假装已有观测结论。每项只保留批次×实际时间的汇总/分位数和少量异常索引；逐原子完整数组为可选诊断，不默认常驻。

|信息|计算与用途|解释边界|
|---|---|---|
|端点到状态的敏感度传递|同一次reverse请求gY与gX；`||gX||/(scale·||gYworld||)`、区域梯度能量分布|反映当前**奖励**经模型的敏感度，不是亲和力头贡献，不是全Jacobian谱|
|方向与原生流/漂移关系|gX分别与Δflow、观测Δnative的余弦；`gX·Δnative`|抗原生方向可能有益；不能简单硬删除负余弦方向|
|跨步稳定性|同一未复制谱系的端点innovation、当前/前步梯度余弦、teacher posterior换模率|随机和离散状态改变会影响innovation，不是模型预测方差的无偏估计|
|区域选择创新|从实际亲子路径计算区域Δendpoint、Δcurrent、Δproposal，分离期望选择项与复制噪声|需在同一事件/亲本背景比较，不能把全群体共同native运动算富集|
|联合空间构型|landmark对/三元组的占据共现、局部点云协方差/方向、相对区域距离角度，固定受体坐标系|不能只把显著边际字段相加；区域重叠和整体姿态要作为混杂处理|
|高值路径与局部变化的联系|在线评分持久性、分数相对事件中位数的margin、优势事件中的空间方向一致性；独立批次与家族权重|在线分数改善参与选择，非native未来因果收益；灭绝后未来保持删失|
|局部控制动作可信度|原始/投影梯度、实际剂量、clash backtrack、窗口内覆盖、dt归一化动作成本|识别被cap、数值no-op或控制未落实，不能仅看终态均分是否变|

`torch.autograd.grad(value.sum(), (X,Y))` 可从已存在图同时取得所需梯度；不需要dense Jacobian，不需要亲和力求导，不需要额外神经前向。精确实施应做小型自动微分测试确认gX与旧实现逐位一致，避免新增无信息机制制造浮点扰动后被误当算法收益。

## 6. 模型模块“重视区域”必须怎样核实

真正的口袋模型主要为 `flowr/models/pocket.py::LigandGenerator/LigandDecoder`，并由 `pocket_util.py::SemlaLayer` 组合self-attention、pocket条件attention及feedforward；不是把旧mol-only `semla.py::SemlaGenerator` 任意当作本次checkpoint架构。LigandDecoder将当前与self-conditioning坐标共同embedding，当前和口袋的RBF距离进入边与interaction模块，端点投影和图投影共享上游表征。旧条件坐标是一条独立输入路径，当前梯度将它冻结。

历史轨迹没有记录hidden activations、attention或模块干预，因此仅由XYZ不能识别“哪个模块因果重视哪个区域”。若确需测试：在保留的正常forward上装可选hook，将 `gen.ligand_dec.layers[i].self_attention/cond_attention` 的equivariant更新、梯度×更新按区域汇总；也可在单独审计的小样本上做条件坐标/指定模块干预，并与正常前向配对。attention大小和gradient×activation仍是解释性代理，不能直接声明亲和力因果贡献。真实模块消融额外调用必须计入审计预算；当前生产推理不加这些重复前向。

来源版本：本轮已下载并核验实际远端 `test/flowcompat30_driver/fm_pocket.py` SHA=`beaace3ab92a8377bc8eab95333c8f40ce370ca9224f7a78927c5a424bb38426`、`integrator.py` SHA=`e9abd83167f708da4712db2ac49ab24ac44a39c59c46a54c20789c7873c58000`。上述动力学、`fm_pocket.py:874` forward、`:1471` predictions、`:1970` selective生成语义已在这两份准确源码上复核。实际源码也含residual selfconditioning/categorical guard选项，但选项存在不代表checkpoint启用，仍应读hparams。

口袋模块结构主要读缓存 `tmp/flowr_root_scnet_source/flowr/models/pocket.py` 与 `pocket_util.py`，尚未逐文件与远端绑定，因此模块hook应以运行时实例名称和捕获源码为准。缓存整文件SHA为 integrator=`ce858a27d25fd268a855181f4434bceb1ad14150d03d401dd28b8cbb165cfc20`，fm_pocket=`e2d10a5789820bc7127e3dc3e3e144785b29db1098c4f739ad22647f5756a2b4`。即便差异只是版本或换行，也不能拿缓存哈希代替实际执行合同。

## 7. 两类值得顺序探索的新机制

**数据挖掘优先：** 学习条件于当前事件、根/父谱系和全局姿态的“优势选择创新”，重点是哪些区域的endpoint改变在高在线margin路径中反复出现，并报告被排除分支的已观察对照、复制运气和删失。对原始/期望/实现选择项分别输出全窗口曲线与批次CI；时间拟合用留一批次验证，曲线和导数只描述时间规律。已有74项压缩边际几何不显著或失败，不能继续把其常数差值重新包装成新规则。

**奖励/控制优先：** 保留R26联合多模态点云基底。先在相同forward中量化敏感度传递、切换与流方向，再注册单轴的“VJP增益/创新可信度→有界剂量”机制，与R26进行相同总剂量的负对照；检验改变来自剂量分配而非额外信息或浮点归一化。另一条独立实验轴是已学区域创新的联合点云方向项，按完整模式而非各区域均值吸引。必须保持窗口由输入声明读取，并在窗口后原生续推。

一个可检验的维度正确的gate为 `G=||gX||/||gY_native||`，`h(G)=G/(G+G_ref)`；gY_native已包含world到native的scale，二者比值无量纲。G_ref可用预先注册的固定常数或先行baseline诊断冻结，不根据当轮最终亲和力即时调节。h只改变剂量分配，不能声称获得了完整条件协方差或value function。避免每个batch重新以自身分位数定门槛导致样本之间相互依赖；配套总剂量匹配的控制。zero开关要求严格走原R26数学运算路径。

若直接在预测端点Y上做输出修正δY，则实际原生状态变化为 `dt·A/(1−t)·δY`（上述源码条件下）；这是一种不同控制接口，不等于R26的J_Fᵀ梯度。需要独立新扩展、准确self-conditioning语义、zero/native逐位检查和剂量匹配。若使用proposal势 `R(Z,s)` 并回传 `J_Zᵀ∇_ZR`，也应明确它优化的是s时刻已学状态，不能声称完整t=1值函数。这两条先作为后续预注册备选，不混入初始R26比较。

## 8. 既有失败约束与实施反馈门槛

历史 `ck2_affinity_geometry30_20261007/30_rounds_workflow_and_reward_analysis.zh-CN.md`、`elite_path20_20261007/report.zh-CN.md`、`dynamic_contrast10_20261008/report.zh-CN.md`、`selection_path20_20261008/report.zh-CN.md` 已记录：压缩/稀疏方向、父子历史奖励、整窗晚期dose重配、宽teacher覆盖、窄核、过高剂量、强化在线分数、普通边际区域对比、终态信用替换、niche/协方差/逐原子robust等不能自动改善R26。新方案必须声明具体新信息/新控制机理，不能只重复这些名字。

分析Agent需要提交实际tool运行receipt及输出SHA，引用正/反证；Designer响应必须改变可执行程序并有公式与参数绑定。生成前进行固定坐标的梯度/剂量响应核验和无信息控制逐位一致测试，生成后确认实际覆盖、非零剂量、窗口外零注入、正常native RNG、相同initial SHA与单步生产forward预算。实施确认不等于性能确认；最终仍以冻结新批次的全部尝试平均pIC50主指标、有效率/PB和应变次指标比较R26与native，历史Steer为非配对参照。失败方案留下报告/commit，恢复原R26源/程序/Skills运行合同。

## 9. 一手文献与迁移边界

[Feng等，On the Guidance of Flow Matching，ICML 2025](https://proceedings.mlr.press/v267/feng25s.html) 的[全文](https://arxiv.org/html/2502.02150)区分一般flow guidance、局部近似与协方差近似；单纯搬用扩散梯度不具有一般保证。其Gaussian/affine条件下Jacobian与端点条件协方差的关系，为审计敏感度与时间尺度提供启发。FLOWR当前self-conditioning、离散随机图和源码SDE不满足直接套用全部条件，未声称复现论文的精确引导或学习新协方差模型。[作者代码](https://github.com/AI4Science-WestlakeU/flow_guidance)当前仅核实项目说明，未逐文件复现。

[Wang等，Training Free Guided Flow Matching with Optimal Control，ICLR 2025](https://arxiv.org/html/2410.18070v3)将奖励、动态控制和运行成本联合考虑，实际算法涉及沿完整ODE轨迹的co-state/VJP及多次更新。它支持把reward方向、控制代价和动态响应分开审计，而不支持把当前单步端点VJP宣称为完整OC-Flow。为保持生产前向预算，本轮借鉴控制成本/条件Jacobian诊断，不执行其全轨迹迭代求解。[作者代码](https://github.com/WangLuran/Guided-Flow-Matching-with-Optimal-Control)已核实仓库身份，未声称复現分子实验。

Feng的Gaussian affine近似若进一步设α=t、β=1−t，reward取正，则速度修正系数变为 `(1−t)/t · ∇X R(F_t(X))`（由其调度公式代入推导）。这个量在t=0形式奇异，理想独立Gaussian路径中的J_F会相应趋零；实际神经预测、自条件、离散标签及当前随机实现不保证抵消。尤其不能把已经归一化的单位gX直接乘1/t而放大。若探索该时间形状，必须有明确有限边界、独立schedule轴、固定dose对照和“近似启发”说明；不把它作为既有FLOWR的严格正确调度。教师点云协方差也不是条件于当前X的真实端点协方差。

[FLOWR.ROOT作者仓库](https://github.com/jule-c/flowr_root)为底层生成与Steer实现来源；实际执行仍由本地/远端捕获的版本和checkpoint SHA绑定，不能以最新README替代运行源码。
