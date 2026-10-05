# Endpoint-anchor proposal 设计审阅

调用身份：Analyst / Designer subagent simulation；2026-10-06。仅生成这三份新docs文件，不运行、编译或导入agents目录，不修改源码/config，不清理数据。

结论：保留新的观察性 lag 证据；固定方向力和区域因果解释均 deferred。建议探索 ASN117 + VAL116、NOS、endpoint-anchored proposal coordinate_mixture，eta=.05、tau=.25、delta=1；这些控制数值是未验证的工程假设。

## 证据与反馈

- 新集合为事先指定的8个CK2区域：ck2:A:ASN117, ck2:A:ASN118, ck2:A:ASP175, ck2:A:GLY46, ck2:A:HIS115, ck2:A:ILE95, ck2:A:LYS68, ck2:A:VAL116。使用discovery0–13，14–19未读取用于决策。
- 全表核查：16/240 adjusted lag q<.05，其中15项NOS、1项all；全部14批、时间覆盖1；请求仅展开其中11项，其余5项以下只按CSV行定位列示，不伪造为合同引用。
- 全部64项proposal selection_shift无q<.05，最小q=0.0959279124621；不能将lag方向当作已证实的选择路径。
- 显著性家族、特征定义和区域集合均改变，不能把本次q相对旧40区域/880项减小解释为单一表征改进。
- 根谱系平均1.2143，11/14批单根；父节点去重不消除共享祖先，提案与下一评分还共享原生随机步和预测模型。
- Round1：valid/PB均98/100；valid target head 7.40776412827628 vs native 7.437039759694313；周围松弛RMS 0.34386082101174437 vs 0.3272142484484583 Å；unique72 vs69。真实整体shape仅改善约1.6766992409593895%。
- Round1报告明确旧当前几何5Å核外承载全部注入平方量；这暴露局部物理接触解释失配，不证明全部性能变化由该问题造成，也不证明新anchor会成功。

## 全部显著lag核查

来源：`results/coordinate_endpoint_anchor_v1/whole_window_evidence.csv`；过滤 `split=discovery, metric=lag_partial_gain_correlation, q<0.05`。每项n_batches=14、mean_time_coverage=1；CI为表内95%CI。

| Feature | Window r | 95% CI | q | Request内可引用 |
|---|---:|---|---:|---|
| ck2:A:ASN117::NOS::centroid_x | 0.03757882 | [0.02497671, 0.05018094] | 0.005264244 | 是 |
| ck2:A:ASN117::NOS::endpoint_transport_z | -0.05390055 | [-0.0808585, -0.0269426] | 0.02330388 | 是 |
| ck2:A:ASN117::NOS::proposal_centroid_x | 0.03303922 | [0.01925532, 0.04682311] | 0.01066108 | 是 |
| ck2:A:ASN117::NOS::proposal_spread | -0.04515871 | [-0.0707874, -0.01953002] | 0.04359257 | 是 |
| ck2:A:ASN117::NOS::spread | -0.0458206 | [-0.0727561, -0.01888511] | 0.04562258 | 是 |
| ck2:A:ASN118::NOS::centroid_x | 0.03497346 | [0.01737539, 0.05257153] | 0.02330388 | 否，仅CSV补充核查 |
| ck2:A:ASN118::NOS::endpoint_transport_z | -0.05076992 | [-0.07347101, -0.02806883] | 0.01573338 | 是 |
| ck2:A:ASN118::NOS::proposal_centroid_x | 0.0315066 | [0.01334484, 0.04966835] | 0.04500656 | 否，仅CSV补充核查 |
| ck2:A:HIS115::NOS::centroid_x | 0.03627919 | [0.0220477, 0.05051068] | 0.008071749 | 是 |
| ck2:A:HIS115::NOS::endpoint_transport_z | -0.04980789 | [-0.07634653, -0.02326925] | 0.02977198 | 否，仅CSV补充核查 |
| ck2:A:HIS115::NOS::proposal_centroid_x | 0.0315825 | [0.01637509, 0.04678991] | 0.02098643 | 是 |
| ck2:A:ILE95::NOS::native_velocity_y | -0.02291201 | [-0.03641568, -0.009408334] | 0.04562258 | 否，仅CSV补充核查 |
| ck2:A:LYS68::all::proposal_centroid_z | -0.03902607 | [-0.06250469, -0.01554746] | 0.04932371 | 否，仅CSV补充核查 |
| ck2:A:VAL116::NOS::centroid_x | 0.03658501 | [0.02266475, 0.05050528] | 0.008071749 | 是 |
| ck2:A:VAL116::NOS::endpoint_transport_z | -0.05084064 | [-0.07705404, -0.02462724] | 0.02541815 | 是 |
| ck2:A:VAL116::NOS::proposal_centroid_x | 0.03184741 | [0.01697702, 0.0467178] | 0.01896236 | 是 |

## 数学与时间合同

在score t的原生forecast y_t上计算区域高斯权重a_ri及条件NOS槽位，求导时stop-gradient；同一步原生proposal u=x_(t+dt)形成后，计算其加权质心c_r=Σa_ri u_i-o_r和spread s_r=sqrt(Σa_ri||u_i-Σa_rj u_j||²)。它们描述预计将参与该区域的槽位在提案时刻的位置，不能称当前残基接触。

只在同一转移内匹配槽位；每次原生forecast重新定义权重，但不得在同一次FD扰动下重算权重。R只对u求导，既不对y_t优化，也不把时间拟合导数当作力。

拼接ASN117和VAL116的NOS proposal xyz/spread为8维z。每批b的真实选中分布给出mu_b(t)=Σp_i z_i及C_b(t)=Σp_i(z_i-mu_b)(z_i-mu_b)^T；这里是预选择候选上的概率权重，不额外按重采样后克隆次数加权。当前实现保存正则化C_b、各spread的未加权背景方差与availability；有效数、根集中度见原始选择与批次诊断。完整背景联合协方差属于后续density-ratio扩展建议，本轮尚未实现。

建议正则Sigma_b=.9 C_b+.1 diag(C_b)，再将特征值下界设为(.1 Å)^2；这两个数是明确的pilot假设、未由当前关联识别。若实施者采用不同已冻结正则，需记录差异，不能静默替换。组件每批等权1/14，不能称14个已证实构象mode。

d_b²=(z-mu_b)^T Sigma_b^-1(z-mu_b)；L_b=delta²(sqrt(1+d_b²/delta²)-1)；R=tau log Σ_b pi_b exp(-L_b/tau)，tau=.25、delta=1。该相关混合用于避免把所有批次平均成单一路径；仍可能集中一个组件，也不能唯一确定分子结构。

原100积分步保持不变。控制窗从metadata读[a,b]=[0,.5]，每步按实际score t和步长dt检查t>=a且t+dt<=b；最后为score .49的proposal X_.5。score .5的proposal X_.51保留越界审计但不得用于X_.5目标或继续注入。t=0初态没有对应proposal控制，首个控制状态是原生第一步之后。

逐score节点保留真实mu/C，若插值用同批节点及SPD安全插值并审计原节点误差；禁止从当前全局函数生成参考。ASN117 NOS proposal_spread一阶拟合在score0为-.030068Å而实测+.027038Å，就是必须保留原节点的反例。

## 备选、控制与可证伪标准

- 选定两区域而非三：ASN117/VAL116提供完整proposal xyz/spread，且有同类NOS lag线索；HIS115虽有x关联，请求未供完整可控制向量，因此不在本合同中增加。相邻两区域以联合协方差处理，不当作两项独立生物收益。
- 不使用固定+x、负z或无界压缩：proposal选择均值未显著；部分lag曲线两端反号；未来gain关联受步后信息与存活混杂。all-spread上尾证据保留，但不能移植为NOS阈值。
- eta=.05作为新表征的低剂量开发值，不按新q值放大；tau/delta与外部eta分离。沿用可审计的per-atom/path/pair/clash限制，global控制使用相同eta、时序、cap及随机种子；记录请求与实际剂量、裁剪、回退、缺失、原生冲突、全局平移和区域外注入。
- 若主流程实施，先做实际proposal条件梯度FD和完整zero/native每步及final tensors一致；额外forward不得改变RNG、原生预测、类别或conditioning。missing NOS跳过并保留分母。
- 最小比较包含matched native、zero、同强度global spread及本endpoint-anchor proposal mixture；相同master42与原始初态。原历史SMC只作既有参照，不新增SMC。该比较评估整套方法，因同时改变anchor/表示/奖励族/通道/强度，不能单独归因于anchor或时序，除非另有匹配消融。
- 优先评价实际X_.5局部与整体双向结构覆盖、全局平移和未控制区域；奖励自身或predicted endpoint达标不能替代。t1统计保留全部尝试、valid/PB、unique/clone、多样性、同构head重评分、有效图MMFF覆盖/收敛与周围松弛。
- 如只提高奖励拟合、未超越global控制或进一步损伤终态合理性，则暂停区域机制性结论；如零对照不一致、窗口外注入、时间错配或缺失补零，则先停止科学解释。没有预言本方案成功，也没有在此开始新实验。

## 可复现来源

- `results/coordinate_endpoint_anchor_v1/agents/Analyst.coordinate.request.json` SHA256 `d6906335b03969e630c6a6f4c13587740cef3c9f1c7ff63886459e2c2d3f56e6`
- `results/coordinate_endpoint_anchor_v1/whole_window_evidence.csv` SHA256 `3846e2c22c619a920e5216bc4116e22827ddbaeccbdecb252e9b6167cbbeaed7`
- `docs/experiments/ck2_coordinate_seed42_20261006/round_01.outcome.json` SHA256 `7ba3e475c802465109ec05367bfc11dedc1e11d1f168b04414374577c1a69ddd`
- `docs/experiments/ck2_coordinate_seed42_20261006/round_01/comparison/comparison.md` SHA256 `6dfb40cd1bb1b41e5638868389c6725bbf25f97d33c5609da37078fa2005f0fe`
- `skills/coordinate-designer/SKILL.md` SHA256 `92a27c08fca8f62d25abd5a9decb336e513c3b7427a8a1ce096f9d9b46be352c`
- `src/evomolsteer/continuous/coordinate_agents.py` SHA256 `df1fb307be1aa1f00c724c394c548e55068af16c623f350fadd7f2f3cde1218c`

合同中的核心引用ID：

- `coordinate:discovery:ck2:A:ASN117::NOS::proposal_centroid_x:lag_partial_gain_correlation`
- `coordinate:discovery:ck2:A:VAL116::NOS::proposal_centroid_x:lag_partial_gain_correlation`
- `coordinate:discovery:ck2:A:ASN117::NOS::proposal_spread:lag_partial_gain_correlation`
- `D_COUNTS`
- `D_LINEAGE`

校验：两个JSON均按当前coordinate-1.0严格schema本地验证；所有合同引用在request中存在；Designer的2×4项proposal NOS观测完整且endpoint锚定。未运行import，因为本任务只允许写三个新docs文件。
