本次审阅由 **Analyst / Designer subagent simulation** 完成，仅使用本地发现集与已完成开发实验；没有修改源码、运行生成或访问远端。建议将新 15 轮用于检验局部与周围槽位的相对几何、尾部质量和剂量时序，而不继续把平均 NOS 形状拟合当作能量或亲和力的替代目标。图可以自由变化，方案比较不引入候选 SMC、图锁定或组成 veto。

**新 motif 证据支持探索，但尚未支持区域因果力场。**

spatial_motif_discovery_v2 使用 discovery 0–13、全部 51 个真实评分节点、预声明 ASN117/VAL116，含 108 个 current/proposal 特征。统计分别报告选择、同期 affinity 和去重父节点后的下一步 gain，不能混用：

| 检验族 | q<.05 | 可以得出的结论 |
| --- | ---: | --- |
| selection_shift | 0/108 | 未建立整窗均值方向 |
| root_balanced_selection_shift | 0/108 | 根均衡后也未建立均值方向 |
| retained / root_balanced_retained shift | 各0/108 | 未建立整窗末端祖先留存均值优势 |
| high / low enrichment | 46/108、9/108 | 部分分位尾部有选择关联 |
| 同期 partial affinity | 0/102 | 没有调整后同期证据 |
| lag_partial，仅调起始 head | 4/108 | 均为两区域 NOS centroid_x |
| lag_controlled，加 global/NOS nuisance | 0/108 | 上述 x 方向未保持稳健 |

例如 VAL116/NOS_C proposal_pair_3 的高分位概率差为 −0.005594798，q=.01294987；proposal_shell_2 为 −0.010581993，q=.00054324。前者的均值选择偏移 q=.579147、根均衡 q=.644383、进一步调整 lag q=.386819。VAL116/NOS proposal centroid_x 的 lag_partial r=.03196629、q=.01094264，进一步调整后 r=.00679474、q=.59696186。这些数值均来自 14 批，报告的时间覆盖为 1；lag 的覆盖域为有下一步观察的 0–.49。

负的 high_enrichment 表示选中概率质量在同节点上四分位中较少，单位是概率差，既不是距离变化，也不意味着应持续排斥该核值。NOS_C 的 centroid/shell 使用 NOS 与 C 的并集；只有 pair 使用跨 NOS–C 原子对。2Å 壳核真实 selected mean 很小：VAL116/NOS_C proposal 在 score=0 约 6.07e−8，在 score=.5 约 .00104；不能把它命名为已形成的 2Å 接触。

**为什么平均局部形状改善而 MMFF 尾部恶化。**

低阶中心、spread、tensor 和几个核值都不能唯一决定键长、键角、扭转或化学图；而旧 NOS-only 条件梯度没有碳邻域协同项。这使“局部目标更像参考、整体化学应变仍大”在数学上完全可能，但尚无逐力项证据证明某一种键或扭转是原因。

R25/26 的 MMFF relief/heavy 中位数 .571034/.562339 相对 native .548773 变化不大，p90 却从 1.706811 增至 2.290123/2.290138。98 个配对有效分子中 76 同图、22 变图；变图子集贡献净周围 RMS 增量的约 96.15%/94.94%，但其中只有 11/22、10/22 的 MMFF 上升。三个最大正向周围 RMS 变化贡献该子集净增量的约 81.17%/93.22%。这是尾部与事后子集的描述，不能据此认定变图有害或锁定 native 图。

当前 energy 是各自同图内的配体 MMFF 松弛释放量，不是结合能。surround RMS 来自无受体力场、未刚体对齐的游离配体松弛，混有整体位姿漂移；不能直接称为受体兼容性。全程 native continuation 也不是 MMFF 最优化器。

Original Steer 的 all-head=7.51035056、unique-valid-head=7.26822154，独特结构24；native 对应 7.43427486、7.27186835，独特结构69。因此应同时报告总体、有效和独特结构指标，不能只追逐克隆权重影响的均值。这并不证明 Steer 优势全由克隆造成。

**可实施的候选与反证。**

| 证据 → 假说 | 可微程序 | 如何推翻或限缩解释 |
| --- | --- | --- |
| NOS 低阶目标不描述周围骨架；NOS_C pair 有尾部选择关联 → 测试相对几何协同 | 对固定 endpoint anchor/类型的实际 proposal，匹配跨 NOS–C 的 1.5/3/5Å 平滑成对核联合分布 | 自身核拟合提高而 head、MMFF p90/CVaR、内部松弛不改善，不能声称协同优势；纯 pair 不保证受体放置 |
| 壳核关联与全局放置高度共享 → 分离 shell、pair、joint | shell 用 2/4/6Å 核；joint 用缩放后的完整相关协方差，不把两区域当独立机制相加 | shell 改善只说明几何投影改变；没有独立结构/亲和力改善便不能称接触机制 |
| 增量选择差很弱 → 检验匹配背景 contrast | 规范化 Gaussian mixture 的 log(p_selected/p_candidate)，再作有界 tanh；保留幅度与饱和到外部剂量 | 组成/根均衡后信号消失或剂量接近零，分别报告；不能把单位归一化放大的微弱差异当强证据 |
| 均值掩盖少数高损失 → 软尾部几何控制 | 对有支持的 motif 区间偏离作平滑最大值惩罚，后续逐对尾部项须另有数据 | energy 尾部无改善或 head/valid 损失，保留该权衡，不靠删除失败样本制造成功 |
| 首步注入占比与时间错配可能重要 → 时间/剂量独立消融 | 真实节点参考、保界插值、全窗口固定尺度；只改 envelope 或一个剂量参数 | 只有启动扰动有贡献就称启动效应，不能描述为全窗口进化规律 |

已实现 motif 的基本式为
\[
f_{rk}(x)=\sum_{i\ne j}\widetilde w_{ijr}
\exp[-(\|x_i-x_j\|-r_k)^2/(2\sigma^2)] ,
\]
其中 NOS_C 仅保留跨类对，\(r_k=(1.5,3,5)\)Å，\(\sigma=.75\)Å，anchor 宽度4Å。梯度同时作用两个槽位；这些长度是预声明统计核，**不是**学得的共价键、氢键或力场参数。固定硬类型与 anchor 求条件坐标导数，不伪造类别/键梯度。

联合目标可使用各发现批次的经验均值/协方差：
\[
q_b=(z-\mu_b)^\top\Sigma_b^{-1}(z-\mu_b)/D,\qquad
R=\tau\log\sum_b\pi_b\exp[-\rho_\delta(q_b)/\tau].
\]
\(z\) 使用全窗口、等批次背景波动尺度；正则化在无量纲空间完成。批次均值并非已识别真实构象 mode。鲁棒能量混合与规范化 Gaussian 密度不同，contrast 必须另含 determinant 与正确指数。

**必须保留的谱系、时间与实现限制。**

t=.5 时 11/14 批只有一条根谱系，平均根数1.214286；当步 selection ESS 仍为48.427/50，不能视作48个独立祖先。主分析保留原 selected 质量；根均衡是改变估计目标的敏感性分析，不能静默替代原始 Steer 分布。借鉴 [STREME 的匹配背景](https://meme-suite.org/meme/doc/streme.html)和 [Felsenstein 的谱系非独立问题](https://www.journals.uchicago.edu/doi/10.1086/284325)，并不意味着序列 motif 检验或 Brownian independent contrasts 已适用于本 SMC 树。

发现两个报告/控制合同问题：

- within_root_affinity_correlation 在批次表已算，但 t=0 根内方差为零，非 lag 的 finite.all(1) 规则让完整窗口汇总全部消失。它是**未汇总**，不是阴性结果；应公开缺失、可识别时域与覆盖。
- 描述性 Legendre 拟合对72条核曲线在1001点审计中有44条越出[0,1]。奖励应使用真实节点及保界插值，协方差保持SPD；时间导数不能当空间力。
- manifest 尚未单列 spatial_motifs.py 的 SHA，本审阅附带该源 hash。联合通道的统一 validity 也需注意：一个 NOS+C 时 NOS_C 已定义，不应因未使用的 NOS-only 双槽要求而连带关闭其合法坐标项。

全部51个原始评分节点仍用于选择分析；proposal 在 score=t 对应状态 t+dt。最后控制是 .49→.50，score=.5 的原始 proposal=.51 不能当作真实 x_.5 或新增注入点。之后 native 完成到1。每个新架构需要固定条件的真实 FD、NumPy/Torch 一致性、完整100步零剂量/native一致及 RNG/self-conditioning 控制。

**新15轮的建议安排。**

前4个名额用于 joint/pair/shell 的预声明表征和执行控制；5–7用于一项背景/谱系/协同消融；8–10依据尾部与节点贡献检验单项软约束或时间剂量；11–12用于方案级候选复现/消融；最后3个名额尽量留给冻结候选的批次扩展或预登记随机流确认。每轮先记录 parent、唯一主要变化、参数来源及反证，再运行。部分指标退化不自动终止，但也不要求跑满预算。

方案比较只借鉴 [NSGA-II 的非支配权衡](https://homes.cs.washington.edu/~sagarwal/nsga2j.pdf)，比较对象是完成一轮的奖励程序，绝不据此筛除或复制生成候选。保持全部 attempted 分母、all/valid/unique head、PB、能量覆盖、median/p90/CVaR 与多样性。相同 seed42 和已反复查看的两批只支持开发性结论，不能作为未见 seed 泛化。

尚缺：逐对极端几何/角度/扭转的发现统计、具体 MMFF 力项归因、真实受体能量和新的全盲确认。新 motif 架构能否改善联合目标仍待运行；完整的证据、精确统计行、公式、来源 hash 与反证条件保存在同目录 JSON。
