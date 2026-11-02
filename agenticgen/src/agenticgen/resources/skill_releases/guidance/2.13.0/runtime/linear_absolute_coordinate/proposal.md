---
name: jump-guidance-proposal
description: Propose up to five executable diverse scale hypotheses and execute every proposed direction.
metadata:
  version: "2.13.0"
---

探索有机会超过已测global best的完整scale组合。倾向多样化、跨窗口重分配和未测区域，不把局部微调当作默认路线。假设指潜在更好的完整落点，不是用于解释窗口作用的研究分支。

阶段一，读标定根、真实最佳父链、回滚坏路径、七窗覆盖、可执行网格和当前预算。逐窗指出证据覆盖到哪个背景；不把最佳路径未改过的窗口视为已收敛，不把均匀高档较差推成该档不能用于某个窗口。

阶段二，先考虑多种完整落点：早弱晚强、递增/递减、交错、多窗重分配、覆盖薄弱窗口、当前边界之外的新机会。优先让多个方向进入不同组合区域，不只围着末窗加减一档。早段弱剂量是待测机会而非预设答案，保留当前或较高剂量的反例。首窗形状按冻结配置执行，不由你更改。

阶段三，比较coverage_profile_suggestions和window_evidence。建议是程序生成的合法未测候选，performance=null，没有收益或稳定保证，也不穷尽全域。可采用建议或提出新的合法组合。足额时优先提出5个不同、可执行的完整假设；最多5个，预算或合法新落点不足时可少于5个。所有提出的方向都会实测，不能先报5个再让综合角色只挑3个。

至少保留一个真实多窗bold机会，其余也应倾向不同的改变窗口集合和非均匀组合；不强制focused/expanded/bold各占一个，也没有固定增减方向配额。合法同背景剂量括界可并列，不靠无意义改变其他窗口伪装多样性。不要为了凑数量提出已测、失败或低于已知参照却无新机会的近重复。

阶段四，调用audit_action_batch（整组准入模式）审计全部提案的累计真实新增成本。每方向是完整50人口，最多250次/轮，五轮最多1250次/教师。精确重复只引用，失败计费不补样；标定另受500总额约束。只读审计不是inference完成证明。

通过原生submit_final_response显式提交decision/proposals/reason；每条包含完整实际target_scale_schedule、真实ID及optimization_target，且相对当前global best计算exploration_tier。不输出updates。理由用简短可核验的证据、机会和未知范围，不编造改善幅度；每个说明字段上限3000字符。hold明确proposals=[]；no_future_execution不再提案。严格数组、证据、预算及唯一一次格式修正不放宽。

七窗口按prompt的guidance_windows排序，末两维独立对应51–75与76–99（第99步原生）。历史六窗末值复制到两维只用于新预实验背景，不证明两段各自最优或存在甜点。允许一段升、另一段降及非均匀联合组合，始终以完整配置实测表现判断。
