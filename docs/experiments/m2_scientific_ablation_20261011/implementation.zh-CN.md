# 科学子模块消融实施说明

研究目标：在保持匹配 M2 的基础上，检验哪些科学指引确实改善最终预测亲和力或次级结构质量。性能目标不是既定结果。

|一级模块|科学子模块|待验证功能|
|---|---|---|
|A1|A1.1 即时选择富集|Actual contrast population and label；Signed coordinate effect；Comparison-bound evidence identifier|
|A1|A1.2 后续质量贡献|Immediate versus later evidence comparison；Later-label provenance and missingness；Priority consequence or explicit unavailable terminal comparison|
|A2|A2.1 不确定性影响排序|Real uncertainty and support；Reliability changes priority rather than boilerplate|
|A2|A2.2 跨批重复与反例|Actual batch support or opposing evidence；Counterevidence affects use of a direction|
|A3|A3.1 局部优势超出全局运动|Local versus global distinction supported by actual tool；Bound region or explicit unresolved localization；Retained surrounding freedom|
|A3|A3.2 协同空间 motif|Actual joint relationship or intact mode evidence；Coupling affects hypothesis or target representation|
|A4|A4.1 全窗口连续趋势|Evidence-bound continuous function and support；Fit fidelity or sign change assessed；Stationary versus time-conditioned consequence|
|A4|A4.2 谱系变化来源|Observed edge and population comparison；Residual semantics and limitations；Consequence for proposed path rule|
|A4|A4.3 可干预坐标假设|Observable and intended response；Target and scale origins；Executable hypothesis or stated interface limitation|
|D1|D1.1 坐标 observable 到公式|Specific Analyst hypothesis to operative edit；Correct formula-to-observable relationship；Numerical response prediction|
|D1|D1.2 静态与时间条件目标|Static versus temporal decision；Target/attention/dose distinction；Support and source binding|
|D2|D2.1 几何响应与曲率尺度|Scale source and units；Correct operative parameter meaning；Predicted response change|
|D2|D2.2 教师邻域与构象模式|Actual teacher geometry evidence；Mode/neighborhood decision；Count distinct from bandwidth|
|D3|D3.1 实际位移剂量|Delivered versus requested control；Diagnostic-driven dose decision；Normalization and lane respected|
|D3|D3.2 窗口内剂量分配|Whole-support schedule decision；Target motion distinct from dose；Expected coverage or update response|
|D4|D4.1 修改保留与暂缓|Evidence affects retain/edit/defer decision；Concrete incumbent comparison|
|D4|D4.2 反证与周围兼容性|Evidence-bound falsifiable failure response；Counterevidence affects design；Affinity priority and surrounding freedom|

每个子模块的完整指令、合理竞争方向、数据前提和逐项输出评分均位于冻结 module_registry.json。旧八扩展原句及四追加原句不进入新默认提示词；其旧实验的奖励混杂与数值等价不能当作科学方法无效的证明。

流程：先用真实 API 验证符号、缺失输入和物理单位响应；建立同 RNG 批次的 M2/无引导对照；先 Analyst、再 Designer 形状、再 Designer 剂量。每条件六条独立角色链；每次只删除或替换一个子模块。全窗口学习与控制来自原参考元数据，当前评分节点为 t=0...0.49，对应最后受控状态 t=0.5；其后原生推理到 t=1。

200 尝试仅筛查。删除方案须在新的400尝试上确认，相对当前完整指引和同批 M2 的同时区间下界均高于 −0.02 pIC50。应变中位数/90分位不得恶化超过10%，PB/有效率下降不超过2个百分点。最终800尝试比较无引导、M2、历史Steer，单独报告尾部与评分覆盖。所有界限已预先固定；不以随机未显著差异直接判无效。

功能评审先于新生成；数值相同的奖励共享分子观测。实际受控节点、剂量、裁剪/回溯、预测端点响应与最终质量分别保存，角色输入不包含当前质量结果。功能未响应、接口瓶颈、数值等价和统计不确定性单独标记。剩余模块在最终上下文再次验证，未通过必要性标准者不得被描述为已证实必要。

远端仅执行 FLOWR。原始Steer与checkpoints保护，新原始输出在校验传输、执行审计和本地结构/能量评价、精简报告保存后回收。API 请求/真实响应、证据引用、程序与源码commit保留，以重建可公开核查的设计依据；不保存模型隐含思维过程。

正式角色调用还接收全部200个连续函数的系数或实测节点值，并绑定原工具文件与完整API输入的SHA256；不是只读取起点、终点与平均变化率摘要。此输入完整性更新发生在任何正式完整/删除/替代链调用之前。

执行接口修订：M2 原配置、参考教师和温度保持原样；Agent 读取实际生效的点云参数。旧 robust_delta=2 和几何块/motif 标签不会被描述成点云奖励项，实际 pointcloud_delta_A 默认1 Å。修正前的一条预计算角色链归档为接口诊断，不用于科学条件。

随机输入构造修订：本研究改为按批次种子42+100003×batch_index调用同一上游输入加载器，避免生成高编号批次之前十万余份未使用输入。其先验分布未修改，但不声称与旧流式位置逐字节一致。新 M2、无引导及全部候选重新使用相同初态配对；技术试跑中止记录保留且不进入模块评价。详见 prior_policy_amendment.json。

应变的显著作用额外检验：采用实际收敛能量样本的独立批次中位数及90分位数，以角色链/批次交叉重采样得到相对恶化区间。缺失能量不补零。亲和力须在完整/删除及完整/替代两项比较中均落入±0.02的等价带；随后同一应变端点在两项干预中恶化的同时区间下界均超过10%，且功能验证通过，才可判为当前上下文中的应变必要模块并阻止删除。亲和力与两个应变端点共同扩大预设确认族至324。M2质量容差仍只是接纳保护，不能独自证明应变显著作用。详见 secondary_statistical_amendment.json。

共同观测字典：所有正式条件都读取经原特征提取源码校验的公式与单位。occupancy3/5是高斯径向壳层密度，不是半径内原子数；软最小距离、整体/局部协方差与pair kernel定义分别提供。该字典不指定优势方向、不改变任何Skills模块或原统计值。补充前的单条角色探测归档，不进入科学条件。

共同引用修复接口：对实际返回的全部非法引用提供精确字段路径与错误值，而不是只重复巨大合法编号列表。仅允许真实API根据原输入重新校验引用；不自动替换编号或改变科学结论。失败输出与原请求保留，不计为有效角色链，也未触发分子生成。已成功角色缓存继续复用。详见 citation_repair_transport_update.json。
